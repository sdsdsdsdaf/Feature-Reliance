#!/usr/bin/env python
"""data/imagenet-c 의 파일 분포를 클래스 단위까지 세어 기록한다.

ImageNet-C 는 ImageNet val 50,000 장(클래스당 50장)을 손상시킨 것이라
`<corruption>/<severity>/<wnid>/ILSVRC2012_val_########.JPEG` 만 있어야 한다.
이 저장소의 사본에는 64x64 `test_####.JPEG` 파일이 섞여 있어서, 어느 셀·어느
클래스가 얼마나 부풀었는지를 삭제 **전에** 확정해 두기 위한 스크립트다.

사용:
    python scripts/audit_imagenet_c.py --out outputs/experiments/T0.1/imagenet_c_audit_before.json
    # (삭제 후)
    python scripts/audit_imagenet_c.py --out outputs/experiments/T0.1/imagenet_c_audit_after.json

두 리포트를 --compare 로 넘기면 클래스별 증감을 표로 뽑는다.
"""
import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

VAL_PREFIX = "ILSVRC2012_val_"
EXTRA_PREFIX = "test_"
# spec 규약: HP 튜닝 전용 4종, 보고용 15종.
HP_CORRUPTIONS = {"gaussian_blur", "saturate", "spatter", "speckle_noise"}


def scan_cell(cell_dir: os.DirEntry) -> dict:
    """corruption/severity 셀 하나를 훑어 클래스별 파일 수를 센다.
    반환: {wnid: {"val": n, "extra": n, "other": n}} — 파일명 접두사로 분류한다."""
    per_class = {}
    with os.scandir(cell_dir.path) as classes:
        for cls in classes:
            if not cls.is_dir() or cls.name.startswith("."):
                continue
            counts = Counter()
            with os.scandir(cls.path) as files:
                for f in files:
                    if not f.is_file():
                        continue
                    if f.name.startswith(VAL_PREFIX):
                        counts["val"] += 1
                    elif f.name.startswith(EXTRA_PREFIX):
                        counts["extra"] += 1
                    else:
                        counts["other"] += 1
            per_class[cls.name] = {"val": counts["val"], "extra": counts["extra"], "other": counts["other"]}
    return per_class


def audit(root: Path) -> dict:
    """data/imagenet-c 전체를 훑어 셀별·클래스별 분포와 오염 요약을 만든다.
    반환: {"cells": [...], "affected_classes": [...], "totals": {...}} 형태의 리포트 dict."""
    cells = []
    affected_sets = {}
    totals = Counter()

    with os.scandir(root) as corruptions:
        corruption_dirs = sorted((e for e in corruptions if e.is_dir()), key=lambda e: e.name)

    for corruption in corruption_dirs:
        with os.scandir(corruption.path) as sevs:
            sev_dirs = sorted((e for e in sevs if e.is_dir()), key=lambda e: e.name)
        for sev in sev_dirs:
            per_class = scan_cell(sev)
            n_val = sum(c["val"] for c in per_class.values())
            n_extra = sum(c["extra"] for c in per_class.values())
            n_other = sum(c["other"] for c in per_class.values())
            affected = sorted(w for w, c in per_class.items() if c["extra"] > 0)
            hist = Counter(c["val"] + c["extra"] + c["other"] for c in per_class.values())

            key = f"{corruption.name}/{sev.name}"
            affected_sets[key] = affected
            totals["val"] += n_val
            totals["extra"] += n_extra
            totals["other"] += n_other

            cells.append(
                {
                    "corruption": corruption.name,
                    "severity": sev.name,
                    "split": "hp" if corruption.name in HP_CORRUPTIONS else "report",
                    "n_classes": len(per_class),
                    "n_val": n_val,
                    "n_extra": n_extra,
                    "n_other": n_other,
                    "n_total": n_val + n_extra + n_other,
                    "n_affected_classes": len(affected),
                    "files_per_class_hist": {str(k): v for k, v in sorted(hist.items())},
                    "extra_per_affected_class": sorted({c["extra"] for w, c in per_class.items() if c["extra"] > 0}),
                }
            )

    # 오염된 셀들이 전부 같은 클래스 집합을 건드렸는지 — 같으면 단일 원인(한 번의 잘못된 추출).
    nonempty = {k: v for k, v in affected_sets.items() if v}
    reference = next(iter(nonempty.values())) if nonempty else []
    identical = all(v == reference for v in nonempty.values())

    return {
        "scanned_root": str(root),
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "totals": dict(totals),
        "n_cells": len(cells),
        "n_contaminated_cells": len(nonempty),
        "affected_class_set_identical_across_cells": identical,
        "affected_classes": reference,
        "n_affected_classes": len(reference),
        "cells": cells,
    }


def print_summary(rep: dict) -> None:
    """리포트를 사람이 읽는 요약으로 출력한다. 셀별 표와 총계."""
    t = rep["totals"]
    print(f"root={rep['scanned_root']}  cells={rep['n_cells']}  오염 셀={rep['n_contaminated_cells']}")
    print(f"총계  val={t.get('val', 0):,}  extra(test_*)={t.get('extra', 0):,}  other={t.get('other', 0):,}")
    print(f"오염 클래스 수={rep['n_affected_classes']}  (모든 오염 셀이 동일 집합? {rep['affected_class_set_identical_across_cells']})")
    print()
    print(f"{'corruption':<20}{'split':<8}{'val':>10}{'extra':>10}{'total':>10}  files/class")
    by_corr = defaultdict(lambda: Counter())
    split_of = {}
    hist_of = defaultdict(Counter)
    for c in rep["cells"]:
        by_corr[c["corruption"]]["val"] += c["n_val"]
        by_corr[c["corruption"]]["extra"] += c["n_extra"]
        split_of[c["corruption"]] = c["split"]
        for k, v in c["files_per_class_hist"].items():
            hist_of[c["corruption"]][k] += v
    for corr in sorted(by_corr):
        v, e = by_corr[corr]["val"], by_corr[corr]["extra"]
        hist = ", ".join(f"{k}장×{n // 5}" for k, n in sorted(hist_of[corr].items(), key=lambda kv: int(kv[0])))
        print(f"{corr:<20}{split_of[corr]:<8}{v:>10,}{e:>10,}{v + e:>10,}  {hist}")


def compare(before: dict, after: dict) -> None:
    """삭제 전/후 리포트를 받아 무엇이 어떻게 줄었는지 출력한다."""
    tb, ta = before["totals"], after["totals"]
    print("== 총계 변화 ==")
    for k in ("val", "extra", "other"):
        b, a = tb.get(k, 0), ta.get(k, 0)
        print(f"  {k:<8}{b:>12,} -> {a:>12,}   ({a - b:+,})")
    print()
    print("== 셀별 클래스당 파일 수 분포 ==")
    bmap = {(c["corruption"], c["severity"]): c for c in before["cells"]}
    changed = 0
    for c in after["cells"]:
        b = bmap[(c["corruption"], c["severity"])]
        if b["files_per_class_hist"] != c["files_per_class_hist"]:
            changed += 1
            if changed <= 5:
                print(f"  {c['corruption']}/{c['severity']}: {b['files_per_class_hist']} -> {c['files_per_class_hist']}")
    print(f"  ... 분포가 바뀐 셀 {changed}/{len(after['cells'])}개")


def main() -> None:
    """CLI 진입점. --compare 두 개를 주면 비교, 아니면 스캔 후 --out 에 저장한다."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/imagenet-c")
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    args = ap.parse_args()

    if args.compare:
        before = json.loads(Path(args.compare[0]).read_text())
        after = json.loads(Path(args.compare[1]).read_text())
        compare(before, after)
        return

    rep = audit(Path(args.root))
    print_summary(rep)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=2))
        print(f"\n리포트 저장: {out}")


if __name__ == "__main__":
    main()
