"""T0.1 DoD 산출물: 모든 데이터셋을 build_dataset()으로 열어 배치 하나를 로드해보고,
샘플/클래스/그룹 수를 실측해 outputs/experiments/T0.1/result.json에 쓴다
(docs/plans/contracts/README.md 공통 헤더 규약, phase0.md T0.1 판정 규칙).

미보유 데이터셋(imagenet-sketch)은 실패가 아니라 status="missing"으로 기록한다.
waterbirds의 n/group_counts는 build_dataset(split="test")이 아니라 metadata.csv
전체(11,788행) 기준이다 — phase0.md의 실측 표와 산출물 예시가 그 값을 쓰고 있고,
"데이터가 실제로 몇 장 있는가"를 보고하는 목적이 split 필터링된 학습용 뷰와는
별개이기 때문이다.
"""

import csv
import datetime
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import DataLoader

from Utils.datasets import AVAILABLE, DatasetUnavailableError, build_dataset

TASK_ID = "T0.1"
SEED = 0
BATCH_SIZE = 4
OUTPUT_PATH = Path("outputs/experiments/T0.1/result.json")

_RESIZE_NEEDED = {"imagenet", "waterbirds", "imagenet-r", "imagenet-a", "imagenet-sketch"}
_EXTRA_KWARGS = {
    "imagenet-c": {"split": "report", "corruption": "gaussian_noise", "severity": 1},
    "imagenet-9": {"split": "original"},
}

# verdict="pass"의 필요조건 3종 (phase0.md 판정 규칙)
_REQUIRED_OK = {"waterbirds", "imagenet-c", "colored-mnist"}


def _resize_224(image):
    """가변 해상도 데이터셋(imagenet 계열)을 224x224로 맞춰 DataLoader 배치가 stack되게 한다."""
    arr = np.asarray(image)
    return cv2.resize(arr, (224, 224))


def git_commit_hash() -> str:
    """현재 HEAD 커밋 해시(40자)를 반환한다. 실패 시 'unknown'."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _open_for_smoke(name: str):
    """name에 맞는 kwargs로 build_dataset을 호출한다(스모크 테스트용 고정 크기 transform 포함)."""
    kwargs = dict(_EXTRA_KWARGS.get(name, {}))
    if name in _RESIZE_NEEDED:
        kwargs["transform"] = _resize_224
    return build_dataset(name, **kwargs)


def _waterbirds_full_counts() -> tuple[int, list[int]]:
    """metadata.csv 전체(split 필터 없이)의 총 샘플 수와 group_counts([n_group0..3])를 센다."""
    metadata_path = Path("data/waterbirds/waterbird_complete95_forest2water2/metadata.csv")
    group_counts = [0, 0, 0, 0]
    n = 0
    with open(metadata_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n += 1
            g = 2 * int(row["y"]) + int(row["place"])
            group_counts[g] += 1
    return n, group_counts


def _probe_dataset(name: str) -> dict:
    """데이터셋 하나를 열고 배치 하나를 로드해 상태 블록(dict)을 만든다."""
    try:
        ds, meta = _open_for_smoke(name)
    except DatasetUnavailableError as e:
        return {"status": "missing", "reason": str(e)}
    except Exception as e:  # noqa: BLE001 — 어떤 원인이든 result.json에 남기고 다음 데이터셋으로 진행
        return {"status": "error", "reason": f"{type(e).__name__}: {e}"}

    try:
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
        batch = next(iter(loader))
        assert batch[0].shape[0] == BATCH_SIZE
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "reason": f"batch load 실패: {type(e).__name__}: {e}"}

    block = {
        "status": "ok",
        "n": meta.num_samples,
        "classes": meta.num_classes,
        "groups": meta.num_groups,
        "source": meta.source,
    }
    if meta.has_group_labels:
        block["group_names"] = meta.group_names

    if name == "waterbirds":
        n_full, group_counts = _waterbirds_full_counts()
        block["n"] = n_full
        block["group_counts"] = group_counts

    return block


def main():
    """AVAILABLE 전체를 스모크 테스트하고 result.json을 쓴다."""
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    datasets_block = {}
    unknown_fields = []
    for name in AVAILABLE:
        block = _probe_dataset(name)
        datasets_block[name] = block
        if block["status"] != "ok":
            unknown_fields.append(f"{name}: {block['status']} — {block.get('reason', 'n/a')}")
        print(
            f"{name}: status={block['status']} "
            f"n={block.get('n')} classes={block.get('classes')} "
            f"groups={block.get('groups')} src={block.get('source')}"
        )

    finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    ok_required = all(datasets_block[n]["status"] == "ok" for n in _REQUIRED_OK)
    verdict = "pass" if ok_required else "fail"
    verdict_reason = (
        "waterbirds/imagenet-c/colored-mnist 모두 ok"
        if ok_required
        else "waterbirds/imagenet-c/colored-mnist 중 하나 이상이 ok가 아님: "
        + ", ".join(n for n in _REQUIRED_OK if datasets_block[n]["status"] != "ok")
    )

    result = {
        "task_id": TASK_ID,
        "git_commit": git_commit_hash(),
        "seed": SEED,
        "started_at": started_at,
        "finished_at": finished_at,
        "config": {
            "available": AVAILABLE,
            "batch_size": BATCH_SIZE,
            "extra_kwargs": _EXTRA_KWARGS,
        },
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "result": {"datasets": datasets_block},
        "unknown_fields": unknown_fields,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nverdict={verdict} ({verdict_reason})")
    print(f"result written to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
