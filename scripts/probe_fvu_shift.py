#!/usr/bin/env python
"""FVU가 corruption severity에 반응하는지 확인하는 소규모 probe (mini-T1.1).

M7(FVU gate)의 전제는 "shift가 오면 SAE 재구성이 무너진다 -> FVU가 올라간다 ->
그때 gain 갱신을 보류한다"이다. 이 전제가 성립하지 않으면 게이트는 발동할 수
없고 M7은 죽은 코드가 된다. 정식 T1.1(15 corruption x 5 severity x 512장)을
돌리기 전에 방향만 싸게 확인하기 위한 스크립트다.

**이건 T1.1이 아니다.** 셀당 이미지 수도 corruption 수도 적으므로 결과는
`outputs/experiments/T1.1/probe_fvu_shift.json`에 별도 아티팩트로 남기고
`result.json`은 건드리지 않는다.

사용:
    PYTHONPATH=. python scripts/probe_fvu_shift.py
    PYTHONPATH=. python scripts/probe_fvu_shift.py --n 256 --corruptions fog,snow
"""
import argparse
import datetime
import json
import subprocess
from pathlib import Path

import timm
import torch
from PIL import Image
from timm.data import create_transform, resolve_model_data_config

from Model.sae_runtime import FrozenSAE
from Utils.datasets import build_dataset
from Utils.SAE_utils import collect_tokens_with_hook

CHECKPOINT_PATH = "outputs/reservoir_sae/vit_b_sae.pt"
OUT_PATH = "outputs/experiments/T1.1/probe_fvu_shift.json"


class _TransformDataset(torch.utils.data.Dataset):
    """build_dataset이 준 (image, label[, group])에 timm transform을 씌워 준다.

    __init__: 원본 dataset과 timm transform을 받아 그대로 보관한다. 원본이
    np.ndarray를 주든 PIL을 주든 PIL로 맞춘 뒤 transform을 적용한다."""

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        """샘플 수."""
        return len(self.dataset)

    def __getitem__(self, idx: int):
        """idx번째 (transform된 이미지, 라벨). group label이 있어도 앞 둘만 쓴다."""
        item = self.dataset[idx]
        image = item[0]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        return self.transform(image), item[1]


def git_commit_hash() -> str:
    """현재 HEAD 커밋 해시(40자). 실패 시 'unknown'."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def measure(frozen, model, transform, dataset, n: int, device: str) -> dict:
    """데이터셋 앞 n장의 block10 패치 토큰을 모아 FVU와 L0를 잰다.
    반환: {"fvu": float, "l0": float, "n_images": int, "n_tokens": int}."""
    subset = torch.utils.data.Subset(dataset, list(range(min(n, len(dataset)))))
    loader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=False)
    tokens = collect_tokens_with_hook(
        model,
        loader,
        max_tokens=None,
        target_block=frozen.meta["target_block"],
        token_scope=frozen.meta["token_scope"],
        device=device,
        cache_dtype=torch.float32,
    ).to(device)
    code = frozen.encode(tokens)
    return {
        "fvu": frozen.fvu(tokens),
        "l0": frozen.l0(code),
        "n_images": len(subset),
        "n_tokens": int(tokens.shape[0]),
    }


def main() -> None:
    """in-domain 기준선과 corruption×severity 격자에서 FVU를 재고 JSON으로 남긴다."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128, help="셀당 이미지 수")
    ap.add_argument("--corruptions", default="gaussian_noise,fog,glass_blur")
    ap.add_argument("--severities", default="1,3,5")
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    frozen = FrozenSAE.from_checkpoint(CHECKPOINT_PATH, device=device)
    model = timm.create_model(frozen.meta["model_name"], pretrained=True).eval().to(device)
    transform = create_transform(**resolve_model_data_config(model), is_training=False)

    ds, _ = build_dataset("imagenet", split="val", root="data")
    in_domain = measure(frozen, model, transform, _TransformDataset(ds, transform), args.n, device)

    cells = []
    for corruption in args.corruptions.split(","):
        for severity in (int(s) for s in args.severities.split(",")):
            ds, _ = build_dataset(
                "imagenet-c", split="report", root="data", corruption=corruption, severity=severity
            )
            m = measure(frozen, model, transform, _TransformDataset(ds, transform), args.n, device)
            m.update(corruption=corruption, severity=severity, fvu_ratio_vs_in_domain=m["fvu"] / in_domain["fvu"])
            cells.append(m)

    # 게이트가 발동할 수 있는가: in-domain FVU를 넘는 OOD 셀이 하나라도 있는가.
    above = [c for c in cells if c["fvu"] > in_domain["fvu"]]
    # severity 방향성: corruption별로 sev가 오를 때 FVU가 오르는지.
    increasing = 0
    by_corruption = {}
    for corruption in args.corruptions.split(","):
        series = sorted((c for c in cells if c["corruption"] == corruption), key=lambda c: c["severity"])
        fvus = [c["fvu"] for c in series]
        mono_up = all(b >= a for a, b in zip(fvus, fvus[1:]))
        by_corruption[corruption] = {"fvu_by_severity": fvus, "monotonically_increasing": mono_up}
        increasing += int(mono_up)

    report = {
        "task_id": "T1.1-probe",
        "note": "예비 probe이지 T1.1이 아니다. 셀·이미지 수가 작아 판정 근거로 쓰지 않는다.",
        "git_commit": git_commit_hash(),
        "seed": 0,
        "started_at": started_at,
        "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config": {
            "checkpoint": CHECKPOINT_PATH,
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "images_per_cell": args.n,
            "corruptions": args.corruptions.split(","),
            "severities": [int(s) for s in args.severities.split(",")],
            "in_domain_source": "ImageNet-1k val",
        },
        "result": {
            "in_domain": in_domain,
            "cells": cells,
            "by_corruption": by_corruption,
            "n_cells_above_in_domain_fvu": len(above),
            "n_corruptions_monotonically_increasing": increasing,
            "n_corruptions_tested": len(args.corruptions.split(",")),
        },
    }

    gate_can_fire = len(above) > 0
    report["verdict"] = "pass" if gate_can_fire and increasing > 0 else "fail"
    report["verdict_reason"] = (
        f"in-domain FVU를 넘는 OOD 셀 {len(above)}개, severity 단조증가 corruption "
        f"{increasing}/{len(args.corruptions.split(','))}개. "
        + ("게이트 발동 가능." if gate_can_fire else "게이트가 발동할 수 없다 -> M7 전제 불성립.")
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"{'condition':<34}{'FVU':>12}{'vs in-dom':>12}{'L0':>9}")
    print(f"{'in-domain (ImageNet val)':<34}{in_domain['fvu']:>12.3e}{'1.00x':>12}{in_domain['l0']:>9.1f}")
    for c in cells:
        label = f"{c['corruption']} sev{c['severity']}"
        print(f"{label:<34}{c['fvu']:>12.3e}{c['fvu_ratio_vs_in_domain']:>11.2f}x{c['l0']:>9.1f}")
    print(f"\nverdict={report['verdict']} — {report['verdict_reason']}")
    print(f"리포트 저장: {out}")


if __name__ == "__main__":
    main()
