#!/usr/bin/env python
"""T1.1 — 실험 1: SAE 계기 신뢰성 + OOD FVU 게이트.

M7(FVU 런타임 게이트)의 전제("shift가 오면 FVU가 올라간다")를 in-domain(ImageNet-1k
val) vs OOD(ImageNet-C 15 test corruption x 5 severity) 전 격자에서 확정 측정한다.
[scripts/probe_fvu_shift.py](../scripts/probe_fvu_shift.py)의 예비 probe(셀당 128장,
3 corruption)는 FVU가 severity에 대해 반대로(감소) 움직이는 것을 봤다 — 이 스크립트가
그 방향을 15x5 전 격자·셀당 512장으로 확정한다. 이겼든 졌든 그대로 보고한다
(docs/plans/contracts/phase1.md T1.1 섹션 참조).

FVU/L0는 `Model.sae_runtime.FrozenSAE.fvu`/`.l0`를 **그대로** 쓴다 — 이 실험이 정한
게이트 임계를 M7이 런타임에 그대로 주입받으므로, 척도가 갈리면 게이트가 무의미해진다.
재구현하지 않는다.

사용:
    python -m experiments.exp1_sae_instrument \
        --sae outputs/reservoir_sae/vit_b_sae.pt \
        --corruptions all --severities 1,2,3,4,5 \
        --max-images-per-cell 512 \
        --out outputs/experiments/T1.1
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from timm.data import create_transform, resolve_model_data_config

from Model.sae_runtime import FrozenSAE
from Utils.datasets import build_dataset

TASK_ID = "T1.1"

# ImageNet-C 15 test corruption (spec 규약: HP holdout 4종 — gaussian_blur/saturate/
# spatter/speckle_noise — 는 여기 포함하지 않는다. Utils.datasets._IMAGENET_C_HP_CORRUPTIONS
# 와 상보 관계이며, 값은 spec에 고정된 것이라 여기 명시적으로 나열한다.)
REPORT_CORRUPTIONS = [
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "shot_noise",
    "snow",
    "zoom_blur",
]


class _TransformDataset(torch.utils.data.Dataset):
    """build_dataset이 준 (image, label[, group])에 timm transform을 씌운다.

    scripts/probe_fvu_shift.py의 동명 클래스와 동일한 계약이라 그대로 가져왔다
    (원본이 np.ndarray를 주든 PIL을 주든 PIL로 맞춘 뒤 transform을 적용한다)."""

    def __init__(self, dataset, transform):
        """원본 dataset과 timm transform을 받아 그대로 보관한다."""
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
        return self.transform(image), int(item[1])


def git_commit_hash() -> str:
    """현재 HEAD 커밋 해시(40자)를 반환한다. 실패 시 'unknown'."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


@torch.no_grad()
def _collect_tokens_and_orig_acc(model, loader, device: str, target_block: int):
    """block10 patch 토큰 전량(CPU float32)과 원본(무개입) forward 정확도를 한 pass에 같이 잰다.

    반환: (tokens [N_tokens,768] cpu float32, correct_orig:int, n_images:int).
    hook은 아무것도 바꾸지 않고 출력을 읽기만 한다(=collect_tokens_with_hook과 동일 계약이되,
    같은 pass에서 로짓도 같이 뽑아 forward를 한 번 아끼는 버전)."""
    token_chunks = []
    captured = {}
    correct = 0
    n_images = 0

    def hook(_module, _inputs, output):
        captured["tokens"] = output[:, 1:, :].reshape(-1, output.shape[-1]).detach().float().cpu()

    handle = model.blocks[target_block].register_forward_hook(hook)
    try:
        for images, labels in loader:
            captured.clear()
            logits = model(images.to(device))
            token_chunks.append(captured["tokens"])
            preds = logits.argmax(dim=-1).cpu()
            correct += int((preds == labels).sum().item())
            n_images += int(images.shape[0])
    finally:
        handle.remove()

    if not token_chunks:
        raise RuntimeError("토큰을 하나도 수집하지 못했다 — 데이터셋이 비어있을 수 있다.")
    tokens = torch.cat(token_chunks, dim=0)
    return tokens, correct, n_images


@torch.no_grad()
def _recon_accuracy(frozen: FrozenSAE, model, loader, device: str, target_block: int):
    """block10 patch 토큰을 `frozen.reconstruct()`(공식 왕복)로 치환한 뒤 계속 forward한 정확도.

    CLS 토큰은 손대지 않는다(SAE는 patch 토큰 스코프로 학습됐다 — ckpt의 token_scope="patch").
    반환: (correct_recon:int, n_images:int)."""
    correct = 0
    n_images = 0

    def hook(_module, _inputs, output):
        patches = output[:, 1:, :]
        b, t, d = patches.shape
        flat = patches.reshape(-1, d)
        recon = frozen.reconstruct(flat).reshape(b, t, d).to(output.dtype)
        new_output = output.clone()
        new_output[:, 1:, :] = recon
        return new_output

    handle = model.blocks[target_block].register_forward_hook(hook)
    try:
        for images, labels in loader:
            logits = model(images.to(device))
            preds = logits.argmax(dim=-1).cpu()
            correct += int((preds == labels).sum().item())
            n_images += int(images.shape[0])
    finally:
        handle.remove()
    return correct, n_images


@torch.no_grad()
def _pooled_fvu_l0_cosine(frozen: FrozenSAE, tokens: torch.Tensor, chunk_size: int = 4096) -> tuple[float, float, float]:
    """FrozenSAE.fvu()/.l0()와 **정확히 같은 공식**(전 원소 pooled mse/var, 토큰당 평균 활성 수)을
    청크 단위로 누적해 계산한다. 한 셀의 토큰(최대 512장 x 196패치 ~= 100k개)을 SAE 코드(K=12288)로
    한 번에 encode하면 GPU가 OOM나기 때문에 필요하다 — 공식을 바꾸는 게 아니라
    `frozen.normalize`/`frozen.encode`/`frozen.decode`(전부 원본 그대로) 호출을 청크로 나눠
    같은 분자(sse)·분모(pooled var)를 정확히 다시 조립하는 것뿐이다.

    반환: (fvu, l0, cosine). fvu/l0 정의는 Model.sae_runtime.FrozenSAE.fvu/.l0 docstring 참조."""
    sse_total = 0.0
    sum_total = 0.0
    sumsq_total = 0.0
    elem_total = 0
    l0_total = 0.0
    cos_total = 0.0
    tok_total = 0

    for start in range(0, tokens.shape[0], chunk_size):
        chunk = tokens[start : start + chunk_size]
        x = frozen.normalize(chunk)
        z = frozen.encode(x, normalized=True)
        xh = frozen.decode(z)

        sse_total += ((x - xh) ** 2).sum().item()
        sum_total += x.sum().item()
        sumsq_total += (x**2).sum().item()
        elem_total += x.numel()
        l0_total += (z > frozen.active_threshold).sum().item()
        cos_total += F.cosine_similarity(x, xh, dim=1).sum().item()
        tok_total += x.shape[0]
        del x, z, xh

    mse = sse_total / max(1, elem_total)
    var = max(sumsq_total / max(1, elem_total) - (sum_total / max(1, elem_total)) ** 2, 1e-12)
    fvu = mse / var
    l0 = l0_total / max(1, tok_total)
    cosine = cos_total / max(1, tok_total)
    return fvu, l0, cosine


def measure_cell(
    frozen: FrozenSAE,
    model,
    transform,
    dataset,
    n: int,
    device: str,
    batch_size: int,
    num_workers: int,
    want_per_image_fvu: bool = False,
) -> dict:
    """데이터셋 앞 n장에서 in-domain/OOD 셀 하나의 전 지표를 잰다.

    FVU/L0/reconstruct는 전부 FrozenSAE의 공식 메서드를 그대로 호출한다(재구현 없음).
    정확도는 원본 forward(acc_orig)와 block10 patch 토큰을 SAE 왕복으로 치환한
    forward(acc_recon) 두 pass로 잰다. want_per_image_fvu=True면 게이트 임계를 뽑을 때
    쓸 이미지 단위 FVU 분포(`frozen.fvu`를 이미지 하나 분량 토큰에 개별 호출)도 같이 낸다.

    반환 키: n_images, n_tokens, fvu, l0, cosine, acc_orig, acc_recon, recon_cost,
    per_image_fvu(want_per_image_fvu=False면 None)."""
    ds = _TransformDataset(dataset, transform)
    subset = torch.utils.data.Subset(ds, list(range(min(n, len(ds)))))
    loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    tokens, correct_orig, n_images = _collect_tokens_and_orig_acc(model, loader, device, frozen.meta["target_block"])
    tokens = tokens.to(device)

    fvu, l0, cosine = _pooled_fvu_l0_cosine(frozen, tokens)

    per_image_fvu = None
    if want_per_image_fvu:
        tokens_per_image = tokens.shape[0] // max(1, n_images)
        per_image_fvu = [
            frozen.fvu(tokens[i * tokens_per_image : (i + 1) * tokens_per_image]) for i in range(n_images)
        ]

    correct_recon, n_images_recon = _recon_accuracy(frozen, model, loader, device, frozen.meta["target_block"])
    if n_images_recon != n_images:
        raise RuntimeError(
            f"acc_orig/acc_recon 이미지 수가 어긋났다: {n_images} vs {n_images_recon} — 셀 평균이 무효화된다."
        )

    return {
        "n_images": n_images,
        "n_tokens": int(tokens.shape[0]),
        "fvu": fvu,
        "l0": l0,
        "cosine": cosine,
        "acc_orig": correct_orig / n_images,
        "acc_recon": correct_recon / n_images,
        "recon_cost": correct_recon / n_images - correct_orig / n_images,
        "per_image_fvu": per_image_fvu,
    }


def _mean_over(cells: list[dict], key: str) -> float:
    """cells 리스트에서 key의 단순 평균. 셀당 이미지 수가 전부 동일하다는 전제 하에서만 유효
    (main()에서 max_images_per_cell을 전 셀 동일하게 고정해 이 전제를 지킨다)."""
    return float(np.mean([c[key] for c in cells]))


def build_ood_summary(ood_cells: list[dict], severities: list[int], corruptions: list[str]) -> dict:
    """ood_summary 블록(scope="15-test")을 산출한다. docs/plans/contracts/phase1.md의
    집계 규약: by_severity는 corruption 15개 평균, by_corruption은 severity 5개 평균."""
    by_severity = []
    for sev in severities:
        cells = [c for c in ood_cells if c["severity"] == sev]
        by_severity.append(
            {
                "severity": sev,
                "acc_orig": _mean_over(cells, "acc_orig"),
                "acc_recon": _mean_over(cells, "acc_recon"),
                "recon_cost": _mean_over(cells, "recon_cost"),
                "fvu": _mean_over(cells, "fvu"),
                "l0": _mean_over(cells, "l0"),
            }
        )

    by_corruption = []
    for corr in corruptions:
        cells = [c for c in ood_cells if c["corruption"] == corr]
        by_corruption.append(
            {
                "corruption": corr,
                "acc_orig": _mean_over(cells, "acc_orig"),
                "acc_recon": _mean_over(cells, "acc_recon"),
                "recon_cost": _mean_over(cells, "recon_cost"),
                "fvu": _mean_over(cells, "fvu"),
                "l0": _mean_over(cells, "l0"),
            }
        )

    return {
        "scope": "15-test",
        "n_cells": len(ood_cells),
        "mean_acc_orig": _mean_over(ood_cells, "acc_orig"),
        "mean_acc_recon": _mean_over(ood_cells, "acc_recon"),
        "mean_recon_cost": _mean_over(ood_cells, "recon_cost"),
        "mean_fvu": _mean_over(ood_cells, "fvu"),
        "mean_l0": _mean_over(ood_cells, "l0"),
        "by_severity": by_severity,
        "by_corruption": by_corruption,
    }


def compute_monotonicity(ood_cells: list[dict], corruptions: list[str], severities: list[int]) -> dict:
    """corruption별로 FVU가 severity에 대해 단조 비감소인지 본다(scripts/probe_fvu_shift.py와
    동일한 정의: 인접 severity 쌍 전부 b>=a). 판정 규칙의 '80% corruption에서 단조 증가' 조건에 쓴다."""
    per_corruption = {}
    n_increasing = 0
    for corr in corruptions:
        series = sorted((c for c in ood_cells if c["corruption"] == corr), key=lambda c: c["severity"])
        fvus = [c["fvu"] for c in series]
        mono_up = all(b >= a for a, b in zip(fvus, fvus[1:]))
        per_corruption[corr] = {"fvu_by_severity": fvus, "monotonically_increasing": mono_up}
        n_increasing += int(mono_up)
    return {
        "per_corruption": per_corruption,
        "n_monotonically_increasing": n_increasing,
        "n_corruptions": len(corruptions),
        "fraction_monotonically_increasing": n_increasing / max(1, len(corruptions)),
    }


def render_result_md(result: dict, severities: list[int], corruptions: list[str]) -> str:
    """사람이 읽는 표(result.md)를 만든다."""
    r = result["result"]
    lines = [
        f"# {TASK_ID} — 실험 1: SAE 계기 신뢰성 + OOD FVU 게이트",
        "",
        f"verdict: **{result['verdict']}** — {result['verdict_reason']}",
        "",
        "## in-domain 기준선",
        "",
        "| dataset | n_images | fvu | l0 | cosine | acc |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| {result['config']['in_domain_source']} | {r['in_domain']['n_images']} "
            f"| {r['in_domain']['fvu']:.6g} | {r['in_domain']['l0']:.2f} "
            f"| {r['in_domain']['cosine']:.4f} | {r['in_domain']['acc']:.4f} |"
        ),
        "",
        f"게이트 임계(`p{result['config']['gate_percentile']}` of in-domain per-image FVU 분포, "
        f"n={len(r['in_domain']['per_image_fvu'] or [])}): **{r['gate_threshold']:.6g}**",
        "",
        "## OOD 셀 (15 test corruption x 5 severity)",
        "",
        "| corruption | severity | fvu | l0 | acc_orig | acc_recon | recon_cost | above_gate |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for corr in corruptions:
        for sev in severities:
            cell = next(c for c in r["ood"] if c["corruption"] == corr and c["severity"] == sev)
            above = "Y" if cell["fvu"] > r["gate_threshold"] else ""
            lines.append(
                f"| {corr} | {sev} | {cell['fvu']:.6g} | {cell['l0']:.1f} "
                f"| {cell['acc_orig']:.4f} | {cell['acc_recon']:.4f} | {cell['recon_cost']:+.4f} | {above} |"
            )

    lines += [
        "",
        "## by_severity (corruption 15개 평균)",
        "",
        "| severity | fvu | l0 | acc_orig | acc_recon | recon_cost |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in r["ood_summary"]["by_severity"]:
        lines.append(
            f"| {row['severity']} | {row['fvu']:.6g} | {row['l0']:.1f} "
            f"| {row['acc_orig']:.4f} | {row['acc_recon']:.4f} | {row['recon_cost']:+.4f} |"
        )

    lines += [
        "",
        "## by_corruption (severity 5개 평균)",
        "",
        "| corruption | fvu | l0 | acc_orig | acc_recon | recon_cost | fvu monotonic-up |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    mono = r["monotonicity"]["per_corruption"]
    for row in r["ood_summary"]["by_corruption"]:
        mono_up = "Y" if mono[row["corruption"]]["monotonically_increasing"] else ""
        lines.append(
            f"| {row['corruption']} | {row['fvu']:.6g} | {row['l0']:.1f} "
            f"| {row['acc_orig']:.4f} | {row['acc_recon']:.4f} | {row['recon_cost']:+.4f} | {mono_up} |"
        )

    lines += [
        "",
        "## 요약",
        "",
        f"- `gate_threshold` = {r['gate_threshold']:.6g} (p{result['config']['gate_percentile']}, source={result['config']['in_domain_source']})",
        f"- `cells_above_gate` = {r['cells_above_gate']} / {r['ood_summary']['n_cells']}",
        (
            f"- `fvu_vs_dacc_pearson` = {r['fvu_vs_dacc_pearson']:.4f}"
            if r["fvu_vs_dacc_pearson"] is not None
            else "- `fvu_vs_dacc_pearson` = null (fvu 또는 recon_cost 분산이 0이라 정의되지 않음)"
        ),
        (
            f"- FVU severity 단조증가 corruption 비율 = {r['monotonicity']['n_monotonically_increasing']}"
            f"/{r['monotonicity']['n_corruptions']} "
            f"({r['monotonicity']['fraction_monotonically_increasing']:.2%})"
        ),
        "",
        "## SAE-source ablation",
        "",
        f"- source: 위 측정 그대로 (checkpoint={result['config']['checkpoint']})",
        f"- proxy: null — {result['result']['source_ablation']['proxy_reason']}",
        f"- public: null — {result['result']['source_ablation']['public_reason']}",
    ]
    return "\n".join(lines) + "\n"


def plot_fvu_vs_severity(r: dict, severities: list[int], corruptions: list[str], out_path: Path):
    """corruption별 FVU-vs-severity 곡선 + in-domain/게이트 임계 기준선."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for corr in corruptions:
        series = sorted((c for c in r["ood"] if c["corruption"] == corr), key=lambda c: c["severity"])
        ax.plot([c["severity"] for c in series], [c["fvu"] for c in series], marker="o", label=corr, alpha=0.8)
    ax.axhline(r["in_domain"]["fvu"], color="black", linestyle="--", linewidth=1.5, label="in-domain")
    ax.axhline(r["gate_threshold"], color="red", linestyle=":", linewidth=1.5, label="gate_threshold")
    ax.set_xlabel("severity")
    ax.set_ylabel("FVU")
    ax.set_title("T1.1 — FVU vs severity (15 test corruptions)")
    ax.set_xticks(severities)
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_acc_vs_severity(r: dict, out_path: Path):
    """acc_orig/acc_recon 두 곡선(by_severity, 15 corruption 평균)을 겹쳐 그려 SAE 왕복이
    치르는 대가(recon_cost)가 severity에 따라 벌어지는지 본다."""
    by_sev = r["ood_summary"]["by_severity"]
    severities = [row["severity"] for row in by_sev]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(severities, [row["acc_orig"] for row in by_sev], marker="o", label="acc_orig")
    ax.plot(severities, [row["acc_recon"] for row in by_sev], marker="s", label="acc_recon")
    ax.axhline(r["in_domain"]["acc"], color="black", linestyle="--", linewidth=1, label="in-domain acc_orig")
    ax.set_xlabel("severity")
    ax.set_ylabel("accuracy")
    ax.set_title("T1.1 — acc_orig vs acc_recon by severity (15-corruption mean)")
    ax.set_xticks(severities)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """in-domain 기준선 -> OOD 15x5 격자 -> 게이트 임계·상관·판정까지 전 동작을 실행하고
    result.json/result.md/figures를 outputs/experiments/T1.1/ 아래에 쓴다."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", default="outputs/reservoir_sae/vit_b_sae.pt", help="source SAE checkpoint")
    ap.add_argument("--sae-proxy", default=None, help="unlabeled proxy(imagenette) SAE checkpoint (없으면 skip)")
    ap.add_argument("--sae-public", default=None, help="공개 pretrained SAE checkpoint (없으면 skip)")
    ap.add_argument("--corruptions", default="all", help="'all' 또는 콤마구분 corruption 이름")
    ap.add_argument("--severities", default="1,2,3,4,5")
    ap.add_argument("--max-images-per-cell", type=int, default=512, help="OOD 셀당 이미지 수(전 셀 동일 고정)")
    ap.add_argument("--n-indomain", type=int, default=2048, help="in-domain 기준선 + 게이트 분포용 이미지 수")
    ap.add_argument("--gate-percentile", type=float, default=95.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"outputs/experiments/{TASK_ID}")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    corruptions = REPORT_CORRUPTIONS if args.corruptions == "all" else args.corruptions.split(",")
    unknown = [c for c in corruptions if c not in REPORT_CORRUPTIONS]
    if unknown:
        raise ValueError(f"report(15-test) 집합에 없는 corruption: {unknown} — 15-test={REPORT_CORRUPTIONS}")
    severities = [int(s) for s in args.severities.split(",")]

    frozen = FrozenSAE.from_checkpoint(args.sae, device=device)
    model = timm.create_model(frozen.meta["model_name"], pretrained=True).eval().to(device)
    transform = create_transform(**resolve_model_data_config(model), is_training=False)

    print(f"device={device} corruptions={len(corruptions)} severities={severities} "
          f"max_images_per_cell={args.max_images_per_cell} n_indomain={args.n_indomain}")

    # 1) in-domain 기준선 (게이트 분포도 여기서 같이 뽑는다)
    ds_indomain, _ = build_dataset("imagenet", split="val", root=args.data_root)
    print("measuring in-domain (ImageNet-1k val)...")
    in_domain = measure_cell(
        frozen, model, transform, ds_indomain, args.n_indomain, device,
        args.batch_size, args.num_workers, want_per_image_fvu=True,
    )
    in_domain["acc"] = in_domain["acc_orig"]

    gate_threshold = float(np.percentile(in_domain["per_image_fvu"], args.gate_percentile))

    # 2) OOD 15 x 5 격자
    ood_cells = []
    for corr in corruptions:
        for sev in severities:
            print(f"measuring OOD cell: {corr} sev{sev} ...")
            ds_ood, meta = build_dataset(
                "imagenet-c", split="report", root=args.data_root, corruption=corr, severity=sev
            )
            if meta.num_samples < args.max_images_per_cell:
                raise RuntimeError(
                    f"{corr} sev{sev}: 셀에 {meta.num_samples}장뿐이라 max_images_per_cell="
                    f"{args.max_images_per_cell}을 못 채운다 — 전 셀 동일 이미지 수 전제가 깨진다."
                )
            cell = measure_cell(
                frozen, model, transform, ds_ood, args.max_images_per_cell, device,
                args.batch_size, args.num_workers, want_per_image_fvu=False,
            )
            cell["corruption"] = corr
            cell["severity"] = sev
            cell.pop("per_image_fvu", None)
            ood_cells.append(cell)

    ood_summary = build_ood_summary(ood_cells, severities, corruptions)
    monotonicity = compute_monotonicity(ood_cells, corruptions, severities)
    cells_above_gate = sum(1 for c in ood_cells if c["fvu"] > gate_threshold)

    fvu_arr = np.array([c["fvu"] for c in ood_cells])
    dacc_arr = np.array([c["recon_cost"] for c in ood_cells])
    if np.std(fvu_arr) > 0 and np.std(dacc_arr) > 0:
        fvu_vs_dacc_pearson = float(np.corrcoef(fvu_arr, dacc_arr)[0, 1])
    else:
        fvu_vs_dacc_pearson = None

    unknown_fields = []
    if fvu_vs_dacc_pearson is None:
        unknown_fields.append(
            {
                "field": "fvu_vs_dacc_pearson",
                "reason": "75개 OOD 셀에서 fvu 또는 recon_cost 분산이 0이라 상관계수가 정의되지 않는다.",
            }
        )

    # 3) SAE-source ablation — 이 세션에서 확보한 건 source ckpt뿐이다. proxy/public은
    #    별도 ckpt가 없어 측정 불가이지 절대 0/기본값으로 채우지 않는다(unknown != absent).
    proxy_reason = "imagenette-trained proxy SAE ckpt가 존재하지 않는다. 학습은 T1.1 범위 밖."
    public_reason = "공개 pretrained SAE ckpt를 받아오지 않았다(다운로드/변환은 T1.1 범위 밖)."
    if args.sae_proxy is None:
        unknown_fields.append({"field": "source_ablation.proxy", "reason": proxy_reason})
    if args.sae_public is None:
        unknown_fields.append({"field": "source_ablation.public", "reason": public_reason})

    fail_reason_no_gate = cells_above_gate == 0
    pass_ok = (not fail_reason_no_gate) and monotonicity["fraction_monotonically_increasing"] >= 0.8
    if fail_reason_no_gate:
        verdict = "fail"
        verdict_reason = (
            f"전 OOD 셀(n={len(ood_cells)}) 중 게이트 임계({gate_threshold:.6g})를 넘은 셀이 0개다 — "
            "게이트가 한 번도 발동할 수 없다는 뜻이고 M7이 무의미해진다. "
            f"FVU severity 단조증가 corruption {monotonicity['n_monotonically_increasing']}/"
            f"{monotonicity['n_corruptions']}."
        )
    elif pass_ok:
        verdict = "pass"
        verdict_reason = (
            f"FVU가 corruption의 {monotonicity['fraction_monotonically_increasing']:.0%}에서 severity에 대해 "
            f"단조 증가하고, 게이트 임계({gate_threshold:.6g}) 밖 셀이 {cells_above_gate}개 존재한다."
        )
    else:
        verdict = "fail"
        verdict_reason = (
            f"게이트 임계를 넘은 셀은 {cells_above_gate}개 있으나, FVU가 severity에 대해 단조 증가하는 "
            f"corruption이 {monotonicity['fraction_monotonically_increasing']:.0%}로 80% 기준을 못 채운다 — "
            "게이트라는 설계가 이 백본·SAE 조합에서 성립하지 않는다는 뜻."
        )

    finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    result_payload = {
        "in_domain": in_domain,
        "ood": ood_cells,
        "ood_summary": ood_summary,
        "gate_threshold": gate_threshold,
        "gate_percentile": args.gate_percentile,
        "cells_above_gate": cells_above_gate,
        "fvu_vs_dacc_pearson": fvu_vs_dacc_pearson,
        "monotonicity": monotonicity,
        "source_ablation": {
            "source": {
                "checkpoint": args.sae,
                "in_domain": {k: v for k, v in in_domain.items() if k != "per_image_fvu"},
                "ood_summary": ood_summary,
                "gate_threshold": gate_threshold,
                "cells_above_gate": cells_above_gate,
            },
            "proxy": None,
            "proxy_reason": proxy_reason,
            "public": None,
            "public_reason": public_reason,
        },
    }

    result = {
        "task_id": TASK_ID,
        "git_commit": git_commit_hash(),
        "seed": args.seed,
        "started_at": started_at,
        "finished_at": finished_at,
        "config": {
            "checkpoint": args.sae,
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_version": torch.__version__,
            "model_name": frozen.meta["model_name"],
            "target_block": frozen.meta["target_block"],
            "token_scope": frozen.meta["token_scope"],
            "active_threshold": frozen.active_threshold,
            "corruptions": corruptions,
            "severities": severities,
            "max_images_per_cell": args.max_images_per_cell,
            "n_indomain": args.n_indomain,
            "in_domain_source": "ImageNet-1k val (HF ILSVRC/imagenet-1k)",
            "gate_percentile": args.gate_percentile,
            "gate_threshold_source": (
                f"in-domain(ImageNet-1k val, n={args.n_indomain} images) 이미지 단위 FVU 분포의 "
                f"p{args.gate_percentile} — imagenette 등 다른 분포를 썼다면 임계가 달라진다"
                "(M1 산출물 기준 in-domain FVU가 데이터셋별로 최대 5배 흔들림, phase1.md 참조)"
            ),
        },
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "result": result_payload,
        "unknown_fields": unknown_fields,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    result_md = render_result_md(result, severities, corruptions)
    (out_dir / "result.md").write_text(result_md, encoding="utf-8")

    figures_dir = out_dir / "figures"
    plot_fvu_vs_severity(result_payload, severities, corruptions, figures_dir / "fvu_vs_severity.png")
    plot_acc_vs_severity(result_payload, figures_dir / "acc_vs_severity.png")

    print(f"\nverdict={verdict} — {verdict_reason}")
    print(f"gate_threshold={gate_threshold:.6g} (p{args.gate_percentile}) cells_above_gate={cells_above_gate}")
    print(f"fvu_vs_dacc_pearson={fvu_vs_dacc_pearson}")
    print(f"결과 저장: {out_dir}/result.json, {out_dir}/result.md, {figures_dir}/*.png")


if __name__ == "__main__":
    main()
