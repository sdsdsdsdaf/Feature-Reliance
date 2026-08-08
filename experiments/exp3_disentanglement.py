#!/usr/bin/env python
"""T1.3 — 실험 3: spurious/causal 분리 검증 (Waterbirds).

thesis의 falsifiable 핵심 — SAE latent 좌표계가 spurious 개념(배경)과 causal
개념(새)을 실제로 분리하는가. docs/plans/contracts/phase1.md T1.3 절이 정본이다.

3층 구조:
  ① 정답    — Waterbirds y/place 라벨의 층화 AUC로 spur_j/caus_j를 정의(순환논증 방지).
  ② 진단    — 라벨 없는 s_k(발화율 shift)·source 라벨만 쓰는 c_k(ablation)가 ①을 복원하는가.
  ③ 해석    — 순수 후보 40개에 Broden IoU로 개념 이름을 붙인다(보조, 판정 게이트 아님).

명제 (A) 분리 — SAE의 purity_rate가 PCA·random보다 유의하게 높은가 (주 결과)
명제 (B) 진단 타당성 — s_k 랭킹이 ①의 순수-spurious 집합을 유의하게 복원하는가
명제 (C) 해석성 — Broden 개념 이름이 붙는가 (보조, 판정 미참여)

사용:
    python -m experiments.exp3_disentanglement \
        --sae outputs/reservoir_sae/vit_b_sae.pt \
        --broden-root data/broden1_227 \
        --out outputs/experiments/T1.3
"""

from __future__ import annotations

import argparse
import datetime
import json
import resource
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from timm.data import create_transform, resolve_model_data_config
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

import broden as broden_mod
from Model.diagnostics import DEFAULT_FIRING_RATE_FLOOR, compute_c_k
from Model.gain_basis import LatentGainBasis
from Model.intervention import GainIntervention
from Model.sae_runtime import FrozenSAE
from Utils.broden_utils import discover_broden, make_image_preprocessor
from Utils.datasets import build_dataset
from experiments.exp3_stats import (
    ap_permutation_p,
    purity_metrics,
    stratified_auc_effect,
    two_proportion_permutation_p,
)

TASK_ID = "T1.3"
SEL_PURE_THRESHOLD = 0.6


# =========================================================
# ConceptAxis 구현 — 세 축이 같은 (scores, ablate) 파이프라인을 탄다 (phase1.md 표)
# =========================================================


class SAEAxis:
    """SAE 개념 딕셔너리 축. scores=sae.encode(정규화 공간에서 이미 ReLU라 >=0)."""

    def __init__(self, sae: FrozenSAE, feature_ids: Tensor):
        """feature_ids: 이 축에서 실제로 다룰 latent index(alive latents만, 전체 K가 아니다)."""
        self.sae = sae
        self.name = "sae"
        self.feature_ids = feature_ids
        self.n_features = int(feature_ids.numel())

    def scores(self, h: Tensor) -> Tensor:
        """[N,768] -> [N,n_features]. alive latent 열만 골라낸다."""
        code = self.sae.encode(h)
        return code.index_select(1, self.feature_ids.to(code.device))

    def ablate(self, h: Tensor, j: int) -> Tensor:
        """feature_ids[j] 하나의 기여만 제거한 활성. c_k와 별도 경로이며 인터페이스 충족용."""
        latent = int(self.feature_ids[j].item())
        code = self.sae.encode(h)
        onehot = torch.zeros_like(code)
        onehot[:, latent] = code[:, latent]
        return h - self.sae.decode_delta(onehot)


class LinearProjectionAxis:
    """PCA/Random 공용 — x=normalize(h) 위에서 정사영 E로 좌표를 바꾸는 축.
    scores/ablate 둘 다 "그 방향 성분을 사영 제거"라는 같은 대수로 정의된다(phase1.md 표)."""

    def __init__(self, sae: FrozenSAE, mean: Tensor, components: Tensor, name: str):
        """mean/components는 normalize(h) 공간에서 정의된다(SAE와 같은 입력 공간을 공유).
        components: [n_features, 768], 행이 unit-norm 기저벡터(PCA는 고유벡터, random은 무작위 unit)."""
        self.sae = sae
        self.mean = mean
        self.components = components
        self.name = name
        self.n_features = int(components.shape[0])

    def scores(self, h: Tensor) -> Tensor:
        """[N,768] -> [N,n_features]. 부호 있는 사영값(AUC가 순위 기반이라 ± 처리가 불필요)."""
        x = self.sae.normalize(h)
        return (x - self.mean) @ self.components.T

    def ablate(self, h: Tensor, j: int) -> Tensor:
        """방향 e_j 성분만 사영 제거. TACT의 trim 1개와 동일 대수(phase1.md)."""
        x = self.sae.normalize(h)
        e_j = self.components[j : j + 1]  # [1,768]
        coeff = ((x - self.mean) @ e_j.T)  # [N,1]
        delta_normalized = coeff * e_j  # [N,768]
        return h - delta_normalized * self.sae.token_std


def build_pca_axis(sae: FrozenSAE, mean: Tensor, components: Tensor) -> LinearProjectionAxis:
    """SAE가 학습된 것과 같은 source(ImageNet-1k) block10 패치 토큰에서 적합한 PCA 통계
    (mean/components, `stream_source_pass`가 스트리밍으로 누적)로 축을 조립한다
    (phase1.md: Waterbirds에서 적합하면 PCA만 test 도메인에 맞춰져 불공정)."""
    return LinearProjectionAxis(sae, mean, components, name="pca")


def make_random_axis(sae: FrozenSAE, dim: int, seed: int, device: str) -> LinearProjectionAxis:
    """같은 d=768, unit-norm, seed 고정 통제군. mean=0(비교 대상인 PCA와 달리 평행이동 없음)."""
    generator = torch.Generator().manual_seed(seed)
    r = torch.randn(dim, dim, generator=generator)
    r = r / r.norm(dim=1, keepdim=True)
    components = r.to(device)
    mean = torch.zeros(1, dim, device=device)
    return LinearProjectionAxis(sae, mean, components, name="random")


# =========================================================
# 데이터 래퍼
# =========================================================


class _WBTransformDataset(Dataset):
    """Waterbirds (image,y,group) 3-튜플에 timm transform을 씌운다."""

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, y, group = self.dataset[idx]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        return self.transform(image), int(y), int(group)


class _TwoTupleView(Dataset):
    """3-튜플 데이터셋에서 (image, y)만 보이게 하는 뷰. compute_c_k의 loader 계약(2-튜플)에 맞춘다."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, y, _group = self.dataset[idx]
        return image, y


class _ImageOnlyTransformDataset(Dataset):
    """(image,label) 2-튜플 데이터셋에 transform을 씌우고 label은 버린다(토큰 수집 전용)."""

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        image = item[0]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        return self.transform(image)


def git_commit_hash() -> str:
    """현재 HEAD 커밋 해시. 실패 시 'unknown'."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


# =========================================================
# 선행 조건 — Waterbirds linear probe (frozen backbone, backbone/SAE 불변)
# =========================================================


def train_probe(backbone, transform, device: str, data_root: str, *, batch_size: int, lr: float, steps: int, seed: int) -> nn.Linear:
    """frozen ImageNet ViT-B/16 위에 2-way linear probe만 학습한다. backbone은 절대 안 바뀐다
    (finetune하면 block10 활성이 바뀌어 SAE가 무효한 계기가 된다 — phase1.md 선행조건 절)."""
    torch.manual_seed(seed)
    ds, _meta = build_dataset("waterbirds", split="train", root=data_root)
    loader = DataLoader(_WBTransformDataset(ds, transform), batch_size=batch_size, shuffle=True, num_workers=4)

    probe = nn.Linear(768, 2).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    backbone.eval()

    step = 0
    while step < steps:
        for x, y, _g in loader:
            with torch.no_grad():
                feat = backbone.forward_head(backbone.forward_features(x.to(device)), pre_logits=True)
            logits = probe(feat)
            loss = F.cross_entropy(logits, y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step >= steps:
                break
    probe.eval()
    return probe


@torch.no_grad()
def evaluate_probe(backbone, probe, transform, device: str, data_root: str, split: str, batch_size: int) -> dict:
    """split(기본 test)에서 전체·그룹별 정확도를 잰다. 공개 Waterbirds 벤치마크(ERM-finetuned
    ResNet-50)와는 백본이 다르므로(frozen ViT-B/16 probe) 직접 비교 불가 — 호출부가 명시한다."""
    ds, _meta = build_dataset("waterbirds", split=split, root=data_root)
    loader = DataLoader(_WBTransformDataset(ds, transform), batch_size=batch_size, shuffle=False, num_workers=4)

    correct_by_group: dict[int, int] = defaultdict(int)
    n_by_group: dict[int, int] = defaultdict(int)
    correct = 0
    n = 0
    for x, y, g in loader:
        feat = backbone.forward_head(backbone.forward_features(x.to(device)), pre_logits=True)
        preds = probe(feat).argmax(dim=1).cpu()
        correct += int((preds == y).sum().item())
        n += y.shape[0]
        for gi, p_i, y_i in zip(g.tolist(), preds.tolist(), y.tolist()):
            n_by_group[gi] += 1
            correct_by_group[gi] += int(p_i == y_i)

    per_group_acc = {str(k): correct_by_group[k] / n_by_group[k] for k in sorted(n_by_group)}
    return {
        "split": split,
        "overall_acc": correct / n,
        "n": n,
        "per_group_acc": per_group_acc,
        "n_by_group": {str(k): v for k, v in sorted(n_by_group.items())},
        "note": (
            "frozen ImageNet ViT-B/16 backbone + 2-way linear probe(Adam)만 학습. "
            "공개 Waterbirds 벤치마크(ERM-finetuned ResNet-50)와 백본이 달라 직접 비교 불가."
        ),
    }


# =========================================================
# block10 patch token 스트리밍 — O(batch) 상주, 이미지 수와 무관
# =========================================================


@torch.no_grad()
def stream_source_pass(
    model, sae: FrozenSAE, loader: DataLoader, device: str, hook_block: int, shard_dir: Path, dtype=torch.float16,
) -> tuple[list[Path], Tensor, Tensor, Tensor]:
    """block{hook_block} patch 토큰(cls 제외)을 단일 순회로 훑으며 세 가지 소비처를 동시에 만든다.

    구버전은 `collect_block_tokens`가 배치마다 `chunks.append(...)`한 뒤 `torch.cat`으로
    이어붙여 [N,T,D] 하나로 반환했다 — Waterbirds 규모(11,788장 x 196 x 768)면 fp16이어도
    3.5GB, `n_imagenet_source`를 키우면 그대로 선형 증가한다(호스트 RAM). 그 큰 텐서를
    PCA 적합(`fit_pca_axis`)·threshold shard 기록(`write_raw_shards`)·alive latent 판정
    (`compute_alive_latents`, 내부에서 `sae.encode(piece)`를 토큰 전체에 반복)까지 세 번
    재사용했었다.

    여기서는 배치 하나를 넘어가는 토큰 텐서를 들고 있지 않는다 — 상주 메모리는
    O(batch)로 고정되고 이미지 수와 무관하다:

      1. fp16 raw 토큰을 배치마다 그 자리에서 shard로 디스크에 쓴다
         (`broden.compute_thresholds_from_shards`가 기대하는 `{"sample_ids", "x"}` 포맷 —
         디스크 총량은 이미지 수에 비례하지만 이건 RSS/VRAM이 아니라 디스크다).
      2. PCA 적합에 필요한 1차/2차 모멘트(정규화 공간, `Model/diagnostics.py`의
         스트리밍 패스와 같은 원리 — 데이터가 바깥, 누적기만 안에 산다)를 D x D
         공분산 하나로만 누적한다. 원본 토큰을 다시 들고 있을 필요가 없다.
      3. alive latent 판정에 쓰는 발화 카운트(threshold=`sae.active_threshold`)를 같은
         pass에서 누적한다 — 별도 pass로 raw 토큰을 재훑지 않는다(구 `compute_alive_latents`
         가 하던 두 번째 전체 순회를 없앤다).

    반환: (shard_paths, pca_mean[1,D], pca_components[D,D] 고유벡터 내림차순, alive_ids)
    """
    captured = {}

    def hook(_module, _inputs, output):
        captured["x"] = output[:, 1:, :].detach()

    handle = model.blocks[hook_block].register_forward_hook(hook)

    d = sae.input_dim
    k = sae.hidden_dim
    sum_x = torch.zeros(d, dtype=torch.float64, device=device)
    sum_xxt = torch.zeros(d, d, dtype=torch.float64, device=device)
    fire_counts = torch.zeros(k, device=device)
    n_tokens = 0
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: list[Path] = []
    try:
        for shard_idx, batch in enumerate(loader):
            x_img = batch[0] if isinstance(batch, (list, tuple)) else batch
            model(x_img.to(device))
            tokens = captured["x"]  # [B,T,D], device 상
            flat = tokens.reshape(-1, tokens.shape[-1]).float()

            xn = sae.normalize(flat).double()
            sum_x += xn.sum(dim=0)
            sum_xxt += xn.T @ xn

            code = sae.encode(flat)
            fire_counts += (code > sae.active_threshold).float().sum(dim=0)
            n_tokens += flat.shape[0]

            shard_path = shard_dir / f"shard_{shard_idx:04d}.pt"
            n_imgs_in_shard = tokens.shape[0]
            torch.save(
                {"sample_ids": list(range(n_imgs_in_shard)), "x": tokens.to(device="cpu", dtype=dtype)}, shard_path
            )
            shard_paths.append(shard_path)
            del tokens, flat, xn, code
    finally:
        handle.remove()

    n_tokens = max(1, n_tokens)
    mean = (sum_x / n_tokens).float()
    cov = (sum_xxt / n_tokens).float() - torch.outer(mean, mean)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    components = eigvecs[:, order].T.contiguous()  # [D,D], 행 = 고유벡터(분산 내림차순)
    alive_ids = (fire_counts / n_tokens >= DEFAULT_FIRING_RATE_FLOOR).nonzero(as_tuple=True)[0].cpu()
    return shard_paths, mean.unsqueeze(0), components, alive_ids


# =========================================================
# 진단 층(②) — s_k(발화율 shift, 라벨 없음) / c_k(source 라벨만)
# =========================================================


# alive latent 판정(발화율 >= DEFAULT_FIRING_RATE_FLOOR, threshold=sae.active_threshold)은
# `stream_source_pass`가 shard 기록과 같은 pass에서 이미 누적한다 — 별도의 두 번째 전체
# 순회(구 `compute_alive_latents`)가 더는 필요 없다.


def compute_firing_rate_from_shards(sae: FrozenSAE, shard_paths: list[Path], latent_ids: Tensor, thresholds: dict, device: str) -> np.ndarray:
    """latent_ids 순서로 발화율(코드가 threshold를 넘는 토큰 비율)을 잰다. s_k(도메인 단위,
    라벨 없음)의 두 항(target/source) 각각에 이 함수를 부른다.

    구버전은 `raw_tokens`(전 이미지 fp16 텐서, `collect_block_tokens`의 반환값)를 인자로
    받아 8,192토큰씩 청크로 재훑었다 — 청크 자체는 작았지만, 그 상위의 `raw_tokens`가
    이미지 수에 선형인 큰 텐서였다(site 1과 같은 문제의 소비처). 여기서는 `stream_source_pass`
    가 배치마다 이미 디스크에 써 둔 shard를 하나씩 다시 읽는다 — shard 하나가 곧 원래
    배치 크기라 상주 메모리가 O(batch)로 고정되고, 이미지 수·shard 개수와 무관하다."""
    thr = torch.tensor([thresholds[int(k)] for k in latent_ids.tolist()], dtype=torch.float32, device=device)
    fire_counts = torch.zeros(latent_ids.numel(), device=device)
    total = 0
    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu")
        flat = shard["x"].to(device=device, dtype=torch.float32).reshape(-1, shard["x"].shape[-1])
        code = sae.encode(flat).index_select(1, latent_ids.to(device))
        fire_counts += (code > thr).sum(dim=0)
        total += flat.shape[0]
        del shard, flat, code
    return (fire_counts / max(1, total)).cpu().numpy()


def build_balanced_group_subset(dataset, groups_all: list[int], seed: int):
    """4개 group에서 최소 group 크기만큼씩 뽑은 균형 부분집합(c_k^bal)."""
    rng = np.random.default_rng(seed)
    by_group: dict[int, list[int]] = defaultdict(list)
    for idx, g in enumerate(groups_all):
        by_group[g].append(idx)
    min_size = min(len(v) for v in by_group.values())
    picked = []
    for g in sorted(by_group):
        idx_g = np.array(by_group[g])
        picked.extend(rng.choice(idx_g, size=min_size, replace=False).tolist())
    picked.sort()
    return torch.utils.data.Subset(dataset, picked), min_size


# =========================================================
# Broden(③) — collect_activations_to_shards/compute_thresholds_from_shards/compute_iou 재사용
# =========================================================


def run_broden_iou(sae: FrozenSAE, token_stats: dict, model, latent_ids: list[int], args, target_block: int, token_scope: str) -> tuple[list[dict], dict]:
    """broden.py의 기존 파이프라인을 그대로 호출해 latent_ids의 개념 IoU를 얻는다(재구현 없음).
    반환: (compute_iou rows, best_rows_by_latent 결과)."""
    concept_by_id, samples = discover_broden(args.broden_root)
    categories = list(broden_mod.BRODEN_CATEGORIES)
    samples = [s for s in samples if any(a.category in categories for a in s.annotations)]
    category_subsets = {c: args.broden_subset for c in categories}
    samples, _counts = broden_mod.select_category_subsets(samples, categories, category_subsets, args.seed)
    if not samples:
        raise RuntimeError("Broden subset이 비었다 — --broden-subset을 늘려야 한다.")

    model_data_config = resolve_model_data_config(model)
    preprocess_pil, _image_to_np, _mask_to_grid, label_to_concept_grid, _meta = make_image_preprocessor(model_data_config)

    broden_args = SimpleNamespace(
        device=args.device,
        activation_cache_dir=str(Path(args.out) / "broden_cache" / f"block{target_block}"),
        num_activation_shards=8,
        rebuild_activation_cache=False,
        activation_cache_dtype="float16",
        batch_size=args.broden_batch_size,
        model_checkpoint=None,
        model_name=args.model_name,
        pretrained=True,
        grid_size=args.grid_size,
    )
    shard_paths = broden_mod.collect_activations_to_shards(samples, model, preprocess_pil, broden_args, target_block, token_scope)
    thresholds = broden_mod.compute_thresholds_from_shards(shard_paths, sae.sae, token_stats, latent_ids, percentile=99.0, chunk_size=256)
    sample_by_id = {int(s.sample_id): s for s in samples}
    latent_index = {int(lid): i for i, lid in enumerate(latent_ids)}
    iou_rows = broden_mod.compute_iou(
        broden_mod.iter_activation_records(shard_paths, sample_by_id, sae.sae, token_stats, latent_ids),
        concept_by_id,
        categories,
        latent_ids,
        thresholds,
        label_to_concept_grid,
        args.grid_size,
        latent_index=latent_index,
        total_records=len(sample_by_id),
    )
    best = broden_mod.best_rows_by_latent(iou_rows, latent_ids)
    return iou_rows, best


# =========================================================
# 렌더링
# =========================================================


def plot_sel_hist(axis_sel: dict, out_path: Path):
    """세 축의 sel_j(informative feature만) 히스토그램을 겹쳐 그린다. 쌍봉 vs 단봉이 핵심 그림."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, sel in axis_sel.items():
        if sel.size:
            ax.hist(sel, bins=30, range=(-1, 1), alpha=0.5, label=f"{name} (n={sel.size})", density=True)
    ax.axvline(SEL_PURE_THRESHOLD, color="black", linestyle=":", linewidth=1)
    ax.axvline(-SEL_PURE_THRESHOLD, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("sel_j = (spur-caus)/(spur+caus)")
    ax.set_ylabel("density")
    ax.set_title("T1.3 — sel_j 분포 (쌍봉=분리, 단봉=entangled)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_spur_vs_caus(axis_points: dict, out_path: Path):
    """(spur_j, caus_j) 산점도, 축별 패널."""
    n = len(axis_points)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    for i, (name, (spur, caus, informative)) in enumerate(axis_points.items()):
        ax = axes[0][i]
        ax.scatter(spur[~informative], caus[~informative], s=4, alpha=0.2, color="gray", label="non-informative")
        ax.scatter(spur[informative], caus[informative], s=6, alpha=0.6, color="crimson", label="informative")
        ax.set_xlabel("spur_j")
        ax.set_ylabel("caus_j")
        ax.set_title(name)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7)
    fig.suptitle("T1.3 — spur_j vs caus_j (축별)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_result_md(result: dict) -> str:
    r = result["result"]
    lines = [
        f"# {TASK_ID} — 실험 3: spurious/causal 분리 검증 (Waterbirds)",
        "",
        f"verdict: **{result['verdict']}** — {result['verdict_reason']}",
        "",
        "## 선행 조건 — Waterbirds linear probe (frozen backbone)",
        "",
        "| split | overall_acc | " + " | ".join(f"group{g}" for g in sorted(r["probe"]["per_group_acc"])) + " |",
        "|---|---:|" + "---:|" * len(r["probe"]["per_group_acc"]),
        (
            f"| {r['probe']['split']} | {r['probe']['overall_acc']:.4f} | "
            + " | ".join(f"{r['probe']['per_group_acc'][g]:.4f}" for g in sorted(r["probe"]["per_group_acc"]))
            + " |"
        ),
        "",
        f"_{r['probe']['note']}_",
        "",
        "## (A) 분리 — 축별 purity_rate",
        "",
        "| axis | n_features | tau_spur | tau_caus | n_informative | n_pure_spurious | n_pure_causal | n_entangled | purity_rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, sep in r["separation"].items():
        pr = sep["purity_rate"]
        pr_str = f"{pr:.4f}" if pr is not None else "null"
        lines.append(
            f"| {name} | {sep['n_features']} | {sep['tau_spur']:.4f} | {sep['tau_caus']:.4f} | "
            f"{sep['n_informative']} | {sep['n_pure_spurious']} | {sep['n_pure_causal']} | "
            f"{sep['n_entangled']} | {pr_str} |"
        )
    lines += [
        "",
        f"`purity_significance`: sae_vs_pca_p = {result['result']['purity_significance']['sae_vs_pca_p']}, "
        f"sae_vs_random_p = {result['result']['purity_significance']['sae_vs_random_p']} "
        f"({result['result']['purity_significance']['test']})",
        "",
        "## (B) 진단 타당성",
        "",
        f"- `s_k_ap_for_pure_spurious` = {r['diagnostic_validity']['s_k_ap_for_pure_spurious']:.4f} "
        f"(p={r['diagnostic_validity']['s_k_ap_p_value']:.4f}), "
        f"`random_baseline_ap` = {r['diagnostic_validity']['random_baseline_ap']:.4f}",
        f"- `s_times_1_minus_c_ap` = {r['diagnostic_validity']['s_times_1_minus_c_ap']:.4f}",
        f"- `corr_c_train_vs_caus` = {r['diagnostic_validity']['corr_c_train_vs_caus']}",
        f"- `corr_c_bal_vs_caus` = {r['diagnostic_validity']['corr_c_bal_vs_caus']}",
        f"- `c_train_misranks_background` = {r['diagnostic_validity']['c_train_misranks_background']}",
        f"- `c_k resolution`(1/n_images): train[{r['diagnostic_validity'].get('c_k_train_split', 'train')}] "
        f"n={r['diagnostic_validity']['c_k_train_n_images']} "
        f"(res={r['diagnostic_validity']['c_k_train_resolution']:.2e}), "
        f"bal[{r['diagnostic_validity'].get('c_k_bal_split', 'train')}] n={r['diagnostic_validity']['c_k_bal_n_images']} "
        f"(res={r['diagnostic_validity']['c_k_bal_resolution']:.2e})",
        "",
        f"_{r['diagnostic_validity'].get('c_k_bal_split_note', '')}_",
        "",
        "## (C) 해석성 — Broden top 40 (보조, 판정 미참여)",
        "",
        "| latent | spur | caus | sel | s_k | c_k_train | c_k_bal | top_concept | category | IoU |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    for row in r["interpretability"]["candidates"]:
        lines.append(
            f"| {row['latent']} | {row['spur']:.3f} | {row['caus']:.3f} | {row['sel']:.3f} | "
            f"{row['s_k']:.4f} | {row['c_k_train']:.4f} | {row['c_k_bal']:.4f} | "
            f"{row['top_concept']} | {row['category']} | {row['iou']:.3f} |"
        )
    return "\n".join(lines) + "\n"


# =========================================================
# main
# =========================================================


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", default="outputs/reservoir_sae/vit_b_sae.pt")
    ap.add_argument("--broden-root", default="data/broden1_227")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default=f"outputs/experiments/{TASK_ID}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-imagenet-source", type=int, default=2048, help="PCA 적합 + s_k source firing에 쓸 ImageNet 이미지 수")
    ap.add_argument("--sel-threshold", type=float, default=SEL_PURE_THRESHOLD)
    ap.add_argument("--top-n-broden", type=int, default=40)
    ap.add_argument("--broden-subset", type=int, default=16, help="Broden 카테고리별 이미지 상한(비용 통제)")
    ap.add_argument("--broden-subset-full", type=int, default=8, help="full_latent_iou 계산용 카테고리별 상한")
    ap.add_argument("--broden-batch-size", type=int, default=8)
    ap.add_argument("--grid-size", type=int, default=14)
    ap.add_argument("--probe-batch-size", type=int, default=64)
    ap.add_argument("--probe-lr", type=float, default=1e-2)
    ap.add_argument("--probe-steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32, help="block10 토큰 수집용 배치 크기")
    ap.add_argument("--c-k-batch-size", type=int, default=8, help="phaseM.md 실측 안전선")
    ap.add_argument(
        "--c-k-max-candidates", type=int, default=0,
        help="c_k를 잴 후보 수 상한(0=제한 없음). ① informative 집합에서 **무작위** 부분표본을 "
             "뽑는다 — 상위 N개로 자르면 corr(c_k, caus_j)가 편향되지만 무작위는 그렇지 않다.",
    )
    ap.add_argument("--c-k-latent-chunk", type=int, default=16, help="phaseM.md 실측 안전선(64는 OOM)")
    ap.add_argument("--c-k-max-images-train", type=int, default=None, help="디버그/스모크용. None이면 train split 전체(4795)")
    ap.add_argument("--c-k-max-images-bal", type=int, default=None, help="디버그/스모크용. None이면 균형 subset 전체")
    ap.add_argument("--skip-broden", action="store_true", help="빠른 재실행/디버그용. 산출물에 unknown_fields로 기록한다")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    unknown_fields = []
    t_run_start = time.monotonic()

    def _peak_rss_mb() -> float:
        """프로세스 시작 이후 누적 최고 RSS(MB). Linux `ru_maxrss`는 KB 단위."""
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    def _stage_start() -> float:
        """스테이지 시작 시각을 찍고, CUDA peak 카운터를 리셋해 이 스테이지만의 peak VRAM을
        따로 잴 수 있게 한다."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return time.monotonic()

    def _stage_done(label: str, t0: float) -> None:
        """스테이지 이름·소요시간·peak RSS·peak VRAM을 찍는다(요구사항: probe/①AUC/permutation
        null/c_k^train/c_k^bal/Broden 각 스테이지). peak VRAM은 `_stage_start`가 리셋한
        시점부터 이 스테이지 안에서의 최고값, peak RSS는 프로세스 전체 누적 최고값(OS 단위라
        스테이지별 리셋이 없다 — 그래도 단조증가라 "지금까지 최악"을 보여준다)."""
        elapsed = time.monotonic() - t0
        peak_rss = _peak_rss_mb()
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
        print(
            f"   [stage done: {label}] elapsed={elapsed:.1f}s peak_rss={peak_rss:.1f}MB "
            f"peak_vram={peak_vram:.1f}MB total_elapsed={time.monotonic() - t_run_start:.1f}s",
            flush=True,
        )

    frozen = FrozenSAE.from_checkpoint(args.sae, device=device)
    args.model_name = frozen.meta["model_name"]
    target_block = frozen.meta["target_block"]
    token_scope = frozen.meta["token_scope"]
    token_stats = {"mean": frozen.token_mean.cpu(), "std": frozen.token_std.cpu()}

    model = timm.create_model(args.model_name, pretrained=True).eval().to(device)
    transform = create_transform(**resolve_model_data_config(model), is_training=False)

    t0 = _stage_start()
    print("1/8 Waterbirds linear probe 학습 (frozen backbone)...")
    probe = train_probe(
        model, transform, device, args.data_root,
        batch_size=args.probe_batch_size, lr=args.probe_lr, steps=args.probe_steps, seed=args.seed,
    )
    probe_eval = evaluate_probe(model, probe, transform, device, args.data_root, split="test", batch_size=args.probe_batch_size)
    print(f"   probe overall_acc={probe_eval['overall_acc']:.4f} per_group={probe_eval['per_group_acc']}")
    _stage_done("probe", t0)

    t0 = _stage_start()
    print("2-4/8 ImageNet source 스트리밍 pass (PCA moment + alive latent + shard 기록, O(batch) 상주)...")
    ds_source, _meta = build_dataset("imagenet", split="val", root=args.data_root)
    subset_source = torch.utils.data.Subset(ds_source, list(range(min(args.n_imagenet_source, len(ds_source)))))
    loader_source = DataLoader(
        _ImageOnlyTransformDataset(subset_source, transform), batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    shard_dir = Path(args.out) / "_source_shards"
    shard_paths, pca_mean, pca_components, alive_ids = stream_source_pass(
        model, frozen, loader_source, device, target_block, shard_dir
    )
    n_alive = int(alive_ids.numel())
    print(f"   alive SAE latents = {n_alive} / {frozen.hidden_dim}")
    sae_axis = SAEAxis(frozen, alive_ids)
    pca_axis = build_pca_axis(frozen, pca_mean, pca_components)
    random_axis = make_random_axis(frozen, dim=frozen.input_dim, seed=args.seed, device=device)
    axes = {"sae": sae_axis, "pca": pca_axis, "random": random_axis}

    print("   source firing threshold(p99) 계산 (broden.compute_thresholds_from_shards 재사용)...")
    alive_ids_list = alive_ids.tolist()
    firing_thresholds = broden_mod.compute_thresholds_from_shards(
        shard_paths, frozen.sae, token_stats, alive_ids_list, percentile=99.0, chunk_size=512
    )
    source_firing = compute_firing_rate_from_shards(frozen, shard_paths, alive_ids, firing_thresholds, device)
    for p in shard_paths:
        p.unlink(missing_ok=True)
    _stage_done("source pass (PCA/alive/threshold/firing)", t0)

    print("5/8 Waterbirds test 스트림 — pooled axis scores(①) + s_k target firing(②)...")
    ds_test, meta_test = build_dataset("waterbirds", split="test", root=args.data_root)
    ds_test_t = _WBTransformDataset(ds_test, transform)
    loader_test = DataLoader(ds_test_t, batch_size=args.batch_size, shuffle=False, num_workers=4)

    pooled_max = {name: [] for name in axes}
    pooled_mean = {name: [] for name in axes}
    y_list, place_list = [], []
    firing_num = np.zeros(n_alive)
    firing_den = 0
    thr_tensor = torch.tensor([firing_thresholds[k] for k in alive_ids_list], dtype=torch.float32, device=device)

    captured = {}

    def hook(_module, _inputs, output):
        captured["x"] = output[:, 1:, :].detach()

    handle = model.blocks[target_block].register_forward_hook(hook)
    try:
        with torch.no_grad():
            for x, y, group in loader_test:
                model(x.to(device))
                h = captured["x"]  # [B,T,768]
                b, t, d = h.shape
                flat = h.reshape(-1, d)

                for name, axis in axes.items():
                    s = axis.scores(flat).reshape(b, t, -1)
                    pooled_max[name].append(s.max(dim=1).values.cpu().numpy())
                    pooled_mean[name].append(s.mean(dim=1).cpu().numpy())

                code_alive = frozen.encode(flat).index_select(1, alive_ids.to(device))
                firing_num += (code_alive > thr_tensor).sum(dim=0).cpu().numpy()
                firing_den += flat.shape[0]

                y_arr = y.numpy()
                place_arr = group.numpy() % 2
                y_list.append(y_arr)
                place_list.append(place_arr)
    finally:
        handle.remove()

    y_all = np.concatenate(y_list)
    place_all = np.concatenate(place_list)
    target_firing = firing_num / max(1, firing_den)
    s_k = np.abs(target_firing - source_firing)

    pooled_max = {name: np.concatenate(v, axis=0) for name, v in pooled_max.items()}
    pooled_mean = {name: np.concatenate(v, axis=0) for name, v in pooled_mean.items()}

    t0 = _stage_start()
    print("6/8 층화 AUC + permutation max-null (P=%d)..." % args.n_perm)
    separation = {}
    axis_sel_for_plot = {}
    axis_points_for_plot = {}
    purity_indicators = {}
    caus_by_axis = {}
    robustness_mean = {}

    for name, scores in pooled_max.items():
        rng_spur = np.random.default_rng(args.seed * 1000 + 1)
        rng_caus = np.random.default_rng(args.seed * 1000 + 2)
        spur, spur_null = stratified_auc_effect(scores, place_all, y_all, n_perm=args.n_perm, rng=rng_spur)
        caus, caus_null = stratified_auc_effect(scores, y_all, place_all, n_perm=args.n_perm, rng=rng_caus)
        tau_spur = float(np.quantile(spur_null, 0.95))
        tau_caus = float(np.quantile(caus_null, 0.95))
        metrics = purity_metrics(spur, caus, tau_spur, tau_caus, sel_threshold=args.sel_threshold)

        separation[name] = {k: v for k, v in metrics.items() if k not in ("sel", "informative", "pure_spurious", "pure_causal")}
        axis_sel_for_plot[name] = metrics["sel"][metrics["informative"]]
        axis_points_for_plot[name] = (spur, caus, metrics["informative"])
        purity_indicator = (metrics["pure_spurious"] | metrics["pure_causal"]).astype(int)[metrics["informative"]]
        purity_indicators[name] = purity_indicator
        caus_by_axis[name] = caus
        if name == "sae":
            sae_spur, sae_caus, sae_metrics = spur, caus, metrics

        # 강건성 병기(mean pooling) — 판정에 쓰지 않는 참고값
        scores_mean = pooled_mean[name]
        rng_spur_m = np.random.default_rng(args.seed * 1000 + 3)
        rng_caus_m = np.random.default_rng(args.seed * 1000 + 4)
        spur_m, spur_null_m = stratified_auc_effect(scores_mean, place_all, y_all, n_perm=args.n_perm, rng=rng_spur_m)
        caus_m, caus_null_m = stratified_auc_effect(scores_mean, y_all, place_all, n_perm=args.n_perm, rng=rng_caus_m)
        metrics_m = purity_metrics(
            spur_m, caus_m, float(np.quantile(spur_null_m, 0.95)), float(np.quantile(caus_null_m, 0.95)), sel_threshold=args.sel_threshold
        )
        robustness_mean[name] = metrics_m["purity_rate"]
    _stage_done("①AUC (stratified spur_j/caus_j, 축별 purity_rate)", t0)

    t0 = _stage_start()
    p_sae_vs_pca = two_proportion_permutation_p(
        purity_indicators["sae"], purity_indicators["pca"], n_perm=args.n_perm, rng=np.random.default_rng(args.seed * 1000 + 5)
    )
    p_sae_vs_random = two_proportion_permutation_p(
        purity_indicators["sae"], purity_indicators["random"], n_perm=args.n_perm, rng=np.random.default_rng(args.seed * 1000 + 6)
    )
    _stage_done("permutation null (tau_spur/tau_caus max-null + sae_vs_pca/random 유의성)", t0)

    print("7/8 c_k^train / c_k^bal (Model.diagnostics.compute_c_k 재사용)...")
    original_head = model.head
    model.head = probe
    basis = LatentGainBasis(frozen)
    intervention = GainIntervention(model, basis, hook_block=target_block, residual=True).to(device)

    # c_k 후보는 ①에서 informative로 판정된 latent만 쓴다(spur_j>=tau_spur 또는
    # caus_j>=tau_caus) — c_k는 (B)의 s_k AP 랭킹·corr(c_k,caus_j)와 (C)의 top-40 Broden에서만
    # 소비되고, 셋 다 non-informative latent에는 아무 관심이 없다. alive latent 전체(수천)
    # 대신 informative 부분집합(실측 ~800-1,000)만 candidates로 넘기면 Model/diagnostics.py의
    # CPU 캐시(4853022) 안에서 candidate chunking 없이 compute_c_k를 직접 호출할 수 있다.
    informative_positions = np.where(sae_metrics["informative"])[0]
    candidate_latent_ids = alive_ids[torch.from_numpy(informative_positions)]
    n_informative = candidate_latent_ids.numel()
    ck_subsampled = False
    if args.c_k_max_candidates and n_informative > args.c_k_max_candidates:
        # informative 전체를 재면 후보 수에 비례해 몇 시간이 든다. c_k의 쓰임새는
        # (B)의 AP 랭킹과 corr(c_k, caus_j)뿐이고 둘 다 표본으로 충분히 추정된다.
        # 상위 N이 아니라 **무작위**로 뽑는 이유: 상위만 남기면 c_k가 큰 쪽으로 잘린
        # 집합에서 상관을 재게 되어 값이 편향된다.
        g = torch.Generator().manual_seed(args.seed)
        pick = torch.randperm(n_informative, generator=g)[: args.c_k_max_candidates]
        candidate_latent_ids = candidate_latent_ids[pick.to(candidate_latent_ids.device)]
        ck_subsampled = True
    print(
        f"   c_k candidates (① informative) = {candidate_latent_ids.numel()} / {n_informative} informative"
        f" / {n_alive} alive" + ("  [무작위 부분표본]" if ck_subsampled else "")
    )

    ds_train, meta_train = build_dataset("waterbirds", split="train", root=args.data_root)
    ds_train_t = _WBTransformDataset(ds_train, transform)
    c_k_train_n = args.c_k_max_images_train or len(ds_train_t)
    loader_c_k_train = DataLoader(_TwoTupleView(ds_train_t), batch_size=args.c_k_batch_size, shuffle=False, num_workers=4)
    t0 = _stage_start()
    c_k_train_full = compute_c_k(
        intervention, loader_c_k_train, device,
        candidates=candidate_latent_ids, latent_chunk=args.c_k_latent_chunk, max_images=c_k_train_n,
    )
    c_k_train = c_k_train_full.index_select(0, alive_ids).numpy()
    _stage_done(f"c_k^train (n_images={c_k_train_n})", t0)

    # c_k^bal은 train이 아니라 test split에서 뽑는다. train의 최소 group(place_bird!=y 조합)은
    # 56장이라 4-group 균형 subset이 4x56=224장 — c_k 분해능(1/n_images)이 4.5e-3으로
    # 실측 c_k 크기(~1e-3)보다 조악해 전부 0으로 양자화된다(phaseM.md M6). test의 최소 group은
    # 642장(균형 2,568장, 분해능 3.9e-4)이라 이 문제가 없다. c_k^train은 그대로 train(균형이
    # 아닌 95% confound 그대로)을 쓴다 — c_k_train이 "편향된 분포에서 측정하면 어떻게 되는가"
    # 자체가 그 arm의 목적이라 바꾸지 않는다.
    #
    # 주의: c_k^bal과 층 ①의 정답(spur_j/caus_j, sae_metrics)이 이제 같은 split(test)에서
    # 나온다 — label leakage는 아니다(c_k는 accuracy 기여도, spur_j/caus_j는 라벨 대비 AUC로
    # 서로 다른 것을 잰다) 하지만 더는 독립 표본이 아니라는 점을 산출물에 명시한다
    # (result_payload.diagnostic_validity.c_k_bal_split_note).
    groups_all_test = [int(ds_test[i][2]) for i in range(len(ds_test))]
    balanced_subset, min_group_size = build_balanced_group_subset(ds_test_t, groups_all_test, seed=args.seed)
    c_k_bal_n = args.c_k_max_images_bal or len(balanced_subset)
    loader_c_k_bal = DataLoader(_TwoTupleView(balanced_subset), batch_size=args.c_k_batch_size, shuffle=False, num_workers=4)
    t0 = _stage_start()
    c_k_bal_full = compute_c_k(
        intervention, loader_c_k_bal, device,
        candidates=candidate_latent_ids, latent_chunk=args.c_k_latent_chunk, max_images=c_k_bal_n,
    )
    c_k_bal = c_k_bal_full.index_select(0, alive_ids).numpy()
    _stage_done(f"c_k^bal (n_images={c_k_bal_n})", t0)

    model.head = original_head  # 원상복구(방어적 — intervention이 이후 다시 안 쓰이지만 관례를 지킨다)

    ap_observed, ap_p = ap_permutation_p(
        sae_metrics["pure_spurious"][sae_metrics["informative"]].astype(int),
        s_k[sae_metrics["informative"]],
        n_perm=args.n_perm,
        rng=np.random.default_rng(args.seed * 1000 + 7),
    )
    random_baseline_ap = float((sae_metrics["pure_spurious"][sae_metrics["informative"]]).mean()) if sae_metrics["n_informative"] > 0 else 0.0

    s_times_1_minus_c = s_k * (1.0 - c_k_train)
    ap_combined, _p_combined = ap_permutation_p(
        (sae_metrics["pure_spurious"][sae_metrics["informative"]]).astype(int),
        s_times_1_minus_c[sae_metrics["informative"]],
        n_perm=args.n_perm,
        rng=np.random.default_rng(args.seed * 1000 + 8),
    )

    def _safe_corr(a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    corr_c_train_vs_caus = _safe_corr(c_k_train, sae_caus)
    corr_c_bal_vs_caus = _safe_corr(c_k_bal, sae_caus)

    pure_spurious_mask = sae_metrics["pure_spurious"]
    if pure_spurious_mask.any():
        median_c_train = float(np.median(c_k_train))
        c_train_misranks_background = float((c_k_train[pure_spurious_mask] > median_c_train).mean())
    else:
        c_train_misranks_background = None
        unknown_fields.append({"field": "diagnostic_validity.c_train_misranks_background", "reason": "n_pure_spurious==0이라 정의되지 않는다"})

    t0 = _stage_start()
    print("8/8 Broden IoU (top-%d 후보 + full alive latent 부산물)..." % args.top_n_broden)
    interpretability_candidates = []
    full_latent_iou_payload = {"status": "skipped", "reason": None}
    if args.skip_broden:
        unknown_fields.append({"field": "interpretability", "reason": "--skip-broden 지정으로 Broden IoU를 건너뛰었다"})
        full_latent_iou_payload = {"status": "skipped", "reason": "--skip-broden"}
    else:
        n_each = max(1, args.top_n_broden // 2)
        sel_full = sae_metrics["sel"]
        alive_pos = np.arange(n_alive)
        spurious_pos = alive_pos[sae_metrics["pure_spurious"]]
        causal_pos = alive_pos[sae_metrics["pure_causal"]]
        top_spurious_pos = spurious_pos[np.argsort(-sel_full[spurious_pos])][:n_each]
        top_causal_pos = causal_pos[np.argsort(sel_full[causal_pos])][:n_each]
        top_positions = np.concatenate([top_spurious_pos, top_causal_pos])
        top_latent_ids = alive_ids[torch.from_numpy(top_positions)].tolist()

        if top_latent_ids:
            _iou_rows, best = run_broden_iou(frozen, token_stats, model, top_latent_ids, args, target_block, token_scope)
            best_by_latent = {int(row["SAE latent"]): row for row in best}
            for pos in top_positions:
                latent = int(alive_ids[pos].item())
                row = best_by_latent.get(latent, {"best Broden concept": "", "category": "", "IoU": "0.000"})
                interpretability_candidates.append(
                    {
                        "latent": latent,
                        "spur": float(sae_spur[pos]),
                        "caus": float(sae_caus[pos]),
                        "sel": float(sel_full[pos]),
                        "s_k": float(s_k[pos]),
                        "c_k_train": float(c_k_train[pos]),
                        "c_k_bal": float(c_k_bal[pos]),
                        "top_concept": row["best Broden concept"],
                        "category": row["category"],
                        "iou": float(row["IoU"]),
                    }
                )
        else:
            unknown_fields.append({"field": "interpretability.candidates", "reason": "순수 후보(pure_spurious/pure_causal)가 0개다"})

        print("   full_latent_iou.json (전 alive latent, T3.1 부산물)...")
        args_full = SimpleNamespace(**vars(args))
        args_full.broden_subset = args.broden_subset_full
        _iou_rows_full, best_full = run_broden_iou(frozen, token_stats, model, alive_ids_list, args_full, target_block, token_scope)
        full_latent_iou_payload = {
            "status": "completed",
            "n_latents": len(alive_ids_list),
            "broden_subset_per_category": args.broden_subset_full,
            "latents": best_full,
        }
    _stage_done("Broden IoU (top-N 후보 + full alive latent)", t0)

    category_dist = {"spurious": {}, "causal": {}}
    for bucket in ("spurious", "causal"):
        cats = [
            row["category"]
            for row in interpretability_candidates
            if (row["sel"] >= args.sel_threshold if bucket == "spurious" else row["sel"] <= -args.sel_threshold)
        ]
        total = len(cats)
        if total:
            counts = defaultdict(int)
            for c in cats:
                counts[c] += 1
            category_dist[bucket] = {k: v / total for k, v in counts.items()}

    # =====================================================
    # 판정
    # =====================================================
    sae_purity = separation["sae"]["purity_rate"] or 0.0
    pca_purity = separation["pca"]["purity_rate"] or 0.0
    random_purity = separation["random"]["purity_rate"] or 0.0

    a_pass = (
        sae_purity > pca_purity
        and sae_purity > random_purity
        and p_sae_vs_pca is not None
        and p_sae_vs_random is not None
        and p_sae_vs_pca < 0.05
        and p_sae_vs_random < 0.05
    )
    b_pass = ap_p < 0.05 and ap_observed > random_baseline_ap

    if not a_pass:
        verdict = "fail"
        verdict_reason = (
            f"(A) 분리 실패 — sae purity_rate={sae_purity:.4f} vs pca={pca_purity:.4f}(p={p_sae_vs_pca}) "
            f"vs random={random_purity:.4f}(p={p_sae_vs_random}). monosemantic basis의 우위가 없으면 "
            "PCA/PLPD로 충분하다는 반론에 진다 — thesis 재검토가 필요하다."
        )
    elif not b_pass:
        verdict = "fail"
        verdict_reason = (
            f"(A) 분리는 통과(sae={sae_purity:.4f} > pca={pca_purity:.4f}, random={random_purity:.4f}, "
            f"p<0.05 양쪽)했으나 (B) 진단 타당성 실패 — s_k_ap={ap_observed:.4f}(p={ap_p:.4f}) vs "
            f"random_baseline={random_baseline_ap:.4f}. 분리는 존재하지만 라벨 없이는 못 집는다는 뜻 — "
            "thesis는 생존하나 c_k anchor 설계·'test time에 spurious를 겨냥한다'는 서술을 전면 수정해야 한다."
        )
    else:
        verdict = "pass"
        verdict_reason = (
            f"(A) sae purity_rate={sae_purity:.4f}가 pca={pca_purity:.4f}·random={random_purity:.4f} "
            f"양쪽보다 유의하게(p<0.05) 높고, (B) s_k_ap={ap_observed:.4f}(p={ap_p:.4f})가 "
            f"random_baseline={random_baseline_ap:.4f}을 유의하게 상회한다."
        )

    finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    result_payload = {
        "probe": probe_eval,
        "separation": separation,
        "purity_significance": {
            "sae_vs_pca_p": p_sae_vs_pca,
            "sae_vs_random_p": p_sae_vs_random,
            "test": f"two-proportion permutation, P={args.n_perm}",
        },
        "pool_pooling": "max",
        "pool_pooling_robustness_mean": robustness_mean,
        "diagnostic_validity": {
            "s_k_ap_for_pure_spurious": float(ap_observed),
            "s_k_ap_p_value": float(ap_p),
            "random_baseline_ap": float(random_baseline_ap),
            "s_times_1_minus_c_ap": float(ap_combined),
            "corr_c_train_vs_caus": corr_c_train_vs_caus,
            "corr_c_bal_vs_caus": corr_c_bal_vs_caus,
            "c_train_misranks_background": c_train_misranks_background,
            "c_k_train_n_images": c_k_train_n,
            "c_k_train_resolution": 1.0 / c_k_train_n,
            "c_k_train_split": "train",
            "c_k_bal_n_images": c_k_bal_n,
            "c_k_bal_resolution": 1.0 / c_k_bal_n,
            "c_k_bal_min_group_size": min_group_size,
            "c_k_bal_split": "test",
            "c_k_bal_split_note": (
                "c_k^bal은 test split에서 뽑는다(train 최소 group=56 -> 균형 224장 -> 분해능 4.5e-3, "
                "실측 c_k(~1e-3)보다 조악해 전부 0으로 양자화됨). test 최소 group=642 -> 균형 2,568장 -> "
                "분해능 3.9e-4. c_k^train은 그대로 train(95% confound 편향 분포)에서 측정한다 — 그 편향 "
                "자체가 c_k^train arm의 목적이라 바꾸지 않는다. 주의: c_k^bal과 층 ①의 정답(spur_j/caus_j)이 "
                "이제 같은 split(test)에서 나온다 — label leakage는 아니다(c_k=accuracy 기여도 ablation, "
                "spur_j/caus_j=라벨 대비 AUC로 서로 다른 양을 잰다) 하지만 더는 독립 표본이 아니다."
            ),
        },
        "interpretability": {
            "candidates": interpretability_candidates,
            "category_dist": category_dist,
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
            "model_name": args.model_name,
            "target_block": target_block,
            "token_scope": token_scope,
            "n_perm": args.n_perm,
            "sel_threshold": args.sel_threshold,
            "n_imagenet_source": args.n_imagenet_source,
            "top_n_broden": args.top_n_broden,
            "note_seed": (
                "이 실험은 seed 1회로 실행됐다(README.md 3-seed 기본과의 편차) — permutation 검정(P="
                f"{args.n_perm})이 판정에 필요한 통계적 유의성을 이미 제공하고, 확률적 요소(probe "
                "학습·PCA/random 축 fit·전체 파이프라인)를 3회 반복하는 비용이 GPU 공유 환경에서 "
                "과도해 단일 seed로 실행했다. 재현성은 --seed로 보장된다."
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

    (out_dir / "result.md").write_text(render_result_md(result), encoding="utf-8")
    with (out_dir / "full_latent_iou.json").open("w", encoding="utf-8") as f:
        json.dump(full_latent_iou_payload, f, indent=2, ensure_ascii=False)

    figures_dir = out_dir / "figures"
    plot_sel_hist(axis_sel_for_plot, figures_dir / "sel_hist.png")
    plot_spur_vs_caus(axis_points_for_plot, figures_dir / "spur_vs_caus.png")

    print(f"\nverdict={verdict} — {verdict_reason}")
    print(f"결과 저장: {out_dir}/result.json, {out_dir}/result.md, {out_dir}/full_latent_iou.json, {figures_dir}/*.png")


if __name__ == "__main__":
    main()
