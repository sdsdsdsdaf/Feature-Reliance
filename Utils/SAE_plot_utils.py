import gc
import math
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from Utils.SAE_utils import (
    collect_tokens_with_hook,
    latent_frequency,
    normalize_tokens,
    normalize_tokens_inplace,
    plot_sae_training_history,
)
from Utils.early_stopping import unwrap_compiled_model


IMAGENETTE_TO_IMAGENET_IDX = torch.tensor([0, 217, 482, 491, 497, 566, 569, 571, 574, 701], dtype=torch.long)
PERTURBATION_KINDS = ("grayscale", "blur", "patch_shuffle")
PERTURBATION_TOP_K = 20
PERTURBATION_OVERLAY_TOP_K = 12
PERTURBATION_BATCH_SIZE = 1
PERTURBATION_ENCODE_CHUNK_SIZE = 256
PERTURBATION_BLUR_KERNEL = 7
PERTURBATION_SHUFFLE_SEED = 0
INTERVENTION_TOP_K = 12
INTERVENTION_RANDOM_TRIALS = 3
INTERVENTION_RANDOM_SEED = 0
INTERVENTION_ALPHA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
INTERVENTION_BATCH_SIZE = 1
INTERVENTION_SAE_CHUNK_SIZE = 128
INTERVENTION_MAX_BATCHES = None
INTERVENTION_TOKEN_SCOPE = "all"
PERTURBATION_FAMILY_NAMES = {
    "grayscale": "color_latents",
    "blur": "texture_latents",
    "patch_shuffle": "shape_latents",
}


def save_plot(fig, output_dir, name, dpi=200):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _base_sae(sae):
    return unwrap_compiled_model(sae)


def _hidden_dim(sae):
    return _base_sae(sae).hidden_dim


def _sae_device(sae):
    return next(_base_sae(sae).parameters()).device


def _dataset_labels(dataset):
    base_ds = dataset.ds if hasattr(dataset, "ds") else dataset
    if hasattr(base_ds, "_labels"):
        return list(base_ds._labels)
    if hasattr(base_ds, "targets"):
        return list(base_ds.targets)
    return [base_ds[i][1] for i in range(len(base_ds))]


def _tokens_per_image_for_scope(token_scope, grid=14):
    token_scope = str(token_scope).lower()
    if token_scope == "cls":
        return 1
    if token_scope == "patch":
        return grid * grid
    return grid * grid + 1


def make_balanced_label_entropy_loader(dataset, max_tokens, token_scope, batch_size=64, seed=0):
    labels = _dataset_labels(dataset)
    classes = sorted(set(labels))
    tokens_per_image = _tokens_per_image_for_scope(token_scope)
    if max_tokens is None:
        per_class = max(Counter(labels).values())
    else:
        per_class = max(1, math.ceil(int(max_tokens) / max(1, len(classes) * tokens_per_image)))
    rng = np.random.default_rng(seed)
    by_class = {c: [] for c in classes}
    for idx, label in enumerate(labels):
        by_class[int(label)].append(int(idx))
    for c in classes:
        rng.shuffle(by_class[c])
        by_class[c] = by_class[c][:per_class]
    indices = []
    for offset in range(per_class):
        for c in classes:
            if offset < len(by_class[c]):
                indices.append(by_class[c][offset])
    subset = torch.utils.data.Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False), indices


@torch.no_grad()
def summarize_sae_latents(sae, tokens, labels=None, batch_size=4096, thresholds=(0.0, 1e-3, 1e-2, 0.1, 0.2), eps=1e-12):
    sae.eval()
    base_sae = _base_sae(sae)
    device = _sae_device(sae)
    labels = labels.detach().cpu().long() if labels is not None else None
    classes = sorted(labels.unique().tolist()) if labels is not None else []
    active_counts = {thr: torch.zeros(base_sae.hidden_dim) for thr in thresholds}
    activation_sum = torch.zeros(base_sae.hidden_dim)
    activation_sq_sum = torch.zeros(base_sae.hidden_dim)
    label_active_counts = torch.zeros(len(classes), base_sae.hidden_dim) if labels is not None else None
    total = 0

    for start in range(0, tokens.shape[0], batch_size):
        xb = tokens[start:start + batch_size].float().to(device)
        z = base_sae.encode(xb).detach().cpu()
        activation_sum += z.sum(dim=0)
        activation_sq_sum += z.pow(2).sum(dim=0)
        for thr in thresholds:
            active = z > float(thr)
            active_counts[thr] += active.float().sum(dim=0)
        if labels is not None:
            yb = labels[start:start + z.shape[0]]
            active_for_entropy = z > float(thresholds[-1])
            for i, cls in enumerate(classes):
                mask = yb == int(cls)
                if mask.any():
                    label_active_counts[i] += active_for_entropy[mask].float().sum(dim=0)
        total += z.shape[0]

    stats = {
        "mean_activation": activation_sum / max(1, total),
        "std_activation": (activation_sq_sum / max(1, total) - (activation_sum / max(1, total)).pow(2)).clamp_min(0).sqrt(),
    }
    for thr in thresholds:
        stats[f"freq_gt_{thr:g}"] = active_counts[thr] / max(1, total)
    if label_active_counts is not None:
        probs = label_active_counts / label_active_counts.sum(dim=0, keepdim=True).clamp_min(eps)
        entropy = -(probs * (probs + eps).log()).sum(dim=0)
        if len(classes) > 1:
            entropy = entropy / math.log(len(classes))
        stats["label_entropy"] = entropy
        stats["classes"] = classes
    return stats


def _format_prob_tick(value):
    if value <= 0:
        return "0"
    if value < 1e-3 or value >= 10:
        return f"{value:.1e}"
    return f"{value:.4f}"


def plot_sae_latent_stats(stats, active_threshold=0.2, title=None, log_scale=True, eps=1e-12):
    sparsity = stats.get(f"freq_gt_{active_threshold:g}")
    if sparsity is None:
        sparsity = stats["freq_gt_0.1"] if "freq_gt_0.1" in stats else stats["mean_activation"].gt(0).float()
    mean_activation = stats["mean_activation"]
    if "label_entropy" not in stats:
        print("label_entropy is missing; colors fall back to 0. Recompute stats with labels.")
    label_entropy = stats.get("label_entropy", torch.zeros_like(mean_activation))
    mask = torch.isfinite(sparsity) & torch.isfinite(mean_activation) & (sparsity > 0) & (mean_activation > 0)
    x = sparsity[mask].numpy()
    y = mean_activation[mask].numpy()
    c = label_entropy[mask].clamp(0, 1).numpy()
    if log_scale:
        x_plot = np.log10(np.clip(x, eps, None))
        y_plot = np.log10(np.clip(y, eps, None))
    else:
        x_plot = x
        y_plot = y
    fig = plt.figure(figsize=(6.2, 4.5))
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 0.9), height_ratios=(1, 4), hspace=0.05, wspace=0.05)
    ax_histx = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0], sharex=ax_histx)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    cax = fig.add_subplot(gs[0, 1])
    sc = ax.scatter(x_plot, y_plot, c=c, s=5, alpha=0.75, cmap="coolwarm", vmin=0.0, vmax=1.0, linewidths=0)
    ax_histx.hist(x_plot, bins=40, color="royalblue", alpha=0.65)
    ax_histy.hist(y_plot, bins=40, orientation="horizontal", color="royalblue", alpha=0.65)
    fig.colorbar(sc, cax=cax, label="Label entropy")
    ax.set_xlabel(f"Sparsity (P(z > {active_threshold:g}))")
    ax.set_ylabel("Mean activation value")
    ax.grid(True, alpha=0.3)
    ax_histx.set_ylabel("count", fontsize=8)
    ax_histy.set_xlabel("count", fontsize=8)
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    if log_scale and x_plot.size and y_plot.size:
        xticks = np.linspace(x_plot.min(), x_plot.max(), num=3)
        yticks = np.linspace(y_plot.min(), y_plot.max(), num=3)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels([_format_prob_tick(10 ** t) for t in xticks])
        ax.set_yticklabels([_format_prob_tick(10 ** t) for t in yticks])
    if title:
        ax.set_title(title, y=-0.35, fontsize=9, fontweight="bold")
    fig.subplots_adjust(left=0.13, right=0.95, bottom=0.18, top=0.95, hspace=0.05, wspace=0.05)
    return fig


@torch.no_grad()
def evaluate_sae_reconstruction_full(sae, tokens, batch_size, threshold, eps=1e-8):
    sae.eval()
    base_sae = _base_sae(sae)
    device = _sae_device(sae)
    total_rows = total_elements = 0
    sse = norm_sse = cosine_sum = l0_sum = l1_sum = 0.0
    active_counts = torch.zeros(base_sae.hidden_dim)
    token_sum = torch.zeros(tokens.shape[1])
    token_sq_sum = torch.zeros(tokens.shape[1])
    for start in range(0, tokens.shape[0], batch_size):
        xb_cpu = tokens[start:start + batch_size].float()
        xb = xb_cpu.to(device)
        x_hat, z = sae(xb)
        x_hat_cpu = x_hat.detach().cpu().float()
        z_cpu = z.detach().cpu().float()
        err = xb_cpu - x_hat_cpu
        sse += float(err.pow(2).sum().item())
        norm_sse += float((err.pow(2).sum(dim=-1) / xb_cpu.pow(2).sum(dim=-1).clamp_min(eps)).sum().item())
        cosine_sum += float(F.cosine_similarity(xb_cpu, x_hat_cpu, dim=-1).sum().item())
        l0_sum += float((z_cpu > threshold).float().sum(dim=-1).sum().item())
        l1_sum += float(z_cpu.abs().sum(dim=-1).sum().item())
        active_counts += (z_cpu > threshold).float().sum(dim=0)
        token_sum += xb_cpu.sum(dim=0)
        token_sq_sum += xb_cpu.pow(2).sum(dim=0)
        total_rows += xb_cpu.shape[0]
        total_elements += xb_cpu.numel()
    mean = token_sum / max(1, total_rows)
    ss_tot = float((token_sq_sum - total_rows * mean.pow(2)).sum().item())
    mse = sse / max(1, total_elements)
    return {
        "mse": mse,
        "token_nmse": norm_sse / max(1, total_rows),
        "r2": 1.0 - sse / max(ss_tot, eps),
        "cosine": cosine_sum / max(1, total_rows),
        "mean_l0": l0_sum / max(1, total_rows),
        "mean_l1": l1_sum / max(1, total_rows),
        "active_ratio": (l0_sum / max(1, total_rows)) / base_sae.hidden_dim,
        "dead_latent_frac": float((active_counts == 0).float().mean().item()),
    }


@torch.no_grad()
def top_latent_tokens(sae, tokens, latent_id, top_k=8, batch_size=4096):
    sae.eval()
    base_sae = _base_sae(sae)
    device = _sae_device(sae)
    values = []
    indices = []
    for start in range(0, tokens.shape[0], batch_size):
        xb = tokens[start:start + batch_size].float().to(device)
        z_i = base_sae.encode(xb)[:, int(latent_id)].detach().cpu()
        k = min(top_k, z_i.numel())
        batch_values, batch_indices = torch.topk(z_i, k=k)
        values.append(batch_values)
        indices.append(batch_indices + start)
    values = torch.cat(values)
    indices = torch.cat(indices)
    k = min(top_k, values.numel())
    top_values, order = torch.topk(values, k=k)
    return top_values, indices[order]


def token_index_to_image_patch(token_idx, token_scope, grid=14):
    token_idx = int(token_idx)
    patch_count = grid * grid
    if token_scope == "patch":
        image_pos = token_idx // patch_count
        patch_id = token_idx % patch_count
        token_type = "patch"
    elif token_scope in {"all", "clspatch"}:
        tokens_per_image = patch_count + 1
        image_pos = token_idx // tokens_per_image
        local_idx = token_idx % tokens_per_image
        if local_idx == 0:
            return {"image_pos": image_pos, "token_type": "cls", "patch_id": None, "patch_y": None, "patch_x": None}
        patch_id = local_idx - 1
        token_type = "patch"
    elif token_scope == "cls":
        return {"image_pos": token_idx, "token_type": "cls", "patch_id": None, "patch_y": None, "patch_x": None}
    else:
        raise ValueError("TOKEN_SCOPE must be one of: 'cls', 'patch', 'all'.")
    return {"image_pos": image_pos, "token_type": token_type, "patch_id": patch_id, "patch_y": patch_id // grid, "patch_x": patch_id % grid}


def model_tensor_to_image(x, mean, std):
    mean = torch.tensor(mean, dtype=x.dtype).view(3, 1, 1)
    std = torch.tensor(std, dtype=x.dtype).view(3, 1, 1)
    return (x.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def show_top_latent_samples(sae, tokens, dataset, latent_id, mean, std, top_k=8, token_scope="patch", grid=14, batch_size=4096):
    values, token_indices = top_latent_tokens(sae, tokens, latent_id, top_k=top_k, batch_size=batch_size)
    rows = math.ceil(len(token_indices) / 4)
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows))
    axes = np.asarray(axes).reshape(-1)
    records = []
    for ax, value, token_idx in zip(axes, values, token_indices):
        meta = token_index_to_image_patch(int(token_idx), token_scope=token_scope, grid=grid)
        image_tensor, label = dataset[int(meta["image_pos"])]
        ax.imshow(model_tensor_to_image(image_tensor, mean, std))
        title = f"latent={int(latent_id)}\nz={float(value):.2f}"
        if meta["token_type"] == "patch":
            patch_h = image_tensor.shape[-2] / grid
            patch_w = image_tensor.shape[-1] / grid
            rect = Rectangle((meta["patch_x"] * patch_w, meta["patch_y"] * patch_h), patch_w, patch_h, fill=False, edgecolor="red", linewidth=2)
            ax.add_patch(rect)
            title += f"\npatch={meta['patch_id']} ({meta['patch_y']},{meta['patch_x']})"
        else:
            title += "\nCLS"
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        records.append({**meta, "latent_id": int(latent_id), "activation": float(value), "label": int(label)})
    for ax in axes[len(token_indices):]:
        ax.axis("off")
    fig.tight_layout()
    return fig, records


@torch.no_grad()
def collect_sae_image_tensors(image_idx, model, dataset, sae, token_stats, target_block, token_scope, device):
    image_tensor, label = dataset[int(image_idx)]
    captured = {}

    def hook(_module, _inputs, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        captured["block_output"] = output.detach().float().cpu()

    handle = model.blocks[target_block].register_forward_hook(hook)
    try:
        model.eval().to(device)
        _ = model(image_tensor.unsqueeze(0).to(device))
    finally:
        handle.remove()

    h = captured["block_output"]
    if token_scope == "patch":
        x_raw = h[:, 1:, :]
        patch_start = 0
    elif token_scope in {"all", "clspatch"}:
        x_raw = h
        patch_start = 1
    else:
        raise ValueError("CLS token only has no 14x14 patch map.")

    x = normalize_tokens(x_raw.reshape(-1, x_raw.shape[-1]), token_stats).reshape(x_raw.shape)
    flat_x = x.reshape(-1, x.shape[-1]).float().to(device)
    base_sae = _base_sae(sae)
    sae.eval().to(device)
    flat_z = base_sae.encode(flat_x)
    flat_x_hat = base_sae.decode(flat_z)
    z = flat_z.detach().cpu().reshape(x.shape[0], x.shape[1], -1)
    x_hat = flat_x_hat.detach().cpu().reshape(x.shape)
    return {"image": image_tensor, "label": label, "x": x[0, patch_start:, :], "z": z[0, patch_start:, :], "x_hat": x_hat[0, patch_start:, :]}


def _normalize_map(m, eps=1e-8):
    m = m.detach().float().cpu()
    m = m - m.min()
    return m / (m.max() + eps)


def _plot_overlay(ax, image_np, patch_map, grid=14, cmap="viridis", alpha=0.55):
    patch_map = _normalize_map(patch_map).reshape(grid, grid)
    ax.imshow(image_np)
    ax.imshow(patch_map, cmap=cmap, alpha=alpha, extent=(0, image_np.shape[1], image_np.shape[0], 0), interpolation="nearest")
    ax.axis("off")


def select_x_channels(x, n_channels=6, mode="variance"):
    if mode == "variance":
        score = x.var(dim=0)
    elif mode == "mean_abs":
        score = x.abs().mean(dim=0)
    else:
        raise ValueError(f"Unknown x channel selection mode: {mode}")
    return score.topk(n_channels).indices.tolist()


def select_z_latents(z, n_latents=6, mode="energy", active_threshold=0.2):
    if mode == "peak_sparse":
        peak = z.max(dim=0).values
        active_count = (z > active_threshold).sum(dim=0)
        score = peak / active_count.clamp_min(1)
    elif mode == "energy":
        score = z.pow(2).mean(dim=0)
    elif mode == "mean":
        score = z.mean(dim=0)
    else:
        raise ValueError(f"Unknown z latent selection mode: {mode}")
    return score.topk(n_latents).indices.tolist()


def _symmetric_vlim(*maps, eps=1e-8):
    vmax = max(float(m.detach().float().abs().max().item()) for m in maps)
    vmax = max(vmax, eps)
    return -vmax, vmax


def plot_sae_channel_maps(
    image_idx,
    model,
    dataset,
    sae,
    token_stats,
    target_block,
    token_scope,
    device,
    mean,
    std,
    n_channels=6,
    x_select_mode="variance",
    z_select_mode="energy",
    active_threshold=0.2,
    grid=14,
):
    sample = collect_sae_image_tensors(image_idx, model, dataset, sae, token_stats, target_block, token_scope, device)
    image_np = model_tensor_to_image(sample["image"], mean, std)
    x = sample["x"]
    z = sample["z"]
    x_hat = sample["x_hat"]
    err = (x - x_hat).abs()
    x_channels = select_x_channels(x, n_channels=n_channels, mode=x_select_mode)
    z_latents = select_z_latents(z, n_latents=n_channels, mode=z_select_mode, active_threshold=active_threshold)

    fig, axes = plt.subplots(4, n_channels + 1, figsize=(2.5 * (n_channels + 1), 10))
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_ylabel("ViT-B x", fontsize=10)
    axes[1, 0].imshow(image_np)
    axes[1, 0].set_ylabel("SAE z", fontsize=10)
    axes[2, 0].imshow(image_np)
    axes[2, 0].set_ylabel("SAE x_hat", fontsize=10)
    axes[3, 0].imshow(image_np)
    axes[3, 0].set_ylabel("|x - x_hat|", fontsize=10)
    for ax in axes[:, 0]:
        ax.axis("off")

    for j, ch in enumerate(x_channels):
        x_map = x[:, ch].reshape(grid, grid)
        x_hat_map = x_hat[:, ch].reshape(grid, grid)
        vmin, vmax = _symmetric_vlim(x_map, x_hat_map)
        axes[0, j + 1].imshow(x_map, cmap="coolwarm", vmin=vmin, vmax=vmax)
        axes[0, j + 1].set_title(f"x dim={ch}", fontsize=8)
        axes[0, j + 1].axis("off")
        axes[2, j + 1].imshow(x_hat_map, cmap="coolwarm", vmin=vmin, vmax=vmax)
        axes[2, j + 1].set_title(f"x_hat dim={ch}", fontsize=8)
        axes[2, j + 1].axis("off")
        axes[3, j + 1].imshow(err[:, ch].reshape(grid, grid), cmap="magma")
        axes[3, j + 1].set_title(f"|x - x_hat| dim={ch}", fontsize=8)
        axes[3, j + 1].axis("off")

    for j, latent_id in enumerate(z_latents):
        z_map = z[:, latent_id].reshape(grid, grid)
        _plot_overlay(axes[1, j + 1], image_np, z_map, grid=grid)
        active_count = int((z[:, latent_id] > active_threshold).sum().item())
        peak = z[:, latent_id].max().item()
        axes[1, j + 1].set_title(f"SAE latent={latent_id}\nactive patches={active_count}, z_max={peak:.2f}", fontsize=8)

    fig.suptitle(
        f"ViT-B/16 block-{target_block} patch-token SAE | x_dim_select={x_select_mode}, "
        f"z_latent_select={z_select_mode}, z_thr={active_threshold}",
        fontsize=11,
    )
    fig.tight_layout()
    return fig, {"image_idx": image_idx, "x_channels": x_channels, "z_latents": z_latents}


def model_tensor_to_unit_tensor(x, mean, std):
    mean = torch.tensor(mean, dtype=x.dtype, device=x.device).view(3, 1, 1)
    std = torch.tensor(std, dtype=x.dtype, device=x.device).view(3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def unit_tensor_to_model_tensor(x, mean, std):
    mean = torch.tensor(mean, dtype=x.dtype, device=x.device).view(3, 1, 1)
    std = torch.tensor(std, dtype=x.dtype, device=x.device).view(3, 1, 1)
    return (x.clamp(0, 1) - mean) / std


def apply_image_perturbation(image_tensor, kind, mean, std, image_idx=0, grid=14, blur_kernel=PERTURBATION_BLUR_KERNEL, seed=PERTURBATION_SHUFFLE_SEED):
    unit = model_tensor_to_unit_tensor(image_tensor, mean, std)
    if kind == "identity":
        perturbed = unit
    elif kind == "grayscale":
        gray = unit.mean(dim=0, keepdim=True)
        perturbed = gray.repeat(3, 1, 1)
    elif kind == "blur":
        k = int(blur_kernel)
        if k % 2 == 0:
            k += 1
        perturbed = F.avg_pool2d(unit.unsqueeze(0), kernel_size=k, stride=1, padding=k // 2).squeeze(0)
    elif kind == "patch_shuffle":
        c, h, w = unit.shape
        patch_h = h // grid
        patch_w = w // grid
        patches = unit[:, :patch_h * grid, :patch_w * grid].reshape(c, grid, patch_h, grid, patch_w).permute(1, 3, 0, 2, 4).reshape(grid * grid, c, patch_h, patch_w)
        g = torch.Generator().manual_seed(int(seed) + int(image_idx))
        patches = patches[torch.randperm(patches.shape[0], generator=g)]
        perturbed = patches.reshape(grid, grid, c, patch_h, patch_w).permute(2, 0, 3, 1, 4).reshape(c, patch_h * grid, patch_w * grid)
        if perturbed.shape[-2:] != unit.shape[-2:]:
            canvas = unit.clone()
            canvas[:, :perturbed.shape[-2], :perturbed.shape[-1]] = perturbed
            perturbed = canvas
    else:
        raise ValueError(f"Unknown perturbation kind: {kind}")
    return unit_tensor_to_model_tensor(perturbed, mean, std)


@torch.no_grad()
def collect_patch_latents_for_batch(images, model, sae, token_stats, target_block, token_scope, device, encode_chunk_size=PERTURBATION_ENCODE_CHUNK_SIZE):
    captured = {}

    def hook(_module, _inputs, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        captured["block_output"] = output.detach().float().cpu()

    handle = model.blocks[target_block].register_forward_hook(hook)
    try:
        model.eval().to(device)
        _ = model(images.to(device))
    finally:
        handle.remove()

    h = captured["block_output"]
    if token_scope in {"patch", "all", "clspatch"}:
        patch_tokens = h[:, 1:, :]
    else:
        raise ValueError("Patch-grid latent maps require TOKEN_SCOPE='patch' or 'all'.")

    flat_tokens = patch_tokens.reshape(-1, patch_tokens.shape[-1]).float()
    mean = token_stats["mean"].float()
    std = token_stats["std"].float()
    z_chunks = []
    base_sae = _base_sae(sae)
    sae.eval().to(device)
    for start in range(0, flat_tokens.shape[0], encode_chunk_size):
        chunk = flat_tokens[start:start + encode_chunk_size]
        chunk = normalize_tokens_inplace(chunk, {"mean": mean, "std": std}).to(device)
        z_chunks.append(base_sae.encode(chunk).detach().cpu())
    z = torch.cat(z_chunks, dim=0)
    return z.reshape(patch_tokens.shape[0], patch_tokens.shape[1], -1)


def _ranking_records(score, frequency, mean_activation, peak_delta, top_k=PERTURBATION_TOP_K):
    k = min(top_k, score.numel())
    top_scores, top_ids = torch.topk(score, k=k)
    records = []
    for rank, (latent_id, value) in enumerate(zip(top_ids.tolist(), top_scores.tolist()), start=1):
        records.append({
            "rank": rank,
            "latent_id": int(latent_id),
            "score_delta": float(value),
            "frequency": float(frequency[latent_id].item()),
            "mean_activation": float(mean_activation[latent_id].item()),
            "peak_abs_delta": float(peak_delta[latent_id].item()),
        })
    return {"latent_ids": top_ids.tolist(), "records": records}


@torch.no_grad()
def rank_perturbation_sensitive_latents(model, dataset, sae, token_stats, config, mean, std, image_indices, kinds=PERTURBATION_KINDS, batch_size=PERTURBATION_BATCH_SIZE, top_k=PERTURBATION_TOP_K):
    accum = {kind: None for kind in kinds}
    count = 0
    iterator = list(range(0, len(image_indices), batch_size))
    for start in tqdm(iterator, total=math.ceil(len(image_indices) / batch_size), desc="ranking perturbation-sensitive latents"):
        idxs = image_indices[start:start + batch_size]
        images = torch.stack([dataset[int(i)][0] for i in idxs], dim=0)
        z_orig = collect_patch_latents_for_batch(images, model, sae, token_stats, config.hook.target_block, config.hook.token_scope, config.extraction_config.device)
        for kind in kinds:
            perturbed = torch.stack([apply_image_perturbation(dataset[int(i)][0], kind, mean, std, image_idx=int(i)) for i in idxs], dim=0)
            z_perturbed = collect_patch_latents_for_batch(perturbed, model, sae, token_stats, config.hook.target_block, config.hook.token_scope, config.extraction_config.device)
            delta = (z_perturbed - z_orig).abs()
            delta_sum = delta.sum(dim=(0, 1))
            peak_delta = delta.amax(dim=(0, 1))
            active = (z_orig > config.sae.active_threshold).float()
            freq_sum = active.sum(dim=(0, 1))
            mean_sum = z_orig.sum(dim=(0, 1))
            if accum[kind] is None:
                accum[kind] = {"delta_sum": delta_sum, "peak_delta": peak_delta, "freq_sum": freq_sum, "mean_sum": mean_sum}
            else:
                accum[kind]["delta_sum"] += delta_sum
                accum[kind]["peak_delta"] = torch.maximum(accum[kind]["peak_delta"], peak_delta)
                accum[kind]["freq_sum"] += freq_sum
                accum[kind]["mean_sum"] += mean_sum
            del z_perturbed, delta, perturbed
        count += z_orig.shape[0] * z_orig.shape[1]
        del z_orig, images
        gc.collect()

    results = {}
    for kind in kinds:
        item = accum[kind]
        score = item["delta_sum"] / max(1, count)
        frequency = item["freq_sum"] / max(1, count)
        mean_activation = item["mean_sum"] / max(1, count)
        results[kind] = _ranking_records(score, frequency, mean_activation, item["peak_delta"], top_k=top_k)
    return results


def plot_perturbation_topk_overlays(model, dataset, sae, token_stats, config, mean, std, image_idx=0, rankings=None, top_k=PERTURBATION_OVERLAY_TOP_K, map_mode="delta"):
    image = dataset[int(image_idx)][0]
    original = image.unsqueeze(0)
    z_orig = collect_patch_latents_for_batch(original, model, sae, token_stats, config.hook.target_block, config.hook.token_scope, config.extraction_config.device)
    image_np = model_tensor_to_image(image, mean, std)
    fig, axes = plt.subplots(1, len(PERTURBATION_KINDS) + 1, figsize=(4 * (len(PERTURBATION_KINDS) + 1), 4))
    axes[0].imshow(image_np)
    axes[0].set_title("original")
    axes[0].axis("off")
    latent_ids_by_kind = {}
    for ax, kind in zip(axes[1:], PERTURBATION_KINDS):
        ids = rankings[kind]["latent_ids"][:top_k]
        latent_ids_by_kind[kind] = [int(x) for x in ids]
        perturbed = apply_image_perturbation(image, kind, mean, std, image_idx=int(image_idx)).unsqueeze(0)
        z_perturbed = collect_patch_latents_for_batch(perturbed, model, sae, token_stats, config.hook.target_block, config.hook.token_scope, config.extraction_config.device)
        if map_mode == "delta":
            patch_map = (z_perturbed[:, :, ids] - z_orig[:, :, ids]).abs().mean(dim=-1)[0]
        else:
            patch_map = z_perturbed[:, :, ids].mean(dim=-1)[0]
        _plot_overlay(ax, image_np, patch_map)
        ax.set_title(f"{kind}\n{map_mode}, top-{len(ids)}", fontsize=9)
    fig.suptitle("Perturbation-sensitive top-k latent overlay", fontsize=11)
    fig.tight_layout()
    return fig, {"latent_ids": latent_ids_by_kind}


def make_intervention_eval_loader(dataset, indices, batch_size=INTERVENTION_BATCH_SIZE):
    subset = torch.utils.data.Subset(dataset, [int(idx) for idx in indices])
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())


def _as_long_tensor(ids):
    if torch.is_tensor(ids):
        return ids.detach().long().cpu()
    return torch.tensor([int(x) for x in ids], dtype=torch.long)


def match_latents_by_frequency_and_magnitude(target_ids, hidden_dim, sae_latent_frequency, sae_latent_stats=None, pool_ids=None, k=None):
    target_ids = [int(x) for x in target_ids]
    if k is None:
        k = len(target_ids)
    if pool_ids is None:
        pool_ids = list(range(hidden_dim))
    blocked = set(target_ids)
    pool_ids = [int(x) for x in pool_ids if int(x) not in blocked]
    if not pool_ids:
        return []
    freq = sae_latent_frequency.float()
    mean_act = sae_latent_stats["mean_activation"].float() if sae_latent_stats is not None else torch.zeros_like(freq)
    target_freq = freq[target_ids].mean() if target_ids else torch.tensor(0.0)
    target_mag = mean_act[target_ids].mean() if target_ids else torch.tensor(0.0)
    pool = torch.tensor(pool_ids, dtype=torch.long)
    score = (freq[pool] - target_freq).abs() + (mean_act[pool] - target_mag).abs()
    order = torch.argsort(score)[:k]
    return pool[order].tolist()


def random_latents(k, hidden_dim, exclude=(), seed=0):
    exclude = set(int(x) for x in exclude)
    candidates = [i for i in range(hidden_dim) if i not in exclude]
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(len(candidates), generator=g)[:k]
    return [candidates[int(i)] for i in perm]


def build_intervention_latent_specs(rankings, hidden_dim, sae_latent_frequency, sae_latent_stats=None, top_k=INTERVENTION_TOP_K, random_trials=INTERVENTION_RANDOM_TRIALS):
    specs = {}
    for kind, result in rankings.items():
        family = PERTURBATION_FAMILY_NAMES.get(kind, f"{kind}_latents")
        cue = [int(x) for x in result["latent_ids"][:top_k]]
        specs[(family, "cue_latents")] = cue
        specs[(family, "matched_frequency_magnitude")] = match_latents_by_frequency_and_magnitude(cue, hidden_dim, sae_latent_frequency, sae_latent_stats, k=len(cue))
        for trial in range(random_trials):
            specs[(family, f"random_{trial}")] = random_latents(len(cue), hidden_dim, exclude=cue, seed=INTERVENTION_RANDOM_SEED + trial + 1000 * len(specs))
    print("[Intervention latent specs]")
    for (family, baseline), ids in specs.items():
        print(f"{family}/{baseline}", ids)
    return specs


@torch.no_grad()
def reconstruct_tokens_with_latent_scaling(tokens, sae, token_stats, latent_ids, alpha, active_threshold=0.2, return_debug=False):
    output_device = tokens.device
    output_dtype = tokens.dtype
    base_sae = _base_sae(sae)
    sae_device = _sae_device(sae)
    flat_tokens = tokens.reshape(-1, tokens.shape[-1]).to(device=sae_device, dtype=torch.float32)
    out_chunks = []
    debug_chunks = []
    ids = _as_long_tensor(latent_ids).to(sae_device) if latent_ids else None
    for start in range(0, flat_tokens.shape[0], INTERVENTION_SAE_CHUNK_SIZE):
        chunk = flat_tokens[start:start + INTERVENTION_SAE_CHUNK_SIZE]
        mean = token_stats["mean"].to(device=sae_device, dtype=chunk.dtype)
        std = token_stats["std"].to(device=sae_device, dtype=chunk.dtype)
        chunk_norm = (chunk - mean) / std
        z = base_sae.encode(chunk_norm)
        if ids is not None and ids.numel() > 0:
            z[:, ids] = z[:, ids] * float(alpha)
        recon_norm = base_sae.decode(z)
        recon = recon_norm * std + mean
        out_chunks.append(recon.detach())
        if return_debug:
            debug_chunks.append(z.detach().cpu())
    recon_tokens = torch.cat(out_chunks, dim=0).reshape_as(tokens).to(device=output_device, dtype=output_dtype)
    if not return_debug:
        return recon_tokens
    z_all = torch.cat(debug_chunks, dim=0)
    return recon_tokens, {"z_mean_abs": z_all.abs().mean().item(), "z_l0": (z_all > active_threshold).float().sum(dim=1).mean().item()}


@torch.no_grad()
def run_model_with_latent_intervention(images, latent_ids, alpha, model, sae, token_stats, target_block, intervention_token_scope, device):
    device = torch.device(device)

    def hook(_module, _inputs, output):
        if isinstance(output, (tuple, list)):
            block_output = output[0]
        else:
            block_output = output
        edited = block_output.detach().clone()
        if intervention_token_scope == "cls":
            edited[:, :1, :] = reconstruct_tokens_with_latent_scaling(block_output[:, :1, :], sae, token_stats, latent_ids, alpha)
        elif intervention_token_scope == "patch":
            edited[:, 1:, :] = reconstruct_tokens_with_latent_scaling(block_output[:, 1:, :], sae, token_stats, latent_ids, alpha)
        elif intervention_token_scope in {"all", "clspatch"}:
            edited = reconstruct_tokens_with_latent_scaling(block_output, sae, token_stats, latent_ids, alpha)
        else:
            raise ValueError(f"Unknown intervention_token_scope: {intervention_token_scope}")
        return edited

    model.eval().to(device)
    sae.eval().to(device)
    handle = model.blocks[target_block].register_forward_hook(hook)
    try:
        logits = model(images.to(device)).detach().cpu()
    finally:
        handle.remove()
    return logits


@torch.no_grad()
def evaluate_latent_intervention_curve(model, sae, token_stats, specs, loader, target_block, device, alphas=INTERVENTION_ALPHA_VALUES, max_batches=INTERVENTION_MAX_BATCHES):
    device = torch.device(device)
    records = []
    model.eval().to(device)
    for (family, baseline), latent_ids in specs.items():
        for alpha in alphas:
            js_sum = ce_sum = logit_l1_sum = acc_sum = clean_acc_sum = 0.0
            rows = 0
            for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"latent intervention curve [{INTERVENTION_TOKEN_SCOPE}]", leave=False)):
                if max_batches is not None and batch_idx >= int(max_batches):
                    break
                images = images.to(device)
                labels_cpu = labels.detach().cpu().long()
                clean_logits = model(images).detach().cpu()
                int_logits = run_model_with_latent_intervention(images, latent_ids, alpha, model, sae, token_stats, target_block, INTERVENTION_TOKEN_SCOPE, device)
                labels_imagenet = IMAGENETTE_TO_IMAGENET_IDX[labels_cpu]
                clean_pred = clean_logits.argmax(dim=1)
                int_pred = int_logits.argmax(dim=1)
                clean_acc = (clean_pred == labels_imagenet).float().mean().item()
                intervention_acc = (int_pred == labels_imagenet).float().mean().item()
                clean_logp = F.log_softmax(clean_logits, dim=1)
                int_logp = F.log_softmax(int_logits, dim=1)
                m = 0.5 * (clean_logp.exp() + int_logp.exp()).clamp_min(1e-12)
                js = 0.5 * (F.kl_div(clean_logp, m, reduction="batchmean") + F.kl_div(int_logp, m, reduction="batchmean"))
                ce = F.cross_entropy(int_logits, labels_imagenet)
                js_sum += float(js.item()) * images.shape[0]
                ce_sum += float(ce.item()) * images.shape[0]
                logit_l1_sum += float((int_logits - clean_logits).abs().mean(dim=1).sum().item())
                acc_sum += intervention_acc * images.shape[0]
                clean_acc_sum += clean_acc * images.shape[0]
                rows += images.shape[0]
            clean_acc = clean_acc_sum / max(1, rows)
            intervention_acc = acc_sum / max(1, rows)
            records.append({
                "family": family,
                "baseline": baseline,
                "name": f"{family}/{baseline}",
                "alpha": float(alpha),
                "js_divergence": js_sum / max(1, rows),
                "cross_entropy": ce_sum / max(1, rows),
                "logit_l1": logit_l1_sum / max(1, rows),
                "clean_acc": clean_acc,
                "intervention_acc": intervention_acc,
                "acc_delta": intervention_acc - clean_acc,
                "acc_drop": clean_acc - intervention_acc,
                "n": rows,
            })
    return records


def print_intervention_curve_summary(records, metric="js_divergence"):
    print("name alpha metric acc_drop ce_delta logit_l1")
    for row in records:
        print(f"{row['name']} {row['alpha']:.2f} {row[metric]:.10e} {row['acc_drop']:.4f} {row['cross_entropy']:.6f} {row['logit_l1']:.10e}")


def plot_intervention_curve(records, metric="js_divergence", figsize=(13, 7), logy=False):
    family_colors = {"color_latents": "tab:red", "texture_latents": "tab:green", "shape_latents": "tab:blue"}
    baseline_styles = {"cue_latents": ("-", "o", 2.6, 1.0), "matched_frequency_magnitude": ("--", "s", 2.2, 0.9), "random_0": (":", "^", 1.6, 0.55), "random_1": (":", "v", 1.6, 0.55), "random_2": (":", "D", 1.6, 0.55)}
    fig, ax = plt.subplots(figsize=figsize)
    names = sorted(set(row["name"] for row in records))
    for name in names:
        family, baseline = name.split("/", 1)
        subset = sorted([row for row in records if row["name"] == name], key=lambda r: r["alpha"])
        xs = [row["alpha"] for row in subset]
        ys = [row[metric] for row in subset]
        linestyle, marker, linewidth, alpha = baseline_styles.get(baseline, ("-", "o", 1.5, 0.8))
        ax.plot(xs, ys, linestyle=linestyle, marker=marker, linewidth=linewidth, alpha=alpha, color=family_colors.get(family), label=name)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.set_xlabel("alpha")
    ax.set_ylabel(metric)
    if logy:
        ax.set_yscale("log")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def save_trial_plots(trial_dir, model, train_history, sae, token_stats, val_tokens, val_labels, train_dataset, val_dataset, config, mean, std):
    plots_dir = Path(trial_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = {}

    fig = plot_sae_training_history(train_history, hidden_dim=_hidden_dim(sae), active_threshold=config.sae.active_threshold)
    save_plot(fig, plots_dir, "sae_train_log", dpi=config.output.plot_dpi)

    label_entropy_loader, _ = make_balanced_label_entropy_loader(val_dataset, max_tokens=config.token.max_val_tokens, token_scope=config.hook.token_scope, batch_size=config.data_config.batch_size)
    entropy_tokens, entropy_labels = collect_tokens_with_hook(
        model,
        label_entropy_loader,
        config.token.max_val_tokens,
        config.hook.target_block,
        config.hook.token_scope,
        config.extraction_config.device,
        cache_dtype=None,
        return_labels=True,
    )
    entropy_tokens = normalize_tokens_inplace(entropy_tokens.float(), token_stats)
    sae_latent_stats = summarize_sae_latents(sae, entropy_tokens, labels=entropy_labels, batch_size=config.sae.batch_size, thresholds=(0.0, 1e-3, 1e-2, 0.1, config.sae.active_threshold))
    fig = plot_sae_latent_stats(sae_latent_stats, active_threshold=config.sae.active_threshold, title=f"SAE for {config.hook.token_scope} token only")
    save_plot(fig, plots_dir, "sae_latent_stats_log", dpi=config.output.plot_dpi)
    fig = plot_sae_latent_stats(sae_latent_stats, active_threshold=config.sae.active_threshold, title=f"SAE for {config.hook.token_scope} token only", log_scale=False)
    save_plot(fig, plots_dir, "sae_latent_stats_linear", dpi=config.output.plot_dpi)

    sae_latent_frequency = latent_frequency(sae, val_tokens, config.sae.batch_size, config.sae.active_threshold, config.extraction_config.device)
    top_freq_values, top_freq_latents = torch.topk(sae_latent_frequency, k=min(10, sae_latent_frequency.numel()))
    for latent_id in top_freq_latents[:3].tolist():
        fig, records = show_top_latent_samples(
            sae,
            val_tokens,
            val_dataset,
            latent_id=int(latent_id),
            mean=mean,
            std=std,
            top_k=8,
            token_scope=config.hook.token_scope,
            batch_size=config.sae.batch_size,
        )
        save_plot(fig, plots_dir, f"top_latent_samples_{int(latent_id):05d}", dpi=config.output.plot_dpi)

    fig, info = plot_sae_channel_maps(
        image_idx=3,
        model=model,
        dataset=val_dataset,
        sae=sae,
        token_stats=token_stats,
        target_block=config.hook.target_block,
        token_scope=config.hook.token_scope,
        device=config.extraction_config.device,
        mean=mean,
        std=std,
        n_channels=6,
        x_select_mode="variance",
        z_select_mode="energy",
        active_threshold=config.sae.active_threshold,
    )
    save_plot(fig, plots_dir, "patch_token_sae_channel_maps_image_3", dpi=config.output.plot_dpi)

    perturbation_indices = list(range(min(12, len(val_dataset))))
    perturbation_rankings = rank_perturbation_sensitive_latents(model, val_dataset, sae, token_stats, config, mean, std, image_indices=perturbation_indices)
    fig, perturbation_overlay_info = plot_perturbation_topk_overlays(model, val_dataset, sae, token_stats, config, mean, std, image_idx=0, rankings=perturbation_rankings)
    save_plot(fig, plots_dir, "perturbation_topk_overlays_image_0", dpi=config.output.plot_dpi)

    intervention_specs = build_intervention_latent_specs(perturbation_rankings, _hidden_dim(sae), sae_latent_frequency, sae_latent_stats)
    intervention_loader = make_intervention_eval_loader(val_dataset, perturbation_indices)
    intervention_records = evaluate_latent_intervention_curve(model, sae, token_stats, intervention_specs, intervention_loader, config.hook.target_block, config.extraction_config.device)
    print_intervention_curve_summary(intervention_records, metric="js_divergence")
    fig = plot_intervention_curve(intervention_records, metric="js_divergence", figsize=(13, 7), logy=False)
    save_plot(fig, plots_dir, "intervention_curve_js_divergence", dpi=config.output.plot_dpi)
    fig = plot_intervention_curve(intervention_records, metric="acc_drop", figsize=(13, 7), logy=False)
    save_plot(fig, plots_dir, "intervention_curve_acc_drop", dpi=config.output.plot_dpi)
    fig = plot_intervention_curve(intervention_records, metric="logit_l1", figsize=(13, 7), logy=False)
    save_plot(fig, plots_dir, "intervention_curve_logit_l1", dpi=config.output.plot_dpi)

    diagnostics["top_active_latents"] = list(zip(top_freq_latents.tolist(), top_freq_values.tolist()))
    diagnostics["channel_map_info"] = info
    diagnostics["perturbation_overlay_info"] = perturbation_overlay_info
    diagnostics["intervention_records"] = intervention_records
    diagnostics["reconstruction_full_metrics"] = evaluate_sae_reconstruction_full(sae, val_tokens, config.sae.batch_size, config.sae.active_threshold)
    return diagnostics
