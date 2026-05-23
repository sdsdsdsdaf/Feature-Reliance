from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import math

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from timm.data import resolve_model_data_config
from tqdm.auto import tqdm

from Utils.Config import DataConfig, DatasetSpec, ModelSpec, TransformHyperParams
from Utils.metric import js_divergence
from Utils.utils import build_dataset, build_transform, cal_accuracy, ensure_dir


def build_loader(
    dataset_spec: DatasetSpec,
    perturbation: str,
    model_spec: ModelSpec,
    transform_hparams: TransformHyperParams,
    data_config: DataConfig,
    normalize: bool = True,
    batch_size: int | None = None,
    shuffle: bool = False,
) -> DataLoader:
    transform = build_transform(
        perturbation=perturbation,
        mean=model_spec.mean,
        std=model_spec.std,
        resize_size=model_spec.resize_size,
        hparams=transform_hparams,
        normalize=normalize,
    )
    dataset = build_dataset(dataset_spec, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size or data_config.batch_size,
        shuffle=shuffle,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
    )


def resize_size_from_timm_config(data_cfg: dict) -> int:
    input_size = data_cfg.get("input_size", (3, 224, 224))
    image_size = int(input_size[-1])
    crop_pct = float(data_cfg.get("crop_pct") or 1.0)
    return int(math.floor(image_size / crop_pct)) if crop_pct > 0 else image_size


def make_timm_model_spec(
    timm,
    timm_model_name: str,
    notebook_model_name: str,
    pretrained_weight_label: str,
    pretrained: bool = True,
) -> ModelSpec:
    model = timm.create_model(timm_model_name, pretrained=pretrained)
    data_cfg = resolve_model_data_config(model)
    input_size = data_cfg.get("input_size", (3, 224, 224))
    if int(input_size[-1]) != 224:
        raise ValueError(
            f"The current perturbation transform center-crops to 224, but {timm_model_name} "
            f"expects input_size={input_size}. Update Utils.transfrom.get_transform first."
        )
    return ModelSpec(
        model_name=notebook_model_name,
        pretrained_weight=pretrained_weight_label if pretrained else "random",
        model=model,
        mean=list(data_cfg["mean"]),
        std=list(data_cfg["std"]),
        resize_size=resize_size_from_timm_config(data_cfg),
    )


def safe_linear_cka(X, Y, eps: float = 1e-12, max_rows: int | None = None) -> float:
    X = X.detach().float().reshape(X.shape[0], -1).cpu()
    Y = Y.detach().float().reshape(Y.shape[0], -1).cpu()
    if max_rows is not None and X.shape[0] > max_rows:
        idx = torch.linspace(0, X.shape[0] - 1, steps=max_rows).long()
        X = X[idx]
        Y = Y[idx]
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    xy = X @ Y.T
    xx = X @ X.T
    yy = Y @ Y.T
    hsic_xy = xy.pow(2).sum()
    hsic_xx = xx.pow(2).sum()
    hsic_yy = yy.pow(2).sum()
    return float((hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + eps)).item())


def activation_to_metric_matrix(A, pool_hw: int = 14, max_images: int | None = 256):
    A = A.detach().float().cpu()
    if max_images is not None:
        A = A[:max_images]
    if A.ndim == 4 and min(A.shape[-2:]) > pool_hw:
        A = F.adaptive_avg_pool2d(A, output_size=(pool_hw, pool_hw))
    return A.flatten(1)


def cosine_distance_from_matrices(X, Y) -> float:
    return float((1.0 - F.cosine_similarity(X.float(), Y.float(), dim=1).mean()).item())


def l2_distance_from_matrices(X, Y) -> float:
    return float((X.float() - Y.float()).norm(dim=1).mean().item())


def specificity_columns(df: pd.DataFrame, prefix: str = "sensitivity"):
    value_cols = [c for c in df.columns if c.startswith(prefix + "_")]
    perturb_cols = {c.removeprefix(prefix + "_"): c for c in value_cols}
    return value_cols, perturb_cols


def to_display_image(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = x.permute(1, 2, 0).numpy()
        else:
            x = x.numpy()
    x = np.asarray(x)
    if x.dtype != np.uint8:
        if x.max() <= 1.5:
            x = np.clip(x * 255.0, 0, 255)
        else:
            x = np.clip(x, 0, 255)
        x = x.astype(np.uint8)
    return x


def build_visual_transforms(
    perturbation_names,
    transform_hparams: TransformHyperParams,
    resize_size: int = 256,
):
    return {
        p: build_transform(
            perturbation=p,
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            resize_size=resize_size,
            hparams=transform_hparams,
            normalize=False,
        )
        for p in perturbation_names
    }


def save_perturbation_preview(
    base_ds,
    sample_global_idx: int,
    perturbations,
    perturbation_to_cue: dict,
    visual_transforms: dict,
    save_dir,
):
    raw_image, label = base_ds[sample_global_idx]
    fig, axes = plt.subplots(1, len(perturbations), figsize=(3.0 * len(perturbations), 3.4))
    if len(perturbations) == 1:
        axes = [axes]

    for ax, perturbation in zip(axes, perturbations):
        transformed = visual_transforms[perturbation](raw_image.copy())
        ax.imshow(to_display_image(transformed))
        ax.set_title(f"{perturbation}\n{perturbation_to_cue[perturbation]}")
        ax.axis("off")

    fig.suptitle(f"global index={sample_global_idx}, label={label}", y=1.04)
    fig.tight_layout()
    save_path = Path(save_dir) / f"perturbation_preview_{sample_global_idx}.png"
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return save_path


@torch.no_grad()
def extract_logits_and_reps(model, loader, device, max_batches: int | None = None):
    model.eval().to(device)
    logits_list, reps_list, labels_list = [], [], []

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="extract logits/reps")):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)

        features = model.forward_features(images)
        try:
            reps = model.forward_head(features, pre_logits=True)
            logits = model.forward_head(features, pre_logits=False)
        except TypeError:
            logits = model(images)
            reps = features.flatten(1) if features.ndim > 2 else features

        logits_list.append(logits.detach().float().cpu())
        reps_list.append(reps.detach().float().cpu())
        labels_list.append(labels.detach().cpu())

    return {
        "logits": torch.cat(logits_list, dim=0),
        "representations": torch.cat(reps_list, dim=0),
        "labels": torch.cat(labels_list, dim=0),
    }


def run_model_level_calibration(
    model_specs,
    dataset_spec,
    perturbation_names,
    perturbation_to_cue: dict,
    build_loader_fn,
    output_dir,
    device,
    max_batches: int | None = None,
):
    rows = []
    for model_spec in model_specs:
        outputs = {}
        for perturbation in perturbation_names:
            loader = build_loader_fn(dataset_spec, perturbation, model_spec)
            outputs[perturbation] = extract_logits_and_reps(
                model_spec.model,
                loader,
                device=device,
                max_batches=max_batches,
            )

        clean = outputs["original"]
        clean_acc = cal_accuracy(clean["logits"], clean["labels"], class_map_name=dataset_spec.class_map_name)

        for perturbation in perturbation_names:
            current = outputs[perturbation]
            acc = cal_accuracy(current["logits"], current["labels"], class_map_name=dataset_spec.class_map_name)
            row = {
                "model": model_spec.model_name,
                "pretrained_weight": model_spec.pretrained_weight,
                "perturbation": perturbation,
                "cue": perturbation_to_cue.get(perturbation, "unknown"),
                "accuracy": acc,
                "accuracy_drop_vs_clean": clean_acc - acc,
            }
            if perturbation != "original":
                row["js_divergence"] = js_divergence(clean["logits"], current["logits"], return_float=True)
                row["cka"] = safe_linear_cka(clean["representations"], current["representations"])
            rows.append(row)

        del outputs
        model_spec.model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(Path(output_dir) / "perturbation_model_metrics.csv", index=False)
    return df


def get_resnet_layers_by_name(model, layer_names):
    all_layers = {
        "conv1": model.conv1,
        "layer1": model.layer1,
        "layer2": model.layer2,
        "layer3": model.layer3,
        "layer4": model.layer4,
    }
    return {name: all_layers[name] for name in layer_names}


def register_activation_hooks(target_layers, activation_cache):
    handles = []
    for layer_name, module in target_layers.items():
        def make_hook(name):
            def hook(_module, _inputs, output):
                activation_cache[name] = output.detach().float().cpu()
            return hook
        handles.append(module.register_forward_hook(make_hook(layer_name)))
    return handles


@torch.no_grad()
def forward_with_resnet_activations(model, images, target_layers, device):
    temp_activations = {}
    handles = register_activation_hooks(target_layers, temp_activations)
    try:
        logits = model(images.to(device, non_blocking=True))
        activations = {name: temp_activations[name] for name in target_layers}
    finally:
        for handle in handles:
            handle.remove()
    return logits.detach().float().cpu(), activations


def init_metric_accumulator():
    return {"n_batches": 0, "cka": 0.0, "cosine_distance": 0.0, "l2_distance": 0.0, "delta_mean": 0.0}


def update_metric_accumulator(acc, A_o, A_p):
    X = activation_to_metric_matrix(A_o, max_images=None)
    Y = activation_to_metric_matrix(A_p, max_images=None)
    acc["n_batches"] += 1
    acc["cka"] += safe_linear_cka(X, Y)
    acc["cosine_distance"] += cosine_distance_from_matrices(X, Y)
    acc["l2_distance"] += l2_distance_from_matrices(X, Y)
    acc["delta_mean"] += float((X - Y).abs().mean().item())


def finalize_metric_accumulator(acc):
    n = max(1, acc["n_batches"])
    return {k: (v / n if k != "n_batches" else v) for k, v in acc.items()}


def standardize_spatial_channels(A, eps: float = 1e-6):
    A = A.detach().float()
    mean = A.mean(dim=(2, 3), keepdim=True)
    std = A.std(dim=(2, 3), keepdim=True).clamp_min(eps)
    return (A - mean) / std


@torch.no_grad()
def run_resnet_streaming_phase1(
    model,
    model_spec,
    dataset_spec,
    perturbation_names,
    target_layer_names,
    perturbation_to_cue: dict,
    shape_perturbations,
    build_loader_fn,
    device,
    max_batches: int | None = None,
):
    model.eval().to(device)
    target_layers = get_resnet_layers_by_name(model, target_layer_names)
    layer_metrics = {
        (perturbation, layer_name): init_metric_accumulator()
        for perturbation in perturbation_names if perturbation != "original"
        for layer_name in target_layers
    }
    channel_sums = defaultdict(dict)
    channel_counts = defaultdict(dict)

    for perturbation in perturbation_names:
        if perturbation == "original":
            continue

        clean_loader = build_loader_fn(dataset_spec, "original", model_spec)
        pert_loader = build_loader_fn(dataset_spec, perturbation, model_spec)
        desc = f"ResNet streaming original vs {perturbation}"

        for batch_idx, ((clean_images, _), (pert_images, _)) in enumerate(
            tqdm(zip(clean_loader, pert_loader), total=len(clean_loader), desc=desc)
        ):
            if max_batches is not None and batch_idx >= max_batches:
                break

            _, A_clean = forward_with_resnet_activations(model, clean_images, target_layers, device=device)
            _, A_pert = forward_with_resnet_activations(model, pert_images, target_layers, device=device)

            for layer_name in target_layers:
                A_o = A_clean[layer_name]
                A_p = A_pert[layer_name]
                update_metric_accumulator(layer_metrics[(perturbation, layer_name)], A_o, A_p)

                sens = (
                    standardize_spatial_channels(A_o)
                    - standardize_spatial_channels(A_p)
                ).abs().mean(dim=(0, 2, 3)).cpu()
                if perturbation not in channel_sums[layer_name]:
                    channel_sums[layer_name][perturbation] = torch.zeros_like(sens)
                    channel_counts[layer_name][perturbation] = 0
                channel_sums[layer_name][perturbation] += sens * A_o.shape[0]
                channel_counts[layer_name][perturbation] += A_o.shape[0]

            del A_clean, A_pert, clean_images, pert_images
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    layer_rows = []
    for (perturbation, layer_name), acc in layer_metrics.items():
        values = finalize_metric_accumulator(acc)
        layer_rows.append({
            "model": "resnet50",
            "layer": layer_name,
            "perturbation": perturbation,
            "cue": perturbation_to_cue[perturbation],
            **values,
        })
    layer_df = pd.DataFrame(layer_rows).sort_values(["cue", "perturbation", "cka"])

    channel_rows = []
    for layer_name, by_perturbation in channel_sums.items():
        mean_sens = {
            p: channel_sums[layer_name][p] / max(1, channel_counts[layer_name][p])
            for p in by_perturbation
        }
        n_channels = next(iter(mean_sens.values())).numel()
        for channel in range(n_channels):
            row = {"model": "resnet50", "layer": layer_name, "channel": channel}
            for perturbation, values in mean_sens.items():
                row[f"sensitivity_{perturbation}"] = float(values[channel].item())

            non_gray = [p for p in mean_sens if p != "grayscale"]
            non_bilat = [p for p in mean_sens if p != "bilateral"]
            shape_ps = [p for p in shape_perturbations if p in mean_sens]
            color_texture_ps = [p for p in ["grayscale", "bilateral"] if p in mean_sens]
            row["color_specificity"] = row.get("sensitivity_grayscale", 0.0) - float(np.mean([row[f"sensitivity_{p}"] for p in non_gray]))
            row["texture_specificity"] = row.get("sensitivity_bilateral", 0.0) - float(np.mean([row[f"sensitivity_{p}"] for p in non_bilat]))
            row["shape_specificity"] = float(np.mean([row[f"sensitivity_{p}"] for p in shape_ps])) - float(np.mean([row[f"sensitivity_{p}"] for p in color_texture_ps]))
            channel_rows.append(row)

    channel_df = pd.DataFrame(channel_rows)
    return layer_df, channel_df


def select_top_resnet_channels(channel_df: pd.DataFrame, top_k: int = 20):
    top = {}
    if channel_df.empty:
        return top
    for cue in ["color", "texture", "shape"]:
        score_col = f"{cue}_specificity"
        cols = ["model", "layer", "channel", score_col]
        top[cue] = (
            channel_df.sort_values(score_col, ascending=False)
            .head(top_k)[cols]
            .to_dict(orient="records")
        )
    return top


def unique_in_order(values):
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def make_position_subset_spec(dataset_spec: DatasetSpec, selected_indices, image_positions, name_suffix: str = "vis_subset"):
    invalid_positions = [pos for pos in image_positions if pos < 0 or pos >= len(selected_indices)]
    if invalid_positions:
        raise IndexError(
            f"VIS_IMAGE_POSITIONS contains out-of-range positions {invalid_positions}; "
            f"valid range is 0..{len(selected_indices) - 1}."
        )

    subset_indices = [int(selected_indices[pos]) for pos in image_positions]
    return DatasetSpec(
        name=f"{dataset_spec.name}_{name_suffix}",
        dataset_type=dataset_spec.dataset_type,
        root=dataset_spec.root,
        split=dataset_spec.split,
        num_classes=dataset_spec.num_classes,
        sample_indices=subset_indices,
        labels_map=list(dataset_spec.labels_map),
        id_dataset_name=dataset_spec.id_dataset_name,
        domain_type=dataset_spec.domain_type,
        shift_type=dataset_spec.shift_type,
        class_map_name=dataset_spec.class_map_name,
        eval_protocol_name=dataset_spec.eval_protocol_name,
    )


def required_resnet_visualization_inputs(top_channels, image_positions, perturbation_to_cue: dict, target_layer_names, records_per_cue: int = 3):
    layers = []
    perturbation_names = ["original"]

    for cue, records in top_channels.items():
        perturbation = next((p for p, c in perturbation_to_cue.items() if c == cue), None)
        if perturbation is not None:
            perturbation_names.append(perturbation)
        for record in records[:records_per_cue]:
            if "layer" in record:
                layers.append(record["layer"])

    if not layers:
        layers = list(target_layer_names)

    return {
        "layers": unique_in_order(layers),
        "perturbations": unique_in_order(perturbation_names),
        "image_positions": unique_in_order([int(pos) for pos in image_positions]),
    }


def resnet_outputs_cover_visualization(outputs, requirements) -> bool:
    if not outputs:
        return False
    for perturbation in requirements["perturbations"]:
        if perturbation not in outputs:
            return False
        activations = outputs[perturbation].get("activations", {})
        if any(layer not in activations for layer in requirements["layers"]):
            return False
        cached_positions = outputs[perturbation].get("image_positions")
        if cached_positions is not None:
            if any(pos not in cached_positions for pos in requirements["image_positions"]):
                return False
        else:
            max_pos = max(requirements["image_positions"], default=-1)
            first_layer = requirements["layers"][0]
            if first_layer not in activations or activations[first_layer].shape[0] <= max_pos:
                return False
    return True


@torch.no_grad()
def extract_resnet_outputs_for_visualization(
    model,
    model_spec,
    dataset_spec,
    perturbation_names,
    layer_names,
    image_positions,
    selected_indices,
    build_loader_fn,
    data_config: DataConfig,
    dtype_for_cache,
    device,
):
    image_positions = unique_in_order([int(pos) for pos in image_positions])
    vis_spec = make_position_subset_spec(dataset_spec, selected_indices, image_positions, name_suffix="resnet_vis")
    target_layers = get_resnet_layers_by_name(model, layer_names)
    outputs = {}

    model.eval().to(device)
    for perturbation in perturbation_names:
        loader = build_loader_fn(
            vis_spec,
            perturbation,
            model_spec,
            batch_size=max(1, min(len(image_positions), data_config.batch_size)),
            shuffle=False,
        )
        logits_list = []
        labels_list = []
        activation_lists = {layer_name: [] for layer_name in layer_names}

        for images, labels in tqdm(loader, desc=f"ResNet vis cache {perturbation}"):
            logits, activations = forward_with_resnet_activations(model, images, target_layers, device=device)
            logits_list.append(logits.to(dtype=dtype_for_cache))
            labels_list.append(labels.detach().cpu())
            for layer_name in layer_names:
                activation_lists[layer_name].append(activations[layer_name].to(dtype=dtype_for_cache))

        outputs[perturbation] = {
            "logits": torch.cat(logits_list, dim=0),
            "labels": torch.cat(labels_list, dim=0),
            "activations": {
                layer_name: torch.cat(chunks, dim=0)
                for layer_name, chunks in activation_lists.items()
            },
            "image_positions": image_positions,
            "selected_indices": [int(selected_indices[pos]) for pos in image_positions],
            "cache_scope": "visualization",
        }

    outputs["_meta"] = {
        "cache_scope": "visualization",
        "image_positions": image_positions,
        "selected_indices": [int(selected_indices[pos]) for pos in image_positions],
        "layers": list(layer_names),
        "perturbations": list(perturbation_names),
    }
    return outputs


def register_vit_block_hooks(model, block_indices, hidden_cache):
    handles = []
    for block_idx in block_indices:
        def make_hook(idx):
            def hook(_module, _inputs, output):
                hidden_cache[idx] = output.detach().float().cpu()
            return hook
        handles.append(model.blocks[block_idx].register_forward_hook(make_hook(block_idx)))
    return handles


def register_vit_attention_input_hooks(model, block_indices, attention_input_cache):
    handles = []
    for block_idx in block_indices:
        def make_hook(idx):
            def hook(_module, inputs):
                attention_input_cache[idx] = inputs[0].detach()
            return hook
        handles.append(model.blocks[block_idx].attn.register_forward_pre_hook(make_hook(block_idx)))
    return handles


def cls_attention_map_from_attention_input(attn_module, x):
    B, N, C = x.shape
    qkv = attn_module.qkv(x).reshape(B, N, 3, attn_module.num_heads, attn_module.head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, _ = qkv.unbind(0)
    if hasattr(attn_module, "q_norm"):
        q = attn_module.q_norm(q)
    if hasattr(attn_module, "k_norm"):
        k = attn_module.k_norm(k)
    attn = (q * attn_module.scale) @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    cls_to_patch = attn[:, :, 0, 1:].mean(dim=1)
    grid = int(math.sqrt(cls_to_patch.shape[-1]))
    if grid * grid != cls_to_patch.shape[-1]:
        raise ValueError(f"Patch count is not square: {cls_to_patch.shape[-1]}")
    return cls_to_patch.reshape(B, grid, grid).detach().float().cpu()


@torch.no_grad()
def forward_with_vit_tokens(model, images, block_indices, device):
    temp_hidden = {}
    handles = register_vit_block_hooks(model, block_indices, temp_hidden)
    try:
        features = model.forward_features(images.to(device, non_blocking=True))
        logits = model.forward_head(features)
        patch_tokens = {idx: temp_hidden[idx][:, 1:, :].cpu() for idx in block_indices}
        cls_tokens = {idx: temp_hidden[idx][:, 0, :].cpu() for idx in block_indices}
    finally:
        for handle in handles:
            handle.remove()
    return logits.detach().float().cpu(), patch_tokens, cls_tokens


@torch.no_grad()
def forward_with_vit_tokens_and_attention(model, images, block_indices, device):
    temp_hidden = {}
    attention_inputs = {}
    handles = register_vit_block_hooks(model, block_indices, temp_hidden)
    handles += register_vit_attention_input_hooks(model, block_indices, attention_inputs)
    try:
        features = model.forward_features(images.to(device, non_blocking=True))
        logits = model.forward_head(features)
        patch_tokens = {idx: temp_hidden[idx][:, 1:, :].cpu() for idx in block_indices}
        cls_tokens = {idx: temp_hidden[idx][:, 0, :].cpu() for idx in block_indices}
        cls_attention = {
            idx: cls_attention_map_from_attention_input(model.blocks[idx].attn, attention_inputs[idx])
            for idx in block_indices
        }
    finally:
        for handle in handles:
            handle.remove()
    return logits.detach().float().cpu(), patch_tokens, cls_tokens, cls_attention


@torch.no_grad()
def run_vit_streaming_phase1(
    model,
    model_spec,
    dataset_spec,
    perturbation_names,
    block_indices,
    perturbation_to_cue: dict,
    build_loader_fn,
    device,
    max_batches: int | None = None,
):
    model.eval().to(device)
    metric_sums = {
        (perturbation, block_idx): init_metric_accumulator()
        for perturbation in perturbation_names if perturbation != "original"
        for block_idx in block_indices
    }

    for perturbation in perturbation_names:
        if perturbation == "original":
            continue

        clean_loader = build_loader_fn(dataset_spec, "original", model_spec)
        pert_loader = build_loader_fn(dataset_spec, perturbation, model_spec)
        desc = f"ViT streaming original vs {perturbation}"

        for batch_idx, ((clean_images, _), (pert_images, _)) in enumerate(
            tqdm(zip(clean_loader, pert_loader), total=len(clean_loader), desc=desc)
        ):
            if max_batches is not None and batch_idx >= max_batches:
                break

            _, H_clean, _ = forward_with_vit_tokens(model, clean_images, block_indices, device=device)
            _, H_pert, _ = forward_with_vit_tokens(model, pert_images, block_indices, device=device)

            for block_idx in block_indices:
                H_o = H_clean[block_idx].float()
                H_p = H_pert[block_idx].float()
                acc = metric_sums[(perturbation, block_idx)]
                X = H_o.flatten(1)
                Y = H_p.flatten(1)
                token_X = H_o.reshape(-1, H_o.shape[-1])
                token_Y = H_p.reshape(-1, H_p.shape[-1])
                acc["n_batches"] += 1
                acc["cka"] += safe_linear_cka(X, Y)
                acc["cosine_distance"] += cosine_distance_from_matrices(token_X, token_Y)
                acc["l2_distance"] += l2_distance_from_matrices(token_X, token_Y)
                acc["delta_mean"] += float((H_o - H_p).norm(dim=-1).mean().item())

            del H_clean, H_pert, clean_images, pert_images
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    rows = []
    for (perturbation, block_idx), acc in metric_sums.items():
        values = finalize_metric_accumulator(acc)
        rows.append({
            "model": "vit-b",
            "block": block_idx,
            "perturbation": perturbation,
            "cue": perturbation_to_cue[perturbation],
            "block_score": values.pop("delta_mean"),
            **values,
        })
    return pd.DataFrame(rows).sort_values(["cue", "perturbation", "cka"])


def select_top_vit_blocks(block_df: pd.DataFrame, top_k: int = 5):
    top = {}
    if block_df.empty:
        return top
    for cue in ["color", "texture", "shape"]:
        cue_df = block_df[block_df["cue"] == cue].copy()
        if cue_df.empty:
            top[cue] = []
            continue
        summary = (
            cue_df.groupby("block", as_index=False)
            .agg(mean_delta=("block_score", "mean"), mean_cka=("cka", "mean"))
        )
        summary["rank_score"] = summary["mean_delta"] - summary["mean_cka"]
        top[cue] = (
            summary.sort_values("rank_score", ascending=False)
            .head(top_k)
            .to_dict(orient="records")
        )
    return top


def required_vit_visualization_inputs(top_blocks, image_positions, perturbation_to_cue: dict, candidate_blocks, records_per_cue: int = 3):
    blocks = []
    perturbation_names = ["original"]

    for cue, records in top_blocks.items():
        perturbation = next((p for p, c in perturbation_to_cue.items() if c == cue), None)
        if perturbation is not None:
            perturbation_names.append(perturbation)
        for record in records[:records_per_cue]:
            if "block" in record:
                blocks.append(int(record["block"]))

    if not blocks:
        blocks = [int(block) for block in candidate_blocks]

    return {
        "blocks": unique_in_order(blocks),
        "perturbations": unique_in_order(perturbation_names),
        "image_positions": unique_in_order([int(pos) for pos in image_positions]),
    }


def vit_cached_image_index(outputs, perturbation: str, image_pos: int):
    cached_positions = outputs[perturbation].get("image_positions")
    if cached_positions is None:
        return int(image_pos)
    if int(image_pos) not in cached_positions:
        raise IndexError(f"image_pos={image_pos} is not in the ViT visualization cache.")
    return cached_positions.index(int(image_pos))


def vit_outputs_cover_visualization(outputs, requirements, require_attention: bool = False) -> bool:
    if not outputs:
        return False
    for perturbation in requirements["perturbations"]:
        if perturbation not in outputs:
            return False
        patch_tokens = outputs[perturbation].get("patch_tokens", {})
        if any(block not in patch_tokens for block in requirements["blocks"]):
            return False
        if require_attention:
            cls_attention = outputs[perturbation].get("cls_attention", {})
            if any(block not in cls_attention for block in requirements["blocks"]):
                return False
        cached_positions = outputs[perturbation].get("image_positions")
        if cached_positions is not None:
            if any(pos not in cached_positions for pos in requirements["image_positions"]):
                return False
        else:
            max_pos = max(requirements["image_positions"], default=-1)
            first_block = requirements["blocks"][0]
            if first_block not in patch_tokens or patch_tokens[first_block].shape[0] <= max_pos:
                return False
    return True


def vit_patch_delta_maps(outputs, block_idx: int, perturbation: str, image_pos: int):
    clean_idx = vit_cached_image_index(outputs, "original", image_pos)
    pert_idx = vit_cached_image_index(outputs, perturbation, image_pos)
    H_o = outputs["original"]["patch_tokens"][block_idx][clean_idx].float()
    H_p = outputs[perturbation]["patch_tokens"][block_idx][pert_idx].float()
    patch_norm_clean = H_o.norm(dim=-1)
    patch_norm_perturbed = H_p.norm(dim=-1)
    patch_delta = (H_o - H_p).norm(dim=-1)
    n = patch_delta.numel()
    grid = int(math.sqrt(n))
    if grid * grid != n:
        raise ValueError(f"Patch count is not square: {n}")
    return {
        "patch_norm_clean": patch_norm_clean.reshape(grid, grid),
        "patch_norm_perturbed": patch_norm_perturbed.reshape(grid, grid),
        "patch_delta": patch_delta.reshape(grid, grid),
    }


@torch.no_grad()
def extract_vit_outputs_for_visualization(
    model,
    model_spec,
    dataset_spec,
    perturbation_names,
    block_indices,
    image_positions,
    selected_indices,
    build_loader_fn,
    data_config: DataConfig,
    device,
    dtype_for_cache=torch.float16,
):
    image_positions = unique_in_order([int(pos) for pos in image_positions])
    vis_spec = make_position_subset_spec(dataset_spec, selected_indices, image_positions, name_suffix="vit_vis")
    outputs = {}

    model.eval().to(device)
    for perturbation in perturbation_names:
        loader = build_loader_fn(
            vis_spec,
            perturbation,
            model_spec,
            batch_size=max(1, min(len(image_positions), data_config.batch_size)),
            shuffle=False,
        )
        logits_list = []
        labels_list = []
        patch_lists = {block_idx: [] for block_idx in block_indices}
        cls_lists = {block_idx: [] for block_idx in block_indices}
        attention_lists = {block_idx: [] for block_idx in block_indices}

        for images, labels in tqdm(loader, desc=f"ViT vis cache {perturbation}"):
            logits, patch_tokens, cls_tokens, cls_attention = forward_with_vit_tokens_and_attention(
                model,
                images,
                block_indices,
                device=device,
            )
            logits_list.append(logits.to(dtype=dtype_for_cache))
            labels_list.append(labels.detach().cpu())
            for block_idx in block_indices:
                patch_lists[block_idx].append(patch_tokens[block_idx].to(dtype=dtype_for_cache))
                cls_lists[block_idx].append(cls_tokens[block_idx].to(dtype=dtype_for_cache))
                attention_lists[block_idx].append(cls_attention[block_idx].to(dtype=dtype_for_cache))

        outputs[perturbation] = {
            "logits": torch.cat(logits_list, dim=0),
            "labels": torch.cat(labels_list, dim=0),
            "patch_tokens": {
                block_idx: torch.cat(chunks, dim=0)
                for block_idx, chunks in patch_lists.items()
            },
            "cls_tokens": {
                block_idx: torch.cat(chunks, dim=0)
                for block_idx, chunks in cls_lists.items()
            },
            "cls_attention": {
                block_idx: torch.cat(chunks, dim=0)
                for block_idx, chunks in attention_lists.items()
            },
            "image_positions": image_positions,
            "selected_indices": [int(selected_indices[pos]) for pos in image_positions],
            "cache_scope": "visualization",
        }

    outputs["_meta"] = {
        "cache_scope": "visualization",
        "image_positions": image_positions,
        "selected_indices": [int(selected_indices[pos]) for pos in image_positions],
        "blocks": [int(block) for block in block_indices],
        "perturbations": list(perturbation_names),
        "resize_size": int(model_spec.resize_size),
    }
    return outputs


@torch.no_grad()
def extract_vit_outputs_for_cache(
    model,
    model_spec,
    dataset_spec,
    perturbation_names,
    block_indices,
    build_loader_fn,
    device,
    dtype_for_cache=torch.float16,
    max_batches: int | None = None,
):
    model.eval().to(device)
    outputs = {}

    for perturbation in perturbation_names:
        loader = build_loader_fn(dataset_spec, perturbation, model_spec)
        logits_list = []
        labels_list = []
        patch_lists = {block_idx: [] for block_idx in block_indices}
        cls_lists = {block_idx: [] for block_idx in block_indices}

        for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"ViT cache {perturbation}")):
            if max_batches is not None and batch_idx >= max_batches:
                break
            logits, patch_tokens, cls_tokens = forward_with_vit_tokens(
                model,
                images,
                block_indices,
                device=device,
            )
            logits_list.append(logits.to(dtype=dtype_for_cache))
            labels_list.append(labels.detach().cpu())
            for block_idx in block_indices:
                patch_lists[block_idx].append(patch_tokens[block_idx].to(dtype=dtype_for_cache))
                cls_lists[block_idx].append(cls_tokens[block_idx].to(dtype=dtype_for_cache))

        outputs[perturbation] = {
            "logits": torch.cat(logits_list, dim=0),
            "labels": torch.cat(labels_list, dim=0),
            "patch_tokens": {
                block_idx: torch.cat(chunks, dim=0)
                for block_idx, chunks in patch_lists.items()
            },
            "cls_tokens": {
                block_idx: torch.cat(chunks, dim=0)
                for block_idx, chunks in cls_lists.items()
            },
        }

    return outputs


def normalize_map(x, eps: float = 1e-8):
    x = np.asarray(x, dtype=np.float32)
    x = x - float(np.nanmin(x))
    denom = float(np.nanmax(x)) + eps
    return x / denom


def overlay_heatmap(image, heatmap, alpha: float = 0.45, cmap_name: str = "magma"):
    image = to_display_image(image).astype(np.float32) / 255.0
    heatmap = normalize_map(heatmap)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    colored = plt.get_cmap(cmap_name)(heatmap)[..., :3]
    overlay = (1.0 - alpha) * image + alpha * colored
    return np.clip(overlay, 0.0, 1.0)


def plot_resnet_channel_comparison(raw_image, A_original, A_perturbed, layer, channel, perturbation, perturbation_to_cue, save_path):
    original_map = A_original[channel].detach().float().cpu().numpy()
    perturbed_map = A_perturbed[channel].detach().float().cpu().numpy()
    diff_map = np.abs(original_map - perturbed_map)

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.3))
    axes[0].imshow(to_display_image(raw_image))
    axes[0].set_title("image")
    axes[1].imshow(normalize_map(original_map), cmap="magma")
    axes[1].set_title("clean act")
    axes[2].imshow(normalize_map(perturbed_map), cmap="magma")
    axes[2].set_title(f"{perturbation} act")
    axes[3].imshow(normalize_map(diff_map), cmap="magma")
    axes[3].set_title("abs diff")
    axes[4].imshow(overlay_heatmap(raw_image, original_map))
    axes[4].set_title("clean overlay")

    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"ResNet {layer} channel={channel} cue={perturbation_to_cue[perturbation]}")
    fig.tight_layout()
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_vit_patch_delta(raw_image, patch_norm_clean, patch_norm_perturbed, patch_delta, block, perturbation, perturbation_to_cue, save_path):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.3))
    axes[0].imshow(to_display_image(raw_image))
    axes[0].set_title("image")
    axes[1].imshow(normalize_map(patch_norm_clean), cmap="magma")
    axes[1].set_title("clean norm")
    axes[2].imshow(normalize_map(patch_norm_perturbed), cmap="magma")
    axes[2].set_title(f"{perturbation} norm")
    axes[3].imshow(normalize_map(patch_delta), cmap="magma")
    axes[3].set_title("delta")
    axes[4].imshow(overlay_heatmap(raw_image, patch_delta))
    axes[4].set_title("delta overlay")

    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"ViT block={block} cue={perturbation_to_cue[perturbation]}")
    fig.tight_layout()
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_vit_attention_activation_summary(
    original_image,
    transformed_image,
    cls_attention_clean,
    cls_attention_perturbed,
    patch_norm_clean,
    patch_norm_perturbed,
    patch_delta,
    block,
    perturbation,
    perturbation_to_cue,
    save_path,
):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.0))
    axes = np.asarray(axes)

    axes[0, 0].imshow(to_display_image(original_image))
    axes[0, 0].set_title("original input")
    axes[0, 1].imshow(to_display_image(transformed_image))
    axes[0, 1].set_title(f"{perturbation} input")
    axes[0, 2].imshow(normalize_map(cls_attention_clean), cmap="magma")
    axes[0, 2].set_title("original CLS attn")
    axes[0, 3].imshow(normalize_map(cls_attention_perturbed), cmap="magma")
    axes[0, 3].set_title(f"{perturbation} CLS attn")

    axes[1, 0].imshow(normalize_map(patch_norm_clean), cmap="magma")
    axes[1, 0].set_title("original CH norm")
    axes[1, 1].imshow(normalize_map(patch_norm_perturbed), cmap="magma")
    axes[1, 1].set_title(f"{perturbation} CH norm")
    axes[1, 2].imshow(normalize_map(patch_delta), cmap="magma")
    axes[1, 2].set_title("patch delta")
    axes[1, 3].imshow(overlay_heatmap(transformed_image, patch_delta))
    axes[1, 3].set_title("delta overlay")

    for ax in axes.reshape(-1):
        ax.axis("off")
    fig.suptitle(f"ViT block={block} cue={perturbation_to_cue[perturbation]}")
    fig.tight_layout()
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return save_path


def load_json_if_exists(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resnet_cached_image_index(outputs, perturbation: str, image_pos: int):
    cached_positions = outputs[perturbation].get("image_positions")
    if cached_positions is None:
        return int(image_pos)
    if int(image_pos) not in cached_positions:
        raise IndexError(f"image_pos={image_pos} is not in the ResNet visualization cache.")
    return cached_positions.index(int(image_pos))
