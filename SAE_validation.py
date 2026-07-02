import copy
import csv
from datetime import timedelta
import itertools
import os
import traceback
from pathlib import Path

import time
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
import torch
import timm
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader
from torchvision.datasets import Imagenette

from Utils.Config import SAEExperimentConfig
from Utils.early_stopping import unwrap_compiled_model
from Utils.SAE_plot_utils import (
    _plot_overlay,
    collect_patch_latents_for_batch,
    model_tensor_to_image,
    save_plot,
    save_trial_plots,
)
from Utils.SAE_utils import (
    TransformDataset,
    collect_tokens_with_hook,
    compute_b_dec_init_streaming,
    describe_dataset,
    evaluate_sae_tokens,
    fit_token_normalizer_streaming,
    get_torch_dtype,
    jsonable,
    normalize_tokens_inplace,
    save_json,
    train_sae_auto,
)


# =============================================================================
# Experiment Settings
# =============================================================================
# Edit this block for normal runs. The dataclasses in Utils/Config.py provide
# defaults, but this script-level block is the visible experiment entry point.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_ROOT = "outputs/SAE_validation"
GRID_DIR_NAME = "grid_search"

# Dataset / backbone
DATA_ROOT = "data"
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
MODEL_NAME = "vit_base_patch16_224"
PRETRAINED = True
TARGET_BLOCK = 10
TOKEN_SCOPE = "all"  # "cls", "patch", "all"

# Dataloader / token extraction
DATALOADER_BATCH_SIZE = 64
DATALOADER_NUM_WORKERS = 0
DATALOADER_PIN_MEMORY = True
MAX_TRAIN_TOKENS = 1_000_000
MAX_VAL_TOKENS = None
TOKEN_SOURCE_MODE = "auto"  # "auto", "cache", "stream"
TOKEN_CACHE_DTYPE = "float16"
TOKEN_CACHE_MAX_CPU_GIB = 8.0

# SAE training
EPOCHS = 350
SAE_LR = 1e-4
SAE_WEIGHT_DECAY = 0.0
EXPANSION = 64
B_DEC_INIT_MODE = "geom"  # "zero", "mean", "geom"
B_DEC_INIT_GRID = [B_DEC_INIT_MODE]  # Set multiple modes here for grid search.
SAE_ACTIVE_THRESHOLD = 0.2
L1_REG = 1e-4
SAE_BATCH_SIZE = 7096
MODEL_COMPILE = True

# AMP / safety checks
USE_AMP = torch.cuda.is_available()
SAE_AMP_DTYPE = "bfloat16"
SAE_CHECK_FINITE = True
MATMUL_PRECISION = "high"

# Early stopping
EARLY_STOPPING_PATIENCE = 30
EARLY_STOPPING_EPS = 5e-5
EARLY_STOPPING_VERBOSE = False

# Grid search
RUN_GRID_SEARCH = True
GRID_MAX_TRIALS = None
GRID_METRIC = "val_nmse"
GRID_MODE = "min"
GRID_SPACE = {
    "sae.expansion": [16, 32, 64],
    "sae.l1_reg": [3e-5, 1e-4, 3e-4],
    "optim_config.lr": [5e-5, 1e-4],
    "sae.dec_bias_mode": B_DEC_INIT_GRID,
    "sae.active_threshold": [0.1, 0.2],
}

# Latent masking / feature-alignment validation
RUN_LATENT_MASKING_ALIGNMENT = True
LATENT_MASK_NUM_LATENTS = 16
LATENT_MASK_NUM_TOKENS = 512
LATENT_MASK_TOP_FEATURES = 12
LATENT_MASK_SEED = 0
LATENT_MASK_ACTIVE_ONLY = True
LATENT_MASK_SAVE_TENSORS = False
RUN_LATENT_OVERLAY_VISUALIZATION = True
LATENT_OVERLAY_NUM_IMAGES = 4
LATENT_OVERLAY_LATENTS_PER_IMAGE = 4
LATENT_OVERLAY_SEED = 0


def build_default_config():
    config = SAEExperimentConfig().validate()

    config.extraction_config.device = DEVICE
    config.output.root_dir = OUTPUT_ROOT
    config.output.grid_dir_name = GRID_DIR_NAME

    config.train_dataset_spec.root = DATA_ROOT
    config.train_dataset_spec.split = TRAIN_SPLIT
    config.val_dataset_spec.root = DATA_ROOT
    config.val_dataset_spec.split = VAL_SPLIT

    if config.model_spec is not None:
        config.model_spec.model_name = MODEL_NAME
        config.model_spec.pretrained_weight = "pretrained" if PRETRAINED else ""

    config.hook.target_block = TARGET_BLOCK
    config.hook.token_scope = TOKEN_SCOPE

    config.data_config.batch_size = DATALOADER_BATCH_SIZE
    config.data_config.num_workers = DATALOADER_NUM_WORKERS
    config.data_config.pin_memory = DATALOADER_PIN_MEMORY
    config.token.max_train_tokens = MAX_TRAIN_TOKENS
    config.token.max_val_tokens = MAX_VAL_TOKENS
    config.token.source_mode = TOKEN_SOURCE_MODE
    config.token.cache_dtype = TOKEN_CACHE_DTYPE
    config.token.cache_max_cpu_gib = TOKEN_CACHE_MAX_CPU_GIB

    config.optim_config.epochs = EPOCHS
    config.optim_config.lr = SAE_LR
    config.optim_config.weight_decay = SAE_WEIGHT_DECAY
    config.optim_config.use_amp = USE_AMP

    config.sae.expansion = EXPANSION
    config.sae.dec_bias_mode = B_DEC_INIT_MODE
    config.sae.active_threshold = SAE_ACTIVE_THRESHOLD
    config.sae.l1_reg = L1_REG
    config.sae.batch_size = SAE_BATCH_SIZE
    config.sae.model_compile = MODEL_COMPILE
    config.sae.amp_dtype = SAE_AMP_DTYPE
    config.sae.check_finite = SAE_CHECK_FINITE
    config.sae.matmul_precision = MATMUL_PRECISION

    config.early_stopping.patience = EARLY_STOPPING_PATIENCE
    config.early_stopping.eps = EARLY_STOPPING_EPS
    config.early_stopping.save_verbose = EARLY_STOPPING_VERBOSE

    config.grid.enabled = RUN_GRID_SEARCH
    config.grid.max_trials = GRID_MAX_TRIALS
    config.grid.metric = GRID_METRIC
    config.grid.mode = GRID_MODE
    config.grid.space = GRID_SPACE

    config.validate()
    if config.extraction_config.device == "cuda" and not torch.cuda.is_available():
        config.extraction_config.device = "cpu"
        config.optim_config.use_amp = False
    return config


def build_backbone_and_transform(config: SAEExperimentConfig):
    model_name = MODEL_NAME
    pretrained = PRETRAINED
    if config.model_spec is not None:
        model_name = config.model_spec.model_name
        pretrained = bool(config.model_spec.pretrained_weight)

    model = timm.create_model(model_name, pretrained=pretrained)
    model.eval().to(config.extraction_config.device)
    data_config = resolve_model_data_config(model)
    transform = create_transform(**data_config, is_training=False)
    mean = list(data_config.get("mean", (0.485, 0.456, 0.406)))
    std = list(data_config.get("std", (0.229, 0.224, 0.225)))
    return model, transform, mean, std


def build_imagenette_dataset(dataset_spec, transform):
    if dataset_spec.dataset_type != "imagenette":
        raise ValueError(f"SAE_validation currently supports dataset_type='imagenette', got {dataset_spec.dataset_type!r}.")
    dataset = Imagenette(
        root=dataset_spec.root or "data",
        split=dataset_spec.split,
        download=False,
    )
    return TransformDataset(dataset, transform=transform)


def build_train_val_loaders(config, transform):
    train_dataset = build_imagenette_dataset(config.train_dataset_spec, transform)
    val_dataset = build_imagenette_dataset(config.val_dataset_spec, transform)
    pin_memory = bool(config.data_config.pin_memory and str(config.extraction_config.device).startswith("cuda"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data_config.batch_size,
        shuffle=True,
        num_workers=config.data_config.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data_config.batch_size,
        shuffle=False,
        num_workers=config.data_config.num_workers,
        pin_memory=pin_memory,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def print_config_summary(config):
    print("\nSAE experiment config")
    print("=" * 32)
    print(f"model_name             : {MODEL_NAME}")
    print(f"pretrained             : {PRETRAINED}")
    print(f"device                 : {config.extraction_config.device}")
    print(f"target_block           : {config.hook.target_block}")
    print(f"token_scope            : {config.hook.token_scope}")
    print(f"max_train_tokens       : {config.token.max_train_tokens}")
    print(f"max_val_tokens         : {config.token.max_val_tokens}")
    print(f"token_source_mode      : {config.token.source_mode}")
    print(f"expansion              : {config.sae.expansion}")
    print(f"dec_bias_mode          : {config.sae.dec_bias_mode}")
    print(f"active_threshold       : {config.sae.active_threshold}")
    print(f"epochs                 : {config.optim_config.epochs}")
    print(f"lr                     : {config.optim_config.lr}")
    print(f"l1_reg                 : {config.sae.l1_reg}")
    print(f"sae_batch_size         : {config.sae.batch_size}")
    print(f"use_amp                : {config.optim_config.use_amp}")
    print(f"amp_dtype              : {config.sae.amp_dtype}")
    print(f"early_stop_patience    : {config.early_stopping.patience}")


def run_sae_trial(config, trial_dir, trial_id=None):
    config = copy.deepcopy(config).validate()
    trial_dir = Path(trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)
    save_json(trial_dir / "config.json", config)
    print_config_summary(config)

    torch.set_float32_matmul_precision(config.sae.matmul_precision)
    model, transform, mean, std = build_backbone_and_transform(config)
    train_dataset, val_dataset, train_loader, val_loader = build_train_val_loaders(config, transform)

    describe_dataset("train_dataset", train_dataset)
    describe_dataset("val_dataset", val_dataset)

    print("\nFitting train-token normalizer from streaming train tokens...")
    token_stats, train_token_count = fit_token_normalizer_streaming(
        model,
        train_loader,
        max_tokens=config.token.max_train_tokens,
        target_block=config.hook.target_block,
        token_scope=config.hook.token_scope,
        device=config.extraction_config.device,
    )
    print(f"train tokens seen for normalizer: {train_token_count:,}")

    print("\nFitting b_dec init from the same train-token stream...")
    b_dec_init = compute_b_dec_init_streaming(
        model,
        train_loader,
        token_stats,
        max_tokens=config.token.max_train_tokens,
        target_block=config.hook.target_block,
        token_scope=config.hook.token_scope,
        device=config.extraction_config.device,
        sae_config=config.sae,
    )
    print(f"b_dec init: mode={config.sae.dec_bias_mode}, shape={tuple(b_dec_init.shape)}")

    print("\nCollecting cached validation tokens...")
    val_tokens, val_labels = collect_tokens_with_hook(
        model,
        val_loader,
        max_tokens=config.token.max_val_tokens,
        target_block=config.hook.target_block,
        token_scope=config.hook.token_scope,
        device=config.extraction_config.device,
        cache_dtype=get_torch_dtype(config.token.cache_dtype),
        return_labels=True,
    )
    val_tokens = normalize_tokens_inplace(val_tokens.float(), token_stats)
    print(f"val tokens cached: {tuple(val_tokens.shape)}")

    input_dim = int(token_stats["mean"].shape[1])
    hidden_dim = int(input_dim * config.sae.expansion)
    checkpoint_path = trial_dir / "best_sae_state.pt"

    trained_sae, history = train_sae_auto(
        model,
        train_loader,
        val_tokens,
        token_stats,
        input_dim,
        hidden_dim,
        b_dec_init,
        config,
        checkpoint_path=checkpoint_path,
        expected_tokens=train_token_count,
    )

    validation_metrics = evaluate_sae_tokens(
        trained_sae,
        val_tokens,
        batch_size=config.sae.batch_size,
        threshold=config.sae.active_threshold,
        device=config.extraction_config.device,
    )

    latent_masking_alignment = run_latent_masking_alignment_validation(
        trained_sae,
        val_tokens,
        val_labels,
        trial_dir,
        config,
    )
    latent_overlay_visualizations = run_latent_overlay_visualizations(
        model,
        val_dataset,
        trained_sae,
        token_stats,
        trial_dir,
        config,
        mean,
        std,
        latent_ids=latent_masking_alignment.get("selected_latents", []),
    )

    diagnostics = save_trial_plots(
        trial_dir=trial_dir,
        model=model,
        train_history=history,
        sae=trained_sae,
        token_stats=token_stats,
        val_tokens=val_tokens,
        val_labels=val_labels,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        mean=mean,
        std=std,
    )

    best_rows = [row for row in history if row.get("is_best")]
    best_row = best_rows[-1] if best_rows else (min(history, key=lambda row: row["normalized_mse"]) if history else {})
    summary = {
        "trial_id": trial_id,
        "status": "completed",
        "trial_dir": str(trial_dir),
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "train_token_count": train_token_count,
        "val_token_count": int(val_tokens.shape[0]),
        "best_epoch": best_row.get("epoch"),
        "best_val_nmse": best_row.get("normalized_mse"),
        "best_active_mean_count": best_row.get("active_mean_count"),
        "final_validation_metrics": validation_metrics,
        "latent_masking_alignment": latent_masking_alignment,
        "latent_overlay_visualizations": latent_overlay_visualizations,
        "checkpoint_path": str(checkpoint_path),
        "diagnostics": diagnostics,
    }
    save_json(trial_dir / "history.json", history)
    save_json(trial_dir / "summary.json", summary)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("trial summary:", jsonable(summary))
    return summary


def iter_grid_items(space):
    keys = list(space.keys())
    values = [space[key] for key in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def set_nested_attr(obj, dotted_key, value):
    parts = dotted_key.split(".")
    target = obj
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


def apply_grid_overrides(config, overrides):
    config = copy.deepcopy(config)
    for key, value in overrides.items():
        set_nested_attr(config, key, value)
    return config.validate()


def format_tuning_hp(overrides):
    return ", ".join(f"{key}={value}" for key, value in overrides.items())


def flatten_dict(data, prefix=""):
    flat = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, name))
        elif isinstance(value, (list, tuple)):
            flat[name] = jsonable(value)
        else:
            flat[name] = jsonable(value)
    return flat


def write_grid_summary_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    flat_rows = [flatten_dict(row) for row in rows]
    fieldnames = sorted({key for row in flat_rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in flat_rows:
            writer.writerow(row)


def _mse_mean(x):
    return float(torch.mean(x.float().pow(2)).item())


def _top_label_counts(labels, mask, k=5):
    if labels is None or labels.numel() == 0 or not bool(mask.any()):
        return []
    active_labels = labels[mask.detach().cpu()]
    if active_labels.numel() == 0:
        return []
    unique, counts = torch.unique(active_labels, return_counts=True)
    order = torch.argsort(counts, descending=True)[: int(k)]
    return [
        {"label": int(unique[idx].item()), "count": int(counts[idx].item())}
        for idx in order
    ]


def _write_latent_alignment_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "latent_id",
        "mean_activation",
        "activation_frequency",
        "active_mean_activation",
        "delta_nmse",
        "masked_nmse",
        "recon_shift_mse",
        "decoder_norm",
        "encoder_norm",
        "encoder_decoder_cosine",
        "top_feature_indices",
        "top_feature_weights",
        "top_feature_activation_corrs",
        "top_active_labels",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (list, tuple, dict)):
                    out[key] = jsonable(value)
                else:
                    out[key] = value
            writer.writerow(out)


@torch.no_grad()
def run_latent_masking_alignment_validation(sae, val_tokens, val_labels, trial_dir, config):
    """Mask random SAE latents and align their reconstruction effect to input features.

    The validation tokens are normalized ViT block features. Each latent is
    aligned to the input feature dimensions where its decoder vector has the
    largest absolute weights. Masking uses the linear decoder identity:
    x_hat_without_j = x_hat - z_j * W_dec[j].
    """

    if not RUN_LATENT_MASKING_ALIGNMENT:
        return {"enabled": False}
    if val_tokens is None or int(val_tokens.shape[0]) == 0:
        return {"enabled": True, "status": "skipped", "reason": "empty val_tokens"}

    start_time = time.perf_counter()
    trial_dir = Path(trial_dir)
    out_dir = trial_dir / "latent_masking_alignment"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_sae = unwrap_compiled_model(sae).eval().to(config.extraction_config.device)
    num_tokens = min(int(LATENT_MASK_NUM_TOKENS), int(val_tokens.shape[0]))
    generator = torch.Generator(device="cpu").manual_seed(int(LATENT_MASK_SEED))
    sample_idx = torch.randperm(int(val_tokens.shape[0]), generator=generator)[:num_tokens]
    sample_tokens_cpu = val_tokens[sample_idx].float().cpu()
    sample_labels = val_labels[sample_idx].detach().cpu() if val_labels is not None else None
    sample_tokens = sample_tokens_cpu.to(config.extraction_config.device)

    x_hat, z = base_sae(sample_tokens)
    z = z.float()
    x_hat = x_hat.float()
    hidden_dim = int(z.shape[1])
    threshold = float(config.sae.active_threshold)
    active_freq = (z > threshold).float().mean(dim=0).cpu()
    mean_activation = z.mean(dim=0).cpu()

    if LATENT_MASK_ACTIVE_ONLY:
        candidates = torch.nonzero(active_freq > 0, as_tuple=False).flatten()
        if candidates.numel() == 0:
            candidates = torch.arange(hidden_dim)
    else:
        candidates = torch.arange(hidden_dim)
    shuffled = candidates[torch.randperm(candidates.numel(), generator=generator)]
    selected = shuffled[: min(int(LATENT_MASK_NUM_LATENTS), int(shuffled.numel()))].tolist()

    centered = sample_tokens.float()
    token_var = torch.mean((centered - centered.mean(dim=0, keepdim=True)).pow(2)).clamp_min(1e-12)
    full_mse = torch.mean((x_hat - centered).pow(2))
    full_nmse = float((full_mse / token_var).item())

    w_dec = base_sae.W_dec.detach().float()
    w_enc = base_sae.W_enc.detach().float()
    rows = []
    details = []
    top_k = min(int(LATENT_MASK_TOP_FEATURES), int(w_dec.shape[1]))

    for latent_id in selected:
        latent_id = int(latent_id)
        activation = z[:, latent_id].detach()
        decoder_vec = w_dec[latent_id].to(x_hat.device)
        contribution = activation[:, None] * decoder_vec[None, :]
        masked_recon = x_hat - contribution
        masked_mse = torch.mean((masked_recon - centered).pow(2))
        masked_nmse = float((masked_mse / token_var).item())
        recon_shift_mse = _mse_mean(contribution)
        active_mask = (activation > threshold).detach().cpu()
        active_values = activation.detach().cpu()[active_mask]

        top_abs, top_idx = torch.topk(w_dec[latent_id].abs().cpu(), k=top_k)
        top_idx = top_idx.tolist()
        top_weights = [float(w_dec[latent_id, idx].item()) for idx in top_idx]
        feature_corrs = []
        activation_cpu = activation.detach().cpu()
        activation_std = activation_cpu.std().clamp_min(1e-8)
        for feature_idx in top_idx:
            feature_values = sample_tokens_cpu[:, feature_idx]
            feature_std = feature_values.std().clamp_min(1e-8)
            corr = torch.mean(
                ((activation_cpu - activation_cpu.mean()) / activation_std)
                * ((feature_values - feature_values.mean()) / feature_std)
            )
            feature_corrs.append(float(corr.item()))

        enc_vec = w_enc[:, latent_id].detach().cpu()
        dec_vec = w_dec[latent_id].detach().cpu()
        enc_dec_cos = torch.nn.functional.cosine_similarity(enc_vec, dec_vec, dim=0).item()
        label_counts = _top_label_counts(sample_labels, active_mask, k=5)
        row = {
            "latent_id": latent_id,
            "mean_activation": float(mean_activation[latent_id].item()),
            "activation_frequency": float(active_freq[latent_id].item()),
            "active_mean_activation": float(active_values.mean().item()) if active_values.numel() else 0.0,
            "delta_nmse": float(masked_nmse - full_nmse),
            "masked_nmse": masked_nmse,
            "recon_shift_mse": recon_shift_mse,
            "decoder_norm": float(dec_vec.norm().item()),
            "encoder_norm": float(enc_vec.norm().item()),
            "encoder_decoder_cosine": float(enc_dec_cos),
            "top_feature_indices": top_idx,
            "top_feature_weights": top_weights,
            "top_feature_activation_corrs": feature_corrs,
            "top_active_labels": label_counts,
        }
        rows.append(row)
        details.append(
            {
                **row,
                "top_feature_abs_weights": [float(v.item()) for v in top_abs],
            }
        )

    if selected:
        selected_contribution = z[:, selected] @ w_dec[selected].to(z.device)
        all_masked_recon = x_hat - selected_contribution
        all_masked_nmse = float((torch.mean((all_masked_recon - centered).pow(2)) / token_var).item())
        all_masked_shift_mse = _mse_mean(selected_contribution)
    else:
        all_masked_nmse = full_nmse
        all_masked_shift_mse = 0.0

    _write_latent_alignment_csv(out_dir / "latent_alignment.csv", rows)
    save_json(out_dir / "latent_alignment.json", details)
    summary = {
        "enabled": True,
        "status": "completed",
        "num_sample_tokens": int(num_tokens),
        "hidden_dim": int(hidden_dim),
        "candidate_pool_size": int(candidates.numel()),
        "selected_latents": [int(x) for x in selected],
        "active_only": bool(LATENT_MASK_ACTIVE_ONLY),
        "active_threshold": threshold,
        "full_reconstruction_nmse": full_nmse,
        "all_selected_masked_nmse": all_masked_nmse,
        "all_selected_delta_nmse": float(all_masked_nmse - full_nmse),
        "all_selected_recon_shift_mse": all_masked_shift_mse,
        "csv": str(out_dir / "latent_alignment.csv"),
        "json": str(out_dir / "latent_alignment.json"),
        "elapsed_seconds": float(time.perf_counter() - start_time),
    }
    if LATENT_MASK_SAVE_TENSORS:
        tensor_path = out_dir / "latent_masking_tensors.pt"
        torch.save(
            {
                "sample_indices": sample_idx,
                "sample_tokens": sample_tokens_cpu,
                "sample_labels": sample_labels,
                "selected_latents": torch.tensor(selected, dtype=torch.long),
                "z_selected": z[:, selected].detach().cpu() if selected else torch.empty(num_tokens, 0),
                "full_reconstruction": x_hat.detach().cpu(),
            },
            tensor_path,
        )
        summary["tensors"] = str(tensor_path)
    save_json(out_dir / "summary.json", summary)
    print(
        "latent masking/alignment:",
        f"sample_tokens={num_tokens}",
        f"selected_latents={len(selected)}",
        f"full_nmse={full_nmse:.6f}",
        f"all_masked_delta_nmse={summary['all_selected_delta_nmse']:.6f}",
    )
    return summary


@torch.no_grad()
def run_latent_overlay_visualizations(
    model,
    val_dataset,
    sae,
    token_stats,
    trial_dir,
    config,
    mean,
    std,
    latent_ids=None,
):
    """Save image-level SAE latent activation overlays for a few validation images."""

    if not RUN_LATENT_OVERLAY_VISUALIZATION:
        return {"enabled": False}
    token_scope = str(config.hook.token_scope).lower()
    if token_scope == "cls":
        return {
            "enabled": True,
            "status": "skipped",
            "reason": "CLS-token SAE has no patch grid to overlay on the image.",
        }
    if not hasattr(model, "blocks"):
        return {
            "enabled": True,
            "status": "skipped",
            "reason": "Overlay visualization requires a ViT-style model with .blocks.",
        }

    start_time = time.perf_counter()
    trial_dir = Path(trial_dir)
    out_dir = trial_dir / "latent_masking_alignment" / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device="cpu").manual_seed(int(LATENT_OVERLAY_SEED))
    num_images = min(int(LATENT_OVERLAY_NUM_IMAGES), len(val_dataset))
    image_indices = torch.randperm(len(val_dataset), generator=generator)[:num_images].tolist()
    candidate_latents = [int(x) for x in latent_ids or []]

    records = []
    for image_idx in image_indices:
        image_tensor, label = val_dataset[int(image_idx)]
        images = image_tensor.unsqueeze(0)
        z = collect_patch_latents_for_batch(
            images,
            model,
            sae,
            token_stats,
            config.hook.target_block,
            config.hook.token_scope,
            config.extraction_config.device,
        )
        z_image = z[0]
        if candidate_latents:
            valid_ids = [latent_id for latent_id in candidate_latents if 0 <= latent_id < z_image.shape[1]]
            if valid_ids:
                scores = torch.tensor([float(z_image[:, latent_id].max().item()) for latent_id in valid_ids])
                order = torch.argsort(scores, descending=True)[: int(LATENT_OVERLAY_LATENTS_PER_IMAGE)]
                selected = [valid_ids[int(i)] for i in order.tolist()]
            else:
                selected = []
        else:
            selected = []
        if not selected:
            selected = z_image.pow(2).mean(dim=0).topk(min(int(LATENT_OVERLAY_LATENTS_PER_IMAGE), z_image.shape[1])).indices.tolist()
            selected = [int(x) for x in selected]

        image_np = model_tensor_to_image(image_tensor, mean, std)
        fig, axes = plt.subplots(1, len(selected) + 1, figsize=(3.2 * (len(selected) + 1), 3.4))
        if len(selected) == 0:
            axes = [axes]
        axes[0].imshow(image_np)
        axes[0].set_title(f"original\nidx={int(image_idx)}, label={int(label)}", fontsize=9)
        axes[0].axis("off")
        latent_records = []
        for ax, latent_id in zip(axes[1:], selected):
            patch_map = z_image[:, int(latent_id)]
            _plot_overlay(ax, image_np, patch_map)
            active_count = int((patch_map > float(config.sae.active_threshold)).sum().item())
            peak = float(patch_map.max().item())
            ax.set_title(f"latent={int(latent_id)}\nactive={active_count}, peak={peak:.2f}", fontsize=9)
            latent_records.append(
                {
                    "latent_id": int(latent_id),
                    "active_patches": active_count,
                    "peak_activation": peak,
                    "mean_activation": float(patch_map.mean().item()),
                }
            )
        fig.suptitle("Random masked SAE latent overlays on original image", fontsize=11)
        fig.tight_layout()
        path = save_plot(fig, out_dir, f"latent_overlay_image_{int(image_idx):05d}", dpi=config.output.plot_dpi)
        records.append(
            {
                "image_idx": int(image_idx),
                "label": int(label),
                "path": str(path),
                "latents": latent_records,
            }
        )

    summary = {
        "enabled": True,
        "status": "completed",
        "num_images": len(records),
        "latents_per_image": int(LATENT_OVERLAY_LATENTS_PER_IMAGE),
        "image_indices": [row["image_idx"] for row in records],
        "candidate_latents": candidate_latents,
        "records": records,
        "output_dir": str(out_dir),
        "elapsed_seconds": float(time.perf_counter() - start_time),
    }
    save_json(out_dir / "summary.json", summary)
    print(
        "latent overlay visualization:",
        f"images={len(records)}",
        f"latents_per_image={LATENT_OVERLAY_LATENTS_PER_IMAGE}",
        f"output_dir={out_dir}",
    )
    return summary


def select_best_trial(summaries, metric, mode):
    completed = [row for row in summaries if row.get("status") == "completed" and row.get(metric) is not None]
    if not completed:
        return None
    reverse = mode == "max"
    return sorted(completed, key=lambda row: row[metric], reverse=reverse)[0]


def run_sae_grid_search(base_config: SAEExperimentConfig):
    base_config = copy.deepcopy(base_config).validate()
    root_dir = Path(base_config.output.root_dir) / base_config.output.grid_dir_name
    root_dir.mkdir(parents=True, exist_ok=True)

    trial_summaries = []
    grid_items = list(iter_grid_items(base_config.grid.space))
    if base_config.grid.max_trials is not None:
        grid_items = grid_items[: int(base_config.grid.max_trials)]

    total_trials = len(grid_items)
    for trial_idx, overrides in enumerate(grid_items):
        trial_dir = root_dir / f"trial_{trial_idx:04d}"
        if trial_dir.exists():
            print(f"Trial {trial_idx + 1}/{total_trials} already exists, skipping...")
            continue

        trial_start_time = time.perf_counter()
        trial_id = f"trial_{trial_idx:04d}"
        trial_dir = root_dir / trial_id
        print(f"\n[{trial_idx + 1}/{total_trials} trial] tuning HP: {format_tuning_hp(overrides)}")
        print(f"===== {trial_id} =====")
        config = apply_grid_overrides(base_config, overrides)

        try:
            summary = run_sae_trial(config, trial_dir=trial_dir, trial_id=trial_id)
            row = {
                "trial_id": trial_id,
                "status": "completed",
                "trial_dir": str(trial_dir),
                "val_nmse": summary["final_validation_metrics"]["normalized_mse"],
                "cosine": summary["final_validation_metrics"]["cosine"],
                "mean_l0": summary["final_validation_metrics"]["mean_l0"],
                "best_epoch": summary.get("best_epoch"),
                "best_val_nmse": summary.get("best_val_nmse"),
                "best_active_mean_count": summary.get("best_active_mean_count"),
                "overrides": overrides,
            }
        except Exception as exc:
            error_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            trial_dir.mkdir(parents=True, exist_ok=True)
            save_json(
                trial_dir / "summary.json",
                {
                    "trial_id": trial_id,
                    "status": "failed",
                    "trial_dir": str(trial_dir),
                    "overrides": overrides,
                    "error": error_text,
                },
            )
            row = {
                "trial_id": trial_id,
                "status": "failed",
                "trial_dir": str(trial_dir),
                "val_nmse": None,
                "cosine": None,
                "mean_l0": None,
                "best_epoch": None,
                "best_val_nmse": None,
                "best_active_mean_count": None,
                "overrides": overrides,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(error_text)
        trial_summaries.append(row)
        save_json(root_dir / "grid_summary.json", trial_summaries)
        write_grid_summary_csv(root_dir / "grid_summary.csv", trial_summaries)
        trial_end_time = time.perf_counter()
        elapsed_time = trial_end_time - trial_start_time
        elapsed_str = str(timedelta(seconds=int(elapsed_time)))
        print(f"Trial {trial_id} completed in {elapsed_str}.")

    best = select_best_trial(trial_summaries, metric=base_config.grid.metric, mode=base_config.grid.mode)
    save_json(root_dir / "best_trial.json", best)
    print(f"\nBest trial: {best}")
    return trial_summaries, best


def main():
    config = build_default_config()
    if config.grid.enabled:
        return run_sae_grid_search(config)

    trial_dir = Path(config.output.root_dir) / "single_trial"
    return run_sae_trial(config, trial_dir=trial_dir, trial_id="single_trial")


if __name__ == "__main__":
    print("Starting SAE validation script...")
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed_time)))
    print(f"SAE validation script completed in {elapsed_str}.")
