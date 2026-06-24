import copy
import csv
from datetime import timedelta
import itertools
import traceback
from pathlib import Path

import time
import torch
import timm
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader
from torchvision.datasets import Imagenette

from Utils.Config import SAEExperimentConfig
from Utils.SAE_plot_utils import save_trial_plots
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
EARLY_STOPPING_VERBOSE = True

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


def select_best_trial(summaries, metric, mode):
    completed = [row for row in summaries if row.get("status") == "completed" and row.get(metric) is not None]
    if not completed:
        return None
    reverse = mode == "max"
    return sorted(completed, key=lambda row: row[metric], reverse=reverse)[0]


def run_sae_grid_search(base_config):
    base_config = copy.deepcopy(base_config).validate()
    root_dir = Path(base_config.output.root_dir) / base_config.output.grid_dir_name
    root_dir.mkdir(parents=True, exist_ok=True)

    trial_summaries = []
    grid_items = list(iter_grid_items(base_config.grid.space))
    if base_config.grid.max_trials is not None:
        grid_items = grid_items[: int(base_config.grid.max_trials)]

    total_trials = len(grid_items)
    for trial_idx, overrides in enumerate(grid_items):
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
