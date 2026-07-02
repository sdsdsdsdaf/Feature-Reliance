#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

_import_start = time.perf_counter()
print("[start] import timm/torch/PIL", flush=True)
import timm
import torch
from PIL import Image
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader
print(f"[done] import timm/torch/PIL ({time.perf_counter() - _import_start:.1f}s)", flush=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_import_start = time.perf_counter()
print("[start] import SAE/project modules", flush=True)
from Model.SAE import VanillaL1SAE
from Utils.Config import SAEExperimentConfig
from Utils.early_stopping import unwrap_compiled_model
from Utils.SAE_utils import (
    collect_tokens_with_hook,
    compute_b_dec_init_streaming,
    evaluate_sae_tokens,
    fit_token_normalizer_streaming,
    get_torch_dtype,
    jsonable,
    normalize_tokens_inplace,
    save_json,
    train_sae_auto,
)
from reservoir_sae.utils.hf_data import (
    build_timm_label_mapping,
    infer_columns,
    load_hf_split,
    select_class_balanced_subset,
)
from reservoir_sae.utils.progress import stage
print(f"[done] import SAE/project modules ({time.perf_counter() - _import_start:.1f}s)", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/package a ViT-B token SAE descriptor using the existing SAE_validation pipeline."
    )
    parser.add_argument("--dataset", default="ILSVRC/imagenet-1k")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--val-split", default=None)
    parser.add_argument("--cache-dir", default="data/hf_cache")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--model-name", default="vit_base_patch16_224")
    parser.add_argument("--target-block", type=int, default=10)
    parser.add_argument("--token-scope", choices=["cls", "patch", "all", "clspatch"], default="patch")
    parser.add_argument("--output", default="outputs/reservoir_sae/vit_b_sae.pt")
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional existing SAE_validation best_sae_state.pt to package with HF token stats. Use with --epochs 0.",
    )
    parser.add_argument("--max-samples", type=int, default=8192)
    parser.add_argument("--max-val-samples", type=int, default=2048)
    parser.add_argument("--max-train-tokens", type=int, default=1_000_000)
    parser.add_argument("--max-val-tokens", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Execution device. Defaults to cuda and falls back to cpu only when CUDA is unavailable.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--run-validation-eval",
        action="store_true",
        help="When packaging --init-checkpoint with --epochs 0, also collect validation tokens and report SAE metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    config = build_sae_config(args)
    torch.set_float32_matmul_precision(config.sae.matmul_precision)

    print(f"Device: {config.extraction_config.device}")
    print(f"Output: {args.output}")
    print(
        f"SAE config: expansion={config.sae.expansion}, dec_bias_mode={config.sae.dec_bias_mode}, "
        f"active_threshold={config.sae.active_threshold}, l1_reg={config.sae.l1_reg}, "
        f"batch_size={config.sae.batch_size}, epochs={config.optim_config.epochs}"
    )

    with stage(f"build ViT model {args.model_name} (create timm model + load pretrained weights)"):
        model = timm.create_model(args.model_name, pretrained=True)
        model.eval().to(config.extraction_config.device)
        transform = create_transform(**resolve_model_data_config(model), is_training=False)

    with stage("build HF train/validation loaders"):
        train_loader, val_loader = build_hf_pair_loaders(args, transform, config)
        print(f"Train batches={len(train_loader):,}; val batches={len(val_loader):,}")

    with stage("fit train-token normalizer"):
        token_stats, train_token_count = fit_token_normalizer_streaming(
            model,
            train_loader,
            max_tokens=config.token.max_train_tokens,
            target_block=config.hook.target_block,
            token_scope=config.hook.token_scope,
            device=config.extraction_config.device,
        )
        print(f"train tokens seen for normalizer: {train_token_count:,}")

    input_dim = int(token_stats["mean"].shape[1])
    if args.init_checkpoint:
        with stage(f"load existing SAE checkpoint {args.init_checkpoint}"):
            if int(args.epochs) != 0:
                raise ValueError("--init-checkpoint is only supported for packaging with --epochs 0.")
            checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
            state_dict = checkpoint.get("sae_state_dict", checkpoint.get("model_state_dict"))
            if state_dict is None:
                raise KeyError(f"No SAE state dict found in --init-checkpoint {args.init_checkpoint}")
            hidden_dim = int(state_dict["b_enc"].shape[0])
            b_dec_init = state_dict["b_dec"].detach().float().clone()
            sae = VanillaL1SAE(input_dim=input_dim, hidden_dim=hidden_dim, b_dec_init=b_dec_init, dec_bias_mode=config.sae.dec_bias_mode)
            sae.load_state_dict(state_dict)
            history = []
            print("Packaging existing SAE weights; skipping SAE training and b_dec re-initialization.")

        val_tokens = None
        metrics = None
        if args.run_validation_eval:
            with stage("collect validation tokens"):
                val_tokens, _val_labels = collect_tokens_with_hook(
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
                print(f"Validation tokens: {tuple(val_tokens.shape)}")
    else:
        with stage("fit b_dec init"):
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

        with stage("collect validation tokens"):
            val_tokens, _val_labels = collect_tokens_with_hook(
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
            print(f"Validation tokens: {tuple(val_tokens.shape)}")

        with stage("train SAE with existing pipeline"):
            hidden_dim = int(input_dim * config.sae.expansion)
            trial_dir = Path(args.output).with_suffix("")
            trial_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = trial_dir / "best_sae_state.pt"
            save_json(trial_dir / "config.json", config)
            sae, history = train_sae_auto(
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
            save_json(trial_dir / "history.json", history)

    base_sae = unwrap_compiled_model(sae).cpu().eval()
    if val_tokens is not None:
        with stage("evaluate SAE on validation tokens"):
            metrics = evaluate_sae_tokens(
                base_sae,
                val_tokens,
                batch_size=config.sae.batch_size,
                threshold=config.sae.active_threshold,
                device=config.extraction_config.device,
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with stage("save ViT-SAE routing checkpoint"):
        package = {
            "sae_state_dict": base_sae.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "token_mean": token_stats["mean"].cpu(),
            "token_std": token_stats["std"].cpu(),
            "b_dec_init": b_dec_init.cpu(),
            "model_name": args.model_name,
            "target_block": int(config.hook.target_block),
            "token_scope": config.hook.token_scope,
            "active_threshold": float(config.sae.active_threshold),
            "source_dataset": args.dataset,
            "source_split": args.split,
            "max_samples": int(args.max_samples),
            "max_train_tokens": int(config.token.max_train_tokens),
            "max_val_tokens": int(config.token.max_val_tokens) if config.token.max_val_tokens is not None else None,
            "init_checkpoint": args.init_checkpoint,
            "sae_config": jsonable(config.sae),
            "optim_config": jsonable(config.optim_config),
            "hook": jsonable(config.hook),
            "token": jsonable(config.token),
            "validation_metrics": metrics,
        }
        torch.save(package, output)
        output.with_suffix(".json").write_text(
            json.dumps(
                {
                    "checkpoint": str(output),
                    "input_dim": input_dim,
                    "hidden_dim": hidden_dim,
                    "num_val_tokens": int(val_tokens.shape[0]) if val_tokens is not None else 0,
                    "train_token_count": int(train_token_count),
                    "model_name": args.model_name,
                    "target_block": int(config.hook.target_block),
                    "token_scope": config.hook.token_scope,
                    "init_checkpoint": args.init_checkpoint,
                    "sae": jsonable(config.sae),
                    "optim_config": jsonable(config.optim_config),
                    "validation_metrics": metrics,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(f"Saved ViT-SAE routing checkpoint: {output}")


def build_sae_config(args: argparse.Namespace) -> SAEExperimentConfig:
    config = SAEExperimentConfig()
    config.extraction_config.device = resolve_device(args.device)
    config.data_config.batch_size = int(args.batch_size)
    config.data_config.num_workers = int(args.num_workers)
    config.data_config.pin_memory = str(config.extraction_config.device).startswith("cuda")
    config.hook.target_block = int(args.target_block)
    config.hook.token_scope = args.token_scope
    config.token.max_train_tokens = int(args.max_train_tokens)
    config.token.max_val_tokens = int(args.max_val_tokens) if args.max_val_tokens is not None else None
    config.token.cache_dtype = "float16"
    config.token.source_mode = "auto"
    config.token.cache_max_cpu_gib = 8.0
    config.token.cache_num_workers = 0

    config.optim_config.epochs = int(args.epochs)
    config.optim_config.lr = float(args.lr)
    config.optim_config.weight_decay = 0.0
    config.optim_config.use_amp = str(config.extraction_config.device).startswith("cuda")

    config.sae.expansion = 32
    config.sae.dec_bias_mode = "geom"
    config.sae.active_threshold = 0.2
    config.sae.l1_reg = 3e-5
    config.sae.batch_size = 7096
    config.sae.bias_init_geom_max_iter = 100
    config.sae.bias_init_geom_tol = 1e-5
    config.sae.model_compile = True
    config.sae.amp_dtype = "bfloat16"
    config.sae.check_finite = True
    config.sae.matmul_precision = "high"

    config.early_stopping.patience = 30
    config.early_stopping.eps = 5e-5
    config.early_stopping.metric_name = "val_nmse"
    config.early_stopping.save_verbose = True
    return config.validate()


def build_hf_pair_loaders(args: argparse.Namespace, transform, config: SAEExperimentConfig):
    print(
        f"HF train dataset={args.dataset} split={args.split} "
        f"cache_dir={args.cache_dir} offline={args.offline}"
    )
    train_dataset = load_hf_split(
        args.dataset,
        args.split,
        cache_dir=args.cache_dir,
        token=args.hf_token,
        offline=args.offline,
    )
    print(
        f"HF val dataset={args.dataset} split={args.val_split or args.split} "
        f"cache_dir={args.cache_dir} offline={args.offline}"
    )
    val_dataset = load_hf_split(
        args.dataset,
        args.val_split or args.split,
        cache_dir=args.cache_dir,
        token=args.hf_token,
        offline=args.offline,
    )
    train_columns = infer_columns(train_dataset)
    val_columns = infer_columns(val_dataset)
    train_label_mapping = build_timm_label_mapping(train_dataset, train_columns)
    val_label_mapping = build_timm_label_mapping(val_dataset, val_columns)
    print(f"Applying class-balanced train subset: max_samples={args.max_samples}")
    train_dataset = select_class_balanced_subset(
        train_dataset,
        train_columns,
        int(args.max_samples),
        seed=args.seed,
        label_mapping=train_label_mapping,
    )
    print(f"Applying class-balanced val subset: max_val_samples={args.max_val_samples}")
    val_dataset = select_class_balanced_subset(
        val_dataset,
        val_columns,
        int(args.max_val_samples),
        seed=args.seed + 1,
        label_mapping=val_label_mapping,
    )
    print(f"Train label mapping: {train_label_mapping.source}; identity={train_label_mapping.is_identity}")
    print(f"Val label mapping: {val_label_mapping.source}; identity={val_label_mapping.is_identity}")
    pin_memory = bool(config.data_config.pin_memory and str(config.extraction_config.device).startswith("cuda"))
    print(f"HF train rows={len(train_dataset):,}; val rows={len(val_dataset):,}")
    return (
        DataLoader(
            HFPairDataset(train_dataset, transform, train_columns, train_label_mapping),
            batch_size=config.data_config.batch_size,
            shuffle=True,
            num_workers=config.data_config.num_workers,
            pin_memory=pin_memory,
        ),
        DataLoader(
            HFPairDataset(val_dataset, transform, val_columns, val_label_mapping),
            batch_size=config.data_config.batch_size,
            shuffle=False,
            num_workers=config.data_config.num_workers,
            pin_memory=pin_memory,
        ),
    )


class HFPairDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform, columns, label_mapping) -> None:
        self.dataset = dataset
        self.transform = transform
        self.columns = columns
        self.label_mapping = label_mapping

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        row = self.dataset[int(idx)]
        image = row[self.columns.image]
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")
        raw_label = int(row[self.columns.label]) if self.columns.label else -1
        label = self.label_mapping.map(raw_label) if raw_label >= 0 else raw_label
        return self.transform(image), label


def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        print("Requested device=cuda but CUDA is unavailable; falling back to cpu.")
        return "cpu"
    return requested


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
