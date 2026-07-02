#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

_import_start = time.perf_counter()
print("[start] import torch/timm/torchvision", flush=True)
import torch
import torch.nn.functional as F
from timm.data import create_transform, resolve_model_data_config
import timm
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.models import VGG19_Weights, vgg19
print(f"[done] import torch/timm/torchvision ({time.perf_counter() - _import_start:.1f}s)", flush=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_import_start = time.perf_counter()
print("[start] import SAE/project modules", flush=True)
from Model.SAE import VanillaL1SAE
from reservoir_sae.utils.progress import progress, stage
from reservoir_sae.utils.hf_data import (
    build_timm_label_mapping,
    infer_columns,
    load_hf_split,
    make_loader,
    select_class_balanced_subset,
)
print(f"[done] import SAE/project modules ({time.perf_counter() - _import_start:.1f}s)", flush=True)


class VGGFeaturePooler(torch.nn.Module):
    def __init__(self, style_idx=(2, 5, 7)) -> None:
        super().__init__()
        self.style_idx = [int(idx) for idx in style_idx]
        max_idx = max(self.style_idx)
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[: max_idx + 1].eval()
        vgg.requires_grad_(False)
        layers = []
        for layer in vgg.children():
            if isinstance(layer, torch.nn.ReLU):
                layer = torch.nn.ReLU(inplace=False)
            layers.append(layer)
        self.vgg = torch.nn.Sequential(*layers)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.features = {}
        for idx in self.style_idx:
            self.vgg[idx].register_forward_hook(self._make_hook(idx))

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.features.clear()
        self.vgg(self.normalize(images))
        pooled = [self.features[idx].mean(dim=(2, 3)) for idx in self.style_idx]
        return torch.cat(pooled, dim=1)

    def _make_hook(self, idx: int):
        def hook(_module, _args, output):
            self.features[idx] = output.detach()

        return hook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small VGG-feature SAE for SAE reservoir routing.")
    parser.add_argument("--dataset", default="ILSVRC/imagenet-1k")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--cache-dir", default="data/hf_cache")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use only datasets already present in the Hugging Face cache.",
    )
    parser.add_argument("--output", default="outputs/reservoir_sae/vgg_sae.pt")
    parser.add_argument("--max-samples", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--sae-batch-size", type=int, default=512)
    parser.add_argument("--expansion", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l1-reg", type=float, default=1e-4)
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Execution device. Defaults to cuda and falls back to cpu only when CUDA is unavailable.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(resolve_device(args.device))

    print(f"Device: {device}")
    print(f"Output: {args.output}")

    with stage("build preprocessing transform (create timm resnet50 + resolve transform)"):
        model_for_transform = timm.create_model("resnet50", pretrained=True)
        transform = create_transform(**resolve_model_data_config(model_for_transform), is_training=False)

    with stage("load HF dataset (cache metadata + Arrow shards)"):
        print(
            f"HF dataset={args.dataset} split={args.split} "
            f"cache_dir={args.cache_dir} offline={args.offline}"
        )
        dataset = load_hf_split(
            args.dataset,
            args.split,
            cache_dir=args.cache_dir,
            token=args.hf_token,
            offline=args.offline,
        )
        columns = infer_columns(dataset)
        label_mapping = build_timm_label_mapping(dataset, columns)
        print(f"Label mapping: {label_mapping.source}; identity={label_mapping.is_identity}")
        print(f"Applying class-balanced train subset: max_samples={args.max_samples}")
        dataset = select_class_balanced_subset(
            dataset,
            columns,
            args.max_samples,
            seed=args.seed,
            label_mapping=label_mapping,
        )
        print(f"Rows: {len(dataset):,}; columns={dataset.column_names}")
        loader = make_loader(
            dataset,
            transform,
            columns,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            label_mapping=label_mapping,
        )

    pooler = VGGFeaturePooler().to(device).eval()
    features = []
    with stage("collect VGG pooled features"):
        with torch.no_grad():
            for step, (images, _labels, _meta) in enumerate(
                progress(loader, desc="VGG feature collection", total=len(loader), log_every=20)
            ):
                feat = pooler(images.to(device, non_blocking=True)).cpu()
                features.append(feat)
                if step % 20 == 0:
                    print(f"batch={step} features={sum(x.shape[0] for x in features)}")
    features_tensor = torch.cat(features, dim=0).float()
    feature_mean = features_tensor.mean(dim=0)
    feature_std = features_tensor.std(dim=0).clamp_min(1e-6)
    normalized = (features_tensor - feature_mean) / feature_std

    input_dim = int(normalized.shape[1])
    hidden_dim = input_dim * int(args.expansion)
    sae = VanillaL1SAE(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(sae.parameters(), lr=args.lr)
    train_loader = DataLoader(
        TensorDataset(normalized),
        batch_size=args.sae_batch_size,
        shuffle=True,
        num_workers=0,
    )

    with stage(f"train VGG-SAE input_dim={input_dim} hidden_dim={hidden_dim} samples={len(normalized)}"):
        for epoch in range(args.epochs):
            total_loss = 0.0
            total_recon = 0.0
            total_l1 = 0.0
            count = 0
            for (batch,) in progress(
                train_loader,
                desc=f"VGG-SAE epoch {epoch + 1}/{args.epochs}",
                total=len(train_loader),
                log_every=50,
            ):
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                recon, z = sae(batch)
                recon_loss = F.mse_loss(recon, batch)
                l1_loss = z.abs().mean()
                loss = recon_loss + args.l1_reg * l1_loss
                loss.backward()
                sae.remove_gradient_parallel_to_decoder_directions()
                optimizer.step()
                sae.set_decoder_norm_to_unit_norm()
                bsz = batch.shape[0]
                total_loss += float(loss.item()) * bsz
                total_recon += float(recon_loss.item()) * bsz
                total_l1 += float(l1_loss.item()) * bsz
                count += bsz
            print(
                f"epoch={epoch + 1:03d} loss={total_loss / count:.6f} "
                f"recon={total_recon / count:.6f} l1={total_l1 / count:.6f}"
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with stage("save VGG-SAE checkpoint"):
        torch.save(
            {
                "sae_state_dict": sae.cpu().state_dict(),
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "style_idx": [2, 5, 7],
                "source_dataset": args.dataset,
                "source_split": args.split,
                "max_samples": int(args.max_samples),
            },
            output,
        )
    meta_path = output.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "checkpoint": str(output),
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "num_samples": int(normalized.shape[0]),
                "epochs": int(args.epochs),
                "expansion": int(args.expansion),
                "l1_reg": float(args.l1_reg),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved SAE checkpoint: {output}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        print("Requested device=cuda but CUDA is unavailable; falling back to cpu.")
        return "cpu"
    return requested


if __name__ == "__main__":
    main()
