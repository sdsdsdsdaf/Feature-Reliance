#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from timm.data import create_transform, resolve_model_data_config
import timm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reservoir_sae.descriptors.sae_adapter import VGGSAEDescriptor
from reservoir_sae.descriptors.stylevec_adapter import VGGStyleVecDescriptor
from reservoir_sae.simple_reservoir import CosinePrototypeReservoir
from reservoir_sae.tta import SpecialistBank, accuracy, configure_bn_only_tent, tent_step
from reservoir_sae.utils.hf_data import (
    filter_corruption_dataset,
    infer_columns,
    load_hf_split,
    make_loader,
)


class RandomDescriptor(torch.nn.Module):
    def __init__(self, dim: int = 128, seed: int = 0) -> None:
        super().__init__()
        self.dim = int(dim)
        self.seed = int(seed)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(torch.randn(1, self.dim, device=images.device), dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal compact ImageNet-C ReservoirTTA + SAE routing run.")
    parser.add_argument(
        "--config",
        default="reservoir_sae/experiments/configs/imagenet_c_minimal.yaml",
        help="Experiment yaml config.",
    )
    parser.add_argument("--routing", choices=["stylevec", "sae", "random"], default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--sae-checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--skip-source-calibration",
        action="store_true",
        help="Use a fixed threshold instead of clean ImageNet descriptor calibration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    apply_overrides(cfg, args)
    set_seed(int(cfg.get("seed", 0)))

    device = torch.device(resolve_device(cfg.get("device", "cuda")))
    output_dir = Path(cfg["output_dir"]) / str(cfg["routing"]["mode"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Routing: {cfg['routing']['mode']}")
    print(f"Output: {output_dir}")

    model = timm.create_model(cfg["model"]["name"], pretrained=bool(cfg["model"].get("pretrained", True)))
    model.to(device)
    data_config = resolve_model_data_config(model)
    transform = create_transform(**data_config, is_training=False)

    batch_size = int(cfg.get("test", {}).get("batch_size", 0) or cfg.get("TEST", {}).get("BATCH_SIZE", 0) or 64)
    if args.batch_size:
        batch_size = args.batch_size

    descriptor = build_descriptor(cfg, device)
    descriptor.eval().to(device)

    print("Loading compact ImageNet-C...")
    c_cfg = cfg["dataset"]
    imagenet_c = load_hf_split(
        dataset_name=c_cfg["imagenet_c_name"],
        config=c_cfg.get("imagenet_c_config"),
        split=c_cfg["imagenet_c_split"],
        cache_dir=c_cfg.get("cache_dir"),
        token=args.hf_token,
    )
    c_columns = infer_columns(imagenet_c)
    imagenet_c = filter_corruption_dataset(
        imagenet_c,
        c_columns,
        c_cfg.get("corruptions"),
        c_cfg.get("severities"),
        c_cfg.get("max_samples"),
    )
    test_loader = make_loader(imagenet_c, transform, c_columns, batch_size=batch_size, num_workers=args.num_workers)

    first_images, _, _ = next(iter(test_loader))
    first_images = first_images.to(device)
    with torch.no_grad():
        first_descriptor = descriptor(first_images)
    descriptor_dim = int(first_descriptor.shape[1])
    threshold = 0.25

    if not args.skip_source_calibration:
        print("Calibrating reservoir threshold from ImageNet validation descriptors...")
        source_descriptors = collect_source_descriptors(cfg, transform, descriptor, device, batch_size, args)
        if source_descriptors.shape[0] >= 2:
            dists = 1.0 - source_descriptors @ source_descriptors.T
            upper = torch.triu(dists, diagonal=1)
            valid = upper[upper > 0]
            if valid.numel() > 0:
                threshold = float(torch.quantile(valid, float(cfg["reservoir"]["threshold_quantile"])).item())
    print(f"Reservoir novelty threshold: {threshold:.4f}")

    reservoir = CosinePrototypeReservoir(
        descriptor_dim=descriptor_dim,
        max_models=int(cfg["reservoir"]["max_models"]),
        threshold=threshold,
        prototype_momentum=float(cfg["reservoir"]["prototype_momentum"]),
        device=device,
    )
    reservoir.initialize(first_descriptor)

    if bool(cfg["tta"]["enabled"]):
        params = configure_bn_only_tent(model)
        optimizer = torch.optim.SGD(
            params,
            lr=float(cfg["tta"]["lr"]),
            momentum=0.9,
            weight_decay=0.0,
        )
        specialists = SpecialistBank(params, optimizer)
    else:
        model.eval()
        optimizer = None
        specialists = None

    metrics_rows: list[dict[str, Any]] = []
    routing_rows: list[dict[str, Any]] = []
    correct_total = 0
    seen_total = 0

    print("Running minimal online stream...")
    for step, (images, labels, meta) in enumerate(test_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.no_grad():
            route_descriptor = descriptor(images)
        route = reservoir.route(route_descriptor)

        if specialists is not None and optimizer is not None:
            specialists.ensure(route.model_idx)
            specialists.load(route.model_idx)
            logits = tent_step(model, images, optimizer, steps=int(cfg["tta"]["steps"]))
            specialists.save(route.model_idx)
        else:
            with torch.no_grad():
                logits = model(images)

        correct, total = accuracy(logits, labels)
        correct_total += correct
        seen_total += total

        corruption = majority(meta["corruption"])
        severity = majority(meta["severity"])
        batch_acc = correct / total if total else float("nan")
        online_acc = correct_total / seen_total if seen_total else float("nan")
        metrics_rows.append(
            {
                "step": step,
                "corruption": corruption,
                "severity": severity,
                "batch_size": int(images.shape[0]),
                "correct": correct,
                "total": total,
                "batch_accuracy": batch_acc,
                "online_accuracy": online_acc,
            }
        )
        routing_rows.append(
            {
                "step": step,
                "corruption": corruption,
                "severity": severity,
                "model_idx": route.model_idx,
                "model_prob": route.model_prob,
                "new_cluster": route.new_cluster,
                "min_distance": route.min_distance,
                "num_models": route.num_models,
            }
        )
        if step % 10 == 0:
            print(
                f"step={step:04d} acc={batch_acc:.3f} online={online_acc:.3f} "
                f"model={route.model_idx} K={route.num_models} {corruption}/{severity}"
            )

    summary = {
        "routing": cfg["routing"]["mode"],
        "num_samples": int(seen_total),
        "online_accuracy": correct_total / seen_total if seen_total else None,
        "num_models": reservoir.num_models,
        "threshold": threshold,
        "dataset": c_cfg["imagenet_c_name"],
        "corruptions": c_cfg.get("corruptions"),
        "severities": c_cfg.get("severities"),
    }
    write_csv(output_dir / "metrics.csv", metrics_rows)
    write_csv(output_dir / "routing_log.csv", routing_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def build_descriptor(cfg: dict[str, Any], device: torch.device) -> torch.nn.Module:
    routing_cfg = cfg["routing"]
    mode = routing_cfg["mode"]
    if mode == "stylevec":
        return VGGStyleVecDescriptor(
            style_idx=routing_cfg.get("style_idx", [2, 5, 7]),
            style_format=routing_cfg.get("style_format", "LOGVAR"),
        )
    if mode == "sae":
        checkpoint = routing_cfg.get("sae_checkpoint")
        if not checkpoint:
            raise ValueError("routing.sae_checkpoint is required for --routing sae")
        return VGGSAEDescriptor(
            checkpoint_path=checkpoint,
            style_idx=routing_cfg.get("style_idx", [2, 5, 7]),
            pooling=routing_cfg.get("sae_pooling", "mean"),
            active_threshold=float(routing_cfg.get("sae_active_threshold", 0.0)),
        )
    if mode == "random":
        return RandomDescriptor(seed=int(cfg.get("seed", 0))).to(device)
    raise ValueError(f"Unknown routing mode: {mode}")


def collect_source_descriptors(
    cfg: dict[str, Any],
    transform,
    descriptor: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    args: argparse.Namespace,
) -> torch.Tensor:
    d_cfg = cfg["dataset"]
    source = load_hf_split(
        dataset_name=d_cfg["imagenet_name"],
        split=d_cfg["imagenet_split"],
        cache_dir=d_cfg.get("cache_dir"),
        token=args.hf_token,
    )
    columns = infer_columns(source)
    max_batches = int(cfg["reservoir"]["source_batches_for_threshold"])
    max_rows = max_batches * batch_size
    source = source.select(range(min(max_rows, len(source))))
    loader = make_loader(source, transform, columns, batch_size=batch_size, num_workers=args.num_workers)
    descriptors = []
    with torch.no_grad():
        for images, _labels, _meta in loader:
            descriptors.append(F.normalize(descriptor(images.to(device)), dim=1).cpu())
    return torch.cat(descriptors, dim=0)


def load_config(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    if args.routing:
        cfg["routing"]["mode"] = args.routing
    if args.max_samples is not None:
        cfg["dataset"]["max_samples"] = args.max_samples
    if args.cache_dir:
        cfg["dataset"]["cache_dir"] = args.cache_dir
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.sae_checkpoint:
        cfg["routing"]["sae_checkpoint"] = args.sae_checkpoint
    if args.device:
        cfg["device"] = args.device


def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def majority(values) -> Any:
    if not values:
        return ""
    counts: dict[Any, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
