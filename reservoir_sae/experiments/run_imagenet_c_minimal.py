#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

_import_start = time.perf_counter()
print("[start] import torch/yaml/timm", flush=True)
import torch
import torch.nn.functional as F
import yaml
from timm.data import create_transform, resolve_model_data_config
import timm
print(f"[done] import torch/yaml/timm ({time.perf_counter() - _import_start:.1f}s)", flush=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_import_start = time.perf_counter()
print("[start] import reservoir_sae modules", flush=True)
from reservoir_sae.descriptors.sae_adapter import VGGSAEDescriptor
from reservoir_sae.descriptors.stylevec_adapter import VGGStyleVecDescriptor
from reservoir_sae.descriptors.vit_sae_adapter import ViTSAEDescriptor
from reservoir_sae.simple_reservoir import CosinePrototypeReservoir
from reservoir_sae.tta import SpecialistBank, accuracy, configure_bn_only_tent, tent_step
from reservoir_sae.utils.progress import progress, stage
from reservoir_sae.utils.hf_data import (
    build_timm_label_mapping,
    build_reservoirtta_imagenet_c_dataset,
    discover_imagenet_c_domains,
    filter_corruption_dataset,
    infer_columns,
    load_hf_split,
    make_loader,
    make_torch_loader,
    resolve_imagenet_c_root,
    select_class_balanced_subset,
)
print(f"[done] import reservoir_sae modules ({time.perf_counter() - _import_start:.1f}s)", flush=True)


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
    parser.add_argument(
        "--routing",
        choices=["stylevec", "sae", "vgg_sae", "vit_sae", "classifier_vit_sae", "random"],
        default=None,
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--stream-mode", choices=["reservoirtta", "mixed_hf"], default=None)
    parser.add_argument("--imagenet-c-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use only datasets already present in the Hugging Face cache.",
    )
    parser.add_argument("--sae-checkpoint", default=None)
    parser.add_argument("--vit-sae-checkpoint", default=None)
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Execution device. Defaults to cuda and falls back to cpu only when CUDA is unavailable.",
    )
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--max-domains",
        type=int,
        default=None,
        help="Limit ReservoirTTA-style ImageNet-C domains for smoke tests.",
    )
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

    with stage(f"build classifier {cfg['model']['name']} (create timm model + load pretrained weights)"):
        model = timm.create_model(cfg["model"]["name"], pretrained=bool(cfg["model"].get("pretrained", True)))
        model.to(device)
        data_config = resolve_model_data_config(model)
        transform = create_transform(**data_config, is_training=False)

    batch_size = int(cfg.get("test", {}).get("batch_size", 0) or cfg.get("TEST", {}).get("BATCH_SIZE", 0) or 64)
    if args.batch_size:
        batch_size = args.batch_size

    with stage(f"build routing descriptor {cfg['routing']['mode']}"):
        descriptor = build_descriptor(cfg, model, device)
        descriptor.eval().to(device)

    c_cfg = cfg["dataset"]
    with stage("load ImageNet-C stream"):
        test_loader, stream_info, domain_segments = build_imagenet_c_loader(
            cfg=cfg,
            args=args,
            transform=transform,
            batch_size=batch_size,
        )
        print(
            f"Stream mode: {stream_info['stream_mode']}; rows={stream_info['num_samples']:,}; "
            f"domains={stream_info['num_domains']}; batch_size={batch_size}"
        )
        if domain_segments:
            print("ReservoirTTA domain sequence preview:")
            for segment in domain_segments[: min(8, len(domain_segments))]:
                print(
                    f"  domain={segment['domain_index']:02d} "
                    f"{segment['corruption']}/{segment['severity']} samples={segment['num_samples']:,} "
                    f"classes={segment.get('num_classes', 'n/a')} "
                    f"per_class={segment.get('selected_min_per_class', 'n/a')}-"
                    f"{segment.get('selected_max_per_class', 'n/a')} "
                    f"label_mapping={segment.get('label_mapping', 'n/a')}"
                )
            if len(domain_segments) > 8:
                print(f"  ... {len(domain_segments) - 8} more domains")

    with stage("initialize reservoir descriptor"):
        first_images, _, _ = next(iter(test_loader))
        first_images = first_images.to(device)
        with torch.no_grad():
            first_descriptor = descriptor(first_images)
        descriptor_dim = int(first_descriptor.shape[1])
        threshold = 0.25

    if not args.skip_source_calibration:
        with stage("calibrate reservoir threshold (load clean ImageNet + extract source descriptors)"):
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
        with stage("configure TENT specialists"):
            params = configure_bn_only_tent(model)
            print(f"Trainable TTA parameter tensors: {len(params)}")
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

    with stage("run online ImageNet-C stream"):
        for step, (images, labels, meta) in enumerate(
            progress(test_loader, desc="online ImageNet-C stream", total=len(test_loader), log_every=10)
        ):
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
            domain_index = majority(meta["domain_index"])
            batch_acc = correct / total if total else float("nan")
            online_acc = correct_total / seen_total if seen_total else float("nan")
            metrics_rows.append(
                {
                    "step": step,
                    "domain_index": domain_index,
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
                    "domain_index": domain_index,
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
                    f"model={route.model_idx} K={route.num_models} domain={domain_index} {corruption}/{severity}"
                )

    domain_rows = summarize_domains(metrics_rows)
    summary = {
        "routing": cfg["routing"]["mode"],
        "num_samples": int(seen_total),
        "online_accuracy": correct_total / seen_total if seen_total else None,
        "num_models": reservoir.num_models,
        "threshold": threshold,
        "stream_mode": stream_info["stream_mode"],
        "dataset": stream_info["dataset"],
        "num_domains": stream_info["num_domains"],
        "corruptions": c_cfg.get("corruptions"),
        "severities": c_cfg.get("severities"),
    }
    write_csv(output_dir / "metrics.csv", metrics_rows)
    write_csv(output_dir / "routing_log.csv", routing_rows)
    write_csv(output_dir / "domain_summary.csv", domain_rows)
    write_csv(output_dir / "domain_sequence.csv", domain_segments)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def build_descriptor(cfg: dict[str, Any], classifier_model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    routing_cfg = cfg["routing"]
    mode = routing_cfg["mode"]
    if mode == "stylevec":
        return VGGStyleVecDescriptor(
            style_idx=routing_cfg.get("style_idx", [2, 5, 7]),
            style_format=routing_cfg.get("style_format", "LOGVAR"),
        )
    if mode in {"sae", "vgg_sae"}:
        checkpoint = routing_cfg.get("sae_checkpoint")
        if not checkpoint:
            raise ValueError("routing.sae_checkpoint is required for --routing sae/vgg_sae")
        return VGGSAEDescriptor(
            checkpoint_path=checkpoint,
            style_idx=routing_cfg.get("style_idx", [2, 5, 7]),
            pooling=routing_cfg.get("sae_pooling", "mean"),
            active_threshold=float(routing_cfg.get("sae_active_threshold", 0.0)),
        )
    if mode in {"vit_sae", "classifier_vit_sae"}:
        checkpoint = routing_cfg.get("vit_sae_checkpoint") or routing_cfg.get("sae_checkpoint")
        if not checkpoint:
            raise ValueError("routing.vit_sae_checkpoint is required for --routing vit_sae/classifier_vit_sae")
        classifier_for_descriptor = classifier_model if mode == "classifier_vit_sae" else None
        return ViTSAEDescriptor(
            checkpoint_path=checkpoint,
            model_name=routing_cfg.get("vit_model_name"),
            target_block=routing_cfg.get("vit_target_block"),
            token_scope=routing_cfg.get("vit_token_scope"),
            pooling=routing_cfg.get("vit_sae_pooling", routing_cfg.get("sae_pooling", "frequency")),
            active_threshold=float(routing_cfg.get("vit_sae_active_threshold", routing_cfg.get("sae_active_threshold", 0.1))),
            encode_chunk_size=int(routing_cfg.get("vit_sae_encode_chunk_size", 16384)),
            classifier_model=classifier_for_descriptor,
        )
    if mode == "random":
        return RandomDescriptor(seed=int(cfg.get("seed", 0))).to(device)
    raise ValueError(f"Unknown routing mode: {mode}")


def build_imagenet_c_loader(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    transform,
    batch_size: int,
):
    c_cfg = cfg["dataset"]
    stream_mode = str(c_cfg.get("stream_mode", "reservoirtta")).lower()
    corruptions = c_cfg.get("corruptions")
    severities = c_cfg.get("severities")
    max_domains = c_cfg.get("max_domains")

    if stream_mode in {"reservoirtta", "reservoir_tta"}:
        root = c_cfg.get("imagenet_c_root")
        root_path = resolve_imagenet_c_root(root)
        hf_dataset = None
        hf_columns = None
        label_mapping = None
        if not root_path or not root_path.exists():
            print(
                f"Local ImageNet-C root not available: {root}. "
                "Trying HF dataset only if it exposes corruption/severity columns."
            )
            print(
                f"HF dataset={c_cfg['imagenet_c_name']} split={c_cfg['imagenet_c_split']} "
                f"config={c_cfg.get('imagenet_c_config')} cache_dir={c_cfg.get('cache_dir')} "
                f"offline={bool(c_cfg.get('offline', False))}"
            )
            hf_dataset = load_hf_split(
                dataset_name=c_cfg["imagenet_c_name"],
                config=c_cfg.get("imagenet_c_config"),
                split=c_cfg["imagenet_c_split"],
                cache_dir=c_cfg.get("cache_dir"),
                token=args.hf_token,
                offline=bool(c_cfg.get("offline", False)),
            )
            hf_columns = infer_columns(hf_dataset)
            label_mapping = build_timm_label_mapping(hf_dataset, hf_columns)
            print(f"HF columns={hf_dataset.column_names}")
            print(f"Label mapping: {label_mapping.source}; identity={label_mapping.is_identity}")

        examples_per_domain = c_cfg.get("num_examples_per_domain", 5000)
        if c_cfg.get("max_samples") is not None:
            examples_per_domain = c_cfg.get("max_samples")
        if root_path and root_path.exists():
            local_domains = discover_imagenet_c_domains(root_path, corruptions=corruptions, severities=severities)
            print(f"Local ImageNet-C domains available after filters: {len(local_domains)}")
        dataset, segments = build_reservoirtta_imagenet_c_dataset(
            transform=transform,
            root=root_path,
            hf_dataset=hf_dataset,
            hf_columns=hf_columns,
            label_mapping=label_mapping,
            corruptions=corruptions,
            severities=severities,
            examples_per_domain=examples_per_domain,
            seed=int(cfg.get("seed", 0)),
            max_domains=max_domains,
        )
        loader = make_torch_loader(dataset, batch_size=batch_size, num_workers=args.num_workers)
        info = {
            "stream_mode": "reservoirtta",
            "dataset": str(root_path) if root_path and root_path.exists() else c_cfg.get("imagenet_c_name"),
            "num_samples": len(dataset),
            "num_domains": len(segments),
        }
        return loader, info, segments

    if stream_mode in {"mixed_hf", "hf_mixed", "mixed"}:
        print(
            f"HF dataset={c_cfg['imagenet_c_name']} split={c_cfg['imagenet_c_split']} "
            f"config={c_cfg.get('imagenet_c_config')} cache_dir={c_cfg.get('cache_dir')} "
            f"offline={bool(c_cfg.get('offline', False))}"
        )
        imagenet_c = load_hf_split(
            dataset_name=c_cfg["imagenet_c_name"],
            config=c_cfg.get("imagenet_c_config"),
            split=c_cfg["imagenet_c_split"],
            cache_dir=c_cfg.get("cache_dir"),
            token=args.hf_token,
            offline=bool(c_cfg.get("offline", False)),
        )
        c_columns = infer_columns(imagenet_c)
        label_mapping = build_timm_label_mapping(imagenet_c, c_columns)
        print(f"Label mapping: {label_mapping.source}; identity={label_mapping.is_identity}")
        if c_cfg.get("max_samples"):
            print(f"Applying class-balanced mixed-HF subset: max_samples={c_cfg.get('max_samples')}")
        imagenet_c = filter_corruption_dataset(
            imagenet_c,
            c_columns,
            c_cfg.get("corruptions"),
            c_cfg.get("severities"),
            c_cfg.get("max_samples"),
            seed=int(cfg.get("seed", 0)),
            label_mapping=label_mapping,
        )
        loader = make_loader(
            imagenet_c,
            transform,
            c_columns,
            batch_size=batch_size,
            num_workers=args.num_workers,
            label_mapping=label_mapping,
        )
        info = {
            "stream_mode": "mixed_hf",
            "dataset": c_cfg["imagenet_c_name"],
            "num_samples": len(imagenet_c),
            "num_domains": 0,
        }
        return loader, info, []

    raise ValueError(f"Unknown dataset.stream_mode={stream_mode!r}. Use 'reservoirtta' or 'mixed_hf'.")


def collect_source_descriptors(
    cfg: dict[str, Any],
    transform,
    descriptor: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    args: argparse.Namespace,
) -> torch.Tensor:
    d_cfg = cfg["dataset"]
    print(
        f"HF source dataset={d_cfg['imagenet_name']} split={d_cfg['imagenet_split']} "
        f"cache_dir={d_cfg.get('cache_dir')} offline={bool(d_cfg.get('offline', False))}"
    )
    source = load_hf_split(
        dataset_name=d_cfg["imagenet_name"],
        split=d_cfg["imagenet_split"],
        cache_dir=d_cfg.get("cache_dir"),
        token=args.hf_token,
        offline=bool(d_cfg.get("offline", False)),
    )
    columns = infer_columns(source)
    label_mapping = build_timm_label_mapping(source, columns)
    print(f"Source label mapping: {label_mapping.source}; identity={label_mapping.is_identity}")
    max_batches = int(cfg["reservoir"]["source_batches_for_threshold"])
    max_rows = max_batches * batch_size
    print(f"Applying class-balanced source calibration subset: max_rows={max_rows}")
    source = select_class_balanced_subset(
        source,
        columns,
        max_rows,
        seed=int(cfg.get("seed", 0)),
        label_mapping=label_mapping,
    )
    loader = make_loader(source, transform, columns, batch_size=batch_size, num_workers=args.num_workers, label_mapping=label_mapping)
    descriptors = []
    with torch.no_grad():
        for images, _labels, _meta in progress(
            loader, desc="source descriptor calibration", total=len(loader), log_every=10
        ):
            descriptors.append(F.normalize(descriptor(images.to(device)), dim=1).cpu())
    return torch.cat(descriptors, dim=0)


def summarize_domains(metrics_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for row in metrics_rows:
        key = (row.get("domain_index"), row.get("corruption"), row.get("severity"))
        item = grouped.setdefault(
            key,
            {
                "domain_index": row.get("domain_index"),
                "corruption": row.get("corruption"),
                "severity": row.get("severity"),
                "num_batches": 0,
                "correct": 0,
                "total": 0,
            },
        )
        item["num_batches"] += 1
        item["correct"] += int(row.get("correct", 0))
        item["total"] += int(row.get("total", 0))
    out = []
    for item in sorted(grouped.values(), key=lambda x: (int(x["domain_index"]), str(x["corruption"]), int(x["severity"]))):
        total = int(item["total"])
        item["accuracy"] = float(item["correct"] / total) if total else None
        out.append(item)
    return out


def load_config(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    if args.routing:
        cfg["routing"]["mode"] = args.routing
    if args.max_samples is not None:
        cfg["dataset"]["max_samples"] = args.max_samples
    if args.max_domains is not None:
        cfg["dataset"]["max_domains"] = args.max_domains
    if args.cache_dir:
        cfg["dataset"]["cache_dir"] = args.cache_dir
    if args.stream_mode:
        cfg["dataset"]["stream_mode"] = args.stream_mode
    if args.imagenet_c_root:
        cfg["dataset"]["imagenet_c_root"] = args.imagenet_c_root
    if args.offline:
        cfg["dataset"]["offline"] = True
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.sae_checkpoint:
        cfg["routing"]["sae_checkpoint"] = args.sae_checkpoint
        if cfg["routing"].get("mode") in {"vit_sae", "classifier_vit_sae"}:
            cfg["routing"]["vit_sae_checkpoint"] = args.sae_checkpoint
    if args.vit_sae_checkpoint:
        cfg["routing"]["vit_sae_checkpoint"] = args.vit_sae_checkpoint
    if args.device:
        cfg["device"] = args.device
    if args.model_name:
        cfg["model"]["name"] = args.model_name


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
