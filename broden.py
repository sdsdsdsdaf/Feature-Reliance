#!/usr/bin/env python3
import argparse
import gc
import json
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

import timm
from timm.data import resolve_model_data_config

from Model.SAE import VanillaL1SAE
from Utils.broden_utils import (
    BRODEN_CATEGORIES,
    BrodenConcept,
    concept_mask_for_sample,
    discover_broden,
    make_image_preprocessor,
    markdown_table,
    parse_float_list,
    parse_category_subsets,
    parse_int_list,
    parse_str_list,
    plot_alignment_overlay,
    plot_broden_preview,
    sample_preview_records,
    save_json,
    select_category_subsets,
    write_csv,
)

DEFAULT_BRODEN_ROOT = "data/broden1_227"
DEFAULT_SAE_CHECKPOINT = "outputs/reservoir_sae/vit_b_sae.pt"
DEFAULT_OUTPUT_DIR = "Cache/sae_broden"
DEFAULT_NUM_ACTIVATION_SHARDS = 32


def parse_args():
    parser = argparse.ArgumentParser(description="Align SAE patch latents to Broden concepts with IoU and overlay plots.")
    parser.add_argument(
        "--broden-root",
        default=DEFAULT_BRODEN_ROOT,
        help=f"Broden dataset root containing index.csv and c_*.csv metadata. Defaults to {DEFAULT_BRODEN_ROOT}.",
    )
    parser.add_argument(
        "--sae-checkpoint",
        default=DEFAULT_SAE_CHECKPOINT,
        help=(
            "SAE checkpoint path or trial directory containing best_sae_state.pt. "
            f"Defaults to {DEFAULT_SAE_CHECKPOINT}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for markdown/csv/plot outputs. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument("--model-checkpoint", default=None, help="Optional backbone state_dict checkpoint.")
    parser.add_argument("--model-name", default="vit_base_patch16_224", help="timm ViT model name.")
    parser.add_argument("--hook-layer", type=int, default=None, help="ViT block index. Defaults to checkpoint target_block or 10.")
    parser.add_argument("--token-scope", default=None, help="Token scope. Overlay requires patch/all. Defaults to checkpoint token_scope or patch.")
    parser.add_argument("--latent-ids", default="all", help="Comma-separated latent ids or all.")
    parser.add_argument("--categories", default=",".join(BRODEN_CATEGORIES), help="Comma-separated Broden categories.")
    parser.add_argument("--subset", type=int, default=None, help="Optional image cap applied independently to every enabled category, e.g. --subset=16.")
    parser.add_argument("--category-subsets", default=None, help="Optional per-category image caps, e.g. object=32,part=16. These override --subset for specified categories.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional image cap for quick runs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Broden image batch size for activation collection.")
    parser.add_argument("--activation-cache-dir", default=None, help="Directory for cached activation shards. Defaults to output-dir/activation_shards.")
    parser.add_argument("--num-activation-shards", type=int, default=DEFAULT_NUM_ACTIVATION_SHARDS, help=f"Number of activation cache shards. Defaults to {DEFAULT_NUM_ACTIVATION_SHARDS}.")
    parser.add_argument("--rebuild-activation-cache", action="store_true", help="Recompute activation shards even when a cache exists.")
    parser.add_argument("--activation-cache-dtype", choices=("float16", "float32"), default="float16", help="Dtype used for cached activation shards.")
    parser.add_argument("--threshold-latent-chunk-size", type=int, default=256, help="Number of SAE latents to encode at once when computing percentile thresholds from cached ViT activations.")
    parser.add_argument("--activation-percentile", type=float, default=99.0, help="Latent activation percentile threshold.")
    parser.add_argument("--sensitivity-percentiles", default="95,98,99", help="Comma-separated percentile list for sensitivity checks.")
    parser.add_argument("--min-active-pixels", type=int, default=1, help="Minimum active patch count for diagnostics.")
    parser.add_argument("--overlay-cmap", default="magma", help="Activation overlay colormap.")
    parser.add_argument("--overlay-alpha", type=float, default=0.45, help="Activation overlay alpha.")
    parser.add_argument("--topk-overlays", type=int, default=3, help="Overlay plots per category.")
    parser.add_argument("--preview-per-category", type=int, default=1, help="Broden data preview plots per category.")
    parser.add_argument("--grid-size", type=int, default=14, help="Patch grid size. ViT-B/16 at 224 uses 14.")
    parser.add_argument("--seed", type=int, default=0, help="Tie-break and preview seed.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device.")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True, help="Load timm pretrained weights.")
    parser.add_argument("--dry-run", action="store_true", help="Only load Broden metadata and write previews/config.")
    return parser.parse_args()


def load_sae_checkpoint(path, device):
    path = Path(path)
    if path.is_dir():
        checkpoint_path = path / "best_sae_state.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Expected best_sae_state.pt inside SAE checkpoint directory: {path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config_path = path / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            checkpoint.setdefault("target_block", config.get("hook", {}).get("target_block"))
            checkpoint.setdefault("token_scope", config.get("hook", {}).get("token_scope"))
    else:
        checkpoint = torch.load(path, map_location="cpu")

    state_dict = checkpoint.get("sae_state_dict", checkpoint.get("model_state_dict"))
    if state_dict is None and all(k in checkpoint for k in ("W_enc", "b_enc", "W_dec", "b_dec")):
        state_dict = checkpoint
    if state_dict is None:
        raise KeyError("SAE checkpoint must contain sae_state_dict, model_state_dict, or raw SAE weights.")

    input_dim = int(checkpoint.get("input_dim", state_dict["W_enc"].shape[0]))
    hidden_dim = int(checkpoint.get("hidden_dim", state_dict["b_enc"].shape[0]))
    b_dec_init = state_dict.get("b_dec", torch.zeros(input_dim)).detach().float().clone()
    sae = VanillaL1SAE(input_dim=input_dim, hidden_dim=hidden_dim, b_dec_init=b_dec_init)
    sae.load_state_dict(state_dict)
    sae.eval().to(device)

    if "token_stats" in checkpoint:
        token_stats = checkpoint["token_stats"]
    elif "token_mean" in checkpoint and "token_std" in checkpoint:
        token_stats = {"mean": checkpoint["token_mean"], "std": checkpoint["token_std"]}
    elif "mean" in checkpoint and "std" in checkpoint:
        token_stats = {"mean": checkpoint["mean"], "std": checkpoint["std"]}
    else:
        token_stats = {"mean": torch.zeros(input_dim), "std": torch.ones(input_dim)}
        print("WARNING: token normalization stats not found in checkpoint; using zero mean and unit std.")
    token_stats = {"mean": token_stats["mean"].detach().float(), "std": token_stats["std"].detach().float().clamp_min(1e-6)}
    meta = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "target_block": checkpoint.get("target_block"),
        "token_scope": checkpoint.get("token_scope"),
    }
    return sae, token_stats, meta


def load_model(args, device):
    model = timm.create_model(args.model_name, pretrained=bool(args.pretrained))
    if args.model_checkpoint:
        checkpoint = torch.load(args.model_checkpoint, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        model.load_state_dict(state_dict, strict=False)
    if not hasattr(model, "blocks"):
        raise ValueError("broden.py currently requires a timm ViT-style model with .blocks for patch overlays.")
    model.eval().to(device)
    return model


def batches(items, batch_size):
    for start in range(0, len(items), int(batch_size)):
        yield start, items[start:start + int(batch_size)]



def _activation_cache_dir(args):
    return Path(args.activation_cache_dir) if args.activation_cache_dir else Path(args.output_dir) / "activation_shards"


def _cache_dtype(name):
    return torch.float16 if str(name).lower() == "float16" else torch.float32


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def activation_shard_paths(cache_dir):
    return sorted(Path(cache_dir).glob("shard_*.pt"))


def load_activation_cache_meta(cache_dir):
    meta_path = Path(cache_dir) / "metadata.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def activation_cache_matches(cache_dir, args, target_block, token_scope, samples):
    meta = load_activation_cache_meta(cache_dir)
    paths = activation_shard_paths(cache_dir)
    if meta is None or not paths:
        return False
    return (
        int(meta.get("cache_version", -1)) == 4
        and str(meta.get("cache_kind")) == "vit_block_activations"
        and int(meta.get("target_block", -1)) == int(target_block)
        and str(meta.get("token_scope")) == str(token_scope)
        and str(meta.get("model_checkpoint")) == str(args.model_checkpoint)
        and str(meta.get("model_name")) == str(args.model_name)
        and bool(meta.get("pretrained")) == bool(args.pretrained)
        and int(meta.get("grid_size", -1)) == int(args.grid_size)
        and int(meta.get("num_samples", -1)) == len(samples)
        and int(meta.get("requested_num_shards", -1)) == int(args.num_activation_shards)
        and int(meta.get("batch_size", -1)) == int(args.batch_size)
        and str(meta.get("dtype")) == str(args.activation_cache_dtype)
        and [int(x) for x in meta.get("sample_ids", [])] == [int(s.sample_id) for s in samples]
    )


def collect_activations_to_shards(samples, model, preprocess_pil, args, target_block, token_scope):
    cache_dir = _activation_cache_dir(args)
    if activation_cache_matches(cache_dir, args, target_block, token_scope, samples) and not args.rebuild_activation_cache:
        print(f"Using cached ViT block activation shards: {cache_dir}")
        return activation_shard_paths(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "metadata.json").unlink(missing_ok=True)
    for old_path in activation_shard_paths(cache_dir):
        old_path.unlink()

    if int(args.num_activation_shards) < 1:
        raise ValueError("--num-activation-shards must be at least 1.")
    if int(args.batch_size) < 1:
        raise ValueError("--batch-size must be at least 1.")

    cache_dtype = _cache_dtype(args.activation_cache_dtype)
    batch_size = int(args.batch_size)
    requested_num_shards = int(args.num_activation_shards)
    total_batches = (len(samples) + batch_size - 1) // batch_size
    max_batches_per_shard = (total_batches + requested_num_shards - 1) // requested_num_shards
    kept_samples = []
    kept_tensors = []
    shard_sample_ids = []
    shard_activations = []
    batches_in_shard = 0
    shard_paths = []
    shard_idx = 0
    print(
        f"Collecting ViT block-{target_block} activations for {len(samples):,} Broden images into at most "
        f"{requested_num_shards:,} shards ({max_batches_per_shard:,} batches per shard maximum)..."
    )

    def collect_batch():
        nonlocal kept_samples, kept_tensors, batches_in_shard
        sample_ids, x = _collect_activation_batch(
            kept_samples, kept_tensors, model, target_block, token_scope,
            args.device, cache_dtype,
        )
        shard_sample_ids.extend(sample_ids)
        shard_activations.append(x)
        batches_in_shard += 1
        kept_samples = []
        kept_tensors = []

    def flush_shard():
        nonlocal shard_idx, shard_sample_ids, shard_activations, batches_in_shard
        shard_path = cache_dir / f"shard_{shard_idx:06d}.pt"
        shard_x = torch.cat(shard_activations, dim=0)
        torch.save({"sample_ids": shard_sample_ids, "x": shard_x}, shard_path)
        shard_paths.append(shard_path)
        shard_idx += 1
        shard_sample_ids = []
        shard_activations = []
        batches_in_shard = 0
        del shard_x
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (ImportError, OSError, AttributeError):
            pass

    for sample in tqdm(samples, desc="ViT activation shards", dynamic_ncols=True):
        try:
            with Image.open(sample.image_path) as image:
                kept_tensors.append(preprocess_pil(image))
            kept_samples.append(sample)
        except Exception as exc:
            print(f"WARNING: failed to preprocess {sample.image_path}: {exc}")
            continue

        if len(kept_tensors) >= batch_size:
            collect_batch()
            if batches_in_shard >= max_batches_per_shard:
                flush_shard()

    if kept_tensors:
        collect_batch()
    if shard_activations:
        flush_shard()

    if not shard_paths:
        raise RuntimeError("No activation shards were written. Check Broden image paths and preprocessing errors.")

    cached_count = 0
    for shard_path in shard_paths:
        cached_count += int(_torch_load(shard_path)["x"].shape[0])
    save_json(cache_dir / "metadata.json", {
        "cache_version": 4,
        "cache_kind": "vit_block_activations",
        "target_block": int(target_block),
        "token_scope": str(token_scope),
        "model_checkpoint": str(args.model_checkpoint),
        "model_name": str(args.model_name),
        "pretrained": bool(args.pretrained),
        "grid_size": int(args.grid_size),
        "num_samples": len(samples),
        "num_cached_samples": cached_count,
        "sample_ids": [int(s.sample_id) for s in samples],
        "batch_size": batch_size,
        "requested_num_shards": requested_num_shards,
        "total_batches": total_batches,
        "max_batches_per_shard": max_batches_per_shard,
        "dtype": str(args.activation_cache_dtype),
    })
    print(f"Cached {len(shard_paths):,} ViT block activation shards under {cache_dir}")
    return shard_paths


def _collect_activation_batch(batch_samples, image_tensors, model, target_block, token_scope, device, cache_dtype):
    batch = torch.stack(image_tensors, dim=0)
    captured = {}

    def hook(_module, _inputs, output):
        captured["x"] = (output[0] if isinstance(output, (tuple, list)) else output).detach()

    handle = model.blocks[target_block].register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = model(batch.to(device))
    finally:
        handle.remove()
    x = captured["x"]
    if token_scope in {"patch", "all", "clspatch"}:
        x = x[:, 1:, :]
    else:
        raise ValueError("Patch-grid activation maps require token_scope=patch or all.")
    return [int(sample.sample_id) for sample in batch_samples], x.to(device="cpu", dtype=cache_dtype).contiguous()


def _encode_cached_activations(x, sae, token_stats, latent_ids, token_chunk_size=32768):
    original_shape = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1])
    device = next(sae.parameters()).device
    ids = torch.as_tensor(latent_ids, dtype=torch.long, device=device)
    mean = token_stats["mean"].to(device=device, dtype=torch.float32)
    std = token_stats["std"].to(device=device, dtype=torch.float32)
    weights = sae.W_enc.index_select(1, ids)
    bias = sae.b_enc.index_select(0, ids)
    decoder_bias = sae.b_dec
    chunks = []
    with torch.no_grad():
        for start in range(0, flat.shape[0], int(token_chunk_size)):
            tokens = flat[start:start + int(token_chunk_size)].to(device=device, dtype=torch.float32)
            tokens = (tokens - mean) / std
            z = torch.relu((tokens - decoder_bias) @ weights + bias)
            chunks.append(z.cpu())
    return torch.cat(chunks, dim=0).reshape(*original_shape, len(latent_ids))


def iter_activation_records(shard_paths, sample_by_id, sae, token_stats, latent_ids, image_batch_size=8):
    for shard_path in shard_paths:
        shard = _torch_load(shard_path)
        sample_ids = shard["sample_ids"]
        x = shard["x"]
        for start in range(0, len(sample_ids), int(image_batch_size)):
            ids = sample_ids[start:start + int(image_batch_size)]
            z = _encode_cached_activations(x[start:start + len(ids)], sae, token_stats, latent_ids)
            for idx, sample_id in enumerate(ids):
                sample = sample_by_id.get(int(sample_id))
                if sample is not None:
                    yield {"sample": sample, "z": z[idx]}


def _latent_columns(latent_ids, latent_index):
    return [latent_index[int(latent_id)] for latent_id in latent_ids]


def compute_thresholds_from_shards(shard_paths, sae, token_stats, latent_ids, percentile, chunk_size):
    if not shard_paths:
        raise RuntimeError("No activation shards available for threshold calculation.")
    quantile = float(percentile) / 100.0
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"Percentile must be between 0 and 100, got {percentile}.")
    thresholds = {}
    chunk_size = max(1, int(chunk_size))
    for start in tqdm(range(0, len(latent_ids), chunk_size), desc="Activation thresholds", dynamic_ncols=True):
        chunk_latents = [int(x) for x in latent_ids[start:start + chunk_size]]
        shard_sizes = []
        for shard_path in shard_paths:
            x = _torch_load(shard_path)["x"]
            shard_sizes.append(int(x.numel() // x.shape[-1]))
            del x
        value_count = sum(shard_sizes)
        if value_count == 0:
            raise RuntimeError("Activation shards contain no values for threshold calculation.")

        fd, temp_path = tempfile.mkstemp(prefix="broden_threshold_", suffix=".mmap")
        os.close(fd)
        values = None
        try:
            values = np.memmap(temp_path, mode="w+", dtype=np.float32, shape=(len(chunk_latents), value_count))
            offset = 0
            for shard_path, shard_size in zip(shard_paths, shard_sizes):
                x = _torch_load(shard_path)["x"]
                z = _encode_cached_activations(x, sae, token_stats, chunk_latents)
                chunk = z.reshape(-1, len(chunk_latents)).numpy()
                values[:, offset:offset + shard_size] = chunk.T
                offset += shard_size
                del x, z, chunk
            values.flush()

            rank = quantile * (value_count - 1)
            lower = int(np.floor(rank))
            upper = int(np.ceil(rank))
            weight = rank - lower
            for row, latent_id in enumerate(chunk_latents):
                values[row].partition((lower, upper))
                low_value = float(values[row, lower])
                high_value = float(values[row, upper])
                thresholds[int(latent_id)] = low_value + (high_value - low_value) * weight
        finally:
            if values is not None:
                values.flush()
                del values
            Path(temp_path).unlink(missing_ok=True)
    return thresholds


def compute_thresholds(records, latent_ids, percentile, latent_index=None):
    records = list(records)
    if not records:
        raise RuntimeError("No activation records available for threshold calculation.")
    latent_index = latent_index or {int(latent_id): int(latent_id) for latent_id in latent_ids}
    cols = _latent_columns(latent_ids, latent_index)
    z_flat = torch.cat([row["z"].reshape(-1, row["z"].shape[-1])[:, cols] for row in records], dim=0)
    q = torch.quantile(z_flat.float(), float(percentile) / 100.0, dim=0)
    return {int(latent_id): float(value) for latent_id, value in zip(latent_ids, q.tolist())}


def print_threshold_values(percentile, thresholds, stage):
    percentile = float(percentile)
    upper_percent = max(0.0, 100.0 - percentile)
    print(f"\n[{stage}] threshold experiment")
    print(f"  relative threshold: P{percentile:g} (top {upper_percent:g}%)")
    print(f"  actual thresholds ({len(thresholds):,} latents):")
    for latent_id in sorted(thresholds):
        print(f"    latent {int(latent_id):>6d}: {float(thresholds[latent_id]):.8g}")
    print(flush=True)


def compute_iou(records, concept_by_id, categories, latent_ids, thresholds, label_to_concept_grid, grid_size, latent_index=None, total_records=None, desc="Broden IoU"):

    accum = defaultdict(lambda: {"intersection": 0, "union": 0, "concept_images": 0, "active_patches": 0, "examples": []})
    concept_ids_by_category = {
        category: sorted(cid for cid, concept in concept_by_id.items() if concept.category == category)
        for category in categories
    }
    latent_index = latent_index or {int(latent_id): int(latent_id) for latent_id in latent_ids}
    for row in tqdm(records, total=total_records, desc=desc, unit="image", dynamic_ncols=True):
        sample = row["sample"]
        z = row["z"]
        active_by_latent = {}
        for latent_id in latent_ids:
            patch_map = z[:, latent_index[int(latent_id)]].reshape(grid_size, grid_size).float()
            active = patch_map >= float(thresholds[int(latent_id)])
            active_by_latent[int(latent_id)] = (active, patch_map)
        for category, concept_ids in concept_ids_by_category.items():
            for concept_id in concept_ids:
                mask = concept_mask_for_sample(sample, concept_id, category, label_to_concept_grid, grid_size)
                if mask is None:
                    continue
                for latent_id, (active, patch_map) in active_by_latent.items():
                    inter = int((active & mask).sum().item())
                    union = int((active | mask).sum().item())
                    key = (latent_id, concept_id)
                    item = accum[key]
                    item["intersection"] += inter
                    item["union"] += union
                    item["concept_images"] += 1
                    item["active_patches"] += int(active.sum().item())
                    if inter > 0:
                        item["examples"].append({
                            "sample": sample,
                            "category": category,
                            "concept_id": concept_id,
                            "latent_id": latent_id,
                            "intersection": inter,
                            "union": union,
                            "peak_activation": float(patch_map.max().item()),
                            "patch_map": patch_map.detach().cpu(),
                            "active": active.detach().cpu(),
                            "mask": mask.detach().cpu(),
                        })
    rows = []
    for (latent_id, concept_id), item in accum.items():
        concept = concept_by_id.get(concept_id, BrodenConcept(concept_id, str(concept_id), "unknown"))
        iou = item["intersection"] / item["union"] if item["union"] else 0.0
        rows.append({
            "latent_id": latent_id,
            "concept_id": concept_id,
            "concept": concept.name,
            "category": concept.category,
            "IoU": iou,
            "intersection": item["intersection"],
            "union": item["union"],
            "concept_images": item["concept_images"],
            "active_patches": item["active_patches"],
            "examples": sorted(item["examples"], key=lambda x: (x["intersection"], x["peak_activation"]), reverse=True),
        })
    rows.sort(key=lambda r: (r["latent_id"], -r["IoU"]))
    return rows



def category_mask_for_sample(sample, category, concept_ids, label_to_concept_grid, grid_size):
    masks = []
    for concept_id in concept_ids:
        mask = concept_mask_for_sample(sample, concept_id, category, label_to_concept_grid, grid_size)
        if mask is not None:
            masks.append(mask.bool())
    if not masks:
        return None
    merged = torch.zeros_like(masks[0], dtype=torch.bool)
    for mask in masks:
        merged |= mask
    return merged


def compute_category_iou(records, concept_by_id, categories, latent_ids, thresholds, label_to_concept_grid, grid_size, latent_index=None, total_records=None):

    accum = defaultdict(lambda: {"intersection": 0, "union": 0, "category_images": 0, "active_patches": 0})
    concept_ids_by_category = {
        category: sorted(cid for cid, concept in concept_by_id.items() if concept.category == category)
        for category in categories
    }
    latent_index = latent_index or {int(latent_id): int(latent_id) for latent_id in latent_ids}
    for row in tqdm(records, total=total_records, desc="Broden category IoU", unit="image", dynamic_ncols=True):
        sample = row["sample"]
        z = row["z"]
        active_by_latent = {}
        for latent_id in latent_ids:
            patch_map = z[:, latent_index[int(latent_id)]].reshape(grid_size, grid_size).float()
            active_by_latent[int(latent_id)] = patch_map >= float(thresholds[int(latent_id)])
        for category, concept_ids in concept_ids_by_category.items():
            category_mask = category_mask_for_sample(sample, category, concept_ids, label_to_concept_grid, grid_size)
            if category_mask is None:
                continue
            for latent_id, active in active_by_latent.items():
                inter = int((active & category_mask).sum().item())
                union = int((active | category_mask).sum().item())
                item = accum[(latent_id, category)]
                item["intersection"] += inter
                item["union"] += union
                item["category_images"] += 1
                item["active_patches"] += int(active.sum().item())
    rows = []
    for (latent_id, category), item in accum.items():
        iou = item["intersection"] / item["union"] if item["union"] else 0.0
        rows.append({
            "latent_id": latent_id,
            "category": category,
            "category_IoU": iou,
            "intersection": item["intersection"],
            "union": item["union"],
            "category_images": item["category_images"],
            "active_patches": item["active_patches"],
        })
    rows.sort(key=lambda r: (r["latent_id"], -r["category_IoU"]))
    return rows

def best_rows_by_latent(iou_rows, latent_ids):
    by_latent = {int(latent_id): [] for latent_id in latent_ids}
    for row in iou_rows:
        by_latent.setdefault(int(row["latent_id"]), []).append(row)
    best = []
    for latent_id in latent_ids:
        rows = sorted(by_latent.get(int(latent_id), []), key=lambda r: r["IoU"], reverse=True)
        if rows:
            row = rows[0]
            best.append({
                "SAE latent": int(latent_id),
                "best Broden concept": row["concept"],
                "category": row["category"],
                "IoU": f"{row['IoU']:.3f}",
            })
        else:
            best.append({"SAE latent": int(latent_id), "best Broden concept": "", "category": "", "IoU": "0.000"})
    return best


def write_outputs(output_dir, best_rows, iou_rows, category_iou_rows, thresholds, args):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    md = markdown_table(best_rows, ["SAE latent", "best Broden concept", "category", "IoU"], aligns={"SAE latent": "right", "IoU": "right"})
    (output_dir / "sae_broden_alignment.md").write_text(md, encoding="utf-8")
    write_csv(output_dir / "sae_broden_alignment.csv", best_rows, ["SAE latent", "best Broden concept", "category", "IoU"])
    long_rows = []
    for row in iou_rows:
        long_rows.append({k: row[k] for k in ["latent_id", "concept_id", "concept", "category", "IoU", "intersection", "union", "concept_images", "active_patches"]})
    write_csv(output_dir / "latent_concept_iou_long.csv", long_rows, ["latent_id", "concept_id", "concept", "category", "IoU", "intersection", "union", "concept_images", "active_patches"])
    category_rows = []
    for row in category_iou_rows:
        category_rows.append({
            "latent_id": row["latent_id"],
            "category": row["category"],
            "category_IoU": row["category_IoU"],
            "intersection": row["intersection"],
            "union": row["union"],
            "category_images": row["category_images"],
            "active_patches": row["active_patches"],
        })
    write_csv(output_dir / "latent_category_iou.csv", category_rows, ["latent_id", "category", "category_IoU", "intersection", "union", "category_images", "active_patches"])
    category_md_rows = [
        {"SAE latent": row["latent_id"], "category": row["category"], "category IoU": f"{row['category_IoU']:.3f}"}
        for row in category_iou_rows
    ]
    (output_dir / "latent_category_iou.md").write_text(
        markdown_table(category_md_rows, ["SAE latent", "category", "category IoU"], aligns={"SAE latent": "right", "category IoU": "right"}),
        encoding="utf-8",
    )
    write_csv(output_dir / "activation_thresholds.csv", [{"latent_id": k, "threshold": v} for k, v in sorted(thresholds.items())], ["latent_id", "threshold"])


def write_previews(output_dir, samples, concept_by_id, categories, preview_per_category, seed, image_to_np, label_to_concept_grid, grid_size):
    preview_dir = Path(output_dir) / "broden_data_preview"
    records = []
    for sample, concept in sample_preview_records(samples, concept_by_id, categories, preview_per_category, seed):
        mask = concept_mask_for_sample(sample, concept.concept_id, concept.category, label_to_concept_grid, grid_size)
        if mask is None:
            continue
        image_np = image_to_np(Image.open(sample.image_path))
        filename = f"{concept.category}_{concept.concept_id}_{sample.sample_id:06d}.png"
        path = plot_broden_preview(
            image_np,
            mask,
            title=f"{concept.category}: {concept.name} ({concept.concept_id})",
            output_path=preview_dir / filename,
        )
        records.append({"category": concept.category, "concept_id": concept.concept_id, "concept": concept.name, "sample_id": sample.sample_id, "path": str(path)})
    if not records:
        raise RuntimeError("Broden preview generation failed. Check image/mask paths and label mapping before IoU.")
    write_csv(preview_dir / "index.csv", records, ["category", "concept_id", "concept", "sample_id", "path"])
    return records


def write_overlays(output_dir, iou_rows, image_to_np, label_to_concept_grid, grid_size, args):
    overlay_dir = Path(output_dir) / "broden_category_overlays"
    index_rows = []
    by_category = defaultdict(list)
    for row in iou_rows:
        if row["examples"]:
            by_category[row["category"]].append(row)
    for category in parse_str_list(args.categories, BRODEN_CATEGORIES):
        rows = sorted(by_category.get(category, []), key=lambda r: r["IoU"], reverse=True)
        written = 0
        for row in rows:
            for example in row["examples"]:
                sample = example["sample"]
                image_np = image_to_np(Image.open(sample.image_path))
                concept = row["concept"]
                filename = f"{category}_latent_{row['latent_id']:05d}_concept_{row['concept_id']}_{written:02d}.png"
                title = f"latent={row['latent_id']} | {category}:{concept} | IoU={row['IoU']:.3f} | image={sample.sample_id}"
                path = plot_alignment_overlay(
                    image_np,
                    broden_mask=example["mask"],
                    activation_map=example["patch_map"].numpy(),
                    binary_activation=example["active"],
                    title=title,
                    output_path=overlay_dir / filename,
                    overlay_cmap=args.overlay_cmap,
                    overlay_alpha=args.overlay_alpha,
                )
                index_rows.append({
                    "category": category,
                    "latent_id": row["latent_id"],
                    "concept_id": row["concept_id"],
                    "concept": concept,
                    "IoU": f"{row['IoU']:.3f}",
                    "sample_id": sample.sample_id,
                    "path": str(path),
                })
                written += 1
                if written >= int(args.topk_overlays):
                    break
            if written >= int(args.topk_overlays):
                break
    write_csv(overlay_dir / "index.csv", index_rows, ["category", "latent_id", "concept_id", "concept", "IoU", "sample_id", "path"])
    md = markdown_table(index_rows, ["category", "latent_id", "concept", "IoU", "sample_id", "path"], aligns={"latent_id": "right", "IoU": "right", "sample_id": "right"})
    (overlay_dir / "index.md").write_text(md, encoding="utf-8")
    return index_rows


def write_threshold_sensitivity(output_dir, shard_paths, sample_by_id, sae, token_stats, concept_by_id, categories, latent_ids, latent_index, percentiles, label_to_concept_grid, grid_size, threshold_chunk_size):
    rows = []
    total_records = len(sample_by_id)
    total_experiments = len(percentiles)
    for experiment_idx, pct in enumerate(percentiles, start=1):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        thresholds = compute_thresholds_from_shards(shard_paths, sae, token_stats, latent_ids, pct, threshold_chunk_size)
        print_threshold_values(pct, thresholds, f"Sensitivity {experiment_idx}/{total_experiments}")
        iou_rows = compute_iou(
            iter_activation_records(shard_paths, sample_by_id, sae, token_stats, latent_ids),
            concept_by_id,
            categories,
            latent_ids,
            thresholds,
            label_to_concept_grid,
            grid_size,
            latent_index=latent_index,
            total_records=total_records,
            desc=f"Broden IoU P{float(pct):g} [{experiment_idx}/{total_experiments}]",
        )
        best = best_rows_by_latent(iou_rows, latent_ids)
        for row in best:
            rows.append({"percentile": pct, **row})
        del best, iou_rows, thresholds

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_csv(Path(output_dir) / "threshold_sensitivity.csv", rows, ["percentile", "SAE latent", "best Broden concept", "category", "IoU"])
    return rows

def main():
    args = parse_args()
    start = time.perf_counter()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    concept_by_id, samples = discover_broden(args.broden_root)
    categories = parse_str_list(args.categories, BRODEN_CATEGORIES)
    unknown_categories = sorted(set(categories) - set(BRODEN_CATEGORIES))
    if unknown_categories:
        raise ValueError(f"Unknown Broden categories: {unknown_categories}; choose from {list(BRODEN_CATEGORIES)}.")
    samples = [s for s in samples if any(a.category in categories for a in s.annotations)]
    if args.max_images is not None:
        samples = samples[: int(args.max_images)]
    if args.subset is not None and args.subset < 1:
        raise ValueError(f"--subset must be at least 1, got {args.subset}.")
    category_subsets = {category: args.subset for category in categories} if args.subset is not None else {}
    category_subsets.update(parse_category_subsets(args.category_subsets))
    inactive_subsets = sorted(set(category_subsets) - set(categories))
    if inactive_subsets:
        raise ValueError(f"--category-subsets contains categories not enabled by --categories: {inactive_subsets}.")
    samples, category_sample_counts = select_category_subsets(samples, categories, category_subsets, args.seed)
    if not samples:
        raise ValueError("No Broden samples remain after applying category and subset filters.")
    print(f"Broden concepts={len(concept_by_id):,}; usable samples={len(samples):,}; categories={categories}; category_samples={category_sample_counts}")

    sae, token_stats, sae_meta = load_sae_checkpoint(args.sae_checkpoint, args.device)
    hidden_dim = int(sae_meta["hidden_dim"])
    latent_ids = parse_int_list(args.latent_ids)
    if latent_ids is None:
        latent_ids = list(range(hidden_dim))
    latent_ids = [int(x) for x in latent_ids if 0 <= int(x) < hidden_dim]
    if not latent_ids:
        raise ValueError("No valid latent ids selected.")

    model = load_model(args, args.device)
    data_config = resolve_model_data_config(model)
    preprocess_pil, image_to_np, _mask_to_grid, label_to_concept_grid, preprocess_meta = make_image_preprocessor(data_config)
    target_block = int(args.hook_layer if args.hook_layer is not None else (sae_meta.get("target_block") if sae_meta.get("target_block") is not None else 10))
    token_scope = str(args.token_scope or sae_meta.get("token_scope") or "patch").lower()
    if token_scope == "cls":
        raise ValueError("Broden spatial overlay requires patch or all token scope, not cls.")

    preview_records = write_previews(
        output_dir,
        samples,
        concept_by_id,
        categories,
        args.preview_per_category,
        args.seed,
        image_to_np,
        label_to_concept_grid,
        args.grid_size,
    )

    run_config = vars(args).copy()
    run_config.update({
        "num_concepts": len(concept_by_id),
        "num_samples": len(samples),
        "category_sample_counts": category_sample_counts,
        "latent_ids": latent_ids,
        "target_block": target_block,
        "token_scope": token_scope,
        "sae_meta": sae_meta,
        "preprocess": preprocess_meta,
        "preview_records": preview_records,
    })
    if args.dry_run:
        run_config["status"] = "dry_run_completed"
        save_json(output_dir / "run_config.json", run_config)
        print(f"Dry run completed. Outputs written to {output_dir}")
        return

    latent_index = {int(latent_id): idx for idx, latent_id in enumerate(latent_ids)}
    sample_by_id = {int(sample.sample_id): sample for sample in samples}
    shard_paths = collect_activations_to_shards(
        samples, model, preprocess_pil, args, target_block, token_scope,
    )
    thresholds = compute_thresholds_from_shards(
        shard_paths,
        sae,
        token_stats,
        latent_ids,
        args.activation_percentile,
        args.threshold_latent_chunk_size,
    )
    print_threshold_values(args.activation_percentile, thresholds, "Main")
    iou_rows = compute_iou(
        iter_activation_records(shard_paths, sample_by_id, sae, token_stats, latent_ids),
        concept_by_id,
        categories,
        latent_ids,
        thresholds,
        label_to_concept_grid,
        args.grid_size,
        latent_index=latent_index,
        total_records=len(sample_by_id),
        desc=f"Broden IoU P{float(args.activation_percentile):g}",
    )
    category_iou_rows = compute_category_iou(
        iter_activation_records(shard_paths, sample_by_id, sae, token_stats, latent_ids),
        concept_by_id,
        categories,
        latent_ids,
        thresholds,
        label_to_concept_grid,
        args.grid_size,
        latent_index=latent_index,
        total_records=len(sample_by_id),
    )
    best_rows = best_rows_by_latent(iou_rows, latent_ids)
    write_outputs(output_dir, best_rows, iou_rows, category_iou_rows, thresholds, args)
    overlay_rows = write_overlays(output_dir, iou_rows, image_to_np, label_to_concept_grid, args.grid_size, args)
    num_iou_rows = len(iou_rows)
    num_category_iou_rows = len(category_iou_rows)
    num_overlay_rows = len(overlay_rows)
    del best_rows, iou_rows, category_iou_rows, thresholds, overlay_rows
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sensitivity_rows = write_threshold_sensitivity(
        output_dir,
        shard_paths,
        sample_by_id,
        sae,
        token_stats,
        concept_by_id,
        categories,
        latent_ids,
        latent_index,
        parse_float_list(args.sensitivity_percentiles),
        label_to_concept_grid,
        args.grid_size,
        args.threshold_latent_chunk_size,
    )
    run_config.update({
        "status": "completed",
        "elapsed_seconds": time.perf_counter() - start,
        "activation_cache_dir": str(_activation_cache_dir(args)),
        "num_activation_shards": len(shard_paths),
        "num_activation_records": int(load_activation_cache_meta(_activation_cache_dir(args)).get("num_cached_samples", 0)),
        "num_iou_rows": num_iou_rows,
        "num_category_iou_rows": num_category_iou_rows,
        "num_overlay_rows": num_overlay_rows,
        "num_sensitivity_rows": len(sensitivity_rows),
        "threshold_policy": f"latent-wise percentile >= {args.activation_percentile}",
    })
    save_json(output_dir / "run_config.json", run_config)
    print((output_dir / "sae_broden_alignment.md").read_text(encoding="utf-8"))
    print(f"Completed Broden alignment in {time.perf_counter() - start:.1f}s. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
