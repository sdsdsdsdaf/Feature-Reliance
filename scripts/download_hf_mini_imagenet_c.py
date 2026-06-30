#!/usr/bin/env python3
"""Download/cache compact ImageNet-C from Hugging Face.

The default dataset is a compact ImageNet-C variant for fast pipeline checks.
It is intended for smoke tests before moving to full ImageNet-C.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download/cache compact HF ImageNet-C.")
    parser.add_argument(
        "--dataset",
        default="niuniandaji/mini-imagenet-c",
        help="Hugging Face dataset id for compact ImageNet-C.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional HF dataset config name, if the selected dataset uses configs.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split to download/cache.",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/hf_cache",
        help="Hugging Face datasets cache directory.",
    )
    parser.add_argument(
        "--output-meta",
        default="data/hf_cache/mini_imagenet_c_metadata.json",
        help="Where to write a small metadata file after download.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional HF token. If omitted, the cached huggingface-cli login is used.",
    )
    return parser.parse_args()


def load_hf_dataset(
    dataset: str,
    config: str | None,
    split: str,
    cache_dir: str,
    token: str | None,
):
    kwargs = {
        "path": dataset,
        "split": split,
        "cache_dir": cache_dir,
    }
    if config is not None:
        kwargs["name"] = config
    if token is not None:
        kwargs["token"] = token

    try:
        return load_dataset(**kwargs)
    except TypeError:
        if token is not None:
            kwargs.pop("token", None)
            kwargs["use_auth_token"] = token
        return load_dataset(**kwargs)


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_hf_dataset(args.dataset, args.config, args.split, str(cache_dir), args.token)

    meta = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "num_rows": len(dataset),
        "features": list(dataset.features.keys()),
        "cache_files": dataset.cache_files,
    }

    output_meta = Path(args.output_meta)
    output_meta.parent.mkdir(parents=True, exist_ok=True)
    output_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Downloaded/cached {args.dataset} split={args.split}")
    if args.config is not None:
        print(f"Config: {args.config}")
    print(f"Rows: {len(dataset):,}")
    print(f"Cache dir: {cache_dir.resolve()}")
    print(f"Metadata: {output_meta.resolve()}")


if __name__ == "__main__":
    main()
