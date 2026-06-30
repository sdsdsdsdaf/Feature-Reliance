#!/usr/bin/env python3
"""Download/cache ImageNet-1K from Hugging Face.

Default target is the validation split because the first ReservoirTTA smoke
tests only need clean ImageNet validation as the source/reference domain.

Before running this script, accept the dataset terms on Hugging Face:
https://huggingface.co/datasets/ILSVRC/imagenet-1k

Then login once:
    huggingface-cli login
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download/cache HF ImageNet-1K.")
    parser.add_argument(
        "--dataset",
        default="ILSVRC/imagenet-1k",
        help="Hugging Face dataset id.",
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=["train", "validation", "test"],
        help="Split to download/cache. Use validation for the minimal experiment.",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/hf_cache",
        help="Hugging Face datasets cache directory.",
    )
    parser.add_argument(
        "--output-meta",
        default="data/hf_cache/imagenet1k_metadata.json",
        help="Where to write a small metadata file after download.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional HF token. If omitted, the cached huggingface-cli login is used.",
    )
    return parser.parse_args()


def load_hf_dataset(dataset: str, split: str, cache_dir: str, token: str | None):
    kwargs = {
        "path": dataset,
        "split": split,
        "cache_dir": cache_dir,
    }
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

    dataset = load_hf_dataset(args.dataset, args.split, str(cache_dir), args.token)

    meta = {
        "dataset": args.dataset,
        "split": args.split,
        "num_rows": len(dataset),
        "features": list(dataset.features.keys()),
        "cache_files": dataset.cache_files,
    }

    output_meta = Path(args.output_meta)
    output_meta.parent.mkdir(parents=True, exist_ok=True)
    output_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Downloaded/cached {args.dataset} split={args.split}")
    print(f"Rows: {len(dataset):,}")
    print(f"Cache dir: {cache_dir.resolve()}")
    print(f"Metadata: {output_meta.resolve()}")


if __name__ == "__main__":
    main()
