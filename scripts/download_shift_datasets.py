#!/usr/bin/env python3
"""Download the distribution-shift datasets required by Plans.md T0.1.

Covers the four datasets still listed as "확보 필요" plus ColoredMNIST:

    imagenet-r       data/imagenet-r/<wnid>/*.jpg              (30k imgs, 200 cls)
    imagenet-a       data/imagenet-a/<wnid>/*.jpg              (7.5k imgs, 200 cls)
    imagenet-sketch  data/imagenet-sketch/<wnid>/*.JPEG        (50.9k imgs, 1000 cls)
    imagenet-9       data/imagenet-9/<variant>/val/<cls>/*.JPEG (BG Challenge)
    colored-mnist    data/colored-mnist/{train1,train2,test}.pt (IRM construction)

Waterbirds and ImageNet-C are already present and are NOT handled here
(ImageNet-C is pinned to the precomputed Zenodo copy — see
scripts/download_hf_imagenet_c.py and the ImageNet-C 규약 in experiment_plan.md).

Usage:
    python scripts/download_shift_datasets.py --datasets imagenet-r imagenet-a
    python scripts/download_shift_datasets.py            # all five
    python scripts/download_shift_datasets.py --dry-run  # print sizes only
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ALL_DATASETS = ["imagenet-r", "imagenet-a", "imagenet-sketch", "imagenet-9", "colored-mnist"]

IMAGENET_R_URL = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
IMAGENET_A_URL = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar"
BG_CHALLENGE_RELEASES = "https://api.github.com/repos/MadryLab/backgrounds_challenge/releases"

# ImageNet-Sketch is distributed via Google Drive upstream; gdown is not installed,
# so we pull the HF mirror instead. Override if the default repo id moves.
SKETCH_HF_REPO = "songweig/imagenet_sketch"

EXPECTED_CLASSES = {"imagenet-r": 200, "imagenet-a": 200, "imagenet-sketch": 1000}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download shift datasets for T0.1.")
    p.add_argument("--datasets", nargs="+", choices=ALL_DATASETS, default=ALL_DATASETS)
    p.add_argument("--data-root", default="data", help="Root for extracted datasets.")
    p.add_argument("--download-dir", default="data/zenodo_cache/shift", help="Archive cache.")
    p.add_argument("--sketch-hf-repo", default=SKETCH_HF_REPO, help="HF dataset repo for ImageNet-Sketch.")
    p.add_argument("--mnist-root", default="data/mnist", help="Where torchvision caches raw MNIST.")
    p.add_argument("--cmnist-size", type=int, default=14, help="ColoredMNIST resolution (IRM uses 14).")
    p.add_argument("--force", action="store_true", help="Delete existing output dirs first.")
    p.add_argument("--keep-archives", action="store_true", help="Keep tars after extraction.")
    p.add_argument("--dry-run", action="store_true", help="Resolve URLs and print sizes; download nothing.")
    p.add_argument("--timeout", type=int, default=60)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    download_dir = Path(args.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, dict] = {}
    for name in args.datasets:
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        out = data_root / name
        if args.force and out.exists():
            print(f"Removing existing {out}")
            shutil.rmtree(out)
        try:
            if name == "imagenet-r":
                report[name] = prepare_tar(IMAGENET_R_URL, "imagenet-r.tar", out, data_root, download_dir, args)
            elif name == "imagenet-a":
                report[name] = prepare_tar(IMAGENET_A_URL, "imagenet-a.tar", out, data_root, download_dir, args)
            elif name == "imagenet-sketch":
                report[name] = prepare_sketch(out, args)
            elif name == "imagenet-9":
                report[name] = prepare_imagenet_9(out, download_dir, args)
            elif name == "colored-mnist":
                report[name] = prepare_colored_mnist(out, args)
        except Exception as exc:  # keep going; one dataset failing shouldn't block the rest
            print(f"[failed] {name}: {exc}", file=sys.stderr)
            report[name] = {"status": "failed", "error": str(exc)}

    meta_path = data_root / "shift_datasets_metadata.json"
    meta_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nMetadata written to {meta_path.resolve()}")

    failed = [k for k, v in report.items() if v.get("status") == "failed"]
    if failed:
        print(f"[warn] failed datasets: {', '.join(failed)}", file=sys.stderr)
        raise SystemExit(1)


# --------------------------------------------------------------------------- tar datasets


def prepare_tar(url: str, archive_name: str, out: Path, data_root: Path, download_dir: Path, args) -> dict:
    """ImageNet-R / ImageNet-A: single tar that extracts to <name>/<wnid>/*.jpg."""
    size = head_size(url, timeout=args.timeout)
    print(f"URL:  {url}\nSize: {format_bytes(size) if size else 'unknown'}")
    if args.dry_run:
        return {"status": "dry-run", "url": url, "bytes": size}

    if out.exists() and any(out.iterdir()):
        print(f"[skip] {out} already populated")
    else:
        archive = download_dir / archive_name
        download_file(url, archive, expected_size=size, timeout=args.timeout)
        print(f"[start] extract {archive.name}")
        with tarfile.open(archive) as handle:
            safe_extract(handle, data_root)  # tar contains a top-level <name>/ dir
        print(f"[done] extract {archive.name}")
        if not args.keep_archives:
            archive.unlink()

    classes = count_class_dirs(out)
    images = count_images(out)
    expected = EXPECTED_CLASSES.get(out.name)
    if expected and classes != expected:
        print(f"[warn] {out.name}: {classes} class dirs, expected {expected}")
    print(f"{out.name}: {images} images / {classes} classes")
    return {"status": "ok", "root": str(out), "images": images, "classes": classes, "bytes": size}


# --------------------------------------------------------------------------- imagenet-sketch


def prepare_sketch(out: Path, args) -> dict:
    """ImageNet-Sketch via the HF mirror (upstream is Google Drive; gdown unavailable)."""
    if args.dry_run:
        print(f"HF repo: {args.sketch_hf_repo} (size resolved at download time)")
        return {"status": "dry-run", "hf_repo": args.sketch_hf_repo}

    if out.exists() and any(out.iterdir()):
        print(f"[skip] {out} already populated")
    else:
        from huggingface_hub import snapshot_download

        print(f"snapshot_download({args.sketch_hf_repo!r}) -> {out}")
        snapshot_download(
            repo_id=args.sketch_hf_repo,
            repo_type="dataset",
            local_dir=str(out),
            max_workers=4,
        )
        for archive in sorted(out.rglob("*.tar.gz")) + sorted(out.rglob("*.tar")):
            print(f"[start] extract {archive.name}")
            with tarfile.open(archive) as handle:
                safe_extract(handle, out)
            if not args.keep_archives:
                archive.unlink()

    images = count_images(out)
    classes = count_class_dirs(out)
    print(f"imagenet-sketch: {images} images / {classes} top-level dirs")
    return {"status": "ok", "root": str(out), "images": images, "classes": classes,
            "hf_repo": args.sketch_hf_repo}


# --------------------------------------------------------------------------- imagenet-9 / BG challenge


def prepare_imagenet_9(out: Path, download_dir: Path, args) -> dict:
    """Backgrounds Challenge (ImageNet-9). Asset names are resolved from the GitHub
    releases API rather than hardcoded, since the repo has renamed them before."""
    print(f"Resolving release assets: {BG_CHALLENGE_RELEASES}")
    req = urllib.request.Request(BG_CHALLENGE_RELEASES, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=args.timeout) as response:
        releases = json.loads(response.read().decode("utf-8"))

    assets = [a for rel in releases for a in rel.get("assets", [])]
    if not assets:
        raise RuntimeError("No release assets found on MadryLab/backgrounds_challenge")

    total = sum(int(a.get("size") or 0) for a in assets)
    print(f"Found {len(assets)} asset(s), total {format_bytes(total)}:")
    for a in assets:
        print(f"  {a['name']:45s} {format_bytes(int(a.get('size') or 0))}")
    if args.dry_run:
        return {"status": "dry-run", "assets": [{"name": a["name"], "bytes": a.get("size")} for a in assets]}

    out.mkdir(parents=True, exist_ok=True)
    for a in assets:
        archive = download_dir / a["name"]
        download_file(a["browser_download_url"], archive, expected_size=int(a.get("size") or 0),
                      timeout=args.timeout)
        if archive.suffix in {".gz", ".tar", ".tgz"} or archive.name.endswith(".tar.gz"):
            print(f"[start] extract {archive.name}")
            with tarfile.open(archive) as handle:
                safe_extract(handle, out)
            print(f"[done] extract {archive.name}")
            if not args.keep_archives:
                archive.unlink()

    variants = sorted(d.name for d in out.iterdir() if d.is_dir())
    images = count_images(out)
    print(f"imagenet-9: {images} images, variants={variants}")
    return {"status": "ok", "root": str(out), "images": images, "variants": variants, "bytes": total}


# --------------------------------------------------------------------------- colored mnist


def prepare_colored_mnist(out: Path, args) -> dict:
    """ColoredMNIST per Arjovsky et al. (IRM): label = digit<5 with 25% label noise,
    colour correlates with the noisy label at rate (1-e). Envs e = 0.2, 0.1 (train)
    and 0.9 (test) — the correlation flip that makes the shortcut fail at test time."""
    if args.dry_run:
        print("Generated locally from torchvision MNIST (~12 MB download).")
        return {"status": "dry-run", "source": "torchvision MNIST"}

    import torch
    from torchvision import datasets

    out.mkdir(parents=True, exist_ok=True)
    mnist = datasets.MNIST(args.mnist_root, train=True, download=True)
    images, labels = mnist.data, mnist.targets

    generator = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(images), generator=generator)
    images, labels = images[perm], labels[perm]

    step = max(1, 28 // args.cmnist_size)
    images = images[:, ::step, ::step]

    def bernoulli(p: float, size: int) -> torch.Tensor:
        return (torch.rand(size, generator=generator) < p).float()

    def xor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a - b).abs()

    def make_env(imgs: torch.Tensor, digits: torch.Tensor, e: float) -> dict:
        y = (digits < 5).float()
        y = xor(y, bernoulli(0.25, len(y)))            # 25% label noise
        colors = xor(y, bernoulli(e, len(y)))          # colour agrees with y at rate 1-e
        stacked = torch.stack([imgs, imgs], dim=1).float()
        stacked[torch.arange(len(stacked)), (1 - colors).long(), :, :] *= 0
        return {"images": stacked / 255.0, "labels": y[:, None], "e": e}

    splits = {"train1": (0, 2, 0.2), "train2": (1, 2, 0.1), "test": (None, None, 0.9)}
    written = {}
    for name, (offset, stride, e) in splits.items():
        if offset is None:                              # test = last third, held out
            sl = slice(2 * len(images) // 3, None)
            env = make_env(images[sl], labels[sl], e)
        else:
            sub = slice(0, 2 * len(images) // 3)
            env = make_env(images[sub][offset::stride], labels[sub][offset::stride], e)
        path = out / f"{name}.pt"
        torch.save(env, path)
        written[name] = {"n": len(env["labels"]), "e": e, "path": str(path)}
        print(f"  {name}: n={len(env['labels'])} e={e} -> {path}")

    return {"status": "ok", "root": str(out), "splits": written,
            "resolution": args.cmnist_size, "channels": 2}


# --------------------------------------------------------------------------- helpers


def head_size(url: str, timeout: int) -> int:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return int(response.headers.get("Content-Length") or 0)
    except Exception:
        return 0


def download_file(url: str, dst: Path, expected_size: int, timeout: int) -> None:
    if dst.exists() and expected_size > 0 and dst.stat().st_size == expected_size:
        print(f"[skip] {dst.name} already downloaded ({format_bytes(expected_size)})")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    existing = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    mode = "ab" if existing else "wb"

    print(f"[start] download {dst.name} ({format_bytes(expected_size) if expected_size else 'unknown size'})")
    start = time.perf_counter()
    downloaded = existing
    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open(mode) as handle:
            if existing and getattr(response, "status", None) == 200:
                handle.truncate(0)          # server ignored Range; restart cleanly
                downloaded = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                print_progress(dst.name, downloaded, expected_size, start)
    except urllib.error.HTTPError as exc:
        if not (existing and exc.code == 416):
            raise
    print()

    if expected_size > 0 and tmp.stat().st_size != expected_size:
        raise RuntimeError(f"Size mismatch for {dst.name}: got {tmp.stat().st_size}, expected {expected_size}")
    tmp.replace(dst)
    print(f"[done] download {dst.name} ({time.perf_counter() - start:.1f}s)")


def print_progress(name: str, downloaded: int, total: int, start: float) -> None:
    elapsed = max(time.perf_counter() - start, 1e-6)
    speed = downloaded / elapsed
    if total > 0:
        print(f"\r  {name}: {100.0 * downloaded / total:5.1f}% "
              f"{format_bytes(downloaded)}/{format_bytes(total)} {format_bytes(speed)}/s", end="", flush=True)
    else:
        print(f"\r  {name}: {format_bytes(downloaded)} {format_bytes(speed)}/s", end="", flush=True)


def safe_extract(handle: tarfile.TarFile, output_dir: Path) -> None:
    output_root = output_dir.resolve()
    for member in handle.getmembers():
        target = (output_dir / member.name).resolve()
        if output_root not in [target, *target.parents]:
            raise RuntimeError(f"Unsafe path in tar archive: {member.name}")
    handle.extractall(output_dir)


def count_class_dirs(path: Path) -> int:
    return sum(1 for item in path.iterdir() if item.is_dir()) if path.exists() else 0


def count_images(path: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in exts)


def format_bytes(num_bytes: float) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}TB"


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[interrupted]", file=sys.stderr)
        raise SystemExit(130) from None
