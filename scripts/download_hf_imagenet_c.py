#!/usr/bin/env python3
"""Download the official ImageNet-C archives from Zenodo.

The filename is kept for backward compatibility with earlier commands. This
script no longer downloads a Hugging Face or Kaggle mirror. It prepares a local
ImageNet-C root like:

    data/imagenet-c/gaussian_noise/5/n01440764/*.JPEG

By default it downloads the standard 15-corruption ImageNet-C archives:
noise.tar, blur.tar, weather.tar, and digital.tar. The Zenodo record also has
extra.tar with additional corruptions; use --include-extra to download it.
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
from typing import Iterable


ZENODO_RECORD_ID = "2235448"
ZENODO_RECORD_API = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

STANDARD_ARCHIVES = ["noise.tar", "blur.tar", "weather.tar", "digital.tar"]
EXTRA_ARCHIVES = ["extra.tar"]

ARCHIVE_CORRUPTIONS = {
    "noise.tar": ["gaussian_noise", "shot_noise", "impulse_noise"],
    "blur.tar": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "weather.tar": ["frost", "snow", "fog", "brightness"],
    "digital.tar": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
    "extra.tar": ["speckle_noise", "spatter", "gaussian_blur", "saturate"],
}

STANDARD_CORRUPTIONS = [name for archive in STANDARD_ARCHIVES for name in ARCHIVE_CORRUPTIONS[archive]]
ALL_CORRUPTIONS = [name for archive in STANDARD_ARCHIVES + EXTRA_ARCHIVES for name in ARCHIVE_CORRUPTIONS[archive]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download official ImageNet-C from Zenodo.")
    parser.add_argument(
        "--output-dir",
        default="data/imagenet-c",
        help="Final ImageNet-C root used by reservoir_sae experiments.",
    )
    parser.add_argument(
        "--download-dir",
        default="data/zenodo_cache/imagenet-c",
        help="Archive download/cache directory.",
    )
    parser.add_argument(
        "--output-meta",
        default="data/imagenet-c/metadata.json",
        help="Metadata json to write after download/extract.",
    )
    parser.add_argument(
        "--archives",
        nargs="+",
        choices=STANDARD_ARCHIVES + EXTRA_ARCHIVES,
        default=None,
        help="Specific Zenodo tar archives to download/extract.",
    )
    parser.add_argument(
        "--include-extra",
        action="store_true",
        help="Also download extra.tar with speckle_noise, spatter, gaussian_blur, and saturate.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove existing output/download directories before downloading.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only extract/normalize archives already present in --download-dir.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded tar files after extraction.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout in seconds for Zenodo API/file requests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    download_dir = Path(args.download_dir)
    archives = selected_archives(args)

    if args.force:
        if output_dir.exists():
            print(f"Removing existing output dir: {output_dir}")
            shutil.rmtree(output_dir)
        if download_dir.exists():
            print(f"Removing existing download dir: {download_dir}")
            shutil.rmtree(download_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    print("Official ImageNet-C Zenodo downloader")
    print(f"Zenodo record: {ZENODO_RECORD_ID}")
    print(f"Archives:      {', '.join(archives)}")
    print(f"Download dir:  {download_dir}")
    print(f"Output root:   {output_dir}")

    zenodo_files = get_zenodo_files(timeout=args.timeout) if not args.skip_download else {}
    if not args.skip_download:
        download_archives(archives, zenodo_files, download_dir, timeout=args.timeout)
    else:
        print("Skipping download; using archives already present in download dir.")

    extract_archives(archives, download_dir, output_dir, keep_archives=args.keep_archives)
    validate_imagenet_c_root(output_dir, expected_corruptions=expected_corruptions(archives))
    meta = write_metadata(archives, download_dir, output_dir, Path(args.output_meta))

    print("ImageNet-C Zenodo download prepared.")
    print(f"Output root: {output_dir.resolve()}")
    print(f"Metadata:    {Path(args.output_meta).resolve()}")
    print(f"Corruptions: {len(meta['corruptions'])}")
    print("Use with:")
    print(f"  --imagenet-c-root {output_dir}")


def selected_archives(args: argparse.Namespace) -> list[str]:
    if args.archives:
        archives = list(args.archives)
    else:
        archives = list(STANDARD_ARCHIVES)
    if args.include_extra and "extra.tar" not in archives:
        archives.append("extra.tar")
    return archives


def expected_corruptions(archives: Iterable[str]) -> list[str]:
    result: list[str] = []
    for archive in archives:
        result.extend(ARCHIVE_CORRUPTIONS[archive])
    return result


def get_zenodo_files(timeout: int) -> dict[str, dict]:
    print(f"Resolving Zenodo record metadata: {ZENODO_RECORD_API}")
    with urllib.request.urlopen(ZENODO_RECORD_API, timeout=timeout) as response:
        record = json.loads(response.read().decode("utf-8"))

    files = {}
    for item in record.get("files", []):
        key = item.get("key")
        if key:
            files[key] = item
    if not files:
        raise RuntimeError(f"No files found in Zenodo record {ZENODO_RECORD_ID}")
    return files


def download_archives(archives: Iterable[str], zenodo_files: dict[str, dict], download_dir: Path, timeout: int) -> None:
    for archive in archives:
        if archive not in zenodo_files:
            available = ", ".join(sorted(zenodo_files))
            raise RuntimeError(f"Zenodo archive {archive!r} not found. Available files: {available}")
        file_info = zenodo_files[archive]
        url = file_info.get("links", {}).get("self")
        size = int(file_info.get("size") or 0)
        if not url:
            raise RuntimeError(f"Zenodo file {archive!r} has no download URL.")
        dst = download_dir / archive
        download_file(url, dst, expected_size=size, timeout=timeout)


def download_file(url: str, dst: Path, expected_size: int, timeout: int) -> None:
    if dst.exists() and expected_size > 0 and dst.stat().st_size == expected_size:
        print(f"[skip] {dst.name} already downloaded ({format_bytes(expected_size)})")
        return

    tmp = dst.with_suffix(dst.suffix + ".part")
    existing = tmp.stat().st_size if tmp.exists() else 0
    headers = {}
    mode = "wb"
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"

    print(f"[start] download {dst.name} ({format_bytes(expected_size)})")
    request = urllib.request.Request(url, headers=headers)
    start = time.perf_counter()
    downloaded = existing
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open(mode) as handle:
            if existing > 0 and getattr(response, "status", None) == 200:
                handle.close()
                mode = "wb"
                downloaded = 0
                print(f"\n  server did not resume {dst.name}; restarting from byte 0")
                return download_file_no_resume(url, dst, expected_size=expected_size, timeout=timeout)
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                print_download_progress(dst.name, downloaded, expected_size, start)
    except urllib.error.HTTPError as exc:
        if existing > 0 and exc.code == 416:
            pass
        else:
            raise
    print()

    if expected_size > 0 and tmp.stat().st_size != expected_size:
        raise RuntimeError(
            f"Downloaded size mismatch for {dst.name}: got {tmp.stat().st_size}, expected {expected_size}"
        )
    tmp.replace(dst)
    print(f"[done] download {dst.name} ({time.perf_counter() - start:.1f}s)")


def download_file_no_resume(url: str, dst: Path, expected_size: int, timeout: int) -> None:
    tmp = dst.with_suffix(dst.suffix + ".part")
    start = time.perf_counter()
    downloaded = 0
    request = urllib.request.Request(url)
    with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            print_download_progress(dst.name, downloaded, expected_size, start)
    print()
    if expected_size > 0 and tmp.stat().st_size != expected_size:
        raise RuntimeError(
            f"Downloaded size mismatch for {dst.name}: got {tmp.stat().st_size}, expected {expected_size}"
        )
    tmp.replace(dst)
    print(f"[done] download {dst.name} ({time.perf_counter() - start:.1f}s)")


def print_download_progress(name: str, downloaded: int, total: int, start: float) -> None:
    elapsed = max(time.perf_counter() - start, 1e-6)
    speed = downloaded / elapsed
    if total > 0:
        pct = 100.0 * downloaded / total
        message = f"\r  {name}: {pct:5.1f}% {format_bytes(downloaded)}/{format_bytes(total)} {format_bytes(speed)}/s"
    else:
        message = f"\r  {name}: {format_bytes(downloaded)} {format_bytes(speed)}/s"
    print(message, end="", flush=True)


def extract_archives(archives: Iterable[str], download_dir: Path, output_dir: Path, keep_archives: bool) -> None:
    for archive_name in archives:
        archive = download_dir / archive_name
        if not archive.exists():
            raise FileNotFoundError(f"Archive not found: {archive}")
        print(f"[start] extract {archive.name}")
        with tarfile.open(archive) as handle:
            safe_extract(handle, output_dir)
        print(f"[done] extract {archive.name}")
        if not keep_archives:
            archive.unlink()


def safe_extract(handle: tarfile.TarFile, output_dir: Path) -> None:
    output_root = output_dir.resolve()
    for member in handle.getmembers():
        target = (output_dir / member.name).resolve()
        if output_root not in [target, *target.parents]:
            raise RuntimeError(f"Unsafe path in tar archive: {member.name}")
    handle.extractall(output_dir)


def validate_imagenet_c_root(root: Path, expected_corruptions: list[str]) -> None:
    missing = [corruption for corruption in expected_corruptions if not (root / corruption).is_dir()]
    if missing:
        raise RuntimeError(f"Missing expected corruptions after extraction: {missing}")

    warnings = []
    for corruption in expected_corruptions:
        for severity in range(1, 6):
            severity_dir = root / corruption / str(severity)
            if not severity_dir.is_dir():
                warnings.append(f"{corruption}/{severity}: missing severity dir")
                continue
            class_count = count_class_dirs(severity_dir)
            if class_count != 1000:
                warnings.append(f"{corruption}/{severity}: {class_count} class dirs, expected 1000")
    if warnings:
        print("Validation warnings:")
        for warning in warnings[:30]:
            print(f"  {warning}")
        if len(warnings) > 30:
            print(f"  ... {len(warnings) - 30} more warnings")


def write_metadata(archives: list[str], download_dir: Path, output_dir: Path, output_meta: Path) -> dict:
    corruptions = {}
    for corruption in expected_corruptions(archives):
        corruption_dir = output_dir / corruption
        if not corruption_dir.exists():
            continue
        severities = {}
        for severity in range(1, 6):
            severity_dir = corruption_dir / str(severity)
            if severity_dir.exists():
                severities[str(severity)] = {
                    "images": count_images(severity_dir),
                    "classes": count_class_dirs(severity_dir),
                }
        corruptions[corruption] = severities

    meta = {
        "source": "zenodo",
        "zenodo_record": ZENODO_RECORD_ID,
        "zenodo_api": ZENODO_RECORD_API,
        "archives": archives,
        "download_dir": str(download_dir),
        "output_dir": str(output_dir),
        "layout": "corruption/severity/class/image",
        "corruptions": corruptions,
    }
    output_meta.parent.mkdir(parents=True, exist_ok=True)
    output_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def count_class_dirs(path: Path) -> int:
    return sum(1 for item in path.iterdir() if item.is_dir())


def count_images(path: Path) -> int:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in image_exts)


def format_bytes(num_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}TB"


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[failed] {exc}", file=sys.stderr)
        raise SystemExit(1) from None
