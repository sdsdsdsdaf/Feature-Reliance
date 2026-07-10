import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


BRODEN_CATEGORIES = ("object", "part", "color", "material", "texture", "scene")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class BrodenConcept:
    concept_id: int
    name: str
    category: str


@dataclass
class BrodenAnnotation:
    category: str
    concept_ids: list[int]
    mask_path: str | None = None
    image_level: bool = False


@dataclass
class BrodenSample:
    sample_id: int
    image_path: str
    annotations: list[BrodenAnnotation]


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(data), f, indent=2, ensure_ascii=False)


def _jsonable(value):
    if dataclass_is_instance(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value


def dataclass_is_instance(value):
    return hasattr(value, "__dataclass_fields__")


def discover_broden(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Broden root does not exist: {root}")
    concept_by_id = load_broden_concepts(root)
    samples = load_broden_index(root, concept_by_id)
    if not samples:
        raise RuntimeError(f"No Broden samples found under {root}. Expected index.csv or image files.")
    return concept_by_id, samples


def load_broden_concepts(root):
    root = Path(root)
    concept_by_id = {}
    for path in sorted(root.rglob("c_*.csv")):
        category = path.stem[2:].lower()
        with path.open("r", encoding="utf-8", newline="") as f:
            sample = f.read(2048)
            f.seek(0)
            has_header = "name" in sample.lower() and ("number" in sample.lower() or "id" in sample.lower())
            reader = csv.DictReader(f) if has_header else csv.reader(f)
            for row in reader:
                if has_header:
                    cid = _first_int(row, ("number", "id", "concept_id", "index"))
                    name = _first_str(row, ("name", "concept", "label"))
                    cat = _first_str(row, ("category", "cat")) or category
                else:
                    if not row or len(row) < 2:
                        continue
                    cid = _parse_int(row[0])
                    name = str(row[1]).strip()
                    cat = category
                if cid is None or not name:
                    continue
                concept_by_id[int(cid)] = BrodenConcept(int(cid), name, str(cat).strip().lower())
    return concept_by_id


def load_broden_index(root, concept_by_id):
    root = Path(root)
    index_path = _find_first(root, "index.csv")
    if index_path is None:
        return _samples_from_images(root)
    with index_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    samples = []
    for sample_id, row in enumerate(rows):
        image_value = _first_str(row, ("image", "img", "filename", "file", "path"))
        if not image_value:
            continue
        image_path = resolve_broden_path(root, image_value)
        if image_path is None:
            continue
        annotations = []
        for category in BRODEN_CATEGORIES:
            if category not in row or not str(row.get(category, "")).strip():
                continue
            annotations.extend(_parse_annotation_cell(root, category, row[category], concept_by_id))
        samples.append(BrodenSample(sample_id=sample_id, image_path=str(image_path), annotations=annotations))
    return samples


def _samples_from_images(root):
    samples = []
    skip_dirs = {"label", "labels", "annotations", "annotation", "segmentation", "segmentations"}
    image_paths = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not any(part.lower() in skip_dirs for part in p.parts)
    ]
    for idx, path in enumerate(sorted(image_paths)):
        samples.append(BrodenSample(sample_id=idx, image_path=str(path), annotations=[]))
    return samples


def _parse_annotation_cell(root, category, value, concept_by_id):
    text = str(value).strip()
    if not text:
        return []
    pieces = [x.strip() for x in text.replace(";", ",").replace("|", ",").split(",") if x.strip()]
    path_like = resolve_broden_path(root, text)
    if path_like is not None and path_like.suffix.lower() in IMAGE_EXTENSIONS:
        ids = _ids_for_category(concept_by_id, category)
        return [BrodenAnnotation(category=category, concept_ids=ids, mask_path=str(path_like), image_level=False)]
    ids = []
    for piece in pieces or [text]:
        cid = _parse_int(piece)
        if cid is not None:
            ids.append(int(cid))
    ids = [cid for cid in ids if cid in concept_by_id or not concept_by_id]
    if not ids:
        return []
    return [BrodenAnnotation(category=category, concept_ids=ids, image_level=category in {"scene", "texture"})]


def _ids_for_category(concept_by_id, category):
    ids = [cid for cid, concept in concept_by_id.items() if concept.category == category]
    return sorted(ids)


def resolve_broden_path(root, value):
    root = Path(root)
    value = str(value).strip()
    if not value:
        return None
    path = Path(value)
    candidates = [path] if path.is_absolute() else [root / value, root / "images" / value, root / "label" / value, root / "labels" / value]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _find_first(root, name):
    matches = sorted(Path(root).rglob(name))
    return matches[0] if matches else None


def _first_str(row, keys):
    lowered = {str(k).lower(): v for k, v in row.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _first_int(row, keys):
    text = _first_str(row, keys)
    return _parse_int(text)


def _parse_int(value):
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def parse_int_list(value):
    if value is None or str(value).lower() == "all":
        return None
    ids = []
    for piece in str(value).replace(";", ",").split(","):
        piece = piece.strip()
        if piece:
            ids.append(int(piece))
    return ids


def parse_str_list(value, default):
    if value is None:
        return list(default)
    return [x.strip().lower() for x in str(value).replace(";", ",").split(",") if x.strip()]


def parse_float_list(value):
    return [float(x.strip()) for x in str(value).replace(";", ",").split(",") if x.strip()]


def parse_category_subsets(value, categories=BRODEN_CATEGORIES):
    """Parse ``category=count`` pairs used to cap Broden samples per category."""
    if value is None or not str(value).strip():
        return {}
    valid_categories = {str(category).lower() for category in categories}
    subsets = {}
    for piece in str(value).replace(";", ",").split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"Invalid category subset {piece!r}; expected category=count (for example object=32).")
        category, count = (part.strip().lower() for part in piece.split("=", 1))
        if category not in valid_categories:
            raise ValueError(f"Unknown Broden category {category!r}; choose from {sorted(valid_categories)}.")
        try:
            count = int(count)
        except ValueError as exc:
            raise ValueError(f"Subset count for {category!r} must be an integer, got {count!r}.") from exc
        if count < 1:
            raise ValueError(f"Subset count for {category!r} must be at least 1, got {count}.")
        subsets[category] = count
    return subsets


def select_category_subsets(samples, categories, subset_sizes, seed=0):
    """Select a deterministic per-category sample union and remove unselected annotations."""
    categories = [str(category).lower() for category in categories]
    if not subset_sizes:
        return list(samples), {category: None for category in categories}
    rng = random.Random(int(seed))
    selected_ids_by_category = {}
    selected_counts = {}
    for category in categories:
        candidates = [sample for sample in samples if any(a.category == category for a in sample.annotations)]
        cap = subset_sizes.get(category)
        if cap is not None and cap < len(candidates):
            candidates = list(candidates)
            rng.shuffle(candidates)
            candidates = candidates[:cap]
        selected_ids_by_category[category] = {int(sample.sample_id) for sample in candidates}
        selected_counts[category] = len(candidates)
    selected_samples = []
    for sample in samples:
        annotations = [a for a in sample.annotations if a.category in selected_ids_by_category and int(sample.sample_id) in selected_ids_by_category[a.category]]
        if annotations:
            selected_samples.append(BrodenSample(sample.sample_id, sample.image_path, annotations))
    return selected_samples, selected_counts


def make_image_preprocessor(data_config):
    input_size = data_config.get("input_size", (3, 224, 224))
    size = int(input_size[-1])
    crop_pct = float(data_config.get("crop_pct", 0.875))
    mean = tuple(float(x) for x in data_config.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(float(x) for x in data_config.get("std", (0.229, 0.224, 0.225)))
    resize_size = int(math.floor(size / crop_pct)) if crop_pct > 0 else size

    def preprocess_pil(image):
        image = image.convert("RGB")
        cropped = resize_center_crop(image, resize_size=resize_size, crop_size=size, resample=Image.BICUBIC)
        arr = np.asarray(cropped).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        mean_t = torch.tensor(mean)[:, None, None]
        std_t = torch.tensor(std)[:, None, None]
        return (tensor - mean_t) / std_t

    def image_to_np(image):
        image = image.convert("RGB")
        cropped = resize_center_crop(image, resize_size=resize_size, crop_size=size, resample=Image.BICUBIC)
        return np.asarray(cropped).astype(np.float32) / 255.0

    def mask_to_grid(mask, grid_size):
        cropped = resize_center_crop(mask, resize_size=resize_size, crop_size=size, resample=Image.NEAREST)
        arr = np.asarray(cropped)
        return downsample_binary_mask(arr, grid_size=grid_size)

    def label_to_concept_grid(mask, concept_id, grid_size, allow_binary_fallback=False):
        cropped = resize_center_crop(mask, resize_size=resize_size, crop_size=size, resample=Image.NEAREST)
        arr = np.asarray(cropped)
        if arr.ndim == 3:
            arr = arr[..., 0]
        binary = arr == int(concept_id)
        if not binary.any() and allow_binary_fallback:
            binary = arr > 0
        if not binary.any():
            return None
        return downsample_binary_mask(binary.astype(np.uint8), grid_size=grid_size)

    return preprocess_pil, image_to_np, mask_to_grid, label_to_concept_grid, {"mean": mean, "std": std, "size": size, "resize_size": resize_size}


def resize_center_crop(image, resize_size, crop_size, resample):
    image = image.resize((resize_size, resize_size), resample=resample)
    left = max(0, (resize_size - crop_size) // 2)
    top = max(0, (resize_size - crop_size) // 2)
    return image.crop((left, top, left + crop_size, top + crop_size))


def downsample_binary_mask(mask_array, grid_size):
    if mask_array.ndim == 3:
        mask_array = mask_array[..., 0]
    mask = torch.from_numpy((mask_array > 0).astype(np.float32))[None, None]
    pooled = F.interpolate(mask, size=(grid_size, grid_size), mode="nearest")
    return pooled[0, 0].bool()


def concept_mask_for_sample(sample, concept_id, category, label_to_concept_grid, grid_size):
    for ann in sample.annotations:
        if ann.category != category or concept_id not in ann.concept_ids:
            continue
        if ann.image_level or ann.mask_path is None:
            return torch.ones(grid_size, grid_size, dtype=torch.bool)
        mask_img = Image.open(ann.mask_path)
        return label_to_concept_grid(
            mask_img,
            concept_id=concept_id,
            grid_size=grid_size,
            allow_binary_fallback=len(ann.concept_ids) == 1,
        )
    return None


def normalize_map(values, eps=1e-8):
    arr = np.asarray(values, dtype=np.float32)
    arr = arr - float(np.nanmin(arr))
    denom = float(np.nanmax(arr)) + eps
    return arr / denom


def plot_broden_preview(image_np, mask_grid, title, output_path, cmap="tab20"):
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
    axes[0].imshow(image_np)
    axes[0].set_title("Broden image", fontsize=9)
    axes[0].axis("off")
    axes[1].imshow(image_np)
    axes[1].imshow(mask_grid.float().numpy(), cmap=cmap, alpha=0.55, extent=(0, image_np.shape[1], image_np.shape[0], 0), interpolation="nearest")
    axes[1].set_title(title, fontsize=9)
    axes[1].axis("off")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_alignment_overlay(image_np, broden_mask, activation_map, binary_activation, title, output_path, overlay_cmap="magma", overlay_alpha=0.45):
    activation_grid = normalize_map(np.asarray(activation_map).reshape(broden_mask.shape))
    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.4))
    axes[0].imshow(image_np)
    axes[0].set_title("image", fontsize=9)
    axes[1].imshow(image_np)
    axes[1].imshow(broden_mask.float().numpy(), cmap="Greens", alpha=0.55, extent=(0, image_np.shape[1], image_np.shape[0], 0), interpolation="nearest")
    axes[1].set_title("Broden mask", fontsize=9)
    axes[2].imshow(image_np)
    axes[2].imshow(activation_grid, cmap=overlay_cmap, alpha=overlay_alpha, extent=(0, image_np.shape[1], image_np.shape[0], 0), interpolation="nearest")
    axes[2].set_title("SAE activation", fontsize=9)
    axes[3].imshow(image_np)
    axes[3].imshow(binary_activation.float().numpy(), cmap="Reds", alpha=0.5, extent=(0, image_np.shape[1], image_np.shape[0], 0), interpolation="nearest")
    axes[3].set_title("thresholded SAE", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def markdown_table(rows, columns, aligns=None):
    aligns = aligns or {}
    header = "| " + " | ".join(columns) + " |"
    sep_cells = []
    for col in columns:
        align = aligns.get(col, "left")
        if align == "right":
            sep_cells.append("---:")
        elif align == "center":
            sep_cells.append(":---:")
        else:
            sep_cells.append("---")
    lines = [header, "| " + " | ".join(sep_cells) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def sample_preview_records(samples, concept_by_id, categories, per_category, seed):
    rng = random.Random(int(seed))
    by_category = {cat: [] for cat in categories}
    for sample in samples:
        for ann in sample.annotations:
            if ann.category in by_category:
                for cid in ann.concept_ids:
                    by_category[ann.category].append((sample, cid))
    selected = []
    for cat in categories:
        candidates = by_category.get(cat, [])
        rng.shuffle(candidates)
        for sample, cid in candidates[: int(per_category)]:
            concept = concept_by_id.get(cid, BrodenConcept(cid, str(cid), cat))
            selected.append((sample, concept))
    return selected
