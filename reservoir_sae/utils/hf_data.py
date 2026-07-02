from __future__ import annotations

import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import datasets.config as datasets_config
from datasets import Dataset, load_dataset
from PIL import Image
from timm.data import ImageNetInfo
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESERVOIRTTA_CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]
IMAGENET_C_EXTRA_CORRUPTIONS = [
    "gaussian_blur",
    "saturate",
    "spatter",
    "speckle_noise",
]
IMAGENET_C_CORRUPTION_ORDER = RESERVOIRTTA_CORRUPTIONS + IMAGENET_C_EXTRA_CORRUPTIONS
RESERVOIRTTA_SEVERITIES = [5, 4, 3, 2, 1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_corruption_dirs(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    for corruption_dir in path.iterdir():
        if not corruption_dir.is_dir():
            continue
        if any(child.is_dir() and child.name.isdigit() for child in corruption_dir.iterdir()):
            count += 1
    return count


def discover_imagenet_c_domains(
    root: str | Path,
    corruptions: Iterable[str] | None = None,
    severities: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    root_path = resolve_imagenet_c_root(root)
    if root_path is None or not root_path.exists():
        return []

    available = {path.name for path in root_path.iterdir() if path.is_dir()}
    if corruptions is None:
        ordered_corruptions = [name for name in IMAGENET_C_CORRUPTION_ORDER if name in available]
        ordered_corruptions.extend(sorted(available - set(ordered_corruptions)))
    else:
        ordered_corruptions = [str(item) for item in corruptions]

    requested_severities = [int(item) for item in severities] if severities is not None else None
    domains: list[dict[str, Any]] = []
    for corruption in ordered_corruptions:
        corruption_dir = root_path / corruption
        if not corruption_dir.is_dir():
            continue
        if requested_severities is None:
            available_severities = sorted(
                (int(path.name) for path in corruption_dir.iterdir() if path.is_dir() and path.name.isdigit()),
                reverse=True,
            )
            ordered_severities = [severity for severity in RESERVOIRTTA_SEVERITIES if severity in available_severities]
            ordered_severities.extend(severity for severity in available_severities if severity not in ordered_severities)
        else:
            ordered_severities = requested_severities
        for severity in ordered_severities:
            segment_root = corruption_dir / str(severity)
            if segment_root.is_dir():
                domains.append(
                    {
                        "corruption": corruption,
                        "severity": int(severity),
                        "path": segment_root,
                    }
                )
    return domains


@dataclass(frozen=True)
class DatasetColumns:
    image: str
    label: str | None
    corruption: str | None
    severity: str | None


@dataclass(frozen=True)
class LabelMapping:
    values: list[int] | None
    source: str
    is_identity: bool

    def map(self, label: int) -> int:
        if self.values is None:
            return int(label)
        return int(self.values[int(label)])


def load_hf_split(
    dataset_name: str,
    split: str,
    cache_dir: str | None = None,
    config: str | None = None,
    token: str | None = None,
    offline: bool = False,
) -> Dataset:
    if offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        datasets_config.HF_DATASETS_OFFLINE = True

    kwargs: dict[str, Any] = {
        "path": dataset_name,
        "split": split,
        "cache_dir": cache_dir,
    }
    if config:
        kwargs["name"] = config
    if token:
        kwargs["token"] = token
    try:
        return load_dataset(**kwargs)
    except TypeError:
        if token:
            kwargs.pop("token", None)
            kwargs["use_auth_token"] = token
        return load_dataset(**kwargs)


def infer_columns(dataset: Dataset) -> DatasetColumns:
    names = set(dataset.column_names)
    image = _first_present(names, ["image", "img", "jpg", "png"])
    label = _first_present(names, ["label", "labels", "class", "target", "fine_label"], required=False)
    corruption = _first_present(
        names,
        ["corruption", "corruption_type", "distortion", "noise_type", "domain"],
        required=False,
    )
    severity = _first_present(names, ["severity", "level", "corruption_severity"], required=False)
    return DatasetColumns(image=image, label=label, corruption=corruption, severity=severity)


def filter_corruption_dataset(
    dataset: Dataset,
    columns: DatasetColumns,
    corruptions: Iterable[str] | None,
    severities: Iterable[int] | None,
    max_samples: int | None,
    seed: int = 0,
    label_mapping: LabelMapping | None = None,
) -> Dataset:
    selected = dataset
    corruption_set = {str(item) for item in corruptions or []}
    severity_set = {int(item) for item in severities or []}

    if corruption_set and columns.corruption:
        selected = selected.filter(lambda row: str(row[columns.corruption]) in corruption_set)
    if severity_set and columns.severity:
        selected = selected.filter(lambda row: int(row[columns.severity]) in severity_set)
    if max_samples is not None and max_samples > 0:
        selected = select_class_balanced_subset(selected, columns, max_samples, seed=seed, label_mapping=label_mapping)
    return selected


def select_class_balanced_subset(
    dataset: Dataset,
    columns: DatasetColumns,
    max_samples: int | None,
    seed: int = 0,
    label_mapping: LabelMapping | None = None,
) -> Dataset:
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    if columns.label is None:
        return dataset.shuffle(seed=seed).select(range(min(max_samples, len(dataset))))

    labels = dataset[columns.label]
    by_label: dict[int, list[int]] = defaultdict(list)
    for idx, raw_label in enumerate(labels):
        label = int(raw_label)
        if label_mapping is not None:
            label = label_mapping.map(label)
        by_label[label].append(idx)

    rng = random.Random(seed)
    for indices in by_label.values():
        rng.shuffle(indices)

    selected: list[int] = []
    class_ids = sorted(by_label)
    while len(selected) < max_samples and class_ids:
        next_class_ids = []
        for class_id in class_ids:
            indices = by_label[class_id]
            if indices:
                selected.append(indices.pop())
                if len(selected) >= max_samples:
                    break
            if indices:
                next_class_ids.append(class_id)
        class_ids = next_class_ids

    rng.shuffle(selected)
    return dataset.select(selected)


def build_timm_label_mapping(dataset: Dataset, columns: DatasetColumns, strict: bool = True) -> LabelMapping:
    """Map HF ClassLabel ids to timm ImageNet-1K classifier ids.

    HF ImageNet-style datasets are not guaranteed to document whether their
    integer labels follow timm's WNID order. This function inspects the
    `ClassLabel.names` metadata and returns an identity mapping when already
    aligned, or a per-label remapping when names are WNIDs/descriptions.
    """

    if columns.label is None:
        return LabelMapping(values=None, source="no label column", is_identity=True)

    feature = dataset.features.get(columns.label)
    names = getattr(feature, "names", None)
    if not names:
        message = f"Label column {columns.label!r} has no ClassLabel names; cannot verify timm alignment."
        if strict:
            raise ValueError(message)
        return LabelMapping(values=None, source=message, is_identity=True)

    info = ImageNetInfo("imagenet-1k")
    timm_wnids = list(info.label_names())
    num_classes = info.num_classes() if callable(info.num_classes) else int(info.num_classes)
    timm_descriptions = [info.index_to_description(i) for i in range(num_classes)]

    if list(names) == timm_wnids:
        return LabelMapping(values=None, source="HF WNID order matches timm ImageNet-1K", is_identity=True)
    if _normalize_imagenet_descriptions(names) == _normalize_imagenet_descriptions(timm_descriptions):
        return LabelMapping(values=None, source="HF description order matches timm ImageNet-1K", is_identity=True)

    if len(names) == len(timm_wnids) and all(isinstance(name, str) and name.startswith("n") for name in names):
        wnid_to_timm = {wnid: idx for idx, wnid in enumerate(timm_wnids)}
        try:
            values = [wnid_to_timm[str(name)] for name in names]
        except KeyError as exc:
            raise ValueError(f"HF label WNID {exc} is not present in timm ImageNet-1K synsets.") from exc
        return LabelMapping(values=values, source="HF WNID order remapped to timm ImageNet-1K", is_identity=values == list(range(len(values))))

    if len(names) == len(timm_descriptions):
        desc_to_timm = {desc: idx for idx, desc in enumerate(_normalize_imagenet_descriptions(timm_descriptions))}
        normalized_names = _normalize_imagenet_descriptions(names)
        if len(desc_to_timm) == len(timm_descriptions) and all(name in desc_to_timm for name in normalized_names):
            values = [desc_to_timm[name] for name in normalized_names]
            return LabelMapping(
                values=values,
                source="HF description order remapped to timm ImageNet-1K",
                is_identity=values == list(range(len(values))),
            )

    message = (
        f"Could not verify HF label alignment with timm ImageNet-1K for label column {columns.label!r}. "
        f"First labels: {list(names)[:5]}"
    )
    if strict:
        raise ValueError(message)
    return LabelMapping(values=None, source=message, is_identity=True)


def _normalize_imagenet_descriptions(names: Iterable[str]) -> list[str]:
    normalized = []
    for name in names:
        value = str(name).strip()
        if value == "crane2":
            value = "crane"
        normalized.append(value)
    return normalized


class HFDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Dataset, transform, columns: DatasetColumns, label_mapping: LabelMapping | None = None) -> None:
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
        label = self.label_mapping.map(raw_label) if self.label_mapping is not None and raw_label >= 0 else raw_label
        meta = {
            "index": int(idx),
            "label": label,
            "raw_label": raw_label,
            "corruption": str(row[self.columns.corruption]) if self.columns.corruption else "",
            "severity": int(row[self.columns.severity]) if self.columns.severity else -1,
        }
        return self.transform(image), label, meta


def make_loader(
    dataset: Dataset,
    transform,
    columns: DatasetColumns,
    batch_size: int,
    num_workers: int,
    label_mapping: LabelMapping | None = None,
) -> DataLoader:
    return DataLoader(
        HFDataset(dataset, transform, columns, label_mapping=label_mapping),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )


def make_torch_loader(dataset: torch.utils.data.Dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )


class ConcatMetaDataset(torch.utils.data.Dataset):
    def __init__(self, datasets: list[torch.utils.data.Dataset]) -> None:
        self.datasets = datasets
        self.cumulative_sizes = []
        total = 0
        for dataset in datasets:
            total += len(dataset)
            self.cumulative_sizes.append(total)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx: int):
        if idx < 0:
            idx += len(self)
        for dataset_idx, cumulative_size in enumerate(self.cumulative_sizes):
            previous = 0 if dataset_idx == 0 else self.cumulative_sizes[dataset_idx - 1]
            if idx < cumulative_size:
                return self.datasets[dataset_idx][idx - previous]
        raise IndexError(idx)


class FixedMetaDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, corruption: str, severity: int, domain_index: int) -> None:
        self.dataset = dataset
        self.corruption = str(corruption)
        self.severity = int(severity)
        self.domain_index = int(domain_index)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, label, meta = self.dataset[idx]
        meta = dict(meta)
        meta["corruption"] = self.corruption
        meta["severity"] = self.severity
        meta["domain_index"] = self.domain_index
        meta["domain_sample_index"] = int(idx)
        return image, label, meta


class ImageNetCSegmentDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str | Path,
        corruption: str,
        severity: int,
        transform,
        max_samples: int | None = None,
        seed: int = 0,
        domain_index: int = 0,
    ) -> None:
        self.root = Path(root)
        self.corruption = str(corruption)
        self.severity = int(severity)
        self.transform = transform
        self.domain_index = int(domain_index)
        self.label_mapping_source = "local ImageNet-C WNID folder -> timm ImageNet-1K index"
        segment_root = self.root / self.corruption / str(self.severity)
        if not segment_root.exists():
            raise FileNotFoundError(
                f"ImageNet-C segment not found: {segment_root}. "
                "Expected ReservoirTTA/RobustBench layout root/corruption/severity/class/image."
            )

        info = ImageNetInfo("imagenet-1k")
        wnid_to_timm = {wnid: idx for idx, wnid in enumerate(info.label_names())}
        by_label: dict[int, list[Path]] = defaultdict(list)
        for class_dir in sorted(path for path in segment_root.iterdir() if path.is_dir()):
            class_name = class_dir.name
            if class_name in wnid_to_timm:
                label = wnid_to_timm[class_name]
            else:
                try:
                    label = int(class_name)
                except ValueError as exc:
                    raise ValueError(f"Cannot map ImageNet-C class folder {class_dir} to timm ImageNet index.") from exc
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    by_label[label].append(path)

        original_class_counts = {int(label): len(paths) for label, paths in by_label.items()}
        pairs: list[tuple[Path, int]] = []
        if max_samples is not None and max_samples > 0:
            rng = random.Random(seed)
            for paths in by_label.values():
                rng.shuffle(paths)
            class_ids = sorted(by_label)
            while len(pairs) < max_samples and class_ids:
                next_class_ids = []
                for class_id in class_ids:
                    paths = by_label[class_id]
                    if paths:
                        pairs.append((paths.pop(), class_id))
                        if len(pairs) >= max_samples:
                            break
                    if paths:
                        next_class_ids.append(class_id)
                class_ids = next_class_ids
            rng.shuffle(pairs)
        else:
            for class_id in sorted(by_label):
                pairs.extend((path, class_id) for path in by_label[class_id])

        if not pairs:
            raise RuntimeError(f"No images found under {segment_root}")
        self.samples = pairs
        self.class_counts = original_class_counts
        selected_counts: dict[int, int] = defaultdict(int)
        for _path, label in self.samples:
            selected_counts[int(label)] += 1
        self.selected_class_counts = dict(selected_counts)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[int(idx)]
        image = Image.open(path).convert("RGB")
        meta = {
            "index": int(idx),
            "label": int(label),
            "raw_label": int(label),
            "corruption": self.corruption,
            "severity": self.severity,
            "domain_index": self.domain_index,
            "domain_sample_index": int(idx),
            "path": str(path),
        }
        return self.transform(image), int(label), meta


def build_reservoirtta_imagenet_c_dataset(
    transform,
    root: str | Path | None = None,
    hf_dataset: Dataset | None = None,
    hf_columns: DatasetColumns | None = None,
    label_mapping: LabelMapping | None = None,
    corruptions: Iterable[str] | None = None,
    severities: Iterable[int] | None = None,
    examples_per_domain: int | None = 5000,
    seed: int = 0,
    max_domains: int | None = None,
) -> tuple[torch.utils.data.Dataset, list[dict[str, Any]]]:
    root_path = resolve_imagenet_c_root(root) if root else None
    use_local = root_path is not None and root_path.exists()
    use_hf_domains = hf_dataset is not None and hf_columns is not None and hf_columns.corruption and hf_columns.severity
    if not use_local and not use_hf_domains:
        raise ValueError(
            "ReservoirTTA-style ImageNet-C requires domain metadata. "
            "Provide dataset.imagenet_c_root with local ImageNet-C layout "
            "root/corruption/severity/class/image, or use an HF dataset with corruption and severity columns. "
            "The current HF mirror has only image/label columns, so real corruption/severity segments cannot be recovered."
        )

    datasets: list[torch.utils.data.Dataset] = []
    segments: list[dict[str, Any]] = []
    domain_index = 0
    if use_local:
        assert root_path is not None
        domain_specs = discover_imagenet_c_domains(root_path, corruptions=corruptions, severities=severities)
        if max_domains is not None:
            domain_specs = domain_specs[: int(max_domains)]
        if not domain_specs:
            raise RuntimeError(
                f"No ImageNet-C domains found under {root_path}. "
                "Expected root/corruption/severity/class/image."
            )
        for spec in domain_specs:
            corruption = str(spec["corruption"])
            severity = int(spec["severity"])
            try:
                segment = ImageNetCSegmentDataset(
                    root=root_path,
                    corruption=corruption,
                    severity=severity,
                    transform=transform,
                    max_samples=examples_per_domain,
                    seed=seed + domain_index,
                    domain_index=domain_index,
                )
            except FileNotFoundError:
                continue

            datasets.append(segment)
            segments.append(
                {
                    "domain_index": domain_index,
                    "corruption": corruption,
                    "severity": severity,
                    "num_samples": len(segment),
                    "num_classes": len(getattr(segment, "selected_class_counts", {})),
                    "selected_min_per_class": min(getattr(segment, "selected_class_counts", {0: 0}).values()),
                    "selected_max_per_class": max(getattr(segment, "selected_class_counts", {0: 0}).values()),
                    "label_mapping": getattr(segment, "label_mapping_source", "unknown"),
                }
            )
            domain_index += 1
    else:
        corruptions = [str(item) for item in (corruptions or RESERVOIRTTA_CORRUPTIONS)]
        severities = [int(item) for item in (severities or RESERVOIRTTA_SEVERITIES)]
        for corruption in corruptions:
            for severity in severities:
                if max_domains is not None and domain_index >= int(max_domains):
                    break
                assert hf_dataset is not None and hf_columns is not None
                selected = hf_dataset.filter(
                    lambda row, corruption=corruption, severity=severity: str(row[hf_columns.corruption]) == corruption
                    and int(row[hf_columns.severity]) == severity
                )
                selected = select_class_balanced_subset(
                    selected,
                    hf_columns,
                    examples_per_domain,
                    seed=seed + domain_index,
                    label_mapping=label_mapping,
                )
                segment = FixedMetaDataset(
                    HFDataset(selected, transform, hf_columns, label_mapping=label_mapping),
                    corruption=corruption,
                    severity=severity,
                    domain_index=domain_index,
                )

                datasets.append(segment)
                segments.append(
                    {
                        "domain_index": domain_index,
                        "corruption": corruption,
                        "severity": severity,
                        "num_samples": len(segment),
                        "num_classes": len(getattr(segment, "selected_class_counts", {})),
                        "selected_min_per_class": min(getattr(segment, "selected_class_counts", {0: 0}).values()),
                        "selected_max_per_class": max(getattr(segment, "selected_class_counts", {0: 0}).values()),
                        "label_mapping": getattr(
                            segment,
                            "label_mapping_source",
                            label_mapping.source if label_mapping is not None else "unknown",
                        ),
                    }
                )
                domain_index += 1
            if max_domains is not None and domain_index >= int(max_domains):
                break

    if not datasets:
        raise RuntimeError("No ImageNet-C samples were loaded.")

    return ConcatMetaDataset(datasets), segments


def resolve_imagenet_c_root(root: str | Path | None) -> Path | None:
    if root is None:
        return None
    root_path = Path(root).expanduser()
    candidates = [root_path]
    if not root_path.is_absolute():
        candidates.append(PROJECT_ROOT / root_path)
    expanded: list[Path] = []
    for candidate in candidates:
        expanded.extend([candidate, candidate / "ImageNet-C", candidate / "imagenet-c"])
    for candidate in expanded:
        if count_corruption_dirs(candidate) > 0:
            return candidate
    return candidates[0]


def collate_batch(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    meta = {
        "index": [item[2]["index"] for item in batch],
        "label": [item[2]["label"] for item in batch],
        "raw_label": [item[2]["raw_label"] for item in batch],
        "corruption": [item[2]["corruption"] for item in batch],
        "severity": [item[2]["severity"] for item in batch],
        "domain_index": [item[2].get("domain_index", -1) for item in batch],
        "domain_sample_index": [item[2].get("domain_sample_index", -1) for item in batch],
    }
    if any("path" in item[2] for item in batch):
        meta["path"] = [item[2].get("path", "") for item in batch]
    return images, labels, meta


def _first_present(names: set[str], candidates: list[str], required: bool = True) -> str | None:
    for candidate in candidates:
        if candidate in names:
            return candidate
    if required:
        raise KeyError(f"Could not infer required dataset column from candidates={candidates}; columns={sorted(names)}")
    return None
