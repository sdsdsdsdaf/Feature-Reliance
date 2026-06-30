from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from datasets import Dataset, load_dataset
from PIL import Image
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class DatasetColumns:
    image: str
    label: str | None
    corruption: str | None
    severity: str | None


def load_hf_split(
    dataset_name: str,
    split: str,
    cache_dir: str | None = None,
    config: str | None = None,
    token: str | None = None,
) -> Dataset:
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
) -> Dataset:
    selected = dataset
    corruption_set = {str(item) for item in corruptions or []}
    severity_set = {int(item) for item in severities or []}

    if corruption_set and columns.corruption:
        selected = selected.filter(lambda row: str(row[columns.corruption]) in corruption_set)
    if severity_set and columns.severity:
        selected = selected.filter(lambda row: int(row[columns.severity]) in severity_set)
    if max_samples is not None and max_samples > 0:
        selected = selected.select(range(min(max_samples, len(selected))))
    return selected


class HFDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Dataset, transform, columns: DatasetColumns) -> None:
        self.dataset = dataset
        self.transform = transform
        self.columns = columns

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        row = self.dataset[int(idx)]
        image = row[self.columns.image]
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")
        label = int(row[self.columns.label]) if self.columns.label else -1
        meta = {
            "index": int(idx),
            "label": label,
            "corruption": str(row[self.columns.corruption]) if self.columns.corruption else "",
            "severity": int(row[self.columns.severity]) if self.columns.severity else -1,
        }
        return self.transform(image), label, meta


def make_loader(dataset: Dataset, transform, columns: DatasetColumns, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        HFDataset(dataset, transform, columns),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )


def collate_batch(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    meta = {
        "index": [item[2]["index"] for item in batch],
        "label": [item[2]["label"] for item in batch],
        "corruption": [item[2]["corruption"] for item in batch],
        "severity": [item[2]["severity"] for item in batch],
    }
    return images, labels, meta


def _first_present(names: set[str], candidates: list[str], required: bool = True) -> str | None:
    for candidate in candidates:
        if candidate in names:
            return candidate
    if required:
        raise KeyError(f"Could not infer required dataset column from candidates={candidates}; columns={sorted(names)}")
    return None

