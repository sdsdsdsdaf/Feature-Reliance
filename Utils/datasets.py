"""데이터셋 팩토리 — 실험 1~8이 쓰는 전 데이터셋을 이름 하나로 여는 단일 창구.

실험 코드가 데이터셋별 디렉토리 구조·라벨 파일 형식을 알 필요가 없게 만든다.
데이터셋별 실제 로딩 로직은 여기 모으고, 기존 `Utils/Dataset.py`의 재사용
가능한 조각(`ImageFolderDS`)은 새로 만들지 않고 그대로 가져다 쓴다.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from Utils.Dataset import ImageFolderDS
from Utils.imagenet_subsets import (
    IMAGENET_A_CLASS_IDS,
    IMAGENET_A_WNIDS,
    IMAGENET_R_CLASS_IDS,
    IMAGENET_R_WNIDS,
    WNID_TO_IMAGENET_INDEX,
    get_subset_class_ids,
    resolve_nested_dir,
)

AVAILABLE = [
    "imagenet",
    "imagenet-c",
    "waterbirds",
    "colored-mnist",
    "imagenet-r",
    "imagenet-a",
    "imagenet-sketch",
    "imagenet-9",
]

# ImageNet-C: HP 튜닝 전용 4종. 나머지 15종은 보고(report)용 — spec의 ImageNet-C 규약.
_IMAGENET_C_HP_CORRUPTIONS = {"gaussian_blur", "saturate", "spatter", "speckle_noise"}

# ImageNet-9(bg_challenge)의 variant 목록 중 fg_mask는 이미지가 아니라 마스크라 분류용에서 제외한다.
_IMAGENET_9_MASK_VARIANT = "fg_mask"

# 디스크의 ColoredMNIST 디렉토리는 언더스코어(colored_mnist)를 쓴다. name 인자는 하이픈(colored-mnist).
_COLORED_MNIST_DIRNAME = "colored_mnist"
_COLORED_MNIST_ENV_FILES = {"train1": "train1.pt", "train2": "train2.pt", "test": "test.pt"}

_WATERBIRDS_SPLIT_TO_CODE = {"train": 0, "val": 1, "test": 2}


class _HiddenSafeImageFolder(ImageFolder):
    """숨김 디렉토리를 클래스로 오인하지 않는 ImageFolder.

    이 환경에서는 툴/세션이 데이터 디렉토리 안에 `.claude/` 같은 빈 보조 디렉토리를
    만들 수 있고, 기본 ImageFolder는 그걸 이미지 0장짜리 클래스로 잡아
    FileNotFoundError를 던지거나(더 나쁘게) 클래스 인덱스를 한 칸씩 밀어버린다.
    점으로 시작하는 디렉토리를 클래스 후보에서 제외해 그 사고를 막는다."""

    def find_classes(self, directory: str):
        """디렉토리에서 클래스 목록을 뽑는다. 숨김(.으로 시작) 디렉토리는 무시한다.
        반환: (정렬된 클래스명 리스트, {클래스명: 인덱스})."""
        classes = sorted(e.name for e in os.scandir(directory) if e.is_dir() and not e.name.startswith("."))
        if not classes:
            raise FileNotFoundError(f"클래스 디렉토리를 찾을 수 없다: {directory}")
        return classes, {name: i for i, name in enumerate(classes)}


class _SafeImageFolderDS(ImageFolderDS):
    """`Utils.Dataset.ImageFolderDS`와 동일한 인터페이스(classes/class_to_idx/
    (arr, label) 반환)이되, 내부 ImageFolder만 숨김 안전 버전으로 바꾼 것.

    __init__: root(클래스 디렉토리들이 든 경로)와 transform(np.ndarray에 적용,
    없으면 원본 RGB np.ndarray)을 받는다. 상위 __init__은 ImageFolder를
    하드코딩하고 있어 호출하지 않고, 같은 속성들을 직접 채운다."""

    def __init__(self, root: str, transform: Optional[Callable] = None):
        Dataset.__init__(self)
        self.ds = _HiddenSafeImageFolder(root)
        self.transform = transform
        self.root = root
        self.classes = self.ds.classes
        self.class_to_idx = self.ds.class_to_idx


class DatasetUnavailableError(FileNotFoundError):
    """요청한 데이터셋이 디스크에 없을 때 던진다(예: imagenet-sketch).
    이건 구현 실패가 아니라 '아직 안 받았다'는 신호이므로, 호출부는 실패가
    아니라 skip으로 처리해야 한다."""


@dataclass
class DatasetMeta:
    """열린 데이터셋의 정체를 담은 카드. 실험 코드는 이걸 보고 분기한다."""

    name: str
    num_classes: int
    num_samples: int
    has_group_labels: bool  # True면 __getitem__이 (img, y, group) 3-튜플을 준다
    num_groups: Optional[int]  # worst-group acc를 몇 개 그룹으로 나눠 재는지
    group_names: Optional[list[str]]
    class_to_idx: dict[str, int]
    source: str  # 실제로 읽은 경로 — 어느 사본을 썼는지 결과에 남기기 위함


def build_dataset(
    name: str,
    split: str = "test",
    *,
    root: str = "data",
    transform: Optional[Callable] = None,
    corruption: Optional[str] = None,
    severity: Optional[int] = None,
) -> tuple[Dataset, DatasetMeta]:
    """이름 하나로 아무 데이터셋이나 열어주는 단일 창구.

    - name: AVAILABLE 목록 중 하나.
    - split: "train"/"val"/"test"가 기본. ImageNet-C는 "hp"/"report",
      ImageNet-9는 variant 이름(예: "mixed_rand")을 split 자리에 준다.
    - corruption/severity: ImageNet-C 전용. 다른 데이터셋에 주면 ValueError.
    - 데이터가 디스크에 없으면(예: imagenet-sketch) 실패가 아니라
      DatasetUnavailableError를 던진다 — 호출부가 skip으로 처리한다.

    반환: (Dataset, 그 데이터셋이 뭔지 설명하는 meta).
    """
    if name not in AVAILABLE:
        raise ValueError(f"알 수 없는 데이터셋 이름: {name!r}. 다음 중 하나여야 한다: {AVAILABLE}")

    if name != "imagenet-c" and (corruption is not None or severity is not None):
        raise ValueError("corruption/severity는 imagenet-c 전용이다")

    root_path = Path(root)

    if name == "imagenet":
        return _build_imagenet(root_path, split, transform)
    if name == "imagenet-c":
        return _build_imagenet_c(root_path, split, transform, corruption, severity)
    if name == "waterbirds":
        return _build_waterbirds(root_path, split, transform)
    if name == "colored-mnist":
        return _build_colored_mnist(root_path, split, transform)
    if name == "imagenet-r":
        return _build_imagenet_subset(root_path, "imagenet-r", transform)
    if name == "imagenet-a":
        return _build_imagenet_subset(root_path, "imagenet-a", transform)
    if name == "imagenet-sketch":
        return _build_imagenet_sketch(root_path, transform)
    if name == "imagenet-9":
        return _build_imagenet9(root_path, split, transform)

    raise AssertionError("도달 불가 — AVAILABLE 목록과 dispatch가 어긋났다")  # pragma: no cover


# =========================================================
# imagenet — HF datasets 캐시(data/hf_cache)에서 로드
# =========================================================


class _HFImageNetDataset(Dataset):
    """HF `ILSVRC/imagenet-1k` 캐시에서 로드한 (image, label) 데이터셋.

    __init__: 이미 로드된 `datasets.Dataset` 객체(hf_dataset)와, np.ndarray
    이미지에 적용할 transform(없으면 None)을 받아 그대로 들고 있는다."""

    def __init__(self, hf_dataset, transform: Optional[Callable] = None):
        self.ds = hf_dataset
        self.transform = transform
        self._label_cache: Optional[list[int]] = None

    @property
    def _labels(self) -> list[int]:
        """전체 라벨 리스트. Arrow 컬럼만 읽어서 이미지 디코드를 피한다.

        이게 없으면 라벨이 필요한 쪽(예: Utils.SAE_plot_utils._dataset_labels)이
        `[ds[i][1] for i in range(len(ds))]`로 떨어져 전체 이미지를 디코드한다 —
        ImageNet 규모에서는 사실상 끝나지 않는다."""
        if self._label_cache is None:
            self._label_cache = [int(v) for v in self.ds["label"]]
        return self._label_cache

    def __len__(self) -> int:
        """전체 샘플 수."""
        return len(self.ds)

    def __getitem__(self, idx: int):
        """idx번째 (image, label). transform이 있으면 RGB np.ndarray에 적용한 결과를 이미지로 준다."""
        item = self.ds[idx]
        image = item["image"].convert("RGB")
        image = np.array(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, int(item["label"])


def _build_imagenet(root: Path, split: str, transform: Optional[Callable]):
    """ImageNet-1k을 HF datasets 캐시(root/hf_cache)에서 연다. split="train"이면 train split, 그 외("test"/"val")는 validation split."""
    hf_cache_dir = root / "hf_cache"
    if not hf_cache_dir.is_dir():
        raise DatasetUnavailableError(str(hf_cache_dir))

    from datasets import load_dataset  # 지연 import — HF datasets는 imagenet 로더 경로에서만 필요

    hf_split = "train" if split == "train" else "validation"
    hf_dataset = load_dataset("ILSVRC/imagenet-1k", split=hf_split, cache_dir=str(hf_cache_dir))

    ds = _HFImageNetDataset(hf_dataset, transform)
    meta = DatasetMeta(
        name="imagenet",
        num_classes=1000,
        num_samples=len(hf_dataset),
        has_group_labels=False,
        num_groups=None,
        group_names=None,
        class_to_idx=dict(WNID_TO_IMAGENET_INDEX),
        source=f"{hf_cache_dir}:{hf_split}",
    )
    return ds, meta


# =========================================================
# imagenet-c — data/imagenet-c/<corruption>/<severity>/<wnid>/*.JPEG
# =========================================================


def _build_imagenet_c(
    root: Path,
    split: str,
    transform: Optional[Callable],
    corruption: Optional[str],
    severity: Optional[int],
):
    """ImageNet-C를 corruption/severity 폴더에서 연다. split이 "hp"/"report" 구분을 강제한다
    (HP 튜닝은 4종 extra에서만, 보고는 15종 test에서 — spec 규약)."""
    if corruption is None or severity is None:
        raise ValueError("imagenet-c는 corruption과 severity가 모두 필요하다")
    if split not in ("hp", "report"):
        raise ValueError('imagenet-c의 split은 "hp" 또는 "report"만 허용된다')

    is_hp_corruption = corruption in _IMAGENET_C_HP_CORRUPTIONS
    if split == "hp" and not is_hp_corruption:
        raise ValueError(f"corruption={corruption!r}는 hp(HP 튜닝용 4종) 목록에 없다: {_IMAGENET_C_HP_CORRUPTIONS}")
    if split == "report" and is_hp_corruption:
        raise ValueError(f"corruption={corruption!r}는 hp 전용이라 report(보고용 15종)에서 쓸 수 없다")

    folder = root / "imagenet-c" / corruption / str(severity)
    if not folder.is_dir():
        raise DatasetUnavailableError(str(folder))

    ds = _SafeImageFolderDS(root=str(folder), transform=transform)
    meta = DatasetMeta(
        name="imagenet-c",
        num_classes=len(ds.classes),
        num_samples=len(ds),
        has_group_labels=False,
        num_groups=None,
        group_names=None,
        class_to_idx=ds.class_to_idx,
        source=str(folder),
    )
    return ds, meta


# =========================================================
# waterbirds — data/waterbirds/waterbird_complete95_forest2water2/
# =========================================================


class WaterbirdsDataset(Dataset):
    """Waterbirds metadata.csv를 split(train/val/test)으로 필터링해
    (image, y, group) 3-튜플을 주는 데이터셋. group = 2*y + place.

    __init__: base_dir(이미지가 있는 루트 디렉토리), rows(이미 split으로
    필터링된 metadata.csv 행들의 리스트), transform(np.ndarray -> 임의 형식,
    없으면 원본 RGB np.ndarray 그대로)을 받아 보관한다."""

    def __init__(self, base_dir: Path, rows: list[dict], transform: Optional[Callable] = None):
        self.base_dir = base_dir
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        """필터링된 행 수(=샘플 수)."""
        return len(self.rows)

    def __getitem__(self, idx: int):
        """idx번째 (image, y, group). y는 0=landbird/1=waterbird, group=2*y+place(0~3)."""
        row = self.rows[idx]
        image = Image.open(self.base_dir / row["img_filename"]).convert("RGB")
        image = np.array(image)
        if self.transform is not None:
            image = self.transform(image)
        y = int(row["y"])
        place = int(row["place"])
        group = 2 * y + place
        return image, y, group


def _build_waterbirds(root: Path, split: str, transform: Optional[Callable]):
    """Waterbirds를 연다. split은 train(0)/val(1)/test(2) 중 하나로 metadata.csv를 필터링한다
    (T1.3이 정답 측정은 test에서, c_k^train은 train에서 재기 위해 이 필터가 필요하다)."""
    base = root / "waterbirds" / "waterbird_complete95_forest2water2"
    metadata_path = base / "metadata.csv"
    if not metadata_path.is_file():
        raise DatasetUnavailableError(str(metadata_path))
    if split not in _WATERBIRDS_SPLIT_TO_CODE:
        raise ValueError(f"waterbirds split은 train/val/test 중 하나여야 한다: {split!r}")

    split_code = _WATERBIRDS_SPLIT_TO_CODE[split]
    with open(metadata_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if int(row["split"]) == split_code]

    ds = WaterbirdsDataset(base, rows, transform)
    meta = DatasetMeta(
        name="waterbirds",
        num_classes=2,
        num_samples=len(rows),
        has_group_labels=True,
        num_groups=4,
        group_names=["landbird_land", "landbird_water", "waterbird_land", "waterbird_water"],
        class_to_idx={"landbird": 0, "waterbird": 1},
        source=str(base),
    )
    return ds, meta


# =========================================================
# colored-mnist — data/colored_mnist/{train1,train2,test}.pt
# =========================================================


class ColoredMNISTDataset(Dataset):
    """ColoredMNIST 환경 하나(train1/train2/test)를 (image, y, group) 3-튜플로 주는 데이터셋.
    color는 파일에 별도 키가 없어 채널에서 복원한다: G채널에 픽셀이 있으면 color=1, 아니면 color=0.
    group = 2*y + color.

    __init__: images(uint8 [N,3,28,28] 텐서), labels(int64 [N] 텐서), transform
    (np.ndarray(H,W,C) -> 임의 형식에 적용, 없으면 원본 CHW uint8 텐서 그대로)을 받아
    color/group을 미리 계산해 들고 있는다."""

    def __init__(self, images, labels, transform: Optional[Callable] = None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.colors = (images[:, 1].sum(dim=(1, 2)) > 0).long()
        self.groups = 2 * labels.long() + self.colors

    def __len__(self) -> int:
        """샘플 수."""
        return int(self.images.shape[0])

    def __getitem__(self, idx: int):
        """idx번째 (image, y, group)."""
        image = self.images[idx]
        if self.transform is not None:
            arr = image.permute(1, 2, 0).numpy()
            image = self.transform(arr)
        y = int(self.labels[idx])
        group = int(self.groups[idx])
        return image, y, group


def _build_colored_mnist(root: Path, split: str, transform: Optional[Callable]):
    """ColoredMNIST를 연다. split은 환경 파일명과 같다: "train1"/"train2"/"test".
    편의상 "train"은 "train1"의 별칭이다(디폴트 트레인 환경 하나가 필요한 호출부용;
    두 트레인 환경을 다 쓰려면 "train1"/"train2"를 각각 명시해서 두 번 연다)."""
    import torch

    env = "train1" if split == "train" else split
    if env not in _COLORED_MNIST_ENV_FILES:
        raise ValueError(f'colored-mnist split은 "train"/"train1"/"train2"/"test" 중 하나여야 한다: {split!r}')

    base = root / _COLORED_MNIST_DIRNAME
    file_path = base / _COLORED_MNIST_ENV_FILES[env]
    if not file_path.is_file():
        raise DatasetUnavailableError(str(file_path))

    payload = torch.load(file_path)
    images = payload["images"]
    labels = payload["labels"]

    ds = ColoredMNISTDataset(images, labels, transform)
    meta = DatasetMeta(
        name="colored-mnist",
        num_classes=2,
        num_samples=len(ds),
        has_group_labels=True,
        num_groups=4,
        group_names=["y0_c0", "y0_c1", "y1_c0", "y1_c1"],
        class_to_idx={"digit0-4": 0, "digit5-9": 1},
        source=str(file_path),
    )
    return ds, meta


# =========================================================
# imagenet-r / imagenet-a — data/<name>/<name>/<wnid>/*.jpg (한 겹 중첩)
# =========================================================


def _build_imagenet_subset(root: Path, name: str, transform: Optional[Callable]):
    """ImageNet-R 또는 ImageNet-A를 연다. ImageFolder가 부여하는 로컬 인덱스(0..199)가
    __getitem__의 target이 되고, class_to_idx는 그 로컬 인덱스에 대응하는 wnid를
    ImageNet-1k 전체 인덱스(0..999)로 매핑한다 — 1000-way head 출력을 이 200개 열로
    마스킹할 때 class_to_idx.values() 순서(=정렬된 wnid 순서)를 그대로 쓰면 된다."""
    folder = resolve_nested_dir(root, name)
    if not folder.is_dir():
        raise DatasetUnavailableError(str(folder))

    if root == Path("data"):
        wnids = IMAGENET_R_WNIDS if name == "imagenet-r" else IMAGENET_A_WNIDS
        class_ids = IMAGENET_R_CLASS_IDS if name == "imagenet-r" else IMAGENET_A_CLASS_IDS
    else:
        wnids, class_ids = get_subset_class_ids(str(root), name)

    if not wnids:
        raise DatasetUnavailableError(str(folder))

    ds = _SafeImageFolderDS(root=str(folder), transform=transform)

    # __getitem__의 target은 ImageFolder가 매긴 로컬 인덱스(0..199)이고, 1000-way head를
    # 마스킹할 때 쓰는 열 순서는 class_ids다. 둘이 같은 순서를 가리켜야만 정확도가 맞는데,
    # 어긋나도 조용히 틀린 숫자가 나올 뿐 예외가 안 난다. 그래서 여기서 못 박는다.
    if list(ds.classes) != list(wnids):
        raise RuntimeError(
            f"{name}: ImageFolder 클래스 순서와 사전계산 wnid 순서가 다르다 — "
            f"이대로 두면 로짓 마스킹이 조용히 어긋난다 "
            f"(n_classes={len(ds.classes)}, n_wnids={len(wnids)})"
        )

    meta = DatasetMeta(
        name=name,
        num_classes=len(ds.classes),
        num_samples=len(ds),
        has_group_labels=False,
        num_groups=None,
        group_names=None,
        class_to_idx=dict(zip(wnids, class_ids)),
        source=str(folder),
    )
    return ds, meta


def _build_imagenet_sketch(root: Path, transform: Optional[Callable]):
    """ImageNet-Sketch를 연다. 현재 디스크에 없으므로(Google Drive 쿼터 실패) 항상
    DatasetUnavailableError를 던진다 — 실험 5의 natural regime에서만 쓰이므로 MVP를 막지 않는다."""
    folder = resolve_nested_dir(root, "imagenet-sketch")
    if not folder.is_dir() or not any(folder.iterdir()):
        raise DatasetUnavailableError(str(folder))
    ds = _SafeImageFolderDS(root=str(folder), transform=transform)
    meta = DatasetMeta(
        name="imagenet-sketch",
        num_classes=len(ds.classes),
        num_samples=len(ds),
        has_group_labels=False,
        num_groups=None,
        group_names=None,
        class_to_idx=ds.class_to_idx,
        source=str(folder),
    )
    return ds, meta


# =========================================================
# imagenet-9 — data/imagenet-9/bg_challenge/<variant>/val/<class>/*.JPEG
# =========================================================


def _build_imagenet9(root: Path, split: str, transform: Optional[Callable]):
    """ImageNet-9(BG Challenge)를 연다. split 자리에 variant 이름을 준다
    (예: "original", "mixed_rand", "only_fg", ...). "fg_mask"는 이미지가 아니라
    마스크라 분류용에서 거부한다."""
    if split == _IMAGENET_9_MASK_VARIANT:
        raise ValueError(f'"{_IMAGENET_9_MASK_VARIANT}"는 마스크 variant라 분류용 split으로 쓸 수 없다')

    folder = root / "imagenet-9" / "bg_challenge" / split / "val"
    if not folder.is_dir():
        raise DatasetUnavailableError(str(folder))

    ds = _SafeImageFolderDS(root=str(folder), transform=transform)
    meta = DatasetMeta(
        name="imagenet-9",
        num_classes=len(ds.classes),
        num_samples=len(ds),
        has_group_labels=False,
        num_groups=None,
        group_names=None,
        class_to_idx=ds.class_to_idx,
        source=str(folder),
    )
    return ds, meta
