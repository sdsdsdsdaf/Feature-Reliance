"""ImageNet-R / ImageNet-A의 wnid -> ImageNet-1k(1000-way) 클래스 인덱스 매핑.

ImageNet-R/A는 원본 ImageNet-1k의 200-class 부분집합이다. 사전학습된 1000-way
classifier의 logits을 이 데이터셋들에 그대로 평가하려면, 그 logits 중 이
200개 클래스에 해당하는 열만 남기고 나머지를 마스킹해야 정확도가 맞는다.
이 파일은 그 매핑을 디스크의 실제 wnid 디렉토리 목록에서 유도해 상수로
고정해 둔다 — 배포 tarball이 자기 이름의 디렉토리를 한 겹 더 품고 있는
`<root>/<name>/<name>/` 중첩도 여기서 흡수한다(sketch가 나중에 다른 구조로
들어올 수 있으므로 루트를 하드코딩하지 않는다).
"""

from __future__ import annotations

from pathlib import Path

from timm.data.imagenet_info import ImageNetInfo

_DEFAULT_ROOT = Path("data")


def _wnid_to_1k_index() -> dict[str, int]:
    """timm이 갖고 있는 표준 ImageNet-1k wnid 순서에서 wnid -> 0..999 인덱스 dict를 만든다."""
    info = ImageNetInfo("imagenet-1k")
    wnids = info.label_names()
    return {wnid: idx for idx, wnid in enumerate(wnids)}


WNID_TO_IMAGENET_INDEX: dict[str, int] = _wnid_to_1k_index()


def resolve_nested_dir(root: Path, name: str) -> Path:
    """<root>/<name> 아래에 <name>이 한 번 더 중첩돼 있으면 한 겹 내려간 경로를 돌려준다.
    (예: data/imagenet-r/imagenet-r/<wnid>/ — 배포 tarball이 자기 이름 디렉토리를 품고 있다.)
    중첩이 없으면 <root>/<name>을 그대로 돌려준다."""
    base = root / name
    nested = base / name
    return nested if nested.is_dir() else base


def _list_wnid_dirs(folder: Path) -> list[str]:
    """folder 아래에서 wnid(nXXXXXXXX) 형태의 디렉토리 이름만 정렬해서 돌려준다. folder가 없으면 빈 리스트."""
    if not folder.is_dir():
        return []
    return sorted(p.name for p in folder.iterdir() if p.is_dir() and p.name.startswith("n"))


def class_ids_for(root: Path, name: str) -> tuple[list[str], list[int]]:
    """<root>/<name>(중첩 자동 해소) 아래 wnid 디렉토리 목록과, 그걸 ImageNet-1k
    인덱스로 매핑한 리스트를 같은 순서로 함께 돌려준다. 데이터가 없으면 ([], [])."""
    folder = resolve_nested_dir(root, name)
    wnids = _list_wnid_dirs(folder)
    class_ids = [WNID_TO_IMAGENET_INDEX[w] for w in wnids if w in WNID_TO_IMAGENET_INDEX]
    return wnids, class_ids


def get_subset_class_ids(root: str, name: str) -> tuple[list[str], list[int]]:
    """build_dataset()이 root를 커스텀으로 받았을 때, 그 root 기준으로 wnid/class_id 매핑을 다시 계산한다."""
    return class_ids_for(Path(root), name)


# 기본 data/ 루트 기준으로 미리 계산해 둔 상수 — import 시점에 디스크를 한 번만 읽는다.
# 데이터가 아직 없어도(예: 준비 전) 예외를 던지지 않고 빈 리스트를 준다.
IMAGENET_R_WNIDS, IMAGENET_R_CLASS_IDS = class_ids_for(_DEFAULT_ROOT, "imagenet-r")
IMAGENET_A_WNIDS, IMAGENET_A_CLASS_IDS = class_ids_for(_DEFAULT_ROOT, "imagenet-a")
