"""T0.1 계약 검증: 각 데이터셋 loader가 배치를 반환하고, 미보유 데이터셋은 skip 처리된다.

`data/imagenet-c`와 `data/imagenet-9`는 loader가 요구하는 추가 인자(corruption/severity,
variant)가 있어 phase0.md의 pseudocode보다 조건이 하나씩 더 붙는다 — AVAILABLE 전체를
같은 시그니처로 스윕할 수 없는 두 데이터셋만 별도 kwargs를 준다.
"""

import cv2
import numpy as np
import pytest
from torch.utils.data import DataLoader

from Utils.datasets import AVAILABLE, DatasetUnavailableError, build_dataset

BATCH_SIZE = 4


def _resize_224(image):
    """자연 이미지 데이터셋(원본 크기가 샘플마다 다름)을 224x224로 맞춰 배치가 stack되게 한다.
    colored-mnist는 이미 CHW 텐서라 이 transform을 쓰지 않는다."""
    arr = np.asarray(image)
    return cv2.resize(arr, (224, 224))


# name -> build_dataset에 추가로 넘길 kwargs. 없으면 기본 인자(split="test")로 연다.
# imagenet/waterbirds/imagenet-r/imagenet-a는 샘플마다 원본 해상도가 달라 배치 stack이
# 실패하므로, DoD 스모크 테스트 목적으로 고정 크기 transform을 준다(정식 학습 파이프라인의
# transform과는 무관 — 실험 코드는 자신의 transform을 transform= 인자로 직접 넘긴다).
_RESIZE_NEEDED = {"imagenet", "waterbirds", "imagenet-r", "imagenet-a", "imagenet-sketch"}
_EXTRA_KWARGS = {
    "imagenet-c": {"split": "report", "corruption": "gaussian_noise", "severity": 1},
    "imagenet-9": {"split": "original"},
}


def _open_dataset(name: str):
    """name에 맞는 kwargs로 build_dataset을 호출한다."""
    kwargs = dict(_EXTRA_KWARGS.get(name, {}))
    if name in _RESIZE_NEEDED:
        kwargs["transform"] = _resize_224
    return build_dataset(name, **kwargs)


@pytest.mark.parametrize("name", AVAILABLE)
def test_build_dataset_returns_batch(name):
    """각 데이터셋이 (Dataset, DatasetMeta)를 돌려주고, 배치 하나가 배치 크기만큼 나온다.
    디스크에 없는 데이터셋(예: imagenet-sketch)은 실패가 아니라 skip한다."""
    try:
        ds, meta = _open_dataset(name)
    except DatasetUnavailableError as e:
        pytest.skip(f"{name}: 데이터 없음 ({e})")
        return

    assert meta.name == name
    assert meta.num_samples == len(ds)
    assert meta.num_classes > 0
    assert meta.source

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    batch = next(iter(loader))

    image_batch = batch[0]
    assert image_batch.shape[0] == BATCH_SIZE

    if meta.has_group_labels:
        assert len(batch) == 3
    else:
        assert len(batch) == 2

    print(
        f"{name}: n={meta.num_samples} classes={meta.num_classes} "
        f"groups={meta.num_groups} src={meta.source}"
    )


def test_unknown_dataset_name_raises_value_error():
    """AVAILABLE에 없는 이름은 ValueError."""
    with pytest.raises(ValueError):
        build_dataset("not-a-real-dataset")


def test_corruption_kwargs_rejected_for_non_imagenet_c():
    """corruption/severity는 imagenet-c 전용 — 다른 데이터셋에 주면 ValueError."""
    with pytest.raises(ValueError):
        build_dataset("waterbirds", corruption="gaussian_noise")


def test_imagenet_c_requires_corruption_and_severity():
    """imagenet-c는 corruption/severity가 둘 다 있어야 한다."""
    with pytest.raises(ValueError):
        build_dataset("imagenet-c", split="report")


def test_imagenet_c_split_enforces_hp_report_partition():
    """split="hp"에 report 전용 corruption을 주면 ValueError (그 반대도 마찬가지)."""
    with pytest.raises(ValueError):
        build_dataset("imagenet-c", split="hp", corruption="gaussian_noise", severity=1)
    with pytest.raises(ValueError):
        build_dataset("imagenet-c", split="report", corruption="gaussian_blur", severity=1)


def test_waterbirds_split_filters_metadata():
    """waterbirds는 split별로 다른 샘플 수를 준다 (train=4795/val=1199/test=5794)."""
    _, meta_train = build_dataset("waterbirds", split="train")
    _, meta_val = build_dataset("waterbirds", split="val")
    _, meta_test = build_dataset("waterbirds", split="test")
    assert meta_train.num_samples == 4795
    assert meta_val.num_samples == 1199
    assert meta_test.num_samples == 5794


def test_imagenet_9_rejects_fg_mask_split():
    """fg_mask는 마스크지 이미지가 아니므로 분류용 split으로 거부된다."""
    with pytest.raises(ValueError):
        build_dataset("imagenet-9", split="fg_mask")
