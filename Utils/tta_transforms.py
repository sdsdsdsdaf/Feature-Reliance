"""M4: TTA용 weak/strong 증강 쌍. `Utils/transfrom.py`(perturbation 실험용)와는 무관한 새 모듈.

정규화 상수는 하드코딩하지 않고 timm의 `resolve_model_data_config`에서 매 호출마다 가져온다
(백본을 바꾸면 상수도 따라 바뀌어야 하므로 모듈 로드 시 캐시하지 않는다).
"""

from __future__ import annotations

from typing import Callable

import timm
from timm.data import resolve_model_data_config
from torch import Tensor
from torchvision import transforms

_BACKBONE_NAME = "vit_base_patch16_224.augreg_in1k"


def _resolve_normalization() -> tuple[tuple[float, ...], tuple[float, ...]]:
    """timm에서 `_BACKBONE_NAME`의 정규화 mean/std를 읽어온다. 하드코딩 금지 조항의 유일한 통로."""
    model = timm.create_model(_BACKBONE_NAME, pretrained=True)
    data_cfg = resolve_model_data_config(model)
    return tuple(data_cfg["mean"]), tuple(data_cfg["std"])


def weak_transform(size: int = 224) -> Callable:
    """가볍게만 흔든 증강. 원본에 가까워 pseudo-label을 뽑는 쪽에 쓴다."""
    mean, std = _resolve_normalization()
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def strong_transform(size: int = 224) -> Callable:
    """색·흐림까지 세게 흔든 증강. 이걸 보고도 같은 답을 내게 학습시키는 쪽에 쓴다."""
    mean, std = _resolve_normalization()
    # kernel_size는 size에 비례하는 홀수(약 10%)로 잡는다. size=224일 때 통상값 23과 같다.
    kernel_size = max(3, int(size * 0.1))
    if kernel_size % 2 == 0:
        kernel_size += 1
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


class TwoCropTransform:
    """weak/strong 두 변환을 한 쌍으로 묶어 하나의 이미지에서 두 뷰를 뽑는 transform."""

    def __init__(self, weak: Callable, strong: Callable):
        """weak/strong 두 변환을 한 쌍으로 묶는다. 보관은 두 Callable뿐."""
        self.weak = weak
        self.strong = strong

    def __call__(self, img) -> tuple[Tensor, Tensor]:
        """같은 이미지 하나에서 (weak, strong) 두 뷰를 만들어 반환한다.
        DataLoader가 이걸 transform으로 받으면 배치가 자동으로 두 뷰 쌍이 된다."""
        return self.weak(img), self.strong(img)
