from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torchvision import transforms
from torchvision.models import VGG19_Weights, vgg19


def _log_var(features: torch.Tensor) -> torch.Tensor:
    var = torch.var(features, dim=(0, 2, 3)).clamp_min(1e-8)
    return torch.log(var)[None, :]


def _mean(features: torch.Tensor) -> torch.Tensor:
    return features.mean(dim=(0, 2, 3), keepdim=False)[None, :]


def _var(features: torch.Tensor) -> torch.Tensor:
    return torch.var(features, dim=(0, 2, 3)).clamp_min(1e-8)[None, :]


def _mean_var(features: torch.Tensor) -> torch.Tensor:
    mean = features.mean(dim=(2, 3))
    variance = features.var(dim=(2, 3)).clamp_min(1e-8)
    return torch.cat([mean, torch.log(variance)], dim=1).mean(dim=0, keepdim=True)


def _gram_diag(features: torch.Tensor) -> torch.Tensor:
    bsz, channels, height, width = features.shape
    flat = features.view(bsz, channels, height * width)
    gram = torch.bmm(flat, flat.transpose(1, 2))
    return gram.diagonal(dim1=-2, dim2=-1).mean(dim=0, keepdim=True)


STYLE_FUNCTIONS = {
    "LOGVAR": _log_var,
    "MEAN": _mean,
    "VAR": _var,
    "MEAN_VAR": _mean_var,
    "GRAM": _gram_diag,
}


class VGGStyleVecDescriptor(nn.Module):
    """ReservoirTTA-compatible VGG StyleVec descriptor.

    The original ReservoirTTA implementation returns one descriptor per batch.
    This adapter keeps that behavior so StyleVec and SAE routing are compared
    at the same batch-routing granularity.
    """

    def __init__(
        self,
        style_idx: Iterable[int] = (2, 5, 7),
        style_format: str = "LOGVAR",
    ) -> None:
        super().__init__()
        self.style_idx = [int(idx) for idx in style_idx]
        self.style_format = style_format.upper()
        if self.style_format not in STYLE_FUNCTIONS:
            raise ValueError(f"Unknown style_format={style_format!r}.")
        self.produce_style = STYLE_FUNCTIONS[self.style_format]

        max_style_idx = max(self.style_idx)
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[: max_style_idx + 1].eval()
        vgg.requires_grad_(False)

        layers = []
        for layer in vgg.children():
            if isinstance(layer, nn.ReLU):
                layer = nn.ReLU(inplace=False)
            layers.append(layer)
        self.vgg = nn.Sequential(*layers)

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.features: dict[int, torch.Tensor] = {}
        for idx in self.style_idx:
            self.vgg[idx].register_forward_hook(self._make_hook(idx))

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.features.clear()
        self.vgg(self.normalize(images))
        descriptors = [self.produce_style(self.features[idx]) for idx in self.style_idx]
        return torch.cat(descriptors, dim=1)

    def _make_hook(self, idx: int):
        def hook(_module, _args, output):
            self.features[idx] = output.detach()

        return hook

