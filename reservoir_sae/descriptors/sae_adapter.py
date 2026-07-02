from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torchvision import transforms
from torchvision.models import VGG19_Weights, vgg19

from Model.SAE import VanillaL1SAE


class VGGSAEDescriptor(nn.Module):
    """SAE latent routing descriptor on top of frozen VGG features.

    Checkpoint format expected by default:
        {
          "sae_state_dict": ...,
          "input_dim": int,
          "hidden_dim": int,
          "feature_mean": Tensor[input_dim], optional
          "feature_std": Tensor[input_dim], optional
        }
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        style_idx: Iterable[int] = (2, 5, 7),
        pooling: str = "mean",
        active_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"SAE checkpoint not found: {self.checkpoint_path}. "
                "Train/save a VGG-SAE checkpoint before running --routing sae."
            )

        self.style_idx = [int(idx) for idx in style_idx]
        self.pooling = pooling.lower()
        self.active_threshold = float(active_threshold)
        if self.pooling not in {"mean", "frequency", "max"}:
            raise ValueError("sae_pooling must be one of: mean, frequency, max")

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        input_dim = int(checkpoint["input_dim"])
        hidden_dim = int(checkpoint["hidden_dim"])
        self.sae = VanillaL1SAE(input_dim=input_dim, hidden_dim=hidden_dim)
        self.sae.load_state_dict(checkpoint["sae_state_dict"])
        self.sae.eval().requires_grad_(False)

        feature_mean = checkpoint.get("feature_mean", torch.zeros(input_dim))
        feature_std = checkpoint.get("feature_std", torch.ones(input_dim))
        self.register_buffer("feature_mean", feature_mean.float().reshape(1, -1))
        self.register_buffer("feature_std", feature_std.float().reshape(1, -1).clamp_min(1e-6))

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
        pooled_features = [self._pool_feature_map(self.features[idx]) for idx in self.style_idx]
        feature = torch.cat(pooled_features, dim=1)
        normalized = (feature - self.feature_mean.to(feature.device)) / self.feature_std.to(feature.device)
        self.sae.to(device=normalized.device, dtype=normalized.dtype)
        z = self.sae.encode(normalized)
        if self.pooling == "frequency":
            descriptor = (z > self.active_threshold).float()
        elif self.pooling == "max":
            descriptor = z
        else:
            descriptor = z
        return descriptor.mean(dim=0, keepdim=True)

    @staticmethod
    def _pool_feature_map(feature: torch.Tensor) -> torch.Tensor:
        return feature.mean(dim=(2, 3))

    def _make_hook(self, idx: int):
        def hook(_module, _args, output):
            self.features[idx] = output.detach()

        return hook
