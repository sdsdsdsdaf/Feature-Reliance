from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class RoutingResult:
    model_idx: int
    model_prob: float
    new_cluster: bool
    min_distance: float
    num_models: int


class CosinePrototypeReservoir:
    """Small reservoir router for minimal StyleVec-vs-SAE experiments."""

    def __init__(
        self,
        descriptor_dim: int,
        max_models: int,
        threshold: float,
        prototype_momentum: float = 0.2,
        device: str | torch.device = "cpu",
    ) -> None:
        self.descriptor_dim = int(descriptor_dim)
        self.max_models = int(max_models)
        self.threshold = float(threshold)
        self.prototype_momentum = float(prototype_momentum)
        self.device = torch.device(device)
        self.prototypes = torch.empty(0, self.descriptor_dim, device=self.device)

    @property
    def num_models(self) -> int:
        return int(self.prototypes.shape[0])

    def initialize(self, descriptor: torch.Tensor) -> None:
        descriptor = self._normalize_descriptor(descriptor)
        self.prototypes = descriptor.detach().clone()

    def route(self, descriptor: torch.Tensor) -> RoutingResult:
        descriptor = self._normalize_descriptor(descriptor)
        if self.num_models == 0:
            self.initialize(descriptor)
            return RoutingResult(0, 1.0, True, 0.0, self.num_models)

        distances = 1.0 - F.cosine_similarity(descriptor, self.prototypes, dim=1)
        min_distance, idx = torch.min(distances, dim=0)
        new_cluster = bool(min_distance.item() > self.threshold and self.num_models < self.max_models)

        if new_cluster:
            self.prototypes = torch.cat([self.prototypes, descriptor.detach().clone()], dim=0)
            model_idx = self.num_models - 1
            model_prob = 1.0
        else:
            model_idx = int(idx.item())
            old = self.prototypes[model_idx]
            momentum = self.prototype_momentum
            self.prototypes[model_idx] = F.normalize((1.0 - momentum) * old + momentum * descriptor[0], dim=0)
            scores = -distances
            model_prob = float(torch.softmax(scores, dim=0)[model_idx].item())

        return RoutingResult(
            model_idx=model_idx,
            model_prob=model_prob,
            new_cluster=new_cluster,
            min_distance=float(min_distance.item()),
            num_models=self.num_models,
        )

    def _normalize_descriptor(self, descriptor: torch.Tensor) -> torch.Tensor:
        descriptor = descriptor.detach().to(self.device).float()
        if descriptor.ndim != 2 or descriptor.shape[0] != 1:
            raise ValueError(f"Expected descriptor shape [1, D], got {tuple(descriptor.shape)}")
        return F.normalize(descriptor, dim=1)

