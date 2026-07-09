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
    parent_model_idx: int | None = None
    model_probs: list[float] | None = None


class PrototypeReservoir:
    """Small ReservoirTTA-style prototype router."""

    def __init__(
        self,
        descriptor_dim: int,
        max_models: int,
        threshold: float,
        prototype_momentum: float = 0.2,
        distance_metric: str = "l2",
        device: str | torch.device = "cpu",
    ) -> None:
        self.descriptor_dim = int(descriptor_dim)
        self.max_models = int(max_models)
        self.threshold = float(threshold)
        self.prototype_momentum = float(prototype_momentum)
        self.distance_metric = str(distance_metric).lower()
        if self.distance_metric not in {"l2", "cosine"}:
            raise ValueError("distance_metric must be one of: l2, cosine")
        self.device = torch.device(device)
        self.prototypes = torch.empty(0, self.descriptor_dim, device=self.device)

    @property
    def num_models(self) -> int:
        return int(self.prototypes.shape[0])

    def route(self, descriptor: torch.Tensor) -> RoutingResult:
        descriptor = self._prepare_descriptor(descriptor)
        if self.num_models == 0:
            self.initialize(descriptor)
            return RoutingResult(0, 1.0, True, 0.0, self.num_models, parent_model_idx=None, model_probs=[1.0])

        distances = self._distance(descriptor, self.prototypes)
        min_distance, idx = torch.min(distances, dim=0)
        nearest_idx = int(idx.item())
        new_cluster = bool(min_distance.item() > self.threshold and self.num_models < self.max_models)

        if new_cluster:
            self.prototypes = torch.cat([self.prototypes, descriptor.detach().clone()], dim=0)
            model_idx = self.num_models - 1
            parent_model_idx = nearest_idx
            distances = self._distance(descriptor, self.prototypes)
            probs = torch.softmax(-distances, dim=0)
            model_prob = float(probs[model_idx].item())
        else:
            model_idx = nearest_idx
            old = self.prototypes[model_idx]
            momentum = self.prototype_momentum
            updated = (1.0 - momentum) * old + momentum * descriptor[0]
            if self.distance_metric == "cosine":
                updated = F.normalize(updated, dim=0)
            self.prototypes[model_idx] = updated
            probs = torch.softmax(-distances, dim=0)
            model_prob = float(probs[model_idx].item())
            parent_model_idx = model_idx

        return RoutingResult(
            model_idx=model_idx,
            model_prob=model_prob,
            new_cluster=new_cluster,
            min_distance=float(min_distance.item()),
            num_models=self.num_models,
            parent_model_idx=parent_model_idx,
            model_probs=[float(x) for x in probs.detach().cpu().tolist()],
        )

    def _prepare_descriptor(self, descriptor: torch.Tensor) -> torch.Tensor:
        descriptor = descriptor.detach().to(self.device).float()
        if descriptor.ndim != 2 or descriptor.shape[0] != 1:
            raise ValueError(f"Expected descriptor shape [1, D], got {tuple(descriptor.shape)}")
        if self.distance_metric == "cosine":
            descriptor = F.normalize(descriptor, dim=1)
        return descriptor

    def initialize(self, descriptor: torch.Tensor) -> None:
        descriptor = self._prepare_descriptor(descriptor)
        self.prototypes = descriptor.detach().clone()

    def _distance(self, descriptor: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        if self.distance_metric == "cosine":
            return 1.0 - F.cosine_similarity(descriptor, prototypes, dim=1)
        return torch.linalg.vector_norm(prototypes - descriptor, ord=2, dim=1)


CosinePrototypeReservoir = PrototypeReservoir
