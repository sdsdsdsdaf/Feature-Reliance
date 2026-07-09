from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=1)
    return -(probs * logits.log_softmax(dim=1)).sum(dim=1)


def entropy_plus_class_marginal(logits: torch.Tensor) -> torch.Tensor:
    ent = softmax_entropy(logits).mean()
    probs = logits.softmax(dim=1)
    avg_probs = probs.mean(dim=0)
    class_marginal = torch.sum(avg_probs * torch.log(avg_probs + 1e-8))
    return ent + class_marginal


def configure_bn_only_tent(model: nn.Module) -> list[nn.Parameter]:
    """Freeze model except normalization affine parameters."""

    model.train()
    model.requires_grad_(False)
    params: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            module.requires_grad_(True)
            for param in module.parameters(recurse=False):
                if param.requires_grad:
                    params.append(param)
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
    return params


class SpecialistBank:
    """Stores specialist trainable-parameter and optimizer states."""

    def __init__(self, params: list[nn.Parameter], optimizer: torch.optim.Optimizer) -> None:
        self.params = params
        self.optimizer = optimizer
        self.source_params = self._clone_params()
        self.source_optimizer = deepcopy(optimizer.state_dict())
        self.param_states = [self._clone_params()]
        self.optimizer_states = [deepcopy(optimizer.state_dict())]

    def ensure(self, idx: int, init_from: int | None = 0) -> None:
        while len(self.param_states) <= idx:
            if init_from is None:
                self.param_states.append([param.clone() for param in self.source_params])
                self.optimizer_states.append(deepcopy(self.source_optimizer))
            else:
                init_idx = min(int(init_from), len(self.param_states) - 1)
                self.param_states.append([param.clone() for param in self.param_states[init_idx]])
                self.optimizer_states.append(deepcopy(self.optimizer_states[init_idx]))

    def load(self, idx: int) -> None:
        self.ensure(idx)
        with torch.no_grad():
            for param, saved in zip(self.params, self.param_states[idx]):
                param.copy_(saved)
        self.optimizer.load_state_dict(self.optimizer_states[idx])

    def load_ensemble(self, weights: list[float] | torch.Tensor) -> None:
        weights_tensor = torch.as_tensor(weights, dtype=self.params[0].dtype, device=self.params[0].device)
        if weights_tensor.numel() > len(self.param_states):
            raise ValueError(f"Got {weights_tensor.numel()} ensemble weights for {len(self.param_states)} specialists.")
        weights_tensor = weights_tensor / weights_tensor.sum().clamp_min(1e-12)
        with torch.no_grad():
            for param_idx, param in enumerate(self.params):
                stacked = torch.stack(
                    [
                        state[param_idx].to(device=param.device, dtype=param.dtype)
                        for state in self.param_states[: weights_tensor.numel()]
                    ],
                    dim=0,
                )
                view_shape = (weights_tensor.numel(),) + (1,) * param.ndim
                param.copy_((stacked * weights_tensor.view(view_shape)).sum(dim=0))

    def save(self, idx: int) -> None:
        self.ensure(idx)
        self.param_states[idx] = self._clone_params()
        self.optimizer_states[idx] = deepcopy(self.optimizer.state_dict())

    def _clone_params(self) -> list[torch.Tensor]:
        return [param.detach().clone() for param in self.params]


def tent_step(
    model: nn.Module,
    images: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    steps: int = 1,
) -> torch.Tensor:
    logits = None
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = softmax_entropy(logits).mean()
        loss.backward()
        optimizer.step()
    if logits is None:
        logits = model(images)
    return logits.detach()


@torch.no_grad()
def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    valid = labels >= 0
    if not bool(valid.any()):
        return 0, 0
    preds = logits.argmax(dim=1)
    correct = int((preds[valid] == labels[valid]).sum().item())
    total = int(valid.sum().item())
    return correct, total
