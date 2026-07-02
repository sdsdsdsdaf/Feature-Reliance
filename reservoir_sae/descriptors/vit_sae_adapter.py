from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
import timm

from Model.SAE import VanillaL1SAE
from Utils.SAE_utils import select_block_tokens


class ViTSAEDescriptor(nn.Module):
    """SAE latent routing descriptor from ViT block tokens.

    The checkpoint should include the SAE weights plus token normalizer:
        {
          "sae_state_dict" or "model_state_dict": ...,
          "input_dim": int,
          "hidden_dim": int,
          "token_mean": Tensor[input_dim],
          "token_std": Tensor[input_dim],
          "target_block": int,
          "token_scope": "cls" | "patch" | "all",
        }

    If classifier_model is supplied, the descriptor hooks that exact model.
    Otherwise it creates a frozen ViT extractor from checkpoint["model_name"].
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        model_name: str | None = None,
        target_block: int | None = None,
        token_scope: str | None = None,
        pooling: str = "frequency",
        active_threshold: float | None = None,
        encode_chunk_size: int = 16384,
        classifier_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path)
        checkpoint = self._load_checkpoint(self.checkpoint_path)

        input_dim = int(checkpoint["input_dim"])
        hidden_dim = int(checkpoint["hidden_dim"])
        self.sae = VanillaL1SAE(input_dim=input_dim, hidden_dim=hidden_dim)
        self.sae.load_state_dict(checkpoint.get("sae_state_dict", checkpoint.get("model_state_dict")))
        self.sae.eval().requires_grad_(False)

        token_mean = checkpoint.get("token_mean", checkpoint.get("feature_mean"))
        token_std = checkpoint.get("token_std", checkpoint.get("feature_std"))
        if token_mean is None or token_std is None:
            raise ValueError(
                f"ViT-SAE checkpoint {self.checkpoint_path} is missing token_mean/token_std. "
                "Re-export or train it with reservoir_sae/experiments/train_vit_sae_descriptor.py."
            )
        self.register_buffer("token_mean", token_mean.float().reshape(1, -1))
        self.register_buffer("token_std", token_std.float().reshape(1, -1).clamp_min(1e-6))

        self.target_block = int(target_block if target_block is not None else checkpoint.get("target_block", 10))
        self.token_scope = str(token_scope or checkpoint.get("token_scope", "patch"))
        self.pooling = str(pooling or checkpoint.get("pooling", "frequency")).lower()
        self.active_threshold = float(
            active_threshold if active_threshold is not None else checkpoint.get("active_threshold", 0.1)
        )
        self.encode_chunk_size = int(encode_chunk_size)
        if self.pooling not in {"mean", "frequency", "max"}:
            raise ValueError("vit_sae_pooling must be one of: mean, frequency, max")

        self.uses_classifier_model = classifier_model is not None
        if classifier_model is not None:
            self.vit = classifier_model
        else:
            resolved_model_name = model_name or checkpoint.get("model_name", "vit_base_patch16_224")
            self.vit = timm.create_model(str(resolved_model_name), pretrained=True).eval()
            self.vit.requires_grad_(False)

        if not hasattr(self.vit, "blocks"):
            raise ValueError("ViT-SAE routing requires a timm ViT-style model with a .blocks attribute.")
        if self.target_block < 0 or self.target_block >= len(self.vit.blocks):
            raise ValueError(f"target_block={self.target_block} is outside model.blocks length {len(self.vit.blocks)}.")

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        captured: dict[str, torch.Tensor] = {}

        def hook(_module, _inputs, output):
            captured["tokens"] = select_block_tokens(output, token_scope=self.token_scope, cpu=False)

        handle = self.vit.blocks[self.target_block].register_forward_hook(hook)
        try:
            _ = self.vit(images)
        finally:
            handle.remove()

        tokens = captured["tokens"].float()
        batch_size = int(images.shape[0])
        tokens_per_image = max(1, tokens.shape[0] // batch_size)
        flat = tokens.reshape(-1, tokens.shape[-1])
        mean = self.token_mean.to(device=flat.device, dtype=flat.dtype)
        std = self.token_std.to(device=flat.device, dtype=flat.dtype)
        self.sae.to(device=flat.device, dtype=flat.dtype)

        z_chunks = []
        for start in range(0, flat.shape[0], self.encode_chunk_size):
            chunk = (flat[start : start + self.encode_chunk_size] - mean) / std
            z_chunks.append(self.sae.encode(chunk))
        z = torch.cat(z_chunks, dim=0).reshape(batch_size, tokens_per_image, -1)

        if self.pooling == "frequency":
            per_image = (z > self.active_threshold).float().mean(dim=1)
        elif self.pooling == "max":
            per_image = z.max(dim=1).values
        else:
            per_image = z.mean(dim=1)
        return per_image.mean(dim=0, keepdim=True)

    @staticmethod
    def _load_checkpoint(path: Path) -> dict[str, Any]:
        if path.is_dir():
            summary_path = path / "summary.json"
            checkpoint_path = path / "best_sae_state.pt"
            if not summary_path.exists() or not checkpoint_path.exists():
                raise FileNotFoundError(f"Expected summary.json and best_sae_state.pt in {path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            checkpoint["input_dim"] = int(summary["input_dim"])
            checkpoint["hidden_dim"] = int(summary["hidden_dim"])
            config_path = path / "config.json"
            if config_path.exists():
                config = json.loads(config_path.read_text(encoding="utf-8"))
                checkpoint["target_block"] = config.get("hook", {}).get("target_block", 10)
                checkpoint["token_scope"] = config.get("hook", {}).get("token_scope", "patch")
                checkpoint["active_threshold"] = config.get("sae", {}).get("active_threshold", 0.1)
            return checkpoint
        if not path.exists():
            raise FileNotFoundError(f"ViT-SAE checkpoint not found: {path}")
        return torch.load(path, map_location="cpu")
