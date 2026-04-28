import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedModel(nn.Module):
    def __init__(self, backbone: nn.Module, model_type: str):
        """
        model_type:
            - "timm_cnn"
            - "timm_vit"
            - "hf_dinov2_cls"
        """
        super().__init__()
        assert model_type in ["timm_cnn", "timm_vit", "hf_dinov2_cls"]

        self.backbone = backbone
        self.model_type = model_type

    def forward(self, x, return_features: bool = False):
        if self.model_type == "timm_cnn":
            feat_map = self.backbone.forward_features(x)
            feat = self.backbone.forward_head(feat_map, pre_logits=True)
            logits = self.backbone.forward_head(feat_map)

        elif self.model_type == "timm_vit":
            feat_tokens = self.backbone.forward_features(x)   # usually (B, N, D)
            if feat_tokens.ndim == 3:
                feat = feat_tokens[:, 0]                      # CLS token
            else:
                feat = feat_tokens                            # already pooled
            logits = self.backbone.forward_head(feat)

        elif self.model_type == "hf_dinov2_cls":
            outputs = self.backbone(
                pixel_values=x,
                output_hidden_states=True,
                return_dict=True,
            )
            logits = outputs.logits
            feat = outputs.hidden_states[-1][:, 0]            # CLS token

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        if return_features:
            return logits, feat

        return logits