import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConsistencyLoss(nn.Module):
    def __init__(
        self,
        mode: str = "feature",  # "kl", "feature", "both", "none"
        feature_loss_type: str = "consine",
        lambda_kl: float = 1.0,
        lambda_feat: float = 1.0,
        temperature: float = 1.0,
        detach_teacher: bool = True,
        normalize_feature: bool = True,
        ce_clean_weight: float = 1.0,
        ce_pert_weight: float = 1.0,
        eps=1e-12,
    ):
        
        super().__init__()
        assert mode in ["kl", "feature", "both", "none"]
        assert feature_loss_type in ["cosine", "mse", "mse_sum"]
        
        self.mode = mode
        self.feature_loss_type = feature_loss_type
        self.eps = eps

        self.lambda_kl = lambda_kl
        self.lambda_feat = lambda_feat
        self.temperature = temperature
        self.detach_teacher = detach_teacher
        self.normalize_feature = normalize_feature
        self.ce_clean_weight = ce_clean_weight
        self.ce_pert_weight = ce_pert_weight

    def kl_consistency(self, original_logits:Tensor, perturbed_logits:Tensor):
        T = self.temperature
        teacher_logits = original_logits.detach() if self.detach_teacher else original_logits
        log_p_perturbed = F.log_softmax(perturbed_logits / T, dim=1)
        p_orig = F.softmax(teacher_logits / T, dim=1)
        
        loss_kl = F.kl_div(log_p_perturbed, p_orig, reduction='batchmean') * (T ** 2)
        
        return loss_kl
        
    def feature_consistency(self, original_features:Tensor, perturbed_features:Tensor):
        teacher_feature = (
            original_features.detach()
            if self.detach_teacher
            else original_features
        )

        student_features = perturbed_features

        if self.normalize_feature:
            teacher_feature = F.normalize(
                teacher_feature,
                p=2,
                dim=-1,
                eps=self.eps,
            )
            student_features = F.normalize(
                student_features,
                p=2,
                dim=-1,
                eps=self.eps
            )

        if self.feature_loss_type == "cosine":
            cosine_sim = F.cosine_similarity(
                student_features,
                teacher_feature,
                dim=-1,
                eps=self.eps
            )
            loss_feat = 1.0 - cosine_sim.mean()

        elif self.feature_loss_type == "mse":
            loss_feat = F.mse_loss(
                student_features,
                teacher_feature,
                reduction="mean"
            )
        elif self.feature_loss_type == "mse_sum":
            # Per-sample squared distance, then batch mean.
            # This avoids the loss becoming artificially tiny due to feature-dimension averaging.            
            loss_feat = (student_features - teacher_feature).pow(2).sum(dim=-1).mean()

        else:
            raise ValueError(f"Unknown feature_loss_type: {self.feature_loss_type}")
        
        return loss_feat
    
    
    def consistency_loss(self, original_logits=None, perturbed_logits=None, original_features=None, perturbed_features=None):
        if self.mode == "none":
            return torch.tensor(
                0.0,
                device=self._get_device(
                    original_logits,
                    perturbed_logits,
                    original_features,
                    perturbed_features,
                ),
            )

        loss = 0.0

        if self.mode in ["kl", "both"]:
            assert original_logits is not None
            assert perturbed_logits is not None

            loss = loss + self.lambda_kl * self.kl_consistency(
                original_logits,
                perturbed_logits,
            )

        if self.mode in ["feature", "both"]:
            assert original_features is not None
            assert perturbed_features is not None

            loss = loss + self.lambda_feat * self.feature_consistency(
                original_features,
                perturbed_features,
            )

        return loss
    
    def forward(
        self,
        original_logits,
        perturbed_logits,
        labels,
        original_features=None,
        perturbed_features=None,
    ):
        ce_clean = F.cross_entropy(original_logits, labels)
        ce_pert = F.cross_entropy(perturbed_logits, labels)

        loss_consistency = self.consistency_loss(
            original_logits=original_logits,
            perturbed_logits=perturbed_logits,
            original_features=original_features,
            perturbed_features=perturbed_features,
        )

        total_loss = (
            self.ce_clean_weight * ce_clean
            + self.ce_pert_weight * ce_pert
            + loss_consistency
        )

        loss_dict = {
            "loss_total": total_loss.detach(),
            "loss_ce_clean": ce_clean.detach(),
            "loss_ce_pert": ce_pert.detach(),
            "loss_consistency": loss_consistency.detach(),
        }

        return total_loss, loss_dict
    
    def _get_device(self, *tensors):
        for tensor in tensors:
            if tensor is not None:
                return tensor.device
        return torch.device("cpu")