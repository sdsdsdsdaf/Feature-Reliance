import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsistencyLoss(nn.Module):
    def __init__(
        self,
        mode: str = "feature",  # "kl", "feature", "both", "none"
        lambda_kl: float = 1.0,
        lambda_feat: float = 1.0,
        temperature: float = 1.0,
        detach_teacher: bool = True,
        normalize_feature: bool = True,
        ce_clean_weight: float = 1.0,
        ce_pert_weight: float = 1.0,
    ):
        
        super().__init__()
        assert mode in ["kl", "feature", "both", "none"]
        
        self.mode = mode
        self.lambda_kl = lambda_kl
        self.lambda_feat = lambda_feat
        self.temperature = temperature
        self.detach_teacher = detach_teacher
        self.normalize_feature = normalize_feature
        self.ce_clean_weight = ce_clean_weight
        self.ce_pert_weight = ce_pert_weight

    def kl_consistency(self, original_logits, perturbed_logits):
        T = self.temperature
        teacher_logits = original_logits.detach() if self.detach_teacher else original_logits
        log_p_perturbed = F.log_softmax(perturbed_logits / T, dim=1)
        p_orig = F.softmax(teacher_logits / T, dim=1)
        
        loss_kl = F.kl_div(log_p_perturbed, p_orig, reduction='batchmean') * (T ** 2)
        
        return loss_kl
    
    """ # 후에 학습이 잘 되지 않을 시 이 함수로 교체
    def feature_consistency(self, original_features, perturbed_features):
        original_features = F.normalize(original_features, p=2, dim=-1)
        perturbed_features = F.normalize(perturbed_features, p=2, dim=-1)

        teacher_features = (
            original_features.detach()
            if self.detach_teacher
            else original_features
        )

        return 1.0 - F.cosine_similarity(
            perturbed_features,
            teacher_features,
            dim=-1,
        ).mean()
    """
        
    def feature_consistency(self, original_features, perturbed_features):
        if self.normalize_feature:
            original_features = F.normalize(original_features, p=2, dim=-1)
            perturbed_features = F.normalize(perturbed_features, p=2, dim=-1)
        
        loss_feat = F.mse_loss(perturbed_features, original_features.detach() if self.detach_teacher else original_features)
        
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