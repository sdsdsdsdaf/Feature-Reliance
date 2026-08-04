"""M3 — AdaContrast 손실 스택.

클래스 균형 memory bank로 k-NN soft-voting pseudo-label을 뽑고, weak/strong
증강 쌍에 대해 same-pseudo negative를 제외한 contrastive 손실과 선택적
diversity 항을 결합한다. 학습 파라미터는 없다 — 유일한 학습 대상은
`Model/intervention.py`의 `gain`이다(전역 불변식).

거리 공간(`distance_space`)이 "h"가 아닐 때는 `bank`가 그 공간과 같은 차원의
feature를 들고 있다고 가정한다(`MemoryBank(dim=...)`가 호출자 책임으로 맞춰짐).
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

_VALID_DISTANCE_SPACES = ("h", "z", "logit")


class MemoryBank:
    def __init__(self, dim: int, capacity: int, num_classes: int, device: str | torch.device = "cpu"):
        """지나간 샘플의 (feature, 예측확률)을 담아둘 큐를 만든다.
        - dim: feature 차원(=backbone h' 크기, distance_space="z"/"logit"이면 그 공간의 차원).
          capacity: 총 슬롯 수. 0이면 in-batch 모드(아무것도 저장하지 않는다).
        - num_classes개의 링 버퍼로 쪼개 클래스당 capacity//num_classes씩 배분한다.
          나머지(capacity % num_classes)는 버려진다(실효 capacity = per_class_cap * num_classes).
        - capacity가 num_classes보다 작아 클래스당 슬롯이 0이 되면(per_class_cap==0)
          사실상 capacity=0과 동일하게 동작한다(아무것도 저장되지 않는다).
        - feats/probs/filled/ptr을 미리 할당해 스트림 중 재할당하지 않는다."""
        if dim <= 0:
            raise ValueError(f"dim은 양수여야 한다: {dim}")
        if capacity < 0:
            raise ValueError(f"capacity는 0 이상이어야 한다: {capacity}")
        if num_classes <= 0:
            raise ValueError(f"num_classes는 양수여야 한다: {num_classes}")

        self.dim = dim
        self.capacity = capacity
        self.num_classes = num_classes
        self.per_class_cap = capacity // num_classes
        self._effective_capacity = self.per_class_cap * num_classes

        device = torch.device(device)
        self.feats = torch.zeros(self._effective_capacity, dim, device=device)
        self.probs = torch.zeros(self._effective_capacity, num_classes, device=device)
        self.filled = torch.zeros(self._effective_capacity, dtype=torch.bool, device=device)
        self.ptr = torch.zeros(num_classes, dtype=torch.long, device=device)

    @property
    def size(self) -> int:
        """현재 큐에 든 유효 항목 수(패딩 슬롯 제외)."""
        return int(self.filled.sum().item())

    @property
    def active_feats(self) -> Tensor:
        """큐에 든 유효 feature만 [size, dim]으로 반환한다(비어 있으면 [0, dim])."""
        return self.feats[self.filled]

    @property
    def active_probs(self) -> Tensor:
        """큐에 든 유효 확률만 [size, num_classes]으로 반환한다."""
        return self.probs[self.filled]

    @property
    def labels(self) -> Tensor:
        """큐에 든 각 항목의 pseudo-label(argmax) [size].
        contrastive에서 같은 라벨을 negative에서 빼는 데 쓴다. 큐가 비어 있으면 빈 텐서."""
        probs = self.active_probs
        if probs.shape[0] == 0:
            return torch.zeros(0, dtype=torch.long, device=self.probs.device)
        return probs.argmax(dim=1)

    @torch.no_grad()
    def enqueue(self, feats: Tensor, probs: Tensor) -> None:
        """이번 배치를 큐에 밀어 넣는다. `probs.argmax(1)` 기준 클래스별 세그먼트에
        가장 오래된 항목을 밀어내며 채운다. `per_class_cap==0`(capacity=0 포함)이면 no-op.
        저장 전 feats를 L2 정규화한다(코사인 kNN이므로). probs는 soft 그대로 저장한다."""
        if self.per_class_cap == 0:
            return
        feats = F.normalize(feats.detach(), dim=1).to(self.feats.device)
        probs = probs.detach().to(self.probs.device)
        labels = probs.argmax(dim=1)
        for i in range(feats.shape[0]):
            c = int(labels[i].item())
            slot = int(self.ptr[c].item())
            pos = c * self.per_class_cap + slot
            self.feats[pos] = feats[i]
            self.probs[pos] = probs[i]
            self.filled[pos] = True
            self.ptr[c] = (slot + 1) % self.per_class_cap

    @torch.no_grad()
    def knn_soft_vote(self, query: Tensor, k: int = 10, logits_weak: Tensor | None = None) -> Tensor:
        """큐에서 가장 닮은 이웃 k개를 찾아 그들의 예측을 평균낸다 -> pseudo-label [N,C].
        '내 예측'이 아니라 '나와 닮은 것들의 합의'라 자기강화가 덜하다.
        큐가 비어 있으면(capacity=0 in-batch 모드 포함, 또는 스트림 초반 size==0)
        `logits_weak`의 softmax로 폴백한다 — 이 경우 `logits_weak`가 필수다.
        `size < k`면 `k`를 `size`로 축소해 에러 없이 동작한다."""
        size = self.size
        if size == 0:
            if logits_weak is None:
                raise ValueError("빈 bank에서 knn_soft_vote를 부르려면 logits_weak 폴백이 필요하다")
            return F.softmax(logits_weak, dim=1)

        q = F.normalize(query, dim=1)
        active_feats = self.active_feats.to(q.device)
        active_probs = self.active_probs.to(q.device)
        sim = q @ active_feats.T  # [N, size]
        eff_k = min(k, size)
        idx = sim.topk(eff_k, dim=1).indices
        return active_probs[idx].mean(dim=1)


class AdaContrastLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        tau: float = 0.07,
        w_ctr: float = 1.0,
        w_div: float = 0.1,
        use_diversity: bool = False,
        distance_space: str = "h",
        knn_k: int = 10,
    ):
        """손실 항들의 가중치와 거리 계산 방식을 고정한다. 학습 파라미터는 없다.
        - tau: contrastive 온도. 작을수록 어려운 negative에 민감
        - w_ctr / w_div: contrastive · diversity 항 가중치
        - use_diversity: 기본 False. True면 한 클래스로 붕괴하는 걸 막는 항이 켜진다
        - distance_space: "h"(기본) | "z" | "logit" — 실험 4B가 뒤집는 축.
          "z"일 때만 forward에 z_weak/z_strong이 필요하다
        - knn_k: bank.knn_soft_vote에 넘길 기본 이웃 수"""
        super().__init__()
        if distance_space not in _VALID_DISTANCE_SPACES:
            raise ValueError(f"distance_space는 {_VALID_DISTANCE_SPACES} 중 하나여야 한다: {distance_space}")
        self.num_classes = num_classes
        self.tau = tau
        self.w_ctr = w_ctr
        self.w_div = w_div
        self.use_diversity = use_diversity
        self.distance_space = distance_space
        self.knn_k = knn_k

    def forward(
        self,
        h_weak: Tensor,
        logits_weak: Tensor,
        h_strong: Tensor,
        logits_strong: Tensor,
        bank: MemoryBank,
        z_weak: Tensor | None = None,
        z_strong: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """약하게/세게 증강한 같은 배치를 받아 손실을 계산한다.
        weak 쪽으로 pseudo-label을 만들고 strong 쪽이 그걸 맞추게 한다.
        z_*는 distance_space="z"일 때만 필요하다.
        반환은 합계 하나가 아니라 항별 dict {"total","pseudo","ctr","div"}."""
        if self.distance_space == "z" and (z_weak is None or z_strong is None):
            raise ValueError("distance_space='z'면 z_weak/z_strong이 필요하다")

        device = h_weak.device
        n = h_weak.shape[0]

        # 1) weak 쪽 pseudo-label. size==0(capacity=0 포함) 이면 자기예측으로 폴백.
        pseudo = bank.knn_soft_vote(h_weak, k=self.knn_k, logits_weak=logits_weak)  # [N, C]
        yhat = pseudo.argmax(dim=1)

        # 2) pseudo-label CE: strong 쪽이 weak 쪽 합의를 맞추게 한다.
        L_pseudo = -(pseudo * F.log_softmax(logits_strong, dim=1)).sum(dim=1).mean()

        # 3) contrastive: 거리를 재는 공간은 distance_space가 결정한다(4B 축).
        if self.distance_space == "h":
            q_raw, k_raw = h_strong, h_weak
        elif self.distance_space == "z":
            q_raw, k_raw = z_strong, z_weak
        else:  # "logit"
            q_raw, k_raw = logits_strong, logits_weak

        q = F.normalize(q_raw, dim=1)
        k_pos = F.normalize(k_raw, dim=1).detach()  # weak 쪽은 pseudo-label 소스, grad 차단
        l_pos = (q * k_pos).sum(dim=1, keepdim=True) / self.tau  # [N, 1]

        if bank.size > 0:
            neg_feats = bank.active_feats.to(device)
            neg_labels = bank.labels.to(device)
        else:
            # capacity=0(in-batch) 모드: negative를 배치 내 다른 샘플에서 취한다.
            # k_pos 자체가 이미 이 배치의 weak-side feature이므로 그대로 negative pool로 쓴다.
            neg_feats = k_pos
            neg_labels = yhat

        l_neg = q @ neg_feats.T / self.tau  # [N, M]
        same_pseudo = yhat.unsqueeze(1) == neg_labels.unsqueeze(0)  # [N, M]
        l_neg = l_neg.masked_fill(same_pseudo, float("-inf"))

        logits_ctr = torch.cat([l_pos, l_neg], dim=1)  # [N, 1+M]
        target = torch.zeros(n, dtype=torch.long, device=device)
        ce_per_row = F.cross_entropy(logits_ctr, target, reduction="none")

        # 엣지 케이스: 한 행의 negative가 전부 same-pseudo로 마스킹되면(모든 bank 항목이
        # 같은 pseudo-label) 그 행의 기여를 0으로 두고 n_valid로만 평균낸다. NaN 방지.
        row_all_masked = same_pseudo.all(dim=1)  # [N]
        ce_per_row = torch.where(row_all_masked, torch.zeros_like(ce_per_row), ce_per_row)
        n_valid = (~row_all_masked).sum().clamp_min(1)
        L_ctr = ce_per_row.sum() / n_valid

        # 4) diversity(옵션): 주변분포 음엔트로피 -> 한 클래스로 붕괴하는 걸 억제.
        if self.use_diversity:
            p_bar = F.softmax(logits_strong, dim=1).mean(dim=0)
            L_div = (p_bar * p_bar.clamp_min(1e-12).log()).sum()
        else:
            L_div = torch.zeros((), device=device, dtype=L_pseudo.dtype)

        total = L_pseudo + self.w_ctr * L_ctr + self.w_div * L_div
        return {"total": total, "pseudo": L_pseudo, "ctr": L_ctr, "div": L_div}
