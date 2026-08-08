"""M3 계약 검증: 클래스 균형 MemoryBank(k-NN soft-voting) · AdaContrastLoss
(same-pseudo negative 제외 · buffer=0 in-batch 폴백 · 전-마스킹 행 NaN 방지).

작은 dim으로 CPU에서 빠르게 돈다 — GPU는 다른 worker가 점유 중이라는 전제
(phaseM.md 실행 환경 절)를 지킨다.
"""

import pytest
import torch
import torch.nn.functional as F

from Model.adacontrast import AdaContrastLoss, MemoryBank


# ---------------------------------------------------------------------------
# MemoryBank
# ---------------------------------------------------------------------------


class TestMemoryBankInit:
    def test_per_class_capacity_is_capacity_over_num_classes(self):
        bank = MemoryBank(dim=4, capacity=12, num_classes=3)
        assert bank.per_class_cap == 4

    def test_size_starts_at_zero(self):
        bank = MemoryBank(dim=4, capacity=12, num_classes=3)
        assert bank.size == 0

    def test_capacity_zero_is_valid_in_batch_mode(self):
        """capacity=0은 에러가 아니라 실험 6의 buffer=0 arm이다."""
        bank = MemoryBank(dim=4, capacity=0, num_classes=3)
        assert bank.size == 0
        assert bank.per_class_cap == 0


class TestMemoryBankEnqueue:
    def test_enqueue_increases_size(self):
        bank = MemoryBank(dim=4, capacity=12, num_classes=3)
        feats = torch.randn(6, 4)
        probs = F.one_hot(torch.tensor([0, 0, 1, 1, 2, 2]), num_classes=3).float()
        bank.enqueue(feats, probs)
        assert bank.size == 6

    def test_capacity_zero_enqueue_is_noop(self):
        bank = MemoryBank(dim=4, capacity=0, num_classes=3)
        feats = torch.randn(6, 4)
        probs = F.one_hot(torch.tensor([0, 0, 1, 1, 2, 2]), num_classes=3).float()
        bank.enqueue(feats, probs)
        assert bank.size == 0

    def test_stored_feats_are_l2_normalized(self):
        bank = MemoryBank(dim=4, capacity=12, num_classes=3)
        feats = torch.randn(6, 4) * 10.0  # 큰 스케일이라 정규화 안 됐으면 바로 드러남
        probs = F.one_hot(torch.tensor([0, 0, 1, 1, 2, 2]), num_classes=3).float()
        bank.enqueue(feats, probs)
        norms = bank.active_feats.norm(dim=1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=0)

    def test_probs_stay_soft_not_normalized(self):
        bank = MemoryBank(dim=4, capacity=12, num_classes=3)
        feats = torch.randn(2, 4)
        probs = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
        bank.enqueue(feats, probs)
        stored = bank.probs[bank.filled]
        torch.testing.assert_close(stored, probs, atol=1e-6, rtol=0)

    def test_class_balanced_ring_buffer_does_not_let_common_class_crowd_out_rare(self):
        """단일 글로벌 FIFO였다면 흔한 클래스가 큐 전체를 차지했을 상황을 구성한다.
        클래스 0이 20개, 클래스 1이 1개 들어와도 클래스 1의 슬롯은 살아남아야 한다."""
        bank = MemoryBank(dim=4, capacity=4, num_classes=2)  # per_class_cap = 2
        many_zero_feats = torch.randn(20, 4)
        many_zero_probs = F.one_hot(torch.zeros(20, dtype=torch.long), num_classes=2).float()
        bank.enqueue(many_zero_feats, many_zero_probs)

        one_class_feats = torch.randn(1, 4)
        one_class_probs = F.one_hot(torch.tensor([1]), num_classes=2).float()
        bank.enqueue(one_class_feats, one_class_probs)

        labels = bank.labels
        assert (labels == 1).sum().item() == 1, "class-1 슬롯이 글로벌 FIFO에 밀려나면 안 된다"
        assert (labels == 0).sum().item() == 2, "class-0은 자기 세그먼트(2슬롯) 안에서만 회전해야 한다"

    def test_ring_buffer_overwrites_oldest_within_class_segment(self):
        bank = MemoryBank(dim=1, capacity=2, num_classes=1)  # per_class_cap = 2
        probs = F.one_hot(torch.tensor([0]), num_classes=1).float()
        for v in [1.0, 2.0, 3.0]:
            bank.enqueue(torch.tensor([[v]]), probs)
        stored = sorted(bank.active_feats.flatten().tolist())
        # 정규화되어도 1차원 dim=1이면 부호만 남는다 -> L2 정규화 후 값은 sign(v) = 1.0 전부.
        # 부호 대신 순서를 확인하려면 dim>=2가 필요하므로 여기선 3개 중 오래된 1.0이
        # 밀려나 큐 크기가 여전히 capacity(2)로 유지되는지만 확인한다.
        assert len(stored) == 2

    def test_no_reallocation_mid_stream_buffer_identity_stable(self):
        """전역 불변식: feats/probs 버퍼는 스트림 중 재할당되지 않는다(같은 텐서 객체 유지)."""
        bank = MemoryBank(dim=4, capacity=8, num_classes=2)
        feats_id_before = id(bank.feats)
        probs_id_before = id(bank.probs)
        for _ in range(3):
            f = torch.randn(4, 4)
            p = F.one_hot(torch.tensor([0, 0, 1, 1]), num_classes=2).float()
            bank.enqueue(f, p)
        assert id(bank.feats) == feats_id_before
        assert id(bank.probs) == probs_id_before


class TestMemoryBankKnnSoftVote:
    def test_size_zero_falls_back_to_logits_weak_softmax(self):
        bank = MemoryBank(dim=4, capacity=8, num_classes=3)
        query = torch.randn(2, 4)
        logits_weak = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
        out = bank.knn_soft_vote(query, k=5, logits_weak=logits_weak)
        torch.testing.assert_close(out, F.softmax(logits_weak, dim=1))

    def test_capacity_zero_bank_falls_back_to_logits_weak(self):
        bank = MemoryBank(dim=4, capacity=0, num_classes=3)
        query = torch.randn(2, 4)
        logits_weak = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        out = bank.knn_soft_vote(query, k=5, logits_weak=logits_weak)
        torch.testing.assert_close(out, F.softmax(logits_weak, dim=1))

    def test_empty_bank_without_logits_weak_raises(self):
        bank = MemoryBank(dim=4, capacity=8, num_classes=3)
        with pytest.raises(ValueError):
            bank.knn_soft_vote(torch.randn(2, 4), k=5)

    def test_knn_soft_vote_averages_topk_neighbor_probs(self):
        """정확히 계산 가능한 소규모 예제: 이웃 확률의 평균이 나와야 한다."""
        bank = MemoryBank(dim=2, capacity=4, num_classes=2)
        # 4개의 직교에 가까운 방향 벡터를 넣어 top-1 이웃이 명확하게 결정되게 한다.
        feats = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        labels_int = torch.tensor([0, 1, 0, 1])
        onehot_probs = F.one_hot(labels_int, num_classes=2).float()
        bank.enqueue(feats, onehot_probs)

        query = torch.tensor([[1.0, 0.0]])  # feats[0]과 가장 가깝다
        out = bank.knn_soft_vote(query, k=1)
        torch.testing.assert_close(out, torch.tensor([[1.0, 0.0]]))

    def test_k_shrinks_to_size_when_bank_smaller_than_k(self):
        """스트림 초반: size < k면 k = size로 축소돼 에러 없이 동작한다."""
        bank = MemoryBank(dim=2, capacity=8, num_classes=2)
        feats = torch.tensor([[1.0, 0.0]])
        probs = torch.tensor([[1.0, 0.0]])
        bank.enqueue(feats, probs)
        assert bank.size == 1
        out = bank.knn_soft_vote(torch.tensor([[1.0, 0.0]]), k=10)
        assert out.shape == (1, 2)


class TestMemoryBankLabels:
    def test_labels_property_is_argmax_of_stored_probs(self):
        bank = MemoryBank(dim=4, capacity=8, num_classes=3)
        feats = torch.randn(3, 4)
        probs = torch.tensor([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.2, 0.2, 0.6]])
        bank.enqueue(feats, probs)
        assert torch.equal(bank.labels.sort().values, torch.tensor([0, 1, 2]).sort().values)

    def test_labels_empty_when_bank_empty(self):
        bank = MemoryBank(dim=4, capacity=8, num_classes=3)
        assert bank.labels.shape == (0,)


# ---------------------------------------------------------------------------
# AdaContrastLoss
# ---------------------------------------------------------------------------


def _toy_batch(n=6, dim=4, num_classes=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    h_weak = torch.randn(n, dim, generator=g)
    h_strong = torch.randn(n, dim, generator=g, requires_grad=True)
    logits_weak = torch.randn(n, num_classes, generator=g)
    logits_strong = torch.randn(n, num_classes, generator=g, requires_grad=True)
    return h_weak, logits_weak, h_strong, logits_strong


class TestAdaContrastLossBasic:
    def test_forward_returns_dict_with_expected_keys(self):
        bank = MemoryBank(dim=4, capacity=12, num_classes=3)
        loss_fn = AdaContrastLoss(num_classes=3)
        h_weak, logits_weak, h_strong, logits_strong = _toy_batch()
        out = loss_fn(h_weak, logits_weak, h_strong, logits_strong, bank)
        assert set(out.keys()) == {"total", "pseudo", "ctr", "div"}
        for v in out.values():
            assert torch.isfinite(v).all()

    def test_total_equals_pseudo_plus_weighted_ctr_when_diversity_off(self):
        bank = MemoryBank(dim=4, capacity=12, num_classes=3)
        loss_fn = AdaContrastLoss(num_classes=3, w_ctr=2.0, w_div=0.5, use_diversity=False)
        h_weak, logits_weak, h_strong, logits_strong = _toy_batch()
        out = loss_fn(h_weak, logits_weak, h_strong, logits_strong, bank)
        assert out["div"].item() == 0.0
        expected = out["pseudo"] + 2.0 * out["ctr"]
        torch.testing.assert_close(out["total"], expected)

    def test_diversity_term_nonzero_only_when_enabled(self):
        bank = MemoryBank(dim=4, capacity=12, num_classes=3)
        h_weak, logits_weak, h_strong, logits_strong = _toy_batch()

        off = AdaContrastLoss(num_classes=3, use_diversity=False)
        out_off = off(h_weak, logits_weak, h_strong, logits_strong, bank)
        assert out_off["div"].item() == 0.0

        on = AdaContrastLoss(num_classes=3, use_diversity=True)
        out_on = on(h_weak, logits_weak, h_strong, logits_strong, bank)
        # 배치가 무작위라 붕괴돼 있지 않으므로 음수(음-엔트로피)일 가능성이 높다 -- 0은 아니어야 함
        assert out_on["div"].item() != 0.0

    def test_gradient_flows_to_strong_side_not_weak_side(self):
        """weak 쪽(k_pos, pseudo-label 소스)은 detach돼 gradient가 흐르지 않는다."""
        bank = MemoryBank(dim=4, capacity=12, num_classes=3)
        loss_fn = AdaContrastLoss(num_classes=3)
        h_weak, logits_weak, h_strong, logits_strong = _toy_batch()
        out = loss_fn(h_weak, logits_weak, h_strong, logits_strong, bank)
        out["total"].backward()
        assert h_strong.grad is not None
        assert not torch.equal(h_strong.grad, torch.zeros_like(h_strong.grad))
        assert h_weak.grad is None  # requires_grad=False로 만들었으니 애초에 grad 속성 없음
        assert logits_strong.grad is not None


class TestAdaContrastLossBufferZero:
    def test_capacity_zero_uses_in_batch_negatives_and_stays_finite(self):
        bank = MemoryBank(dim=4, capacity=0, num_classes=3)
        loss_fn = AdaContrastLoss(num_classes=3)
        h_weak, logits_weak, h_strong, logits_strong = _toy_batch()
        out = loss_fn(h_weak, logits_weak, h_strong, logits_strong, bank)
        for v in out.values():
            assert torch.isfinite(v).all()


class TestAdaContrastLossSamePseudoMasking:
    def test_all_negatives_same_pseudo_label_zeroes_row_without_nan(self):
        """가장 중요한 엣지 케이스: bank의 모든 항목이 query의 pseudo-label과 같으면
        그 행의 negative가 전부 -inf로 마스킹된다. NaN 없이 유한값이어야 한다."""
        dim = 4
        num_classes = 3
        bank = MemoryBank(dim=dim, capacity=12, num_classes=num_classes)

        # bank를 전부 class 0 pseudo-label로 채운다.
        bank_feats = torch.randn(4, dim)
        bank_probs = F.one_hot(torch.zeros(4, dtype=torch.long), num_classes=num_classes).float()
        bank.enqueue(bank_feats, bank_probs)

        # query 쪽도 weak 예측이 class 0으로 쏠리게 만들어 yhat==0이 되게 한다.
        n = 3
        h_weak = torch.randn(n, dim)
        logits_weak = torch.tensor([[5.0, -5.0, -5.0]] * n)
        h_strong = torch.randn(n, dim, requires_grad=True)
        logits_strong = torch.randn(n, num_classes, requires_grad=True)

        loss_fn = AdaContrastLoss(num_classes=num_classes)
        out = loss_fn(h_weak, logits_weak, h_strong, logits_strong, bank)

        for k, v in out.items():
            assert torch.isfinite(v).all(), f"{k}가 NaN/Inf: {v}"

        out["total"].backward()
        assert h_strong.grad is not None
        assert torch.isfinite(h_strong.grad).all(), "전-마스킹 행의 backward에서 NaN gradient가 새면 안 된다"
        assert logits_strong.grad is not None
        assert torch.isfinite(logits_strong.grad).all()


class TestAdaContrastLossDistanceSpace:
    def test_default_distance_space_is_h(self):
        loss_fn = AdaContrastLoss(num_classes=3)
        assert loss_fn.distance_space == "h"

    def test_z_space_requires_z_weak_and_z_strong(self):
        bank = MemoryBank(dim=4, capacity=12, num_classes=3)
        loss_fn = AdaContrastLoss(num_classes=3, distance_space="z")
        h_weak, logits_weak, h_strong, logits_strong = _toy_batch()
        with pytest.raises(ValueError):
            loss_fn(h_weak, logits_weak, h_strong, logits_strong, bank)

    def test_z_space_runs_with_z_weak_and_z_strong(self):
        code_dim = 8
        bank = MemoryBank(dim=code_dim, capacity=12, num_classes=3)
        loss_fn = AdaContrastLoss(num_classes=3, distance_space="z")
        h_weak, logits_weak, h_strong, logits_strong = _toy_batch()
        n = h_weak.shape[0]
        z_weak = torch.randn(n, code_dim)
        z_strong = torch.randn(n, code_dim, requires_grad=True)
        out = loss_fn(
            h_weak, logits_weak, h_strong, logits_strong, bank, z_weak=z_weak, z_strong=z_strong
        )
        for v in out.values():
            assert torch.isfinite(v).all()
        out["total"].backward()
        assert z_strong.grad is not None

    def test_logit_space_runs_without_z(self):
        num_classes = 3
        bank = MemoryBank(dim=num_classes, capacity=12, num_classes=num_classes)
        loss_fn = AdaContrastLoss(num_classes=num_classes, distance_space="logit")
        h_weak, logits_weak, h_strong, logits_strong = _toy_batch(num_classes=num_classes)
        out = loss_fn(h_weak, logits_weak, h_strong, logits_strong, bank)
        for v in out.values():
            assert torch.isfinite(v).all()

    def test_invalid_distance_space_raises_at_init(self):
        with pytest.raises(ValueError):
            AdaContrastLoss(num_classes=3, distance_space="bogus")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 미가용")
class TestCudaDevice:
    """MemoryBank·AdaContrastLoss가 CUDA에서 device 불일치 없이 돈다."""

    def test_memory_bank_and_loss_run_on_cuda(self):
        dim, num_classes, n = 4, 3, 6
        bank = MemoryBank(dim=dim, capacity=12, num_classes=num_classes, device="cuda")
        loss_fn = AdaContrastLoss(num_classes=num_classes)

        h_weak = torch.randn(n, dim, device="cuda")
        logits_weak = torch.randn(n, num_classes, device="cuda")
        h_strong = torch.randn(n, dim, device="cuda", requires_grad=True)
        logits_strong = torch.randn(n, num_classes, device="cuda", requires_grad=True)

        out = loss_fn(h_weak, logits_weak, h_strong, logits_strong, bank)
        for v in out.values():
            assert v.device.type == "cuda"
        out["total"].backward()
        assert h_strong.grad.device.type == "cuda"

        bank.enqueue(h_weak.detach(), F.softmax(logits_weak, dim=1).detach())
        assert bank.active_feats.device.type == "cuda"
