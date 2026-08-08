"""M5 계약 검증: `GainAnchor`의 3 모드 전환과 `lam=0`이 `off`와 수치적으로
동일함을 확인한다(phaseM.md M5 절 + M5 DoD).

CPU에서만 도는 순수 텐서 연산이라 backbone/timm 의존이 없다. GPU 개입 텐서에
대해서도 같은 불변식이 성립하는지 확인하는 CUDA 게이트 테스트를 맨 아래에 둔다
(다른 테스트 모듈에서 device 버그가 세 번 잡혔다는 계약 README의 경고를 따른다).
"""

import warnings

import pytest
import torch

from Model.anchor import GainAnchor


def _gain(values):
    return torch.tensor(values, dtype=torch.float32)


class TestOffMode:
    def test_off_returns_zero_scalar(self):
        anchor = GainAnchor(mode="off", lam=5.0)
        gain = _gain([1.0, 2.0, 0.5, 1.0])
        penalty = anchor(gain)
        assert penalty.shape == ()
        assert penalty.item() == 0.0

    def test_off_ignores_lam_and_c_k(self):
        """off는 lam·c_k가 뭐든 항상 0이다."""
        c_k = torch.tensor([0.1, -0.2, 0.3, 0.0])
        anchor = GainAnchor(mode="off", lam=99.0, c_k=c_k)
        gain = _gain([10.0, -3.0, 5.0, 1.0])
        assert anchor(gain).item() == 0.0

    def test_off_dtype_device_matches_gain(self):
        anchor = GainAnchor(mode="off", lam=1.0)
        gain = torch.zeros(3, dtype=torch.float64)
        penalty = anchor(gain)
        assert penalty.dtype == torch.float64


class TestL2Mode:
    def test_l2_penalty_matches_closed_form(self):
        anchor = GainAnchor(mode="l2", lam=2.0)
        gain = _gain([1.5, 0.5, 1.0, 2.0])
        expected = (2.0 / 2) * ((gain - 1.0) ** 2).sum()
        assert torch.allclose(anchor(gain), expected)

    def test_l2_penalty_is_nonneg(self):
        anchor = GainAnchor(mode="l2", lam=3.0)
        for _ in range(20):
            gain = torch.randn(16)
            assert anchor(gain).item() >= 0.0

    def test_l2_zero_at_noop(self):
        anchor = GainAnchor(mode="l2", lam=7.0)
        gain = torch.ones(8)
        assert anchor(gain).item() == 0.0


class TestCkMode:
    def test_ck_requires_c_k(self):
        with pytest.raises(ValueError):
            GainAnchor(mode="ck", lam=1.0)

    def test_ck_negative_clamped_to_zero(self):
        """음수 c_k(제거가 오히려 도움된 latent)는 보존 대상이 아니므로 가중치 0."""
        c_k = torch.tensor([-1.0, 0.0, 0.0])
        anchor = GainAnchor(mode="ck", lam=4.0, c_k=c_k)
        gain = _gain([5.0, 5.0, 5.0])  # latent 0만 크게 벗어나도 페널티는 0
        assert anchor(gain).item() == 0.0

    def test_ck_weight_normalized_to_unit_mean_over_positive(self):
        """0이 아닌 가중치의 평균이 1이 되도록 정규화한다(모듈 docstring 결정 2)."""
        c_k = torch.tensor([0.002, 0.004, -0.001, 0.0])
        anchor = GainAnchor(mode="ck", lam=1.0, c_k=c_k)
        assert anchor._ck_weight[2].item() == 0.0  # 음수 -> 0
        assert anchor._ck_weight[3].item() == 0.0  # 원래 0(제외된 latent) -> 0
        positive = anchor._ck_weight[anchor._ck_weight > 0]
        assert torch.allclose(positive.mean(), torch.tensor(1.0), atol=1e-6)
        # 상대 크기는 보존된다: latent1(0.004)이 latent0(0.002)의 정확히 2배
        assert torch.allclose(anchor._ck_weight[1] / anchor._ck_weight[0], torch.tensor(2.0))

    def test_ck_all_nonpositive_degenerates_to_zero_penalty(self):
        """전부 0 이하인 c_k는 정규화 분모가 없어 가중치를 전부 0으로 둔다."""
        c_k = torch.tensor([-0.1, -0.2, 0.0])
        anchor = GainAnchor(mode="ck", lam=100.0, c_k=c_k)
        gain = _gain([50.0, -20.0, 3.0])
        assert anchor(gain).item() == 0.0

    def test_ck_penalty_matches_closed_form_with_normalized_weight(self):
        c_k = torch.tensor([0.001, 0.003])
        anchor = GainAnchor(mode="ck", lam=2.0, c_k=c_k)
        gain = _gain([1.5, 0.0])
        w = anchor._ck_weight
        expected = (2.0 / 2) * (w * (gain - 1.0) ** 2).sum()
        assert torch.allclose(anchor(gain), expected)

    def test_ck_shape_mismatch_raises(self):
        c_k = torch.tensor([0.1, 0.2, 0.3])
        anchor = GainAnchor(mode="ck", lam=1.0, c_k=c_k)
        with pytest.raises(ValueError):
            anchor(torch.ones(4))


class TestInvalidMode:
    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            GainAnchor(mode="bogus", lam=1.0)


class TestLamZeroMatchesOff:
    """M5 DoD: lam=0이면 mode와 무관하게 off와 수치적으로(비트 단위로) 동일하다."""

    @pytest.mark.parametrize("mode", ["l2", "ck"])
    def test_lam_zero_bitwise_equals_off(self, mode):
        gain = torch.randn(10)
        c_k = torch.rand(10) if mode == "ck" else None
        anchor_zero = GainAnchor(mode=mode, lam=0.0, c_k=c_k)
        anchor_off = GainAnchor(mode="off", lam=0.0)
        penalty_zero = anchor_zero(gain)
        penalty_off = anchor_off(gain)
        assert torch.equal(penalty_zero, penalty_off)
        assert penalty_zero.item() == 0.0

    def test_lam_zero_with_extreme_gain_still_zero(self):
        """gain이 1에서 크게 벗어나도 lam=0이면 페널티가 정확히 0이어야 한다
        (0 * 유한값 == 0이 IEEE754에서 성립함을 실제로 확인)."""
        gain = _gain([1e6, -1e6, 0.0, 1e-9])
        anchor = GainAnchor(mode="l2", lam=0.0)
        assert anchor(gain).item() == 0.0


class TestNoTrainableParameters:
    def test_no_parameters_regardless_of_mode(self):
        """전역 불변식: GainAnchor는 학습 파라미터를 갖지 않는다(gain만 학습)."""
        c_k = torch.rand(5)
        for anchor in [
            GainAnchor(mode="off", lam=1.0),
            GainAnchor(mode="l2", lam=1.0),
            GainAnchor(mode="ck", lam=1.0, c_k=c_k),
        ]:
            assert list(anchor.parameters()) == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 미가용 환경")
class TestCuda:
    def test_ck_mode_on_cuda(self):
        c_k = torch.tensor([0.001, 0.002, -0.003]).cuda()
        anchor = GainAnchor(mode="ck", lam=1.0, c_k=c_k).cuda()
        gain = torch.tensor([1.2, 0.8, 5.0]).cuda()
        penalty = anchor(gain)
        assert penalty.device.type == "cuda"
        assert penalty.item() >= 0.0

    def test_lam_zero_bitwise_equals_off_on_cuda(self):
        gain = torch.randn(8).cuda()
        anchor_zero = GainAnchor(mode="l2", lam=0.0).cuda()
        anchor_off = GainAnchor(mode="off", lam=0.0).cuda()
        assert torch.equal(anchor_zero(gain), anchor_off(gain))


def test_ck_mode_warns_when_no_positive_ck():
    """양수 c_k가 하나도 없으면 anchor가 항상 0을 반환해 off와 구별되지 않는다.
    4B의 anchor 축이 '차이 없음'으로 나올 때 원인을 설계가 아니라 c_k 입력에서
    찾을 수 있어야 하므로, 조용히 넘어가지 않고 경고해야 한다."""
    c_k = torch.tensor([-0.1, 0.0, -0.02, 0.0])
    with pytest.warns(RuntimeWarning, match="양수 c_k가 하나도 없다"):
        anchor = GainAnchor(mode="ck", lam=1.0, c_k=c_k)
    gain = torch.tensor([2.0, 3.0, 0.5, -1.0])
    assert torch.equal(anchor(gain), torch.zeros(()))


def test_ck_mode_does_not_warn_when_some_ck_positive():
    """정상 입력(양수가 하나라도 있음)에서는 경고하지 않는다."""
    c_k = torch.tensor([-0.1, 1e-3, 0.0, 2e-3])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        GainAnchor(mode="ck", lam=1.0, c_k=c_k)
