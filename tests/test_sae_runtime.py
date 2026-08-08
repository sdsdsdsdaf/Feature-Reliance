"""M1 계약 검증: FrozenSAE 정규화 왕복 · decode_delta · frozen grad · FVU/L0 정의 일치."""

import math
import json
import os

import pytest
import torch

from Model.SAE import VanillaL1SAE
from Model.sae_runtime import FrozenSAE
from Utils.SAE_utils import evaluate_sae_tokens

CHECKPOINT_PATH = "outputs/reservoir_sae/vit_b_sae.pt"


def _make_toy_sae(input_dim=8, hidden_dim=16, seed=0):
    """작은 크기의 VanillaL1SAE를 재현 가능한 시드로 만든다(빠른 unit test용)."""
    torch.manual_seed(seed)
    b_dec_init = torch.randn(input_dim)
    sae = VanillaL1SAE(input_dim, hidden_dim, b_dec_init=b_dec_init, dec_bias_mode="zero")
    sae.set_decoder_norm_to_unit_norm()
    return sae


def _make_toy_frozen_sae(input_dim=8, hidden_dim=16, seed=0, active_threshold=0.2):
    """FrozenSAE 인스턴스를 실 checkpoint 없이 조립한다(정규화 통계는 무작위지만 고정 시드)."""
    torch.manual_seed(seed + 1)
    sae = _make_toy_sae(input_dim=input_dim, hidden_dim=hidden_dim, seed=seed)
    token_mean = torch.randn(1, input_dim)
    token_std = torch.rand(1, input_dim) + 0.5  # 0으로 나뉘지 않게
    meta = {"model_name": "toy", "target_block": 10, "token_scope": "patch"}
    return FrozenSAE(sae, token_mean, token_std, active_threshold, meta)


class TestNormalizeRoundTrip:
    def test_denormalize_normalize_is_exact(self):
        """normalize -> denormalize 왕복이 원본과 사실상 정확히 같아야 한다."""
        frozen = _make_toy_frozen_sae()
        h = torch.randn(32, 8)
        x = frozen.normalize(h)
        h_back = frozen.denormalize(x)
        torch.testing.assert_close(h_back, h, rtol=0, atol=1e-6)

    def test_normalize_denormalize_is_exact(self):
        """denormalize -> normalize 왕복도 원본과 사실상 정확히 같아야 한다."""
        frozen = _make_toy_frozen_sae()
        x = torch.randn(32, 8)
        h = frozen.denormalize(x)
        x_back = frozen.normalize(h)
        torch.testing.assert_close(x_back, x, rtol=0, atol=1e-6)

    def test_normalize_broadcasts_over_rank3(self):
        """[B,T,768] 형태 입력에도 정규화가 마지막 축 기준으로 broadcast된다."""
        frozen = _make_toy_frozen_sae()
        h = torch.randn(4, 5, 8)
        x = frozen.normalize(h)
        assert x.shape == h.shape


class TestDecodeDelta:
    def test_zero_delta_returns_exact_zero(self):
        """code_delta가 0이면 decode_delta는 정확히 0을 반환한다(M2 no-op 비트 동일성의 근거)."""
        frozen = _make_toy_frozen_sae()
        zero_delta = torch.zeros(10, frozen.hidden_dim)
        out = frozen.decode_delta(zero_delta)
        assert torch.equal(out, torch.zeros(10, frozen.input_dim))

    def test_decode_delta_matches_two_decode_subtraction_closely(self):
        """decode_delta(Δ)는 denormalize(decode(z+Δ)) - denormalize(decode(z))와 (수치상) 일치한다."""
        frozen = _make_toy_frozen_sae()
        z = torch.rand(10, frozen.hidden_dim)
        delta = torch.randn(10, frozen.hidden_dim) * 0.01
        direct = frozen.decode_delta(delta)
        via_two_decodes = frozen.denormalize(frozen.decode(z + delta)) - frozen.denormalize(frozen.decode(z))
        torch.testing.assert_close(direct, via_two_decodes, rtol=1e-4, atol=1e-5)

    def test_decode_delta_shape(self):
        """[N,K] -> [N,input_dim] 형태 변환을 확인한다."""
        frozen = _make_toy_frozen_sae()
        delta = torch.randn(7, frozen.hidden_dim)
        out = frozen.decode_delta(delta)
        assert out.shape == (7, frozen.input_dim)


class TestFrozenGrad:
    def test_encode_is_always_no_grad(self):
        """encode 결과에는 grad_fn이 없어야 한다(상수 취급)."""
        frozen = _make_toy_frozen_sae()
        h = torch.randn(5, 8, requires_grad=True)
        z = frozen.encode(h)
        assert z.requires_grad is False
        assert z.grad_fn is None

    def test_frozen_params_have_no_grad_after_backward(self):
        """encode -> decode 왕복 뒤 backward를 걸어도 SAE 파라미터의 .grad는 None이다."""
        frozen = _make_toy_frozen_sae()
        h = torch.randn(6, 8)
        z = frozen.encode(h)
        z = z.clone().requires_grad_(True)  # decode 쪽 grad 경로만 별도로 확인
        out = frozen.decode(z)
        out.sum().backward()
        for name, p in frozen.sae.named_parameters():
            assert p.grad is None, f"{name}에 grad가 채워짐"

    def test_sae_params_frozen(self):
        """FrozenSAE 내부 sae 파라미터는 전부 requires_grad=False다."""
        frozen = _make_toy_frozen_sae()
        for p in frozen.sae.parameters():
            assert p.requires_grad is False


class TestGlobalInvariant:
    def test_no_trainable_parameters_at_all(self):
        """전역 불변식: M1에는 gain이 없으므로 학습 가능한 파라미터가 전혀 없어야 한다."""
        frozen = _make_toy_frozen_sae()
        trainable = [n for n, p in frozen.named_parameters() if p.requires_grad]
        assert trainable == [], f"학습 가능한 파라미터 발견: {trainable}"


class TestFvuL0Definitions:
    def test_fvu_matches_evaluate_sae_tokens_normalized_mse(self):
        """FrozenSAE.fvu가 Utils.SAE_utils.evaluate_sae_tokens의 normalized_mse와 동일 정의여야 한다."""
        frozen = _make_toy_frozen_sae()
        h = torch.randn(200, frozen.input_dim) * 2.0 + 1.0
        x = frozen.normalize(h)

        fvu = frozen.fvu(h)
        ref = evaluate_sae_tokens(frozen.sae, x, batch_size=64, threshold=frozen.active_threshold, device="cpu")
        assert fvu == pytest.approx(ref["normalized_mse"], rel=1e-5, abs=1e-8)

    def test_l0_matches_evaluate_sae_tokens_mean_l0(self):
        """FrozenSAE.l0이 Utils.SAE_utils.evaluate_sae_tokens의 mean_l0와 동일 정의여야 한다."""
        frozen = _make_toy_frozen_sae()
        h = torch.randn(200, frozen.input_dim) * 2.0 + 1.0
        x = frozen.normalize(h)
        z = frozen.encode(x, normalized=True)

        l0 = frozen.l0(z)
        ref = evaluate_sae_tokens(frozen.sae, x, batch_size=64, threshold=frozen.active_threshold, device="cpu")
        assert l0 == pytest.approx(ref["mean_l0"], rel=1e-5, abs=1e-8)

    def test_l0_accepts_explicit_threshold(self):
        """l0(z, threshold=...)가 active_threshold를 무시하고 명시된 값을 쓴다."""
        frozen = _make_toy_frozen_sae()
        z = torch.tensor([[0.0, 0.1, 0.3], [0.5, 0.5, 0.5]])
        # row0: >0.2 -> {0.3} = 1개, row1: >0.2 -> {0.5,0.5,0.5} = 3개 -> mean 2.0
        assert frozen.l0(z, threshold=0.2) == pytest.approx(2.0)
        # row0: >0.0 -> {0.1,0.3} = 2개, row1: >0.0 -> 3개 -> mean 2.5
        assert frozen.l0(z, threshold=0.0) == pytest.approx(2.5)


class TestReconstruct:
    def test_reconstruct_is_encode_then_decode_raw(self):
        """reconstruct(h) == decode_raw(encode(h))."""
        frozen = _make_toy_frozen_sae()
        h = torch.randn(9, frozen.input_dim)
        torch.testing.assert_close(frozen.reconstruct(h), frozen.decode_raw(frozen.encode(h)))


@pytest.mark.skipif(not os.path.exists(CHECKPOINT_PATH), reason="실 checkpoint 없음")
class TestFromCheckpoint:
    def test_from_checkpoint_measured_shapes(self):
        """ckpt에서 복원한 필드가 phaseM.md에 기록된 실측치와 일치하는지 확인한다."""
        frozen = FrozenSAE.from_checkpoint(CHECKPOINT_PATH, device="cpu")
        assert frozen.input_dim == 768
        assert frozen.hidden_dim == 12288
        assert frozen.token_mean.shape == (1, 768)
        assert frozen.token_std.shape == (1, 768)
        assert frozen.active_threshold == pytest.approx(0.2)
        assert frozen.meta["model_name"] == "vit_base_patch16_224"
        assert frozen.meta["target_block"] == 10
        assert frozen.meta["token_scope"] == "patch"
        # W_dec 행 unit-norm 유지 확인
        row_norms = frozen.sae.W_dec.norm(dim=1)
        torch.testing.assert_close(row_norms, torch.ones_like(row_norms), rtol=1e-3, atol=1e-3)

    def test_from_checkpoint_no_trainable_parameters(self):
        """실 checkpoint 로드 후에도 전역 불변식(학습 파라미터 0개)을 만족해야 한다."""
        frozen = FrozenSAE.from_checkpoint(CHECKPOINT_PATH, device="cpu")
        trainable = [n for n, p in frozen.named_parameters() if p.requires_grad]
        assert trainable == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 미가용")
class TestCudaDevice:
    """CUDA에서만 드러나는 device 불일치를 잡는다.

    M1은 CPU에서 개발·검증됐지만 실제 실험(T1.1 이후)은 전부 GPU에서 돈다.
    CPU 테스트만으로는 '함수 안에서 CPU 리터럴 텐서를 만들어 쓰는' 류의 버그가
    보이지 않으므로(실제로 fvu()에 있었다) 전 공개 API를 CUDA에서 한 번씩 태운다."""

    def test_all_public_apis_run_on_cuda(self):
        """CUDA 입력에 대해 encode/decode/decode_delta/decode_raw/reconstruct/fvu/l0가
        device 오류 없이 돌고, 결과가 전부 같은 device에 남는지 확인한다."""
        frozen = FrozenSAE.from_checkpoint(CHECKPOINT_PATH, device="cuda")
        h = torch.randn(64, frozen.input_dim, device="cuda")

        z = frozen.encode(h)
        assert z.device.type == "cuda"
        assert frozen.decode(z).device.type == "cuda"
        assert frozen.decode_delta(torch.zeros_like(z)).device.type == "cuda"
        assert frozen.decode_raw(z).device.type == "cuda"
        assert frozen.reconstruct(h).device.type == "cuda"

        # 스칼라로 떨어지는 두 지표는 값이 유한하기만 하면 된다.
        assert math.isfinite(frozen.fvu(h))
        assert math.isfinite(frozen.l0(z))

    def test_cpu_and_cuda_agree(self):
        """같은 입력에 대해 CPU와 CUDA의 FVU가 일치해야 한다.
        어긋나면 device별로 다른 경로를 타고 있다는 뜻이다."""
        h = torch.randn(256, 768)
        cpu_fvu = FrozenSAE.from_checkpoint(CHECKPOINT_PATH, device="cpu").fvu(h)
        cuda_fvu = FrozenSAE.from_checkpoint(CHECKPOINT_PATH, device="cuda").fvu(h.cuda())
        assert cpu_fvu == pytest.approx(cuda_fvu, rel=1e-4)
