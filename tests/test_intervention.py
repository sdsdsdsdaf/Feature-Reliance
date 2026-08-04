"""M2 계약 검증: gain=1 비트 동일 no-op · frozen grad 경로 · gain 유일 학습 파라미터 ·
gradient sparsity · reset · residual on/off 전환 · 3 basis 교체.

작은 vit_tiny_patch16_224(embed_dim=192, depth=12)를 CPU에서 쓴다 — GPU는 다른
worker가 점유 중이라는 전제(phaseM.md 실행 환경 절)를 지킨다.
"""

import pytest
import torch
import timm

from Model.SAE import VanillaL1SAE
from Model.sae_runtime import FrozenSAE
from Model.gain_basis import ChannelGainBasis, LatentGainBasis, RandomDictGainBasis
from Model.intervention import GainIntervention

EMBED_DIM = 192
HOOK_BLOCK = 10  # vit_tiny도 depth=12라 production과 동일하게 blocks[11]만 downstream


def _make_backbone(seed=0):
    """CPU에서 빠르게 도는 작은 timm ViT를 만든다. dropout류는 기본 0이라 결정론적."""
    torch.manual_seed(seed)
    m = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10)
    return m.eval()


def _backbone_activation_at_hook(backbone, x, hook_block=HOOK_BLOCK):
    """GainIntervention.forward의 상류 부분을 그대로 재현해 block10 출력을 얻는다.
    실측 활성으로 SAE 정규화 통계를 잡을 때 쓴다(dead-latent 재현성 확보용)."""
    with torch.no_grad():
        h = backbone.patch_embed(x)
        h = backbone._pos_embed(h)
        h = backbone.patch_drop(h)
        h = backbone.norm_pre(h)
        for blk in backbone.blocks[: hook_block + 1]:
            h = blk(h)
    return h


def _make_toy_frozen_sae(flat_activation, hidden_dim=384, seed=1, active_threshold=0.2):
    """FrozenSAE를 실제 backbone 활성 분포에 맞춘 정규화 통계로 조립한다(빠른 unit test용)."""
    torch.manual_seed(seed)
    input_dim = flat_activation.shape[-1]
    b_dec_init = torch.randn(input_dim)
    sae = VanillaL1SAE(input_dim, hidden_dim, b_dec_init=b_dec_init, dec_bias_mode="zero")
    sae.set_decoder_norm_to_unit_norm()
    token_mean = flat_activation.mean(0, keepdim=True)
    token_std = flat_activation.std(0, keepdim=True) + 0.5
    meta = {"model_name": "toy", "target_block": HOOK_BLOCK, "token_scope": "patch"}
    return FrozenSAE(sae, token_mean, token_std, active_threshold, meta)


@pytest.fixture
def backbone_and_x():
    """공용 backbone + 고정 입력 배치. 각 테스트가 독립된 backbone 인스턴스를 갖는다
    (intervention이 backbone을 freeze/eval로 in-place 변형하므로 공유하면 오염된다)."""
    backbone = _make_backbone()
    torch.manual_seed(42)
    x = torch.randn(2, 3, 224, 224)
    return backbone, x


@pytest.fixture
def toy_sae(backbone_and_x):
    """backbone_and_x의 실제 block10 활성 분포에 맞춘 toy FrozenSAE."""
    backbone, x = backbone_and_x
    flat = _backbone_activation_at_hook(backbone, x).reshape(-1, EMBED_DIM)
    return _make_toy_frozen_sae(flat)


def _latent_basis(toy_sae):
    return LatentGainBasis(toy_sae)


def _channel_basis():
    return ChannelGainBasis(dim=EMBED_DIM)


def _random_dict_basis(toy_sae, target_l0=None):
    # 3으로 두면 이 toy 설정(hidden_dim=384, 392 토큰)에서 안정적으로 dead latent가
    # 생겨(실측 29/384) gradient sparsity 테스트가 흔들리지 않는다.
    if target_l0 is None:
        target_l0 = 3
    return RandomDictGainBasis(toy_sae, target_l0=target_l0, seed=7)


class TestBitIdenticalNoOp:
    """불변식 2: gain==1이면 residual=True에서 무개입 forward와 비트 단위로 동일하다."""

    @pytest.mark.parametrize("basis_factory", ["latent", "channel", "random_dict"])
    def test_gain_one_matches_unmodified_backbone(self, backbone_and_x, toy_sae, basis_factory):
        backbone, x = backbone_and_x
        basis = {
            "latent": lambda: _latent_basis(toy_sae),
            "channel": lambda: _channel_basis(),
            "random_dict": lambda: _random_dict_basis(toy_sae),
        }[basis_factory]()

        # 개입 없는 참조: contract pseudocode와 동일한 파이프라인(backbone.forward와 등가)
        with torch.no_grad():
            ref_logits = backbone(x)
            ref_feat = backbone.forward_head(backbone.forward_features(x), pre_logits=True)

        intervention = GainIntervention(backbone, basis, hook_block=HOOK_BLOCK, residual=True)
        with torch.no_grad():
            logits, feat = intervention(x)

        assert torch.equal(logits, ref_logits), "gain=1인데 logits가 무개입 forward와 다르다"
        assert torch.equal(feat, ref_feat), "gain=1인데 feat가 무개입 forward와 다르다"

    def test_delta_is_exact_zero_at_gain_one(self, toy_sae, backbone_and_x):
        """delta(code, gain=1)이 근사가 아니라 정확히 0 텐서다."""
        backbone, x = backbone_and_x
        basis = _latent_basis(toy_sae)
        flat = _backbone_activation_at_hook(backbone, x).reshape(-1, EMBED_DIM)
        code = basis.encode(flat)
        gain = torch.ones(basis.gain_dim)
        delta = basis.delta(code, gain)
        assert torch.equal(delta, torch.zeros_like(delta))


class TestGlobalInvariant:
    """전역 불변식: gain 벡터 하나만 학습 파라미터."""

    def test_trainable_parameters_is_exactly_gain(self, backbone_and_x, toy_sae):
        backbone, x = backbone_and_x
        intervention = GainIntervention(backbone, _latent_basis(toy_sae), hook_block=HOOK_BLOCK)
        params = intervention.trainable_parameters()
        assert len(params) == 1
        assert params[0] is intervention.gain

    def test_named_parameters_trainable_is_gain_only(self, backbone_and_x, toy_sae):
        backbone, x = backbone_and_x
        intervention = GainIntervention(backbone, _latent_basis(toy_sae), hook_block=HOOK_BLOCK)
        trainable = [n for n, p in intervention.named_parameters() if p.requires_grad]
        assert trainable == ["gain"], f"gain 외 학습 파라미터 발견: {trainable}"


class TestFrozenGradPath:
    """불변식 3: blocks[:hook_block+1]과 SAE는 backward 후에도 .grad가 None."""

    def test_upstream_blocks_and_sae_never_receive_grad(self, backbone_and_x, toy_sae):
        backbone, x = backbone_and_x
        intervention = GainIntervention(backbone, _latent_basis(toy_sae), hook_block=HOOK_BLOCK)

        logits, feat = intervention(x)
        (logits.sum() + feat.sum()).backward()

        for i, blk in enumerate(backbone.blocks[: HOOK_BLOCK + 1]):
            for name, p in blk.named_parameters():
                assert p.grad is None, f"blocks[{i}].{name}에 grad가 채워짐(상류는 no_grad여야 함)"
        for name, p in toy_sae.sae.named_parameters():
            assert p.grad is None, f"SAE.{name}에 grad가 채워짐(SAE는 frozen)"

    def test_downstream_blocks_have_no_grad_either_since_frozen(self, backbone_and_x, toy_sae):
        """tail(blocks[hook_block+1:])도 frozen이라 weight 자체는 grad가 채워지지 않는다.
        학습 파라미터는 gain 뿐이라는 전역 불변식의 또 다른 표현."""
        backbone, x = backbone_and_x
        intervention = GainIntervention(backbone, _latent_basis(toy_sae), hook_block=HOOK_BLOCK)

        logits, feat = intervention(x)
        (logits.sum() + feat.sum()).backward()

        for name, p in backbone.named_parameters():
            assert p.grad is None, f"backbone.{name}에 grad가 채워짐(backbone 전체가 frozen이어야 함)"

    def test_gain_receives_nonzero_grad(self, backbone_and_x, toy_sae):
        """gain=1이어도 gain 자체의 gradient는 0이 아니다(delta의 gain에 대한 미분은 code라서)."""
        backbone, x = backbone_and_x
        intervention = GainIntervention(backbone, _latent_basis(toy_sae), hook_block=HOOK_BLOCK)

        logits, feat = intervention(x)
        (logits.sum() + feat.sum()).backward()

        assert intervention.gain.grad is not None
        assert not torch.equal(intervention.gain.grad, torch.zeros_like(intervention.gain.grad))


class TestGradientSparsity:
    """한 번도 발화하지 않은 latent는 gain gradient가 정확히 0이다.
    ChannelGainBasis는 code가 조밀(=h 자체)해서 대상에서 제외한다."""

    @pytest.mark.parametrize("basis_name", ["latent", "random_dict"])
    def test_dead_latent_has_exact_zero_gain_grad(self, backbone_and_x, toy_sae, basis_name):
        backbone, x = backbone_and_x
        basis = _latent_basis(toy_sae) if basis_name == "latent" else _random_dict_basis(toy_sae)
        intervention = GainIntervention(backbone, basis, hook_block=HOOK_BLOCK)

        flat = _backbone_activation_at_hook(backbone, x).reshape(-1, EMBED_DIM)
        code = basis.encode(flat)
        dead_mask = code.abs().sum(dim=0) == 0
        assert dead_mask.any(), "테스트 전제가 깨짐: 한 번도 발화하지 않은 latent가 없다"
        dead_idx = int(dead_mask.nonzero()[0].item())

        logits, feat = intervention(x)
        (logits.sum() + feat.sum()).backward()

        assert intervention.gain.grad[dead_idx].item() == 0.0


class TestReset:
    """불변식 4: reset() 후 gain이 정확히 1."""

    def test_reset_restores_gain_to_exactly_one(self, backbone_and_x, toy_sae):
        backbone, x = backbone_and_x
        intervention = GainIntervention(backbone, _latent_basis(toy_sae), hook_block=HOOK_BLOCK)
        with torch.no_grad():
            intervention.gain.fill_(3.5)
        intervention.reset()
        assert torch.equal(intervention.gain, torch.ones_like(intervention.gain))


class TestResidualSwitch:
    """residual=False는 absolute()가 있는 basis만 허용하고, 없으면 __init__에서 즉시 실패한다."""

    def test_residual_false_with_channel_basis_raises_in_init(self, backbone_and_x):
        backbone, x = backbone_and_x
        with pytest.raises(ValueError):
            GainIntervention(backbone, _channel_basis(), hook_block=HOOK_BLOCK, residual=False)

    def test_residual_false_with_latent_basis_does_not_raise_and_runs(self, backbone_and_x, toy_sae):
        backbone, x = backbone_and_x
        intervention = GainIntervention(backbone, _latent_basis(toy_sae), hook_block=HOOK_BLOCK, residual=False)
        with torch.no_grad():
            logits, feat = intervention(x)
        assert logits.shape == (2, 10)
        assert feat.shape[0] == 2

    def test_residual_false_gain_one_differs_from_backbone_by_reconstruction_error(self, backbone_and_x, toy_sae):
        """residual=False는 gain=1이어도 ĥ와의 재구성 오차만큼 무개입 forward와 어긋난다(정확히 같으면 안 된다)."""
        backbone, x = backbone_and_x
        with torch.no_grad():
            ref_logits = backbone(x)
        intervention = GainIntervention(backbone, _latent_basis(toy_sae), hook_block=HOOK_BLOCK, residual=False)
        with torch.no_grad():
            logits, _ = intervention(x)
        assert not torch.equal(logits, ref_logits)


class TestBasisSwap:
    """3 basis가 같은 GainIntervention 루프에 그대로 꽂힌다(shape·타입 교체 확인)."""

    @pytest.mark.parametrize("basis_factory", ["latent", "channel", "random_dict"])
    def test_all_bases_run_in_same_loop(self, backbone_and_x, toy_sae, basis_factory):
        backbone, x = backbone_and_x
        basis = {
            "latent": lambda: _latent_basis(toy_sae),
            "channel": lambda: _channel_basis(),
            "random_dict": lambda: _random_dict_basis(toy_sae),
        }[basis_factory]()

        intervention = GainIntervention(backbone, basis, hook_block=HOOK_BLOCK, residual=True)
        with torch.no_grad():
            logits, feat = intervention(x)
        assert logits.shape == (2, 10)
        assert intervention.gain.shape == (basis.gain_dim,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 미가용")
class TestCudaDevice:
    """M1의 CUDA 스모크와 동일한 취지: device 불일치 버그를 CI 밖에서 잡는다."""

    def test_forward_and_backward_run_on_cuda(self):
        backbone = _make_backbone().cuda()
        torch.manual_seed(0)
        x = torch.randn(2, 3, 224, 224, device="cuda")
        flat = _backbone_activation_at_hook(backbone, x).reshape(-1, EMBED_DIM)
        frozen = _make_toy_frozen_sae(flat.cpu()).cuda()
        # toy_sae 통계는 cpu 활성 기준으로 계산했으므로 .cuda()로 옮겨 device를 맞춘다.
        basis = LatentGainBasis(frozen)
        intervention = GainIntervention(backbone, basis, hook_block=HOOK_BLOCK).cuda()

        logits, feat = intervention(x)
        assert logits.device.type == "cuda"
        assert feat.device.type == "cuda"
        (logits.sum() + feat.sum()).backward()
        assert intervention.gain.grad is not None
        assert intervention.gain.grad.device.type == "cuda"
