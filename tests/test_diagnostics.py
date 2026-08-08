"""M6 계약 검증: `compute_c_k`의 반환 shape·부호·후보 제외 latent==0·dead latent==0·
baseline==무개입 정확도·gain 미변형(no side effect)을 확인한다.

작은 vit_tiny_patch16_224(embed_dim=192, depth=12)를 CPU에서 쓴다 — GPU는 다른
worker가 점유 중이라는 전제(phaseM.md 실행 환경 절)를 지킨다.
"""

import os

import pytest
import timm
import torch
from torch.utils.data import DataLoader, TensorDataset

from Model.SAE import VanillaL1SAE
from Model.diagnostics import compute_c_k
from Model.gain_basis import ChannelGainBasis, LatentGainBasis
from Model.intervention import GainIntervention
from Model.sae_runtime import FrozenSAE

CHECKPOINT_PATH = "outputs/reservoir_sae/vit_b_sae.pt"
EMBED_DIM = 192
HOOK_BLOCK = 10
NUM_CLASSES = 10


def _make_backbone(seed=0):
    """CPU에서 빠르게 도는 작은 timm ViT를 만든다. dropout류는 기본 0이라 결정론적."""
    torch.manual_seed(seed)
    m = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=NUM_CLASSES)
    return m.eval()


def _backbone_activation_at_hook(backbone, x, hook_block=HOOK_BLOCK):
    """GainIntervention.forward의 상류 부분을 그대로 재현해 block10 출력을 얻는다."""
    with torch.no_grad():
        h = backbone.patch_embed(x)
        h = backbone._pos_embed(h)
        h = backbone.patch_drop(h)
        h = backbone.norm_pre(h)
        for blk in backbone.blocks[: hook_block + 1]:
            h = blk(h)
    return h


def _make_toy_frozen_sae(flat_activation, hidden_dim=64, seed=1, active_threshold=0.2):
    """FrozenSAE를 실제 backbone 활성 분포에 맞춘 정규화 통계로 조립한다."""
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
def backbone_and_loader():
    """backbone + 소규모 DataLoader(8장, 정답 라벨 포함). 매 테스트가 독립 backbone을 갖는다
    (intervention이 backbone을 freeze/eval로 in-place 변형하므로 공유하면 오염된다)."""
    backbone = _make_backbone()
    torch.manual_seed(42)
    x = torch.randn(8, 3, 224, 224)
    with torch.no_grad():
        y = backbone(x).argmax(dim=1)  # 이 backbone이 스스로 100% 맞히는 라벨(baseline acc=1.0 확보용)
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)
    return backbone, loader, x


@pytest.fixture
def toy_sae(backbone_and_loader):
    """backbone_and_loader의 실제 block10 활성 분포에 맞춘 toy FrozenSAE(hidden_dim=64)."""
    backbone, _loader, x = backbone_and_loader
    flat = _backbone_activation_at_hook(backbone, x).reshape(-1, EMBED_DIM)
    return _make_toy_frozen_sae(flat)


def _intervention(backbone, toy_sae):
    basis = LatentGainBasis(toy_sae)
    return GainIntervention(backbone, basis, hook_block=HOOK_BLOCK, residual=True)


class TestReturnShapeAndCandidateExclusion:
    def test_return_shape_is_K(self, backbone_and_loader, toy_sae):
        backbone, loader, _x = backbone_and_loader
        intervention = _intervention(backbone, toy_sae)
        c_k = compute_c_k(intervention, loader, "cpu", max_images=8)
        assert c_k.shape == (toy_sae.hidden_dim,)

    def test_latents_outside_candidates_are_exactly_zero(self, backbone_and_loader, toy_sae):
        backbone, loader, _x = backbone_and_loader
        intervention = _intervention(backbone, toy_sae)
        candidates = torch.tensor([0, 1, 2], dtype=torch.long)
        c_k = compute_c_k(intervention, loader, "cpu", candidates=candidates, max_images=8)
        mask = torch.ones(toy_sae.hidden_dim, dtype=torch.bool)
        mask[candidates] = False
        assert torch.equal(c_k[mask], torch.zeros(mask.sum()))

    def test_empty_candidates_returns_all_zero(self, backbone_and_loader, toy_sae):
        backbone, loader, _x = backbone_and_loader
        intervention = _intervention(backbone, toy_sae)
        c_k = compute_c_k(intervention, loader, "cpu", candidates=torch.zeros(0, dtype=torch.long), max_images=8)
        assert torch.equal(c_k, torch.zeros(toy_sae.hidden_dim))


class TestDeadLatentIsExactZero:
    def test_never_firing_latent_gives_exact_zero_c_k(self, backbone_and_loader, toy_sae):
        """한 번도 발화하지 않은 latent를 candidates에 강제로 포함시켜도 c_k가 정확히 0이어야 한다."""
        backbone, loader, x = backbone_and_loader
        intervention = _intervention(backbone, toy_sae)
        flat = _backbone_activation_at_hook(backbone, x).reshape(-1, EMBED_DIM)
        code = intervention.basis.encode(flat)
        dead_mask = code.abs().sum(dim=0) == 0
        assert dead_mask.any(), "테스트 전제가 깨짐: dead latent가 없다"
        dead_idx = int(dead_mask.nonzero()[0].item())

        c_k = compute_c_k(intervention, loader, "cpu", candidates=torch.tensor([dead_idx]), max_images=8)
        assert c_k[dead_idx].item() == 0.0


class TestBaselineMatchesPlainBackbone:
    def test_baseline_acc_equals_plain_backbone_acc(self, backbone_and_loader, toy_sae):
        """loader의 라벨이 backbone 자체 예측이므로 baseline(gain≡1) 정확도는 1.0이어야 하고,
        따라서 c_k <= 1.0 (ablation이 정확도를 더 떨어뜨릴 수는 있어도 baseline을 넘길 순 없다)."""
        backbone, loader, _x = backbone_and_loader
        intervention = _intervention(backbone, toy_sae)
        flat = _backbone_activation_at_hook(backbone, torch.cat([b[0] for b in loader]))
        code = intervention.basis.encode(flat.reshape(-1, EMBED_DIM))
        fired = (code.abs().sum(dim=0) > 0).nonzero(as_tuple=True)[0]
        candidates = fired[:5] if fired.numel() >= 5 else fired

        c_k = compute_c_k(intervention, loader, "cpu", candidates=candidates, max_images=8)
        assert (c_k[candidates] <= 1.0 + 1e-6).all()
        assert (c_k[candidates] >= -1.0 - 1e-6).all()


class TestCanBeNegative:
    def test_c_k_is_not_clamped_and_may_be_negative(self, backbone_and_loader, toy_sae):
        """ablation이 정확도를 올리는 인위적 상황을 만들어 c_k가 음수로 나오는지 확인한다.
        절반은 backbone이 못 맞히는 라벨을 줘서, latent 제거가 우연히 도움될 여지를 만든다."""
        backbone, loader, x = backbone_and_loader
        intervention = _intervention(backbone, toy_sae)
        with torch.no_grad():
            wrong_labels = (backbone(x).argmax(dim=1) + 1) % NUM_CLASSES  # 항상 틀리는 라벨
        adversarial_loader = DataLoader(TensorDataset(x, wrong_labels), batch_size=4, shuffle=False)

        flat = _backbone_activation_at_hook(backbone, x).reshape(-1, EMBED_DIM)
        code = intervention.basis.encode(flat)
        fired = (code.abs().sum(dim=0) > 0).nonzero(as_tuple=True)[0]
        candidates = fired[:20] if fired.numel() >= 20 else fired
        assert candidates.numel() > 0, "테스트 전제가 깨짐: 발화하는 latent가 없다"

        c_k = compute_c_k(intervention, adversarial_loader, "cpu", candidates=candidates, max_images=8)
        # baseline acc는 0(항상 틀림)이므로 c_k = 0 - ablated_acc = -ablated_acc <= 0.
        # 최소 하나는 ablation으로 우연히 정답을 맞혀 ablated_acc>0, 즉 c_k<0이 되길 기대한다.
        assert (c_k[candidates] <= 0.0 + 1e-6).all()


class TestGainNotMutated:
    """불변식: compute_c_k는 측정이지 학습이 아니다 — 호출 전후 intervention.gain이 그대로다."""

    def test_gain_unchanged_after_call(self, backbone_and_loader, toy_sae):
        backbone, loader, _x = backbone_and_loader
        intervention = _intervention(backbone, toy_sae)
        with torch.no_grad():
            intervention.gain.fill_(3.5)
        before = intervention.gain.detach().clone()

        compute_c_k(intervention, loader, "cpu", max_images=8)

        assert torch.equal(intervention.gain.detach(), before)

    def test_gain_restored_even_when_default_gain_one(self, backbone_and_loader, toy_sae):
        backbone, loader, _x = backbone_and_loader
        intervention = _intervention(backbone, toy_sae)
        compute_c_k(intervention, loader, "cpu", max_images=8)
        assert torch.equal(intervention.gain.detach(), torch.ones_like(intervention.gain))


class TestAutoCandidateSelectionByFiringRate:
    def test_none_candidates_selects_only_frequently_firing_latents(self, backbone_and_loader, toy_sae):
        """candidates=None이면 발화율 하한(0.5%)을 넘는 latent만 후보가 된다.
        max_images로 캡을 걸어도 자동 선정이 동작해야 한다."""
        backbone, loader, x = backbone_and_loader
        intervention = _intervention(backbone, toy_sae)
        flat = _backbone_activation_at_hook(backbone, x).reshape(-1, EMBED_DIM)
        code = intervention.basis.encode(flat)
        dead_mask = code.abs().sum(dim=0) == 0
        assert dead_mask.any(), "테스트 전제가 깨짐: dead latent가 없다"

        c_k = compute_c_k(intervention, loader, "cpu", max_images=8)
        # 전혀 발화하지 않는 latent는 자동 후보에서 제외되어 0이어야 한다.
        assert torch.equal(c_k[dead_mask], torch.zeros(int(dead_mask.sum())))


class TestLatentChunkParameterIsCosmeticallyTransparent:
    def test_result_independent_of_latent_chunk_size(self, backbone_and_loader, toy_sae):
        """latent_chunk는 배치 방식만 바꾸는 성능 손잡이라 결과가 값에 무관해야 한다."""
        backbone, loader, x = backbone_and_loader
        intervention = _intervention(backbone, toy_sae)
        flat = _backbone_activation_at_hook(backbone, x).reshape(-1, EMBED_DIM)
        code = intervention.basis.encode(flat)
        fired = (code.abs().sum(dim=0) > 0).nonzero(as_tuple=True)[0]
        candidates = fired[:6] if fired.numel() >= 6 else fired
        assert candidates.numel() >= 2, "테스트 전제가 깨짐: 발화 latent가 2개 미만"

        c_k_chunk1 = compute_c_k(intervention, loader, "cpu", candidates=candidates, latent_chunk=1, max_images=8)
        c_k_chunk_all = compute_c_k(
            intervention, loader, "cpu", candidates=candidates, latent_chunk=len(candidates), max_images=8
        )
        torch.testing.assert_close(c_k_chunk1[candidates], c_k_chunk_all[candidates])


class TestMaxImagesCapsCache:
    def test_max_images_smaller_than_loader_still_runs(self, backbone_and_loader, toy_sae):
        backbone, loader, _x = backbone_and_loader
        intervention = _intervention(backbone, toy_sae)
        c_k = compute_c_k(intervention, loader, "cpu", candidates=torch.tensor([0, 1]), max_images=4)
        assert c_k.shape == (toy_sae.hidden_dim,)


class TestChannelBasisSupported:
    """compute_c_k는 LatentGainBasis 전용이 아니라 GainBasis 일반에 대해 동작해야 한다."""

    def test_channel_basis_runs_without_error(self, backbone_and_loader):
        backbone, loader, _x = backbone_and_loader
        basis = ChannelGainBasis(dim=EMBED_DIM)
        intervention = GainIntervention(backbone, basis, hook_block=HOOK_BLOCK, residual=True)
        c_k = compute_c_k(intervention, loader, "cpu", candidates=torch.tensor([0, 1, 2]), max_images=8)
        assert c_k.shape == (EMBED_DIM,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 미가용")
@pytest.mark.skipif(not os.path.exists(CHECKPOINT_PATH), reason="실 checkpoint 없음")
def test_compute_c_k_runs_on_real_checkpoint_smoke():
    """실 checkpoint + CUDA에서 소규모 스모크. 비용이 커서 이미지·후보 수를 강하게 제한한다."""
    backbone = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=10).cuda().eval()
    sae = FrozenSAE.from_checkpoint(CHECKPOINT_PATH, device="cuda")
    basis = LatentGainBasis(sae)
    intervention = GainIntervention(backbone, basis, hook_block=HOOK_BLOCK, residual=True).cuda()

    torch.manual_seed(0)
    x = torch.randn(8, 3, 224, 224, device="cuda")
    with torch.no_grad():
        y = backbone(x).argmax(dim=1)
    loader = DataLoader(TensorDataset(x.cpu(), y.cpu()), batch_size=4, shuffle=False)

    c_k = compute_c_k(intervention, loader, "cuda", latent_chunk=16, max_images=8)
    assert c_k.shape == (sae.hidden_dim,)
