"""M8 계약 검증: OnlineTTARunner가 M2(gain 개입)+M3(AdaContrast 손실)+M4(weak/strong aug)를
하나의 스트리밍 루프로 배선하는지 확인한다.

핵심 불변식:
- 평가 프로토콜: 배치는 그 배치 자신의 gain 갱신이 반영되기 '전' 예측으로 채점한다
  (정보 누설 방지 — phaseM.md M8 절).
- `distance_space`는 memory bank의 dim·enqueue 텐서까지 따라간다(h/z/logit 3공간).
- `buffer`(0/정수/"full") · `passes`(1/2/5 등)가 config만 바꿔 전환된다.
- `gate=None`만 지원한다(M7은 T1.1로 dropped) — gate!=None은 NotImplementedError.

작은 vit_tiny_patch16_224(embed_dim=192, depth=12)를 CPU에서 쓴다 — GPU는 다른
worker가 점유 중이라는 전제(phaseM.md 실행 환경 절)를 지킨다. CUDA+실 checkpoint+
실 데이터가 있을 때만 도는 smoke는 맨 아래에 스킵 가드로 분리한다.
"""

import os

# cv2가 dlopen하는 시스템 libgobject가 torch보다 늦게 로드되면 심볼 충돌로 죽는
# 이 conda 환경의 기존 이슈(Utils.Dataset이 cv2를 쓴다) — torch/timm보다 먼저
# import해 회피한다. datasets 모듈 자체를 안 쓰는 대부분의 테스트에는 영향 없다.
import Utils.datasets as datasets_mod  # noqa: E402

import pytest
import timm
import torch

from Model.SAE import VanillaL1SAE
from Model.adacontrast import AdaContrastLoss
from Model.gain_basis import LatentGainBasis
from Model.intervention import GainIntervention
from Model.sae_runtime import FrozenSAE
from Model.tta_runner import OnlineTTARunner, TTAConfig, _pool_code_to_image

CHECKPOINT_PATH = "outputs/reservoir_sae/vit_b_sae.pt"
EMBED_DIM = 192
HOOK_BLOCK = 10
NUM_CLASSES = 5
HIDDEN_DIM = 32  # 작은 toy SAE 코드 차원(=basis.gain_dim)


def _make_backbone(seed=0, num_classes=NUM_CLASSES):
    torch.manual_seed(seed)
    m = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes)
    return m.eval()


def _backbone_activation_at_hook(backbone, x, hook_block=HOOK_BLOCK):
    with torch.no_grad():
        h = backbone.patch_embed(x)
        h = backbone._pos_embed(h)
        h = backbone.patch_drop(h)
        h = backbone.norm_pre(h)
        for blk in backbone.blocks[: hook_block + 1]:
            h = blk(h)
    return h


def _make_toy_frozen_sae(flat_activation, hidden_dim=HIDDEN_DIM, seed=1, active_threshold=0.2):
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
def toy_setup():
    """intervention + loss_fn을 만들 수 있는 최소 구성 요소 묶음."""
    backbone = _make_backbone()
    torch.manual_seed(42)
    x = torch.randn(4, 3, 224, 224)
    flat = _backbone_activation_at_hook(backbone, x).reshape(-1, EMBED_DIM)
    toy_sae = _make_toy_frozen_sae(flat)
    basis = LatentGainBasis(toy_sae)
    return {"backbone": backbone, "x": x, "toy_sae": toy_sae, "basis": basis}


def _make_intervention(toy_setup, residual=True):
    # backbone은 intervention이 in-place로 freeze/eval하므로 매번 새로 만든다.
    backbone = _make_backbone()
    return GainIntervention(backbone, toy_setup["basis"], hook_block=HOOK_BLOCK, residual=residual)


def _make_batch(batch_size=4, seed=0):
    torch.manual_seed(seed)
    x_weak = torch.randn(batch_size, 3, 224, 224)
    x_strong = torch.randn(batch_size, 3, 224, 224)
    y = torch.randint(0, NUM_CLASSES, (batch_size,))
    return x_weak, x_strong, y


class TestTTAConfigValidation:
    def test_invalid_distance_space_raises(self):
        with pytest.raises(ValueError):
            TTAConfig(distance_space="bogus")

    def test_buffer_full_without_capacity_raises(self):
        with pytest.raises(ValueError):
            TTAConfig(buffer="full", full_buffer_capacity=None)

    def test_buffer_full_with_capacity_is_valid(self):
        cfg = TTAConfig(buffer="full", full_buffer_capacity=100)
        assert cfg.buffer == "full"
        assert cfg.full_buffer_capacity == 100

    def test_negative_int_buffer_raises(self):
        with pytest.raises(ValueError):
            TTAConfig(buffer=-1)


class TestDistanceSpaceBankWiring:
    """⚠️ distance_space는 memory bank까지 따라가야 한다 (M8 절)."""

    def test_h_space_bank_dim_matches_feature_dim(self, toy_setup):
        intervention = _make_intervention(toy_setup)
        loss_fn = AdaContrastLoss(num_classes=NUM_CLASSES, distance_space="h")
        config = TTAConfig(distance_space="h", buffer=8)
        runner = OnlineTTARunner(intervention, loss_fn, anchor=None, config=config)
        assert runner.bank.dim == intervention.backbone.num_features == EMBED_DIM

    def test_z_space_bank_dim_matches_basis_gain_dim(self, toy_setup):
        intervention = _make_intervention(toy_setup)
        loss_fn = AdaContrastLoss(num_classes=NUM_CLASSES, distance_space="z")
        config = TTAConfig(distance_space="z", buffer=8)
        runner = OnlineTTARunner(intervention, loss_fn, anchor=None, config=config)
        assert runner.bank.dim == toy_setup["basis"].gain_dim == HIDDEN_DIM

    def test_logit_space_bank_dim_matches_num_classes(self, toy_setup):
        intervention = _make_intervention(toy_setup)
        loss_fn = AdaContrastLoss(num_classes=NUM_CLASSES, distance_space="logit")
        config = TTAConfig(distance_space="logit", buffer=8)
        runner = OnlineTTARunner(intervention, loss_fn, anchor=None, config=config)
        assert runner.bank.dim == NUM_CLASSES

    def test_mismatched_distance_space_raises_early(self, toy_setup):
        intervention = _make_intervention(toy_setup)
        loss_fn = AdaContrastLoss(num_classes=NUM_CLASSES, distance_space="z")
        config = TTAConfig(distance_space="h", buffer=8)
        with pytest.raises(ValueError):
            OnlineTTARunner(intervention, loss_fn, anchor=None, config=config)

    def test_z_space_pooling_produces_image_level_vectors(self, toy_setup):
        intervention = _make_intervention(toy_setup)
        x_weak, _, _ = _make_batch(batch_size=4)
        logits, feat, code = intervention(x_weak, return_code=True)
        pooled = _pool_code_to_image(code, batch_size=4)
        assert pooled.shape == (4, HIDDEN_DIM)


class TestGainInterventionReturnCode:
    """M2 확장: forward(x, return_code=True) -> (logits, feat, code). 기본값 False는 기존과 동일."""

    def test_default_matches_two_tuple(self, toy_setup):
        intervention = _make_intervention(toy_setup)
        x_weak, _, _ = _make_batch(batch_size=4)
        with torch.no_grad():
            out = intervention(x_weak)
        assert len(out) == 2

    def test_return_code_true_gives_three_tuple_with_matching_logits_feat(self, toy_setup):
        intervention = _make_intervention(toy_setup)
        x_weak, _, _ = _make_batch(batch_size=4, seed=1)
        torch.manual_seed(0)
        with torch.no_grad():
            logits_ref, feat_ref = intervention(x_weak)
        torch.manual_seed(0)
        with torch.no_grad():
            logits, feat, code = intervention(x_weak, return_code=True)
        assert torch.equal(logits, logits_ref)
        assert torch.equal(feat, feat_ref)
        assert code.shape[1] == HIDDEN_DIM
        assert code.shape[0] % 4 == 0  # B*T_patch


class TestGateNotImplemented:
    def test_gate_none_is_supported(self, toy_setup):
        intervention = _make_intervention(toy_setup)
        loss_fn = AdaContrastLoss(num_classes=NUM_CLASSES, distance_space="h")
        config = TTAConfig(distance_space="h", buffer=8)
        OnlineTTARunner(intervention, loss_fn, anchor=None, config=config, gate=None)

    def test_gate_not_none_raises_not_implemented(self, toy_setup):
        intervention = _make_intervention(toy_setup)
        loss_fn = AdaContrastLoss(num_classes=NUM_CLASSES, distance_space="h")
        config = TTAConfig(distance_space="h", buffer=8)
        with pytest.raises(NotImplementedError):
            OnlineTTARunner(intervention, loss_fn, anchor=None, config=config, gate=object())


class TestNoLeakageProtocol:
    """배치는 그 배치 자신의 gain 갱신이 반영되기 전 예측으로 채점해야 한다."""

    def test_pre_adaptation_prediction_unaffected_by_this_batch_update(self, toy_setup):
        intervention = _make_intervention(toy_setup)
        loss_fn = AdaContrastLoss(num_classes=NUM_CLASSES, distance_space="h")
        config = TTAConfig(distance_space="h", buffer=8, lr=1e-1)  # 큰 lr로 gain을 확실히 움직인다
        runner = OnlineTTARunner(intervention, loss_fn, anchor=None, config=config)

        x_weak, x_strong, y = _make_batch(batch_size=4)
        gain_before = intervention.gain.detach().clone()
        with torch.no_grad():
            logits_ref, _ = intervention(x_weak)
        expected_pre_correct = int((logits_ref.argmax(dim=1) == y).sum().item())

        pre_correct, bsz, _ = runner._adapt_one_batch(x_weak, x_strong, y)

        gain_after = intervention.gain.detach().clone()
        assert not torch.equal(gain_before, gain_after), "테스트 전제가 깨짐: gain이 갱신되지 않았다"
        assert pre_correct == expected_pre_correct
        assert bsz == 4


class TestBufferAndPassesConfigSwitch:
    def test_buffer_zero_is_in_batch_mode(self, toy_setup):
        intervention = _make_intervention(toy_setup)
        loss_fn = AdaContrastLoss(num_classes=NUM_CLASSES, distance_space="h")
        config = TTAConfig(distance_space="h", buffer=0)
        runner = OnlineTTARunner(intervention, loss_fn, anchor=None, config=config)
        assert runner.bank.capacity == 0

    def test_buffer_full_uses_full_buffer_capacity(self, toy_setup):
        intervention = _make_intervention(toy_setup)
        loss_fn = AdaContrastLoss(num_classes=NUM_CLASSES, distance_space="h")
        config = TTAConfig(distance_space="h", buffer="full", full_buffer_capacity=40)
        runner = OnlineTTARunner(intervention, loss_fn, anchor=None, config=config)
        assert runner.bank.capacity == 40

    def test_passes_multiplies_samples_seen(self, toy_setup):
        intervention = _make_intervention(toy_setup)
        loss_fn = AdaContrastLoss(num_classes=NUM_CLASSES, distance_space="h")
        stream = [_make_batch(batch_size=4, seed=i) for i in range(2)]  # 8 샘플/pass

        config1 = TTAConfig(distance_space="h", buffer=8, passes=1)
        runner1 = OnlineTTARunner(intervention, loss_fn, anchor=None, config=config1)
        result1 = runner1.run(stream)
        assert result1["n_seen"] == 8

        intervention2 = _make_intervention(toy_setup)
        config2 = TTAConfig(distance_space="h", buffer=8, passes=3)
        runner2 = OnlineTTARunner(intervention2, loss_fn, anchor=None, config=config2)
        result2 = runner2.run(stream)
        assert result2["n_seen"] == 24


class TestRunReturnsBothProtocols:
    def test_run_returns_expected_keys_and_both_accuracies(self, toy_setup):
        intervention = _make_intervention(toy_setup)
        loss_fn = AdaContrastLoss(num_classes=NUM_CLASSES, distance_space="h")
        config = TTAConfig(distance_space="h", buffer=8, passes=1)
        runner = OnlineTTARunner(intervention, loss_fn, anchor=None, config=config)
        stream = [_make_batch(batch_size=4, seed=i) for i in range(3)]

        result = runner.run(stream)

        for key in ("online_acc", "final_acc", "acc_vs_time", "gain_trace", "loss_trace", "gate_skipped", "n_seen"):
            assert key in result
        assert result["n_seen"] == 12
        assert len(result["acc_vs_time"]) == 3
        assert len(result["gain_trace"]) == 3
        assert 0.0 <= result["online_acc"] <= 1.0
        assert 0.0 <= result["final_acc"] <= 1.0

    def test_final_acc_reflects_post_adaptation_gain_without_further_updates(self, toy_setup):
        """final_acc는 적응이 끝난 gain으로 다시 평가한 값이지, 적응 도중 갱신을 계속하지 않는다."""
        intervention = _make_intervention(toy_setup)
        loss_fn = AdaContrastLoss(num_classes=NUM_CLASSES, distance_space="h")
        config = TTAConfig(distance_space="h", buffer=8, passes=1, lr=1e-1)
        runner = OnlineTTARunner(intervention, loss_fn, anchor=None, config=config)
        stream = [_make_batch(batch_size=4, seed=i) for i in range(2)]

        runner.run(stream)
        gain_after_run = intervention.gain.detach().clone()

        # run() 종료 후 gain이 그대로 유지된다(final_acc 재평가가 추가로 gain을 바꾸지 않았다).
        assert torch.equal(intervention.gain.detach(), gain_after_run)


class TestAnchorInjection:
    def test_anchor_none_behaves_like_zero_penalty(self, toy_setup):
        intervention = _make_intervention(toy_setup)
        loss_fn = AdaContrastLoss(num_classes=NUM_CLASSES, distance_space="h")
        config = TTAConfig(distance_space="h", buffer=8)
        runner = OnlineTTARunner(intervention, loss_fn, anchor=None, config=config)
        penalty = runner._anchor_penalty()
        assert penalty.item() == 0.0

    def test_anchor_callable_is_invoked_with_gain(self, toy_setup):
        intervention = _make_intervention(toy_setup)
        loss_fn = AdaContrastLoss(num_classes=NUM_CLASSES, distance_space="h")
        config = TTAConfig(distance_space="h", buffer=8)

        calls = []

        def fake_anchor(gain):
            calls.append(gain)
            return (gain - 1.0).pow(2).sum()

        runner = OnlineTTARunner(intervention, loss_fn, anchor=fake_anchor, config=config)
        penalty = runner._anchor_penalty()
        assert len(calls) == 1
        assert penalty.item() == 0.0  # gain은 아직 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 미가용")
class TestCudaDevice:
    def test_run_completes_on_cuda(self, toy_setup):
        backbone = _make_backbone().cuda()
        torch.manual_seed(0)
        x = torch.randn(4, 3, 224, 224, device="cuda")
        flat = _backbone_activation_at_hook(backbone, x).reshape(-1, EMBED_DIM)
        frozen = _make_toy_frozen_sae(flat.cpu()).cuda()
        basis = LatentGainBasis(frozen)
        intervention = GainIntervention(backbone, basis, hook_block=HOOK_BLOCK).cuda()
        loss_fn = AdaContrastLoss(num_classes=NUM_CLASSES, distance_space="h")
        config = TTAConfig(distance_space="h", buffer=8)
        runner = OnlineTTARunner(intervention, loss_fn, anchor=None, config=config)

        stream = []
        for i in range(2):
            xw, xs, y = _make_batch(batch_size=4, seed=i)
            stream.append((xw.cuda(), xs.cuda(), y.cuda()))

        result = runner.run(stream)
        assert result["n_seen"] == 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 미가용")
@pytest.mark.skipif(not os.path.exists(CHECKPOINT_PATH), reason="실 checkpoint 없음")
@pytest.mark.skipif(not os.path.isdir("data/imagenet-c"), reason="ImageNet-C 데이터 없음")
class TestOneCorruptionSmoke:
    """DoD: 1 corruption smoke에서 no-adapt 대비 acc 개선 + online/final 두 프로토콜 병기."""

    def test_adapted_final_acc_improves_over_no_adapt(self):
        from Utils.tta_transforms import DEFAULT_BACKBONE, strong_transform, weak_transform

        device = "cuda"
        sae = FrozenSAE.from_checkpoint(CHECKPOINT_PATH, device=device)
        basis = LatentGainBasis(sae)

        corruption = "gaussian_noise"
        severity = 5
        try:
            ds, meta = datasets_mod.build_dataset(
                "imagenet-c", split="report", corruption=corruption, severity=severity
            )
        except datasets_mod.DatasetUnavailableError:
            pytest.skip("imagenet-c report split 없음")

        n_images = min(300, len(ds))
        subset = torch.utils.data.Subset(ds, list(range(n_images)))

        weak = weak_transform(224, DEFAULT_BACKBONE)
        strong = strong_transform(224, DEFAULT_BACKBONE)

        def make_loader():
            import numpy as np
            from PIL import Image

            class _TwoView(torch.utils.data.Dataset):
                def __len__(self):
                    return len(subset)

                def __getitem__(self, idx):
                    arr, y = subset[idx]
                    img = Image.fromarray(arr) if isinstance(arr, np.ndarray) else arr
                    return weak(img), strong(img), y

            return torch.utils.data.DataLoader(_TwoView(), batch_size=32, shuffle=False)

        backbone_noadapt = timm.create_model(DEFAULT_BACKBONE, pretrained=True, num_classes=meta.num_classes)
        intervention_noadapt = GainIntervention(backbone_noadapt.to(device), basis, hook_block=10).to(device)
        loss_fn = AdaContrastLoss(num_classes=meta.num_classes, distance_space="h")
        cfg_noadapt = TTAConfig(distance_space="h", buffer=256, passes=1, lr=0.0)
        runner_noadapt = OnlineTTARunner(intervention_noadapt, loss_fn, anchor=None, config=cfg_noadapt)
        result_noadapt = runner_noadapt.run(make_loader())

        backbone_adapt = timm.create_model(DEFAULT_BACKBONE, pretrained=True, num_classes=meta.num_classes)
        intervention_adapt = GainIntervention(backbone_adapt.to(device), basis, hook_block=10).to(device)
        cfg_adapt = TTAConfig(distance_space="h", buffer=256, passes=2, lr=1e-2)
        runner_adapt = OnlineTTARunner(intervention_adapt, loss_fn, anchor=None, config=cfg_adapt)
        result_adapt = runner_adapt.run(make_loader())

        print(
            f"\n[M8 smoke] {corruption}/sev{severity} n={n_images}\n"
            f"  no-adapt : online_acc={result_noadapt['online_acc']:.4f} final_acc={result_noadapt['final_acc']:.4f}\n"
            f"  adapted  : online_acc={result_adapt['online_acc']:.4f} final_acc={result_adapt['final_acc']:.4f}\n"
        )

        assert result_adapt["final_acc"] >= result_noadapt["final_acc"]
