"""ghost gradient (PatchSAE src/sae_training/sparse_autoencoder.py 이식) 검증.

ghost grad가 고치려는 실패는 ReLU+L1의 흡수 상태다: pre-activation이 한 번 충분히
음수로 내려가면 z=0 -> 기울기 0 -> b_enc/W_enc가 안 움직임 -> 영원히 z=0. exp()는
음수 구간에서도 기울기가 0이 아니라 이 상태를 빠져나올 수 있다.

그래서 여기서 확인하는 것은 "손실이 줄었나"가 아니라 **기울기가 죽은 latent에만,
그리고 실제로 0이 아니게 흐르는가**다.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import Utils.SAE_utils as U
from Model.SAE import VanillaL1SAE
from Utils.Config import SAEExperimentConfig
from Utils.SAE_utils import DeadLatentTracker, _make_dead_tracker, _run_sae_step

DIM, HIDDEN = 8, 16


def _sae():
    torch.manual_seed(0)
    return VanillaL1SAE(input_dim=DIM, hidden_dim=HIDDEN, b_dec_init=torch.zeros(DIM), dec_bias_mode="zero")


def _cfg(use_ghost=True, window=0, recon_space="norm"):
    c = SAEExperimentConfig()
    c.sae.recon_space = recon_space
    c.sae.check_finite = True
    c.sae.use_ghost_grads = use_ghost
    c.sae.dead_feature_window = window
    c.optim_config.use_amp = False
    return c


def _step(sae, xb, cfg, tracker, token_std=None):
    opt = torch.optim.Adam(sae.parameters(), lr=0.0)  # lr=0 이라 파라미터는 안 움직이고 기울기만 본다
    scaler = U._make_grad_scaler("cpu", False, torch.float32)
    return _run_sae_step(sae, opt, xb, scaler, cfg.sae, cfg.optim_config, "cpu", None, token_std, tracker)


def _sae_with_dead_half():
    """무작위 초기화로는 16개가 전부 켜져서 dead가 안 생긴다.

    b_enc를 크게 음수로 밀어 앞쪽 절반을 확실히 죽여 놓는다 — 이게 ghost가 고치려는
    상태(pre-activation이 음수라 z=0, 따라서 기울기 0)와 같은 모양이다."""
    sae = _sae()
    with torch.no_grad():
        sae.b_enc[: HIDDEN // 2] = -50.0
    return sae


# --- forward 계약 -------------------------------------------------------------


def test_forward_default_shape_is_unchanged():
    """평가·진단 경로 6곳이 2-튜플로 받고 있다. 기본 반환 형태가 바뀌면 전부 깨진다."""
    sae, x = _sae(), torch.randn(4, DIM)
    assert len(sae(x)) == 2
    x_hat, z, pre = sae(x, return_pre=True)
    assert torch.equal(z, torch.relu(pre))
    assert torch.equal(x_hat, sae.decode(z))


def test_encode_matches_relu_of_encode_pre():
    sae, x = _sae(), torch.randn(4, DIM)
    assert torch.allclose(sae.encode(x), torch.relu(sae.encode_pre(x)))


# --- dead 추적 ----------------------------------------------------------------


def test_tracker_counts_steps_since_fired():
    t = DeadLatentTracker(hidden_dim=4, window=1, threshold=1e-8, device="cpu")
    # latent 0만 발화하는 배치를 두 번 먹인다
    z = torch.zeros(2, 4)
    z[0, 0] = 1.0
    t.update(z)
    t.update(z)
    assert t.steps_since_fired.tolist() == [0, 2, 2, 2]
    # window=1이므로 2스텝 안 켜진 1,2,3만 dead다
    assert t.dead_mask().tolist() == [False, True, True, True]


def test_tracker_threshold_is_not_the_reporting_threshold():
    """0.2(보고용)로 재면 작게 켜지는 latent까지 dead로 몰린다. 판정은 1e-8이어야 한다."""
    t = DeadLatentTracker(hidden_dim=2, window=0, threshold=1e-8, device="cpu")
    z = torch.tensor([[0.05, 0.0]])  # 0.05는 active_threshold(0.2) 미만이지만 발화는 발화다
    t.update(z)
    assert t.dead_mask().tolist() == [False, True]


def test_make_dead_tracker_returns_none_when_disabled():
    assert _make_dead_tracker(_cfg(use_ghost=False).sae, HIDDEN, "cpu") is None
    assert _make_dead_tracker(_cfg(use_ghost=True).sae, HIDDEN, "cpu") is not None


# --- ghost 손실 ---------------------------------------------------------------


def test_no_ghost_loss_before_any_latent_is_dead():
    """첫 스텝에는 카운터가 0이라 dead가 없다 — ghost는 None이어야 한다."""
    sae, cfg = _sae(), _cfg(window=5)
    tracker = _make_dead_tracker(cfg.sae, HIDDEN, "cpu")
    *_, ghost = _step(sae, torch.randn(32, DIM), cfg, tracker)
    assert ghost is None


def test_ghost_loss_appears_once_window_passes():
    sae, cfg = _sae_with_dead_half(), _cfg(window=0)
    tracker = _make_dead_tracker(cfg.sae, HIDDEN, "cpu")
    xb = torch.randn(32, DIM)
    _step(sae, xb, cfg, tracker)          # 1스텝 뒤 안 켜진 latent는 steps_since_fired=1 > 0
    *_, ghost = _step(sae, xb, cfg, tracker)
    assert ghost is not None and torch.isfinite(ghost)


def test_ghost_grads_reach_dead_latents_only():
    """핵심 계약. 살아있는 latent의 기울기는 ghost 유무와 무관해야 하고,
    죽은 latent는 ghost가 없으면 기울기가 정확히 0이어야 한다."""
    torch.manual_seed(3)
    xb = torch.randn(64, DIM)

    def grads(use_ghost):
        sae, cfg = _sae_with_dead_half(), _cfg(use_ghost=use_ghost, window=0)
        tracker = _make_dead_tracker(cfg.sae, HIDDEN, "cpu")
        if tracker is not None:
            _step(sae, xb, cfg, tracker)
        else:
            _step(sae, xb, cfg, None)
        _step(sae, xb, cfg, tracker)
        return sae.b_enc.grad.clone(), (tracker.dead_mask() if tracker else None)

    g_ghost, dead = grads(True)
    g_plain, _ = grads(False)

    assert dead[: HIDDEN // 2].all(), "b_enc=-50으로 민 latent는 dead여야 한다"
    # 죽은 latent: ghost 없으면 0, 있으면 0이 아니다
    assert torch.count_nonzero(g_plain[dead]) == 0
    assert torch.count_nonzero(g_ghost[dead]) > 0
    # 살아있는 latent: ghost가 손대지 않는다
    assert torch.allclose(g_ghost[~dead], g_plain[~dead], atol=1e-6)


def test_ghost_loss_value_tracks_recon_loss():
    """PatchSAE의 재스케일은 ghost 손실의 '값'을 recon 항과 같게 만든다.

    rescale = (recon_elem / (ghost_elem + eps)).detach() 이므로 rescale*ghost_elem은
    수치상 recon_elem과 같아진다. 값이 아니라 기울기 크기를 맞추려는 장치라 이게 정상이고,
    로그에서 ghost != mse로 보이면 오히려 이식이 틀어진 것이다."""
    sae, cfg = _sae_with_dead_half(), _cfg(window=0)
    tracker = _make_dead_tracker(cfg.sae, HIDDEN, "cpu")
    xb = torch.randn(64, DIM)
    _step(sae, xb, cfg, tracker)
    _loss, recon, _l1, _xh, _z, ghost = _step(sae, xb, cfg, tracker)
    assert float(ghost.item()) == pytest.approx(float(recon.item()), rel=1e-3)


def test_ghost_is_added_to_total_loss():
    sae, cfg = _sae_with_dead_half(), _cfg(window=0)
    tracker = _make_dead_tracker(cfg.sae, HIDDEN, "cpu")
    xb = torch.randn(64, DIM)
    _step(sae, xb, cfg, tracker)
    loss, recon, l1, _xh, _z, ghost = _step(sae, xb, cfg, tracker)
    expected = float(recon.item()) + cfg.sae.l1_reg * float(l1.item()) + float(ghost.item())
    assert float(loss.item()) == pytest.approx(expected, rel=1e-5)


def test_raw_space_ghost_uses_the_same_sigma_weighting_as_recon():
    """recon_space='raw'면 잔차도 ghost 출력도 sigma로 가중돼야 한다.

    한쪽만 가중하면 ghost가 엉뚱한 공간의 잔차를 쫓게 되고, 그러면 살려낸 latent가
    배포 공간에서 쓸모없는 방향을 학습한다."""
    torch.manual_seed(5)
    xb = torch.randn(64, DIM)
    sd = torch.rand(1, DIM) * 3 + 0.5

    sae, cfg = _sae_with_dead_half(), _cfg(window=0, recon_space="raw")
    tracker = _make_dead_tracker(cfg.sae, HIDDEN, "cpu")
    _step(sae, xb, cfg, tracker, token_std=sd)
    _loss, recon, _l1, _xh, _z, ghost = _step(sae, xb, cfg, tracker, token_std=sd)
    # 같은 재스케일 항등식이 raw 공간에서도 성립해야 한다
    assert float(ghost.item()) == pytest.approx(float(recon.item()), rel=1e-3)


def test_max_rows_subsamples_without_changing_which_latents_get_gradient():
    """VRAM 탈출구가 기울기의 '대상'을 바꾸면 안 된다 — 표본만 줄여야 한다."""
    torch.manual_seed(7)
    xb = torch.randn(64, DIM)

    def dead_grad_mask(max_rows):
        sae, cfg = _sae_with_dead_half(), _cfg(window=0)
        cfg.sae.ghost_grad_max_rows = max_rows
        tracker = _make_dead_tracker(cfg.sae, HIDDEN, "cpu")
        _step(sae, xb, cfg, tracker)
        _step(sae, xb, cfg, tracker)
        return (sae.b_enc.grad != 0), tracker.dead_mask()

    full, dead = dead_grad_mask(None)
    sub, _ = dead_grad_mask(8)
    assert full[dead].all() and sub[dead].all()
    assert torch.equal(full, sub)


def test_ghost_off_is_bit_identical_to_before():
    """기본값(use_ghost_grads=False)에서는 손실이 예전과 완전히 같아야 한다."""
    torch.manual_seed(11)
    xb = torch.randn(32, DIM)

    sae_a, cfg_a = _sae(), _cfg(use_ghost=False)
    loss_a, recon_a, l1_a, _xh, _z, ghost_a = _step(sae_a, xb, cfg_a, None)
    assert ghost_a is None

    sae_b = _sae()
    x_hat, z = sae_b(xb)
    expected_recon = (x_hat - xb).pow(2).mean()
    expected_l1 = z.abs().sum(dim=-1).mean()
    assert float(recon_a.item()) == pytest.approx(float(expected_recon.item()), rel=1e-6)
    assert float(l1_a.item()) == pytest.approx(float(expected_l1.item()), rel=1e-6)
    assert float(loss_a.item()) == pytest.approx(
        float(expected_recon.item()) + cfg_a.sae.l1_reg * float(expected_l1.item()), rel=1e-6
    )
