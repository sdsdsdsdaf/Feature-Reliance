"""재구성 손실/지표를 denorm(raw) 공간에서 재는 경로 테스트.

SAE는 정규화된 활성으로 학습하지만 배포에서는 denorm된 재구성이 ViT에 다시 꽂힌다.
recon_space="raw"는 그 불일치를 없앤다. 검증하는 계약:

- raw 손실 == denorm을 명시적으로 계산한 MSE (mu가 상쇄된다는 항등식)
- sigma가 전부 1이면 raw == norm
- sigma가 균일하지 않으면 둘이 갈린다 (안 갈리면 가중이 안 걸린 것이다)
- token_std 없이 raw를 쓰면 조용히 norm으로 떨어지지 않고 에러가 난다
- 학습 함수 세 갈래 전부 token_std를 실제로 전달한다 (예전에 3곳 중 1곳을 빠뜨린 적이 있다)
"""

import inspect

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import Utils.SAE_utils as U
from Model.SAE import VanillaL1SAE
from Utils.Config import SAEExperimentConfig
from Utils.SAE_utils import _run_sae_step, evaluate_sae_tokens, train_sae_auto

DIM = 6
HIDDEN = 12


def _sae():
    torch.manual_seed(0)
    return VanillaL1SAE(input_dim=DIM, hidden_dim=HIDDEN, b_dec_init=torch.zeros(DIM), dec_bias_mode="zero")


def _cfg(recon_space="raw"):
    c = SAEExperimentConfig()
    c.sae.recon_space = recon_space
    c.sae.check_finite = False
    c.optim_config.use_amp = False
    return c


def _step(sae, xb, cfg, token_std):
    opt = torch.optim.Adam(sae.parameters(), lr=0.0)  # lr=0 이라 손실 계산만 본다
    scaler = U._make_grad_scaler("cpu", False, torch.float32)
    _loss, recon, _l1, _xh, _z = _run_sae_step(sae, opt, xb, scaler, cfg.sae, cfg.optim_config, "cpu", None, token_std)
    return float(recon.item())


def test_raw_loss_equals_explicit_denorm_mse():
    """mu가 상쇄된다는 항등식을 실제 숫자로 확인한다."""
    torch.manual_seed(1)
    sae = _sae()
    xb = torch.randn(32, DIM)
    mu = torch.randn(1, DIM) * 5.0          # mu가 커도 결과가 안 변해야 한다
    sd = torch.rand(1, DIM) * 3 + 0.5

    got = _step(sae, xb, _cfg("raw"), sd)

    with torch.no_grad():
        x_hat, _ = sae(xb)
        expected = F.mse_loss(x_hat * sd + mu, xb * sd + mu).item()
    assert got == pytest.approx(expected, rel=1e-5)


def test_raw_equals_norm_when_sigma_is_one():
    torch.manual_seed(2)
    xb = torch.randn(16, DIM)
    sd = torch.ones(1, DIM)
    raw = _step(_sae(), xb, _cfg("raw"), sd)
    norm = _step(_sae(), xb, _cfg("norm"), None)
    assert raw == pytest.approx(norm, rel=1e-6)


def test_raw_differs_from_norm_when_sigma_is_uneven():
    """가중이 실제로 걸리는지 — 안 갈리면 sigma가 무시되고 있는 것이다."""
    torch.manual_seed(3)
    xb = torch.randn(16, DIM)
    sd = torch.linspace(0.5, 4.0, DIM).unsqueeze(0)
    raw = _step(_sae(), xb, _cfg("raw"), sd)
    norm = _step(_sae(), xb, _cfg("norm"), None)
    assert raw != pytest.approx(norm, rel=1e-3)


def test_raw_without_token_std_raises():
    """조용히 norm으로 떨어지면 안 된다 — 손실이 바뀌었는데 아무도 모르게 된다."""
    with pytest.raises(ValueError, match="token_std"):
        _step(_sae(), torch.randn(8, DIM), _cfg("raw"), None)


def test_unknown_recon_space_rejected():
    with pytest.raises(ValueError, match="recon_space"):
        _step(_sae(), torch.randn(8, DIM), _cfg("denormalized"), torch.ones(1, DIM))


def test_evaluate_raw_fvu_matches_hand_computed():
    """지표 쪽도 raw 공간에서 FVU가 나와야 한다 (분자도 분모도 raw)."""
    torch.manual_seed(4)
    sae = _sae()
    tokens = torch.randn(64, DIM)
    stats = {"mean": torch.randn(1, DIM) * 2, "std": torch.rand(1, DIM) * 2 + 0.5}

    out = evaluate_sae_tokens(sae, tokens, batch_size=16, threshold=0.2, device="cpu", token_stats=stats)

    with torch.no_grad():
        x_hat, _ = sae(tokens)
        sd, mu = stats["std"], stats["mean"]
        sse = (x_hat * sd + mu - (tokens * sd + mu)).square().sum().item()
        var = (tokens * sd).square().sum().item()
    assert out["recon_space"] == "raw"
    assert out["mse"] == pytest.approx(sse / tokens.numel(), rel=1e-5)
    assert out["normalized_mse"] == pytest.approx(sse / var, rel=1e-5)


def test_evaluate_without_stats_keeps_normalized_behaviour():
    torch.manual_seed(5)
    sae = _sae()
    tokens = torch.randn(64, DIM)
    out = evaluate_sae_tokens(sae, tokens, batch_size=16, threshold=0.2, device="cpu")
    with torch.no_grad():
        x_hat, _ = sae(tokens)
        sse = (tokens - x_hat).square().sum().item()
    assert out["recon_space"] == "norm"
    assert out["mse"] == pytest.approx(sse / tokens.numel(), rel=1e-5)


def test_l0_metrics_are_unaffected_by_recon_space():
    """L0는 z에 대한 것이라 공간과 무관하다 — 바뀌면 뭔가 잘못 섞인 것이다."""
    torch.manual_seed(6)
    sae = _sae()
    tokens = torch.randn(64, DIM)
    stats = {"mean": torch.zeros(1, DIM), "std": torch.rand(1, DIM) * 3 + 0.5}
    a = evaluate_sae_tokens(sae, tokens, batch_size=16, threshold=0.2, device="cpu")
    b = evaluate_sae_tokens(sae, tokens, batch_size=16, threshold=0.2, device="cpu", token_stats=stats)
    for k in ("mean_l0", "l0_raw", "l0_ratio_raw"):
        assert a[k] == pytest.approx(b[k])


def test_every_run_sae_step_call_site_passes_token_std():
    """호출 지점이 하나라도 빠지면 그 경로만 조용히 norm 손실로 학습된다.

    예전에 warmup scheduler를 3곳 중 2곳에만 붙여서 실제로 쓰이던 경로가 누락된 적이 있다."""
    src = inspect.getsource(U)
    calls = [ln for ln in src.splitlines() if "config.sae, config.optim_config, device, scheduler" in ln]
    assert len(calls) == 4, f"_run_sae_step 호출이 4곳이어야 한다: {len(calls)}"
    for ln in calls:
        assert ln.rstrip().endswith("token_std"), f"token_std 누락: {ln.strip()}"


class _Blk(nn.Module):
    def forward(self, x):
        return x


class _TinyViT(nn.Module):
    def __init__(self, dim=DIM, tokens=4):
        super().__init__()
        self.blocks = nn.ModuleList([_Blk()])
        self.tokens = tokens
        self.dim = dim

    def forward(self, images):
        b = images.shape[0]
        x = images.reshape(b, 1, self.dim).expand(b, self.tokens, self.dim).contiguous()
        for blk in self.blocks:   # 훅이 blocks[0]에 걸리므로 반드시 통과시켜야 한다
            x = blk(x)
        return x


@pytest.mark.parametrize("mode", ["token_budget", "epoch"])
def test_training_end_to_end_in_raw_space(tmp_path, mode):
    """두 스케줄 모두 raw 공간으로 끝까지 돈다 — sigma가 불균일해도 터지지 않는다."""
    c = _cfg("raw")
    c.extraction_config.device = "cpu"
    c.hook.target_block = 0
    c.hook.token_scope = "all"
    c.sae.expansion = 2
    c.sae.dec_bias_mode = "zero"
    c.sae.batch_size = 8
    c.sae.model_compile = False
    c.optim_config.epochs = 2
    c.schedule.mode = mode
    c.schedule.total_train_tokens = 64
    c.schedule.eval_every_steps = 2
    c.early_stopping.patience = None
    c.early_stopping.save_verbose = False
    c.token.max_train_tokens = 64
    c.token.max_val_tokens = 32
    c.output.root_dir = str(tmp_path)

    images = torch.randn(32, DIM)
    loader = DataLoader(TensorDataset(images, torch.zeros(32, dtype=torch.long)), batch_size=8)
    stats = {"mean": torch.randn(1, DIM), "std": torch.linspace(0.5, 4.0, DIM).unsqueeze(0)}

    _sae_out, history = train_sae_auto(
        _TinyViT(), loader, torch.randn(32, DIM), stats, DIM, DIM * 2,
        torch.zeros(DIM), c, checkpoint_path=tmp_path / "best.pt", expected_tokens=64,
    )
    assert history
    assert history[0]["recon_space"] == "raw"
