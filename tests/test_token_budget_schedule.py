"""token_budget 학습 스케줄 테스트.

검증하는 계약:
- 예산만큼만 학습하고 멈춘다
- 검증이 epoch이 아니라 eval_every_steps 스텝마다 일어난다
- 같은 활성을 두 번 쓰지 않는다(스트림을 한 번만 흘린다)
- epoch 모드는 기존 동작 그대로다
"""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from Utils.Config import SAEExperimentConfig
from Utils.SAE_utils import train_sae_auto


class _TinyBlock(nn.Module):
    """[B, tokens, dim]을 그대로 돌려주는 훅 대상 블록."""

    def forward(self, x):
        return x


class _TinyViT(nn.Module):
    """collect/iter_token_batches_with_hook이 기대하는 최소 인터페이스만 갖춘 가짜 백본."""

    def __init__(self, dim=8, tokens=5, depth=1):
        super().__init__()
        self.blocks = nn.ModuleList([_TinyBlock() for _ in range(depth)])
        self.dim = dim
        self.tokens = tokens
        self.forward_calls = 0

    def forward(self, images):
        self.forward_calls += 1
        b = images.shape[0]
        x = images.reshape(b, 1, -1).expand(b, self.tokens, self.dim).contiguous()
        for blk in self.blocks:
            x = blk(x)
        return x


def _loader(num_images=64, dim=8):
    images = torch.randn(num_images, dim)
    return DataLoader(TensorDataset(images, torch.zeros(num_images, dtype=torch.long)), batch_size=8)


def _config(tmp_path, mode, total_tokens=None, eval_every=4, epochs=2):
    c = SAEExperimentConfig()
    c.extraction_config.device = "cpu"
    c.hook.target_block = 0
    c.hook.token_scope = "all"
    c.sae.expansion = 2
    c.sae.dec_bias_mode = "zero"
    c.sae.batch_size = 16
    c.sae.model_compile = False
    c.sae.check_finite = False
    c.optim_config.epochs = epochs
    c.optim_config.use_amp = False
    c.schedule.mode = mode
    c.schedule.total_train_tokens = total_tokens
    c.schedule.eval_every_steps = eval_every
    c.early_stopping.patience = None
    c.early_stopping.save_verbose = False
    c.token.max_train_tokens = 160
    c.token.max_val_tokens = 64
    c.output.root_dir = str(tmp_path)
    return c


def _run(tmp_path, config, model=None, dim=8):
    model = model or _TinyViT(dim=dim)
    val_tokens = torch.randn(64, dim)
    stats = {"mean": torch.zeros(1, dim), "std": torch.ones(1, dim)}
    return train_sae_auto(
        model,
        _loader(dim=dim),
        val_tokens,
        stats,
        dim,
        dim * config.sae.expansion,
        torch.zeros(dim),
        config,
        checkpoint_path=tmp_path / "best.pt",
        expected_tokens=160,
    )


def test_stops_at_token_budget(tmp_path):
    """예산을 넘겨서 학습하지 않는다."""
    config = _config(tmp_path, "token_budget", total_tokens=96, eval_every=2)
    _sae, history = _run(tmp_path, config)
    assert history[-1]["tokens_seen"] >= 96
    # 배치 하나(16 토큰) 이상 초과하지 않는다
    assert history[-1]["tokens_seen"] < 96 + config.sae.batch_size


def test_evaluates_every_n_steps_not_every_epoch(tmp_path):
    """검증 회차가 스텝 주기를 따른다."""
    config = _config(tmp_path, "token_budget", total_tokens=160, eval_every=3)
    _sae, history = _run(tmp_path, config)
    steps = [row["step"] for row in history]
    # 마지막 한 줄은 예산 소진으로 찍히므로 주기에서 벗어날 수 있다
    for step in steps[:-1]:
        assert step % 3 == 0
    assert history[-1]["step"] == max(steps)


def test_row_schema_matches_epoch_mode(tmp_path):
    """기존 소비자(EarlyStopper, plot_sae_training_history)가 읽는 키가 그대로 있어야 한다."""
    config = _config(tmp_path, "token_budget", total_tokens=96, eval_every=2)
    _sae, history = _run(tmp_path, config)
    required = {"epoch", "train_loss", "train_mse", "train_l1", "mse", "normalized_mse",
                "cosine", "mean_l0", "l0_raw", "active_total", "is_best"}
    assert required <= set(history[0]), required - set(history[0])
    assert [row["epoch"] for row in history] == list(range(1, len(history) + 1))


def test_single_pass_over_activations(tmp_path):
    """활성을 다시 만들지 않는다 — ViT forward 횟수가 예산에 선형이고 한 바퀴를 안 넘는다.

    loader는 64장 / batch 8 = 8 배치이고 배치당 8x5 = 40 토큰이라 한 바퀴가 320 토큰이다.
    epoch 모드였다면 같은 활성을 epochs번 다시 만들거나 캐시해야 한다."""
    dim = 8

    full = _TinyViT(dim=dim)
    _run(tmp_path / "full", _config(tmp_path / "full", "token_budget", total_tokens=320, eval_every=100), model=full, dim=dim)
    assert full.forward_calls == 8  # 정확히 한 바퀴, 재계산 없음

    half = _TinyViT(dim=dim)
    _run(tmp_path / "half", _config(tmp_path / "half", "token_budget", total_tokens=160, eval_every=100), model=half, dim=dim)
    assert half.forward_calls == 4  # 예산의 절반이면 forward도 절반


def test_budget_larger_than_dataset_cycles_the_loader(tmp_path):
    """예산이 데이터 한 바퀴보다 크면 loader를 다시 돈다 — 정확히 필요한 바퀴 수만큼.

    한 바퀴 = 8 배치 x (8장 x 5토큰) = 320 토큰이다. PatchSAE도 ImageNet을 약 2.05바퀴 돈다."""
    dim = 8
    for passes in (1, 2, 3):
        model = _TinyViT(dim=dim)
        out = tmp_path / f"p{passes}"
        _sae, history = _run(out, _config(out, "token_budget", total_tokens=320 * passes, eval_every=1000), model=model, dim=dim)
        assert model.forward_calls == 8 * passes, f"{passes}바퀴: forward {model.forward_calls}회"
        assert history[-1]["tokens_seen"] == 320 * passes


def test_epoch_mode_unchanged(tmp_path):
    """mode='epoch'이면 기존 경로로 간다 — history 길이가 epoch 수와 같다."""
    config = _config(tmp_path, "epoch", epochs=3)
    _sae, history = _run(tmp_path, config)
    assert len(history) == 3
    assert "tokens_seen" not in history[0]


def test_token_budget_requires_total(tmp_path):
    config = _config(tmp_path, "token_budget", total_tokens=None)
    with pytest.raises(ValueError, match="total_train_tokens"):
        _run(tmp_path, config)


def test_unknown_mode_rejected(tmp_path):
    config = _config(tmp_path, "sliding_window")
    with pytest.raises(ValueError, match="schedule.mode"):
        _run(tmp_path, config)
