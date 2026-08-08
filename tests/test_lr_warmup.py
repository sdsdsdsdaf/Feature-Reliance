"""constant-with-warmup 스케줄러 테스트.

PatchSAE 참조 구현(src/sae_training/utils.get_scheduler의 'constantwithwarmup')과
같은 식이어야 한다: lr_scale = min(1.0, (step + 1) / warmup_steps).
"""

import torch
from torch import nn

from Utils.Config import OptimConfig
from Utils.SAE_utils import _make_scheduler


def _optimizer(lr):
    param = nn.Parameter(torch.zeros(1))
    param.grad = torch.zeros(1)
    return torch.optim.AdamW([param], lr=lr)


def test_no_scheduler_when_warmup_disabled():
    """lr_warmup_steps=0이면 스케줄러를 만들지 않아 기존 고정 lr 동작이 유지된다."""
    assert _make_scheduler(_optimizer(1e-4), OptimConfig(lr=1e-4, lr_warmup_steps=0)) is None


def test_missing_field_is_treated_as_disabled():
    """lr_warmup_steps가 없는 옛 config 객체가 들어와도 터지지 않아야 한다."""

    class Legacy:
        lr = 1e-4

    assert _make_scheduler(_optimizer(1e-4), Legacy()) is None


def test_warmup_ramps_linearly_then_holds():
    lr, warmup = 4e-4, 10
    opt = _optimizer(lr)
    sched = _make_scheduler(opt, OptimConfig(lr=lr, lr_warmup_steps=warmup))

    # step 0은 첫 optimizer.step() 이전 상태 = (0 + 1) / 10
    assert opt.param_groups[0]["lr"] == lr * 1 / warmup

    seen = []
    for _ in range(warmup + 5):
        opt.step()  # 실제 학습 루프와 같은 순서(optimizer 먼저, scheduler 나중)
        sched.step()
        seen.append(opt.param_groups[0]["lr"])

    # warmup 구간은 선형 증가
    for i in range(warmup - 1):
        assert abs(seen[i] - lr * min(1.0, (i + 2) / warmup)) < 1e-12
    # warmup 이후는 base lr에 고정
    for value in seen[warmup - 1:]:
        assert abs(value - lr) < 1e-12


def test_matches_patchsae_lambda():
    """참조 구현의 lambda와 스텝별로 정확히 일치해야 한다."""
    lr, warmup = 4e-4, 500
    opt = _optimizer(lr)
    sched = _make_scheduler(opt, OptimConfig(lr=lr, lr_warmup_steps=warmup))

    for step in range(0, 1200):
        expected = lr * min(1.0, (step + 1) / warmup)
        assert abs(opt.param_groups[0]["lr"] - expected) < 1e-12, step
        opt.step()
        sched.step()


# ---------------------------------------------------------------------------
# 후반부 lr 감쇠 (2026-08-07 추가)
#
# 고치려는 문제: warmup 뒤 lr이 끝까지 고정이라 분지에 들어간 뒤에도 파라미터가
# 계속 배회했다. 고정 검증셋의 val_nmse가 회차마다 1.2~1.9% 흔들려 평탄구간 전체
# 개선폭이 잡음의 1.2~1.7배에 그쳤고, 그 탓에 체크포인트가 운으로 뽑혔다.
# ---------------------------------------------------------------------------

import math

import pytest

import Utils.SAE_utils as U


def _cfg(decay="cosine", start=0.8, final=0.0, warmup=0):
    return OptimConfig(lr=1e-3, lr_warmup_steps=warmup, lr_decay=decay,
                       lr_decay_start_frac=start, lr_final_frac=final)


def _trace(cfg, total):
    opt = _optimizer(cfg.lr)
    sched = _make_scheduler(opt, cfg, total_steps=total)
    out = []
    for _ in range(total):
        out.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return out


def test_decay_none_is_unchanged():
    """기본값은 감쇠 없음 — 기존 실행과 수치가 같아야 한다."""
    assert _make_scheduler(_optimizer(1e-3), _cfg(decay="none", warmup=0), total_steps=100) is None
    lrs = _trace(_cfg(decay="none", warmup=10), 50)
    assert lrs[20:] == pytest.approx([1e-3] * 30)


def test_decay_requires_total_steps():
    """감쇠를 켜놓고 total_steps를 안 주면 조용히 고정 lr로 도는 대신 터져야 한다."""
    with pytest.raises(ValueError, match="total_steps"):
        _make_scheduler(_optimizer(1e-3), _cfg(), total_steps=None)


def test_unknown_decay_rejected():
    with pytest.raises(ValueError, match="lr_decay"):
        _make_scheduler(_optimizer(1e-3), _cfg(decay="exponential"), total_steps=100)


@pytest.mark.parametrize("decay", ["linear", "cosine"])
def test_flat_until_start_then_monotone_to_zero(decay):
    total = 1000
    lrs = _trace(_cfg(decay=decay, start=0.8), total)
    # 앞 80%는 고정
    assert lrs[:800] == pytest.approx([1e-3] * 800)
    # 뒤 20%는 단조 감소하고 끝에서 0에 닿는다
    tail = lrs[800:]
    assert all(b <= a + 1e-12 for a, b in zip(tail, tail[1:])), "감쇠 구간이 단조가 아니다"
    assert tail[-1] == pytest.approx(0.0, abs=1e-5)


def test_cosine_starts_gentler_and_meets_linear_at_the_midpoint():
    """코사인은 감쇠 초반이 더 완만하고, 정확히 절반 지점에서 선형과 만난다.

    0.5*(1+cos(pi*p)) 는 p=0.5 에서 0.5 라 선형(1-p)과 같아진다. 그래서 비교는
    중간이 아니라 1/4 지점에서 해야 의미가 있다."""
    total = 1000
    lin = _trace(_cfg(decay="linear"), total)
    cos = _trace(_cfg(decay="cosine"), total)
    quarter = 850   # 감쇠 구간(800~1000)의 1/4
    assert cos[quarter] > lin[quarter]
    assert cos[quarter] == pytest.approx(0.854e-3, rel=0.02)
    assert lin[quarter] == pytest.approx(0.75e-3, rel=0.02)
    # lr_lambda 가 s = step + 1 을 쓰므로 p = 0.5 가 되는 건 step 899 다
    half = 899
    assert cos[half] == pytest.approx(lin[half], rel=1e-6)
    assert cos[half] == pytest.approx(0.5e-3, rel=1e-6)


def test_final_frac_floor_is_respected():
    lrs = _trace(_cfg(decay="linear", final=0.1), 1000)
    assert lrs[-1] == pytest.approx(1e-4, rel=1e-3)
    assert min(lrs) >= 1e-4 - 1e-12


def test_warmup_and_decay_compose():
    """warmup 상승 -> 고정 -> 감쇠 세 구간이 한 스케줄에서 이어져야 한다."""
    lrs = _trace(_cfg(decay="cosine", start=0.8, warmup=100), 1000)
    assert lrs[0] == pytest.approx(1e-5)          # 첫 스텝 = lr/warmup
    assert lrs[99] == pytest.approx(1e-3)         # warmup 끝
    assert lrs[500] == pytest.approx(1e-3)        # 고정 구간
    assert lrs[-1] == pytest.approx(0.0, abs=1e-5)


def test_every_training_loop_passes_total_steps():
    """호출 지점이 하나라도 빠지면 그 경로만 감쇠 없이 학습된다.

    warmup 때 3곳 중 2곳에만 붙여 실제로 쓰이던 경로가 누락된 전례가 있다."""
    import inspect

    import ast

    tree = ast.parse(inspect.getsource(U))
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_make_scheduler"
    ]
    assert len(calls) == 3, f"_make_scheduler 호출이 3곳이어야 한다: {len(calls)}"
    for c in calls:
        kw = {k.arg for k in c.keywords}
        assert "total_steps" in kw, f"total_steps 누락 (line {c.lineno})"
