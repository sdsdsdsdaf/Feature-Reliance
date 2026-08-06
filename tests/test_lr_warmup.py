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
