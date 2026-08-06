"""희소성 제약 기반 best 선정 테스트.

학습 손실은 이 변경의 범위 밖이다 — 여기서 검증하는 건 "어느 epoch / 어느 trial을
남기는가"뿐이다. 실제 학습 없이 가짜 epoch row와 더미 모듈로 EarlyStopper를 돌린다.
"""

import torch
from torch import nn

from SAE_validation import select_best_trial
from Utils.early_stopping import EarlyStopper


class _DummySAE(nn.Module):
    """state_dict 저장만 되면 되는 최소 모듈."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(2))


def _row(epoch, nmse, l0_raw, mean_l0=None):
    """EarlyStopper.step이 읽는 필드만 채운 epoch row."""
    return {
        "epoch": epoch,
        "normalized_mse": nmse,
        "l0_raw": l0_raw,
        "mean_l0": mean_l0 if mean_l0 is not None else l0_raw / 100.0,
        "active_mean_count": mean_l0 if mean_l0 is not None else l0_raw / 100.0,
    }


def _run(tmp_path, rows, l0_max=None, l0_metric="l0_raw", patience=None):
    """rows를 순서대로 흘려보내고 (stopper, 저장된 checkpoint) 를 돌려준다."""
    ckpt = tmp_path / "best.pt"
    stopper = EarlyStopper(
        patience=patience,
        eps=5e-5,
        checkpoint_path=ckpt,
        l0_metric=l0_metric,
        l0_max=l0_max,
        verbose=False,
    )
    model = _DummySAE()
    for row in rows:
        stopper.step(row["normalized_mse"], model, row)
    saved = torch.load(ckpt, map_location="cpu", weights_only=False) if ckpt.exists() else None
    return stopper, saved


def test_no_constraint_keeps_legacy_behaviour(tmp_path):
    """l0_max=None이면 제약 이전과 동일하게 val_nmse 최소 epoch을 남긴다."""
    rows = [_row(1, 0.05, 9000), _row(2, 0.01, 9500), _row(3, 0.02, 300)]
    stopper, saved = _run(tmp_path, rows, l0_max=None)
    assert stopper.best_epoch == 2
    assert saved["epoch"] == 2


def test_all_infeasible_keeps_lowest_l0(tmp_path):
    """제약을 아무도 못 맞추면 val_nmse가 아니라 L0가 가장 낮은 epoch을 남긴다."""
    rows = [_row(1, 0.05, 9000), _row(2, 0.01, 9500), _row(3, 0.02, 4000)]
    stopper, saved = _run(tmp_path, rows, l0_max=150)
    assert stopper.best_epoch == 3
    assert stopper.best_is_feasible is False
    # checkpoint 파일이 없으면 load_best가 trial을 통째로 죽인다.
    assert saved is not None
    assert saved["l0_feasible"] is False


def test_feasible_beats_infeasible_even_with_worse_nmse(tmp_path):
    """제약을 만족하는 epoch은 nmse가 더 나빠도 위반 epoch을 이긴다."""
    rows = [_row(1, 0.001, 9000), _row(2, 0.30, 120)]
    stopper, saved = _run(tmp_path, rows, l0_max=150)
    assert stopper.best_epoch == 2
    assert stopper.best_is_feasible is True
    assert saved["epoch"] == 2


def test_best_nmse_within_feasible_region(tmp_path):
    """feasible이 여러 개면 그중 val_nmse 최소를 고르고, 이후 위반 epoch에 뺏기지 않는다."""
    rows = [
        _row(1, 0.30, 9000),   # 위반
        _row(2, 0.20, 140),    # 만족
        _row(3, 0.10, 100),    # 만족, 더 좋음
        _row(4, 0.01, 5000),   # 위반 — nmse가 제일 좋지만 후보가 아니다
    ]
    stopper, saved = _run(tmp_path, rows, l0_max=150)
    assert stopper.best_epoch == 3
    assert saved["epoch"] == 3


def test_patience_resets_when_constraint_first_met(tmp_path):
    """위반 구간에서 patience가 소진돼 feasible 구간 진입 전에 멈추면 안 된다."""
    rows = [_row(1, 0.30, 9000)] + [_row(e, 0.30, 9000) for e in range(2, 6)] + [_row(6, 0.40, 100)]
    stopper, _ = _run(tmp_path, rows, l0_max=150, patience=3)
    assert stopper.bad_epochs == 0
    assert stopper.should_stop is False
    assert stopper.best_epoch == 6


def test_l0_raw_and_mean_l0_disagree(tmp_path):
    """z>0.2 기준으로는 통과하지만 z>0 기준으로는 dense한 SAE가 걸러져야 한다.

    trial_0025 실측(mean_l0 30.5 / l0_raw 13,519)을 그대로 넣은 회귀 테스트다."""
    rows = [_row(1, 0.003, 13519, mean_l0=30.5)]
    _, saved_raw = _run(tmp_path / "raw", rows, l0_max=150, l0_metric="l0_raw")
    assert saved_raw["l0_feasible"] is False

    rows = [_row(1, 0.003, 13519, mean_l0=30.5)]
    _, saved_thr = _run(tmp_path / "thr", rows, l0_max=150, l0_metric="mean_l0")
    assert saved_thr["l0_feasible"] is True


def _trial(trial_id, nmse, l0_raw):
    return {"trial_id": trial_id, "status": "completed", "val_nmse": nmse, "l0_raw": l0_raw}


def test_select_best_trial_without_constraint():
    trials = [_trial("t0", 0.01, 9000), _trial("t1", 0.02, 100)]
    best = select_best_trial(trials, metric="val_nmse", mode="min")
    assert best["selected"]["trial_id"] == "t0"


def test_select_best_trial_prefers_feasible():
    trials = [_trial("t0", 0.001, 9000), _trial("t1", 0.02, 100), _trial("t2", 0.03, 140)]
    best = select_best_trial(trials, metric="val_nmse", mode="min", l0_metric="l0_raw", l0_max=150)
    assert best["selected"]["trial_id"] == "t1"
    assert best["selection"]["feasible"] is True
    assert best["selection"]["feasible_trials"] == 2


def test_select_best_trial_reports_when_nothing_feasible():
    """전부 위반이면 아무거나 승격시키지 않고 왜 못 골랐는지를 남긴다."""
    trials = [_trial("t0", 0.001, 9000), _trial("t1", 0.02, 2937)]
    best = select_best_trial(trials, metric="val_nmse", mode="min", l0_metric="l0_raw", l0_max=150)
    assert best["selected"] is None
    assert best["selection"]["feasible"] is False
    assert best["selection"]["closest_trial"] == "t1"
    assert best["selection"]["closest_l0"] == 2937


def test_select_best_trial_returns_none_without_completed_trials():
    assert select_best_trial([], metric="val_nmse", mode="min", l0_metric="l0_raw", l0_max=150) is None
