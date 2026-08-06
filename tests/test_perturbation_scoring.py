"""섭동 민감도 점수의 상대화/특이도 축 테스트.

각 모드가 "왜" 다른 latent를 고르는지를 고정한다. 시나리오는 latent 4개다:

  id 0  big_generic   : 항상 크게 켜지고(mean z 1.0) 모든 kind에 똑같이 0.10 흔들린다.
                        절대 변화량이 가장 크므로 absolute가 항상 이걸 고른다. 상대 변화는 10%.
  id 1  small_specific: 작게 켜지고(mean z 0.01) grayscale에만 0.005 흔들린다.
                        절대 변화는 20배 작지만 상대 변화는 50%로 5배 크고, kind 특이적이다.
  id 2  big_specific  : 크게 켜지고(mean z 1.0) grayscale에만 0.30 흔들린다.
                        절대·특이도 모두 강하다.
  id 3  ghost         : 거의 안 켜지는데(frequency 1e-9) 켜질 때는 자기 크기만큼 흔들린다
                        (mean z 1e-6, dz 1e-6 -> 상대 변화 약 100%). 절대값이 미미해
                        absolute에서는 보이지도 않지만 relative에서는 1등이 된다
                        — min_frequency 하한이 필요한 이유.
"""

import pytest
import torch

from Utils.SAE_plot_utils import (
    PERTURBATION_SCORE_MODES,
    perturbation_scores,
    score_perturbation_accum,
)

KINDS = ("grayscale", "blur", "patch_shuffle")
BIG_GENERIC, SMALL_SPECIFIC, BIG_SPECIFIC, GHOST = 0, 1, 2, 3

# mean |dz| per kind
ABS = {
    "grayscale": torch.tensor([0.10, 0.005, 0.30, 1e-6]),
    "blur": torch.tensor([0.10, 0.000, 0.00, 0.0]),
    "patch_shuffle": torch.tensor([0.10, 0.000, 0.00, 0.0]),
}
MEAN_ACT = {k: torch.tensor([1.0, 0.01, 1.0, 1e-6]) for k in KINDS}
FREQ = {k: torch.tensor([0.5, 0.02, 0.5, 1e-9]) for k in KINDS}


def _scores(mode, min_frequency=0.0):
    with warnings_suppressed():
        return perturbation_scores(ABS, FREQ, MEAN_ACT, KINDS, score_mode=mode, min_frequency=min_frequency)


class warnings_suppressed:
    def __enter__(self):
        import warnings
        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        self._ctx.__exit__(*exc)
        return False


def test_absolute_picks_the_biggest_latent_regardless_of_specificity():
    """기존 동작 — 절대 변화량만 본다. blur/patch_shuffle에는 big_generic만 반응하므로 그게 1등."""
    s = _scores("absolute")
    assert int(s["blur"].argmax()) == BIG_GENERIC
    assert int(s["patch_shuffle"].argmax()) == BIG_GENERIC
    # grayscale은 big_specific(0.30)이 big_generic(0.10)보다 커서 이긴다
    assert int(s["grayscale"].argmax()) == BIG_SPECIFIC
    # small_specific은 절대값이 작아 뒤로 밀린다
    assert s["grayscale"][SMALL_SPECIFIC] < s["grayscale"][BIG_GENERIC]


def test_relative_lifts_the_small_but_strongly_responding_latent():
    """상대화 — small_specific의 상대 변화 50%가 big_generic의 10%를 이긴다."""
    s = _scores("relative", min_frequency=1e-3)
    assert s["grayscale"][SMALL_SPECIFIC] > s["grayscale"][BIG_GENERIC]
    assert s["grayscale"][SMALL_SPECIFIC] == pytest.approx(0.5, rel=1e-4)
    assert s["grayscale"][BIG_GENERIC] == pytest.approx(0.1, rel=1e-4)


def test_relative_without_min_frequency_is_hijacked_by_a_ghost_latent():
    """하한이 없으면 거의 안 켜지는 latent가 분모 때문에 1등이 된다 — 하한이 필요한 이유."""
    s = _scores("relative")
    assert int(s["grayscale"].argmax()) == GHOST
    s_guarded = _scores("relative", min_frequency=1e-3)
    assert torch.isinf(s_guarded["grayscale"][GHOST]) and s_guarded["grayscale"][GHOST] < 0
    assert int(s_guarded["grayscale"].argmax()) == SMALL_SPECIFIC


def test_relative_warns_when_min_frequency_is_off():
    with pytest.warns(RuntimeWarning, match="min_frequency"):
        perturbation_scores(ABS, FREQ, MEAN_ACT, KINDS, score_mode="relative", min_frequency=0.0)


def test_specific_zeroes_out_the_latent_that_responds_to_everything():
    """특이도 — 모든 kind에 똑같이 반응하는 latent는 대비에서 0이 된다."""
    s = _scores("specific")
    assert s["grayscale"][BIG_GENERIC] == pytest.approx(0.0, abs=1e-7)
    assert s["blur"][BIG_GENERIC] == pytest.approx(0.0, abs=1e-7)
    # grayscale에만 반응하는 latent는 자기 값을 그대로 유지한다(다른 kind가 0이므로)
    assert s["grayscale"][BIG_SPECIFIC] == pytest.approx(0.30, rel=1e-4)
    assert int(s["grayscale"].argmax()) == BIG_SPECIFIC


def test_specific_makes_a_generic_latent_negative_in_the_kinds_it_ignores():
    """grayscale에만 반응하는 latent는 blur 쪽에서 음수가 되어 blur의 cue에서 빠진다."""
    s = _scores("specific")
    assert s["blur"][BIG_SPECIFIC] < 0
    assert s["patch_shuffle"][BIG_SPECIFIC] < 0


def test_relative_specific_combines_both_axes():
    """상대화 후 대비 — small_specific(상대 50%, 특이)이 big_specific(상대 30%, 특이)을 이긴다."""
    s = _scores("relative_specific", min_frequency=1e-3)
    assert int(s["grayscale"].argmax()) == SMALL_SPECIFIC
    assert s["grayscale"][BIG_GENERIC] == pytest.approx(0.0, abs=1e-7)
    # absolute에서는 순서가 반대였다
    a = _scores("absolute")
    assert a["grayscale"][SMALL_SPECIFIC] < a["grayscale"][BIG_SPECIFIC]


def test_axes_are_independent():
    """네 모드가 서로 다른 1등을 낸다 — 두 축이 실제로 분리돼 있다는 뜻이다."""
    winners = {
        "absolute": int(_scores("absolute")["grayscale"].argmax()),
        "relative": int(_scores("relative", min_frequency=1e-3)["grayscale"].argmax()),
        "specific": int(_scores("specific")["grayscale"].argmax()),
        "relative_specific": int(_scores("relative_specific", min_frequency=1e-3)["grayscale"].argmax()),
    }
    assert winners == {
        "absolute": BIG_SPECIFIC,
        "relative": SMALL_SPECIFIC,
        "specific": BIG_SPECIFIC,
        "relative_specific": SMALL_SPECIFIC,
    }


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="score_mode"):
        perturbation_scores(ABS, FREQ, MEAN_ACT, KINDS, score_mode="l2")


def test_specific_needs_at_least_two_kinds():
    one = ("grayscale",)
    with pytest.raises(ValueError, match="at least 2"):
        perturbation_scores({"grayscale": ABS["grayscale"]}, {"grayscale": FREQ["grayscale"]},
                            {"grayscale": MEAN_ACT["grayscale"]}, one, score_mode="specific")


def _accum(n_tokens=1000):
    return {k: {
        "delta_sum": ABS[k] * n_tokens,
        "peak_delta": ABS[k] * 2,
        "freq_sum": FREQ[k] * n_tokens,
        "mean_sum": MEAN_ACT[k] * n_tokens,
    } for k in KINDS}


def test_score_perturbation_accum_absolute_matches_legacy_formula():
    """누적/채점 분리 후에도 absolute는 delta_sum/count 그대로다."""
    n = 1000
    out = score_perturbation_accum(_accum(n), n, kinds=KINDS, top_k=3, score_mode="absolute")
    assert out["grayscale"]["latent_ids"][0] == BIG_SPECIFIC
    rec = out["grayscale"]["records"][0]
    assert rec["score_delta"] == pytest.approx(0.30, rel=1e-4)
    assert rec["score_absolute"] == pytest.approx(0.30, rel=1e-4)


def test_score_perturbation_accum_keeps_absolute_score_for_comparison():
    """모드를 바꿔도 raw 절대 점수가 record에 남아 있어야 비교가 된다."""
    n = 1000
    out = score_perturbation_accum(_accum(n), n, kinds=KINDS, top_k=2,
                                   score_mode="relative_specific", min_frequency=1e-3)
    top = out["grayscale"]["records"][0]
    assert top["latent_id"] == SMALL_SPECIFIC
    assert top["score_delta"] != pytest.approx(top["score_absolute"])
    assert top["score_absolute"] == pytest.approx(0.005, rel=1e-4)


def test_min_frequency_too_high_raises_instead_of_returning_garbage_ids():
    n = 1000
    with pytest.raises(ValueError, match="min_frequency"):
        score_perturbation_accum(_accum(n), n, kinds=KINDS, top_k=3,
                                 score_mode="relative", min_frequency=0.9)


def test_all_declared_modes_are_reachable():
    n = 1000
    for mode in PERTURBATION_SCORE_MODES:
        out = score_perturbation_accum(_accum(n), n, kinds=KINDS, top_k=2, score_mode=mode, min_frequency=1e-3)
        assert len(out["grayscale"]["latent_ids"]) == 2
