"""T1.3 실험 3의 통계 커널(`experiments/exp3_stats.py`) 검증.

이 실험은 `[tdd:skip:analysis]`이라 Red 증적은 요구되지 않지만, 계약이 명시적으로
"층화 AUC와 permutation null 특히"는 non-obvious하니 테스트가 필요하다고 못박았다.
핵심적으로 검증하는 것:
1. 층화가 y-place 95% confound를 실제로 제거하는가 (순수 배경 탐지기가 caus로 안 잡히는가).
2. permutation max-null이 순수 잡음에서는 낮고, feature 수가 늘면(다중비교) 자동으로 올라가는가.
3. purity_metrics의 분모 0 가드·경계값(sel=+-0.6)이 기대대로 동작하는가.
4. two-proportion permutation test가 대칭 케이스에서 p~1, 뚜렷한 차이에서 p가 작은가.
5. average_precision이 sklearn과 일치하는가, permutation p-value가 방향대로 움직이는가.
"""

import numpy as np
import pytest

from experiments.exp3_stats import (
    ap_permutation_p,
    average_precision,
    purity_metrics,
    stratified_auc_effect,
    two_proportion_permutation_p,
)


def _make_confounded_dataset(n=2000, seed=0):
    """y와 place가 95% 상관인 합성 Waterbirds 축소판.

    feature 0: place만 완벽히 판별(배경 탐지기) — 순수 spurious여야 한다.
    feature 1: y만 완벽히 판별(새 몸통 탐지기) — 순수 causal이어야 한다.
    feature 2: 순수 잡음 — 둘 다 무정보여야 한다.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    flip = rng.random(n) >= 0.95  # 5%만 뒤집어 95% 상관을 만든다
    place = np.where(flip, 1 - y, y)

    noise0 = rng.normal(0, 0.01, size=n)
    noise1 = rng.normal(0, 0.01, size=n)
    noise2 = rng.normal(0, 1.0, size=n)
    scores = np.stack(
        [
            place.astype(float) + noise0,  # feature 0: place 탐지기
            y.astype(float) + noise1,  # feature 1: y 탐지기
            noise2,  # feature 2: 잡음
        ],
        axis=1,
    )
    return scores, y, place


class TestStratifiedAucRemovesConfound:
    def test_pure_background_detector_is_spurious_not_causal(self):
        """층화 없이 AUC(a_j -> y)를 그냥 재면 배경 탐지기도 causal처럼 보인다는 게
        phase1.md의 핵심 주장이다 — 층화된 spur_j/caus_j는 그 함정에 빠지지 않아야 한다."""
        scores, y, place = _make_confounded_dataset()
        spur, spur_null = stratified_auc_effect(scores, place, y, n_perm=50, rng=np.random.default_rng(1))
        caus, caus_null = stratified_auc_effect(scores, y, place, n_perm=50, rng=np.random.default_rng(2))

        assert spur[0] > 0.9, "feature 0(배경 탐지기)은 spur가 거의 1이어야 한다"
        assert caus[0] < 0.2, "feature 0은 y를 층화해도(=place 고정) y를 못 맞히므로 caus가 낮아야 한다"

        assert caus[1] > 0.9, "feature 1(새 탐지기)은 caus가 거의 1이어야 한다"
        assert spur[1] < 0.2, "feature 1은 place를 못 맞히므로 spur가 낮아야 한다"

        assert spur[2] < 0.3 and caus[2] < 0.3, "잡음 feature는 둘 다 낮아야 한다"

    def test_unstratified_auc_would_have_been_fooled(self):
        """대조 실측: 층화하지 않은 단순 AUC(a_j -> y)는 배경 탐지기(feature 0)도 0.9에
        가까운 값을 준다는 걸 sanity-check해, 층화의 필요성 자체를 재확인한다."""
        from sklearn.metrics import roc_auc_score

        scores, y, _place = _make_confounded_dataset()
        naive_auc_feature0 = roc_auc_score(y, scores[:, 0])
        assert naive_auc_feature0 > 0.85, "층화 없는 단순 AUC는 배경 탐지기를 causal처럼 보이게 한다"


class TestPermutationMaxNull:
    def test_pure_noise_gives_low_but_nonzero_null(self):
        rng = np.random.default_rng(3)
        n = 500
        scores = rng.normal(size=(n, 20))
        y = rng.integers(0, 2, size=n)
        strata = rng.integers(0, 2, size=n)
        effect, null_max = stratified_auc_effect(scores, y, strata, n_perm=100, rng=np.random.default_rng(4))
        assert null_max.min() >= 0.0
        tau = np.quantile(null_max, 0.95)
        assert 0.0 < tau < 0.6, f"잡음뿐인 데이터의 max-null 임계가 비정상적으로 크다: {tau}"

    def test_larger_pool_raises_the_null_threshold(self):
        """max 통계를 쓰므로 feature(pool) 수가 늘수록 tau가 자동으로 올라가야 한다
        (다중비교 보정 — SAE ~6000 vs PCA 768의 공정성 기준)."""
        rng = np.random.default_rng(5)
        n = 400
        y = rng.integers(0, 2, size=n)
        strata = rng.integers(0, 2, size=n)

        small = rng.normal(size=(n, 10))
        large = rng.normal(size=(n, 500))
        _e_small, null_small = stratified_auc_effect(small, y, strata, n_perm=100, rng=np.random.default_rng(6))
        _e_large, null_large = stratified_auc_effect(large, y, strata, n_perm=100, rng=np.random.default_rng(6))

        assert np.quantile(null_large, 0.95) > np.quantile(null_small, 0.95)

    def test_single_class_stratum_raises(self):
        scores = np.random.default_rng(7).normal(size=(10, 3))
        y = np.zeros(10, dtype=int)
        strata = np.zeros(10, dtype=int)
        with pytest.raises(ValueError):
            stratified_auc_effect(scores, y, strata, n_perm=10)


class TestPurityMetrics:
    def test_boundary_and_zero_division_guard(self):
        spur = np.array([1.0, 0.0, 0.0, 0.5])
        caus = np.array([0.0, 1.0, 0.0, 0.5])
        out = purity_metrics(spur, caus, tau_spur=0.3, tau_caus=0.3, sel_threshold=0.6)
        # feature 0: sel=1.0 (>=0.6, informative) -> pure spurious
        # feature 1: sel=-1.0 (<=-0.6, informative) -> pure causal
        # feature 2: spur=caus=0 -> not informative, sel=0 (분모 0 가드)
        # feature 3: spur=caus=0.5 -> informative(spur>=tau), sel=0 -> entangled
        assert out["n_pure_spurious"] == 1
        assert out["n_pure_causal"] == 1
        assert out["n_informative"] == 3
        assert out["n_entangled"] == 1
        assert out["sel"][2] == 0.0
        assert out["purity_rate"] == pytest.approx(2 / 3)

    def test_purity_rate_none_when_no_informative_features(self):
        spur = np.array([0.01, 0.02])
        caus = np.array([0.01, 0.02])
        out = purity_metrics(spur, caus, tau_spur=0.9, tau_caus=0.9)
        assert out["n_informative"] == 0
        assert out["purity_rate"] is None


class TestTwoProportionPermutation:
    def test_identical_proportions_give_high_p(self):
        rng = np.random.default_rng(8)
        a = rng.integers(0, 2, size=300)
        b = rng.integers(0, 2, size=300)
        p = two_proportion_permutation_p(a, b, n_perm=200, rng=np.random.default_rng(9))
        assert p > 0.05

    def test_clear_difference_gives_low_p(self):
        a = np.ones(200, dtype=int)  # purity_rate=1.0
        b = np.zeros(200, dtype=int)  # purity_rate=0.0
        p = two_proportion_permutation_p(a, b, n_perm=200, rng=np.random.default_rng(10))
        assert p < 0.01

    def test_empty_indicator_returns_none(self):
        assert two_proportion_permutation_p(np.array([]), np.array([1, 0])) is None


class TestAveragePrecision:
    def test_matches_sklearn(self):
        from sklearn.metrics import average_precision_score

        rng = np.random.default_rng(11)
        y_true = rng.integers(0, 2, size=200)
        y_score = rng.normal(size=200) + y_true * 1.5
        ours = average_precision(y_true, y_score)
        theirs = average_precision_score(y_true, y_score)
        assert ours == pytest.approx(theirs, abs=1e-9)

    def test_perfect_ranking_ap_is_one(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        assert average_precision(y_true, y_score) == pytest.approx(1.0)

    def test_no_positives_returns_zero(self):
        y_true = np.zeros(10, dtype=int)
        y_score = np.random.default_rng(12).normal(size=10)
        assert average_precision(y_true, y_score) == 0.0

    def test_ap_permutation_p_detects_signal(self):
        rng = np.random.default_rng(13)
        y_true = np.zeros(300, dtype=int)
        y_true[:60] = 1
        y_score = rng.normal(size=300)
        y_score[:60] += 2.0  # 강한 신호
        ap, p = ap_permutation_p(y_true, y_score, n_perm=200, rng=np.random.default_rng(14))
        assert ap > 0.6
        assert p < 0.05

    def test_ap_permutation_p_high_for_random_ranker(self):
        rng = np.random.default_rng(15)
        y_true = np.zeros(300, dtype=int)
        y_true[:60] = 1
        rng.shuffle(y_true)
        y_score = rng.normal(size=300)  # y_true와 무관한 랭커
        ap, p = ap_permutation_p(y_true, y_score, n_perm=200, rng=np.random.default_rng(16))
        assert p > 0.05
