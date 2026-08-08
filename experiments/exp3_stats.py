"""T1.3 실험 3의 순수 통계 커널 — torch/timm/데이터 의존이 전혀 없는 numpy 함수만 모은다.

`experiments/exp3_disentanglement.py`가 이 모듈의 함수를 오케스트레이션하고,
`tests/test_exp3_stats.py`가 합성 데이터로 이 함수들의 정확성(특히 층화 AUC와
permutation max-null)을 검증한다. GPU/데이터셋 없이도 pytest가 즉시 돌 수 있게
분리했다 — docs/plans/contracts/phase1.md T1.3 절의 "①정답 — 층화 AUC"와
"①임계 — permutation max-null" 절을 그대로 구현한다.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def stratified_auc_effect(
    scores: np.ndarray,
    target: np.ndarray,
    strata: np.ndarray,
    *,
    n_perm: int = 200,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """feature별 층화 분리도 effect_j와 permutation max-null 분포를 함께 낸다.

    `effect_j = mean over s in strata의 값 of 2*|AUC(scores[:,j] -> target | strata==s) - 0.5|`.
    각 층 안에서 target을 셔플(구조 보존)한 permutation을 `n_perm`회 반복해, 매 permutation마다
    `max_j effect_j(shuffled)`를 하나씩 낸 것이 `null_max`다(phase1.md의 tau_spur/tau_caus 원료).

    AUC는 rank-sum(Mann-Whitney U)으로 계산한다 — 층별로 rank를 한 번만 매기면
    permutation마다는 셔플된 라벨에 해당하는 rank의 합만 다시 구하면 되므로,
    O(n_perm) 루프를 O(n_perm) 행렬곱 하나로 벡터화할 수 있다(퍼뮤테이션 행렬 @ rank 행렬).

    scores: [n_images, n_features]. target/strata: [n_images], 둘 다 정수형(0/1 등).
    strata 값이 둘 이상의 클래스를 갖지 못하는 층(전부 0 또는 전부 1)은 AUC가 정의되지
    않으므로 건너뛴다(분모가 0인 상황이 실제로는 발생하지 않아야 하지만 방어적으로 처리).

    반환: (effect: [n_features], null_max: [n_perm]).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n_features = scores.shape[1]
    effect_sum = np.zeros(n_features, dtype=np.float64)
    perm_effect_sum = np.zeros((n_perm, n_features), dtype=np.float64)
    n_valid_strata = 0

    for stratum_value in np.unique(strata):
        idx = np.where(strata == stratum_value)[0]
        n_s = idx.size
        sub_scores = scores[idx].astype(np.float64)
        sub_target = target[idx]
        n1 = int(sub_target.sum())
        n2 = n_s - n1
        if n1 == 0 or n2 == 0:
            continue

        ranks = rankdata(sub_scores, axis=0)  # [n_s, n_features], tie-average rank 1..n_s

        R1 = ranks[sub_target == 1].sum(axis=0)
        auc = (R1 - n1 * (n1 + 1) / 2.0) / (n1 * n2)
        effect_sum += 2.0 * np.abs(auc - 0.5)

        base = sub_target.astype(np.float64)
        perm_matrix = np.empty((n_perm, n_s), dtype=np.float64)
        for p in range(n_perm):
            perm_matrix[p] = rng.permutation(base)
        R1p = perm_matrix @ ranks  # [n_perm, n_features]
        aucp = (R1p - n1 * (n1 + 1) / 2.0) / (n1 * n2)
        perm_effect_sum += 2.0 * np.abs(aucp - 0.5)

        n_valid_strata += 1

    if n_valid_strata == 0:
        raise ValueError("모든 층에서 target이 단일 클래스다 — AUC를 정의할 수 없다.")

    effect = effect_sum / n_valid_strata
    perm_effect = perm_effect_sum / n_valid_strata
    null_max = perm_effect.max(axis=1)
    return effect, null_max


def purity_metrics(
    spur: np.ndarray,
    caus: np.ndarray,
    tau_spur: float,
    tau_caus: float,
    *,
    sel_threshold: float = 0.6,
) -> dict:
    """spur_j/caus_j와 두 permutation 임계로부터 축 하나의 분리 성적표를 만든다.

    informative_j = (spur_j >= tau_spur) or (caus_j >= tau_caus) — 각자 자기 임계와 비교한다.
    sel_j = (spur_j-caus_j)/(spur_j+caus_j), 분모 0이면(둘 다 0인 non-informative 축소) 0으로 둔다.
    purity_rate = |informative & |sel|>=sel_threshold| / |informative| (informative가 0개면 None).

    반환: informative_mask/sel(둘 다 array로 호출부가 히스토그램에 씀) 포함 dict.
    """
    denom = spur + caus
    sel = np.divide(spur - caus, denom, out=np.zeros_like(spur), where=denom > 0)
    informative = (spur >= tau_spur) | (caus >= tau_caus)
    pure_spurious = informative & (sel >= sel_threshold)
    pure_causal = informative & (sel <= -sel_threshold)
    n_informative = int(informative.sum())
    n_pure_spurious = int(pure_spurious.sum())
    n_pure_causal = int(pure_causal.sum())
    n_entangled = n_informative - n_pure_spurious - n_pure_causal
    purity_rate = (n_pure_spurious + n_pure_causal) / n_informative if n_informative > 0 else None
    return {
        "n_features": int(spur.shape[0]),
        "tau_spur": float(tau_spur),
        "tau_caus": float(tau_caus),
        "n_informative": n_informative,
        "n_pure_spurious": n_pure_spurious,
        "n_pure_causal": n_pure_causal,
        "n_entangled": n_entangled,
        "purity_rate": purity_rate,
        "sel": sel,
        "informative": informative,
        "pure_spurious": pure_spurious,
        "pure_causal": pure_causal,
    }


def two_proportion_permutation_p(
    indicator_a: np.ndarray,
    indicator_b: np.ndarray,
    *,
    n_perm: int = 200,
    rng: np.random.Generator | None = None,
) -> float:
    """purity_rate 두 축(a vs b)의 차이가 우연히 나올 확률(양측)을 permutation으로 잰다.

    indicator_*: informative feature 각각이 pure(1)인지 아닌지의 0/1 배열(축마다 길이가 다를 수
    있다 — SAE ~수천 vs PCA/random 768). 귀무가설 "두 축의 purity 확률이 같다"를 세우고,
    두 indicator를 풀링한 뒤 원래 크기(len(a), len(b))로 무작위 재분할해 차이의 permutation
    분포를 만든다(표준 two-proportion permutation test). 어느 한쪽이 비어 있으면(informative
    feature가 0개) 비교가 정의되지 않으므로 None을 반환한다.
    """
    if indicator_a.size == 0 or indicator_b.size == 0:
        return None
    if rng is None:
        rng = np.random.default_rng(0)
    n_a = indicator_a.size
    pooled = np.concatenate([indicator_a, indicator_b])
    observed = abs(indicator_a.mean() - indicator_b.mean())
    n_ge = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(pooled)
        diff = abs(shuffled[:n_a].mean() - shuffled[n_a:].mean())
        if diff >= observed:
            n_ge += 1
    return n_ge / n_perm


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """랭커 y_score로 y_true(0/1)를 얼마나 잘 뽑는지 average precision으로 잰다.
    sklearn과 동일 정의(재현율 스텝마다 정밀도를 평균)를 직접 구현해 의존성 없이 쓴다.
    y_score를 내림차순 정렬한 뒤 순차 정밀도·재현율에서 재현율이 오른 지점만 평균한다."""
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    n_pos = y_sorted.sum()
    if n_pos == 0:
        return 0.0
    tp_cum = np.cumsum(y_sorted)
    precision_at_k = tp_cum / (np.arange(y_sorted.shape[0]) + 1)
    return float((precision_at_k * y_sorted).sum() / n_pos)


def ap_permutation_p(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_perm: int = 200,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """s_k 랭킹의 AP가 무작위 랭킹보다 유의하게 높은지 permutation으로 검정한다.
    y_true(양성 여부)를 셔플해 귀무 AP 분포를 만들고, 관측 AP 이상이 나온 비율을 p-value로 낸다.
    반환: (observed_ap, p_value)."""
    if rng is None:
        rng = np.random.default_rng(0)
    observed = average_precision(y_true, y_score)
    n_ge = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(y_true)
        null_ap = average_precision(shuffled, y_score)
        if null_ap >= observed:
            n_ge += 1
    return observed, n_ge / n_perm
