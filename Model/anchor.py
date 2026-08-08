"""M5 — `Model/anchor.py`.

`gain`을 초기값 1 쪽으로 잡아당기는 정규화 항. 학습 파라미터는 없다(`nn.Module`을
쓰는 건 `.to(device)`로 buffer를 함께 옮기기 위해서일 뿐, `named_parameters()`는
항상 빈 이터레이터를 준다 — `GainIntervention`의 전역 불변식 "gain 외 학습
파라미터 없음"과 무관하다).

**이론적 근거** (논문 명제로 쓸 수 있는 유일한 보증, phaseM.md M5 절):
정류점에서 `∇L_task(g*) + lam·(g*−1) = 0` 이므로

    ‖g* − 1‖ = ‖∇L_task(g*)‖ / lam ≤ G / lam

즉 no-op(`gain ≡ 1`)으로부터의 이탈 반경이 `G/lam` 이내로 묶인다(비폭주). 이건
수렴 보증이 아니다 — 하류가 비선형이고 pseudo-label이 `gain`에 의존해 목적함수가
매 스텝 바뀐다. 이 선을 넘어 주장하지 말 것.

**`ck` 모드의 두 가지 설계 결정** (계약이 정하지 않아 여기서 명시적으로 정한다):

1. **음수 `c_k`를 클램프한다(`|c_k|`가 아니라 `clamp(min=0)`).** `c_k`가 음수라는
   건 "그 latent를 껐더니 오히려 정확도가 올랐다" — 즉 인과적으로 유해했다는
   뜻이다(phaseM.md M6). 그런 latent를 `gain=1` 쪽으로 붙잡을 이유가 없다.
   `|c_k|`를 쓰면 "해로운 latent를 건드리지 말라"는 페널티가 생겨 정확히 반대
   신호가 된다 — 클램프가 유일하게 방향이 맞는 선택이다.
2. **클램프 후 0이 아닌 항목들의 평균이 1이 되도록 재정규화한다.** M6에서 실측한
   `c_k`는 1e-3 자릿수다(phaseM.md M6 절 "`c_k`는 실측상 매우 작다"). 원값을
   그대로 가중치로 쓰면 `lam·Σ c_k(g_k−1)²`가 같은 `lam`의 `l2` 모드보다 약
   1000배 약해져, 두 모드가 같은 `lam` 격자에서 비교 불가능해진다 — 이러면
   실험 4B의 anchor 축("`ck` 가중이 균일 L2 대비 기여하는가")이 `lam`의 스케일
   차이만 재는 무의미한 실험이 된다. 0이 아닌 가중치의 평균을 1로 맞추면 `ck`
   모드는 `l2`와 **같은 `lam` 스케일에서 "균일하게 당길지 선택적으로 당길지"만
   달라지는 비교**가 된다. 상대적 크기(어떤 latent를 더 세게 붙잡을지)는
   정규화가 스칼라 배율이라 그대로 보존된다.
   - 클램프 후 전부 0이면(측정된 모든 `c_k`가 0 이하) 정규화를 생략하고 가중치를
     전부 0으로 둔다 — 이 경우 `ck` 모드는 `lam`과 무관하게 상시 0을 반환한다.
     이는 퇴화 케이스이며, 그 latent 집합에 대해 "보존할 근거가 없다"는 신호로
     읽어야 한다(측정 데이터 부족 신호일 수도 있다 — M6의 `n_images` 각주 참고).

`lam == 0`은 `mode`와 무관하게 `off`와 **비트 단위로 동일**해야 한다(M5 DoD).
`0.0 * 유한값 == 0.0`이 IEEE 754에서 정확히 성립하므로, `off`처럼 별도 분기를
두지 않고 공식 `(lam / 2) * (w * (gain - 1) ** 2).sum()`을 그대로 평가해도
`lam=0`이면 정확히 0이 나온다. `off`만 별도 조기 반환을 둔 이유는 `w`(특히 `ck`
모드의 정규화된 `c_k`)를 아예 만들 필요가 없어 계산을 스킵할 수 있어서다.
"""

from __future__ import annotations

import warnings

import torch
from torch import Tensor, nn

_VALID_MODES = ("off", "l2", "ck")


class GainAnchor(nn.Module):
    def __init__(self, mode: str, lam: float, c_k: Tensor | None = None):
        """gain을 초기값 1 쪽으로 잡아당기는 정규화 항. 학습 파라미터는 없다.
        - mode: "off"(항상 0) | "l2"(모든 latent 균일 가중 1) | "ck"(c_k가 큰
          latent를 더 세게 1에 붙잡음). "off"/"l2"/"ck" 외 값은 ValueError.
        - lam: 당기는 세기. 클수록 개입이 보수적이 된다. 0이면 mode와 무관하게
          항상 0을 반환한다(off와 수치적으로 동일).
        - c_k: mode="ck"일 때만 필요(그 외 모드에서 None이 아니어도 무시된다).
          None이면 ValueError. buffer `self._ck_weight`로 클램프·정규화까지
          끝낸 최종 가중치를 보관한다(모듈 docstring의 두 결정 참고) — forward마다
          다시 계산하지 않는다."""
        super().__init__()
        if mode not in _VALID_MODES:
            raise ValueError(f"mode는 {_VALID_MODES} 중 하나여야 한다: {mode!r}")
        if mode == "ck" and c_k is None:
            raise ValueError('mode="ck"는 c_k가 필요하다')
        self.mode = mode
        self.lam = float(lam)

        if mode == "ck":
            clamped = c_k.clamp(min=0.0)
            positive_mask = clamped > 0
            if positive_mask.any():
                weight = clamped / clamped[positive_mask].mean()
            else:
                # 퇴화: 보존할 만한 latent가 하나도 없다. 벌점이 항상 0이 되어
                # 이 모드가 mode="off"와 구별되지 않는다 — 실험 4B의 anchor 축이
                # "차이 없음"으로 나오는데 원인은 설계가 아니라 c_k 입력이다.
                # 조용히 넘어가면 그 진단이 불가능하므로 반드시 알린다.
                warnings.warn(
                    f'mode="ck"인데 양수 c_k가 하나도 없다(전체 {c_k.numel()}개). '
                    "anchor가 항상 0을 반환해 사실상 off와 같아진다. "
                    "c_k를 충분한 n_images로 측정했는지 확인할 것 "
                    "(c_k의 분해능은 1/n_images이고 실측값은 1e-3 자릿수다 — "
                    "docs/plans/contracts/phaseM.md M6 참조).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                weight = clamped  # 전부 0
            self.register_buffer("_ck_weight", weight)
        else:
            self.register_buffer("_ck_weight", torch.empty(0))

    def forward(self, gain: Tensor) -> Tensor:
        """현재 gain이 1에서 얼마나 벗어났는지에 대한 벌점 스칼라를 반환한다.
        적응 손실에 더해져 gain이 폭주하는 걸 막는다.
        - "off": gain과 같은 dtype/device의 0-dim 텐서 0.0을 그대로 반환.
        - "l2": (lam/2) * Σ(gain-1)^2.
        - "ck": (lam/2) * Σ w_k (gain_k-1)^2, w는 __init__에서 정규화한 가중치.
          gain.shape != w.shape면 ValueError(둘 다 [gain_dim] 1차원이어야 한다)."""
        if self.mode == "off":
            return gain.new_zeros(())

        if self.mode == "l2":
            w = gain.new_ones(())
        else:  # "ck"
            if self._ck_weight.shape != gain.shape:
                raise ValueError(
                    f"c_k.shape {tuple(self._ck_weight.shape)} != gain.shape {tuple(gain.shape)}"
                )
            w = self._ck_weight.to(device=gain.device, dtype=gain.dtype)

        return (self.lam / 2) * (w * (gain - 1.0) ** 2).sum()
