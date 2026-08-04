"""M2 — gain을 '무엇에 주소지정하느냐'를 정하는 basis 스위치.

`GainBasis`를 구현하는 클래스는 전부 학습 파라미터가 없는 순수 소프트웨어
인터페이스다. 신경망 모듈이 아니고 `Model/Adaptor.py`(LinearAdaptor 등)와
아무 관계가 없다. 유일한 학습 파라미터 `gain`은 `Model/intervention.py`의
`GainIntervention`이 갖는다.
"""

from typing import Protocol

import torch
from torch import Tensor
from torch.nn import functional as F

from Model.sae_runtime import FrozenSAE


class GainBasis(Protocol):
    """gain을 '무엇에 주소지정하느냐'를 정하는 스위치.

    학습 파라미터가 없는 순수 소프트웨어 인터페이스이며, 신경망 모듈이 아니다.
    """

    name: str
    """로그·결과표에 찍히는 arm 이름 ("latent" / "channel" / "random_dict")."""

    gain_dim: int
    """gain 벡터의 길이. GainIntervention이 이 값으로 파라미터를 만든다."""

    supports_absolute: bool
    """absolute()가 실제로 값을 반환할 수 있는지. False면 GainIntervention이
    residual=False 요청을 __init__ 시점에 즉시 ValueError로 거부한다."""

    def encode(self, h: Tensor) -> Tensor:
        """활성을 '무엇을 얼마나 켰나' 좌표로 바꾼다. [N,768] -> [N,D]. 항상 no_grad(상수)."""
        ...

    def delta(self, code: Tensor, gain: Tensor) -> Tensor:
        """gain이 1에서 벗어난 만큼을 활성 공간의 보정 벡터로 되돌린다. -> [N,768].
        gain이 전부 1이면 정확히 0을 반환해야 한다(no-op 보장).
        residual=True(기본) 경로가 쓴다: h+ = h + delta(...)"""
        ...

    def absolute(self, code: Tensor, gain: Tensor) -> Tensor:
        """h를 통째로 대체할 절대값을 만든다. -> [N,768].
        residual=False 경로가 쓴다: h+ = absolute(...)
        재구성 경로가 있는 딕셔너리 기반 basis만 구현한다."""
        ...


class LatentGainBasis:
    """SAE 개념 딕셔너리(K=12288)에 gain을 주소지정하는 제안 방법의 basis."""

    def __init__(self, sae: FrozenSAE):
        """SAE 하나만 들고 있으면 된다. gain_dim = sae.hidden_dim, name = "latent"."""
        self.sae = sae
        self.name = "latent"
        self.gain_dim = sae.hidden_dim
        self.supports_absolute = True

    def encode(self, h: Tensor) -> Tensor:
        """raw h를 받아 내부에서 정규화 후 SAE 코드로 인코딩한다. 항상 no_grad(상수)."""
        return self.sae.encode(h)

    def delta(self, code: Tensor, gain: Tensor) -> Tensor:
        """gain이 1에서 벗어난 만큼(code 단위)을 raw 활성 공간의 차분으로 되돌린다."""
        return self.sae.decode_delta((gain - 1.0) * code)

    def absolute(self, code: Tensor, gain: Tensor) -> Tensor:
        """gain으로 스케일한 코드를 raw 활성 공간의 절대값으로 되돌린다(residual=False 경로)."""
        return self.sae.decode_raw(gain * code)


class ChannelGainBasis:
    """SAE를 거치지 않고 raw 채널(D=768)에 직접 gain을 거는 통제군.

    모든 토큰에 동일하게 적용되는 고정 대각 사상이라 "개념이 있는 곳에서만
    누른다"를 표현할 수 없다 — 이게 4A가 격리하려는 차이다. 재구성 경로가
    없으므로 residual=False(absolute)는 정의되지 않는다.
    """

    def __init__(self, dim: int = 768, centered: bool = False, token_mean: Tensor | None = None):
        """SAE를 안 쓰므로 들고 있을 게 거의 없다. gain_dim = dim, name = "channel".
        centered=True면 token_mean이 필요하다(스케일 정합 확인용 변형)."""
        if centered and token_mean is None:
            raise ValueError("centered=True인데 token_mean이 주어지지 않음")
        self.dim = dim
        self.centered = centered
        self.token_mean = token_mean
        self.name = "channel"
        self.gain_dim = dim
        self.supports_absolute = False

    def encode(self, h: Tensor) -> Tensor:
        """코드가 곧 raw 활성 h다(SAE를 거치지 않음). detach로 상수 취급을 강제한다."""
        return h.detach()

    def delta(self, code: Tensor, gain: Tensor) -> Tensor:
        """gain이 1에서 벗어난 만큼을 raw 공간에서 elementwise로 적용한다.
        centered=True면 token_mean을 뺀 편차에 적용한다(스케일 정합 확인용)."""
        if self.centered:
            return (gain - 1.0) * (code - self.token_mean)
        return (gain - 1.0) * code

    def absolute(self, code: Tensor, gain: Tensor) -> Tensor:
        """채널 basis는 재구성 경로가 없어 residual=False가 정의되지 않는다."""
        raise NotImplementedError("채널 basis는 재구성 경로가 없어 residual=False가 정의되지 않는다")


class RandomDictGainBasis:
    """SAE의 '학습된 딕셔너리' 자리에만 무작위 딕셔너리를 꽂은 통제군.

    SAE arm과 구조·희소성·파라미터 수가 동일하고 딕셔너리만 학습되지 않았다.
    "왜 학습된 SAE인가"를 격리하는 가장 날카로운 통제군.
    """

    def __init__(self, sae: FrozenSAE, *, target_l0: int, seed: int):
        """정규화 통계는 SAE 것을 그대로 쓴다(정규화 공간을 맞춰야 비교가 성립).
        - seed 고정 R을 보관, 학습하지 않는다
        - target_l0: FrozenSAE.l0()로 측정한 SAE 실측값. 희소성을 맞추는 유일한 손잡이
        - gain_dim = sae.hidden_dim (K), name = "random_dict"
        하드코딩하지 말 것 — 호출자가 실측값을 주입해야 한다."""
        self.sae = sae
        self.target_l0 = target_l0
        self.seed = seed
        self.name = "random_dict"
        self.gain_dim = sae.hidden_dim
        self.supports_absolute = True

        # CPU generator로 뽑아야 device·GPU 종류와 무관하게 같은 seed가 같은 R을 준다.
        # 만든 뒤 SAE가 있는 device로 옮긴다 — basis는 nn.Module이 아니라서
        # GainIntervention.to(device)가 여기까지 닿지 않기 때문이다.
        generator = torch.Generator().manual_seed(seed)
        r = torch.randn(sae.hidden_dim, sae.input_dim, generator=generator)
        r = r / r.norm(dim=1, keepdim=True)  # W_dec와 동일하게 행 unit-norm
        self.R = r.to(sae.token_std.device)

    def _topk_sparsify(self, c: Tensor) -> Tensor:
        """행마다 target_l0개만 남기고 나머지는 0으로 만들어 SAE 실측 L0에 sparsity를 맞춘다."""
        k = self.target_l0
        if k >= c.shape[1]:
            return c
        _, idx = c.topk(k, dim=1)
        mask = torch.zeros_like(c, dtype=torch.bool)
        mask.scatter_(1, idx, True)
        return c * mask

    def encode(self, h: Tensor) -> Tensor:
        """raw h를 SAE와 동일한 정규화 공간으로 옮긴 뒤 무작위 딕셔너리로 인코딩하고
        target_l0에 맞춰 top-k만 남긴다. 항상 no_grad(상수) — 호출자가 code를 재사용한다."""
        with torch.no_grad():
            x = self.sae.normalize(h)
            c = F.relu(x @ self.R.T)
            return self._topk_sparsify(c)

    def delta(self, code: Tensor, gain: Tensor) -> Tensor:
        """FrozenSAE.decode_delta와 완전히 같은 식이되 딕셔너리만 R로 바뀐다.
        sae.decode_delta를 못 쓰는 유일한 이유가 '딕셔너리가 다르다'는 것이고,
        그 차이가 곧 이 통제군이 격리하려는 변수다."""
        return ((gain - 1.0) * code) @ self.R * self.sae.token_std

    def absolute(self, code: Tensor, gain: Tensor) -> Tensor:
        """residual=False 경로. R로 만든 재구성으로 h를 통째 대체한다.
        b_dec에 해당하는 게 없으므로 token_mean만 되돌린다."""
        return (gain * code) @ self.R * self.sae.token_std + self.sae.token_mean
