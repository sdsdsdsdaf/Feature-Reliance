# Contract Brief — Phase M (방법 구현)

> [Plans.md](../Plans.md) Phase M task의 **구현 계약**. task ledger는 Plans.md, 정답 조건은 [experiment_plan.md](../experiment_plan.md).
> 여기 적힌 **시그니처·동작·불변식이 계약**이고 내부 구현 방식은 자유. 불변식은 전부 테스트로 강제한다.
> 배치 규약: 모듈은 `Model/`, 변환은 `Utils/`. 새 top-level 패키지를 만들지 않는다.

## 전역 불변식 (모든 모듈 테스트에 포함)

```python
def assert_only_gain_trainable(module):
    trainable = [n for n, p in module.named_parameters() if p.requires_grad]
    assert trainable == ["gain"], f"gain 외 학습 파라미터 발견: {trainable}"
```

`gain` 벡터 하나만 학습한다. 백본·SAE·무작위 딕셔너리 전부 frozen이고 **네트워크에 새 모듈을 추가하지 않는다.**

## 검증된 전제 (이 세션 실측)

| 항목 | 값 | 출처 |
|---|---|---|
| `W_enc` / `W_dec` | `[768, 12288]` / `[12288, 768]`, `W_dec` 행 unit-norm | ckpt |
| `token_mean` / `token_std` | `[1, 768]` | ckpt |
| `active_threshold` | `0.2` | ckpt |
| hidden_dim | `12288` = 768 × **16** (json의 `expansion: 32`는 stale) | ckpt |
| 패치당 실측 `L0` | `≈ 497` (thr 0.2), 활성 질량 top-100이 21% | imagenette 200장 측정 |
| in-domain FVU | `≈ 4e-4` | 동일 측정 |
| timm forward | `patch_embed → _pos_embed → patch_drop → norm_pre → blocks[0..11] → norm → forward_head` | `inspect.getsource` |

---

## M1 — `Model/sae_runtime.py`

**목적**: SAE를 "추론 계기"로 감싼다. 정규화 왕복을 캡슐화해 호출자가 `token_mean`/`token_std`를 잊는 사고를 원천 차단.

```python
class FrozenSAE:
    def __init__(self, sae: VanillaL1SAE, token_mean: Tensor, token_std: Tensor,
                 active_threshold: float, meta: dict):
        """학습된 SAE와 그 정규화 통계를 한 객체로 묶는다.
        보관: sae(파라미터 전부 requires_grad=False, eval), token_mean/token_std/
              active_threshold는 buffer로, meta는 ckpt의 model_name·target_block·token_scope.
        직접 부르는 일은 드물고 보통 from_checkpoint를 쓴다."""

    @classmethod
    def from_checkpoint(cls, path: str, device) -> "FrozenSAE":
        """ckpt 파일 하나에서 SAE와 정규화 통계를 전부 복원해 조립한다."""

    def normalize(self, h: Tensor) -> Tensor:
        """백본 활성(raw) -> SAE 입력 공간.  (h - token_mean) / token_std.
        SAE는 정규화된 토큰으로 학습됐으므로 raw h를 그냥 넣으면 안 된다."""

    def denormalize(self, x: Tensor) -> Tensor:
        """normalize의 역변환. SAE 출력(정규화 공간)을 백본이 이해하는 raw 공간으로 되돌린다."""

    def encode(self, h: Tensor, *, normalized: bool = False) -> Tensor:
        """활성 -> 개념 코드. [N,768] -> [N,12288] 비음수 희소 벡터.
        normalized=False면 내부에서 normalize를 먼저 부른다. 항상 no_grad."""

    def decode(self, z: Tensor) -> Tensor:
        """개념 코드 -> 활성. encode의 역방향, 단 정규화 공간 출력이다."""

    def decode_delta(self, code_delta: Tensor) -> Tensor:
        """코드의 '변화량'을 raw 활성 공간의 변화량으로 되돌린다. [N,K] -> [N,768].
        decode()는 절대량이라 b_dec·token_mean이 붙지만, 이건 차분이라 그 덧셈 항들이
        상쇄되고 곱셈 항 token_std만 남는다:  code_delta @ W_dec * token_std
        gain 개입이 쓰는 유일한 경로 — 덕분에 호출자가 W_dec·token_std를 직접 만질 일이 없다."""

    def decode_raw(self, z: Tensor) -> Tensor:
        """코드를 raw 활성 공간의 '절대량'으로 되돌린다. denormalize(decode(z)).
        decode_delta가 차분이라면 이건 전체 값 — residual=False(h 통째 대체) 경로가 쓴다."""

    def reconstruct(self, h: Tensor) -> Tensor:
        """encode -> decode_raw 왕복. SAE를 통과시킨 h를 raw 공간으로 돌려준다.
        '이 SAE가 표현을 얼마나 보존하나'를 눈으로 볼 때 쓴다. = decode_raw(encode(h))"""

    def fvu(self, h: Tensor) -> float:
        """재구성 실패율. 0이면 완벽, 클수록 SAE가 이 입력을 못 담고 있다는 뜻.
        실험 1의 게이트 임계와 M7의 런타임 판정이 이 값을 공유한다."""

    def l0(self, z: Tensor, threshold: float | None = None) -> float:
        """토큰 하나당 평균 몇 개의 개념이 켜졌나. 희소성 지표.
        M2의 RandomDictGainBasis가 sparsity를 맞출 때 이 측정값을 받아 쓴다."""
```

### 동작

**`from_checkpoint`**
1. `ck = torch.load(path, map_location="cpu")`
2. `VanillaL1SAE(ck["input_dim"], ck["hidden_dim"], b_dec_init=ck["b_dec_init"], dec_bias_mode="geom")` 생성 후 `load_state_dict(ck["sae_state_dict"])`
3. `.eval()`, 모든 파라미터 `requires_grad_(False)`, `.to(device)`
4. `token_mean`/`token_std`/`active_threshold`를 buffer로 등록. `meta`에 `model_name`·`target_block`·`token_scope` 보관
5. **검증**: `token_mean.shape == (1, input_dim)`, `W_dec.norm(dim=1)`이 전부 1에 근사(unit-norm 유지 확인)

**`normalize(h)`** → `(h - token_mean) / token_std`. `[..., 768]` 어떤 랭크든 브로드캐스트.
**`denormalize(x)`** → `x * token_std + token_mean`.

**`encode(h, normalized=False)`**
1. `x = h if normalized else self.normalize(h)`
2. `torch.no_grad()` 안에서 `z = relu((x - b_dec) @ W_enc + b_enc)` → `[N, 12288]`
3. **항상 no_grad.** 우리 방법에서 `z`는 상수다(gradient는 `delta`를 통해서만 `gain`에 도달). 진단용으로도 grad가 필요 없다.

**`decode(z)`** → `z @ W_dec + b_dec`, **정규화 공간** `[N, 768]`.
**`reconstruct(h)`** → `denormalize(decode(encode(h)))`, raw 공간.

**`decode_delta(code_delta)`** → `(code_delta @ W_dec) * token_std`, raw 공간 `[N, 768]`.

정의상 두 재구성의 차이이고, 전개하면 덧셈 항이 전부 상쇄된다:

```
denormalize(decode(z + Δ)) − denormalize(decode(z))
  = [((z+Δ) @ W_dec + b_dec) * std + mean] − [(z @ W_dec + b_dec) * std + mean]
  = (Δ @ W_dec) * std
```

- **`decode` 두 번 부르고 빼는 방식으로 구현하지 말 것.** 큰 수 두 개의 차라서 정밀도가 나빠지고 연산도 두 배다. 위 한 줄로 직접 계산한다.
- `code_delta`가 0이면 정확히 0을 반환한다 — M2의 no-op 비트 동일성이 여기서 나온다.
- **이 메서드가 `W_dec`·`token_std`를 외부에 노출하지 않기 위한 유일한 통로다.** basis 구현이 `sae.W_dec`를 직접 만지면 M1의 캡슐화가 깨지고, 정규화를 빠뜨리는 사고가 다시 가능해진다.

**`fvu(h)`**
1. `x = normalize(h)`, `xh = decode(encode(x, normalized=True))`
2. `mse = ((x - xh) ** 2).mean()`, `var = x.var(unbiased=False)` — **전 원소 기준**
3. `return (mse / max(var, 1e-12)).item()`
4. **반드시 `Utils/SAE_utils.evaluate_sae_tokens`의 `normalized_mse`와 동일 정의를 쓸 것.** 정의가 갈리면 실험 1의 게이트 임계와 M7의 런타임 판정이 서로 다른 척도가 되어 게이트가 무의미해진다.

**`l0(z, threshold=None)`** → `(z > (threshold or self.active_threshold)).sum(-1).float().mean().item()`

### 엣지 케이스
- `h`가 `[B, T, 768]`이면 내부에서 `[B*T, 768]`로 펴서 처리하고 원래 랭크로 복원
- dtype: 내부 연산은 `float32`로 승격(ckpt가 bf16으로 학습됐어도 진단 수치는 fp32에서 낸다)

---

## M2 — `Model/gain_basis.py` + `Model/intervention.py`

**목적**: 실험 4A의 세 arm이 **같은 루프에 꽂히도록** gain의 주소지정만 분리한다.

```python
class GainBasis(Protocol):
    """gain을 '무엇에 주소지정하느냐'를 정하는 스위치.
    학습 파라미터가 없는 순수 소프트웨어 인터페이스이며, 신경망 모듈이 아니고
    기존 Model/Adaptor.py(LinearAdaptor 등)와 아무 관계가 없다."""

    name: str            # 로그·결과표에 찍히는 arm 이름 ("latent" / "channel" / "random_dict")
    gain_dim: int        # gain 벡터의 길이. GainIntervention이 이 값으로 파라미터를 만든다

    def encode(self, h: Tensor) -> Tensor:
        """활성을 '무엇을 얼마나 켰나' 좌표로 바꾼다. [N,768] -> [N,D]. 항상 no_grad(상수)."""

    def delta(self, code: Tensor, gain: Tensor) -> Tensor:
        """gain이 1에서 벗어난 만큼을 활성 공간의 보정 벡터로 되돌린다. -> [N,768].
        gain이 전부 1이면 정확히 0을 반환해야 한다(no-op 보장).
        residual=True(기본) 경로가 쓴다: h⁺ = h + delta(...)"""

    def absolute(self, code: Tensor, gain: Tensor) -> Tensor:
        """h를 통째로 대체할 절대값을 만든다. -> [N,768].
        residual=False 경로가 쓴다: h⁺ = absolute(...)
        재구성 경로가 있는 딕셔너리 기반 basis만 구현한다.
        ChannelGainBasis는 ĥ에 해당하는 게 없으므로 NotImplementedError를 던진다."""
```

`delta`는 전 basis 필수, `absolute`는 딕셔너리 기반만 구현한다. 두 경로의 관계:

```
absolute(code, gain) − delta(code, gain) = ĥ   (재구성, gain과 무관)
h + delta  vs  absolute   →  차이는 정확히 h − ĥ (재구성 오차)
```

### `LatentGainBasis` (제안, `D = K = 12288`)

```python
class LatentGainBasis:
    def __init__(self, sae: FrozenSAE):
        """SAE 하나만 들고 있으면 된다. gain_dim = sae.hidden_dim (12288), name = "latent"."""
        self.sae = sae

    def encode(self, h):
        return self.sae.encode(h)                 # raw h를 받아 내부에서 정규화

    def delta(self, code, gain):
        return self.sae.decode_delta((gain - 1.0) * code)

    def absolute(self, code, gain):
        return self.sae.decode_raw(gain * code)
```

두 줄의 대비가 앞서 나온 `(gain−1)` vs `gain` 질문의 답이다 — **차분이냐 절대량이냐가 `-1`의 유무를 결정한다.**

**`W_dec`·`token_std`를 직접 만지지 않는다.** 행렬곱과 역정규화는 전부 `FrozenSAE.decode_delta` 안에 있고, 이 클래스가 하는 일은 "gain이 1에서 벗어난 만큼을 코드에 곱해 넘기는 것"뿐이다. 수식 `h⁺ = h + sd ⊙ W_dec^T((a−1)⊙z)`의 `sd ⊙ W_dec^T` 부분이 M1 소관이라는 뜻이다.

**왜 `gain * code`가 아니라 `(gain − 1) * code`인가** — `delta`가 `h`에 *더해지고*, `h`에는 이미 그 신호가 들어 있기 때문이다.

```
원하는 값:  (gain⊙z) @ W_dec + b_dec
현재 h:          z  @ W_dec + b_dec      (≈ ĥ)
더해야 할 차이:  ((gain−1)⊙z) @ W_dec
```

`gain * code`를 넘기면 `gain=1`에서 `h⁺ = h + ĥ ≈ 2h`가 되어 무개입 지점에서 신호가 두 배로 뛴다.
**`-1`은 `h +`와 짝을 이루는 항이라 둘 중 하나만 바꾸면 깨진다.** `gain * code`가 맞는 경우는 residual off, 즉 `h`를 통째로 갈아끼울 때뿐이다:

```python
# residual ON  (기본) — gain=1이 비트 동일 무개입
h_plus = h + sae.decode_delta((gain - 1.0) * code)
# residual OFF (4B ablation) — gain=1이어도 재구성 오차만큼 어긋남
h_plus = sae.denormalize(sae.decode(gain * code))
```

residual on이 기본인 이유: `h`에 남아 있는 **SAE가 재구성하지 못하는 잔차**를 보존한다. 개입이 SAE가 이해하는 성분만 건드리고 나머지는 그대로 통과한다. off는 그 잔차를 통째로 버리므로, FVU가 큰 계기(희소한 SAE일수록 크다)에서 손실이 커진다.

### `ChannelGainBasis` (통제군, `D = 768`)

```python
def __init__(self, dim: int = 768, centered: bool = False, token_mean: Tensor | None = None):
    """SAE를 안 쓰므로 들고 있을 게 거의 없다. gain_dim = dim (768), name = "channel".
    centered=True면 token_mean이 필요하다(스케일 정합 확인용 변형)."""

encode(h):     return h                                   # 코드가 곧 h
delta(code, gain):
    centered=False:  return (gain - 1.0) * code           # raw 공간 elementwise
    centered=True:   return (gain - 1.0) * (code - token_mean)
absolute(code, gain):
    raise NotImplementedError("채널 basis는 재구성 경로가 없어 residual=False가 정의되지 않는다")
```

SAE를 거치지 않는다. 모든 토큰에 **동일하게 적용되는 고정 대각 사상**이라 "개념이 있는 곳에서만 누른다"를 표현할 수 없다 — 이게 4A가 격리하려는 차이다.
`centered` 플래그는 스케일 정합 확인용. 기본 False로 보고하되 4A에서 두 변형을 모두 돌려 결론이 뒤집히지 않는지 확인한다.

### `RandomDictGainBasis` (통제군, `D = K`)

```python
class RandomDictGainBasis:
    def __init__(self, sae: FrozenSAE, *, target_l0: int, seed: int):
        """SAE의 '학습된 딕셔너리' 자리에만 무작위 딕셔너리를 꽂은 통제군.
        정규화 통계는 SAE 것을 그대로 쓴다(정규화 공간을 맞춰야 비교가 성립).
        - seed 고정 R을 buffer로 보관, 학습하지 않는다
        - target_l0: M1의 l0()로 측정한 SAE 실측값. 희소성을 맞추는 유일한 손잡이
        - gain_dim = sae.hidden_dim (K), name = "random_dict\""""
        R = randn(K, dim, generator=Generator().manual_seed(seed))
        self.R = R / R.norm(dim=1, keepdim=True)          # W_dec와 동일하게 행 unit-norm

    def encode(self, h):
        1. x = self.sae.normalize(h)                      # 정규화는 M1에 위임
        2. c = relu(x @ self.R.T)                         # [N,K] 비음수화
        3. TopK per row: target_l0 개만 남기고 나머지 0     # SAE 실측 L0에 sparsity 매칭
        4. return c

    def delta(self, code, gain):
        """FrozenSAE.decode_delta와 완전히 같은 식이되 딕셔너리만 R로 바뀐다.
        sae.decode_delta를 못 쓰는 유일한 이유가 '딕셔너리가 다르다'는 것이고,
        그 차이가 곧 이 통제군이 격리하려는 변수다."""
        return ((gain - 1.0) * code) @ self.R * self.sae.token_std

    def absolute(self, code, gain):
        """residual=False 경로. R로 만든 재구성으로 h를 통째 대체한다.
        b_dec에 해당하는 게 없으므로 token_mean만 되돌린다."""
        return (gain * code) @ self.R * self.sae.token_std + self.sae.token_mean
```

SAE arm과 **구조·희소성·파라미터 수가 동일하고 딕셔너리만 학습되지 않았다.** 이게 "왜 학습된 SAE인가"를 격리하는 가장 날카로운 통제군이다.
`target_l0`은 하드코딩하지 말고 **M1의 `l0()`로 같은 데이터에서 측정한 값**을 주입한다.

### `GainIntervention(nn.Module)`

```python
def __init__(self, backbone, basis: GainBasis, hook_block: int = 10,
             num_prefix_tokens: int = 1, residual: bool = True):
    """백본과 basis를 묶고 이 모듈의 유일한 학습 파라미터를 만든다.
    - backbone 전체를 requires_grad_(False)로 얼린다
    - self.gain = nn.Parameter(torch.ones(basis.gain_dim))   ← 유일한 학습 대상
    - hook_block: 몇 번째 블록 출력에 주입할지 (기본 10 = SAE가 학습된 지점)
    - num_prefix_tokens: cls 토큰 수. 패치 토큰만 개입 대상이라 앞에서 잘라낸다
    - residual: True면 h + delta(원본 보존), False면 absolute로 h 통째 대체.
      False인데 basis가 absolute를 구현 안 했으면 __init__ 시점에 ValueError로 즉시 실패시킨다
      (스트림 한참 돌다가 터지면 안 된다)"""

def trainable_parameters(self) -> list[Parameter]:
    """optimizer에 넘길 파라미터. 항상 [self.gain] 하나뿐이어야 한다(전역 불변식)."""

def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
    """이미지 배치를 받아 (logits, h') 반환.
    h'는 대조학습에서 거리를 재는 backbone feature(분류기 직전 표현)다."""

def reset(self) -> None:
    """gain을 1로 되돌려 무개입 상태로 만든다. 도메인 전환·시드 반복 사이에 부른다."""
```

**동작 (`forward`)** — timm 내부 API를 직접 호출해 상류를 `no_grad`로 묶는다. 훅+detach 방식은 forward 시 상류 그래프를 그대로 쌓아 메모리를 낭비하므로 쓰지 않는다.

```python
with torch.no_grad():
    h = backbone.patch_embed(x)
    h = backbone._pos_embed(h)          # cls 토큰 결합 -> [B, 197, 768]
    h = backbone.patch_drop(h)
    h = backbone.norm_pre(h)
    for blk in backbone.blocks[: hook_block + 1]:
        h = blk(h)
    h = h.detach()
    prefix, patch = h[:, :num_prefix_tokens], h[:, num_prefix_tokens:]
    flat = patch.reshape(-1, patch.shape[-1])            # [B*196, 768]
    code = basis.encode(flat)                            # 상수

if self.residual:                                        # ← grad 경로 시작 (양쪽 모두)
    patch_plus = (flat + basis.delta(code, self.gain)).view_as(patch)
else:
    patch_plus = basis.absolute(code, self.gain).view_as(patch)   # h를 통째로 대체
h_plus = torch.cat([prefix, patch_plus], dim=1)

for blk in backbone.blocks[hook_block + 1:]:             # grad 통과 (weight는 frozen)
    h_plus = blk(h_plus)
h_plus = backbone.norm(h_plus)

feat   = backbone.forward_head(h_plus, pre_logits=True)  # 대조학습 거리를 재는 h'
logits = backbone.head(feat)
return logits, feat
```

`forward_head(pre_logits=True)`를 쓰는 이유: pooling·`fc_norm`을 손으로 재구현하면 timm 버전이 바뀔 때 조용히 어긋난다.

**불변식**
1. `trainable_parameters() == [self.gain]` — 정확히 1개
2. **`gain == 1`이면 `delta`가 정확히 0이라 무개입 forward와 비트 단위로 동일하다.** 근사가 아니다. `(gain−1)*code == 0` → `dh == 0`.
   ⚠️ Plans.md T0.3 시절 DoD의 "±재구성오차"는 **residual off 변형에만** 해당한다. residual on(기본)은 오차가 0이다. 실험 4B의 residual 축이 정확히 이 차이를 잰다.
3. `backbone.blocks[:hook_block+1]`의 파라미터에 `.grad`가 절대 채워지지 않는다
4. `reset()` 후 `gain`이 정확히 1

**두 경로의 관계** (4B가 재는 축):

| `residual` | 식 | `gain=1`일 때 | SAE가 못 담는 성분 `h − ĥ` |
|---|---|---|---|
| `True` (기본) | `h + delta(code, gain)` | **`h`, 비트 동일 무개입** | 그대로 통과 |
| `False` | `absolute(code, gain)` | `ĥ`, 재구성 오차만큼 어긋남 | **버려짐** |

두 경로의 차이는 정확히 `h − ĥ`다. 현 ckpt는 FVU가 `4e-4`라 거의 같지만, `L0` 수십 급의 진짜 희소한 SAE로 올리면 FVU가 커져 **희소할수록 residual on의 이득이 커진다.** 4B는 이 관계를 실측으로 확인하는 것이다.

`residual=False`는 `ChannelGainBasis`에 정의되지 않는다(`ĥ`가 없다) → 4A의 arm 비교는 **항상 `residual=True`로 고정**하고, residual 축은 4B에서 SAE arm 위에서만 돌린다.

---

## M3 — `Model/adacontrast.py`

### `MemoryBank`

```python
def __init__(self, dim: int, capacity: int, num_classes: int):
    """지나간 샘플의 (feature, 예측확률)을 담아둘 큐를 만든다.
    - dim: feature 차원(=backbone h' 크기).  capacity: 총 슬롯 수. 0이면 in-batch 모드
    - num_classes개의 링 버퍼로 쪼개 클래스당 capacity//num_classes씩 배분
    - feats/probs/ptr/size를 buffer로 미리 할당(스트림 중 재할당 금지)"""

def enqueue(self, feats: Tensor, probs: Tensor) -> None:
    """이번 배치를 큐에 밀어 넣는다. 가장 오래된 항목이 밀려난다. capacity=0이면 no-op."""

def knn_soft_vote(self, query: Tensor, k: int = 10, logits_weak: Tensor | None = None) -> Tensor:
    """큐에서 가장 닮은 이웃 k개를 찾아 그들의 예측을 평균낸다 -> pseudo-label [N,C].
    '내 예측'이 아니라 '나와 닮은 것들의 합의'라 자기강화가 덜하다."""

@property
def labels(self) -> Tensor:
    """큐에 든 각 항목의 pseudo-label(argmax) [size].
    contrastive에서 같은 라벨을 negative에서 빼는 데 쓴다."""
```

**동작**
- 저장 전 `feats`를 **L2 정규화**한다(코사인 kNN이므로). `probs`는 soft 그대로.
- **클래스 균형**: 클래스당 `capacity // num_classes` 크기의 링 버퍼를 두고 `probs.argmax(1)` 기준으로 배분. 전역 FIFO 하나면 흔한 클래스가 큐를 독점한다.
- `capacity == 0` → in-batch 모드. 저장하지 않고 `knn_soft_vote`가 **자기예측으로 폴백**한다(실험 6의 `buffer=0` 경로).
  폴백하려면 호출자의 weak 예측이 필요하므로 `logits_weak`를 인자로 받는다 — 초안 시그니처엔 없었고, 없으면 이 폴백을 구현할 방법이 없다. `size == 0`인 스트림 초반에도 같은 경로를 탄다.

```
knn_soft_vote(query, k):
    if capacity == 0:
        return softmax(logits_weak, dim=1)          # 호출자가 넘긴 weak 예측
    q   = normalize(query)                           # [N, dim]
    sim = q @ self.feats[:size].T                    # [N, size]
    idx = sim.topk(min(k, size), dim=1).indices
    return self.probs[idx].mean(dim=1)               # [N, C] soft
```

### `AdaContrastLoss(nn.Module)`

```python
def __init__(self, num_classes: int, tau: float = 0.07,
             w_ctr: float = 1.0, w_div: float = 0.1,
             use_diversity: bool = False,
             distance_space: str = "h"):
    """손실 항들의 가중치와 거리 계산 방식을 고정한다. 학습 파라미터는 없다.
    - tau: contrastive 온도. 작을수록 어려운 negative에 민감
    - w_ctr / w_div: contrastive · diversity 항 가중치
    - use_diversity: 기본 False. True면 한 클래스로 붕괴하는 걸 막는 항이 켜진다
    - distance_space: "h"(기본) | "z" | "logit" — 실험 4B가 뒤집는 축"""

def forward(self, h_weak, logits_weak, h_strong, logits_strong,
            bank, z_weak=None, z_strong=None) -> dict[str, Tensor]:
    """약하게/세게 증강한 같은 배치를 받아 손실을 계산한다.
    weak 쪽으로 pseudo-label을 만들고 strong 쪽이 그걸 맞추게 한다.
    z_*는 distance_space="z"일 때만 필요하다.
    반환은 합계 하나가 아니라 항별 dict {"total","pseudo","ctr","div"}."""
```

**동작**
1. `pseudo = bank.knn_soft_vote(h_weak, k)` → `[N, C]` soft. `yhat = pseudo.argmax(1)`
2. `L_pseudo = -(pseudo * log_softmax(logits_strong, 1)).sum(1).mean()` (soft-target CE)
3. **contrastive** — 거리를 재는 공간은 `distance_space`가 결정한다(4B 축):
   - `"h"` (기본): `q = normalize(h_strong)`, `k_pos = normalize(h_weak).detach()`
   - `"z"`: `h` 대신 `z_weak`/`z_strong`을 L2 정규화해 사용
   - `"logit"`: `logits_*`를 사용
   ```
   l_pos = (q * k_pos).sum(1, keepdim=True) / tau              # [N,1]
   l_neg = q @ bank.feats[:size].T / tau                       # [N,size]
   mask  = yhat[:, None] == bank.labels[None, :]               # same-pseudo 제외
   l_neg = l_neg.masked_fill(mask, float("-inf"))
   L_ctr = cross_entropy(cat([l_pos, l_neg], 1), zeros(N, dtype=long))
   ```
   `capacity == 0`이면 negative를 **배치 내 다른 샘플**에서 취하고 동일 마스킹을 적용한다.
4. `use_diversity`면 `p_bar = softmax(logits_strong,1).mean(0)`, `L_div = (p_bar * p_bar.log()).sum()` (주변분포 음엔트로피 → 한 클래스 붕괴 억제)
5. `total = L_pseudo + w_ctr * L_ctr + w_div * L_div`
6. **반환은 항별 dict** `{"total","pseudo","ctr","div"}` — 어느 항이 붕괴를 일으키는지 로그로 분리 추적해야 한다

**엣지 케이스**
- `mask`가 한 행을 전부 `-inf`로 만들면(모든 bank 항목이 같은 pseudo-label) 그 행의 `L_ctr`을 0으로 두고 `n_valid`로 평균낸다. 안 그러면 `NaN`.
- 스트림 초반 `size < k`면 `k = size`로 축소, `size == 0`이면 pseudo는 자기예측 폴백.

---

## M4 — `Utils/tta_transforms.py`

```python
def weak_transform(size: int = 224) -> Callable:
    """가볍게만 흔든 증강. 원본에 가까워 pseudo-label을 뽑는 쪽에 쓴다."""

def strong_transform(size: int = 224) -> Callable:
    """색·흐림까지 세게 흔든 증강. 이걸 보고도 같은 답을 내게 학습시키는 쪽에 쓴다."""

class TwoCropTransform:
    def __init__(self, weak, strong):
        """weak/strong 두 변환을 한 쌍으로 묶는다. 보관은 두 Callable뿐."""

    def __call__(self, img) -> tuple[Tensor, Tensor]:
        """같은 이미지 하나에서 (weak, strong) 두 뷰를 만들어 반환한다.
        DataLoader가 이걸 transform으로 받으면 배치가 자동으로 두 뷰 쌍이 된다."""
```

- **weak**: `RandomResizedCrop(size, scale=(0.8,1.0))` + `RandomHorizontalFlip` + normalize
- **strong**: weak + `ColorJitter(0.4,0.4,0.4,0.1)` p=0.8 + `RandomGrayscale` p=0.2 + `GaussianBlur` + normalize
- normalize 상수는 **timm의 `resolve_data_config`에서 가져온다.** 하드코딩 금지(백본 교체 시 조용히 어긋남)
- 기존 `Utils/transfrom.py`는 perturbation 실험용(`ChannelShuffle`·`BilateralFilter` 등)이라 재사용하지 않는다

---

## M5 — `Model/anchor.py`

```python
class GainAnchor(nn.Module):
    def __init__(self, mode: str, lam: float, c_k: Tensor | None = None):
        """gain을 초기값 1 쪽으로 잡아당기는 정규화 항. 학습 파라미터는 없다.
        - mode: "off" | "l2"(모든 latent 균일) | "ck"(c_k가 큰 latent를 더 세게 붙잡음)
        - lam: 당기는 세기. 클수록 개입이 보수적이 된다
        - c_k: mode="ck"일 때만 필요. buffer로 보관하고 음수는 0으로 클램프"""

    def forward(self, gain: Tensor) -> Tensor:
        """현재 gain이 1에서 얼마나 벗어났는지에 대한 벌점 스칼라를 반환한다.
        적응 손실에 더해져 gain이 폭주하는 걸 막는다."""
```

**동작**: `L_anchor = (lam / 2) * (w * (gain - 1) ** 2).sum()`

| `mode` | `w` | 용도 |
|---|---|---|
| `"off"` | — | `return gain.new_zeros(())` |
| `"l2"` | `1` | 균일 보존 (L2-to-init 통제군) |
| `"ck"` | `c_k` | 선택적 보존 — causal latent만 1에 붙잡음 |

**불변식**
- `lam == 0`이면 `mode` 무관하게 `mode="off"`와 **수치적으로 동일**
- `mode="ck"`인데 `c_k is None`이면 `ValueError`
- `c_k.shape == (gain_dim,)`, 음수는 0으로 클램프(정확도가 오히려 오른 latent는 보존 대상이 아님)

**이론적 근거** (논문 명제로 쓸 수 있는 유일한 보증):
정류점에서 `∇L_task(g*) + lam·(g*−1) = 0` 이므로

```
‖g* − 1‖ = ‖∇L_task(g*)‖ / lam ≤ G / lam
```

즉 **no-op으로부터의 이탈 반경이 `G/lam` 이내로 묶인다**(비폭주). `W_dec` 행이 unit-norm이라 `G`도 활성값으로 제어된다: `|∂L/∂g_k| ≤ ‖∂L/∂h⁺‖ · ‖sd‖_∞ · code_k`.
수렴 보증이 아니다 — 하류가 비선형이고 pseudo-label이 `gain`에 의존해 목적함수가 매 스텝 바뀐다. 이 선을 넘어 주장하지 말 것.

---

## M6 — `Model/diagnostics.py`

```python
def compute_c_k(intervention: GainIntervention, loader, device, *,
                candidates: Tensor | None = None,
                latent_chunk: int = 64,
                max_images: int | None = None) -> Tensor: ...
```

**정의**: `c_k = acc(gain ≡ 1) − acc(gain with gain_k = 0)`

`gain_k = 0`은 residual 주입에서 latent k의 기여만 빼는 것과 정확히 같다. 별도 ablation 경로를 만들 필요가 없고, **baseline이 무개입 forward와 비트 동일**이라 재구성 오차가 섞이지 않는다.

**동작**
1. `candidates`가 없으면 발화율로 후보를 추린다 — 전체 `K=12288` 중 실측상 한 번도 안 켜지는 게 절반이다. `latent_frequency`로 발화율 하한(예: 0.5%)을 넘는 것만.
2. 이미지마다 `h`(block10 출력)와 `code`를 **한 번만 계산해 캐시**한다.
3. baseline: `gain ≡ 1`로 꼬리(blocks 11~12 + norm + head)만 돌려 정확도 산출.
4. 후보를 `latent_chunk`개씩 끊어, 각 k에 대해 `gain_k=0`인 gain 벡터로 **꼬리만** 재실행. `delta`는 `code[:, k:k+1]` 한 열만 관여하므로 값싸다.
5. `c_k[k] = baseline_acc − ablated_acc[k]`

**비용** (반드시 문서화할 것): 캐시 덕에 latent당 비용이 전체 forward가 아니라 **꼬리 2/12 블록**이다. 그래도 `후보수 × 이미지수`에 비례하므로 `max_images`(예: 5000)와 후보 축소가 필수다. 전수 `K × 전체 source`는 현실적으로 불가능하다.

**불변식**: 반환 `[K]`, 후보에서 제외된 latent는 `0.0`. `c_k`는 부호가 있을 수 있다(음수 = 제거했더니 오히려 정확도 상승).

### ⚠️ `c_k`는 실측상 매우 작다 — 표본 수가 곧 분해능이다 (2026-08-05 측정)

`c_k`는 정확도 차이라 **분해능이 `1/n_images`로 양자화**된다. 실측(ImageNet-1k val, `LatentGainBasis`, 발화율 상위 200 latent):

| `n_images` | 분해능 | `c_k ≠ 0`인 latent | `c_k` 범위 |
|---|---|---|---|
| 64 | 1.6e-02 | **1 / 200** | 음수 1개뿐 |
| 256 | 3.9e-03 | 4 / 200 | `[0, +0.0039]` |
| 1024 | 9.8e-04 | **66 / 200** | `[−0.00098, +0.0029]` (양수 63 / 음수 3) |

**작은 표본에서 "`c_k`가 전부 0"으로 보이는 건 신호가 없어서가 아니라 눈금이 굵어서다.** `n`을 16배 늘리자 검출 수가 1 → 66으로 늘었다. 실제 `c_k`는 **1e-3 자릿수**이고, 그건 눈금 바로 위다.

**따라서**:
- **`max_images`를 아껴서는 안 된다.** `n ≤ 256`에서 잰 `c_k`는 사실상 양자화 잡음이다. 계약이 예로 든 5,000장이 최소선에 가깝다.
- 1000-way ImageNet에서 개별 latent 하나를 끄는 것의 효과가 작은 건 **자연스럽다** — 딕셔너리가 과완비·중복이고 주입 지점 뒤에 블록이 하나뿐이다.
- **Waterbirds(2-class, `y`↔`place` 95% 상관)에서는 훨씬 커야 한다.** T1.3이 `c_k`를 쓰는 곳이 거기이고, 배경 latent를 끄면 편향된 train 정확도가 눈에 띄게 흔들려야 정상이다. 그렇지 않다면 그것 자체가 T1.3 명제 (B)에 대한 부정 결과다.
- **M5의 `ck` 가중 anchor·T1.4b의 `anchor="ck"`는 이 크기에 의존한다.** `c_k`가 전부 1e-3이면 가중이 사실상 균일 L2와 구분되지 않는다 — 4B가 그 차이를 재는 실험이므로, 결과가 "차이 없음"으로 나오면 여기를 먼저 의심한다.

### 발화율 임계 (계약이 정하지 않았던 것)

후보 자동 선택의 "발화" 기준은 **`sae.active_threshold`(=0.2)** 다. `code > 0`으로 잡으면 ReLU 코드가 정확히 0인 경우가 거의 없어 **12288개 전부가 후보로 뽑혀** 축소가 무의미해진다. 임계 0.2·발화율 하한 0.5%로 잡으면 실측 **4,738 / 12,288 (38.6%)** 이 남는다. SAE가 아닌 basis(`ChannelGainBasis`)는 `active_threshold`가 없으므로 0.0으로 폴백한다.

### 비용 실측 (RTX 4070, 11.6GB)

`batch_size=8`, `latent_chunk=32`, 후보 300개, 64장 → **4.45초** (후보당 ≈0.015초), peak VRAM 2.78GB.
`batch_size=32`·`latent_chunk=64`는 **OOM** — 메가배치 `chunk×B×T`가 attention 메모리를 빠르게 먹는다. `batch_size=8`, `latent_chunk=16~32`가 안전선이다.

**T1.3의 호출 방식**: 실험 3은 이 함수를 **loader만 바꿔 두 번** 호출한다 — `c_k^train`(Waterbirds train, `y`↔`place` 95% 상관)과 `c_k^bal`(4 그룹을 최소 그룹 크기로 맞춘 균형 부분집합). 두 값의 격차가 "편향 source에서 `c_k`가 배경을 causal로 오판하는 폭"이고, 그게 [plan.md](../plan.md) §7 리스크의 실측값이다. 함수 자체는 수정할 필요가 없다.

---

## M7 — `Model/fvu_gate.py`

```python
class FVUGate:
    def __init__(self, threshold: float, mode: str = "skip"):
        """SAE 재구성이 못 믿을 상황을 판정하는 문지기. 학습 파라미터 없음.
        - threshold: 실험 1이 source FVU 분포의 상위 분위수로 정해 주입하는 값
        - mode: "skip"(넘으면 아예 갱신 중단) | "scale"(초과분만큼 완만히 약화)
        - 내부에 skipped 카운터를 두고 몇 배치가 걸렸는지 누적한다"""

    def scale_for(self, fvu: float) -> float:
        """이번 배치의 재구성 상태를 보고 gain 갱신을 얼마나 허용할지 0~1로 답한다.
        1.0이면 그대로, 0.0이면 이 배치는 학습하지 않는다."""
```

**동작**
- `mode="skip"`: `1.0 if fvu <= threshold else 0.0`
- `mode="scale"`: `clamp(threshold / max(fvu, eps), 0.0, 1.0)` — 초과분에 반비례 감쇠

**계약**: `threshold`는 하드코딩하지 않고 **실험 1이 source FVU 분포의 상위 분위수로 산출한 값을 주입**받는다. `scale_for`가 반환한 값은 M8에서 **손실에 곱해** gain 갱신 크기를 줄인다(옵티마이저 step을 건너뛰는 것보다 상태가 단순).
`scale == 0.0`이었던 배치 수를 반드시 카운트해 로그로 남긴다 — 게이트가 한 번도 안 걸리면 게이트가 무의미하다는 신호다(현 ckpt의 in-domain FVU가 4e-4라 실제로 그럴 위험이 있다).

---

## M8 — `Model/tta_runner.py`

```python
@dataclass
class TTAConfig:
    lr: float = 1e-3
    batch_size: int = 64
    steps_per_batch: int = 1
    buffer: int = 16384          # 0 = in-batch            (실험 6 축)
    passes: int = 1              # 1 = single-pass online  (실험 6 축)
    knn_k: int = 10
    tau: float = 0.07
    anchor_mode: str = "off"     # off | l2 | ck           (실험 4B 축)
    anchor_lam: float = 0.0
    distance_space: str = "h"    # h | z | logit           (실험 4B 축)
    use_diversity: bool = False
    fvu_threshold: float | None = None
    seed: int = 0

class OnlineTTARunner:
    def __init__(self, intervention, loss_fn, anchor, config, gate=None):
        """부품을 조립하고 optimizer·memory bank를 만든다.
        - optimizer = Adam(intervention.trainable_parameters(), lr=config.lr)
          ← trainable_parameters()가 [gain] 하나만 주므로 다른 건 절대 안 움직인다
        - bank = MemoryBank(dim=feature_dim, capacity=config.buffer, num_classes=...)
        - gate=None이면 FVU 게이트 없이 항상 갱신한다
        - config.seed로 난수를 고정한다"""

    def run(self, stream) -> dict:
        """스트림을 처음부터 끝까지 한 번(또는 config.passes회) 흘리며 적응시킨다.
        배치마다 '적응 전 예측'을 먼저 채점하고 그 다음에 gain을 갱신한다.
        반환: 정확도·궤적·게이트 통계가 담긴 결과 dict."""
```

**동작 (`run`)** — 배치마다:

```
1. (x_weak, x_strong, y) = batch
2. 적응 전 예측으로 online acc 집계:
       with no_grad: logits_pre, _ = intervention(x_weak)
       online_correct += (logits_pre.argmax(1) == y).sum()
   ↑ Tent 이래의 표준 프로토콜. 이 배치의 갱신 결과로 이 배치를 평가하면 정보 누설이다.
3. scale = gate.scale_for(sae.fvu(h_block10)) if gate else 1.0
4. for _ in range(steps_per_batch):
       logits_w, h_w = intervention(x_weak)
       logits_s, h_s = intervention(x_strong)
       losses = loss_fn(h_w, logits_w, h_s, logits_s, bank)
       L = losses["total"] + anchor(intervention.gain)
       opt.zero_grad(); (scale * L).backward(); opt.step()
5. bank.enqueue(key_w.detach(), softmax(logits_w, 1).detach())   # key_w는 아래 표 참조
6. 기록: acc_vs_time, ‖gain-1‖, 항별 손실, gate 발동 여부
```

### ⚠️ `distance_space`는 memory bank까지 따라가야 한다 (M3 구현 중 발견)

`AdaContrastLoss`는 negative를 `bank.feats`에서 가져오고 query와 내적한다. **query가 사는 공간과 bank가 저장한 공간이 다르면 차원부터 안 맞는다.** 위 초안처럼 `distance_space`와 무관하게 `h_w`를 넣으면 `"z"`·`"logit"` 조건이 **아예 실행되지 않고**, 그러면 실험 4B의 거리공간 축이 통째로 사라진다.

`bank`의 `dim`과 `enqueue`에 넣는 텐서를 **둘 다** `config.distance_space`에 맞춘다:

| `distance_space` | `MemoryBank(dim=...)` | `key_w` (enqueue 대상) |
|---|---|---|
| `"h"` (기본) | `feature_dim` (h' 차원) | `h_w` |
| `"z"` | `basis.gain_dim` (= K) | **패치 토큰 코드를 이미지 단위로 mean-pool한 `[B, K]`** |
| `"logit"` | `num_classes` | `logits_w` |

**`"z"`의 pooling은 계약에 없던 결정이다.** 코드 `z`는 패치 토큰 단위 `[B*196, K]`인데 bank는 이미지 단위로 저장한다. **패치 축 mean-pool**을 쓴다 — 이미지를 "어떤 개념이 얼마나 켜졌나"의 가방으로 보는 것이고, `s_k`(발화율)를 재는 방식과 같은 관점이다. `max`-pool은 robustness 확인용으로만 둔다.

**`GainIntervention.forward`는 코드를 반환하지 않는다.** `"z"`를 쓰려면 접근 경로가 필요하므로 **`forward(x, return_code: bool = False)`** 로 확장한다 — 기본값이 `False`라 기존 호출부(M6·T1.1·테스트)는 그대로 동작하고, `True`일 때만 `(logits, feat, code)` 3-튜플을 준다. 이 확장은 M8 담당이다.

`"h"` 외의 조건에서 bank 차원과 enqueue 텐서가 어긋나면 **런타임 shape 오류로 죽는다**(조용히 틀리지는 않는다). 그래도 4B 격자를 돌리다 중간에 죽는 건 비용이므로, `OnlineTTARunner.__init__`에서 세 값의 정합을 미리 확인하고 어긋나면 즉시 `ValueError`를 던진다.

`passes > 1`이면 위 루프를 스트림 전체에 대해 `passes`회 반복하고, 종료 후 **adapt-then-eval**로 재평가한다.

**반환**
```python
{"online_acc": float,          # 적응 전 예측 누적 (single-pass에서만 의미)
 "final_acc": float,           # 스트림 종료 후 재평가
 "acc_vs_time": list[float],
 "gain_trace": list[float],    # ‖gain − 1‖ 궤적
 "loss_trace": list[dict],
 "gate_skipped": int,
 "n_seen": int}
```

**불변식**
- `config`만 바꿔 실험 **4A·4B·5·6·7을 전부** 돌릴 수 있어야 한다. 실험별 코드 분기 금지.
- `passes == 1`이면 각 샘플을 정확히 한 번만 본다(순수 online TTA).
- 시드 고정 시 `gain_trace`가 재현된다.

---

## 구현 순서와 검증 게이트

| 순서 | task | 이것만으로 확인 가능한 것 |
|---|---|---|
| 1 | M1 | FVU·L0가 `SAE_validation` 수치와 일치 |
| 2 | M2 | `gain=1` 비트 동일 · `gain.grad≠0` · frozen `.grad is None` · 3 basis 교체 |
| 3 | M4 | weak/strong 쌍 생성 (M1·M2와 독립이라 병렬 가능) |
| 4 | M3 | 손실 항별 유한값 · same-pseudo 마스킹 · `buffer=0` 폴백 |
| 5 | M8 | 1 corruption smoke에서 no-adapt 대비 개선 |
| 6 | M6 → M5 | `c_k` 산출 후 anchor의 `ck` 모드 활성화 |
| 7 | M7 | 실험 1의 임계 확정 후 게이트 결선 |

M5는 `off`/`l2` 모드만 먼저 만들면 M6 없이도 M8에 붙일 수 있다. `ck` 모드만 M6에 의존한다.
