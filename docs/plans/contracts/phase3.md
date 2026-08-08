# Contract Brief — Phase 3 (Extend, 실험 7·8)

> [Plans.md](../Plans.md) T3.1~T3.2. 공통 규약은 [README.md](README.md).
> T3.1은 안정성 fallback(Recommended), T3.2는 사실상 **Paper B**(Optional).

---

## T3.1 — 실험 7: clustering fallback (개별 gain vs 개념 클러스터 gain)

**목적**: `K=12288` 개별 gain은 라벨 없는 TTA에서 노이즈·과적합 위험이 있다. 개념 클러스터 단위 group gain이 파라미터를 수십~수백으로 줄이면서 안정성·worst-group을 지키는지 검증한다.

**진입점**: `experiments/exp7_cluster_gain.py`

### 새 basis 구현

```python
# Model/gain_basis.py 에 추가
class ClusterGainBasis:
    """latent K개를 M개 클러스터로 묶어 group gain을 공유한다.
    GainBasis Protocol을 그대로 만족하므로 M8 루프는 수정 없이 재사용된다."""

    def __init__(self, sae: FrozenSAE, assignment: Tensor, num_clusters: int):
        """어느 latent가 어느 클러스터에 속하는지를 확정해 들고 있는다.
        - assignment: [K] int, 각 latent의 소속 클러스터 (-1 = 미할당 → gain 고정 1)
        - num_clusters(M): 학습할 gain의 길이. K=12288 대신 M개만 학습한다
        - assignment는 buffer로 보관(학습 대상 아님). gain_dim = M"""
        self.gain_dim = num_clusters

    def encode(self, h):
        """LatentGainBasis와 완전히 동일 — 코드는 여전히 latent 단위 [N,K]다.
        묶는 것은 gain이지 코드가 아니다."""
        return self.sae.encode(h)

    def delta(self, code, gain):
        """M개짜리 gain을 assignment로 K개 자리에 펼친 뒤 LatentGainBasis와 똑같이 넘긴다.
        딕셔너리가 SAE 것 그대로이므로 decode_delta를 재사용한다."""
        return self.sae.decode_delta((self._expand(gain) - 1.0) * code)

    def absolute(self, code, gain):
        """residual=False 경로. 딕셔너리가 SAE 것이므로 decode_raw를 재사용한다."""
        return self.sae.decode_raw(self._expand(gain) * code)

    def _expand(self, gain):
        """[M] 클러스터 gain -> [K] latent gain. 미할당(-1)은 1로 고정한다."""
        expanded = gain[self.assignment]
        return torch.where(self.assignment >= 0, expanded, torch.ones_like(expanded))
```

**핵심**: `gain`은 `[M]`이지만 `delta` 내부에서 `assignment`로 `[K]`에 전개된다. 학습 파라미터는 여전히 `gain` 하나이고 전역 불변식이 유지된다.

### 군집 방법 3종

| 방법 | 절차 | 근거 |
|---|---|---|
| `decoder-cos` | `W_dec` 행(unit-norm) 코사인 유사도로 k-means 또는 agglomerative | 같은 방향을 쓰는 원자는 같은 개념일 가능성 |
| `broden` | T1.3의 부산물 `full_latent_iou.json`(alive latent 전체의 최고 IoU 개념/category)으로 그룹화 | 의미 기반. 개념 라벨이 없는 latent는 `-1` |
| `co-activation` | 토큰 단위 공발화 행렬(`z>thr`의 상관)로 군집 | 함께 켜지는 것끼리 |

3종을 비교하고 앙상블(다수결 할당)도 한 조건으로 넣는다.

### 동작

```
for method in ["decoder-cos", "broden", "co-activation"]:
    assignment = cluster(method, M)
    for M in [32, 128, 512, K]:              # K = 개별 gain (baseline 조건)
        for seed in [0, 1, 2]:
            run with ClusterGainBasis(sae, assignment, M)
```

`M=K`는 `LatentGainBasis`와 동치여야 한다 — **이걸 sanity check로 assert한다**(할당이 항등일 때 두 basis의 출력이 일치).

### 산출물

```json
{"result": {
  "grid": [{"method": "decoder-cos", "M": 128, "worst_group_acc": 0.73,
            "std_over_seeds": 0.006, "params": 128, "steps_to_converge": 140}],
  "individual_baseline": {"M": 12288, "worst_group_acc": 0.72, "std_over_seeds": 0.021},
  "best": {"method": "decoder-cos", "M": 128},
  "stable_range": [64, 256],
  "sanity_M_equals_K_matches_latent_basis": true}}
```

### 판정 규칙

- **`pass`**: 중간 `M`에서 **시드 간 분산이 개별 gain보다 작으면서** worst-group acc가 유지 또는 향상. 파라미터는 1~2 자릿수 감소.
- `M`이 너무 작으면 표현력 부족으로 하락, 너무 크면 개별과 수렴 → **적정 `M`의 안정 구간**이 존재해야 한다. 단조 곡선이면 클러스터링이 의미를 못 만든 것.
- 개별 gain이 불안정한 regime(biased)에서 이득이 가장 커야 한다.
- `sanity_M_equals_K_matches_latent_basis: false`면 **구현 버그**이므로 즉시 정지.

---

## T3.2 — 실험 8: continual / recurring (Paper B)

**목적**: 스트림이 장기화·재발하면 단일 gain은 도메인 간 간섭·forgetting을 겪는다. **SAE descriptor 라우팅 + 도메인당 gain 저장/재로드**가 ReservoirTTA 대비 메모리·해석성에서 이득인지, 그 이득이 스텝 연산으로 상쇄되지 않는지 정직하게 잰다.

**진입점**: `experiments/exp8_continual.py`

### 재사용 (재작성 금지)

[reservoir_sae/simple_reservoir.py](../../../reservoir_sae/simple_reservoir.py)의 `PrototypeReservoir`가 이미 라우터다:

```python
PrototypeReservoir(descriptor_dim, max_models, threshold,
                   prototype_momentum=0.2, distance_metric="l2"|"cosine", device)
    .route(descriptor) -> RoutingResult(model_idx, ..., is_new, ..., num_models,
                                        parent_model_idx, model_probs)
    .initialize(descriptor)
    .num_models
```

우리가 새로 만드는 건 **descriptor 정의**와 **gain 저장소**뿐이다.

### 새 구성요소

```python
# Model/domain_router.py
def sae_descriptor(code: Tensor, *, threshold: float, normalize: bool = True) -> Tensor:
    """배치의 SAE 코드를 도메인 지문으로 요약한다.
    기본: 발화율 벡터 mean(code > threshold, dim=0) -> [K], L2 정규화.
    ReservoirTTA의 StyleVec(외부·불투명)을 내부·결정 정렬된 것으로 대체하는 지점."""

class GainStore:
    """도메인당 gain을 sparse하게 저장/재로드한다.
    1에서 벗어난 성분만 (index, value)로 보관 — 이게 경량성 주장의 실체다."""

    def __init__(self, max_domains: int):
        """도메인 인덱스 -> (indices, values) 딕셔너리 하나를 들고 있는 게 전부다.
        모델 사본을 저장하는 ReservoirTTA와 대비되는 지점이라 일부러 가볍게 유지한다."""

    def save(self, domain_idx: int, gain: Tensor, *, eps: float = 1e-3) -> int:
        """이 도메인에서 학습된 gain 중 1에서 eps 넘게 벗어난 성분만 골라 저장한다.
        반환: 저장된 non-trivial 성분 수 — 이 숫자가 곧 경량성 주장의 근거다."""

    def load(self, domain_idx: int, gain_dim: int) -> Tensor:
        """저장된 성분을 1로 채운 벡터 위에 되살려 [gain_dim] gain을 복원한다.
        처음 보는 도메인이면 전부 1인 벡터(=무개입 시작점)를 준다."""

    def total_bytes(self) -> int:
        """전 도메인 저장량 합계. ReservoirTTA와의 메모리 대조표에 그대로 들어간다."""
```

### 동작

1. **recurring 스트림 구성**: 도메인 A→B→C→A→B→C… 순열. ImageNet-C precomputed에서 **순서만 재배열**(재생성 금지 — spec의 ImageNet-C 규약).
2. 배치마다 `descriptor = sae_descriptor(code)` → `reservoir.route(descriptor)`
3. `is_new`면 새 gain(init 1)에서 시작, 기존 버킷이면 `GainStore.load`로 재로드
4. 배치 적응 후 `GainStore.save`
5. 혼합 배치는 샘플 단위 클러스터링(FIND식), 완만한 drift는 버킷 프로토타입이 EMA로 따라가게(FreDA식)
6. **ReservoirTTA와 대조**: 동일 스트림에서 `third_party/ReservoirTTA` 실행

**도메인 감지 임계**: source descriptor 간 거리 분포의 **상위 분위수**로 정한다(ReservoirTTA식). target 라벨을 쓰지 않는다.

### 측정

| 지표 | 정의 |
|---|---|
| forgetting Δ | 도메인 A 첫 방문 정확도 − 재방문 정확도. **음수(회복)면 좋다** |
| acc-vs-time | 배치 인덱스별 누적 정확도 |
| 도메인당 메모리 | `GainStore`의 non-trivial 성분 수 × 4바이트 |
| 스텝 FLOPs | 백본 + **SAE encode/decode 포함** |
| routing 정확도 | 진짜 도메인 라벨 대비 버킷 할당 일치율(사후 분석용, 적응엔 미사용) |

### 산출물

```json
{"result": {
  "forgetting_delta": {"ours": -0.004, "reservoir_tta": 0.021, "single_gain": 0.068},
  "memory_per_domain_bytes": {"ours": 18432, "reservoir_tta": 294912},
  "memory_ratio": 16.0,
  "nontrivial_gain_components": {"mean": 4608, "min": 3102, "max": 5940},
  "step_flops": {"ours": 1.94e10, "reservoir_tta": 1.75e10, "ratio": 1.11},
  "routing_accuracy": 0.87,
  "num_buckets_created": 5, "true_num_domains": 5}}
```

### 판정 규칙

- **`pass`**: 도메인당 저장이 ReservoirTTA보다 유의하게 가볍고, 재발 도메인의 forgetting Δ가 작다.
- **스텝 FLOPs는 SAE forward 때문에 꼭 싸지지 않는다.** 이건 숨기지 말고 **메모리·해석성 이득 vs 연산 비용의 trade-off 표**로 정직하게 낸다([plan.md](../plan.md) §2.7이 이미 ⚠️로 인정한 지점).
- **⚠️ 경량성 주장의 실측 리스크**: 현 ckpt는 패치당 `L0 ≈ 497`, 이미지당 발화 support가 4,000개를 넘는다. 이러면 "도메인당 수십~수백 스칼라"라는 plan.md의 셀링 포인트가 **성립하지 않는다**(ReservoirTTA의 LN affine ~74k와 같은 자릿수가 된다).
  `nontrivial_gain_components`가 수천이면 `memory_ratio`가 1에 가까워지므로 **경량성 주장을 철회하거나**, `L0` 수십 급 계기로 올린 뒤 재측정해야 한다. 이 숫자를 반드시 먼저 확인하고 논문 문구를 정한다.
- `num_buckets_created`가 `true_num_domains`와 크게 다르면 임계 설정 실패 → 라우팅 결론을 내지 않는다(`inconclusive`).
