# Contract Brief — Phase 2 (Main, baseline·실험 5·6)

> [Plans.md](../Plans.md) T2.1~T2.3. 공통 규약은 [README.md](README.md).
> 논문의 **주 표**가 여기서 나온다. T2.2가 crossover, T2.3이 TTA 정체성 방어.

---

## T2.1 — baseline 스위트

**목적**: 실험 5가 요구하는 9종을 **동일 백본(ViT-B/16)·동일 스트림·동일 평가 프로토콜**로 재현한다. 공개 수치 인용이 아니라 우리가 돌린 결과라야 비교가 성립한다.

### 공통 인터페이스

```python
# Model/baselines/__init__.py
class TTABaseline(Protocol):
    """9종 baseline과 우리 방법이 같은 스트림 루프에 꽂히도록 맞추는 공통 규격.
    구현체마다 __init__ 인자는 다르지만(Tent는 lr, TACT는 augmentation 수 n·제거 성분 수 m,
    CoTTA는 EMA 계수 등) 아래 세 메서드는 반드시 같은 의미를 가져야 한다."""

    name: str    # 결과표에 찍히는 이름

    def reset(self) -> None:
        """적응 상태를 초기값으로 되돌린다. 도메인/시드가 바뀔 때 부른다."""

    def adapt_and_predict(self, x: Tensor, *, x_strong: Tensor | None = None) -> Tensor:
        """배치 1개를 받아 (a) 적응 전 예측 logits를 반환하고 (b) 내부 상태를 갱신한다.
        반환 logits는 반드시 '이 배치의 갱신이 반영되기 전' 값이어야 한다 — Tent 이래 표준.
        x_strong은 증강 쌍을 쓰는 방법(AdaContrast·우리)만 사용하고 나머지는 무시한다."""

    def trainable_parameter_count(self) -> int:
        """test time에 갱신되는 파라미터 수. 메모리 비교표의 근거이자,
        TACT처럼 backprop-free인 방법이 0을 반환하는지 확인하는 sanity check."""
```

`OnlineTTARunner`(M8)와 동일한 스트림 루프에 꽂힌다. 우리 방법도 이 Protocol을 구현해 **같은 표에 같은 방식으로** 올린다.

### 9종 현황과 작업량

| baseline | 현재 보유 | 필요 작업 |
|---|---|---|
| No-adapt | — | 자명 (forward만) |
| **Tent** | `reservoir_sae/tta.py`의 `configure_bn_only_tent` | **BN 전용이라 ViT에 못 씀.** LN affine(`norm1`·`norm2`의 weight/bias) 수집으로 재작성 |
| EATA | 없음 | entropy 필터 + Fisher 정규화. source Fisher 사전계산 필요 |
| SAR | 없음 | SAM optimizer + entropy 필터 + 모델 리셋 |
| DeYO | 없음 | PLPD(패치 셔플 후 예측 변화) 샘플 필터 + LN affine 적응 |
| TACT | 없음 | **backprop-free.** augmentation `n`개 → per-sample PCA → 상위 `m`(≤16) 성분 하드 제거 → trimmed prototype 유사도 분류. 학습 파라미터 0 |
| CoTTA | 없음 | teacher-student EMA + augmentation averaging + stochastic restore. 9종 중 가장 무겁다 |
| AdaContrast | 없음 | **M3+M8을 그대로 쓰되 backbone 전체를 학습**하는 변형. 우리와의 차이가 "무엇을 적응시키나"만 남는다 |
| ReservoirTTA | `third_party/ReservoirTTA` | 어댑터로 감싸 Protocol에 맞춤 |

**TACT 주의**: 학습 파라미터가 없고 개입 지점이 prototype 분류용 최종 표현이다. `trainable_parameter_count() == 0`이 정상이며, `adapt_and_predict`가 gradient를 쓰지 않는다. 실험 4A의 basis arm으로 넣으려 했던 것이 애초에 범주 오류였던 이유다.

### 검증

각 baseline이 ImageNet-C 1 corruption(severity 3) smoke에서
1. no-adapt 대비 정확도가 **떨어지지 않고**(Tent가 collapse하는 조건은 예외로 기록)
2. 공개 보고치와 **자릿수가 맞는지**(±5%p 이내면 통과, 벗어나면 재현 실패로 기록)

### 산출물

```json
{"result": {"baselines": {
  "tent": {"impl": "reimplemented-ln-affine", "params": 73728,
           "smoke_acc": 0.55, "reference_acc": 0.57, "within_tolerance": true},
  "tact": {"impl": "reimplemented", "params": 0, "smoke_acc": 0.52, ...}}}}
```

**판정 규칙**: `pass` ⟺ 9종 전부 Protocol 준수 + smoke 통과. 재현이 공개치와 크게 어긋나는 baseline은 `within_tolerance: false`로 기록하고 **논문에 그 사실을 각주로 남긴다**(조용히 빼지 않는다).

---

## T2.2 — 실험 5: 메인 crossover

**진입점**: `experiments/exp5_online_crossover.py`

### 동작

regime × method 격자를 동일 스트림에서 돌린다.

| regime | 데이터 | 주 지표 | 기대 |
|---|---|---|---|
| spurious | Waterbirds, ColoredMNIST | worst-group acc | **SAE 크게 우세** |
| natural | ImageNet-R/Sketch/A | acc / rel-acc | SAE 우세 |
| corruption | ImageNet-C 15×5 | acc | **SAE ≈ baseline (정직)** |
| wild/continual | mixed·recurring IN-C | acc-vs-time | SAE 우세 |

**HP 선택 규약**: 모든 HP(우리 것·baseline 것 포함)를 **ImageNet-C 4 extra hold-out**(`gaussian_blur`·`saturate`·`spatter`·`speckle_noise`)에서 고르고, **보고는 15 test에서** 한다. target 라벨로 튜닝했다는 반론을 막는 유일한 장치다. 이 분리를 코드로 강제한다(`build_dataset(..., split="hp"|"report")`).

**민감도 그래프 필수**: `lam`·lr을 넓은 범위로 스윕한 acc 곡선. TTA reviewer의 1순위 공격이 하이퍼파라미터이고, AdaContrast·T3A는 "hyperparameter insensitivity"를 장점으로 판다.

### 산출물

```json
{"result": {
  "grid": [{"regime": "spurious", "dataset": "waterbirds", "method": "ours",
            "worst_group_acc": 0.71, "mean_acc": 0.85, "std": 0.012}],
  "crossover": {"corruption": {"ours": 0.612, "tent": 0.608, "tact": 0.615,
                               "gap_vs_best_baseline": -0.003},
                "spurious":   {"ours": 0.71, "best_baseline": 0.62,
                               "gap_vs_best_baseline": 0.09}},
  "sensitivity": {"lam": [[0.0, 0.68], [0.1, 0.71], [1.0, 0.70], [10.0, 0.63]],
                  "lr":  [[1e-4, 0.66], [1e-3, 0.71], [1e-2, 0.69]]},
  "hp_selected_on": "imagenet-c-4extra", "reported_on": "imagenet-c-15test"}}
```

### 판정 규칙

- **`pass` = crossover가 보인다**: corruption에서 baseline과 비등(±1%p 이내면 "비등"으로 보고), spurious·wild에서 유의하게 우세.
- **corruption에서 이기려 하지 말 것.** [plan.md](../plan.md) §6.1의 전략이 "정확도 정면승부 회피"다. corruption에서 이기면 좋지만 그게 논거가 아니고, 오히려 "easy에선 비등"이 정직성의 근거로 쓰인다.
- **`fail`**: spurious regime에서도 우세하지 않으면 논문의 주 논거가 없다.
- 민감도 곡선이 넓은 범위에서 평평하지 않으면 `inconclusive` — 튜닝 운이었을 가능성을 배제 못 한다.

**비용**: 9 baseline + ours × 4 regime × (ImageNet-C는 15×5 셀). **이 Phase 전체에서 가장 비싸다.** severity 서브샘플링·셀당 이미지 상한을 config로 두고 실제 사용값을 `result.json`에 기록한다(무엇을 줄였는지 숨기지 않는다).

---

## T2.3 — 실험 6: 프로토콜·메모리 ablation (TTA 정체성 방어)

**목적**: default가 AdaContrast 원본(memory bank·multi-epoch 가능)이라 "이거 사실 SFDA/UDA 아니냐"는 공격을 받는다. **default를 바꾸지 않고** 축을 좁혀 순수 online TTA로도 성립함을 보인다.

**진입점**: `experiments/exp6_protocol_ablation.py`

### 동작

`TTAConfig`의 두 축만 스윕한다. 개입·손실은 실험 5와 완전히 동일하다.

```
buffer ∈ {0, 256, 2048, full(16384)}   ×   passes ∈ {1, 2, 5}
```

- `buffer=0` → in-batch 전용. negative는 배치 내에서, pseudo-label은 **weak-aug 자기예측 폴백**(M3 계약). 가장 순수한 online TTA.
- `buffer` ↑ → AdaContrast 원본에 수렴.
- `passes=1` → single-pass online. `passes>1` → multi-epoch(SFDA 영역).

**평가 프로토콜을 조건마다 정직하게 병기한다**: `passes=1`은 online 누적 acc, `passes>1`은 adapt-then-eval. 두 수를 한 표에 섞지 않는다.

### 산출물

```json
{"result": {
  "grid": [{"buffer": 0, "passes": 1, "protocol": "online-cumulative",
            "acc": 0.66, "worst_group_acc": 0.68, "std": 0.014},
           {"buffer": 16384, "passes": 5, "protocol": "adapt-then-eval",
            "acc": 0.73, "worst_group_acc": 0.74, "std": 0.009}],
  "pure_tta": {"buffer": 0, "passes": 1, "worst_group_acc": 0.68},
  "sfda_ceiling": {"buffer": "full", "passes": 5, "worst_group_acc": 0.74},
  "protocol_gap": 0.06,
  "baseline_margin_at_pure_tta": 0.05}}
```

### 판정 규칙

- **`pass` (정체성 방어 성공)**: `buffer=0 · passes=1`에서도 **baseline 대비 우위가 유지**된다. 이게 핵심이다 — bank 없이 순수 TTA로 성립함을 보이면 "SFDA 아니냐"가 닫힌다.
- **`protocol_gap`이 작으면** 우리 이득이 SFDA 예산이 아니라 **basis(SAE)** 에서 온다는 증거다. 크면 반대로 예산 덕이었다는 뜻이므로 논문의 주장을 그만큼 약화시켜 적는다.
- biased regime에서 in-batch 이득이 corruption보다 커야 한다(spurious 억제가 배치 내에서도 작동한다는 뜻).
- `buffer` 증가에 따라 완만히 개선 후 포화하는 곡선이 정상. 단조 감소면 memory bank 구현 버그를 의심한다.
