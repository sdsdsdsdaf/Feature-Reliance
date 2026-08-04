# Contract Brief — Phase 1 (MVP, 실험 1~4B)

> [Plans.md](../Plans.md) T1.1~T1.4b. 공통 규약·산출물 헤더는 [README.md](README.md).
> 이 Phase는 **저비용 falsify 게이트**다. 각 실험은 실패 시 상위 thesis를 반증하도록 설계돼 있고, 판정 규칙에 그 조건을 명시한다.

---

## T1.1 — 실험 1: SAE 계기 신뢰성 + OOD FVU 게이트

**진입점**: `experiments/exp1_sae_instrument.py`

```bash
python -m experiments.exp1_sae_instrument \
    --sae outputs/reservoir_sae/vit_b_sae.pt \
    --corruptions all --severities 1,2,3,4,5 \
    --max-images-per-cell 512 \
    --out outputs/experiments/T1.1
```

### 동작

1. **in-domain 기준선**: ImageNet-1k val에서 `FrozenSAE.fvu(h)`·`l0(z)`를 배치 누적. `M1`의 정의(`mse/var`, 전 원소 기준)를 그대로 쓴다 — 실험 1이 정한 임계를 M7이 런타임에 쓰므로 척도가 갈리면 게이트가 무의미해진다.
2. **OOD**: ImageNet-C 15 test × 5 severity 각 셀에서 동일 측정. 셀당 `max_images_per_cell`로 제한.
3. **게이트 임계**: source FVU 분포의 상위 분위수(기본 `p95`)를 `threshold`로 확정. M7이 이 값을 주입받는다.
4. **재구성-정확도 상관**: 각 셀에서 (a) 원본 forward 정확도, (b) `reconstruct(h)`로 치환한 forward 정확도를 재고 `FVU vs Δacc` 상관계수 산출.
5. **SAE-source ablation**: 동일 측정을 `source(ImageNet-1k)` / `unlabeled proxy(imagenette)` / `공개 pretrained SAE` 3조건에서 반복. proxy·public은 별도 ckpt를 `--sae`로 받는다.

### 산출물

```json
{"result": {
  "in_domain": {"fvu": 0.0004, "l0": 497.0, "cosine": 0.9998, "acc": 0.79},

  "ood": [{"corruption": "gaussian_noise", "severity": 5,
           "fvu": 0.031, "l0": 388.0, "acc_orig": 0.21, "acc_recon": 0.19,
           "recon_cost": -0.02}],

  "ood_summary": {
    "scope": "15-test",
    "n_cells": 75,
    "mean_acc_orig": 0.382, "mean_acc_recon": 0.361, "mean_recon_cost": -0.021,
    "mean_fvu": 0.012, "mean_l0": 421.0,
    "by_severity": [
      {"severity": 1, "acc_orig": 0.58, "acc_recon": 0.57, "recon_cost": -0.01,
       "fvu": 0.003, "l0": 470.0}
    ],
    "by_corruption": [
      {"corruption": "gaussian_noise", "acc_orig": 0.24, "acc_recon": 0.22,
       "recon_cost": -0.02, "fvu": 0.021, "l0": 402.0}
    ]
  },

  "gate_threshold": 0.0012,
  "gate_percentile": 95,
  "cells_above_gate": 18,
  "fvu_vs_dacc_pearson": -0.71,
  "source_ablation": {"source": {...}, "proxy": {...}, "public": {...}}}}
```

**`ood_summary` 집계 규약**

- `scope: "15-test"` — **15 test corruption에서만 집계한다.** HP holdout인 4 extra(`gaussian_blur`·`saturate`·`spatter`·`speckle_noise`)는 평균에 섞지 않는다. 두 집합을 다 보고할 때는 `scope`를 바꿔 항목을 하나 더 만든다.
- `n_cells` = 15 × 5 = 75. 셀별 이미지 수가 같아야 단순 평균이 유효하므로 `max_images_per_cell`을 전 셀 동일하게 고정한다.
- `recon_cost = acc_recon − acc_orig` (보통 음수). **SAE 왕복이 정확도로 치르는 대가**이고, `mean_recon_cost`가 계기 충실도의 한 줄 요약이다.
- `by_severity`는 corruption 15개 평균, `by_corruption`은 severity 5개 평균. ImageNet-C 관례대로 둘 다 낸다.
- `in_domain.acc`도 같이 둔다 — OOD 평균이 얼마나 떨어졌는지 볼 기준선이 없으면 숫자 해석이 안 된다.
- `cells_above_gate` = FVU가 `gate_threshold`를 넘은 셀 수. **0이면 게이트가 한 번도 안 걸린다는 뜻이고 곧 `fail` 조건이다.**

`figures/fvu_vs_severity.png` — corruption별 곡선.
`figures/acc_vs_severity.png` — `acc_orig`/`acc_recon` 두 곡선을 겹쳐 그려 대가가 severity에 따라 벌어지는지 본다.

### ✅ 확정 결과 (2026-08-05, 정식 실행) — `verdict = fail`

**15 test corruption × 5 severity = 75셀, 셀당 512장, in-domain 2,048장으로 실행 완료.** 예비 probe의 방향이 전체 규모에서 그대로 확인됐다.

| | |
|---|---|
| in-domain (ImageNet-1k val, n=2048) | FVU **1.526e-03**, L0 786.2, cosine 0.9996, acc 0.858 |
| 게이트 임계 (in-domain 이미지단위 FVU의 p95) | **2.960e-03** |
| **`cells_above_gate`** | **0 / 75** |
| severity 단조 **증가** corruption | **0 / 15** |
| `mean_recon_cost` | **+0.0000** (SAE 왕복이 정확도로 치르는 대가가 없다) |
| `fvu_vs_dacc_pearson` | 0.158 (약하고, 기대와 부호도 어긋난다) |

**severity가 오를수록 정확도는 무너지는데 FVU는 좋아진다** — 이 표가 결론이다.

| severity | FVU | L0 | acc |
|---|---|---|---|
| 1 | 1.650e-03 | 773.1 | 0.855 |
| 2 | 1.569e-03 | 758.4 | 0.823 |
| 3 | 1.504e-03 | 752.5 | 0.798 |
| 4 | 1.395e-03 | 740.5 | 0.747 |
| 5 | **1.263e-03** | 731.9 | **0.647** |

정확도가 21%p 붕괴하는 동안 FVU는 오히려 24% 낮아진다. **FVU는 shift를 전혀 추적하지 못한다.**

**결론**: FVU는 이 조합(ViT-B/16 + `L0≈497~800` PatchSAE)에서 shift 감지기가 아니라 **입력 복잡도 측정기**다. 손상된 이미지는 고주파가 뭉개져 딕셔너리가 맞추기 더 쉬워지고, L0도 함께 떨어진다(786 → 732).

**이건 SAE가 나쁘다는 뜻이 아니다.** cosine이 0.9996이고 `recon_cost`가 0이다 — 계기로서의 충실도는 매우 좋다. 무너진 건 **VS2식 FVU 게이트라는 설계**이고, 그게 ViT+PatchSAE 계열에 그대로 이식되지 않는다는 것 자체가 보고할 결과다.

**→ [M7 FVU gate를 Optional로 강등한다.](../Plans.md)** `L0` 수십 급 계기로 올린 뒤 재측정할 여지는 남긴다.

**미측정**: `source_ablation`의 `proxy`·`public`은 `null`이다(해당 ckpt가 없고 학습·수급이 T1.1 범위 밖). `unknown_fields`에 사유가 기록돼 있다. **strict-no-source 방어는 아직 미완**이므로 별도 task가 필요하다.

산출물: `outputs/experiments/T1.1/{result.json,result.md,figures/}`, 진입점 [experiments/exp1_sae_instrument.py](../../../experiments/exp1_sae_instrument.py).

### 예비 probe (2026-08-05, 위 결과로 대체됨)

정식 T1.1 전에 [scripts/probe_fvu_shift.py](../../../scripts/probe_fvu_shift.py)로 방향만 싸게 확인했던 것(셀당 128장, 3 corruption × sev {1,3,5}). 결과는 `outputs/experiments/T1.1/probe_fvu_shift.json`. 전체 실행이 이를 확증했으므로 참고용으로만 남긴다.

| condition | FVU | in-domain 대비 | L0 |
|---|---|---|---|
| **in-domain** (ImageNet-1k val) | 1.626e-03 | 1.00× | 793.3 |
| gaussian_noise sev1 / 3 / 5 | 9.19 / 8.89 / 8.57 e-04 | 0.57 / 0.55 / **0.53×** | 666 / 650 / 671 |
| fog sev1 / 3 / 5 | 8.90 / 8.58 / 7.92 e-04 | 0.55 / 0.53 / **0.49×** | 736 / 794 / 802 |
| glass_blur sev1 / 3 / 5 | 9.21 / 8.72 / 7.66 e-04 | 0.57 / 0.54 / **0.47×** | 684 / 663 / 665 |

**FVU가 severity에 따라 올라가지 않고 내려간다.** 9개 셀 전부 in-domain보다 낮고(0.47~0.57×), 3개 corruption 계열 **전부**에서 severity에 대해 단조 **감소**한다. `n_cells_above_in_domain_fvu = 0`, `n_corruptions_monotonically_increasing = 0/3`.

**해석**: 손상된 이미지는 고주파 디테일이 뭉개져 활성이 더 평범해지고, 딕셔너리가 맞추기 **더 쉬운** 입력이 된다(L0도 793 → 650~670으로 함께 떨어진다). 즉 FVU는 여기서 shift 감지기가 아니라 **입력 복잡도 측정기**로 작동한다.

**주의**
- 이건 T1.1이 **아니다.** 셀당 128장·3 corruption뿐이라 판정 근거로 쓰지 않는다. 다만 방향이 9/9 셀에서 예외 없고 효과가 2배라 정식 측정에서 뒤집힐 가능성은 낮다.
- **in-domain 기준선이 데이터에 따라 크게 흔들린다** — ImageNet-1k val에서 `1.6e-3`인데 imagenette(10 클래스)에서는 `3.4e-4`로 **약 5배** 차이다(M1 산출물). 정식 T1.1은 게이트 임계를 **어느 분포에서** 뽑는지를 반드시 고정해 기록해야 한다.
- `L0 ≈ 497~800`짜리 무딘 계기 탓일 수 있다. PatchSAE급(L0 수십)으로 올리면 달라질 여지는 남아 있다.

### 판정 규칙

- `pass`: FVU가 severity에 대해 **단조 증가**하고(corruption의 ≥80%에서), 게이트 임계 밖 셀이 존재한다.
- **`fail`: 전 OOD 셀의 FVU가 임계 이하** → 게이트가 한 번도 안 걸린다는 뜻이고, M7이 무의미해진다.
  **위 probe에서 이 실패 모드가 실제로 관측됐다**(임계 초과 셀 0개, 게다가 방향이 반대). 정식 측정에서 확정되면 **M7을 Optional로 강등**하고 그 사실을 spec에 반영한다.
- **`fail`은 "SAE가 나쁘다"가 아니라 "게이트라는 설계가 이 백본·SAE 조합에서 성립하지 않는다"는 뜻이다.** 재구성 품질(FVU 1e-3, cosine 0.9998)은 오히려 매우 좋다. 부정 결과 그대로 논문에 싣는다 — VS2식 FVU 게이트가 ViT+PatchSAE 계열에 그대로 이식되지 않는다는 것 자체가 보고할 값어치가 있다.
- `proxy`/`public` 조건이 `source`와 같은 결론을 주면 "source 의존은 편의일 뿐 필수 아님"이 입증된다(strict-no-source 방어).

**비용**: forward-only. 15×5×512 ≈ 38k 이미지 + in-domain. 학습 없음.

---

## T1.2 — 실험 2: 개입 메커니즘 sanity + capacity 상한

**진입점**: `experiments/exp2_intervention_sanity.py` + `tests/test_intervention.py`

### 동작

**A. 항등성·gradient 경로 (테스트, 결정론적)**

```python
# 1. no-op 항등성 — residual on이면 근사가 아니라 비트 동일
logits_ref, feat_ref = backbone_plain(x)
logits_int, feat_int = intervention(x)            # gain == 1
assert torch.equal(logits_ref, logits_int)

# 2. gradient 경로
loss = intervention(x)[0].logsumexp(-1).sum(); loss.backward()
assert intervention.gain.grad.abs().sum() > 0
for p in backbone.parameters(): assert p.grad is None
for p in sae.parameters():      assert p.grad is None

# 3. optimizer가 gain만 바꾼다
before = {n: p.clone() for n, p in backbone.named_parameters()}
opt.step()
for n, p in backbone.named_parameters(): assert torch.equal(p, before[n])

# 4. gradient 희소성 — 발화 안 한 latent의 gain은 안 움직인다
dead = (code.sum(0) == 0)
assert intervention.gain.grad[dead].abs().max() == 0
```

3종 basis 전부에 대해 반복한다. `채널 gain`은 4번을 건너뛴다(dense code).

**B. capacity 상한 (라벨 supervised)**

`gain`만 학습하되 **target 라벨을 주고** 도달 가능한 정확도 상한을 잰다. 비교군은 `LN affine`(Tent 파라미터화).

```
for arm in [latent_gain, channel_gain, ln_affine]:
    reset(); opt = Adam(arm.params, lr=lr)
    for step in range(300):
        loss = cross_entropy(model(x), y_true)      # ← 라벨 사용 (오라클)
        opt.zero_grad(); loss.backward(); opt.step()
    ceiling[arm] = eval_acc()
```

lr은 arm별 `{1e-3, 3e-3, 1e-2}` 스윕 후 best. 데이터는 ImageNet-C 약(severity 1)·강(severity 5) 각 1종.

### 산출물

```json
{"result": {
  "asserts": {"no_op_bitwise": true, "gain_grad_nonzero": true,
              "frozen_grad_none": true, "grad_sparsity_respected": true},
  "capacity": {"latent_gain": {"lr": 3e-3, "ceiling": 0.58},
               "channel_gain": {"lr": 1e-3, "ceiling": 0.61},
               "ln_affine":    {"lr": 1e-3, "ceiling": 0.64}},
  "gap_vs_ln_affine": {"latent_gain": -0.06, "channel_gain": -0.03}}}
```

### 판정 규칙

- `fail`: assert 4종 중 하나라도 실패 → **M2 구현 버그.** 이후 실험 전부 무효이므로 즉시 정지.
- capacity 격차는 **정직하게 보고할 대가**이지 새 모듈로 메울 대상이 아니다(v1 = gain 단독). 격차가 크면 FB-cluster·손실 설계로 대응할지 판단하는 입력이 된다.
- **주의**: 이건 오라클 상한이지 성능 예측이 아니다. 실제 thesis는 "라벨 없을 때 어느 basis가 옳은 방향을 찾나"이고, 그건 T1.4a가 잰다. 여기 격차만 보고 결론 내지 말 것.

---

## T1.3 — 실험 3: spurious/causal 분리 검증 (Waterbirds)

**thesis의 falsifiable 핵심.** SAE latent 좌표계가 **spurious 개념과 causal 개념을 분리**하는가.

**진입점**: `experiments/exp3_disentanglement.py`

### 측정의 3층 구조 — 이 순서가 뒤집히면 순환논증이다

"`s_k` 높고 `c_k` 낮은 걸 뽑아서 spurious라 부른다"는 것 자체로는 아무것도 증명하지 못한다. **무엇이 spurious인지를 정하는 독립적인 정답**이 먼저 있어야 한다.

| 층 | 무엇 | 쓰는 라벨 | 역할 |
|---|---|---|---|
| ① **정답** | `spur_j` / `caus_j` (층화 AUC) | Waterbirds `y`·`place` | 무엇이 spurious인지를 **정의**한다 |
| ② **진단** | `s_k` (발화율 shift) | 없음 | test time에 쓸 수 있는 대용물 |
| ② **진단** | `c_k` (ablation 정확도 하락) | source 라벨만 | 동상 |
| ③ **해석** | Broden IoU | Broden 개념 마스크 | 그 feature가 **어떤** 개념인지 이름 붙인다 |

실험이 검정하는 명제는 세 개다.

- **(A) 분리** — SAE 축에는 "배경만 보는 feature"와 "새만 보는 feature"가 실제로 따로 존재하는가. PCA·무작위 축은 그렇지 않은가. ← **주 결과**
- **(B) 진단 타당성** — 라벨 없는 `s_k`(+ `c_k`)가 ①의 정답을 복원하는가. 이게 서야 test time에 라벨 없이 spurious를 집어낼 수 있다는 주장이 성립한다.
- **(C) 해석성** — 분리된 feature에 사람이 읽는 개념 이름(`water`, `beak`)이 붙는가. ← **보조**

**왜 Waterbirds가 ①을 줄 수 있나**: 이 데이터셋은 새를 배경 위에 **합성해서** 만들었다. 그래서 `place`(land/water)가 spurious이고 `y`(landbird/waterbird)가 causal이라는 게 **설계상 확정**돼 있고, 두 라벨이 `metadata.csv`에 그대로 들어있다(`y, place, split`). 정답을 Broden으로 우회해서 알아낼 필요가 없다.

### ① 정답 — 층화 AUC

feature `j`의 이미지 단위 점수 `a_j(image)` = 패치 점수의 pooling(기본 `max`, robustness로 `mean` 병기).

```
spur_j = mean over y ∈ {0,1} of  2·|AUC(a_j → place | y 고정) − 0.5|
caus_j = mean over place ∈ {0,1} of  2·|AUC(a_j → y | place 고정) − 0.5|
```

둘 다 `[0,1]`. 1이면 그 축 하나로 완벽히 판별, 0이면 무정보.

**층화가 필수인 이유**: Waterbirds는 `y`와 `place`가 **95% 상관**이다. 층화하지 않고 그냥 `AUC(a_j → y)`를 재면 **순수 배경 탐지기도 0.95에 가까운 AUC**를 받아 causal처럼 보인다. `y`를 고정한 안에서 `place`를 판별하게 하면 남는 신호는 배경뿐이고, 반대도 마찬가지다. 이 층화가 confound를 완전히 제거한다.

**AUC 부호 처리**: `|AUC − 0.5|`이므로 feature가 물에서 켜지든 뭍에서 켜지든 동일하게 잡힌다. → **PCA의 `±e_j` 부호 문제가 사라진다.** 성분을 2배로 늘리는 처리가 필요 없다.

**평가 split**: Waterbirds `split==2`(test, 그룹이 대체로 균형). train(95% 편향)은 ②의 `c_k` 측정에만 쓴다.

### ① 임계 — permutation max-null (pool 크기 보정)

`n_features`가 축마다 다르므로(SAE ~6,000 vs PCA 768) 고정 임계를 쓰면 **후보를 많이 가진 축이 우연히 유리해진다.** 축마다 자기 pool 크기에 맞는 귀무 임계를 뽑는다.

```
for p in 1..P:                      # P = 200
    place' = y 층 안에서 place를 셔플     # 층화 구조는 보존
    null_max[p] = max over j of spur_j(place')
tau_spur = quantile(null_max, 0.95)
```

`caus`도 동일하게 `place` 층 안에서 `y`를 셔플해 `tau_caus`를 얻는다.
**max 통계를 쓰므로 pool이 클수록 `tau`가 자동으로 높아진다** — 앞서 문제였던 다중비교/pool 크기 비대칭이 여기서 해소된다. 축마다 임계를 따로 뽑는 것이 공정성의 기준이다.

효율: 각 feature의 점수 순위를 **한 번만** 계산해두면 permutation마다 순위합만 다시 더하면 된다(AUC = 정규화된 Mann-Whitney U).

### ① 축별 분리도 지표

```
informative_j : max(spur_j, caus_j) ≥ 해당 tau
sel_j = (spur_j − caus_j) / (spur_j + caus_j)     # +1 순수 배경, −1 순수 새, 0 entangled
purity_rate = |{j : informative_j and |sel_j| ≥ 0.6}| / |{j : informative_j}|
```

**축의 성적 = `purity_rate`와 `sel_j` 분포 모양.** ±1 근처 쌍봉이면 분리 성공, 0 근처 단봉이면 entangled.
**비율로 보고하므로 pool 크기가 분자·분모에서 상쇄된다** — 최댓값을 비교하던 이전 설계의 결함이 여기서 사라진다. 원 개수(`n_informative`, `n_pure_spurious`, `n_pure_causal`, `n_entangled`)도 같이 낸다.

### 축 추상화 — 세 축이 같은 파이프라인을 탄다

"feature"란 **패치 토큰마다 스칼라 하나를 주는 함수**다. 그것만 있으면 위 계산이 전부 동일하게 돈다.

```python
class ConceptAxis(Protocol):
    """block10 패치 토큰 -> feature별 스칼라 맵. 세 축이 이 규격만 맞추면 나머지는 공유."""
    n_features: int
    def scores(self, h: Tensor) -> Tensor:
        """[N,768] -> [N, n_features]. 값이 클수록 그 feature가 강하게 켜진 것.
        AUC는 순위 기반이라 축 간 스케일 차이를 신경 쓸 필요가 없다."""
    def ablate(self, h: Tensor, j: int) -> Tensor:
        """[N,768] -> [N,768]. feature j의 기여만 제거한 활성. c_j 측정에 쓴다.
        ②에서 SAE 축에만 호출된다 — 대조축(PCA/random)은 c_j가 필요 없다."""
```

**`ablate`는 SAE 축에서만 실제로 돈다.** 주 지표 (A)는 `scores`만으로 계산되고, `c_k`는 ②(SAE 진단의 타당성)에서만 쓰이므로 **PCA/random의 `ablate` 구현은 인터페이스 충족용**이다(사영 제거 1줄). 대조축에 대해 `c_j` 루프를 돌 필요가 없다 — 이전 설계 대비 큰 비용 절감.

| 축 | `n_features` | `scores(h)` | `ablate(h, j)` |
|---|---|---|---|
| **SAE** | 발화하는 latent 수 (실측 ~6,000) | `sae.encode(h)` — ReLU라 이미 `≥0` | `gain_j=0`인 residual 주입 = `h − decode_delta(z_j e_j)` |
| **PCA** | `d`(=768, 부호 처리 불필요) | `x = normalize(h)`; `p = (x−μ) @ E.T` | 그 방향 성분을 사영 제거: `h − ⟨x−μ, e_j⟩ e_j` (TACT의 trim 1개와 동일) |
| **Random** | `d` | 같은 식, `E`가 무작위 unit 벡터 | 동일 |

`scores`의 값 자체는 AUC에서 순위로만 쓰이므로 **percentile 이진화·마스크 밀도 정합이 주 경로에 필요 없다.** 이진화는 `s_k` 발화율 정의와 ③ Broden에서만 쓴다.

### ② 진단 타당성 — 라벨 없는 신호가 정답을 복원하는가

| 신호 | 정의 | 라벨 |
|---|---|---|
| `s_k` | `\|firing_k(Waterbirds 스트림) − firing_k(source=ImageNet)\|` | **없음** (도메인 단위) |
| `c_k^train` | Waterbirds train(95% 편향)에서 `ablate(h,k)` 후 정확도 하락 | source 라벨 |
| `c_k^bal` | 그룹 균형 부분집합에서 동일 측정 | source 라벨 |

- 발화 = `scores_j > threshold_j`, 임계는 `broden.py`의 `compute_thresholds_from_shards(..., percentile=99, ...)`.
- **`s_k`는 `place` 라벨을 쓰지 않는다.** 이전 정의(`|firing(place=1) − firing(place=0)|`)는 정답 라벨을 진단에 흘려 넣는 것이라 폐기한다.
- **검정**: `s_k`(및 `s_k·(1−c_k)`)를 랭커로 써서 ①의 순수-spurious 집합을 얼마나 잘 뽑는지 **average precision**으로 잰다. `corr(c_k, caus_j)`도 같이 낸다.
- **`c_k^train` vs `c_k^bal` 대비가 그 자체로 결과다.** 95% 상관 하에서 배경 latent를 끄면 train 정확도가 **떨어지므로** `c_k^train`은 배경을 causal로 오판한다. 그 오판 폭이 곧 [plan.md](../plan.md) §7이 인정한 "`c_k`는 도메인 불변 가정에 의존" 리스크의 실측값이다.

### ③ 해석성 — Broden IoU (보조)

**①에서 순수 spurious/causal로 판정된 SAE feature 상위 `N=40`개에 대해서만** 돌린다. Broden은 "분리되나"가 아니라 **"그게 무슨 개념인가"** 에 답하는 도구이므로 판정 게이트가 아니다.

**기존 [broden.py](../../../broden.py)를 재사용한다** (재작성 금지):
- `collect_activations_to_shards(...)` — 활성 캐시
- `compute_thresholds_from_shards(..., percentile, ...)` — feature별 활성 임계
- `compute_iou(records, concept_by_id, categories, latent_ids, thresholds, ...)` — 개념 IoU
- `Utils/broden_utils.discover_broden / load_broden_concepts / load_broden_index`

축 객체를 받도록 `_encode_cached_activations`만 일반화한다. 대조축 IoU도 같은 호출로 싸게 얻을 수 있으므로 **참고 수치로 병기하되 판정에는 쓰지 않는다**(마스크 밀도 민감성 때문).

**T3.1 의존**: 실험 7의 `broden` 군집 방식은 **전 latent의 top-IoU**를 요구한다. 후보 40개만으로는 부족하므로, 동일 캐시에서 `latent_ids=alive` 전체를 한 번 더 돌려 `full_latent_iou.json`을 부산물로 남긴다.

### 선행 조건 — Waterbirds 분류기 (spec에 없던 결정)

Waterbirds는 2-class(`y`) 문제인데 **ImageNet head로는 못 푼다.** 그렇다고 backbone을 finetune하면 block10 활성이 바뀌어 **ImageNet에서 학습한 SAE가 유효한 계기가 아니게 된다**(실험 1의 FVU 검증도 무효화).

→ **frozen ImageNet ViT-B/16 위 linear probe만 학습한다.** backbone·SAE 불변.
공개 Waterbirds 벤치마크 수치(ERM-finetuned ResNet-50)와는 **직접 비교 불가**임을 명시한다.

**PCA 적합 위치**: **SAE가 학습된 것과 같은 source(ImageNet-1k) block10 패치 토큰**에서 `E`를 적합한다. Waterbirds에서 적합하면 PCA만 test 도메인에 맞춰진 셈이라 불공정하다. Random은 같은 `d`, unit-norm, seed 고정.

### 산출물

```json
{"result": {
  "separation": {
    "sae":    {"n_features": 6401, "tau_spur": 0.34, "tau_caus": 0.35,
               "n_informative": 812, "n_pure_spurious": 231, "n_pure_causal": 186,
               "n_entangled": 395, "purity_rate": 0.513},
    "pca":    {"n_features": 768, "tau_spur": 0.28, "tau_caus": 0.29,
               "n_informative": 96, "n_pure_spurious": 7, "n_pure_causal": 4,
               "n_entangled": 85, "purity_rate": 0.115},
    "random": {"n_features": 768, "purity_rate": 0.086, "...": null}
  },
  "purity_significance": {"sae_vs_pca_p": 0.001, "sae_vs_random_p": 0.001,
                          "test": "two-proportion permutation, P=200"},
  "pool_pooling": "max",

  "diagnostic_validity": {
    "s_k_ap_for_pure_spurious": 0.61, "random_baseline_ap": 0.29,
    "s_times_1_minus_c_ap": 0.68,
    "corr_c_train_vs_caus": 0.12, "corr_c_bal_vs_caus": 0.57,
    "c_train_misranks_background": 0.44},

  "interpretability": {
    "spurious": [{"latent": 8123, "spur": 0.84, "caus": 0.07, "sel": 0.85,
                  "s_k": 0.61, "c_k_train": 0.019, "c_k_bal": 0.002,
                  "top_concept": "water", "category": "material", "iou": 0.34}],
    "causal":   [{"latent": 407, "spur": 0.06, "caus": 0.71, "sel": -0.84,
                  "s_k": 0.03, "c_k_train": 0.048, "c_k_bal": 0.051,
                  "top_concept": "beak", "category": "part", "iou": 0.21}],
    "category_dist": {"spurious": {"scene": 0.41, "material": 0.28, "object": 0.07},
                      "causal":   {"part": 0.44, "object": 0.31, "scene": 0.05}},
    "control_iou_reference": {"pca": 0.06, "random": 0.04, "sae": 0.24}}}}
```

`figures/sel_hist.png` — 축 3종의 `sel_j` 히스토그램 겹쳐 그리기. **쌍봉 vs 단봉이 이 실험의 핵심 그림이다.**
`figures/spur_vs_caus.png` — `(spur_j, caus_j)` 산점도, 축별 패널. 축에 붙은 점 = 분리된 feature.

### 판정 규칙

- **`pass`**: (A) SAE의 `purity_rate`가 PCA·random 양쪽보다 permutation 검정에서 유의하게 높다. **(B) `s_k` 기반 랭킹의 AP가 random baseline(= 순수-spurious 비율)을 유의하게 상회한다.**
- **`fail` (A 실패) → thesis 재검토.** monosemantic basis의 우위가 없으면 PCA/PLPD로 충분하다는 반론에 진다.
- **`fail` (A는 통과, B 실패)**: 분리는 존재하지만 **라벨 없이는 못 집는다**는 뜻. thesis가 죽지는 않지만 v1의 `c_k` anchor 설계와 "test time에 spurious를 겨냥한다"는 서술을 **전면 수정**해야 한다. T1.4b의 `anchor="ck"` 조건도 근거를 잃는다.
- ③(Broden)은 `pass`/`fail`을 가르지 않는다. 개념 이름이 안 붙어도 (A)(B)가 서면 통과이고, 해석성 주장만 약화된다.
- **실패를 계기 탓과 구분할 것.** 현 ckpt는 패치당 `L0 ≈ 497`(768차원 내)로 희소하지 않다. `purity_rate`가 낮게 나오면 thesis가 틀린 게 아니라 계기가 무딘 것일 수 있으므로, `L0` 수십 급 SAE(PatchSAE 등)로 재측정한 뒤 판정한다. 이 재측정 없이 `fail`을 확정하지 않는다.

---

## T1.4a — 실험 4A: 정적 basis head-to-head

**진입점**: `experiments/exp4a_basis.py`

### 동작

```
for arm in ["latent_gain", "channel_gain", "random_dict"]:
    for lr in [1e-3, 3e-3, 1e-2]:
        for seed in [0, 1, 2]:
            reset(); intervention = GainIntervention(backbone, BASIS[arm])
            runner = OnlineTTARunner(intervention, AdaContrastLoss(...),
                                     GainAnchor("l2", lam=LAMBDA_FIXED),
                                     TTAConfig(lr=lr, passes=PASSES_STATIC, ...))
            r = runner.run(target_stream)
    report[arm] = best_over_lr(mean_over_seeds)      # ← arm별 best-lr을 보고
```

**고정해야 하는 것** (하나라도 arm 간에 다르면 실험 무효):
- 주입 지점 block10, 손실 **AdaContrast 고정**(entropy 병기 금지 — 4B의 거리공간 축이 성립하려면 contrastive 항이 있어야 함)
- step 수, batch size, anchor `lam` **동일값으로 전 arm에 켠 채 고정**(끄면 collapse가 basis 효과를 덮음)
- `random_dict`의 `target_l0`은 하드코딩이 아니라 **M1의 `l0()`로 같은 데이터에서 측정한 값**

**arm별로 달라야 하는 것**: lr만. 축마다 활성 스케일이 달라 단일 lr 비교는 basis 효과가 아니라 최적화 아티팩트를 잰다.

### 산출물

```json
{"result": {
  "arms": {"latent_gain":  {"best_lr": 3e-3, "worst_group_acc": 0.72, "std": 0.011,
                            "mean_acc": 0.86, "cka_drift": 0.94},
           "channel_gain": {"best_lr": 1e-3, "worst_group_acc": 0.64, "std": 0.019, ...},
           "random_dict":  {"best_lr": 3e-3, "worst_group_acc": 0.61, "std": 0.023, ...}},
  "target_l0_used": 497,
  "margin_vs_channel": 0.08, "margin_vs_random": 0.11}}
```

### 판정 규칙

- `pass`: `latent_gain`의 worst-group acc가 **`channel_gain`과 `random_dict` 양쪽보다** 시드 표준편차를 넘어 우세.
- **`fail` (채널 gain이 이기거나 비등)**: test time에 SAE가 불필요하다는 뜻 → **thesis 붕괴 게이트.**
- **`random_dict`가 근접**: 이득의 원인이 "학습된 딕셔너리"가 아니라 "sparse overcomplete 구조" 자체다. 기여 주장을 그쪽으로 정정해야 한다(실패는 아니지만 논문의 claim이 바뀐다).
- **계기 조건 각주**: 현 ckpt에서는 세 arm의 유효 rank가 가까워(패치당 `L0 ≈ 497` / 768차원) **분리가 약할 수 있다.** 결론이 애매하면 계기를 올린 뒤 재측정한다.

---

## T1.4b — 실험 4B: SAE arm 설계 ablation

**진입점**: `experiments/exp4b_design_ablation.py`

arm을 `latent_gain` 하나로 고정하고 3축 factorial. 세 축 모두 SAE arm 위에서만 정의되므로 4A와 같은 표에 들어갈 수 없다.

| 축 | 값 |
|---|---|
| anchor | `off` / `l2`(균일) / `ck`(`c_k` 가중) |
| 거리공간 | `h'`(backbone feat) / `z`(latent) / `logit` |
| residual | `on`(`h + Δ`) / `off`(`ĥ + Δ`, 재구성 치환) |

3×3×2 = 18 조건 × 3 seed. `lam ∈ {0.1, 1.0}`, `tau = 0.07`, 나머지는 4A와 동일.

**의존**: `anchor="ck"`가 T1.3의 `c_k` 산출물을 그대로 읽는다. **`c_k^train`(편향 source에서 잰 현실적 버전)을 쓴다** — `c_k^bal`은 그룹 라벨을 알아야 만들 수 있어 TTA 가정에 어긋난다. T1.3의 `corr_c_train_vs_caus`가 낮게 나왔다면 이 조건의 기대치를 그만큼 낮춰 해석한다.

### 산출물

```json
{"result": {
  "grid": [{"anchor": "ck", "distance_space": "h", "residual": "on",
            "worst_group_acc": 0.72, "std": 0.011,
            "gain_dev_final": 0.31, "collapsed": false}],
  "marginal": {"distance_space": {"h": 0.71, "z": 0.66, "logit": 0.63},
               "anchor": {"off": 0.64, "l2": 0.70, "ck": 0.72},
               "residual": {"on": 0.71, "off": 0.68}}}}
```

`gain_dev_final` = `‖gain − 1‖`. `collapsed` = 예측이 한 클래스로 붕괴했는지.

### 판정 규칙

- **거리공간**: `h'`가 `z`·`logit`보다 우세하면 설계의 "거리는 backbone에서"가 정당화된다.
- **anchor**: `ck`가 `l2`보다 유의하게 나으면 **선택적 보존의 값**이 입증. 차이가 없으면 `c_k`를 v1에서 빼는 근거가 된다(부정 결과도 결론이다).
- **residual**: `on`이 안정적이어야 한다. `off`는 SAE 재구성 오차가 경로에 들어와 `gain=1`조차 무개입과 달라진다.
- **`anchor=off`에서 `collapsed=true`가 다수**면 M5 anchor의 비폭주 보증(`‖g−1‖ ≤ G/lam`)이 경험적으로 확인된 것이다 — 이 대비를 논문 명제의 근거로 인용한다.
