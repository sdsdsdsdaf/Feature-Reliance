# 제안할 방법

> 근거: `plan.md`(SAE 기반 Feature Disentanglement × TTA) + 세션 논의로 확정한 v1 설계.
> 수식은 VS Code 렌더 문제로 코드 스팬 사용. 생성 2026-08-04.

## 방법

### 한 줄
Frozen ViT-B/16 + frozen SAE 위에서, SAE latent `z`에 붙는 **학습 gain `alpha`만** AdaContrast식 손실로 test-time에 GD로 학습한다. 개입은 **monosemantic latent 축**에서 하되, contrastive의 **거리(pos/neg)는 backbone feature에서** 잰다.

### 구성요소
| 구성요소 | 상태 | 갱신 방식 |
|---|---|---|
| ViT-B/16 backbone (block0~12) | **frozen** | — |
| SAE (encoder + decoder), block10·patch·`K=12288` | **frozen** | — (측정 계기) |
| **latent gain `alpha ∈ R^K`** (init=1) | **learnable** | **GD (optimizer엔 alpha만)** |
| memory bank (pseudo-label용 feature/label 큐) | 갱신 | EMA/enqueue (gradient 아님) |
| (fallback) anchor 세기 `λ`, cluster gain `alpha_g` | 상황부 | GD/스케줄 |

### 적응 손실 (AdaContrast식)
```
L = L_pseudo(weak-aug, self-training)  +  L_ctr(contrastive, on backbone feature)  [+ L_div(diversity)]
```
-> L_div(diversity)는 추후 선택으로 쓸 수 있도록 구현한다. ex) Loss(use_diversity=True)
- **pseudo-label**: weak-aug feature를 memory bank에서 k-NN soft voting → 소프트 라벨. (backbone frozen이라 momentum encoder 불필요, weak/strong augmentation 차이만 사용.)
- **contrastive `L_ctr`**: strong-aug query와 positive를 당기고 negative를 민다. 같은 pseudo-label 샘플은 negative에서 제외(false-negative 방지). **거리는 개입 후 backbone feature `h'`(또는 CLS 표현)에서** 계산. `z` 위가 아님.
- **`c_k` 진단 항은 v1 적응 손실에 미포함** (아래 fallback에서만 복귀).

### 적응 프로토콜 (default = AdaContrast 그대로)
**기본은 AdaContrast 원본 레시피를 그대로 따른다** — memory bank(k-NN soft-voting pseudo-label) · weak/strong augmentation · contrastive(same-pseudo negative 제외) · diversity(옵션). 우리 쪽 변경은 **backbone·SAE frozen + `alpha`만 학습 + 거리는 backbone feature** 뿐이고, contrastive 손실·pseudo-label 정제 로직은 건드리지 않는다.
- **단 메모리·pass를 sweepable 축으로 구현**한다 → 같은 코드로 `buffer ∈ {0(=in-batch), 256, 2048, full-queue}` × `pass ∈ {single-pass, multi-epoch}` 를 갈아끼울 수 있게(측정은 실험 6).
- `buffer=0`이면 negative=in-batch · pseudo=weak-aug 자기예측(=가장 순수한 online TTA), 값이 커질수록 AdaContrast 원본에 수렴. **default는 원본(bank 사용), in-batch는 옵션 경로.**

### gradient 경로 (핵심 트릭)
`alpha`는 `z` 위에서 움직이지만 손실은 `h'`에서 잰다 →
```
L(h') ──backprop──▶ frozen block11~12 ──▶ frozen decoder W_dec ──▶ alpha
```
`requires_grad=False`는 **weight의 `.grad`만** 막고 **통과 grad(`grad_x = Wᵀ·grad_y`)는 살린다**. 그래서 backbone·SAE를 전부 얼려도 `alpha`만 학습된다. 구현은 상류(block0~10·encoder)를 `no_grad`+`detach`로 상수화하고, **주입 지점(block10) 이후 꼬리만** 그래프에 남겨 저비용.

### 진단(오프라인·참고용, v1 적응엔 미사용)
- **쓸모 점수 `c_k = acc(source) − acc(source | latent k off)`** — causal 여부 프록시(source 라벨로 1회).
- **흔들림 점수 `s_k = |firing_k(target) − firing_k(source)|`** — 도메인 단위 발화율 차. **라벨 불필요.**
- 위 둘은 **진단**이지 정답이 아니다. 무엇이 spurious인지의 **정답**은 Waterbirds의 `y`/`place` 라벨로 따로 정의하고(실험 3 ①), 실험 3은 `s_k`·`c_k`가 그 정답을 복원하는지를 검정한다.
- **Broden IoU** — 분리된 feature에 사람이 읽는 개념 이름(`water`, `beak`)을 붙이는 **해석 도구**(`broden.py`). 분리 여부의 판정 기준이 아니다.

### Fallback / 확장 (contrastive 단독이 부족할 때)
| 순위 | 방법 | 트리거 |
|---|---|---|
| 1 (main) | AdaContrast + **개별 latent gain `alpha`** | 기본 |
| 2 (FB-anchor) | `+ λ·Σ_k c_k(a_k−1)²` 또는 `+ λ·‖alpha−1‖²` | collapse/불안정 시 |
| 3 (FB-cluster) | K개 latent → **M개 개념 클러스터**의 **group gain `alpha_g`** | 개별 가중이 노이즈/과적합일 때 |
| 비고 (EXT) | **Continual Learning 전환** — recurring/prolonged 스트림, SAE descriptor 라우팅, 도메인당 gain 저장·재로드, forgetting 관리 | 스트림 장기화 |

### 가정 & source-free 준수
"source-free"는 **test 시점에 source data가 없다**는 뜻이지 "source를 한 번도 안 봤다"가 아니다(분류기 자체가 source 학습됨). 배포 땐 **frozen SAE + latent sketch만** 들고 가고 source data는 안 들고 간다 — Tent/T3A/SHOT/AdaContrast와 같은 급의 오프라인 산물.

| 방법 | 오프라인 source 산물 (test 시 미보유) |
|---|---|
| Tent | BN 통계 |
| T3A | class prototype |
| SHOT/AdaContrast | source-trained weight |
| **우리** | **frozen SAE + latent sketch** |

- **v1은 더 가볍다**: SAE는 비지도라 **source 라벨 불필요**. 라벨 쓰는 `c_k`는 v1 제외 → v1의 source 의존 = **unlabeled source feature뿐**.
- **정직한 각주(reviewer 공격 지점)**: 순수 Tent보다 오프라인 가정이 약간 무겁다 — 분류기 외에 **SAE라는 extra module을 source에서 학습**(TTT가 source에서 self-sup head 학습하는 것과 동급). 반칙은 아니나 "extra"는 맞음.
- **strict-no-source fallback**: source 자체 금지 시 (a) unlabeled target/proxy로 SAE 학습, (b) 공개 pretrained SAE 재사용. → **실험 1의 SAE-source ablation**으로 "source 의존은 편의일 뿐 필수 아님" 입증.

## 전체 파이프라인

### 오프라인 (배포 전, source에서 1회)
```
source 표현(ViT-B block10 patch) ─▶ SAE 비지도 학습 ─▶ frozen SAE
                                                  └▶ (참고) s_k·c_k 진단 측정, Broden IoU 개념 라벨링
```

### 온라인 (test-time, 라벨 없음)
```
x_t ─▶ ViT block0..10 ─(no_grad)─▶ h ─▶ SAE.encode ─(detach)─▶ z            [상수]
                                                    │
                                     z' = alpha ⊙ z │  (alpha만 학습, init 1 → no-op)
                                                    ▼
        h⁺ = h + sd ⊙ W_dec^T((alpha − 1) ⊙ z) ─▶ ViT block11..12 ─▶ h'(backbone feat) ─▶ head ─▶ logits
        (sd = ckpt의 `token_std`. SAE는 정규화 공간 `(h−mu)/sd`에서 동작하므로 주입 전 역정규화 필요)
                                                    │
   [AdaContrast]                                    │
     weak-aug(x_t) ─▶ h'_w ─▶ memory bank k-NN ─▶ pseudo-label ŷ
     strong-aug(x_t) ─▶ h'_s ─▶ L_ctr(h'_s, pos/neg; same-ŷ는 neg 제외) + L_pseudo(h'_s, ŷ) + L_div
                                                    │
                        L ──backprop(통과 grad)──▶ decoder·block11~12 거쳐 ▶ alpha 만 update
```

### frozen / learnable 요약
- **frozen**: backbone weight, SAE(E, W_dec). **learnable**: `alpha`(또는 cluster `alpha_g`). **EMA/큐**: memory bank, (EXT) router centroid·도메인 gain.
- ViT는 **LayerNorm**(BN 아님) → Tent식 BN 통계 재보정 불가. gain `alpha`가 그 자리를 대신.

---

# 실험

> 설계 원칙: **적응 엔진(TTA 루프)은 고정하고 "어디서(basis) 적응하나"만 변수로** 두어 "왜 SAE"를 인과적으로 격리. 싸게 falsify되는 것부터 배치.
> 공통 모델(별도 명시 없으면): backbone `vit_base_patch16_224`, SAE `outputs/reservoir_sae/vit_b_sae.pt`(block10·patch·`input_dim=768`·`hidden_dim=12288`(**expansion 16**)). SAE는 정규화 공간에서 동작 — ckpt의 `token_mean`/`token_std`를 적용/역적용해야 함.
> 데이터 보유 현황 — **보유**: ImageNet-1k, ImageNet-C(19종=15 test+4 extra, `data/imagenet-c`), imagenette, Broden(`data/broden1_227`), **Waterbirds(`data/waterbirds/waterbird_complete95_forest2water2`)**. **확보 필요**: ColoredMNIST, ImageNet-R/Sketch/A, ImageNet-9(BG Challenge).
> **ImageNet-C 규약**: `data/imagenet-c`(공식 precomputed, Zenodo 2235448)를 **그대로 사용, 재생성(on-the-fly) 안 함** — 라이브러리 버전 드리프트로 baseline과 비교 불가해지는 것 방지. recurring 스트림도 precomputed에서 **순서만 재배열**. **HP 튜닝은 4 extra(`gaussian_blur·saturate·spatter·speckle_noise`) hold-out에서, 보고는 15 test에서** → target 라벨 없는 HP 선택 방어.

---

## 실험 1. SAE 계기(measurement) 신뢰성 + OOD FVU 게이트

## 해야 하는 이유
방법 전체가 "SAE가 표현을 충실히·sparse하게 재구성한다"에 의존한다. 특히 plan.md §7의 리스크 — **source로 학습한 SAE가 shift에서 재구성이 무너지면 gain·진단이 오염**된다. 적응 실험 이전에 계기부터 검증하지 않으면 이후 결과 해석 불가. (plan.md Phase 0 + OOD 검증은 본 계획의 추가.)

## 실험과정
1. `SAE_validation.py`로 block10 SAE 재구성 손실·sparsity(활성 latent 수·L0) 재확인.
2. **In-domain(source)** vs **OOD(ImageNet-C 각 corruption·severity)** 에서 재구성 지표 비교: `FVU`(fraction of variance unexplained), 활성 latent 분포 이동.
3. **FVU 게이트 기준선** 설정: FVU가 임계 초과하는 shift를 "SAE 신뢰 불가 → 개입 보류/약화" 구역으로 표시(VS2식).
4. 재구성 품질 vs 다운스트림 정확도(재구성 표현으로 forward) 상관 확인.
5. **SAE-source ablation** (source-free 방어): SAE를 `source(ImageNet-1k)` vs `unlabeled proxy` vs `공개 pretrained SAE`로 각각 학습/로드해 재구성·게이트·다운스트림이 유지되는지 → "source 의존은 편의일 뿐 필수 아님(strict-no-source)" 입증.

## 실험세팅
### 1. 데이터셋
- source: ImageNet-1k val(또는 학습에 쓴 토큰 소스), OOD: ImageNet-C precomputed 15 test × 5 severity, (선택) ImageNet-R. ablation용 proxy: imagenette/unlabeled 소량.

### 2. 모델
- `vit_base_patch16_224` + `outputs/reservoir_sae/vit_b_sae.pt`. 개입/학습 없음(forward-only).

### 3. 하이퍼파라미터
- token scope: patch, hook block: 10. active_threshold 0.2(체크포인트값), 활성 percentile 99.
- 지표: FVU, L0(=평균 활성 latent 수), reconstruction cosine, severity별 곡선. FVU 게이트 임계는 source FVU 분포의 상위 분위수로 설정.

## 예상결과
- source에선 FVU 낮고 sparse 유지. corruption severity↑ → FVU 단조 증가(특히 noise류). **easy corruption은 완만, 강한 shift에서 급증**하는 곡선 → FVU 게이트의 정당성 확보. 게이트 밖(신뢰 가능) 구역에서만 이후 개입을 신뢰한다는 경계가 서면 성공. **ablation: proxy/public SAE로도 재구성·게이트가 유지되면 source-free 방어 완료.**

---

## 실험 2. 개입 메커니즘 sanity — gradient 경로·no-op 안전성·표현력

## 해야 하는 이유
"`alpha`만 학습, backbone·SAE frozen, `alpha=1`은 no-op, 거리는 backbone feature" 설계가 **실제 ViT+SAE에서 그대로 성립하는지**를 정확도 실험 이전에 격리 검증해야 한다. 여기서 새는 게 있으면(예: no-op가 정확도를 떨어뜨림, gain 표현력이 너무 작음) 이후 모든 표가 무의미. (본 계획의 추가 실험 — 세션의 frozen-grad 원리 검증을 실모델로 확장.)

## 실험과정
1. **no-op 항등성**: `alpha≡1` 주입 표현으로 forward한 정확도 == 무개입 정확도 인지(재구성 오차 범위 내) 확인.
2. **gradient 경로**: 손실을 `h'`에서 잡고 backward 후 `alpha.grad ≠ 0`, backbone·SAE weight의 `.grad is None`, optimizer step이 `alpha`만 바꾸는지 assert.
3. **표현력(capacity) 상한**: source/약한 shift에서 라벨을 주고 `alpha`만 학습했을 때 도달 가능한 정확도 상한 측정 → **채널 affine(LN affine, Tent 파라미터화)** 의 상한과 비교(gain 재가중이 부족한지 조기 진단, plan.md §7 capacity 리스크).
4. 격차가 크게 나오면 **새 모듈을 넣어 메우지 않는다**(v1 = gain 단독). FB-cluster·손실 설계로 대응할지, 대가로 보고할지를 여기서 판단한다.

## 실험세팅
### 1. 데이터셋
- ImageNet-1k val(항등성·gradient), ImageNet-C 약·강 severity 각 1종(capacity 상한).

### 2. 모델
- `vit_base_patch16_224` + SAE. 개입 지점 block10 patch. 비교군: LN affine 파라미터화(Tent).

### 3. 하이퍼파라미터
- optimizer: Adam, lr(alpha) `1e-3`(gain은 스케일 민감 → `{1e-3,3e-3,1e-2}` 스윕). init `alpha=1`. steps 100~300(상한 측정이므로 라벨 supervised).
- assert 항목: `alpha.grad` norm>0, frozen `.grad is None`, `‖alpha−1‖` 궤적.

## 예상결과
- no-op 정확도 == 무개입(±재구성 오차). `alpha.grad`만 채워지고 frozen weight 불변. capacity: **gain 재가중은 채널 affine보다 상한이 다소 낮을 수 있음** — 이건 **정직하게 보고할 대가**이지 새 모듈로 메울 대상이 아니다(v1 = gain 단독). 격차 크기가 FB-cluster·손실 설계로 대응할지를 정하는 게이트.

---

## 실험 3. spurious/causal 분리 검증 (Waterbirds) — MVP 핵심

## 해야 하는 이유
thesis의 **falsifiable 핵심**: SAE latent 좌표계가 정말 **spurious 개념과 causal 개념을 분리**하는가. 이게 안 되면 "monosemantic basis 우위"라는 논거 자체가 붕괴 → PCA/PLPD로 충분하다는 반론에 진다. 정확도보다 **분리 가능성**을 먼저 증명해야 한다. (plan.md Phase 1.)

**측정을 3층으로 분리한다.** "`s_k` 높고 `c_k` 낮은 걸 뽑아서 spurious라 부른다"는 것 자체로는 아무것도 증명하지 못한다(순환논증). 무엇이 spurious인지를 정하는 **독립적인 정답**이 먼저 있어야 한다.

| 층 | 무엇 | 쓰는 라벨 | 역할 |
|---|---|---|---|
| ① **정답** | `spur_j`/`caus_j` (층화 AUC) | Waterbirds `y`·`place` | 무엇이 spurious인지를 **정의** |
| ② **진단** | `s_k`(발화율 shift) / `c_k`(ablation) | 없음 / source 라벨만 | test time에 쓸 수 있는 대용물 |
| ③ **해석** | Broden IoU | Broden 개념 마스크 | feature에 개념 **이름**을 붙임 |

Waterbirds는 새를 배경 위에 **합성해서** 만든 데이터셋이므로 `place`(land/water)가 spurious, `y`(landbird/waterbird)가 causal이라는 게 **설계상 확정**돼 있고 두 라벨이 `metadata.csv`에 그대로 있다. 정답을 Broden으로 우회해 알아낼 필요가 없다. **Broden은 판정 게이트가 아니라 해석 도구로 강등한다.**

검정할 명제:
- **(A) 분리** — SAE 축에 "배경만 보는 feature"와 "새만 보는 feature"가 따로 존재하는가. PCA·무작위 축은 아닌가. ← **주 결과**
- **(B) 진단 타당성** — 라벨 없는 `s_k`(+`c_k`)가 ①의 정답을 복원하는가. 이게 서야 "test time에 라벨 없이 spurious를 겨냥한다"는 주장이 성립한다.
- **(C) 해석성** — 분리된 feature에 `water`·`beak` 같은 개념 이름이 붙는가. ← **보조**

## 실험과정
1. **정답 정의(①)**: feature `j`의 이미지 단위 점수 `a_j`(패치 점수 pooling, 기본 `max`)에 대해
   `spur_j = mean_y 2|AUC(a_j→place | y 고정) − 0.5|`, `caus_j = mean_place 2|AUC(a_j→y | place 고정) − 0.5|`.
   **층화가 필수** — Waterbirds는 `y`와 `place`가 95% 상관이라 층화 없이 재면 순수 배경 탐지기도 causal처럼 보인다. `|AUC−0.5|`이므로 **부호 무관·스케일 무관·임계 불필요**.
2. **임계(permutation max-null)**: 축마다 층 안에서 라벨을 셔플해 `max_j spur_j`의 귀무분포(P=200)를 만들고 95분위를 `tau`로 쓴다. **max 통계라 pool이 클수록 `tau`가 자동으로 높아져** 축 간 후보 수 비대칭(SAE ~6,000 vs PCA 768)이 보정된다.
3. **분리도 산출**: `sel_j = (spur_j − caus_j)/(spur_j + caus_j)`(+1 순수 배경, −1 순수 새, 0 entangled),
   `purity_rate = |{informative & |sel_j|≥0.6}| / |{informative}|`. **비율이라 pool 크기가 상쇄된다.**
4. **대조축**: 동일 절차를 **PCA 축**·**무작위 축**에 적용 → SAE만 `purity_rate`가 유의하게 높은지(basis가 원인).
5. **진단 타당성(②)**: `s_k`(및 `s_k·(1−c_k)`)를 랭커로 ①의 순수-spurious 집합을 뽑는 **average precision**. `corr(c_k, caus_j)`. **`c_k`는 편향 train에서 잰 `c_k^train`과 그룹 균형에서 잰 `c_k^bal`을 병기** — 95% 상관 하에서 `c_k^train`이 배경을 causal로 오판하는 폭이 곧 §7 리스크의 실측값.
6. **해석성(③)**: ①에서 순수로 판정된 SAE feature 상위 40개에만 `broden.py` IoU를 돌려 개념·category 분포를 낸다. 실험 7(군집)용으로 alive latent 전체 IoU도 부산물로 남긴다.

## 실험세팅
### 1. 데이터셋
- **Waterbirds**(보유: `data/waterbirds/waterbird_complete95_forest2water2`). ①·②는 `metadata.csv`의 `y`/`place`/`split`만 쓴다. **①은 `split==2`(test, 그룹 균형)**, `c_k^train`은 train split. ③의 개념 라벨은 Broden(`data/broden1_227`). 보조: ImageNet-9/BG Challenge(확보 필요).

### 2. 모델
- `vit_base_patch16_224` + SAE. **frozen backbone 위 linear probe만 학습**(finetune하면 block10 활성이 바뀌어 SAE가 유효한 계기가 아니게 됨 → 공개 Waterbirds 수치와 직접 비교 불가임을 명시). PCA/random 축의 `E`는 **SAE와 같은 source(ImageNet-1k) block10 토큰**에서 적합. `broden.py`(activation-percentile 99, sensitivity 95/98/99).

### 3. 하이퍼파라미터
- pooling `max`(robustness로 `mean` 병기), permutation `P=200`, purity 컷 `|sel|≥0.6`, Broden 후보 `N=40`.
- `c_k`: latent off = 해당 latent 재구성 기여 제거 후 정확도 하락. `s_k`: **target 스트림 vs source의 발화율 차**(`place` 라벨 미사용 — 정답 라벨을 진단에 흘리면 ②가 무의미해진다). Broden category: object,part,color,material,texture,scene.

## 예상결과
- **(A)** SAE의 `sel_j` 분포가 ±1 근처 **쌍봉**, PCA/무작위는 0 근처 **단봉** → `purity_rate`가 유의하게 높음. disentanglement가 SAE 축 고유 특성임을 입증.
- **(B)** `s_k` 랭킹의 AP가 무작위 baseline을 상회. **`c_k^train`은 배경을 causal로 오판**하고 `c_k^bal`이 정답과 잘 맞을 것 — 이 격차가 `c_k` 가정의 정직한 실측 보고가 된다.
- **(C)** SPURIOUS feature → **water/land·scene·material**, CAUSAL feature → **beak/wing 등 object·part**.
- **(A) 실패 시 thesis 재검토** — 이 실험이 저비용 게이트인 이유. **(A) 통과·(B) 실패**면 thesis는 살지만 v1의 `c_k` anchor 설계와 "test time에 spurious를 겨냥한다"는 서술을 전면 수정해야 한다(실험 4B의 `anchor=ck` 조건도 근거를 잃음). (C)는 판정을 가르지 않는다.

---

## 실험 4A. 정적 basis head-to-head — latent gain vs 채널 gain vs random dictionary

## 해야 하는 이유
"왜 SAE 축인가"는 **같은 주입 지점·같은 손실·같은 학습량에서 gain의 주소지정 방식만 바꿔** 이겨야 성립한다. 온라인 루프의 혼란 없이 정적으로 격리. (plan.md Phase 2 + §4.4 통제군.)

**TACT/DeYO는 이 실험에서 제외한다.** TACT는 augmentation 집합의 per-sample PCA로 분산 상위 `m`(≤16)개 성분을 하드 제거하는 **backprop-free 방법**이고 개입 지점도 prototype 분류용 최종 표현이다. DeYO는 **샘플 keep/drop 필터**로 자체 적응 파라미터가 없다. 둘 다 "교체할 적응 파라미터"가 없어 이 틀에 들어가지 않으므로 **실험 5 baseline 표**에서 method 대 method로 비교한다. (구 계획의 `PCA-dim gain` arm은 TACT에 존재하지 않는 구성이라 삭제.)

## 실험과정
1. 공통 정적 TTA 루프(라벨 없이 target set에 AdaContrast 손실로 개입)에서 **gain의 주소지정만 교체**. 주입 지점은 3종 모두 **block10 출력으로 동일**:
   - **`latent gain`(제안)**: `h⁺ = h + sd ⊙ W_dec^T((alpha − 1) ⊙ z)`, `alpha ∈ R^K`, init 1. 개념(딕셔너리 원자)마다 스칼라 1개 — support가 **입력에 따라 달라짐**.
   - **`채널 gain`**: `h⁺ = h + (gamma − 1) ⊙ h`, `gamma ∈ R^768`, init 1. **SAE를 거치지 않고** 생 채널마다 스칼라 1개 — 모든 토큰에 **동일하게 적용되는 고정 대각 사상**. Tent의 파라미터화를 우리 주입 지점으로 옮겨 변수를 하나로 줄인 형태(Tent 원본 method는 실험 5 baseline).
   - **`random dictionary gain`**: `W_dec`를 unit-norm **무작위** 행렬 `R`로 교체하고 SAE와 **동일 `K`·동일 sparsity**(TopK를 SAE 실측 L0에 맞춤). `h⁺ = h + sd ⊙ R^T((g − 1) ⊙ c)`. 조건부성·파라미터 수를 맞추고 **딕셔너리가 학습된 것이냐만** 남기는 통제.
   - (참고행, 선택) **`채널 affine`**: `gamma ⊙ h + beta` — Tent 원본 형태. gain 단독 대비 shift 항의 기여 확인용.
2. biased·natural 데이터에서 worst-group / 평균 정확도로 순위. 표현 drift는 CKA/JS로 병기.

## 실험세팅
### 1. 데이터셋
- biased: Waterbirds(보유), ColoredMNIST(확보 필요). natural: ImageNet-R(200)(확보 필요). corruption 참조: ImageNet-C 일부.

### 2. 모델
- `vit_base_patch16_224` + SAE. 3종 arm은 **동일 백본·동일 주입 지점·동일 손실·동일 step**에서만 상이.

### 3. 하이퍼파라미터
- 손실: **AdaContrast로 고정**(구 계획의 "entropy 또는" 삭제 — 4B의 거리공간 축이 성립하려면 contrastive 거리 항이 있어야 한다).
- optimizer Adam, steps 200~500(정적). **lr은 arm별로 `{1e-3, 3e-3, 1e-2}` 스윕 후 arm별 best를 보고** — 축마다 활성 스케일이 달라 단일 lr 비교는 basis 효과가 아니라 최적화 아티팩트를 잰다.
- anchor는 **전 arm에 균일 `λ‖g−1‖²`를 같은 값으로 켠 채 고정**(4A에서는 변수가 아님 — 켜지 않으면 collapse가 basis 효과를 덮는다). 거리 temperature `τ=0.07`.

## 예상결과
- **latent gain이 채널 gain 대비 worst-group에서 우세**(특히 biased). 채널 gain이 이기거나 비등하면 **test time에 SAE가 불필요**하다는 뜻 → thesis 붕괴 게이트.
- **random dictionary가 latent gain에 근접하면** 이득의 원인이 "학습된 딕셔너리"가 아니라 "sparse overcomplete 구조" 자체 → 기여 주장을 그쪽으로 정정해야 한다.
- **계기 조건 각주**: 현 checkpoint는 패치당 실측 `L0 ≈ 497`(768차원 내)이라 세 arm의 유효 rank가 가까워 **분리가 약할 수 있다**. `L0` 수십 급 계기(PatchSAE 등)로 올린 뒤 재측정 권장.

---

## 실험 4B. SAE arm 설계 ablation — anchor · 거리공간 · residual

## 해야 하는 이유
우리 설계의 특이점인 **"개입은 latent `z`에서, 거리는 backbone feature `h'`에서"** 가 실제로 이득인지 검증해야 한다. 단 이 세 축은 **SAE arm 위에서만 정의된다** — PCA축엔 `c_k`가 없고, 채널 gain엔 코드 `z`가 없고, 재구성 경로가 없는 arm엔 residual 선택지 자체가 없다. 그래서 4A와 같은 표에 넣으면 빈칸투성이 격자가 되므로 별도 실험으로 분리한다. (본 계획의 추가.)

## 실험과정
1. arm을 **`latent gain` 하나로 고정**하고 3축 factorial:
   - **anchor**: `{off, 균일 λ‖alpha−1‖², c_k 가중 λ·Σ_k c_k(alpha_k−1)²}`
   - **거리공간**: `{backbone h', latent z, logit}`
   - **residual 주입**: `{on = h + Δ, off = ĥ + Δ(재구성으로 치환)}`
2. worst-group / 평균 정확도 + 학습 안정성(`‖alpha−1‖` 궤적, 시드 간 분산)으로 판정.

## 실험세팅
### 1. 데이터셋
- 4A와 동일(Waterbirds 중심).

### 2. 모델
- `vit_base_patch16_224` + SAE. `c_k`는 **실험 3 산출물의 `c_k^train`을 그대로 사용**(그룹 라벨이 필요한 `c_k^bal`은 TTA 가정 위반) → 4B는 실험 3에 의존. 실험 3의 (B)가 실패했다면 이 조건의 기대치를 그만큼 낮춰 해석한다.

### 3. 하이퍼파라미터
- 손실 AdaContrast 고정. `λ ∈ {0.1, 1.0}`, `τ=0.07`. optimizer/step은 4A와 동일.

## 예상결과
- **거리공간 = backbone feature가 `z`/logit보다 안정·정확** → 설계의 "거리는 backbone에서" 정당화.
- **`c_k` 가중이 균일 L2 대비 추가 기여**하면 선택적 보존의 값이 입증됨. 차이가 없으면 `c_k`를 v1에서 빼는 근거가 된다.
- residual on이 off보다 안정(no-op 안전성 + SAE 재구성 오차가 경로에 들어오지 않음).

---

## 실험 5. 온라인 AdaContrast TTA — 메인 결과·crossover

## 해야 하는 이유
논문의 주 표. **easy corruption에선 Tent/TACT와 비등(정직)** 하고 **biased/wild에서만 우세**한 **crossover** 를 전 baseline 대비 스트림에서 보여야 "왜 SAE"가 성능으로도 성립. (plan.md Phase 3.)

## 실험과정
1. 온라인 스트림에서 배치별 1-step 적응(§ 파이프라인 그대로): weak-aug pseudo-label + strong-aug contrastive(거리 backbone feat) → `alpha`만 GD.
2. 전 baseline과 동일 스트림·백본으로 비교. regime별(spurious/natural/corruption/wild) 정확도·acc-vs-time·표현 drift 기록.
3. **hyperparameter 민감도 그래프**(넓은 `λ`·lr 범위에서 평탄) 필수 — TTA reviewer 1순위 공격 대응.

## 실험세팅
### 1. 데이터셋
| regime | 데이터 | 지표 |
|---|---|---|
| spurious | Waterbirds, ColoredMNIST | worst-group acc |
| natural | ImageNet-R/Sketch/A | acc/rel-acc |
| corruption | ImageNet-C 15×5 | acc |
| wild/continual | mixed·recurring IN-C | acc-vs-time |

### 2. 모델
- `vit_base_patch16_224` + SAE. head는 source 분류기.

### 3. 하이퍼파라미터
- optimizer Adam, lr(alpha) `1e-3`, batch 64, step/batch=1(온라인). memory bank size 매 클래스 균형 큐(예: 총 16k), k-NN `k=10`, `τ=0.07`, diversity weight `0.1`, anchor `λ`는 off(v1) 기준·민감도용 스윕.
- baseline: No-adapt, Tent, EATA, SAR, **DeYO**, **TACT**, CoTTA, AdaContrast, ReservoirTTA(동일 셋업).

## 예상결과
- **corruption(easy): SAE ≈ Tent/TACT(비등, 정직)**. **spurious·wild: SAE가 worst-group·acc-vs-time에서 우세**. crossover 곡선이 드러나고, 민감도 그래프가 넓은 범위에서 평탄하면 핵심 논거 성립. Tent가 wild에서 collapse할 때 우리는 안정(monosemantic + 방향 제약).

---

## 실험 6. 적응 프로토콜·메모리 ablation — in-batch ↔ bounded ↔ full AdaContrast (TTA 정체성 방어)

## 해야 하는 이유
default가 AdaContrast 원본(memory bank·multi-epoch 가능)이라 "이거 사실 SFDA/UDA 아니냐"는 공격을 받는다. **default는 바꾸지 않고**, 같은 방법을 메모리·pass 축에서 좁혀 (a) **bank 없이(in-batch)도 성립**함을 보이면 **순수 online TTA 정체성**이 서고, (b) 원본으로 넓히면 상한을 보며 **TTA↔SFDA 스펙트럼 상의 위치**를 정량화한다. "가능성"만 확보하는 게 목적이라 별도 실험으로 분리.

## 실험과정
1. **default = AdaContrast 원본**(실험 5와 동일 코드)을 기준점으로 고정.
2. `buffer ∈ {0(=in-batch), 256, 2048, full-queue}` 스윕 — negative 수·pseudo-label 출처(self-예측 vs k-NN soft-voting)가 성능에 미치는 영향.
3. `pass ∈ {single-pass(=online), multi-epoch(2·5)}` 스윕 — single-pass(TTA) vs multi-epoch(SFDA)의 정확도 차이. 평가는 각각 **online 누적 / adapt-then-eval** 로 정직하게 병기.
4. `buffer=0·single-pass`(순수 TTA)와 `full·multi-epoch`(SFDA 상한)의 격차를 측정 → 우리 이득이 **프로토콜(예산) 덕인지 basis(SAE) 덕인지** 격리.

## 실험세팅
### 1. 데이터셋
- corruption: ImageNet-C 일부(빠른 스윕). biased: Waterbirds(정체성이 중요한 곳). natural: ImageNet-R.

### 2. 모델
- `vit_base_patch16_224` + SAE. 개입·손실은 실험 5 그대로, **메모리·pass만 변수**.

### 3. 하이퍼파라미터
- `buffer ∈ {0,256,2048,full}`. `epochs ∈ {1(=online),2,5}`. k-NN `k=10`(buffer>0일 때만), `τ=0.07`, batch 64. 나머지는 실험 5와 동일.
- 평가 프로토콜 명시: single-pass=online 누적 acc, multi-epoch=adapt 후 eval.

## 예상결과
- **buffer=0(in-batch)도 acc가 크게 무너지지 않으면** → "bank 없이 순수 TTA로 성립" = 정체성 방어 완료(핵심). buffer↑ → 완만한 개선 후 포화. **multi-epoch가 상한은 올리되 single-pass 대비 이득이 작으면** → 우리 값이 SFDA 예산이 아니라 **basis(SAE)** 에서 온다는 증거. biased에서 in-batch 이득이 corruption보다 큼.

---

## 실험 7. Clustering fallback — 개별 gain vs 개념 클러스터 gain

## 해야 하는 이유
`K=12288` 개별 gain은 **노이즈·과적합** 위험(라벨 없는 TTA에서 특히). 사용자가 지정한 fallback인 **개념 클러스터 단위 group gain**이 파라미터를 수십~수백으로 줄이면서 안정성·worst-group을 지키는지 검증해야 v1 대안이 선다. (사용자 지정 fallback — 본 계획에서 실험화.)

## 실험과정
1. SAE latent을 **개념 클러스터 M개**로 군집: (a) decoder weight 코사인 유사도, (b) Broden 개념 라벨(**실험 3의 alive latent 전체 IoU 부산물** 사용 — 후보 40개로는 부족), (c) co-activation 통계 중 택1/앙상블.
2. 클러스터별 **group gain `alpha_g`**(latent은 소속 클러스터 gain 공유) 학습 — 나머지는 실험 5와 동일.
3. **개별 gain vs cluster gain(M∈{32,128,512})** 비교: worst-group acc, 안정성(분산), 파라미터 수, 학습 step 대비 수렴.
4. anchor(FB-anchor)와의 조합 효과도 격자로 확인.

## 실험세팅
### 1. 데이터셋
- Waterbirds, ColoredMNIST(안정성이 드러나는 biased), ImageNet-R(일반화).

### 2. 모델
- `vit_base_patch16_224` + SAE. 개입 파라미터만 개별↔클러스터 교체.

### 3. 하이퍼파라미터
- 클러스터 수 `M ∈ {32,128,512, K(=개별)}`. 군집 방법 3종 비교. optimizer/lr/배치는 실험 5와 동일. anchor `λ ∈ {0,0.1}`.

## 예상결과
- **cluster gain(M 중간값)이 개별 gain 대비 분산↓·worst-group 유지 또는 향상**, 파라미터는 1~2 자릿수↓. 너무 작은 M은 표현력 부족으로 하락, 너무 크면 개별과 수렴 → **적정 M의 안정 구간** 확인. 개별 gain이 불안정한 regime에서 cluster의 이득이 가장 큼.

---

## 실험 8. Continual / recurring 전환 — routing·forgetting·효율

## 해야 하는 이유
스트림이 장기화·재발하면 단일 gain은 도메인 간 간섭·forgetting을 겪는다. **도메인 지문(SAE descriptor) 라우팅 + 도메인당 gain 저장/재로드**(Paper B)가 ReservoirTTA 대비 **메모리·해석성**에서 이득인지, 그리고 그 이득이 스텝 연산으로 상쇄되지 않는지 정직하게 보여야 한다. (plan.md Phase 4 + Paper B, continual 확장.)

## 실험과정
1. recurring 스트림(도메인 A→B→C→A…) 구성. 배치별 **SAE descriptor**(어떤 latent이 켜지나 요약)를 기억된 지문과 거리 비교 → 기존 버킷/신규 버킷/재발 매칭.
2. 재발 도메인은 저장된 `alpha`(또는 `alpha_g`) 재로드, 신규는 새 gain 학습. 혼합 배치는 샘플 단위 클러스터링(FIND식), 완만 drift는 버킷 통계 추종(FreDA식).
3. **forgetting Δ**(복귀 시 정확도 회복), acc-vs-time, **도메인당 메모리·스텝 FLOPs** 를 ReservoirTTA와 대조.

## 실험세팅
### 1. 데이터셋
- mixed·recurring ImageNet-C(재발 순열), 보조로 natural shift 혼합 스트림.

### 2. 모델
- `vit_base_patch16_224` + frozen SAE(공유). 라우터 centroid는 EMA 갱신. 비교: `third_party/ReservoirTTA`.

### 3. 하이퍼파라미터
- 도메인 감지 임계 = source descriptor 거리 상위 분위수(ReservoirTTA식). EMA momentum `0.99`. 도메인당 저장 = sparse gain(1에서 벗어난 성분만). FLOPs는 SAE encode/decode(토큰당 ~19M MAC) 포함해 정직 보고.

## 예상결과
- **도메인당 저장이 ReservoirTTA(모델/LN affine ~74k)보다 1~2 자릿수 가벼움**, 재발 도메인 **forgetting Δ 작음**(gain 재로드). **스텝 FLOPs는 SAE forward로 꼭 싸진 않음** → 메모리·해석성 이득 vs 연산 비용의 정직한 trade-off 표. routing 정확도(도메인 매칭)로 재사용 효과 뒷받침.

---

## 부록: 실험 간 의존/우선순위
- **MVP(먼저)**: 실험 1 → 2 → 3 → 4A → 4B. ("SAE 계기 신뢰 → 메커니즘 성립 → **spurious/causal 분리** → 정적 basis 우위 → 설계 ablation"까지 저비용·저-knob으로 확정.) **4B는 `c_k` 때문에 실험 3에 의존**, 4A는 의존하지 않으므로 실험 3과 병행 가능.
- **메인**: 실험 5(온라인 crossover, default=AdaContrast 원본). **정체성 방어**: 실험 6(프로토콜·메모리 ablation, in-batch 포함). **보강**: 실험 7(안정성 fallback), 8(continual·효율).
- 각 실험은 **실패 시 상위 thesis를 반증**하도록 설계(특히 3·4A). 데이터 확보 필요 항목(ColoredMNIST·ImageNet-R/Sketch/A·ImageNet-9)은 MVP 착수 전 우선 수급(Waterbirds는 보유).
