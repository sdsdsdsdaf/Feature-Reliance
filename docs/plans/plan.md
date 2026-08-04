# 연구 계획 — SAE 기반 Feature Disentanglement × TTA

작성 근거: `paper/summary/` 요약 22편 + 설계 논의. 생성 2026-08-04.

---

## 0. 한 줄 thesis
**"TTA가 어느 feature에 의존하는가"** 라는 질문(DeYO/TACT)에, SAE의 **지속적·monosemantic latent basis** 위에서 답한다. DeYO/TACT는 옳은 일을 **나쁜 basis**(스칼라 / per-batch PCA 부분공간) 위에서 하고 있고, SAE가 그 basis를 갈아끼운다. 적응은 **각 latent에 붙는 학습 가능한 gain `a_k`** 로 하고, 그 gain을 GD로 학습한다. 즉 **feature-reliance TTA를 sparse latent 축 위에서 재정식화**하는 것이 기여이고, DeYO/TACT는 그 coarse special case.

---

## 1. 문제의식 & 왜 SAE / 왜 TTA

### 1.1 DeYO → TACT → SAE 는 같은 아이디어의 basis 업그레이드
| | feature 단위 | basis 성격 | 한계 |
|---|---|---|---|
| DeYO | 스칼라 1개 (PLPD=shape 의존) | 입력 perturbation | 1축뿐, 개념 분해 없음 |
| TACT | 부분공간 1개 (PCA non-causal) | per-batch·직교·**entangled** | monosemantic 아님, batch마다 바뀜, augmentation 가정 의존 |
| **SAE (제안)** | 수천 disentangled latent | 고정·공유·(준)monosemantic | monosemanticity 불완전(§7) |

### 1.2 SAE-FT를 TTA로 옮길 이유 (핵심 논리)
SAE-FT는 fine-tuning에서 "L2 정규화와 정확도 비등, 강점은 해석성"이라고 스스로 인정. **왜 fine-tuning에선 L2로 충분한가?** 라벨 gradient가 "어느 방향으로 갈지"를 이미 알려주기 때문. **TTA엔 라벨이 없어** 방향 결정을 정규화/선택이 떠안음 → **방향 지식의 품질이 처음으로 값을 하는 곳이 TTA.** 그래서 `L2 ≈ SAE-FT` 등식이 TTA에선 깨질 것 = 검증 가능한 motivation.

### 1.3 pseudo-labeling을 써도 되나? → 써라, 근데 그게 곧 SAE가 필요한 이유
pseudo-label은 라벨이 아니라 **모델 자신의 믿음**(self-reinforcing). shift 하에서 예측은 종종 spurious feature에 근거 → pseudo-label = **"확신에 찬 틀린 방향"** 을 인코딩. 위험이 "방향 없음"이 아니라 "확신에 찬 틀린 방향"으로 바뀜 → **방향을 골라내는 필터**가 필요 = SAE의 선택성. (DeYO/TACT/AdaContrast가 문서화한 실패 모드.)
- 조합: `L_tta = L_pseudo + SAE-보존/선택 정규화` (경쟁 아님).
- **경계(정직)**: pseudo-label이 이미 좋은 easy shift에선 SAE 이득 작음. SAE의 값은 **biased/wild/prolonged** regime에서 나옴.

### 1.4 SAE만이 주는 것 (PCA/PLPD 불가)
1. **지속·공유 좌표계** — 스트림 전체에서 같은 latent 추적(프로파일링/라우팅의 공통 축).
2. **개념 수준 선택성** — "latent 37 줄이고 102 유지" 가능(스칼라/부분공간은 불가).
3. **augmentation-free 진단** — shift를 표현 활성 통계로 직접 측정(TACT의 augmentation 설계 가정 불필요).
4. **해석 가능** — 어느 개념이 shift됐는지 읽고 라벨 없이 개입 규칙 결정(MSAE/SAE-mono/Broden).

---

## 2. 방법 설계

### 2.1 스위치(latent)마다 두 점수
SAE는 표현을 수천 개 **개념 스위치(latent)** 로 바꾼다. 스위치 하나 = "이 개념이 있나?"(예: #37 물 텍스처, #102 부리).

- **흔들림 점수 (shift score, `s_k`)** — 새 도메인에서 켜지는 비율 vs source에서 켜지는 비율의 차이. **라벨 불필요**(forward만). *v1에선 명시적으로 안 써도 됨(§2.3).*
  `s_k(D_t) = |firing_k(D_t) − firing_k(source)|`
- **쓸모 점수 (usefulness score, `c_k`)** — source에서 이 스위치를 끄면 정확도가 얼마나 떨어지나. **source 라벨로 오프라인 1회** 측정.
  `c_k = acc(source) − acc(source | latent k off)`

### 2.2 판정 (개념도)
|  | 쓸모 낮음 | 쓸모 높음 |
|---|---|---|
| **흔들림 높음** | 🔴 SPURIOUS → 줄임 | SHIFTED-CAUSAL(예: 안개 속 물체) → 조심히 유지 |
| **흔들림 낮음** | noise → 무시 | 🟢 INVARIANT-CAUSAL → 보존 |

`c_k`는 causal(도메인 불변) 여부의 프록시. "causal feature는 도메인 불변"이라는 TACT/DeYO 공통 가정에 의존(§7).

**이 표는 진단이지 정답이 아니다.** `s_k`·`c_k`로 뽑아놓고 "이게 spurious다"라고 부르면 순환논증이다. 무엇이 spurious인지의 **정답은 따로** 정의해야 하고, Waterbirds가 그걸 준다 — 새를 배경 위에 합성해 만든 데이터셋이라 `place`(배경)가 spurious·`y`(새 종류)가 causal이라는 게 설계상 확정돼 있고 두 라벨이 메타데이터에 있다. 실험 3은 **그 정답을 기준선으로 놓고, 라벨 없는 `s_k`(+`c_k`)가 그걸 복원하는지**를 검정한다(§4.1).

### 2.3 개입 = **latent별 학습 gain `a_k` (GD)** — v1
통계적 on/off 게이팅이 아니라, **각 latent에 붙는 학습 가능한 스칼라 gain `a`** 를 GD로 학습한다(백본·SAE는 frozen).
```
adapt 표현:  z'_k = a_k · z_k                     # a init = 1
주입(residual): h⁺ = h + sd ⊙ W_dec^T((a − 1) ⊙ z) # a=1이면 no-op → 안전 (sd=token_std, 역정규화)
loss:        L = entropy 또는 pseudo-label(x_t)  +  λ · Σ_k c_k (a_k − 1)²
학습:        a (스칼라 K개)만 GD.  백본 + SAE frozen
```
- **`c_k` 정규화의 의미**: 쓸모 높은(causal) latent는 `a_k`를 1에 붙잡아 **보존**, 쓸모 낮은(spurious) latent는 자유롭게 → **GD가 알아서 내림**.
- **온라인 `s_k` 불필요**: GD가 entropy/pseudo를 낮추려고 spurious latent gain을 자연히 내림 → §2.1의 통계·임계·게이팅 knob이 v1에서 사라짐(§3 걱정 해소).
- **collapse 방어**: init=1, `c_k` 정규화, `‖a−1‖` 상한, residual 주입.
- **해석성**: 학습된 `a`를 읽으면 "이 도메인에서 어느 개념을 얼마나 줄였나" 가 그대로 보임.
- **(옵션) forward-only 변형**: 학습 없이 `s_k×c_k`로 gain을 정하는 training-free 버전 — 저비용·해석용, 성능 상한은 낮음.
- **capacity 참고**: gain 재가중은 Tent의 채널 affine보다 표현력이 작을 수 있음(§7). **v1은 gain 단독으로 확정 — 네트워크에 새 모듈을 넣지 않는다.** 부족하면 FB-cluster(개념 클러스터 group gain)·손실 설계로 대응한다.

### 2.4 도메인마다 다름 → 온라인 누적 + 도메인 감지(=라우팅)
- **누적 평균(EMA)** 으로 통계/프로토타입을 천천히 갱신 → 배치 노이즈에 안 흔들림. 도메인이 진짜 바뀔 때만 크게 이동.
- **도메인 감지(라벨 없이)** = 도메인 **지문(descriptor: 어떤 latent가 켜지나 요약)** 을 기억된 지문들과 **거리 비교**: 가까우면 그 버킷, 다 멀면 새 버킷, 예전 것과 매치되면 **재사용(recurring)**. (= ReservoirTTA/FIND 라우팅 = **Paper B**.)
- **도메인별 gain**: 버킷마다 별도 `a` 유지, 재발 도메인은 학습된 `a` 재로드. 혼합 배치→샘플 단위 클러스터링(FIND), 완만한 drift→버킷 통계가 따라가게(FreDA).

### 2.5 무엇이 frozen / 갱신되나
| 구성요소 | test-time | 갱신 |
|---|---|---|
| 백본 body (ViT weight) | frozen | — |
| **latent gain `a`** (유일한 학습 파라미터) | 갱신 | **gradient** (entropy/pseudo + `c_k` reg) |
| **SAE (E, decoder)** | **완전 frozen** | 없음 (측정 계기) |
| prototype·router centroid·(`s_k`) | 갱신 | **EMA/통계** (gradient 아님) |

주의: ViT는 **LayerNorm**(BN 아님) — Tent식 "BN 통계 재보정"은 직접 안 됨. gain `a` 학습이 그 자리를 대신하거나 LN affine과 병행.

### 2.6 SAE 학습에 source 필요? → source-free 준수
SAE는 **배포 전 오프라인·비지도**로 source 표현에 학습. 배포 땐 **frozen SAE + latent sketch(prototype)** 만 들고 감(데이터 아님). "source-free = test 시 source 없음"이지 "본 적 없음"이 아님 → FOA/SAID/SAE-FT/T3A/Ada-ReAlign 전부 동일한 오프라인-source-통계 사용. strict-no-source면 (a) target warm-up/unlabeled proxy로 SAE 학습 or (b) 공개 CLIP SAE 재사용.

### 2.7 효율성 & ReservoirTTA 대비 (셀링 포인트)
| 축 | ReservoirTTA | 이 설계 (latent gain) | 판정 |
|---|---|---|---|
| **도메인당 저장** | specialist 모델(적응 파라미터 복제, ViT LN affine ≈ **~74k**) | gain 벡터 `a`, **sparse(1에서 벗어난 것만)** = 수십~수백 스칼라 | ✅ **한두 자릿수 가벼울 수 있음** |
| **공유(1개)** | 백본 | 백본 + **frozen SAE** | — |
| **스텝당 연산** | 라우팅된 specialist forward | 백본 + **SAE encode/decode**(≈ FFN 1개 분량, 토큰당 ~19M MAC) | ⚠️ **꼭 싸진 않음** — FLOPs 보고 필요 |
| **capacity** | LN 전체(이상) 적응 → 큼 | gain 재가중만 → 작음 | ⚠️ 정직히 감수하는 대가 |
| **해석성** | StyleVec(외부·불투명) | 도메인당 **읽히는 개념-gain** | ✅ |

**포지셔닝**: *ReservoirTTA = 도메인당 **모델** 저장 / 이 설계 = 도메인당 **sparse 개념-gain** 저장.* → 메모리·해석성 ✅, 스텝 연산·용량 ⚠️(대가). 재발 재사용은 gain 재로드로 동일하게 지원.

---

## 3. 하이퍼파라미터 대응 (reviewer 1순위 공격)
TTA 공통 약점(target 튜닝 불가). 목표 = "target 라벨 없이 정하고, 둔감하게".
1. **latent gain GD(v1)로 이미 대폭 완화** — 온라인 `s_k`/임계/게이팅 knob 제거, 남는 건 `λ`(정규화) 정도.
2. **source/가짜-shift 은행에서 분위수로 설정** — 기존 perturbation suite + ImageNet-C/R를 shift 은행으로. 도메인 감지 임계 = source 지문 거리의 상위 분위수(ReservoirTTA식).
3. **entropy로 자동 선택** — `λ`·개입 세기를 "entropy 최소화 / augmentation 일관성"으로 온라인 자동 선택(Tent-C).
4. **민감도 그래프 필수** — 넓은 범위에서 평평함을 증명(AdaContrast/TCA/T3A는 "hyperparameter insensitivity"를 장점으로 셀링).
5. **저-knob 정적 버전부터** — thesis는 정적(오프라인)에서 먼저 증명.

---

## 4. 실험 계획

### 4.1 설계 원칙: 적응 엔진 고정, **적응 파라미터화(basis)만 변수**
공통 TTA 루프(ViT-B/16, loss=entropy/pseudo, 같은 학습량) + 교체형 적응 파라미터:
- **`latent gain GD` (제안)** / `채널 gain` (SAE 미경유, 생 채널 768개에 스칼라 1개씩 — Tent 파라미터화를 동일 주입 지점으로 이식) / `random dictionary gain` (동일 `K`·동일 sparsity, 딕셔너리만 무작위). 주입 지점·손실·학습량 전부 동일, 나머지 동결 → 차이는 오직 **gain을 무엇에 주소지정하나** → "왜 SAE"가 인과적으로 증명.
- **TACT/DeYO는 여기 넣지 않는다**: TACT는 augmentation 집합의 per-sample PCA로 상위 성분을 하드 제거하는 backprop-free **method**이고, DeYO는 샘플 keep/drop **필터**다. 교체할 적응 파라미터가 없으므로 §4.5 baseline에서 method 대 method로 비교한다.

### 4.2 단계 (falsify 싼 것부터)
- **Phase 0** — `SAE_validation.py`로 ViT-B block10 SAE 학습, 재구성·sparsity 확인.
- **Phase 1 (제일 중요)** — **분리 검증**: Waterbirds의 `y`/`place` 라벨을 정답으로 놓고, SAE 축에 **"배경만 보는 feature"와 "새만 보는 feature"가 실제로 따로 존재하는지**를 층화 AUC로 잰다(PCA·무작위 축이 대조군). 이어서 **라벨 없는 `s_k`가 그 정답을 복원하는지** 검정하고, 분리된 feature에 `broden.py` IoU로 개념 이름을 붙인다. 성공=disentanglement 실제 작동(PCA/PLPD 불가 증거). **실패면 thesis 재검토.**
- **Phase 2** — 정적 개입: **(4A) latent gain vs 채널 gain vs random dictionary** head-to-head + **(4B) SAE arm 설계 ablation**(anchor·거리공간·residual) (Waterbirds/ImageNet-R).
- **Phase 3** — online TTA 루프: entropy/pseudo, 전체 baseline·스트림 비교.
- **Phase 4** — prolonged/recurring: 재발 스트림, forgetting(복귀 정확도 Δ), **메모리/FLOPs vs ReservoirTTA**.

### 4.3 데이터 (regime별) — crossover가 곧 증거
| regime | 데이터 | 기대 | 지표 |
|---|---|---|---|
| spurious(biased) | Waterbirds, ColoredMNIST (+ImageNet-9/BG) | **SAE 크게 우세** | worst-group acc |
| natural | ImageNet-R(200), Sketch, ImageNet-A | SAE 우세 | acc/rel-acc |
| corruption(easy) | ImageNet-C 15×5 | **SAE ≈ baseline(정직)** | acc |
| wild/continual | mixed·recurring IN-C | SAE 우세 | acc-vs-time, forgetting Δ |

→ "easy는 Tent/PCA와 비등, biased/wild에서만 SAE 승"의 **crossover** 가 핵심 논거.

### 4.4 통제군 (여기서 논문이 서고 무너짐)

**(4A) basis 통제 — arm 교체**
- **채널 gain control**: SAE를 안 거치고 생 채널 768개에 gain. 모든 토큰에 **동일 적용되는 고정 대각 사상**이라 "개념이 있는 곳에서만 누른다"를 못 함. 여기서 지면 **test time에 SAE가 불필요**하다는 뜻 → 최대 위협 통제군.
- **random dictionary control**: 동일 `K`·동일 sparsity의 **무작위 overcomplete 딕셔너리**에 gain. 조건부성·파라미터 수를 맞춰 **"딕셔너리가 학습된 것이냐"만** 남김. (구 계획의 `PCA-dim gain`은 TACT에 존재하지 않는 구성이라 삭제 — PCA 계수는 부호 있고 조밀해 선택적 억제를 표현할 수 없다.)

**(4B) 설계 통제 — SAE arm 내부** (세 축 모두 SAE arm 위에서만 정의됨)
- **`c_k` 정규화 on/off**: causal 보존이 실제로 기여하는지.
- **L2-to-init control**: `c_k` 가중 대신 순수 `‖a−1‖` 균일. (선택성의 값)
- **거리공간 / residual 주입 on-off**: §2.3 설계의 두 특이점 검증.

### 4.5 baseline / 지표
- baseline(고정 백본): No-adapt, Tent, EATA, SAR, **DeYO**, **TACT**, CoTTA, ReservoirTTA.
- 지표: (biased)worst-group acc, (일반)acc/rel-acc, 스트림 안정성(acc-vs-time), forgetting Δ, **분리 품질(`purity_rate` / 진단 AP)**, **해석성(Broden-IoU, 보조)**, 표현 drift(CKA/JS), **효율(도메인당 메모리·FLOPs)**.

### 4.6 추천 MVP
`Phase 1(Waterbirds 분리 검증)` + `Phase 2(4A: latent gain vs 채널 gain vs random dictionary / 4B: c_k·거리공간·residual ablation)`. 온라인 루프·라우팅 붙이기 전에 **"SAE 축이 (a)spurious와 causal을 실제로 분리하고 (b)라벨 없이도 그 분리를 집어낼 수 있으며 (c)그 축에서의 gain 적응이 생 채널·무작위 딕셔너리보다 낫다"** 를 최소비용·저-knob으로 확인.

---

## 5. 두 논문(A/B)과의 매핑
- **Paper A (feature reliance intervention)**: §2.1–2.3, 2.5 = 진단(`c_k`) + **latent gain GD** 개입/보존. SAE-FT(보존) + DeYO/TACT(진단) 확장.
- **Paper B (SAE reservoir routing)**: §2.4, 2.7 = 도메인 지문 라우팅 + **도메인당 sparse gain 저장**. ReservoirTTA의 StyleVec/모델저장을 SAE descriptor/gain으로 교체.
- 둘은 **같은 SAE latent 좌표계** 공유 → 하나의 시스템.

---

## 6. 각 선행연구 대비 우리 방법이 해결하는 것
| 선행연구 | 그들의 한계 | 우리 방법이 해결/개선하는 것 |
|---|---|---|
| [DeYO](../../paper/summary/DEYO_summary.md) | shape 1축(PLPD), 입력 perturbation 간접 | monosemantic **다축**에서 causal-vs-spurious를 **개념별로** 진단·개입 |
| [TACT](../../paper/summary/TACT_summary.md) | per-batch PCA 부분공간(entangled), augmentation 사전지식 필요 | **고정·공유 monosemantic basis**, augmentation-free, **개념 선택적** gain |
| [SAE-FT](../../paper/summary/SAE-FT_summary.md) | **train-time·source 라벨** 필요, 균일 보존(L2와 비등) | **test-time·무라벨**로 이식, `c_k` 가중 **선택적** — 방향 지식이 값을 하는 TTA로 |
| [VS2](../../paper/summary/VS2_summary.md) | **고정** γ steering, task-정렬 없음 | **학습 gain(GD)** 로 도메인별 자동 조정 + `c_k` 보존 |
| [Tent](../../paper/summary/T3A_summary.md) 참조/EATA/SAR | 채널 affine(**entangled**), 방향 모름 → wild서 spurious drift | **latent gain(monosemantic)** + `c_k`로 causal 보존해 **방향 제약** |
| [ReservoirTTA](../../paper/summary/ReservoirTTA_summary.md) | 도메인당 **모델 저장**(무겁), **StyleVec**(외부·불투명) 라우팅 | 도메인당 **sparse concept-gain**(가벼움·해석), **SAE descriptor**(내부·결정 정렬) |
| [FIND](../../paper/summary/FIND_summary.md)/[FreDA](../../paper/summary/FreDA_summary.md) | 혼합 분리하되 **정규화/주파수(비의미)** 축 | **의미 latent** 로 분리·개입, 진단·라우팅·개입 **통합** |
| [T3A](../../paper/summary/T3A_summary.md)/[AdaNPC](../../paper/summary/AdaNPC_summary.md) | prototype/memory 라우팅 but **dense feature** | **monosemantic latent descriptor** 라우팅 + **개입까지** 확장 |
| [TiME](../../paper/summary/TiME_summary.md)/[MoETTA](../../paper/summary/MoETTA_summary.md)/[MoASE++](../../paper/summary/MoASE++_summary.md) | expert/sparsity 라우팅 but **불투명** | **해석 가능한 sparse SAE 축**에서 라우팅+개입, 도메인당 경량 gain |

### 6.1 vs TENT / DeYO / TACT — 장단점 (정직)

| baseline | 우리 장점 | 우리 단점 |
|---|---|---|
| **TENT** (entropy + affine) | 해석 가능 · spurious **겨냥 억제**(Tent는 wild서 collapse) · routing·경량 저장 · monosemantic knob | **압도적으로 복잡**(Tent는 5줄) · SAE forward 비용 · 오프라인 SAE+`c_k` 필요 |
| **DeYO** (entropy + PLPD 선택) | **모델 내부 진단**(PLPD 3결함 회피: confound / 도메인고정 "shape=causal" / coarse 1축) · **개념별 개입**(DeYO는 샘플 keep/drop만) · 해석·routing | DeYO가 **더 단순·학습불필요 진단** · 우리 진단(`c_k`·invariance)도 자체 가정 · DeYO는 검증됨 |
| **TACT** (PCA trim, bp-free) | **monosemantic** vs entangled 부분공간 · **고정·공유 basis**(routing 가능) · **학습 gain(적응형)** | TACT는 **backprop-free**(싸고 안정) · 이론근거·multi-modal · **SAE가 병목** |

- **전체 단점**: 복잡성·부품 수 / **SAE 단일 의존**(monosemantic·OOD 재구성) / 스텝 연산 / 증명 부담 / **더 날카로운 shortcut 지렛대**(monosemantic이라 Tent보다 shortcut을 더 잘 만듦 → safeguard 필수)
- **전체 장점**: 해석성 / 개념별·도메인 적응 개입 / 지속 basis→routing·경량 / 내부 진단
- **전략**: 정확도(easy corruption ≈ 비등 → "TENT/TACT로 되잖아" 함정) 말고 **해석·hard-regime·worst-group·경량 라우팅**에서 승부.

---

## 7. 열린 리스크 (정직)
- **shift ≠ spurious**: `c_k`로 완화하나 "causal feature는 도메인 불변" 가정에 의존. 의미가 도메인 간 바뀌면 `c_k` 전이 실패. **실험 3이 이걸 실측한다** — 편향 source에서 잰 `c_k^train`과 그룹 균형에서 잰 `c_k^bal`을 병기하면, 95% 상관 하에서 `c_k`가 배경을 causal로 오판하는 폭이 숫자로 나온다.
- **진단이 정답을 복원 못 할 위험**: 분리가 SAE 축에 존재해도 **라벨 없는 `s_k`로 그걸 집어내지 못하면** "test time에 spurious를 겨냥한다"는 주장이 무너진다. thesis 자체는 살지만 v1의 anchor 설계를 다시 짜야 한다. 실험 3의 명제 (B)가 이 전용 게이트다.
- **SAE monosemanticity 불완전** (MSAE: feature splitting/absorption) → invariant/spurious 분리가 항상 깨끗하진 않음 = basis 우위의 상한. 실험 3의 `sel_j` 분포가 0 근처 단봉이면 이 리스크가 실현된 것.
- **SAE OOD 재구성 취약** — source 학습 SAE가 shift에서 무너지면 gain·진단 오염 → VS2식 `FVU-gate` 필요.
- **capacity**: gain 재가중만으론 hard shift에 부족 가능. **v1은 gain 단독**(네트워크에 새 모듈 추가 없음)이 전제이므로, 부족분은 FB-cluster·손실 설계로 대응하고 남는 격차는 정직하게 보고한다.
- **연산**: SAE forward(스텝당 ~FFN 1개)로 메모리 이득이 연산으로 상쇄될 수 있음 → 정직한 FLOPs 보고.
- **핵심 증명 부담**: **채널 gain·random dictionary** 대비 이겨야 "왜 SAE"가 성립. 특히 채널 gain에 지면 test time에 SAE가 불필요하다는 결론이 된다.

---

## 8. 참고 선행연구 (요약 링크)
- 진단·개입: [DEYO](../../paper/summary/DEYO_summary.md), [TACT](../../paper/summary/TACT_summary.md), [SAE-FT](../../paper/summary/SAE-FT_summary.md), [VS2](../../paper/summary/VS2_summary.md)
- SAE 품질/해석: [MSAE](../../paper/summary/MSAE_summary.md), [SAE-monosemantic](../../paper/summary/SAE-monosemantic_summary.md), [SAID](../../paper/summary/SAID_summary.md)
- 라우팅/prolonged: [ReservoirTTA](../../paper/summary/ReservoirTTA_summary.md), [FIND](../../paper/summary/FIND_summary.md), [FreDA](../../paper/summary/FreDA_summary.md), [TiME](../../paper/summary/TiME_summary.md), [MoETTA](../../paper/summary/MoETTA_summary.md), [Ada-ReAlign](../../paper/summary/Ada-ReAlign_summary.md)
- prototype/memory: [T3A](../../paper/summary/T3A_summary.md), [AdaNPC](../../paper/summary/AdaNPC_summary.md), [AdaContrast](../../paper/summary/AdaContrast_summary.md)
- 전체 표: [paper.md](../../paper/summary/paper.md)
