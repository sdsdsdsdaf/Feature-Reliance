# Plans — SAE-latent-gain × AdaContrast TTA

Purpose: `experiment_plan.md`(= product/spec contract, SSOT)를 실행 가능한 task ledger로 전개한다. thesis = "TTA feature-reliance를 frozen SAE latent 축 위에서 학습 gain `alpha`로 재정식화하고, easy shift에선 Tent/TACT와 비등·biased/wild에선 우세한 crossover를 보인다." precedence: `experiment_plan.md`(spec) > `plan.md`(설계 근거) > 본 Plans.md(task).

- Spec SSOT: [experiment_plan.md](experiment_plan.md) / 설계 근거: [plan.md](plan.md)
- 공통 모델: `vit_base_patch16_224` + SAE `outputs/reservoir_sae/vit_b_sae.pt`(block10·patch·`K=12288`, expansion 16)
- unknown data contract: 미확인 사실은 `unknown`으로 명시(`not_observed != absent`).
- **구현 계약**: [contracts/](contracts/) — 각 Phase의 시그니처·절차·산출 아티팩트 스키마·판정 규칙. 착수 전 해당 절을 읽는다.

## Spec delta
- `experiment_plan.md` 갱신 반영: (1) `### 가정 & source-free 준수` 서브섹션 추가, (2) ImageNet-C = precomputed(Zenodo 2235448) 고정·재생성 금지 + 4-extra HP holdout 규약, (3) 실험 1에 SAE-source ablation, (4) 적응 프로토콜 default=AdaContrast 원본 + buffer/pass sweep 축, (5) 실험 6(프로토콜·메모리 ablation) 신설·후속 재번호.
- root `spec.md` 없음 → 본 연구의 product contract는 `experiment_plan.md`를 SSOT로 채택(consumer 승인/수정만).
- **2026-08-04 개정 (실험 4)**: (1) 실험 4를 **4A(basis head-to-head) / 4B(SAE arm 설계 ablation)** 로 분리 — 통제 3축(`c_k`·거리공간·residual)이 SAE arm 위에서만 정의돼 5×3 격자가 성립하지 않았음. (2) arm 목록에서 **`PCA-dim gain`·`PLPD-select` 삭제** — TACT는 backprop-free method, DeYO는 샘플 필터로 교체할 적응 파라미터가 없음 → 실험 5 baseline으로 이관. (3) `채널 affine(Tent)` → **`채널 gain`**(동일 주입 지점, 단일 변수화), **`random dictionary gain`** 신설. (4) 실험 4 손실을 **AdaContrast로 고정**(거리공간 축 성립 조건). (5) 주입 수식에 `token_std` **역정규화 반영**. (6) 사실 정정: expansion **16**(32 아님), Waterbirds **보유**(`data/waterbirds`).
- **2026-08-05 개정 (실험 3 재정식화)**: 실험 3이 **순환논증 구조**였음 — `s_k`·`c_k`로 후보를 뽑아놓고 그걸 spurious라 부른 뒤 Broden IoU로 확인하는 형태라, "무엇이 spurious인가"의 정답이 진단 자신에게서 나왔다. 이에 따라 (1) **정답을 Waterbirds `y`/`place` 라벨로 분리** — `y`/`place` **층화 AUC**로 `spur_j`/`caus_j`를 정의(층화 없이는 95% 상관 때문에 배경 탐지기가 causal로 보임). (2) **주 지표를 `purity_rate`(분리도)로 교체**, Broden IoU는 **해석 도구로 강등**(판정 게이트 아님). (3) 대조축 공정성 문제 해소 — AUC가 순위 기반이라 **부호·스케일·마스크 밀도 정합이 불필요**해지고(`ConceptAxis`의 PCA `2d` 부호 처리 삭제 → `d`), **permutation max-null 임계**가 pool 크기(SAE ~6,000 vs PCA 768) 비대칭을 자동 보정, 비율 지표라 pool 크기가 상쇄. (4) `s_k` 정의 수정 — 구 정의 `|firing(place=1)−firing(place=0)|`은 **정답 라벨을 진단에 흘려 넣는 것**이라 폐기하고 plan.md §2.1의 도메인 단위 정의(`|firing(target)−firing(source)|`)로 복귀. (5) **`c_k^train`/`c_k^bal` 병기** — 편향 source에서 `c_k`가 배경을 causal로 오판하는 폭이 plan.md §7 리스크의 실측값. (6) **판정 명제를 (A) 분리 / (B) 진단 타당성 / (C) 해석성으로 분해** — (A) 실패 = thesis 재검토, (A) 통과·(B) 실패 = thesis 생존하나 `c_k` anchor 설계·"test time에 spurious 겨냥" 서술 전면 수정(T1.4b `anchor=ck` 근거 상실), (C)는 판정 미참여. (7) T3.1 `broden` 군집이 요구하는 **alive latent 전체 IoU**를 T1.3 부산물(`full_latent_iou.json`)로 명시.
- **2026-08-05 실측 (M7 FVU 게이트 전제 붕괴 관측)**: 정식 T1.1 전 예비 probe([scripts/probe_fvu_shift.py](../../scripts/probe_fvu_shift.py), 셀당 128장 × 3 corruption × sev{1,3,5})에서 **FVU가 severity에 대해 단조 증가가 아니라 단조 감소**했고, OOD 9개 셀 전부가 in-domain보다 **낮았다**(0.47~0.57×). 임계 초과 셀 0개 → **게이트가 발동할 수 없다.** 손상 이미지는 고주파가 뭉개져 활성이 평범해지고 딕셔너리가 맞추기 더 쉬워지기 때문으로 보인다(L0도 793→650~670 동반 하락). 이에 따라 (1) **M7을 `blocked`로 전환**하고 Depends에 T1.1 추가 — T1.1이 확정하기 전엔 착수하지 않는다. (2) **T2.2의 Depends에서 M7을 조건부로 강등**(게이트 없이도 진행 가능). (3) 부수 발견: **in-domain FVU 기준선이 데이터에 따라 5배 흔들린다**(ImageNet-1k val `1.6e-3` vs imagenette `3.4e-4`) → T1.1은 게이트 임계를 뽑은 분포를 반드시 명시해야 한다. (4) 확정 시 이건 **논문에 실을 부정 결과**다 — VS2식 FVU 게이트가 ViT+PatchSAE 계열에 그대로 이식되지 않는다는 보고. 재구성 품질 자체(cosine 0.9998)는 오히려 매우 좋으므로 "SAE가 나쁘다"는 결론이 아니다. 계기를 `L0` 수십 급으로 올린 뒤 재측정할 여지는 남긴다.
- **2026-08-04 개정 (방법 구현)**: (1) **Phase M 신설** — 기존 ledger가 실험 task만 담고 있어 논문에서 주장할 방법 자체의 구현(SAE 추론 wrapper·gain 주입·AdaContrast 손실 스택·aug·anchor·`c_k`·FVU gate·runner)이 T0.3/T2.1 두 줄에 묻혀 있었음. M1~M8로 분해하고 실험 task 의존을 전부 재배선. (2) **T2.1을 baseline 스위트로 교체** — 실험 5가 baseline 9종을 요구하는데 구현 task가 없었음(보유는 `third_party/ReservoirTTA`와 BN 전용 Tent뿐). (3) **adaptor 하이브리드 언급 전면 삭제**(plan.md 4곳·experiment_plan.md 3곳) — v1은 **gain 단독**이고 네트워크에 새 모듈을 넣지 않는 구조. capacity 부족분은 FB-cluster·손실 설계로 대응하고 남는 격차는 정직하게 보고.

## team_validation_mode: manual-pass
subagent 대량 spawn 대신 5관점 분리 평가(세션에서 대부분 합의된 설계라 비용대비 적정).
- **Product**: 목표=crossover(biased/wild 우세, easy 비등)·해석성·경량 라우팅. 정확도 정면승부 회피는 의도된 포지셔닝. ✅
- **Architecture**: frozen backbone+SAE, `alpha`만 학습, block10 이후 tail만 grad → 저메모리·구현 단순. ✅
- **Security**: secret read 없음. 위험은 **공개 데이터셋 다운로드(external-fetch)** 뿐 → 사전확인에 선언. 파괴적 조작 없음.
- **QA**: 모든 DoD를 산출물(표·곡선·assert 통과)로 검증가능화. MVP(1~4)가 싸게 falsify.
- **Skeptic**: 리스크(=대응 실험) — `c_k` causal 가정(실험 3), SAE OOD(실험 1 FVU 게이트), gain capacity(실험 2), TTA↔SFDA 정체성(실험 6), 개별 gain 노이즈(실험 7). 각 리스크에 전용 게이트 존재. ✅

## Phase 0 — Prep (착수 전 필수)

→ 구현 계약: **[phase0.md](contracts/phase0.md)**

| Task | 내용 | DoD | Depends | Status |
|---|---|---|---|---|
| T0.1 데이터 수급 | `[lane:gate][tdd:skip:data-acquisition]` ColoredMNIST·ImageNet-R/Sketch/A·ImageNet-9(BG) 확보(**Waterbirds는 보유** `data/waterbirds`). ImageNet-C는 보유분(precomputed) 사용, **재생성 금지** | 각 데이터셋 loader가 배치 반환 + 샘플 수·클래스 수 로그 출력. ImageNet-C loader가 `corruption/severity/class` 경로에서 로드 | - | cc:완료 [75ddd31 — `Utils/datasets.py` `build_dataset()`, 29 tests, sketch만 `missing`] |
| T0.2 lint/format baseline | `[lane:fast][tdd:skip:setup]` ruff/black baseline 확인·설정 | `ruff check`·`black --check` 통과 or 설정 파일 커밋 | - | cc:완료 [1f41202 — `pyproject.toml` ruff `select=["F"]`, black 미도입(계약대로), 기존 11건은 per-file-ignores 기준선] |

## Phase M — 방법 구현 (논문에서 주장할 방법 그 자체)

> **원칙: 학습 파라미터는 gain 벡터 하나뿐.** 백본·SAE는 frozen이고 **네트워크에 새 모듈(adaptor 류)을 추가하지 않는다.** M2의 "basis 스위치"는 학습 파라미터가 없는 **소프트웨어 인터페이스**이지 신경망 모듈이 아니다(기존 `Model/Adaptor.py`와 무관).
> 현재 코드 실태: SAE **학습** 코드만 존재(`Utils/SAE_utils.py`). AdaContrast 계열(memory bank·k-NN soft voting·contrastive·weak/strong aug)은 **전무**. `Model/Loss.py`·`Utils/transfrom.py`는 이전 perturbation 프로젝트 산물이라 재사용 불가. `reservoir_sae/tta.py`는 **BN 전용 Tent**라 ViT(LN)에 그대로 못 씀.

| Task | 내용 | DoD | Depends | Status |
|---|---|---|---|---|
| M1 SAE 추론 wrapper | `[lane:gate][tdd:required]` frozen SAE encode/decode + `token_mean`/`token_std` 정규화 왕복, FVU·L0 산출 API | 왕복 후 재구성이 `SAE_validation` 수치와 일치 · frozen weight `.grad is None` | - | cc:완료 [89d5646 — FrozenSAE(Model/sae_runtime.py), 16 tests, FVU=3.45e-4/L0=502.5 vs ref FVU~4e-4/L0~497 on imagenette 200장] |
| M2 gain 주입 모듈 + basis 스위치 | `[lane:gate][tdd:required]` `h⁺ = h + delta(code, gain)` 주입, block10 hook, 상류 `no_grad`+`detach`, tail만 grad, optimizer=gain only. basis 3종을 `encode(h)->code` / `delta(code,gain)->dh` 두 함수로 교체 가능하게(`latent`/`채널`/`random dictionary`) | ✅ smoke: `gain.grad≠0` · frozen `.grad is None` · **`gain=1`이 무개입 forward와 비트 동일**(residual on이면 delta가 정확히 0 — 재구성 오차가 개입하지 않음) · 3종이 동일 루프에서 교체 · `residual` on/off 전환(off는 딕셔너리 basis만, 채널은 즉시 ValueError) · **gain 외 학습 파라미터 0개 assert** | M1 | cc:완료 [daca0da+779fe2c — Model/gain_basis.py·intervention.py, 21 tests, 실 ckpt·GPU에서 3 basis 전부 비트동일 no-op 확인] |
| M3 AdaContrast 손실 스택 | `[lane:gate][tdd:required]` memory bank(클래스 균형 큐) · k-NN soft-voting pseudo-label · contrastive(same-pseudo negative 제외) · diversity(옵션 플래그). **거리는 backbone feature `h'`에서**. `buffer`·`pass`를 config 축으로 | 원본 레시피 재현 + `buffer∈{0,256,2048,full}`·`pass∈{1,2,5}`가 코드 변경 없이 전환 | M2 | cc:완료 [f7235a8 — Model/adacontrast.py, 29 tests. `-inf` 전체마스킹 행·`capacity=0` in-batch 경로 검증] |
| M4 weak/strong augmentation | `[lane:fast][tdd:required]` AdaContrast 원본 aug 정책(기존 `Utils/transfrom.py`는 perturbation용이라 신규 작성) | weak/strong 배치가 동일 샘플에서 생성·shape 검증 | - | cc:완료 [9faf73a+713c37e — Utils/tta_transforms.py, 7 tests, 정규화 상수는 백본 인자로 해결] |
| M5 anchor 정규화 항 | `[lane:fast][tdd:required]` `{off, 균일 λ‖g−1‖², c_k 가중 λΣc_k(g_k−1)²}` 스위치 | 3 모드 전환 + `λ=0`이 off와 수치 동일 | M2, M6 | cc:완료 [c09128a — Model/anchor.py, 21 tests. 음수 c_k는 clamp(min=0), 가중치 평균을 1로 재정규화(4B 비교 가능성 확보)] |
| M6 `c_k` 진단 모듈 | `[lane:gate][tdd:required]` offline·source 라벨. `c_k = acc(source) − acc(source \| latent k off)` | `c_k` 벡터 산출·저장 + latent off 경로가 재구성 기여만 제거하는지 검증 | M1, **M2** | cc:완료 [b1c3d1e — Model/diagnostics.py, 13 tests. 캐시된 tail 재실행 + delta 선형성 probe. ⚠️ c_k는 1e-3 자릿수라 n_images가 곧 분해능] |
| M7 FVU gate | `[lane:gate][tdd:required]` VS2식 런타임 게이트. **⛔ T1.1이 전제를 반증했다 — 구현하지 않는다** | (해당 없음 — 게이트가 발동할 수 없음이 실측으로 확정) | M1, M2, T1.1 | **Optional 강등 [T1.1 8c56764]** — 75셀 전부 임계 미달, severity 단조증가 0/15. 계기를 `L0` 수십 급으로 올린 뒤 재측정할 때만 부활 |
| M8 online TTA runner | `[lane:gate][tdd:required]` 스트림 배치별 1-step 적응, 평가 프로토콜 2종(online 누적 / adapt-then-eval) 병기 | 1 corruption smoke에서 no-adapt 대비 acc 개선 로그 + 두 프로토콜 수치 동시 출력 | M2, M3, M4 | cc:완료 [08705aa — Model/tta_runner.py, 23 tests. gaussian_noise/5 smoke: final_acc 0.6700(no-adapt) -> 0.6967(적응)] |

→ 구현 계약: **[contracts/phaseM.md](contracts/phaseM.md)** (class/function 시그니처·동작·불변식·엣지 케이스)

전역 불변식 (전 모듈 테스트에 포함):

```python
trainable = [n for n, p in module.named_parameters() if p.requires_grad]
assert trainable == ["gain"], f"gain 외 학습 파라미터 발견: {trainable}"
```

## Phase 1 — MVP (실험 1~4, 저비용 falsify 게이트)

→ 구현 계약: **[phase1.md](contracts/phase1.md)**

| Task | 내용 | DoD | Depends | Status |
|---|---|---|---|---|
| T1.1 실험 1 — SAE 계기+FVU+source ablation | `[lane:gate][tdd:skip:analysis]` in-domain vs OOD 재구성/sparsity, FVU 게이트 임계, SAE-source(source/proxy/public) ablation | FVU-vs-severity 곡선 + 게이트 임계값 + source/proxy/public 재구성 비교표 산출·저장 | M1 | cc:완료 [8c56764 — 75셀×512장 실행. **verdict=fail**: cells_above_gate 0/75, severity 단조증가 0/15. proxy·public ablation 미측정(ckpt 없음)] |
| T1.2 실험 2 — 개입 sanity+capacity | `[lane:gate][tdd:required]` no-op 항등성·grad 경로 assert·gain vs LN affine capacity 상한 | assert 3종 통과 + gain/affine capacity 상한표 산출. 격차 기록(대가로 보고할지, FB-cluster로 대응할지 판단) | M2 | cc:TODO |
| T1.3 실험 3 — spurious/causal 분리 | `[lane:gate][tdd:skip:analysis]` ①정답: Waterbirds `y`/`place` 층화 AUC(`spur_j`/`caus_j`) + permutation max-null 임계 → 축별 `purity_rate`(SAE vs PCA vs random). ②진단: 라벨없는 `s_k`(+`c_k^train`/`c_k^bal`)가 ①을 복원하는지 AP로 검정. ③해석: 순수 후보 40개에 `broden.py` IoU로 개념 이름 부여(+ 실험 7용 alive latent 전체 IoU 부산물) | (A) 축 3종 `purity_rate`·`sel_j` 히스토그램·permutation p값 표 산출 + (B) `s_k`→순수-spurious 랭킹 AP와 무작위 baseline 대비표, `corr(c_k^train/^bal, caus_j)` 산출 + (C) 후보 Broden 개념·category 분포 표 산출. `verdict`는 (A)·(B) 두 조건으로 자동 판정 | T0.1, M1, M6 | cc:TODO |
| T1.4a 실험 4A — 정적 basis head-to-head | `[lane:gate][tdd:skip:analysis]` `latent gain` vs `채널 gain`(SAE 미경유, 768) vs `random dictionary gain`(동일 K·동일 sparsity). 주입 지점·손실(AdaContrast 고정)·step 동일, **lr은 arm별 스윕 후 arm별 best 보고**, anchor는 전 arm 균일 고정 | 3 arm worst-group/acc 표(arm별 best-lr) + CKA/JS drift 병기. **채널 gain 대비 우위 / random dictionary 대비 우위** 판정 산출 | T0.1, M2, M3, M4, T1.2 | cc:TODO |
| T1.4b 실험 4B — SAE arm 설계 ablation | `[lane:gate][tdd:skip:analysis]` `latent gain` 고정 후 3축 factorial: anchor{off/균일 L2/`c_k` 가중} × 거리공간{`h'`/`z`/logit} × residual{on/off} | 3축 ablation 표 + `‖alpha−1‖` 궤적·시드 분산. 거리공간 `h'` 우위 판정, `c_k` 가중이 균일 L2 대비 기여하는지 판정 산출 | M2, M3, M4, M5, M6, T1.3 | cc:TODO |

## Phase 2 — Main (온라인 crossover)

→ 구현 계약: **[phase2.md](contracts/phase2.md)**

| Task | 내용 | DoD | Depends | Status |
|---|---|---|---|---|
| T2.1 baseline 스위트 | `[lane:gate][tdd:required]` 실험 5가 요구하는 9종을 **동일 백본(ViT-B/16)·동일 스트림**에서 재현: No-adapt·Tent·EATA·SAR·DeYO·TACT·CoTTA·AdaContrast·ReservoirTTA. 현재 보유는 `third_party/ReservoirTTA`와 **BN 전용** `reservoir_sae/tta.py`뿐이라 **ViT(LN)용 재구현 필요** | 9종이 동일 인터페이스로 1 corruption smoke 통과 + 공개 보고치와 자릿수 일치 확인 | T0.1, M8 | cc:TODO |
| T2.2 실험 5 — 메인 crossover | `[lane:release][tdd:skip:analysis]` 전 baseline × regime(spurious/natural/corruption/wild). HP는 4-extra holdout에서 선택 | regime별 acc/worst-group/acc-vs-time 표 + baseline 비교 + HP 민감도 그래프(평탄) 산출 | T2.1, T1.4a (~~M7~~ — 게이트 전제가 무너지면 M7 없이 진행) | cc:TODO |
| T2.3 실험 6 — 프로토콜·메모리 ablation | `[lane:gate][tdd:skip:analysis]` `buffer∈{0,256,2048,full}`×`pass∈{single,multi-epoch}` sweep, in-batch 성립·TTA↔SFDA 위치 | buffer/pass sweep 표 + `buffer=0·single-pass`(순수 TTA) 성립 여부 판정 산출 | M3, M8 | cc:TODO |

## Phase 3 — Extend (보강·확장)

→ 구현 계약: **[phase3.md](contracts/phase3.md)**

| Task | 내용 | DoD | Depends | Status |
|---|---|---|---|---|
| T3.1 실험 7 — clustering fallback | `[lane:gate][tdd:skip:analysis]` latent→M 개념 클러스터 group gain(FB-cluster). 군집 3방식(decoder cos/Broden/co-activation) | 개별 vs `M∈{32,128,512}` 분산·worst-group·param수 표 산출. 적정 M 안정구간 식별 | M2, M8 | cc:TODO |
| T3.2 실험 8 — continual/recurring (Paper B) | `[lane:release][tdd:skip:analysis]` SAE descriptor 라우팅 + 도메인당 gain 저장/재로드, forgetting·mem/FLOPs vs ReservoirTTA. 기존 `reservoir_sae/simple_reservoir.py`(PrototypeReservoir) 재사용 | forgetting Δ·acc-vs-time·도메인당 mem/FLOPs 대조표 + routing 정확도 산출 | M8, T2.1 | cc:TODO |

## 우선순위 분류
- **Required**: T0.1, **M1~M6·M8(방법 구현)**, T1.1, T1.2, T1.3, T1.4a, T1.4b, T2.1, T2.2, T2.3 — thesis 성립의 최소 골격(방법 구현→진단→메커니즘→basis→crossover→정체성). **Phase M이 없으면 Phase 1·2가 전부 착수 불가.**
- **Recommended**: T0.2(lint 위생), T3.1(안정성 fallback).
- **Optional(강등 확정)**: **M7** — FVU 게이트. T1.1이 전제를 반증했다(cells_above_gate 0/75, severity 단조증가 0/15). **구현하지 않고 부정 결과로 보고한다.** 계기를 `L0` 수십 급으로 올린 뒤에만 재고.
- **Optional**: T3.2(Paper B continual — 사실상 2번째 논문, 여력 시).
- **Reject**: ImageNet-C on-the-fly 재생성(버전 드리프트로 baseline 비교 불가) — precomputed 고정으로 대체.

## 사전확인
- 사항: external-fetch — Waterbirds·ColoredMNIST·ImageNet-R/Sketch/A·ImageNet-9(BG) 공개 데이터 다운로드(HF/kaggle/zenodo/공식 배포처)
  이유: T0.1 데이터 수급 및 실험 3~8 실행에 필요
  scope: Phase 0 / Task T0.1
- 사항: external-fetch(옵션) — 공개 pretrained SAE 다운로드
  이유: T1.1 SAE-source ablation의 public-SAE 조건
  scope: Phase 1 / Task T1.1
- (secret-read / destructive 조작 없음)

## 다음 액션
새로운 세션의 起動 커맨드: `claude`
起動 후 최초 입력: `/harness-work M1`
향하는 장면: 전체 병목은 **Phase M(방법 구현)** 이다 — 지금 코드에는 SAE 학습기만 있고 제안 방법(gain 주입·AdaContrast 손실·runner)이 없어 Phase 1·2가 통째로 착수 불가. `M1 → M2`가 서면 T1.1·T1.2 게이트가 바로 붙는다. 데이터(T0.1)와 augmentation(M4)은 의존이 없으므로 다른 세션에서 병행 가능.
