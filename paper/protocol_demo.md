# SAE Validation Demo: 논문 방식 기반 실험 코드 계획

이 문서는 `paper/refer/`의 논문 방식 중 현재 보유한 ViT-B SAE weight로 바로 실행 가능한 항목만 골라 demo 실험 코드로 옮기기 위한 계획이다. 기존 repo의 validation 코드를 기준으로 재구성하지 않고, 논문에서 사용한 검증 절차를 실험 단위로 그대로 가져오는 것을 원칙으로 한다.

## 0. 필요한 데이터셋

### 필수 데이터셋

1. `data/imagenette2/train`
   - 용도: SAE 학습 weight와 동일한 train distribution.
   - 사용 위치: token normalizer 재계산, top activating reference pool, class-level latent statistics.
   - 논문 대응: PatchSAE의 ImageNet train reference image pool, VLM-SAE의 ImageNet top activating image pool.

2. `data/imagenette2/val`
   - 용도: 최종 validation split.
   - 사용 위치: reconstruction/fidelity, sparsity, monosemanticity score, top-k reference image, heatmap, class alignment, intervention.
   - 논문 대응: PatchSAE의 classification benchmark validation, Matryoshka의 ImageNet-1K image modality evaluation.

3. `outputs/SAE_validation/grid_search`
   - 용도: 현재 실험에 사용할 SAE weight 저장소.
   - 사용 파일:
     - `grid_summary.csv`
     - `trial_XXXX/config.json`
     - `trial_XXXX/summary.json`
     - `trial_XXXX/best_sae_state.pt`

### 필수 입력 파일

1. `protocol_demo/concepts/user_concepts.txt`
   - 용도: CB-SAE식 concept coverage audit에 사용할 관심 feature/concept 목록.
   - 최소 구성: Imagenette class name, Broden concept name, corruption/domain keyword, 직접 검증하려는 feature name.
   - 논문 대응: CB-SAE의 user-specified concept set `C_user`.

### 선택 데이터셋

1. `data/broden`
   - 용도: feature-latent alignment의 강한 검증.
   - 사용 위치: Broden concept mask와 SAE latent patch activation mask의 IoU 계산.
   - 논문 대응: Network Dissection의 Broden dataset 기반 unit-concept alignment.
   - 출처: Network Dissection: Quantifying Interpretability of Deep Visual Representations, Section 2.1/2.2.
   - 참고 링크: https://arxiv.org/abs/1704.05796

2. `data/imagenet-c`
   - 용도: OOD/corruption별 latent activation 변화 확인.
   - 주의: ReservoirTTA routing 실험으로 쓰지 않는다.
   - 논문 대응: PatchSAE의 cross-dataset generalization, UniversalSAE의 OOD generalization 아이디어 중 “단일 모델 activation generalization”에 해당하는 부분만 사용.

3. `data/imagenette2/noisy_imagenette.csv`
   - 용도: noisy label metadata가 필요할 때만 사용.
   - demo 1차 구현에서는 제외한다.

## 1. 이번 demo의 범위

### 사용 가능한 논문 방식

현재 weight는 `vit_base_patch16_224`의 block 10 patch-token activation에 대해 학습된 `VanillaL1SAE`이다. 따라서 다음 논문 방식만 사용한다.

1. PatchSAE 방식
   - top activating images
   - activation frequency / mean activation
   - label entropy
   - patch-level heatmap
   - image/class-level aggregation
   - top-k latent masking 또는 intervention

2. VLM-SAE Monosemanticity 방식
   - top-16 image grid
   - activation-weighted pairwise similarity 기반 MonoSemanticity score
   - MS score 상위/하위 latent 비교
   - 사람 검토용 image grid 생성

3. Matryoshka SAE의 평가 metric 방식
   - FVU / EVR / cosine reconstruction
   - L0 / active latent count
   - dead latent ratio
   - decoder orthogonality
   - sparsity-fidelity 비교

4. Network Dissection / Broden 방식
   - densely labeled concept mask와 latent activation mask의 IoU
   - object / part / material / texture / color / scene concept label 부여
   - top-image coherence보다 강한 feature-latent alignment evidence

5. CB-SAE 방식 중 현재 weight로 가능한 진단
   - latent별 interpretability score와 steerability score 분리
   - `high I/high S`, `high I/low S`, `low I/high S`, `low I/low S` quadrant 분류
   - user-specified concept set coverage audit
   - low-utility latent pruning 후보와 missing concept 목록 산출

### 제외할 논문 방식

1. ReservoirTTA 관련 항목
   - reservoir routing
   - specialist assignment
   - recurring domain routing consistency
   - online TTA update
   - stream prototype update

2. UniversalSAE
   - 이유: UniversalSAE는 여러 모델의 activation을 shared concept space로 공동 학습해야 한다.
   - 현재 보유 weight는 단일 ViT-B SAE이므로 cross-model reconstruction, firing entropy across models, co-fire proportion across models는 수행하지 않는다.

3. MP-SAE
   - 이유: MP-SAE는 matching-pursuit sequential encoder architecture가 필요하다.
   - 현재 보유 weight는 `VanillaL1SAE`이므로 MP-SAE의 residual-guided inference, synthetic hierarchy recovery, feature absorption benchmark는 수행하지 않는다.

4. Matryoshka hierarchy objective
   - 이유: 현재 weight는 Matryoshka objective로 학습되지 않았다.
   - 단, Matryoshka 논문의 evaluation metric인 FVU/EVR/L0/dead neuron/decoder orthogonality는 사용한다.

5. CLIP text embedding과 decoder column 직접 matching
   - 이유: 현재 SAE decoder는 CLIP shared embedding space가 아니라 `timm` ViT-B block feature space에 있다.
   - 따라서 Matryoshka Appendix A의 CLIP vocabulary cosine threshold `> 0.42` 방식은 이번 demo에서 제외한다.

6. CB-SAE 재학습과 concept bottleneck neuron 추가
   - 이유: 새 논문의 실제 CB-SAE는 CLIP/LVLM activation, CLIP-Dissect, CLIP zero-shot pseudo concept activation, concept bottleneck autoencoder 학습이 필요하다.
   - 현재 demo는 이미 학습된 ViT-B `VanillaL1SAE` weight를 검증하는 것이 목적이므로, CB-SAE의 training objective는 구현하지 않는다.
   - 단, 논문의 핵심 진단인 concept coverage와 interpretability/steerability 분리는 그대로 수행한다.

## 2. 사용할 weight 선택

### weight 후보

`grid_summary.csv` 기준 1차 demo에서는 세 종류의 checkpoint를 비교한다.

| 목적 | trial | 이유 |
| --- | --- | --- |
| fidelity 우선 | `trial_0000` | 가장 낮은 `val_nmse`, 매우 높은 reconstruction cosine |
| balanced | `trial_0015` | fidelity가 높고 active latent 수가 중간 수준 |
| sparse/interpretability 우선 | `trial_0025` | active latent 수가 작아 top image와 heatmap 검토가 쉬움 |

### 최종 deep dive 우선순위

1. 먼저 세 trial 전체에 대해 Matryoshka식 sparsity-fidelity 표를 만든다.
2. 그다음 PatchSAE/VLM-SAE식 interpretability 분석은 `trial_0025`를 우선한다.
3. `trial_0025`에서 monosemantic latent가 너무 적으면 `trial_0015`를 보조 deep dive로 사용한다.

## 3. 전체 실험 흐름

```text
1. Load ViT-B backbone and selected SAE checkpoint
2. Extract block-10 patch token activations
3. Normalize activations using train split statistics
4. Encode tokens with SAE
5. Aggregate patch latent activations to image-level latent activations
6. Run paper-style validation protocols:
   A. Matryoshka metric table
   B. PatchSAE latent statistics and reference images
   C. PatchSAE patch heatmaps
   D. Network Dissection Broden concept-mask IoU
   E. VLM-SAE MonoSemanticity score
   F. PatchSAE class-level latent aggregation
   G. PatchSAE latent masking/intervention
   H. CB-SAE concept coverage and interpretability-steerability quadrant
7. Save CSV/JSON metrics and paper-style figures
```

## 4. Protocol A: Matryoshka식 reconstruction/sparsity 평가

### 출처

- `paper/refer/Matroshoka.pdf`
- Section 4.1 `Evaluation Metrics`
- Section 4.2 `Sparsity-Fidelity Trade-off`

### 목적

SAE가 원 ViT-B activation을 충분히 복원하면서 sparse latent를 만들었는지 확인한다.

### 입력

- selected trial checkpoint
- Imagenette val patch-token activations
- normalized token matrix `X`
- SAE reconstruction `X_hat`
- SAE latent activation `Z`

### 계산 metric

1. MSE
   - `mean((X_hat - X)^2)`

2. FVU
   - `MSE / variance(X)`
   - Matryoshka 논문의 normalized reconstruction error에 해당.

3. EVR
   - `1 - FVU`

4. cosine reconstruction
   - token별 `cos(X, X_hat)` 평균.

5. L0 또는 active count
   - token별 `Z > active_threshold` 개수 평균.

6. active ratio
   - `mean(active_count / hidden_dim)`

7. dead latent ratio
   - val set 전체에서 한 번도 threshold를 넘지 않는 latent 비율.

8. decoder orthogonality
   - SAE decoder row vector pairwise cosine의 off-diagonal 평균 및 상위 quantile.
   - Matryoshka 논문의 decoder orthogonality metric에 해당.

### 산출물

```text
protocol_demo/matryoshka_metrics/trial_metrics.csv
protocol_demo/matryoshka_metrics/sparsity_fidelity_scatter.png
protocol_demo/matryoshka_metrics/decoder_orthogonality_hist.png
protocol_demo/matryoshka_metrics/dead_latent_summary.json
```

### 판정

- fidelity checkpoint는 EVR이 가장 높아야 한다.
- sparse checkpoint는 L0가 낮고 dead latent가 과도하게 높지 않아야 한다.
- decoder orthogonality가 너무 낮으면 latent가 중복 feature를 많이 담는 것으로 본다.
- 이후 interpretability 분석은 fidelity만 보지 않고 sparse/active count가 해석 가능한 trial을 우선한다.

## 5. Protocol B: PatchSAE식 latent summary statistics

### 출처

- `paper/refer/PatchSAE.pdf`
- Section 3.2 `Analysis Method and Evaluation Setup`
- Figure 2, Figure 3

### 목적

각 SAE latent가 얼마나 자주, 얼마나 강하게 켜지고, 어떤 class label에 집중되는지 확인한다.

### 입력

- image별 patch latent activation `Z_image`
  - shape: `[num_images, num_patches, hidden_dim]`
- image label `y`

### image-level activation 집계

PatchSAE의 image/class/dataset aggregation 방식을 따른다.

각 image `i`, patch `j`, latent `s`에 대해:

```text
h_ij[s] = SAE latent activation
a_ij[s] = 1[h_ij[s] > tau]
image_active_count_i[s] = sum_j a_ij[s]
image_mean_i[s] = mean_j h_ij[s]
image_max_i[s] = max_j h_ij[s]
```

demo에서는 latent ranking에는 `image_active_count`와 `image_mean`을 모두 저장한다.

### latent별 summary

각 latent `s`에 대해:

1. activation frequency
   - `image_active_count_i[s] > 0`인 이미지 비율.

2. mean positive activation
   - active image에서의 `image_mean_i[s]` 평균.

3. mean active patches
   - active image에서 몇 개 patch가 켜지는지 평균.

4. label entropy
   - latent activation으로 weighted class distribution을 만들고 entropy 계산.

5. top class
   - 가장 activation mass가 큰 Imagenette class.

6. class purity
   - top class activation mass / total activation mass.

### 산출물

```text
protocol_demo/patchsae_stats/trial_XXXX/latent_summary.csv
protocol_demo/patchsae_stats/trial_XXXX/frequency_vs_mean_activation.png
protocol_demo/patchsae_stats/trial_XXXX/frequency_vs_label_entropy.png
protocol_demo/patchsae_stats/trial_XXXX/top_latents_by_region.json
```

### 판정

- high mean activation + moderate frequency: 의미 있는 visual feature 후보.
- high frequency + high entropy: generic 또는 background feature 후보.
- low entropy + high purity: class-discriminative feature 후보.
- low frequency + high activation: rare feature 후보.

## 6. Protocol C: PatchSAE식 top activating reference images

### 출처

- `paper/refer/PatchSAE.pdf`
- Section 3.2 `Reference images for SAE latents`
- Appendix A.3 cross-dataset reference image examples

### 목적

latent direction을 사람이 해석할 수 있도록 top activating image grid를 만든다.

### latent 선택

다음 group에서 latent를 고른다.

1. mean activation top 20
2. frequency top 20
3. low label entropy top 20
4. class purity top 20
5. MonoSemanticity score top 20
6. intervention effect top 20

중복 제거 후 최대 100개 latent를 reference image 생성 대상으로 둔다.

### top image 선택 방식

각 latent `s`에 대해:

```text
score_i[s] = image_active_count_i[s]
tie_breaker = image_mean_i[s]
top_k_images = argsort(score_i[s], tie_breaker)[:K]
```

K는 논문 방식에 맞춰 기본 `16`으로 둔다.

### figure 구성

각 latent마다 하나의 grid를 만든다.

```text
title:
  latent id
  activation frequency
  mean positive activation
  label entropy
  top class / purity
  MS score if available

each tile:
  image
  class name
  active patch count
  max activation
```

### 산출물

```text
protocol_demo/reference_images/trial_XXXX/latent_000123_top16.png
protocol_demo/reference_images/trial_XXXX/latent_000123_top16.json
protocol_demo/reference_images/trial_XXXX/contact_sheet_top_latents.png
```

### 판정

- top-16 이미지가 같은 object, part, texture, color, pose, background 중 하나로 묶이면 interpretable latent 후보.
- class는 다르지만 visual texture가 같으면 texture/style latent로 기록한다.
- 이미지들이 시각적으로 무관하면 polysemantic 후보로 기록한다.

## 7. Protocol D: PatchSAE식 patch-level spatial attribution

### 출처

- `paper/refer/PatchSAE.pdf`
- Section 3.2 `Localizing patch-level SAE latent activations`
- Figure 2(c)

### 목적

latent가 이미지의 어느 patch에서 켜지는지 확인한다.

### 계산 방식

각 selected latent와 top activating image에 대해:

```text
patch_map = h_ij[s] for all patches j
patch_map_grid = reshape(patch_map, 14, 14)
overlay = resize patch_map_grid to image resolution
```

현재 ViT-B patch size가 16이고 input이 224이면 patch grid는 `14 x 14`이다.

### 산출물

```text
protocol_demo/patch_heatmaps/trial_XXXX/latent_000123_image_000045_overlay.png
protocol_demo/patch_heatmaps/trial_XXXX/latent_000123_heatmap_grid.png
protocol_demo/patch_heatmaps/trial_XXXX/spatial_stats.csv
```

### spatial stats

- peak patch index
- peak activation
- active patch count
- active patch ratio
- concentration score
  - 예: `max_activation / sum_positive_activation`
- image class

### 판정

- object part feature: heatmap이 object 일부에 국소적으로 집중되어야 한다.
- texture/color feature: 해당 texture/color 영역에 분포해야 한다.
- global/background feature: 전역적으로 켜질 수 있으나 top images가 일관되어야 한다.

## 8. Protocol E: Network Dissection / Broden concept-mask alignment

### 출처

- 외부 추가 논문: Network Dissection: Quantifying Interpretability of Deep Visual Representations
- 링크: https://arxiv.org/abs/1704.05796
- Section 2.1 `Broden: Broadly and Densely Labeled Dataset`
- Section 2.2 `Scoring Unit Interpretability`

### 왜 추가하는가

top activating images, MS score, class entropy, patch heatmap은 모두 alignment의 간접 증거다. 이들만으로는 latent가 어떤 feature와 align되었다고 강하게 말할 수 없다. Network Dissection은 Broden의 dense concept annotation을 사용해 unit activation map이 실제 concept mask와 얼마나 겹치는지 IoU로 평가한다. 따라서 이 protocol은 “feature-latent alignment 후보”를 “mask-level alignment evidence가 있는 latent”로 끌어올리는 역할을 한다.

### Broden dataset 구성

Network Dissection의 Broden은 여러 densely labeled dataset을 합쳐 다음 concept category를 제공한다.

- object
- part
- scene
- texture
- material
- color

object, part, material, color는 pixel-level mask와 잘 맞고, scene/texture는 image-level label 성격이 강하다. 현재 ViT-B patch SAE에서는 pixel-level mask를 `14 x 14` patch grid로 downsample해서 latent activation mask와 비교한다.

### 입력

- Broden image `x`
- Broden concept label `c`
- Broden concept mask `L_c(x)`
- SAE latent patch activation map `A_k(x)`

### 계산 방식

Network Dissection의 unit scoring을 ViT patch latent에 맞게 변형한다.

1. Broden image를 ViT-B transform으로 resize/crop한다.
2. ViT-B block 10 patch token을 추출한다.
3. SAE encoder로 patch latent activation `A_k(x)`를 얻는다.
4. Broden pixel mask `L_c(x)`를 ViT patch grid 크기인 `14 x 14`로 downsample한다.
5. latent activation threshold `T_k`를 정한다.
   - Network Dissection 원 논문은 dataset-wide activation top quantile을 사용한다.
   - demo 기본값: `P(A_k > T_k) = 0.005`가 되도록 latent별 threshold 계산.
6. latent activation mask를 만든다.

```text
M_k(x) = 1[A_k(x) >= T_k]
```

7. concept별 dataset-wide IoU를 계산한다.

```text
IoU(k, c) = sum_x |M_k(x) intersect L_c(x)| / sum_x |M_k(x) union L_c(x)|
```

8. latent `k`의 label은 IoU가 가장 높은 concept `c*`로 둔다.

```text
c*(k) = argmax_c IoU(k, c)
```

### 산출물

```text
protocol_demo/broden_alignment/trial_XXXX/broden_iou_scores.csv
protocol_demo/broden_alignment/trial_XXXX/broden_best_concepts.csv
protocol_demo/broden_alignment/trial_XXXX/broden_category_summary.csv
protocol_demo/broden_alignment/trial_XXXX/latent_000123_broden_examples.png
protocol_demo/broden_alignment/trial_XXXX/latent_000123_mask_overlay.png
```

### 기록할 컬럼

- latent id
- best concept
- concept category
- IoU
- activation threshold `T_k`
- number of concept images
- number of active patches
- top activating Broden images
- top activating Imagenette images
- agreement between Broden concept and Imagenette top-image interpretation

### 판정 기준

Network Dissection 원 논문은 `IoU > 0.04`를 concept detector 기준으로 사용했다. 이 threshold는 CNN unit과 Broden mask 기준이므로 ViT patch SAE에서는 그대로 절대 기준으로 고정하지 않고, 다음처럼 tier를 둔다.

| tier | 조건 | 의미 |
| --- | --- | --- |
| Broden-0 | IoU가 낮고 top concept 불안정 | mask-level alignment evidence 없음 |
| Broden-1 | best IoU가 random/control보다 높음 | 약한 concept-mask association |
| Broden-2 | best IoU가 높고 top images와 concept가 일치 | concept alignment 후보 강화 |
| Broden-3 | IoU 높음 + top images/MS/heatmap 모두 일치 | strong feature-latent alignment evidence |
| Broden-4 | Broden alignment + masking/intervention effect | behaviorally relevant aligned latent |

### control

Broden IoU가 진짜 alignment인지 보려면 baseline을 같이 계산한다.

1. random latent baseline
   - 같은 activation frequency를 가진 random latent의 IoU.

2. shuffled mask baseline
   - Broden mask를 image 간 shuffle.

3. random rotation baseline
   - 가능하면 latent basis를 random orthogonal mixing한 뒤 IoU 감소 확인.
   - Network Dissection이 axis-aligned interpretability 검증에 사용한 사고방식과 대응된다.

### 이 protocol로도 말할 수 없는 것

Broden IoU가 높아도 “latent = feature”를 증명하는 것은 아니다.

- Broden에 없는 concept는 검출하지 못한다.
- 하나의 feature가 여러 latent에 분산되어 있을 수 있다.
- 하나의 latent가 Broden concept와 겹치면서도 다른 shortcut을 함께 담을 수 있다.

따라서 최종 문구는 “latent is aligned with concept c”가 아니라 “latent shows Broden mask-level alignment evidence for concept c”로 쓴다.

## 9. Protocol F: VLM-SAE식 MonoSemanticity score

### 출처

- `paper/refer/Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models.pdf`
- Section 3.2 `Monosemanticity Score`
- Figure 2, Figure 3
- Section 4.2.1 `Alignment of MS with human perception`

### 목적

각 latent가 한 가지 concept에 집중되는지 자동 점수로 평가한다.

### 입력

- evaluation image set `I`
- image semantic embedding `E(x)`
- image-level latent activation `a_k(x)`

### semantic embedding 선택

논문은 pretrained image encoder embedding을 사용한다. 현재 demo에서는 다음 우선순위를 둔다.

1. 기본: ViT-B final CLS embedding
   - 현재 backbone과 동일해 별도 model을 요구하지 않는다.
2. 선택: CLIP image embedding
   - 설치/weight가 준비되어 있을 때만 사용.
   - 이 경우 MS score가 사람의 semantic similarity와 더 가까울 수 있다.

### 계산 방식

이미지 `x_n`, `x_m`에 대해:

```text
S_nm = cosine(E(x_n), E(x_m))
```

latent `k`의 image-level activation vector를 min-max normalize:

```text
a_tilde_k[n] = (a_k[n] - min(a_k)) / (max(a_k) - min(a_k))
R_k[n,m] = a_tilde_k[n] * a_tilde_k[m]
MS_k = sum_{n<m} R_k[n,m] * S_nm / sum_{n<m} R_k[n,m]
```

### human-review artifact

VLM-SAE의 user study 방식처럼 latent pair 비교용 grid를 만든다.

```text
latent_A_top16.png
latent_B_top16.png
question: Which set of images looks more similar and focused on the same thing?
```

실제 user study를 바로 수행하지 않더라도, review CSV에 사람이 선택할 수 있는 column을 둔다.

### 산출물

```text
protocol_demo/monosemanticity/trial_XXXX/ms_scores.csv
protocol_demo/monosemanticity/trial_XXXX/ms_hist.png
protocol_demo/monosemanticity/trial_XXXX/ms_top_latents_contact_sheet.png
protocol_demo/monosemanticity/trial_XXXX/ms_bottom_latents_contact_sheet.png
protocol_demo/monosemanticity/trial_XXXX/human_review_pairs.csv
protocol_demo/monosemanticity/trial_XXXX/human_review_grids/
```

### 판정

- high MS + visually consistent top images: monosemantic latent.
- high MS + single class concentration: class-specific monosemantic latent.
- high MS + multi-class same texture: texture/color monosemantic latent.
- low MS + high frequency: generic/polysemantic latent.

## 10. Protocol G: PatchSAE식 class-level latent aggregation

### 출처

- `paper/refer/PatchSAE.pdf`
- Section 3.2 `Discovering active SAE latents in diverse levels`
- Section 4.1 `Top-k SAE latent masking`

### 목적

latent가 Imagenette class-discriminative 정보를 담는지 확인한다.

### 계산 방식

PatchSAE의 class-level activation:

```text
a_c[s] = sum_{i in class c} image_active_count_i[s]
```

추가로 다음 값을 계산한다.

- class mean activation
- class active frequency
- class purity
- label entropy
- one-vs-rest score
  - class c 이미지의 activation과 나머지 이미지 activation을 분리하는 정도.

### 산출물

```text
protocol_demo/class_alignment/trial_XXXX/class_latent_matrix.csv
protocol_demo/class_alignment/trial_XXXX/class_top_latents.csv
protocol_demo/class_alignment/trial_XXXX/class_latent_heatmap.png
protocol_demo/class_alignment/trial_XXXX/class_reference_grids/
```

### 판정

- 특정 class에서만 강하게 켜지는 latent는 class-discriminative feature 후보.
- 여러 class에서 비슷하게 켜지는 latent는 style/texture/background 후보.
- class-level top latent가 top image grid에서도 같은 class 또는 같은 visual part를 보이면 strong alignment.

## 11. Protocol H: PatchSAE식 top-k latent masking

### 출처

- `paper/refer/PatchSAE.pdf`
- Section 4.1 `Impact of SAE Latents on Classification`
- Section 4.1.2 `Key Findings`

### 목적

class-level로 선택한 top-k latent가 classifier prediction에 실제로 중요한지 확인한다.

### 실험 방식

ViT-B block 10 activation `H`를 SAE로 encode/decode한다.

```text
Z = SAE.encode(H)
H_hat = SAE.decode(Z)
```

mask 종류:

1. `on_all`
   - 전체 latent를 사용한 SAE reconstruction.
   - 원 model activation 대체 시 성능이 크게 유지되어야 한다.

2. `off_all`
   - latent를 모두 끈 reconstruction.
   - 성능이 크게 떨어지는 sanity check.

3. `on_class_topk`
   - 정답 class 또는 predicted class의 class-level top-k latent만 켠다.

4. `off_class_topk`
   - class-level top-k latent만 끈다.

5. `on_random_k`
   - 같은 개수의 random latent만 켠다.

6. `off_random_k`
   - 같은 개수의 random latent만 끈다.

7. `on_dataset_topk`
   - dataset 전체에서 자주 켜지는 top-k latent만 켠다.

### k 값

PatchSAE 방식처럼 k sweep을 한다.

```text
k = 1, 2, 4, 8, 16, 32, 64
```

sparse checkpoint에서는 hidden active 수가 작으므로 `k=1,2,4,8,16`을 우선한다.

### metric

- top-1 accuracy
- clean vs reconstructed accuracy
- confidence drop
- logit L1
- JS divergence between softmax outputs

### 산출물

```text
protocol_demo/topk_masking/trial_XXXX/masking_results.csv
protocol_demo/topk_masking/trial_XXXX/masking_accuracy_curve.png
protocol_demo/topk_masking/trial_XXXX/masking_logit_shift_curve.png
```

### 판정

- `on_all`이 원 model 성능을 대부분 회복해야 reconstruction이 classifier-relevant 정보를 보존한다고 본다.
- `on_class_topk`가 `on_random_k`보다 높으면 class top latent가 class-discriminative 정보를 담는다고 본다.
- `off_class_topk`가 `off_random_k`보다 더 큰 성능 저하를 만들면 해당 latent set의 behavioral relevance가 있다고 본다.

## 12. Protocol I: VLM-SAE식 intervention curve

### 출처

- `paper/refer/Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models.pdf`
- Section 3.3 `Steering MLLMs with Vision SAEs`
- 단, 현재 demo는 MLLM이 아니라 ViT-B classifier activation에 같은 latent manipulation 원리를 적용한다.

### 목적

monosemantic latent 또는 class-specific latent를 조작했을 때 classifier output이 변하는지 확인한다.

### 실험 방식

논문의 neuron intervention 형태를 ViT-B classifier에 맞춰 적용한다.

selected latent set `S`에 대해:

```text
Z = SAE.encode(H)
Z'_s = alpha * Z_s for s in S
Z'_r = Z_r for r not in S
H' = SAE.decode(Z')
logits' = ViT.forward_from_block_10(H')
```

alpha sweep:

```text
alpha = 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0
```

selected latent set:

1. high MS latents
2. low label entropy class latents
3. top-k reference image visually valid latents
4. random matched active-frequency baseline

### metric

- accuracy delta
- target class probability delta
- predicted class flip rate
- JS divergence
- logit L1

### 산출물

```text
protocol_demo/intervention/trial_XXXX/intervention_curves.csv
protocol_demo/intervention/trial_XXXX/acc_delta_by_alpha.png
protocol_demo/intervention/trial_XXXX/js_by_alpha.png
protocol_demo/intervention/trial_XXXX/class_probability_shift.png
```

### 판정

- alpha를 0으로 낮출 때 관련 class probability가 떨어지면 해당 latent는 classifier-relevant.
- alpha를 키울 때 관련 class probability가 오르면 causal handle 후보.
- random baseline과 비슷하면 관찰상 해석은 가능하지만 behavioral relevance는 약하다고 본다.

## 13. Protocol J: CB-SAE식 concept coverage와 utility quadrant

### 출처

- `paper/refer/Kulkarni_Interpretable_and_Steerable_Concept_Bottleneck_Sparse_Autoencoders_CVPR_2026_paper.pdf`
- Section 3 `Background`: CLIP-Dissect interpretability score와 downstream steering score.
- Section 4 `Interpretability vs Steerability in SAEs`: neuron을 interpretability/steerability 2축으로 나누는 분석.
- Section 4 Expt. 2: Broden, VLG-CBM, DECIDER, common English words 기반 concept coverage 측정.
- Figure 4 및 Section 5.1: `I + S`가 낮은 neuron을 low-utility pruning 후보로 두는 방식.
- Section 6.1 `Evaluation Metrics`: MS, Unit-Vec, White Image steering 평가.

### 목적

top image, MS, heatmap, Broden IoU가 있더라도 “각 feature와 latent가 align되었다”고 바로 말할 수는 없다. CB-SAE 논문처럼 각 latent에 대해 interpretability와 steerability를 분리하고, `user_concepts.txt`의 각 concept가 latent 공간에서 실제로 발견되는지 coverage를 계산한다.

### 현재 demo에서 바꾸는 점

CB-SAE 원 논문은 CLIP/LVLM SAE에서 CLIP-Dissect와 LLaVA/UnCLIP steering을 사용한다. 현재 weight는 `timm` ViT-B classifier block feature SAE이므로 다음 proxy를 사용한다.

| CB-SAE 원 논문 | 현재 ViT-B demo proxy |
| --- | --- |
| CLIP-Dissect interpretability | Broden IoU, MS score, top-image coherence, class purity |
| LLaVA text steering | classifier logit/probability intervention shift |
| UnCLIP image steering | 사용하지 않음 |
| user-specified concept set | `protocol_demo/concepts/user_concepts.txt` |
| low-utility neuron pruning | 재학습 없이 retained/prune 후보 CSV만 생성 |

### 입력

- `protocol_demo/concepts/user_concepts.txt`
- Protocol B의 latent summary
- Protocol C/D의 top image와 heatmap
- Protocol E의 Broden IoU table, if available
- Protocol F의 MS score
- Protocol G의 class latent aggregation
- Protocol H/I의 masking/intervention result

### concept assignment

각 latent `k`에 대해 primary concept 후보를 다음 우선순위로 정한다.

1. Broden IoU가 있으면 best Broden concept.
2. Broden이 없으면 top class 또는 top domain label.
3. 사람이 검토한 top image label이 있으면 manual label로 override.
4. 어느 evidence도 안정적이지 않으면 `unassigned`.

이 assignment는 최종 truth가 아니라 coverage 계산을 위한 후보 label이다.

### interpretability score `I_k`

각 score를 0-1로 normalize한 뒤 weighted average를 사용한다.

```text
I_k = 0.40 * broden_iou_norm
    + 0.25 * MS_norm
    + 0.20 * top_image_coherence
    + 0.15 * class_or_domain_purity
```

`data/broden`이 없으면 Broden 항목을 제외하고 나머지 weight를 renormalize한다. top-image coherence는 1차 구현에서는 human review CSV가 있으면 사용하고, 없으면 class purity로 대체한다.

### steerability score `S_k`

Protocol H/I 결과에서 random baseline 대비 target behavior shift를 계산한다.

```text
S_k = normalize(
  max_alpha target_logit_shift(k, alpha)
  - mean_random_latent_shift(alpha)
)
```

class concept latent이면 해당 class logit shift를 사용한다. Broden object/part/material/texture concept처럼 직접 class logit과 연결이 약한 경우에는 `S_k`를 missing으로 두고, `interpretability-only` concept로 분리한다.

### quadrant 분류

threshold는 CB-SAE 논문처럼 평균선 또는 validation percentile로 나눈다. demo에서는 우선 `tau_I = median(I)`와 `tau_S = median(S)`를 사용하고, report에는 threshold 값을 기록한다.

| quadrant | 조건 | 해석 |
| --- | --- | --- |
| high-I/high-S | `I_k >= tau_I`, `S_k >= tau_S` | aligned and behaviorally useful 후보 |
| high-I/low-S | `I_k >= tau_I`, `S_k < tau_S` | 설명은 되지만 causal evidence 약함 |
| low-I/high-S | `I_k < tau_I`, `S_k >= tau_S` | 중요한 latent일 수 있으나 feature label 불안정 |
| low-I/low-S | `I_k < tau_I`, `S_k < tau_S` | pruning/ignore 후보 |

### concept coverage audit

`user_concepts.txt`의 각 concept `c`에 대해 다음을 계산한다.

```text
best_latent(c) = argmax_k score(k, c)
covered_interpretable(c) = I_best >= tau_I
covered_behavioral(c) = I_best >= tau_I and S_best >= tau_S
```

보고서에는 concept별로 다음 컬럼을 저장한다.

- `concept`
- `best_latent_id`
- `assignment_source`: Broden, class, domain, manual, unassigned
- `interpretability_score`
- `steerability_score`
- `covered_interpretable`
- `covered_behavioral`
- `evidence_level`
- `missing_reason`

### 산출물

```text
protocol_demo/cb_sae_diagnostics/latent_interpretability_steerability.csv
protocol_demo/cb_sae_diagnostics/concept_coverage.csv
protocol_demo/cb_sae_diagnostics/quadrant_scatter.png
protocol_demo/cb_sae_diagnostics/missing_concepts.md
protocol_demo/cb_sae_diagnostics/low_utility_latents.csv
```

### 판정

- feature-latent alignment를 강하게 주장하려면 최소 `covered_interpretable=True`와 Broden/MS/top-image 중 둘 이상의 독립 evidence가 필요하다.
- model behavior까지 주장하려면 `covered_behavioral=True`가 필요하다.
- `high-I/low-S` latent는 “보이는 feature와 맞는 latent”라고만 쓰고, classifier reliance나 causal handle이라고 쓰지 않는다.
- `low-I/high-S` latent는 중요한 internal factor 후보지만 feature 이름을 붙이지 않는다.
- missing concept는 “해당 feature가 현재 SAE latent에서 확인되지 않음”으로 기록한다.

## 14. Protocol K: optional OOD corruption latent profile

### 출처

- `paper/refer/PatchSAE.pdf`
- Appendix A.3 `SAE's generalizability to different datasets`
- UniversalSAE Appendix A.4의 OOD generalization 아이디어 중 cross-model 부분 제외

### 목적

Imagenette에서 해석된 latent가 corrupted image에서도 같은 visual factor로 켜지는지 확인한다.

### 입력

- `data/imagenet-c` 또는 Imagenette와 class overlap이 있는 corruption subset.

### 계산

corruption/severity별로:

- mean activation
- active frequency
- top activating images
- top latent overlap with clean val

### 산출물

```text
protocol_demo/ood_corruption/corruption_latent_profile.csv
protocol_demo/ood_corruption/clean_vs_corruption_top_latent_overlap.csv
protocol_demo/ood_corruption/corruption_reference_grids/
```

### 주의

이 protocol은 routing, ReservoirTTA, online adaptation을 평가하지 않는다. 오직 latent generalization과 corruption sensitivity만 평가한다.

## 15. 최종 report 구조

최종 report는 논문 figure에 대응되는 순서로 구성한다.

```text
protocol_demo/report.md
protocol_demo/report.json
```

### report section

1. Dataset and checkpoint
   - 사용 dataset
   - selected trials
   - 제외한 논문 방식

2. Matryoshka-style sparsity-fidelity
   - FVU/EVR/cosine/L0/dead latent/decoder orthogonality

3. PatchSAE-style latent atlas
   - frequency vs mean activation scatter
   - label entropy coloring
   - top activating reference images

4. PatchSAE-style spatial attribution
   - patch heatmap overlays
   - active patch statistics

5. Network Dissection / Broden-style concept-mask alignment
   - Broden IoU table
   - concept category summary
   - Broden mask overlay examples
   - Broden tier assignment

6. VLM-SAE-style monosemanticity
   - MS score histogram
   - high/low MS reference grids
   - human review pair CSV

7. PatchSAE-style class aggregation
   - class-latent heatmap
   - class top latent table

8. PatchSAE-style masking
   - on/off top-k accuracy curves
   - random baseline comparison

9. VLM-SAE-style intervention
   - alpha curve
   - output shift
   - random matched baseline

10. CB-SAE-style concept coverage and utility quadrant
   - interpretability/steerability quadrant scatter
   - concept coverage table
   - missing concept list
   - low-utility latent list

11. Final latent evidence levels
   - Level 0: no alignment evidence
   - Level 1: top-image coherence only
   - Level 2: top-image + MS + heatmap consistency
   - Level 3: class/domain statistical association
   - Level 4: Broden mask-level IoU evidence
   - Level 5: Broden evidence + masking/intervention + human review agreement
   - Level 6: user-specified concept coverage + high-I/high-S quadrant

## 16. 구현 우선순위

### Phase 1: 논문 figure 재현형 artifact 생성

우선 paper-style artifact만 만든다.

- Matryoshka metric table
- PatchSAE latent scatter
- top-16 reference image grid
- patch heatmap overlay
- MS score table
- Broden IoU table, if `data/broden` exists

### Phase 2: class alignment과 masking

Broden mask alignment와 PatchSAE Section 4.1에 맞춰 class-level top-k latent masking을 구현한다.

- Broden concept-mask IoU
- class latent aggregation
- top-k on/off masking
- random baseline
- dataset-level top-k baseline

### Phase 3: intervention curve

VLM-SAE의 activation intervention 원리를 ViT-B classifier에 적용한다.

- alpha sweep
- high-MS/class-latent/random baseline 비교
- final category assignment

### Phase 4: CB-SAE diagnostic report

새 CVPR 2026 CB-SAE 논문의 진단 방식을 재학습 없이 적용한다.

- `user_concepts.txt` 생성
- latent별 `I_k`, `S_k` 계산
- quadrant scatter 생성
- concept coverage table 생성
- missing concept와 low-utility latent list 작성

## 17. 최소 성공 기준

1. selected trial checkpoint를 1개 이상 load한다.
2. Matryoshka식 FVU/EVR/L0/dead latent metric이 생성된다.
3. PatchSAE식 latent summary scatter가 생성된다.
4. top-16 reference image grid가 최소 50개 latent에 대해 생성된다.
5. patch heatmap overlay가 최소 50장 생성된다.
6. `data/broden`이 있으면 Broden IoU score가 최소 100개 latent에 대해 생성된다.
7. MS score가 최소 100개 latent에 대해 생성된다.
8. class-level latent heatmap이 생성된다.
9. top-k masking curve가 random baseline과 함께 생성된다.
10. intervention alpha curve가 최소 3개 latent set에 대해 생성된다.
11. `user_concepts.txt`의 concept coverage table이 생성된다.
12. latent별 interpretability/steerability quadrant가 생성된다.

## 18. 최종 제언

이번 demo는 “현재 repo 코드가 이미 하는 분석을 예쁘게 재실행”하는 것이 아니라, refer 논문들의 검증 방식을 현재 ViT-B SAE weight에 맞게 이식하는 것이다. 핵심은 다음 순서다.

1. Matryoshka 방식으로 SAE weight의 fidelity/sparsity를 먼저 판정한다.
2. PatchSAE 방식으로 latent atlas를 만든다.
3. Network Dissection/Broden 방식으로 concept mask와 latent activation mask의 IoU를 계산한다.
4. VLM-SAE 방식으로 monosemanticity를 정량화한다.
5. PatchSAE 방식으로 class-level top-k latent가 classifier behavior에 필요한지 masking한다.
6. VLM-SAE 방식으로 latent activation intervention curve를 만든다.
7. CB-SAE 방식으로 concept coverage와 interpretability/steerability quadrant를 계산한다.

UniversalSAE와 MP-SAE는 좋은 reference이지만 현재 단일 ViT-B `VanillaL1SAE` weight로는 그대로 수행할 수 없으므로 demo protocol에서는 명시적으로 제외한다. CB-SAE도 전체 재학습은 제외하지만, “해석 가능함”과 “조작 가능함”을 분리하고 missing concept를 보고하는 진단은 현재 weight에서도 반드시 수행한다.
