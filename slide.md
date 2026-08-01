---
marp: true
theme: default
paginate: true
size: 16:9
title: Broden을 이용한 SAE Latent–Concept Alignment 검증
---

# Broden을 이용한 SAE Latent–Concept Alignment 검증

SAE latent가 사람이 이해할 수 있는 시각 개념과 실제로 정렬되는가?

Feature Reliance Project

---

# 1. Introduction

## 연구 배경

- Vision Transformer의 내부 표현에는 물체, 색, 질감, 형태와 같은 여러 시각 정보가 섞여 있다.
- Sparse Autoencoder(SAE)는 이 복잡한 표현을 sparse latent들의 조합으로 분해한다.
- 각 latent가 하나의 해석 가능한 시각 feature를 나타낸다면 모델의 내부 판단 근거를 더 구체적으로 설명할 수 있다.

## 문제점

- Top activating image만 보고 latent의 의미를 정하면 해석이 주관적이다.
- 특정 이미지에서 activation이 높다는 사실만으로 어떤 물체나 영역 때문에 활성화됐는지 알 수 없다.
- 따라서 latent activation과 사람이 직접 표시한 시각 concept을 공간적으로 비교할 필요가 있다.

---

# 1. Introduction

## 실험 목적

SAE latent가 활성화되는 위치와 human-annotated visual concept의 위치를 비교하여 다음을 확인한다.

1. 특정 latent가 일관된 시각 concept에 대응하는가?
2. 그 latent는 concept이 존재하는 이미지 영역에서 활성화되는가?
3. 이러한 관계가 일부 예시가 아니라 데이터셋 전체에서 반복되는가?

## 핵심 아이디어

> 특정 latent가 `wheel`을 나타낸다면, 자동차 이미지에서 단순히 활성화되는 것을 넘어 실제 바퀴가 표시된 위치에서 활성화되어야 한다.

---

# 2. Dataset

## Broden Dataset

Broden은 Network Dissection에서 신경망 내부 unit의 해석 가능성을 평가하기 위해 구축한 visual concept dataset이다.

여러 dense annotation dataset을 통합하여 서로 다른 의미 수준의 concept을 함께 제공한다.

| Category | 설명 | 예시 |
| --- | --- | --- |
| Object | 이미지에 존재하는 물체 | car, dog, building |
| Part | 물체의 구성 요소 | wheel, head, door |
| Material | 표면의 재질 | wood, metal, glass |
| Texture | 반복되는 시각 패턴 | striped, zigzagged |
| Color | 지각적 색상 | red, white, blue |
| Scene | 이미지 전체의 장소 | bedroom, fire station |

공간 annotation이 있는 concept은 pixel-level mask로 제공되며, scene과 일부 texture concept은 image-level label로 제공된다.

> 출처: Bau et al., *Network Dissection: Quantifying Interpretability of Deep Visual Representations*, **CVPR 2017** — [Paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Bau_Network_Dissection_Quantifying_CVPR_2017_paper.html) · [Project](https://netdissect.csail.mit.edu/) · [Dataset/Code](https://github.com/CSAILVision/NetDissect)

---

# 2. Dataset

## 실제 Broden 이미지와 annotation

### 실제 이미지

![Broden object image](data/broden1_227/images/ade20k/ADE_train_00018550.jpg)

Broden image ID 18 · Object example: `car`

---

# 2. Dataset

## 실제 Broden 이미지와 label overlay

![Broden object label overlay](Cache/sae_broden/broden_data_preview/object_car.png)

- Category: `object`
- Concept: `car`
- 원본 이미지와 사람이 지정한 label 영역이 함께 표시되어 있다.

> 이미지 및 annotation 출처: Zhou et al., *Scene Parsing through ADE20K Dataset*, **CVPR 2017** — [Paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhou_Scene_Parsing_Through_CVPR_2017_paper.html)  
> Broden 통합 데이터 출처: Bau et al., *Network Dissection*, **CVPR 2017** — [Project](https://netdissect.csail.mit.edu/)

---

# 3. 실험 목적과 간단한 과정

## 실험에서 확인하려는 것

SAE latent와 Broden concept 사이에 의미 있는 공간적 정렬이 존재하는지 확인한다.

## 전체 과정

1. Broden 이미지를 Vision Transformer에 입력한다.
2. 중간 layer의 patch representation을 추출한다.
3. 학습된 SAE를 이용해 patch별 latent activation을 계산한다.
4. 각 latent의 activation을 이미지 위의 spatial map으로 변환한다.
5. Broden concept mask와 latent activation mask를 비교한다.
6. 데이터셋 전체에서 가장 잘 겹치는 concept을 latent의 의미 후보로 정한다.
7. 정량 결과와 실제 overlay를 함께 확인한다.

---

# 3. 실험 목적과 간단한 과정

## 실험의 논리

```text
Broden image
    ↓
ViT patch representation
    ↓
SAE latent activation map
    ↓
Broden concept mask와 공간적으로 비교
    ↓
Latent–concept alignment score
    ↓
Latent별 best matching concept
```

## 기대하는 결과

- `wheel` latent는 실제 wheel 영역에서 활성화된다.
- `red` latent는 이미지의 red 영역에서 활성화된다.
- `metal` latent는 metal로 표시된 표면에서 활성화된다.
- 같은 관계가 여러 이미지에서 반복된다.

---

# 4. 실제 실험 방법

## 1단계: Latent activation map 생성

- Broden 이미지를 ViT 입력 크기에 맞게 전처리한다.
- SAE를 학습할 때 사용한 것과 동일한 ViT layer의 patch token을 추출한다.
- Patch token을 SAE encoder에 입력한다.
- 각 patch에서 latent별 activation을 계산한다.
- Patch activation을 원래의 2차원 위치로 배치하여 latent activation map을 만든다.

ViT-B/16의 224×224 입력을 사용하는 경우 이미지는 14×14 patch grid로 표현된다.

```text
Image → 14×14 patch representation → SAE → latent별 14×14 activation map
```

---

# 4. 실제 실험 방법

## 2단계: Broden mask와 비교

- ViT patch grid를 Broden의 pixel-level concept mask와 동일한 크기로 변환한다.
- Latent activation이 thrshhold보다 높은 patch를 선택하여 binary activation mask를 만든다.
- 동일 이미지에서 latent mask와 concept mask가 겹치는 영역을 계산한다.

## Dataset-level IoU

\[
IoU(k,c)=
\frac{\sum_x |M_k(x) \cap L_c(x)|}
     {\sum_x |M_k(x) \cup L_c(x)|}
\]

- \(M_k(x)\): 이미지 \(x\)에서 latent \(k\)가 활성화된 영역
- \(L_c(x)\): 이미지 \(x\)에서 concept \(c\)가 표시된 영역
- 모든 이미지의 intersection과 union을 합산하여 계산한다.

---

# 4. 실제 실험 방법

## 3단계: Latent의 concept 결정

각 latent에 대해 IoU가 가장 높은 Broden concept을 찾는다.

\[
c^*(k)=\arg\max_c IoU(k,c)
\]

## 결과의 신뢰성 확인

- Activation threshold가 달라져도 같은 concept이 선택되는지 확인한다.
- 높은 IoU를 보인 이미지의 overlay를 직접 확인한다.
- Top activating image에서 추측한 의미와 Broden concept이 일치하는지 비교한다.

높은 IoU 하나만으로 결론을 내리지 않고 정량 결과, 안정성, 시각화를 함께 사용한다.

---

# 5. 결과

## 실험 실행 규모

전체 latent–concept IoU 계산이 완료되었다.

| 항목 | 실제 설정 / 결과 |
| --- | ---: |
| Model / layer | ViT-B/16, block 10 patch token |
| 평가 SAE latent | 12,288개 |
| Broden category | object, part, color, material, texture, scene |
| 평가 image | category별 100장, 중복 제거 후 총 598장 |
| 비교 concept | 1,197개 |
| Activation threshold | latent별 99th percentile |
| 계산된 latent–concept IoU | 3,231,744개 |

- 95/98/99th percentile에서 총 36,864개 threshold sensitivity 결과를 계산했다.
- Category-level IoU 73,728개와 대표 overlay 18개도 생성되었다.

---

# 5. 결과

## 99th percentile의 대표 best match

| SAE latent | Best Broden concept | Category | IoU |
| ---: | --- | --- | ---: |
| 3705 | rope | part | 1.000 |
| 3688 | flecked | texture | 0.923 |
| 6524 | flecked | texture | 0.913 |
| 1289 | flecked | texture | 0.883 |
| 3047 | flecked | texture | 0.872 |
| 8379 | warehouse-indoor-s | scene | 0.827 |
| 12040 | bullpen-s | scene | 0.796 |
| 21 | bed | part | 0.667 |
| 83 | track | object | 0.565 |
| 177 | flecked | texture | 0.515 |

Part와 texture에서 특히 높은 IoU가 관찰되었고, scene과 object에서도 대표 best match가 확인되었다.

> 단, category별 최대 100장을 사용했지만 concept별 표본 수가 불균형한 평가이므로 IoU 1.0은 넓은 데이터에서의 일반화를 의미하지 않는다.

---

# 5. 결과

## Threshold sensitivity

| Latent | 95th percentile | 98th percentile | 99th percentile |
| ---: | --- | --- | --- |
| 3705 | rope (1.000) | rope (1.000) | rope (1.000) |
| 3688 | flecked (0.929) | flecked (0.929) | flecked (0.923) |
| 1289 | flecked (0.908) | flecked (0.893) | flecked (0.883) |
| 177 | flecked (0.735) | flecked (0.597) | flecked (0.515) |
| 21 | pottedplant (0.500) | bed (0.333) | bed (0.667) |
| 42 | zigzagged (0.765) | zigzagged (0.393) | striped (0.209) |
| 214 | zigzagged (0.628) | zigzagged (0.454) | chimney (0.500) |

- Latent 3705, 3688, 1289, 177은 threshold가 바뀌어도 best concept이 유지되었고, 특히 3705는 모든 threshold에서 IoU 1.0이었다.
- Latent 21, 42, 214는 threshold에 따라 concept이 바뀌어, 단일 threshold의 높은 IoU만으로 의미를 확정하기 어렵다.

> 현재 산출물에는 random/shuffled baseline이 없으므로 alignment의 통계적 유의성은 아직 결론낼 수 없다.

---

# 6. 분석

## 어떤 결과를 강한 alignment로 볼 것인가?

### 강한 alignment

- IoU가 random 또는 shuffled baseline보다 높다.
- 여러 이미지에서 같은 위치 관계가 반복된다.
- Activation threshold가 달라져도 같은 concept이 선택된다.
- Latent activation과 Broden annotation이 overlay에서 실제로 겹친다.
- Top activating image에서 관찰한 의미와 Broden concept이 일치한다.

### 불확실한 alignment

- 일부 이미지에서만 우연히 겹친다.
- Threshold에 따라 best concept이 계속 바뀐다.
- 큰 배경 영역 때문에 IoU가 높아진다.
- Top image의 의미와 spatial alignment 결과가 다르다.
---

# 6. 분석

## 예상되는 한계

- Broden vocabulary에 없는 concept은 평가할 수 없다.
- 14×14 patch grid에서는 작은 object나 part의 경계가 손실될 수 있다.
- 하나의 concept이 여러 latent에 나뉘어 표현될 수 있다.
- 하나의 latent가 여러 concept에 동시에 반응할 수 있다.
- Scene과 texture의 image-level label은 object mask와 동일하게 해석하기 어렵다.
- 높은 공간적 alignment가 모델 예측에 대한 인과적 중요성을 직접 증명하지는 않는다.

---

# 6. 분석

## 결론과 다음 단계

Broden 실험은 SAE latent의 의미를 top image만 보고 추측하는 단계에서 벗어나, human annotation을 이용해 공간적으로 검증하는 방법을 제공한다.

```text
Top image와 heatmap
        ↓
Latent 의미 후보 생성
        ↓
Broden mask-level alignment 검증
        ↓
Masking 및 intervention
        ↓
모델이 실제로 의존하는 feature인지 검증
```

최종적으로는 다음 두 조건을 모두 확인해야 한다.

1. Latent가 사람이 이해할 수 있는 concept과 정렬된다.
2. 해당 latent에 개입했을 때 모델의 예측이 실제로 변한다.

> 출처: Fong & Vedaldi, *Net2Vec*, **CVPR 2018** — [Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Fong_Net2Vec_Quantifying_and_CVPR_2018_paper.html)  
> Bau et al., *Understanding the Role of Individual Units in a Deep Neural Network*, **PNAS 2020** — [Paper](https://www.pnas.org/doi/10.1073/pnas.1907375117)
