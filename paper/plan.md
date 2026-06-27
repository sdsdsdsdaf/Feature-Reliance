# AAAI Draft Plan

## Paper A: SAE Latent Analysis and Intervention for Domain Adaptation

### Abstract

- 문제: input perturbation 기반 feature reliance 분석은 color/texture/shape 같은 사전 정의 cue에 의존해 실제 시각 feature를 충분히 분리하지 못한다.
- 제안: 하나의 vision SAE를 학습해 여러 domain의 sparse latent activation profile을 비교하고, domain shift에 민감한 latent를 찾아 adaptation/intervention에 사용한다.
- 검증 방향: ImageNet, ImageNet-R, synthetic perturbation에서 SAE latent 분포와 activation frequency를 분석하고, 관련 latent 조절이 OOD 성능과 feature reliance profile을 바꾸는지 평가한다.
- 예상 기여: domain-specific feature reliance를 더 세밀한 latent 단위로 측정하고, 해석 가능한 adaptation signal로 연결한다.

### 1. Introduction

- Paragraph 1: ImageNet pretrained model의 성능은 domain이 바뀌면 크게 변하며, 이는 모델이 어떤 visual cue에 의존하는지가 domain마다 달라질 수 있음을 시사한다.
- Paragraph 2: 기존 input perturbation 방식은 grayscale, bilateral filtering, patch shuffle, local warp 등으로 cue를 약화시키지만, perturbation이 feature를 순수하게 분리하지 못하고 복수 cue를 동시에 손상시킨다는 한계가 있다. (실제 논문에서 texture는 Shape score와 Texture score를 동시에 낮추고 있는 걸 보여준다)
- Paragraph 3: 따라서 color/texture/shape라는 coarse category 대신 모델 내부 representation에서 발견되는 더 미세한 feature unit을 분석할 필요가 있다.
(실제로 visudal cue는 다른것도 있다는 걸 그거 논문을 찾아 명시)
- Paragraph 4: SAE를 vision representation, 특히 ViT block token에 적용해 sparse latent vector를 얻고, domain별 activation frequency와 magnitude를 비교하는 접근을 제안한다.
- Paragraph 5: 본 논문의 핵심 질문을 정리한다. 어떤 SAE latent가 domain-invariant인지, 어떤 latent가 domain shift-specific인지, 그리고 latent intervention이 adaptation에 도움을 주는지.
- Paragraph 6: 기여 요약. input perturbation 한계의 실증적 동기화, SAE 기반 feature reliance profile, domain-sensitive latent scoring, latent reweighting/intervention protocol.

### 2. Related Work

- Feature reliance and shortcut learning: texture bias, shape bias, cue conflict, input perturbation 기반 분석을 다룬다.
- Domain generalization/adaptation and TTA: source-free adaptation, test-time adaptation, robustness under natural/synthetic shifts를 배경으로 둔다.
- Sparse autoencoders for representation analysis: NLP mechanistic interpretability에서의 SAE와 vision으로 확장되는 PatchSAE류 연구를 연결한다.
- Representation similarity and perturbation metrics: 코드에서 사용하는 accuracy drop, relative accuracy, JS divergence, CKA, perturbation validation metrics를 관련 평가 축으로 설명한다.

### 3. Sparse Latent Feature Reliance

#### 3.1 SAE Architecture and Training Objectives

- Problem setup: pretrained classifier `f`, domain set `D`, perturbation set `P`, internal representation/token `h`, SAE latent `z`를 정의한다.
- **Stage 1**, SAE training: ViT-B/16 block output을 hook으로 수집하고 cls/patch/all token을 정규화한 뒤 Vanilla L1 SAE를 학습한다. loss는 reconstruction MSE 과 L1 sparsity weighted sum.
- Current SAE: Vanilla L1 SAE with ReLU encoder, linear decoder, unit-norm decoder rows, reconstruction MSE plus L1 activation penalty.
- Planned SAE variants: JumpReLU SAE and Top-K SAE.

#### 3.2 Analysis Method and Evaluation Setup

- patch/image/class/dataset-level activation 분석을 사용한다.
- patch-level activation은 latent가 이미지의 어느 영역에서 켜지는지 보여준다.
- image-level activation은 한 이미지의 sparse feature profile로 사용한다.
- class-level activation은 특정 class에 반복적으로 나타나는 discriminative latent를 찾는 데 사용한다.
- dataset/domain-level activation은 ImageNet, ImageNet-R, synthetic perturbation 사이의 feature reliance profile을 비교하는 데 사용한다.
- top-k masking을 사용해 자주 활성화되는 latent가 실제 classification에 중요한지 평가한다: on top-k, off top-k, random-k, class/domain-level top-k를 비교한다.
- Metrics: accuracy, relative accuracy, accuracy drop, JS divergence between logits, CKA between representations, feature reliance score, perturbation validation scores, SAE NMSE, cosine reconstruction, mean active latent count.

#### 3.3 SAE Discovers Domain-Sensitive Visual Concepts

- **Stage  2** , domain-sensitive latent discovery: ID와 OOD 또는 clean과 perturbed domain 사이의 activation 차이를 기준으로 invariant latent와 shift-sensitive latent를 나눈다.
- ImageNet-R 같은 style shift에서는 특정 latent cluster의 frequency가 증가하거나 감소할 가능성이 있다.

### 4. Analyzing Domain Adaptation Behavior via SAE

#### 4.1 Impact of SAE Latents on Classification

- **Stage 3**, latent intervention/adaptation: shift-sensitive latent의 activation 또는 decoder contribution을 up/down-weighting하여 classifier feature를 조절하고 OOD 성능 변화를 측정한다.

##### 4.1.1 Analysis Method and Experiment Setup

- Datasets: ImageNet validation, ImageNet-200 subset aligned with ImageNet-R classes, ImageNet-R.
- Models: ResNet50 pretrained on ImageNet-1K, ViT-B/16 pretrained or augmented ImageNet-1K variant.
- Perturbations: original, grayscale, bilateral, patchshuffle, patchrotation, localwarp.
- **Evaluation**: accuracy/drop, JS divergence, CKA, feature reliance score, latent sparsity, reconstruction NMSE, intervention gain을 보고한다.

##### 4.1.2 Key Findings

- SAE latent profile이 input perturbation보다 더 세밀한 domain 차이를 드러내는지 확인한다.
- latent intervention이 모든 shift에 보편적으로 이득을 주지는 않더라도, 어떤 latent가 성능 저하와 연결되는지 해석 가능한 분석을 제공하는 것을 목표로 한다.

#### 4.2 Understanding Adaptation Mechanisms

- before/after adaptation 또는 clean/OOD 비교에서 high, high-to-low, low-to-high latent group을 나누어 adaptation이 새 latent를 활성화하는지, 기존 latent를 억제하는지, 이미 활성화된 latent를 remapping하는지 분석한다.

##### 4.2.1 Analysis Method and Experiment Setup

- clean domain과 OOD domain의 class-level 또는 dataset-level SAE latent activation을 같은 latent space에서 비교한다.
- adaptation/intervention 전후의 latent activation scatter plot을 만들고 high, high-to-low, low-to-high group을 나눈다.

##### 4.2.2 Key Findings

- 핵심 예상: adaptation gain은 완전히 새로운 feature 발견보다는 기존 sparse visual concept의 선택적 억제 또는 재가중에서 나올 가능성이 있다.
- shift-sensitive latent가 OOD 성능 저하와 연결된다면, SAE는 domain adaptation을 해석 가능한 latent 단위로 분석하는 도구가 될 수 있다.

### 5. Discussion

- Vanilla L1 SAE 이후 JumpReLU SAE, Top-K SAE로 확장해 sparsity-quality trade-off와 novelty를 탐색한다.
- latent intervention의 목표는 항상 accuracy를 올리는 것이 아니라, domain shift에서 어떤 내부 feature reliance가 변하는지 설명하는 것이다.

### 6. Conclusion

- 본 논문은 perturbation label보다 더 세밀한 SAE latent 단위로 domain-specific feature reliance를 측정하는 방향을 제안한다.
- 예상되는 핵심 결과는 domain shift-sensitive latent를 찾아 OOD 성능 변화와 연결하고, 이를 해석 가능한 adaptation signal로 사용하는 것이다.

## Paper B: SAE-Latent Reservoir Routing for Test-Time Adaptation

### Abstract

- 문제: test-time adaptation은 변화하는 domain에서 안정성과 plasticity를 동시에 요구한다.
- 제안: ReservoirTTA의 StyleVec routing을 SAE latent vector로 대체해, sample의 sparse feature activation pattern을 기준으로 specialist reservoir에 routing한다.
- 검증 방향: 동일한 single SAE encoder를 사용해 domain tendency를 추정하고, 유사한 latent pattern의 sample을 같은 specialist에 할당해 adaptation stability와 OOD 성능을 평가한다.
- 예상 기여: hand-crafted style descriptor 대신 model-internal sparse latent descriptor를 사용하는 reservoir-based TTA framework.

### 1. Introduction

- Paragraph 1: 실제 배포 환경에서는 test stream의 domain이 시간에 따라 바뀌며, 하나의 모델을 계속 업데이트하면 catastrophic drift나 unstable adaptation이 발생할 수 있다.
- Paragraph 2: ReservoirTTA는 여러 reservoir specialist를 유지하고 style vector로 sample을 routing해 안정성을 높이는 baseline이다.
- Paragraph 3: 그러나 style vector는 외부적/저차원 style descriptor에 가까워, 내부 feature 변화와 완전히 일치하지 않을 수 있으며 해석이 불가능하다.
- Paragraph 4: SAE latent vector는 pretrained vision model의 internal representation을 sparse하고 비교 가능한 pattern으로 변환하므로 routing signal로 적합하다.
- Paragraph 5: 본 논문은 ReservoirTTA의 StyleVec를 SAE latent로 대체하고, latent similarity 기반 routing 및 specialist adaptation을 제안한다.
- Paragraph 6: 기여 요약. single SAE 기반 domain tendency descriptor, SAE latent routing, ReservoirTTA baseline과 직접 비교 가능한 TTA protocol, 효율성/해석성 trade-off 분석.

### 2. Related Work

- Test-time adaptation: entropy minimization, batch normalization adaptation, continual/source-free TTA, stability-plasticity 문제를 다룬다.
- Reservoir/specialist-based adaptation: multiple specialists, routing, memory/reservoir를 사용하는 adaptation 방법을 배경으로 둔다.
- Style-based routing and domain descriptors: style statistics, texture/style shift descriptor, domain clustering 방법과 비교한다.
- Sparse latent representation: SAE가 routing descriptor로 쓰일 때의 장점과 한계, vision SAE/PatchSAE 관련 연구를 연결한다.

### 3. SAE-Latent Reservoir Routing

#### 3.1 SAE Descriptor Architecture and Extraction

- Problem setup: unlabeled test stream `x_t`, frozen or partially adaptable backbone `f`, SAE encoder `E_sae`, specialist set `{g_k}`를 정의한다.
- SAE descriptor extraction: 입력 sample에서 ViT token 또는 backbone feature를 추출하고 SAE encoder로 sparse latent `z_t`를 얻는다. image-level descriptor는 token latent의 mean/max/frequency pooling으로 만든다.

#### 3.2 Analysis Method and Evaluation Setup

- patch/image/class/dataset-level activation 분석을 routing descriptor 검증에도 사용한다.
- dataset/domain-level activation은 stream segment 또는 reservoir prototype이 실제 domain tendency를 반영하는지 확인하는 데 사용한다.
- top-k overlap은 SAE latent routing이 StyleVec보다 어떤 sparse feature를 기준으로 sample을 묶는지 해석하는 데 사용한다.

#### 3.3 SAE Latents as Domain Routing Signals

- Reservoir routing: `z_t`와 reservoir prototypes 사이의 cosine distance, JS divergence, or top-k overlap을 사용해 가장 가까운 specialist를 선택한다.
- Reservoir maintenance: 각 specialist의 latent prototype과 memory를 moving average 또는 bounded queue로 갱신한다.

### 4. Analyzing Test-Time Adaptation Behavior via SAE Routing

#### 4.1 Impact of SAE-Latent Routing on Adaptation

- Specialist update: 선택된 specialist만 entropy minimization, feature consistency, or pseudo-label objective로 업데이트한다. 나머지 specialist는 drift를 막기 위해 유지한다.

##### 4.1.1 Analysis Method and Experiment Setup

- ImageNet-C/ImageNet-R/synthetic perturbation stream을 사용한다.
- recurring and evolving domain stream을 구성해 reservoir 구조의 안정성을 평가한다.
- Comparison to ReservoirTTA: 기존 StyleVec routing을 SAE latent routing으로 교체한 것을 main variant로 두고, StyleVec, random routing, single model TTA와 비교한다.

##### 4.1.2 Key Findings

- 핵심 예상: SAE latent routing은 style vector보다 model-internal feature shift를 더 잘 반영해 특정 domain shift에서 routing purity와 adaptation 성능을 높일 가능성이 있다.
- Reservoir 구조 덕분에 하나의 모델을 계속 갱신하는 방식보다 안정적일 수 있다.

#### 4.2 Understanding Reservoir Adaptation Mechanisms

- before/after adaptation latent profile을 비교해 specialist update가 domain-specific latent를 새로 활성화하는지, 억제하는지, 기존 latent mapping을 바꾸는지 분석한다.

##### 4.2.1 Analysis Method and Experiment Setup

- specialist별 latent prototype을 비교하고, 각 specialist가 어떤 sparse feature reliance profile을 담당하는지 확인한다.
- Evaluation: ImageNet-C/ImageNet-R/synthetic perturbation stream에서 accuracy, online error, adaptation stability, routing purity, model size, inference time을 측정한다.

##### 4.2.2 Key Findings

- 핵심 예상: SAE latent prototype이 recurring domain을 안정적으로 구분한다면, reservoir specialist가 해석 가능한 domain-specific feature profile로 분화될 수 있다.
- 반대로 routing-adaptation 결합 때문에 해석성이 흐려질 수 있으며, 이 경우 hybrid StyleVec + SAE routing이나 lightweight latent pooling을 검토한다.

### 5. Discussion

- 단점으로 specialist 수 증가에 따른 메모리/추론 비용, ReservoirTTA framework 의존으로 인한 novelty 약화, routing-adaptation 결합으로 인한 해석성 감소를 논의한다.
- 향후 JumpReLU/Top-K SAE, lightweight latent pooling, specialist pruning으로 효율성을 개선할 수 있다.

### 6. Conclusion

- 본 논문은 hand-crafted style descriptor 대신 model-internal sparse latent descriptor를 reservoir routing에 사용하는 TTA framework를 제안한다.
- 예상되는 핵심 결과는 SAE latent routing이 recurring/evolving domain stream에서 안정성과 해석 가능한 routing signal을 동시에 제공할 수 있음을 보이는 것이다.
