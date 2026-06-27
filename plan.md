# AAAI Draft Plan

## Paper A: SAE Latent Analysis and Intervention for Domain Adaptation

### Abstract
- 문제: input perturbation 기반 feature reliance 분석은 color/texture/shape 같은 사전 정의 cue에 의존해 실제 시각 feature를 충분히 분리하지 못한다.
- 제안: 하나의 vision SAE를 학습해 여러 domain의 sparse latent activation profile을 비교하고, domain shift에 민감한 latent를 찾아 adaptation/intervention에 사용한다.
- 검증 방향: ImageNet, ImageNet-R, synthetic perturbation에서 SAE latent 분포와 activation frequency를 분석하고, 관련 latent 조절이 OOD 성능과 feature reliance profile을 바꾸는지 평가한다.
- 예상 기여: domain-specific feature reliance를 더 세밀한 latent 단위로 측정하고, 해석 가능한 adaptation signal로 연결한다.

### Introduction
- Paragraph 1: ImageNet pretrained model의 성능은 domain이 바뀌면 크게 변하며, 이는 모델이 어떤 visual cue에 의존하는지가 domain마다 달라질 수 있음을 시사한다.
- Paragraph 2: 기존 input perturbation 방식은 grayscale, bilateral filtering, patch shuffle, local warp 등으로 cue를 약화시키지만, perturbation이 feature를 순수하게 분리하지 못하고 복수 cue를 동시에 손상시킨다는 한계가 있다. (실제 논문에서 texture는 Shape score와 Texture score를 동시에 낮추고 있는 걸 보여준다)
- Paragraph 3: 따라서 color/texture/shape라는 coarse category 대신 모델 내부 representation에서 발견되는 더 미세한 feature unit을 분석할 필요가 있다.
(실제로 visudal cue는 다른것도 있다는 걸 그거 논문을 찾아 명시)
- Paragraph 4: SAE를 vision representation, 특히 ViT block token에 적용해 sparse latent vector를 얻고, domain별 activation frequency와 magnitude를 비교하는 접근을 제안한다.
- Paragraph 5: 본 논문의 핵심 질문을 정리한다. 어떤 SAE latent가 domain-invariant인지, 어떤 latent가 domain shift-specific인지, 그리고 latent intervention이 adaptation에 도움을 주는지.
- Paragraph 6: 기여 요약. input perturbation 한계의 실증적 동기화, SAE 기반 feature reliance profile, domain-sensitive latent scoring, latent reweighting/intervention protocol.

### Related Work
- Feature reliance and shortcut learning: texture bias, shape bias, cue conflict, input perturbation 기반 분석을 다룬다.
- Domain generalization/adaptation and TTA: source-free adaptation, test-time adaptation, robustness under natural/synthetic shifts를 배경으로 둔다.
- Sparse autoencoders for representation analysis: NLP mechanistic interpretability에서의 SAE와 vision으로 확장되는 PatchSAE류 연구를 연결한다.
- Representation similarity and perturbation metrics: 코드에서 사용하는 accuracy drop, relative accuracy, JS divergence, CKA, perturbation validation metrics를 관련 평가 축으로 설명한다.

### Method
- Problem setup: pretrained classifier `f`, domain set `D`, perturbation set `P`, internal representation/token `h`, SAE latent `z`를 정의한다.
- **Stage 1**, SAE training: ViT-B/16 block output을 hook으로 수집하고 cls/patch/all token을 정규화한 뒤 Vanilla L1 SAE를 학습한다. loss는 reconstruction MSE 과 L1 sparsity weighted sum.
- **Stage  2** , domain-sensitive latent discovery: ID와 OOD 또는 clean과 perturbed domain 사이의 activation 차이를 기준으로 invariant latent와 shift-sensitive latent를 나눈다.
- **Stage 3**, latent intervention/adaptation: shift-sensitive latent의 activation 또는 decoder contribution을 up/down-weighting하여 classifier feature를 조절하고 OOD 성능 변화를 측정한다.
- **Evaluation**: accuracy/drop, JS divergence, CKA, feature reliance score, latent sparsity, reconstruction NMSE, intervention gain을 보고한다.

### Expected Results and Discussion Themes
- SAE latent profile이 input perturbation보다 더 세밀한 domain 차이를 드러낼 것이라는 예상.
- ImageNet-R 같은 style shift에서는 특정 latent cluster의 frequency가 증가하거나 감소할 가능성.
- latent intervention이 모든 shift에 보편적으로 이득을 주지는 않더라도, 어떤 latent가 성능 저하와 연결되는지 해석 가능한 분석을 제공할 수 있음.
- Vanilla L1 SAE 이후 JumpReLU SAE, Top-K SAE로 확장해 sparsity-quality trade-off와 novelty를 탐색한다.

## Paper B: SAE-Latent Reservoir Routing for Test-Time Adaptation

### Abstract
- 문제: test-time adaptation은 변화하는 domain에서 안정성과 plasticity를 동시에 요구한다.
- 제안: ReservoirTTA의 StyleVec routing을 SAE latent vector로 대체해, sample의 sparse feature activation pattern을 기준으로 specialist reservoir에 routing한다.
- 검증 방향: 동일한 single SAE encoder를 사용해 domain tendency를 추정하고, 유사한 latent pattern의 sample을 같은 specialist에 할당해 adaptation stability와 OOD 성능을 평가한다.
- 예상 기여: hand-crafted style descriptor 대신 model-internal sparse latent descriptor를 사용하는 reservoir-based TTA framework.

### Introduction
- Paragraph 1: 실제 배포 환경에서는 test stream의 domain이 시간에 따라 바뀌며, 하나의 모델을 계속 업데이트하면 catastrophic drift나 unstable adaptation이 발생할 수 있다.
- Paragraph 2: ReservoirTTA는 여러 reservoir specialist를 유지하고 style vector로 sample을 routing해 안정성을 높이는 baseline이다.
- Paragraph 3: 그러나 style vector는 외부적/저차원 style descriptor에 가까워, 모델이 실제로 의존하는 내부 feature 변화와 완전히 일치하지 않을 수 있다.
- Paragraph 4: SAE latent vector는 pretrained vision model의 internal representation을 sparse하고 비교 가능한 pattern으로 변환하므로 routing signal로 적합하다.
- Paragraph 5: 본 논문은 ReservoirTTA의 StyleVec를 SAE latent로 대체하고, latent similarity 기반 routing 및 specialist adaptation을 제안한다.
- Paragraph 6: 기여 요약. single SAE 기반 domain tendency descriptor, SAE latent routing, ReservoirTTA baseline과 직접 비교 가능한 TTA protocol, 효율성/해석성 trade-off 분석.

### Related Work
- Test-time adaptation: entropy minimization, batch normalization adaptation, continual/source-free TTA, stability-plasticity 문제를 다룬다.
- Reservoir/specialist-based adaptation: multiple specialists, routing, memory/reservoir를 사용하는 adaptation 방법을 배경으로 둔다.
- Style-based routing and domain descriptors: style statistics, texture/style shift descriptor, domain clustering 방법과 비교한다.
- Sparse latent representation: SAE가 routing descriptor로 쓰일 때의 장점과 한계, vision SAE/PatchSAE 관련 연구를 연결한다.

### Method
- Problem setup: unlabeled test stream `x_t`, frozen or partially adaptable backbone `f`, SAE encoder `E_sae`, specialist set `{g_k}`를 정의한다.
- SAE descriptor extraction: 입력 sample에서 ViT token 또는 backbone feature를 추출하고 SAE encoder로 sparse latent `z_t`를 얻는다. image-level descriptor는 token latent의 mean/max/frequency pooling으로 만든다.
- Reservoir routing: `z_t`와 reservoir prototypes 사이의 cosine distance, JS divergence, or top-k overlap을 사용해 가장 가까운 specialist를 선택한다.
- Specialist update: 선택된 specialist만 entropy minimization, feature consistency, or pseudo-label objective로 업데이트한다. 나머지 specialist는 drift를 막기 위해 유지한다.
- Reservoir maintenance: 각 specialist의 latent prototype과 memory를 moving average 또는 bounded queue로 갱신한다.
- Comparison to ReservoirTTA: 기존 StyleVec routing을 SAE latent routing으로 교체한 것을 main variant로 두고, StyleVec, random routing, single model TTA와 비교한다.
- Evaluation: ImageNet-C/ImageNet-R/synthetic perturbation stream에서 accuracy, online error, adaptation stability, routing purity, model size, inference time을 측정한다.

### Expected Results and Discussion Themes
- SAE latent routing은 style vector보다 model-internal feature shift를 더 잘 반영해 특정 domain shift에서 routing purity와 adaptation 성능을 높일 가능성.
- Reservoir 구조 덕분에 하나의 모델을 계속 갱신하는 방식보다 안정적일 수 있음.
- 단점으로 specialist 수 증가에 따른 메모리/추론 비용, ReservoirTTA framework 의존으로 인한 novelty 약화, routing-adaptation 결합으로 인한 해석성 감소를 논의한다.
- 향후 JumpReLU/Top-K SAE, lightweight latent pooling, specialist pruning으로 효율성을 개선할 수 있다.

## Shared Experimental Material from Current Code

- Datasets: ImageNet validation, ImageNet-200 subset aligned with ImageNet-R classes, ImageNet-R.
- Models: ResNet50 pretrained on ImageNet-1K, ViT-B/16 pretrained or augmented ImageNet-1K variant.
- Perturbations: original, grayscale, bilateral, patchshuffle, patchrotation, localwarp.
- Metrics: accuracy, relative accuracy, accuracy drop, JS divergence between logits, CKA between representations, feature reliance score, perturbation validation scores, SAE NMSE, cosine reconstruction, mean active latent count.
- Current SAE: Vanilla L1 SAE with ReLU encoder, linear decoder, unit-norm decoder rows, reconstruction MSE plus L1 activation penalty.
- Planned SAE variants: JumpReLU SAE and Top-K SAE.
