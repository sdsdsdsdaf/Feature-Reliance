# SAE Latent 학습 품질 및 Feature Alignment 검증 프로토콜

이 문서는 `paper/refer/`의 SAE 관련 논문들을 바탕으로, Vision/VLM SAE latent가 잘 학습되었는지, 그리고 각 latent가 실제 feature/concept와 정렬되었는지 확인하기 위한 실험 프로토콜을 정리한다. 핵심 관점은 단순히 reconstruction loss가 낮은지 보는 것이 아니라, sparse latent가 인간이 해석 가능한 단일 feature를 안정적으로 잡고, 위치/클래스/도메인/모델 행동과 일관되게 연결되는지 검증하는 것이다.

## 1. 각 논문이 사용한 방식

### 1.1 PatchSAE

PatchSAE는 CLIP ViT의 중간 residual stream activation을 SAE로 복원하면서, patch token 단위의 sparse latent를 학습한다. latent 하나를 하나의 시각 concept 후보로 보고, 해당 latent가 어떤 이미지와 어떤 patch에서 강하게 켜지는지를 분석한다. 이 논문의 중요한 점은 SAE latent를 단순 전역 embedding feature로만 보지 않고, patch-level activation을 spatial attribution mask처럼 사용했다는 점이다.

출처: `paper/refer/PatchSAE.pdf`
- Section 3.1 `PatchSAE Architecture and Training Objectives`: CLIP ViT-B/16 residual stream hook, MSE + L1 objective, all-token 학습, expansion factor, hook layer 설정.
- Section 3.2 `Analysis Method and Evaluation Setup`: top-k reference images, summary statistics, activation frequency, mean activation, label entropy, label standard deviation, patch-level localization.
- Section 4.1 `Impact of SAE Latents on Classification`: top-k SAE latent masking, class-wise representative latent, random/dataset-level ablation 비교.
- Section 4.2 `Understanding Adaptation Mechanisms`: CLIP과 MaPLe의 class-discriminative latent overlap 분석.
- Appendix A.1/A.2/A.3: hyperparameter, layer ablation, 다른 dataset으로의 generalization.

#### 사용한 데이터셋과 방법

- CLIP ViT-B/16을 backbone으로 사용하고, 주로 두 번째 마지막 attention block의 residual stream을 hook한다.
- ImageNet train activation으로 SAE를 학습한다.
- SAE는 MSE reconstruction loss와 L1 sparsity regularizer로 학습한다.
- 모든 token, 즉 CLS token과 image patch token을 함께 사용한다.
- latent별 top-k activating image를 reference image로 저장한다.
- patch activation을 thresholding한 뒤 image-level, class-level, dataset-level activation으로 aggregate한다.
- top-k class-representative latent만 켜거나 끄는 masking experiment를 통해 classification 성능 변화를 본다.
- ImageNet 외 Flowers102, Caltech101, OxfordPets, Food101, DTD, EuroSAT, UCF101 등 여러 downstream dataset에서 latent의 generalization을 확인한다.

#### 사용한 이유와 검증하고자 한 것

- top-k reference image: latent가 사람이 볼 수 있는 일관된 concept를 잡는지 확인한다.
- activation frequency/mean activation: latent가 너무 흔한 noise인지, 드물지만 강한 concept인지 구분한다.
- label entropy와 label standard deviation: latent가 특정 class, 상위 semantic group, texture/style/color 중 무엇을 잡는지 granularity를 추정한다.
- patch-level mask: latent가 실제 이미지 안의 해당 object part나 texture 위치에 켜지는지 확인한다.
- class-level latent masking: latent가 예측에 필요한 class-discriminative 정보를 담는지 확인한다.
- adapted CLIP/MaPLe 비교: adaptation이 새로운 concept를 만드는지, 기존 concept와 class mapping을 바꾸는지 확인한다.

#### 우리 프로토콜에 가져올 점

- latent별 `top activating images + patch heatmap + activation statistics`를 기본 리포트 단위로 둔다.
- class/domain 단위로 latent activation을 aggregate해서, feature reliance가 class/distribution별로 달라지는지 본다.
- top-k latent on/off masking으로 “해석 가능한 latent가 실제 모델 행동에 필요한가”를 검증한다.

### 1.2 Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models

이 논문은 SAE latent의 monosemanticity를 정량화하는 MonoSemanticity score(MS)를 제안한다. 한 neuron 또는 latent가 강하게 반응하는 이미지들이 서로 얼마나 비슷한지를 activation-weighted pairwise similarity로 계산한다. 또한 이 점수가 사람의 판단과 맞는지 user study로 검증하고, monosemantic latent에 개입해 MLLM 출력을 steer할 수 있음을 보인다.

출처: `paper/refer/Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models.pdf`

- Abstract 및 Figure 1: VLM SAE가 monosemantic feature를 학습한다는 주장과 top activating image 비교.
- Section 3.2 `Monosemanticity Score`: pairwise image embedding similarity와 activation weighting으로 MS score를 계산하는 공식.
- Section 3.3 `Steering MLLMs with Vision SAEs`: CLIP vision encoder activation에 SAE를 삽입하고 특정 neuron activation을 조작하는 intervention 방식.
- Section 4.1 `Experimental Settings`: CLIP/SigLIP/AIMv2/WebSSL, ImageNet activation, layer/token 설정.
- Section 4.2.1 `Alignment of MS with human perception`: Mechanical Turk user study와 MS-human alignment rate.
- Section 4.2.2 `Monosemanticity of SAEs`: expansion factor, sparsity, SAE architecture별 MS 비교.
- Appendix B 및 Appendix E.3: human study 상세 절차, iNaturalist taxonomy 기반 hierarchy alignment.

#### 사용한 데이터셋과 방법

- CLIP, SigLIP, AIMv2, WebSSL 등 여러 vision encoder에 SAE를 적용한다.
- ImageNet image activation으로 SAE를 학습하고 평가한다.
- CLIP의 여러 layer, CLS token, 일부 token embedding에 대해 SAE를 학습한다.
- MS score는 evaluation image set의 embedding pairwise cosine similarity를 latent activation의 pairwise product로 가중 평균해 계산한다.
- top-16 activating image grid를 사람에게 보여주고 어느 grid가 더 한 concept에 집중되어 있는지 묻는다.
- LLaVA의 vision encoder activation에 SAE를 삽입한 뒤 특정 latent activation을 고정/증폭해 concept insertion/suppression을 평가한다.
- iNaturalist taxonomy를 사용해 Matryoshka hierarchy와 인간 정의 taxonomy depth의 정렬도 추가로 확인한다.

#### 사용한 이유와 검증하고자 한 것

- MS score: top activating image가 얼마나 한 concept에 모이는지 정량화한다.
- human alignment: 자동 지표가 인간이 보는 monosemanticity와 일치하는지 확인한다.
- expansion factor/sparsity 비교: wider, sparser latent가 더 monosemantic한지 확인한다.
- intervention: latent가 단순히 관찰 가능한 상관 feature가 아니라 모델 출력을 제어하는 causal handle인지 확인한다.
- taxonomy alignment: hierarchy-aware SAE가 coarse-to-fine concept 구조를 실제 생물 분류 체계와 맞게 학습했는지 확인한다.

#### 우리 프로토콜에 가져올 점

- latent별 MS score를 자동 산출한다.
- 상위 latent에 대해 top-16 image grid를 만들고, 필요한 경우 human/LLM-assisted review를 붙인다.
- SAE latent intervention으로 concept insertion/suppression 또는 classifier logit shift를 측정한다.
- hierarchy가 중요한 경우 LCA depth, WordNet distance, taxonomy depth 등을 이용해 coarse/fine alignment를 본다.

### 1.3 Universal Sparse Autoencoders

Universal SAE는 여러 vision model의 activation을 하나의 shared sparse concept space로 정렬한다. 한 모델 activation을 encode한 sparse code가 다른 모델 activation도 복원하도록 학습해, model-specific feature와 universal feature를 구분한다.

출처: `paper/refer/Universal.pdf`

- Figure 1 및 Abstract: multiple pretrained DNN의 activation을 shared concept space로 정렬하는 USAE 개요.
- Section 3.2 `Training USAEs`: 한 모델 encoder로 얻은 shared code를 모든 model-specific decoder로 복원하는 universal loss.
- Section 3.3 `Coordinated Activation Maximization`: 같은 universal concept index를 여러 모델에서 동시에 activation maximization하는 방법.
- Section 4 `Experimental Results` 및 implementation details: DinoV2, SigLIP, ViT final layer activation, ImageNet train/validation 설정.
- Section 4.1 `Universal Concept Visualizations`: concept energy와 universal concept heatmap.
- Section 4.2 `Validation of Cross-Model Reconstruction`: cross-model reconstruction R2 matrix.
- Section 4.3 `Measuring Concept Universality and Importance`: firing entropy, co-fire proportion, energy-cofire correlation.
- Section 4.4 `Concept Consistency Between USAEs and SAEs`: independent SAE와 Universal SAE decoder vector cosine matching 및 Hungarian matching.
- Appendix A.4/A.5: DTD/CelebA OOD generalization과 top co-firing concept consistency.

#### 사용한 데이터셋과 방법

- DinoV2, SigLIP, ViT 등 서로 다른 training objective와 architecture를 가진 vision model을 사용한다.
- ImageNet train set의 final-layer activation으로 Universal SAE를 학습하고 ImageNet validation으로 평가한다.
- 하나의 shared code space를 만들고 model-specific decoder로 각 모델 activation을 복원한다.
- cross-model reconstruction R2를 confusion matrix처럼 측정한다.
- concept energy를 reconstruction contribution으로 정의한다.
- firing entropy와 co-fire proportion으로 concept universality를 측정한다.
- 독립적으로 학습한 SAE와 Universal SAE의 decoder vector를 cosine similarity와 Hungarian matching으로 비교한다.
- coordinated activation maximization으로 같은 universal concept가 각 모델에서 어떤 시각 패턴으로 나타나는지 시각화한다.
- DTD, CelebA에서 OOD generalization을 확인한다.

#### 사용한 이유와 검증하고자 한 것

- cross-model R2: shared latent가 특정 모델 전용 code가 아니라 여러 모델 activation을 설명하는지 확인한다.
- firing entropy: latent가 한 모델에서만 켜지는지, 여러 모델에서 균등하게 켜지는지 확인한다.
- co-fire proportion: 같은 input에서 여러 모델이 같은 latent를 동시에 사용하는지 확인한다.
- energy vs co-fire correlation: 중요한 concept일수록 universal한지 확인한다.
- independent SAE와의 concept overlap: universal training이 기존 단일 모델 SAE concept와 얼마나 일관되는지 확인한다.
- OOD generalization: ImageNet에서 배운 concept가 DTD/CelebA에서도 같은 의미로 켜지는지 확인한다.

#### 우리 프로토콜에 가져올 점

- backbone이 여러 개라면 model별 SAE를 따로 보는 것에 그치지 말고, shared concept인지 model-specific concept인지 구분한다.
- 같은 corruption/domain에서 latent가 여러 model 또는 여러 specialist에서 co-fire하는지 측정한다.
- recurring domain 실험에서는 “반복될 때 같은 latent group이 다시 켜지는가”를 co-fire/entropy 방식으로 평가한다.

### 1.4 Matryoshka SAE

Matryoshka SAE는 단일 top-k나 L1 sparsity 대신 여러 granularity의 top-k objective를 동시에 학습해 coarse-to-fine hierarchical concept를 얻는다. CLIP embedding space에서 reconstruction-sparsity Pareto frontier, concept matching, similarity search, bias analysis를 통해 학습 품질과 concept alignment를 평가한다.

출처: `paper/refer/Matroshoka.pdf`

- Abstract 및 Figure 1: MSAE가 hierarchical coarse-to-fine concept representation을 학습한다는 개요.
- Section 3.2 `Matryoshka SAE Architecture`: 여러 top-k granularity를 동시에 최적화하는 architecture.
- Section 3.3 `Training and Inference`: CLIP embedding normalization, modality별 mean/scale 처리, inference 설정.
- Section 4.1 `Evaluation Metrics`: L0, FVU/EVR, linear probing, CKNNA, decoder orthogonality, dead neuron count.
- Section 4.2-4.8: sparsity-fidelity trade-off, granularity ablation, semantic quality, decoder orthogonality, progressive recovery, modality ablation.
- Appendix A `Concept Detection and Validation`: CLIP vocabulary matching, cosine similarity threshold `> 0.42`, top/second ratio `> 2.0`, one concept per neuron, manual semantic consistency.
- Appendix E `CKNNA`: local neighborhood alignment metric 상세.
- Appendix G.3 `Gender Bias Analysis in CelebA`: concept magnitude distribution, highest activation images, concept manipulation.

#### 사용한 데이터셋과 방법

- CC3M train set에서 CLIP ViT-L/14 또는 ViT-B/16 embedding으로 SAE를 학습한다.
- ImageNet-1K image modality와 CC3M text modality에서 평가한다.
- FVU/EVR, cosine similarity, linear probing, CKNNA, decoder orthogonality, dead neuron count를 사용한다.
- predefined vocabulary의 CLIP text embedding과 SAE decoder column의 cosine similarity로 neuron-concept matching을 한다.
- validation threshold로 cosine similarity `> 0.42`, top similarity / second-highest similarity `> 2.0`, one concept per neuron 조건을 사용한다.
- top activating ImageNet/CC3M examples로 valid/invalid concept를 수동 확인한다.
- CelebA에서 gender-associated concept activation 분포와 concept manipulation을 사용해 bias alignment를 확인한다.

#### 사용한 이유와 검증하고자 한 것

- FVU/EVR/cosine similarity: SAE가 원 activation을 충실히 복원하는지 확인한다.
- L0/dead neuron: sparse code가 적절히 쓰이고, 죽은 latent가 많지 않은지 확인한다.
- CKNNA: SAE activation space가 원 CLIP embedding의 local neighborhood geometry를 보존하는지 확인한다.
- decoder orthogonality: latent direction들이 과도하게 중복되지 않는지 확인한다.
- CLIP vocabulary matching: latent direction이 텍스트 concept와 직접 정렬되는지 확인한다.
- threshold + ratio: 최고 similarity만 보고 생기는 spurious assignment를 줄인다.
- concept manipulation: 특정 concept 크기를 바꿨을 때 downstream classifier가 예측을 어떻게 바꾸는지 확인한다.

#### 우리 프로토콜에 가져올 점

- reconstruction 지표와 interpretability 지표를 분리해서 둘 다 통과해야 한다.
- decoder-text concept matching은 자동 라벨링의 1차 후보로 쓰되, threshold/ratio/manual check를 반드시 붙인다.
- downstream classifier 또는 TTA specialist에서 concept magnitude가 prediction과 통계적으로 연결되는지 확인한다.

### 1.5 MP-SAE

MP-SAE는 일반 SAE가 가정하는 “feature는 독립적이고 선형적으로 접근 가능하다”는 전제의 한계를 지적하고, Matching Pursuit 방식의 residual-guided sequential encoder로 hierarchical/nonlinear feature를 복원한다. 이 논문은 특히 feature absorption과 hierarchical feature recovery를 검증하는 synthetic benchmark를 잘 제안한다.

출처: `paper/refer/MPSAE.pdf`

- Abstract 및 Introduction: standard SAE의 LRH 가정 한계, hierarchical/nonlinear/multidimensional feature 문제 제기.
- Section 2 `Formalizing Linear Representation Hypothesis and Sparse Autoencoders`: LRH, overcomplete dictionary, quasi-orthogonality, sparsity 전제.
- Section 3: Matching Pursuit 기반 residual-guided sequential encoder, conditional orthogonality, nonlinear accessible feature 설명.
- Section 4.1 `A Synthetic Generative Model of Hierarchical Features`: parent-child hierarchical generative process, feature absorption, flat MSE/hierarchical MSE.
- Section 4.2 `Representations from Pretrained Models`: ImageNet-1K activation, CLIP/DINOv2/SigLIP/ViT 비교, R2-sparsity Pareto, effective rank, Babel/coherence, inference-time sparsity sweep.
- Appendix B.1: synthetic tree data generation, activation probability, firing magnitude variance, training details.
- Appendix B.2/B.3: large vision experiment setup, co-activation effective rank, dictionary coherence.
- Appendix B.4: COCO image-caption embedding, modality score, multimodal concept recovery.

#### 사용한 데이터셋과 방법

- synthetic hierarchical tree를 만든다. parent concept가 켜져야 child concept가 켜질 수 있는 조건부 activation 구조를 둔다.
- ground-truth dictionary와 sparse code가 있으므로 learned dictionary와 직접 비교한다.
- Vanilla SAE, BatchTopK, Matryoshka SAE, MP-SAE를 비교한다.
- learned dictionary self-similarity, ground-truth alignment, sparse code recovery를 본다.
- flat MSE와 hierarchical MSE로 intra-level correlation과 inter-level separation을 평가한다.
- ImageNet-1K final-layer activation으로 CLIP, DINOv2, SigLIP, ViT SAE를 학습해 R2-sparsity Pareto frontier를 비교한다.
- effective rank of co-activation, Babel/coherence score, inference-time sparsity 변화에 따른 monotonic reconstruction을 평가한다.
- COCO image-caption embedding으로 multimodal concept recovery와 modality score를 평가한다.

#### 사용한 이유와 검증하고자 한 것

- synthetic tree: SAE가 계층 feature를 실제로 분리하는지, child가 parent에 흡수되는 feature absorption이 생기는지 확인한다.
- ground-truth alignment: 해석 가능한 feature recovery를 정량적으로 검증한다.
- effective rank: active feature 조합이 다양하고 중복되지 않는지 확인한다.
- conditional coherence: 전체 dictionary는 유사한 feature를 포함해도, 같은 input에서 함께 선택되는 feature들이 서로 덜 간섭하는지 확인한다.
- inference-time sparsity sweep: k를 바꿔도 reconstruction이 안정적으로 좋아지는지 확인한다.
- modality score: VLM에서 image-only/text-only feature로 분리되지 않고 multimodal shared concept를 잡는지 확인한다.

#### 우리 프로토콜에 가져올 점

- 실제 데이터만으로는 alignment 정답이 없으므로, synthetic controlled benchmark를 별도로 둬 SAE training recipe의 feature recovery 능력을 먼저 검증한다.
- hierarchical feature가 예상되는 경우 parent-child absorption을 체크한다.
- latent co-activation matrix와 effective rank로 feature 사용 패턴의 다양성을 본다.

### 1.6 Interpretable and Steerable Concept Bottleneck Sparse Autoencoders (CB-SAE)

CB-SAE는 기존 SAE latent가 “해석 가능한가”와 “실제로 downstream output을 조작할 수 있는가”를 분리해서 측정한다. 논문은 많은 SAE neuron이 둘 중 하나 또는 둘 다 낮으며, 사용자가 원하는 concept가 SAE latent 공간에 없을 수도 있다는 점을 보인다. 그래서 먼저 기존 SAE neuron별 interpretability score와 steerability score를 계산하고, 낮은 neuron을 pruning한 뒤, 빠진 user-specified concept를 concept bottleneck neuron으로 보강한다.

출처: `paper/refer/Kulkarni_Interpretable_and_Steerable_Concept_Bottleneck_Sparse_Autoencoders_CVPR_2026_paper.pdf`

- Abstract 및 Figure 1: low-utility SAE neuron pruning, user-defined concept bottleneck 추가, interpretability와 steerability 개선 주장.
- Section 3 `Background`: SAE 학습식, CLIP-Dissect 기반 neuron interpretability, LLaVA steering output 기반 steerability score 정의.
- Section 4 `Interpretability vs Steerability in SAEs`: 65,536개 SAE neuron을 interpretability/steerability 2축으로 분석하고 네 quadrant로 분류.
- Section 4 Expt. 1: CLIP-ViT-L/14-336 layer 22, ImageNet-1K activation, Matryoshka Batch Top-k SAE expansion factor 64, Broden 1,197 concepts 사용.
- Section 4 Expt. 2: Broden, VLG-CBM, DECIDER, common English words concept set에 대한 concept coverage 측정.
- Section 5 `Our Approach: CB-SAE`: `I + S`가 낮은 neuron pruning, user-specified concept set 중 retained SAE에 없는 concept만 CB-SAE concept로 사용.
- Section 5.2: reconstruction loss, CLIP zero-shot pseudo concept activation을 이용한 interpretability loss, cyclic reconstruction 기반 steerability loss.
- Section 6.1 `Setup` 및 `Evaluation Metrics`: CLIP-Dissect score, MS score, Unit-Vec steering, White Image steering, Sentence-BERT/DINOv2 similarity 기반 평가.
- Table 1/2 및 Figure 5/6: baseline SAE, discarded neuron, retained neuron, CB neuron의 interpretability/steerability 비교.

#### 사용한 데이터셋과 방법

- baseline SAE는 ImageNet-1K의 CLIP-ViT-L/14-336 중간 activation으로 학습한다.
- concept labeling에는 Broden 1,197개 concept를 사용한다.
- concept coverage 분석에는 Broden, VLG-CBM, DECIDER, 3k/20k common English words를 사용한다.
- interpretability는 CLIP-Dissect로 neuron별 best concept와 score를 산출한다.
- steerability는 특정 neuron을 강하게 활성화한 뒤 downstream model output이 CLIP-Dissect가 붙인 concept와 의미적으로 가까워지는지 본다.
- text steering에서는 LLaVA output과 assigned concept의 Sentence-BERT similarity를 사용한다.
- image steering에서는 UnCLIP output과 해당 neuron의 top-16 activating image 사이 DINOv2 embedding similarity를 사용한다.
- pruning은 neuron별 interpretability score `I`와 steerability score `S`를 더한 값이 낮은 bottom-M neuron을 제거하는 방식이다.
- CB-SAE는 retained SAE를 freeze하고, user concept set 중 retained SAE가 이미 잘 표현하는 concept를 제외한 나머지를 concept bottleneck neuron으로 학습한다.

#### 사용한 이유와 검증하고자 한 것

- top activating image가 그럴듯해도 neuron이 downstream output을 조작하지 못할 수 있음을 보인다.
- 반대로 output을 크게 바꾸는 neuron이 사람이 이해할 수 있는 concept와 정렬되지 않을 수 있음을 보인다.
- SAE latent 공간이 사용자가 요구하는 concept set을 충분히 cover하는지 별도로 측정한다.
- low interpretability/low steerability neuron은 downstream explanation/control에 적합하지 않으므로 pruning 후보로 둔다.
- 누락된 concept는 기존 SAE latent에 억지로 이름을 붙이지 않고, concept bottleneck처럼 명시적 concept channel을 추가해야 한다는 결론을 제시한다.

#### 우리 프로토콜에 가져올 점

- feature alignment 판정에 `interpretability score`와 `steerability score`를 분리해서 기록한다.
- latent를 `high I/high S`, `high I/low S`, `low I/high S`, `low I/low S` 네 그룹으로 나눈다.
- user-specified concept list를 만들고, 각 concept가 적어도 하나의 latent로 cover되는지 concept coverage를 계산한다.
- cover되지 않은 concept는 “해당 feature latent가 발견되지 않음”으로 보고, label forcing이나 top-image cherry-picking을 하지 않는다.
- 실제 프로젝트에서는 CB-SAE 전체 학습을 바로 하지 않더라도, `I + S` 기반 retained/prune 후보와 missing concept 목록을 반드시 report에 포함한다.

## 2. 공통점 및 차이점

### 2.1 공통점

- 모든 논문은 reconstruction 품질만으로 SAE를 평가하지 않는다. MSE, FVU, EVR, R2 같은 fidelity 지표는 필수지만, 이것만으로 “latent가 잘 학습되었다”고 결론 내리지 않는다.
- top activating examples는 거의 모든 논문에서 가장 기본적인 interpretability check로 사용된다. latent가 강하게 켜지는 이미지/patch/text가 같은 concept로 묶이는지 본다.
- sparsity는 단순히 낮을수록 좋은 값이 아니라, reconstruction과 interpretability 사이의 trade-off로 평가된다.
- latent가 model behavior와 연결되는지 확인한다. masking, ablation, intervention, steering, classifier probability shift 등이 사용된다.
- interpretability와 steerability는 같은 값이 아니다. 사람이 이름 붙일 수 있는 latent라도 output control에는 약할 수 있고, output을 강하게 바꾸는 latent라도 의미가 불명확할 수 있다.
- 자동 concept labeling은 threshold와 수동 검증이 필요하다. vocabulary similarity만으로는 hierarchy, synonym, polysemy 때문에 오류가 생긴다.
- user-specified concept coverage를 별도로 본다. 평가자가 관심 있는 feature가 latent 공간에 없으면, top image가 아무리 좋아도 그 feature와 align되었다고 말할 수 없다.
- domain/model generalization을 본다. ImageNet에서 학습한 latent가 다른 dataset, 다른 model, 다른 modality에서도 같은 의미로 켜지는지 확인한다.
- feature alignment는 여러 해상도에서 본다. patch-level, image-level, class-level, dataset-level, model-level, hierarchy-level이 함께 쓰인다.

### 2.2 차이점

| 논문 | 주요 질문 | 핵심 검증 | 강점 | 주의점 |
| --- | --- | --- | --- | --- |
| PatchSAE | patch-level latent가 visual concept와 위치적으로 정렬되는가 | top images, patch heatmap, class latent masking | spatial attribution과 model behavior 연결이 강함 | 정량 monosemanticity는 제한적 |
| VLM SAE monosemanticity | latent가 단일 concept에 집중되는가 | MS score, human study, MLLM steering | 자동 점수와 인간 판단 연결 | pairwise similarity encoder 선택에 민감 |
| Universal SAE | 여러 model이 공유하는 concept인가 | cross-model R2, firing entropy, co-fire, concept overlap | model-specific vs universal feature 구분 | shared space 학습 비용이 큼 |
| Matryoshka SAE | coarse-to-fine hierarchy와 좋은 Pareto frontier를 얻는가 | EVR/FVU, CKNNA, vocabulary matching, manipulation | 실용적 metric set과 threshold 제안 | vocabulary coverage에 따라 valid concept 수가 달라짐 |
| MP-SAE | 비선형/계층 feature를 기존 SAE가 놓치는가 | synthetic ground truth, absorption, effective rank, modality score | feature recovery 검증이 가장 엄밀함 | 실제 실험에는 architecture 변경이 필요 |
| CB-SAE | 해석 가능한 latent가 실제로 조작 가능하고, 원하는 concept를 cover하는가 | CLIP-Dissect, LLaVA/UnCLIP steering, concept coverage, pruning | interpretability와 steerability를 분리하고 missing concept를 명시 | CLIP/LVLM 기반 평가 의존성이 있고 concept set 선택에 민감 |

### 2.3 우리 문제에 대한 해석

SAE latent가 “잘 학습되었다”는 말은 최소 네 가지를 동시에 만족해야 한다.

1. Activation fidelity: SAE reconstruction이 원 모델 activation의 정보를 충분히 보존한다.
2. Concept alignment: 각 latent가 top activating examples, spatial mask, text/vocab label, class/domain statistics에서 일관된 feature로 해석된다.
3. Behavioral relevance: latent를 제거하거나 증폭했을 때 classifier/TTA specialist/routing decision이 예측 가능한 방향으로 변한다.
4. Concept coverage: 우리가 검증하려는 feature set의 각 concept가 latent 공간 안에 실제로 존재하거나, 없다는 사실이 명시적으로 보고된다.

따라서 reservoir SAE나 feature reliance 분석에서는 latent를 routing vector로 쓰기 전에, latent가 domain corruption, object part, texture, class-discriminative feature 중 무엇을 잡는지 먼저 검증해야 한다.

## 3. 최종 프로토콜

### 3.1 목표

최종 목표는 SAE latent별로 다음 질문에 답하는 것이다.

- 이 latent는 reconstruction에 기여하는가?
- 이 latent는 sparse하고 죽지 않았는가?
- 이 latent가 강하게 켜지는 sample들은 같은 feature를 공유하는가?
- 이 latent는 patch 위치, class label, domain/corruption label, text concept와 정렬되는가?
- 이 latent를 조작하면 모델 예측, routing, adaptation behavior가 바뀌는가?
- 이 latent는 해석 가능성과 조작성 중 어느 쪽을 만족하는가?
- 관심 concept set 중 어떤 concept가 latent 공간에 있고 어떤 concept가 빠져 있는가?
- 같은 domain이 반복될 때 같은 latent profile이 재등장하는가?

### 3.2 Step A: 학습 품질 기본 점검

#### 측정 항목

- Reconstruction MSE
- FVU 또는 Normalized MSE
- EVR 또는 R2
- cosine similarity between original activation and reconstructed activation
- L0 또는 active latent count
- activation frequency
- mean positive activation
- dead latent ratio
- decoder cosine similarity 또는 coherence

#### 판정 기준

- 원 activation을 SAE reconstruction으로 대체했을 때 downstream accuracy 또는 CLIP similarity가 크게 무너지지 않아야 한다.
- dead latent가 과도하게 많으면 sparsity coefficient, top-k, expansion factor를 재조정한다.
- activation frequency가 너무 높은 latent는 generic/background/noise 후보로 분류하고, 너무 낮은 latent는 rare concept 또는 dead/unstable latent로 분리한다.
- fidelity가 높아도 monosemanticity가 낮으면 routing feature로 바로 쓰지 않는다.

### 3.3 Step B: latent별 reference set 구축

각 latent `z_k`에 대해 다음 artifact를 저장한다.

- top-16 또는 top-32 activating images
- activation value와 rank
- patch/token heatmap
- image-level activation count
- class/domain/corruption/severity label 분포
- label entropy
- WordNet 또는 taxonomy distance/std
- mean positive activation
- activation frequency

#### 판정 기준

- top activating images가 시각적으로 한 concept, object part, texture, color, corruption pattern, domain style 중 하나로 묶이면 valid 후보로 둔다.
- top images가 서로 무관하거나, label entropy와 visual evidence가 모두 불안정하면 polysemantic/noisy 후보로 둔다.
- high frequency + low visual consistency latent는 routing에 쓰지 않거나 down-weight한다.

### 3.4 Step C: MonoSemanticity score 계산

VLM-SAE 논문 방식을 따라 latent별 MS score를 계산한다.

1. 평가 이미지 `N`개를 모은다.
2. 별도 image encoder, 예를 들어 CLIP/DINOv2 embedding으로 pairwise cosine similarity matrix `S`를 만든다.
3. latent activation vector `a_k`를 min-max normalize한다.
4. relevance `R_nm = a_k,n * a_k,m`를 계산한다.
5. diagonal을 제외하고 activation-weighted pairwise similarity를 평균해 `MS_k`를 얻는다.

#### 권장 리포트

- latent별 MS score
- MS 상위/하위 latent의 top-16 image grid
- MS와 activation frequency의 scatter plot
- MS와 label entropy의 scatter plot
- MS와 reconstruction contribution의 scatter plot

#### 판정 기준

- high MS + consistent top images: monosemantic latent
- high MS + single dataset/class overfit: narrow but possibly useful latent
- low MS + high activation frequency: generic/polysemantic latent
- low MS + high reconstruction contribution: 중요한데 해석이 어려운 latent이므로 단독 routing feature로 쓰기 전에 추가 검증 필요

### 3.5 Step D: text/vocabulary concept alignment

Matryoshka SAE 방식을 따라 decoder direction 또는 latent activation prototype을 text concept와 매칭한다.

1. candidate vocabulary를 만든다.
   - object: ImageNet class, WordNet synset
   - part: nose, wing, wheel, face, texture 등
   - style/domain: blur, snow, fog, sketch, texture, satellite 등
   - color/material: red, blue, metal, wood 등
   - corruption: Gaussian noise, defocus blur, frost 등
2. 각 vocab phrase를 CLIP text embedding으로 변환한다.
3. SAE decoder column 또는 latent-conditioned image prototype과 cosine similarity를 계산한다.
4. top concept와 second concept의 ratio를 계산한다.
5. threshold를 적용한다.

#### 1차 threshold

- cosine similarity `> 0.42`
- top similarity / second similarity `> 2.0`
- one primary concept per latent

이 값은 Matryoshka SAE의 기준을 시작점으로 둔 것이므로, backbone/dataset이 다르면 validation set에서 calibration한다.

#### 판정 기준

- threshold 통과 + top activating images와 일치: aligned concept
- threshold 통과 + image evidence 불일치: text false positive
- threshold 미통과 + image evidence 일관: vocabulary coverage 부족 가능
- top/second ratio 낮음: hierarchical 또는 synonym cluster 후보

### 3.6 Step E: spatial alignment

PatchSAE 방식을 사용한다.

1. patch/token activation을 원 이미지 grid로 reshape한다.
2. heatmap 또는 soft segmentation mask를 만든다.
3. 가능하면 segmentation mask, bounding box, saliency map, human-visible object location과 비교한다.
4. object/part latent의 경우 top activation patch가 해당 object/part에 위치하는지 본다.
5. corruption/style latent의 경우 전역적으로 켜지는지, 특정 texture 영역에서 켜지는지 본다.

#### 판정 기준

- object part latent: heatmap이 해당 part에 국소적으로 집중되어야 한다.
- color/texture latent: 해당 color/texture region과 겹쳐야 한다.
- corruption/domain latent: 여러 object에 걸쳐 안정적으로 켜질 수 있으며, class label보다 corruption/severity label과 더 강하게 연결될 수 있다.

### 3.7 Step F: class/domain/distribution alignment

latent activation을 group 단위로 aggregate한다.

- class-level activation
- corruption-level activation
- severity-level activation
- domain-level activation
- recur-index별 activation
- reservoir specialist별 activation

#### 분석

- group별 top latent를 뽑는다.
- latent activation distribution의 entropy를 계산한다.
- 같은 class이지만 다른 corruption에서 유지되는 latent와, class와 무관하게 corruption에서 켜지는 latent를 분리한다.
- recurring stream에서는 같은 domain이 반복될 때 같은 latent profile이 재등장하는지 cosine similarity 또는 Jaccard overlap으로 본다.

#### 판정 기준

- class-discriminative latent: class label 기준 entropy가 낮고 top-k masking 때 accuracy에 영향이 크다.
- domain/corruption latent: class entropy는 높지만 corruption/domain entropy가 낮다.
- recurring-domain latent: recur index가 달라도 같은 domain segment에서 profile similarity가 높다.

### 3.8 Step G: behavioral relevance 검증

SAE latent가 모델 행동에 실제로 영향을 주는지 확인한다.

#### Masking

- class/domain별 top-k latent만 on
- class/domain별 top-k latent만 off
- random k latent off와 비교
- dataset-level frequent latent off와 비교

#### Intervention

- latent activation을 `0`, mean, high quantile, fixed alpha 등으로 조작한다.
- decoder로 복원한 activation을 원 모델 중간 activation에 대체한다.
- classifier logit, CLIP similarity, TTA routing decision, reservoir specialist assignment 변화를 기록한다.

#### 판정 기준

- aligned latent를 끄면 관련 class/domain 성능이 떨어져야 한다.
- unrelated latent를 끄는 random baseline보다 효과가 커야 한다.
- latent를 키웠을 때 관련 concept의 logit/similarity/routing probability가 증가해야 한다.
- intervention이 unrelated output을 과도하게 망치면 latent가 entangled 되었을 가능성이 있다.

### 3.9 Step H: concept coverage 및 interpretability-steerability quadrant

CB-SAE 논문 방식을 따라, 각 latent의 “이름 붙일 수 있음”과 “모델 행동을 바꿀 수 있음”을 분리해 기록한다. 이 단계는 feature-latent alignment를 주장하기 전에, 관심 feature가 SAE latent 공간에 실제로 존재하는지 확인하는 broadened dataset/concept 검증이다.

#### 입력

- user-specified concept set `C_user`
  - class concept: ImageNet/Imagenette class, WordNet synset
  - visual concept: Broden object/part/material/texture/color/scene
  - domain concept: blur, noise, frost, sketch, texture, satellite 등
  - project-specific feature: 사용자가 분석하려는 feature reliance 후보
- latent별 top image, heatmap, MS score, class/domain entropy
- 가능하면 Broden IoU 또는 CLIP-Dissect score
- Step G의 masking/intervention 결과

#### Interpretability score

가능한 경우 CB-SAE처럼 CLIP-Dissect score를 사용한다. CLIP-Dissect를 바로 쓸 수 없는 non-CLIP ViT에서는 다음 proxy를 calibration해서 쓴다.

- Broden concept-mask IoU
- top activating image의 semantic consistency
- MS score
- class/domain label purity
- human/LLM-assisted review agreement

#### Steerability score

downstream task에 맞춰 latent 조작 효과를 측정한다.

- classifier: latent activation을 0/mean/high quantile/fixed alpha로 바꾸고 target class logit 또는 probability shift 측정
- VLM: steered text output과 assigned concept의 sentence embedding similarity
- image generation: steered image와 top activating reference image의 embedding similarity
- routing/TTA: routing probability 또는 specialist assignment 변화

현재 모델이 classifier라면 `target logit increase - random latent baseline`을 0-1로 normalize해 steerability proxy로 둔다.

#### Concept coverage

각 concept `c in C_user`에 대해 다음을 계산한다.

```text
covered(c) = exists latent k such that
  assigned_concept(k) == c
  and interpretability_score(k) >= tau_I
  and, if behavior is required, steerability_score(k) >= tau_S
```

coverage report에는 다음을 반드시 포함한다.

- covered concept list
- missing concept list
- concept별 best latent id와 score
- concept별 top image/heatmap/Broden overlay
- concept가 missing인 이유 후보: dataset에 없음, vocabulary에 없음, SAE capacity 부족, feature가 distributed representation임

#### Quadrant 판정

| 그룹 | 의미 | 사용 방침 |
| --- | --- | --- |
| High I / High S | 해석 가능하고 조작 가능한 latent | feature reliance/routing/control 후보 |
| High I / Low S | 사람이 이해하기 쉽지만 행동 영향은 약함 | 설명 artifact로 사용, causal claim 금지 |
| Low I / High S | 행동 영향은 크지만 의미가 불명확함 | 추가 해석 필요, 단독 feature label 금지 |
| Low I / Low S | 해석/조작성 모두 낮음 | pruning 또는 ignore 후보 |

#### 판정 기준

- feature alignment를 강하게 주장하려면 `high interpretability + independent evidence + non-random steerability`가 필요하다.
- 관심 concept가 missing이면, 가장 가까운 latent에 억지 label을 붙이지 않는다.
- missing concept가 프로젝트 핵심 feature라면 dataset을 넓히거나, supervised/concept-bottleneck 보강을 별도 실험으로 설계한다.

### 3.10 Step I: cross-model 또는 cross-specialist alignment

Universal SAE 방식을 reservoir/TTA 상황에 맞게 변형한다.

#### 분석

- model 또는 specialist별 latent prototype을 만든다.
- 같은 input/domain에서 latent co-fire proportion을 계산한다.
- firing entropy로 universal latent와 specialist-specific latent를 구분한다.
- independent SAE 또는 specialist별 SAE를 쓴다면 decoder vector cosine similarity와 Hungarian matching으로 concept overlap을 측정한다.
- shared latent의 reconstruction contribution energy를 계산한다.

#### 판정 기준

- universal latent: 여러 model/specialist에서 균등하게 자주 co-fire한다.
- specialist-specific latent: 특정 specialist/domain에서만 강하게 켜진다.
- high energy + high co-fire latent: 안정적 routing feature 후보
- high energy + low co-fire latent: 특정 specialist/domain diagnostic feature 후보

### 3.11 Step J: synthetic controlled benchmark

MP-SAE 논문처럼 ground-truth concept가 있는 작은 synthetic dataset을 만든다.

#### 구성

- parent-child hierarchy를 갖는 synthetic feature tree를 만든다.
- parent가 켜져야 child가 켜지는 조건부 activation을 둔다.
- parent와 child firing magnitude는 완전 상관이 되지 않게 variance를 둔다.
- intra-level correlation을 0, 0.3, 0.6, 0.9처럼 조절한다.

#### 평가

- learned decoder와 ground-truth dictionary cosine alignment
- support recovery
- sparse code magnitude recovery
- feature absorption score
- flat MSE
- hierarchical MSE
- effective rank of co-activation

#### 판정 기준

- child feature가 parent direction에 흡수되면 hierarchy recovery 실패다.
- 같은 level correlation은 보존하되 parent-child 분리는 유지되어야 한다.
- 실제 데이터 실험 전에 SAE architecture와 sparsity setting을 이 benchmark에서 먼저 고른다.

## 4. 최종 판정표

| 항목 | 통과 조건 | 실패 시 해석 | 조치 |
| --- | --- | --- | --- |
| Reconstruction | EVR/R2 높고 downstream 성능 보존 | activation 정보 손실 | expansion/lr/loss/layer 재조정 |
| Sparsity | active count가 안정적이고 dead latent 낮음 | 너무 dense 또는 dead feature 많음 | sparsity coefficient/top-k 조정 |
| Monosemanticity | MS 높고 top images 일관 | polysemantic latent | routing feature에서 제외 또는 분해 |
| Text alignment | cosine/ratio threshold 통과 | vocab mismatch 또는 spurious match | vocab 확장, manual check |
| Spatial alignment | heatmap이 feature 위치와 일치 | shortcut/background 가능성 | segmentation/occlusion 추가 검증 |
| Class/domain alignment | label/domain entropy가 목적과 일치 | feature가 목표 factor와 불일치 | group-level top latent 재선정 |
| Behavioral relevance | masking/intervention 효과가 random보다 큼 | 관찰 상관일 뿐 causal하지 않음 | latent set 재선정 |
| Interpretability/steerability | high I/high S latent가 충분함 | 설명 가능하지만 causal하지 않거나, causal하지만 해석 불가 | quadrant별로 용도 분리 |
| Concept coverage | 관심 concept가 하나 이상의 latent로 cover됨 | 원하는 feature가 latent 공간에 없음 | broaden dataset, concept bottleneck, 재학습 |
| Generalization | OOD/recur에서도 같은 concept 유지 | dataset-specific artifact | train data 다양화 |
| Cross-model/specialist | co-fire/entropy가 목적에 맞음 | model-specific collapse | universal/shared training 또는 normalization |

## 5. 권장 실험 산출물

각 SAE run마다 다음 파일을 저장한다.

- `metrics/reconstruction.json`: MSE, FVU, EVR, R2, cosine similarity
- `metrics/sparsity.json`: L0, frequency, mean activation, dead latent ratio
- `metrics/monosemanticity.csv`: latent id, MS, frequency, mean activation, entropy
- `metrics/concept_alignment.csv`: latent id, top vocab, cosine, ratio, validation flag
- `metrics/group_alignment.csv`: class/domain/corruption/severity/recur별 top latent
- `metrics/intervention.csv`: latent id, manipulation value, output/logit/routing change
- `metrics/interpretability_steerability.csv`: latent id, interpretability score, steerability score, quadrant, retained/prune flag
- `metrics/concept_coverage.csv`: concept, covered flag, best latent id, best score, missing reason
- `figures/top_images/latent_{k}.png`
- `figures/heatmaps/latent_{k}.png`
- `figures/scatter_interpretability_steerability.png`
- `figures/scatter_ms_entropy.png`
- `figures/scatter_energy_cofire.png`
- `figures/recur_profile_similarity.png`

## 6. 프로토콜 단계별 출처 매핑

| 프로토콜 단계 | 주 출처 | 참고 위치 | 가져온 핵심 아이디어 |
| --- | --- | --- | --- |
| Step A 학습 품질 기본 점검 | `Matroshoka.pdf`, `MPSAE.pdf`, `Universal.pdf` | Matryoshka Section 4.1/4.2, MP-SAE Section 4.2, Universal Section 4.2 | FVU/EVR/R2, sparsity-fidelity Pareto, dead neurons, reconstruction preservation |
| Step B reference set 구축 | `PatchSAE.pdf`, `Matroshoka.pdf` | PatchSAE Section 3.2, Matryoshka Appendix A/G | top activating images, activation frequency, mean activation, label entropy, manual semantic consistency |
| Step C MonoSemanticity score | `Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models.pdf` | Section 3.2, Section 4.2.1, Appendix B | activation-weighted pairwise similarity, top-16 grid, human judgment validation |
| Step D text/vocabulary alignment | `Matroshoka.pdf` | Appendix A `Concept Detection and Validation` | CLIP text embedding과 decoder vector cosine matching, `> 0.42`, ratio `> 2.0`, one concept per neuron |
| Step E spatial alignment | `PatchSAE.pdf`, `Universal.pdf` | PatchSAE Section 3.2/Figure 2, Universal Section 4.1/Figure 4 | patch/token heatmap, concept localization, universal concept heatmap |
| Step F class/domain/distribution alignment | `PatchSAE.pdf`, `Universal.pdf` | PatchSAE Section 3.2/4.1/4.2, Universal Section 4.3/Appendix A.4 | class-level/dataset-level activation, firing entropy, co-fire proportion, OOD generalization |
| Step G behavioral relevance | `PatchSAE.pdf`, `Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models.pdf`, `Matroshoka.pdf` | PatchSAE Section 4.1, VLM-SAE Section 3.3/4.3, Matryoshka Appendix G.3 | latent masking, intervention, steering, classifier probability shift |
| Step H concept coverage 및 interpretability-steerability quadrant | `Kulkarni_Interpretable_and_Steerable_Concept_Bottleneck_Sparse_Autoencoders_CVPR_2026_paper.pdf` | Section 3, Section 4 Expt. 1/2, Section 5, Section 6.1, Table 1/2 | CLIP-Dissect, steering score, four-quadrant neuron 분류, concept coverage, low-utility pruning |
| Step I cross-model/specialist alignment | `Universal.pdf`, `MPSAE.pdf` | Universal Section 3.2/4.2/4.3/4.4, MP-SAE Appendix B.4 | shared concept space, cross-model reconstruction, firing entropy, co-fire, modality score |
| Step J synthetic controlled benchmark | `MPSAE.pdf` | Section 4.1, Appendix B.1 | parent-child hierarchy, feature absorption, support recovery, ground-truth dictionary alignment |

## 7. 최종 제언

SAE latent를 feature reliance 또는 reservoir routing에 쓰려면, 먼저 “복원 잘 됨”과 “해석 가능함”을 분리해 평가해야 한다. 복원 지표가 좋은 SAE도 polysemantic latent를 만들 수 있고, top image가 그럴듯한 latent도 실제 prediction이나 routing에는 영향이 없을 수 있다. 따라서 최종 프로토콜은 다음 순서를 권장한다.

1. reconstruction/sparsity Pareto로 사용할 SAE checkpoint를 고른다.
2. top activating images, MS score, label/domain entropy로 latent를 1차 필터링한다.
3. text vocabulary matching과 patch heatmap으로 feature 이름과 위치 정렬을 확인한다.
4. class/domain/corruption/recur 단위 activation profile로 어떤 latent가 어떤 factor를 잡는지 분류한다.
5. masking/intervention으로 behavioral relevance를 확인한다.
6. CB-SAE 방식으로 interpretability/steerability quadrant와 user concept coverage를 산출한다.
7. missing concept는 “아직 발견되지 않은 feature”로 보고, 데이터 확장이나 concept bottleneck 보강 후보로 남긴다.
8. recurring domain stream에서는 같은 domain이 돌아왔을 때 latent profile과 reservoir specialist assignment가 함께 재현되는지 본다.
9. 최종적으로 routing에는 `high MS`, `stable activation`, `domain/recur alignment`, `behavioral relevance`, `high interpretability/high steerability`를 모두 만족하는 latent subset만 사용한다.

우리 프로젝트의 Reservoir SAE 목적에는 특히 `PatchSAE의 patch/class aggregation`, `VLM-SAE의 MS score`, `Network Dissection/Broden의 mask IoU`, `CB-SAE의 interpretability/steerability 및 concept coverage`, `Universal SAE의 co-fire/firing entropy`, `Matryoshka SAE의 vocabulary threshold`, `MP-SAE의 hierarchy/absorption check`를 조합하는 것이 가장 적합하다. 이 조합은 SAE latent가 domain shift를 설명하는 feature인지, 단순 class shortcut인지, 또는 recurring domain을 안정적으로 식별하는 routing signal인지 구분하는 데 필요한 최소 검증 세트다.
