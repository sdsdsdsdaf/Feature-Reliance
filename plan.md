# ReservoirTTA + SAE Routing Experiment Plan

## Goal

ReservoirTTA 원본 코드를 이 프로젝트 안에 내려받고, 원 논문의 StyleVec 기반 routing을 이 프로젝트의 SAE latent descriptor로 대체한 최소 실험 코드를 만든다. 첫 실험 범위는 최대한 작게 잡아 ImageNet clean validation과 ImageNet-C만 사용한다.

## High-Level Idea

ReservoirTTA는 test stream의 각 sample에 대해 StyleVec descriptor를 만들고, 그 descriptor를 reservoir prototype과 비교해 specialist model을 선택한다. 이번 실험에서는 reservoir 구조, specialist update, evaluation loop는 최대한 원본 코드 흐름을 유지하고, routing descriptor만 다음처럼 바꾼다.

```text
Original:
image -> StyleVec extractor -> style descriptor -> reservoir routing

SAE version:
image -> backbone feature hook -> SAE encoder -> sparse latent descriptor -> reservoir routing
```

이렇게 하면 비교가 단순해진다. 같은 ReservoirTTA 프레임워크 안에서 StyleVec routing과 SAE latent routing만 바꿔 ImageNet-C stream에서 online accuracy, forgetting, routing stability를 비교한다.

## Routing Model Strategy

ReservoirTTA 원 논문은 routing용 StyleVec을 classifier 자체가 아니라 frozen ImageNet-trained VGG-19의 early/intermediate feature statistics에서 뽑는다. ImageNet-C classifier는 주로 ResNet-50을 쓰고, 추가로 ViT-B/16도 평가한다. 따라서 SAE routing 실험은 비교 공정성과 novelty를 분리해서 설계한다.

### Track A. Fair Controlled Comparison: VGG-SAE Routing

목표:

```text
same routing extractor network, different descriptor
```

구성:

```text
classifier/adaptation model: ReservoirTTA 원본 ImageNet-C 설정, 우선 ResNet-50
routing extractor: frozen ImageNet-trained VGG-19
baseline descriptor: VGG StyleVec, log-variance/channel statistics
ours descriptor: VGG feature 위에 학습한 SAE latent descriptor
```

장점:

- ReservoirTTA 원본의 가장 중요한 routing 조건을 유지한다.
- StyleVec vs SAE descriptor만 비교하므로 실험 통제가 깔끔하다.
- reviewer가 "extractor를 바꿔서 좋아진 것 아닌가?"라고 묻는 것을 줄일 수 있다.

한계:

- SAE latent가 classifier 내부 feature reliance를 직접 반영한다는 novelty는 약해진다.
- VGG feature의 sparse descriptor이므로 원 논문의 style prior와 더 가까운 변형으로 보일 수 있다.

### Track B. Novelty/Main Claim: Classifier-Aligned SAE Routing

목표:

```text
routing descriptor comes from the same pretrained model being adapted
```

구성:

```text
classifier/adaptation model: ResNet-50 또는 ViT-B/16
routing extractor: same classifier backbone feature
ours descriptor: 해당 classifier feature 위에 학습한 SAE latent descriptor
baseline descriptor: ReservoirTTA VGG StyleVec
```

장점:

- "model-internal sparse latent descriptor"라는 논문 novelty가 가장 잘 살아난다.
- specialist routing이 실제 classifier feature reliance 변화와 연결된다는 주장을 할 수 있다.

한계:

- StyleVec baseline과 extractor network가 달라져서 controlled comparison으로는 약하다.
- classifier별 SAE를 학습해야 하므로 compute와 구현 비용이 늘어난다.

### Practical Decision

첫 구현은 Track A로 시작한다. 즉, ReservoirTTA와 마찬가지로 routing extractor는 VGG-19를 쓰되, StyleVec을 VGG feature SAE latent로 대체한다. 이 단계에서 reservoir 코드, compact ImageNet-C loader, routing log, metric 저장이 모두 작동하는지 확인한다.

그 다음 Track B를 추가한다. ResNet-50 ImageNet-C classifier와 같은 ResNet-50 내부 feature에서 SAE를 학습하고, VGG StyleVec baseline 대비 classifier-aligned SAE routing을 main novelty experiment로 둔다. 현재 프로젝트의 ViT SAE는 pipeline smoke test에는 쓸 수 있지만, ImageNet-C/ReservoirTTA의 공정 비교 결과로는 별도 표시한다.

## Directory Plan

```text
Feature-Reliance/
  third_party/
    ReservoirTTA/                  # GitHub 원본 clone, 원본 변경 최소화
  reservoir_sae/
    descriptors/
      stylevec_adapter.py          # 원본 StyleVec descriptor wrapper
      sae_adapter.py               # 현재 프로젝트 SAE latent descriptor wrapper
    experiments/
      run_imagenet_c_minimal.py    # ImageNet + ImageNet-C 최소 실행 entrypoint
      configs/
        imagenet_c_minimal.yaml
    scripts/
      prepare_imagenet_c_subset.py # 필요 시 corruption/severity subset index 생성
    README.md
  outputs/
    reservoir_sae/                 # 실험 로그와 csv/json 결과
```

원본 코드를 직접 많이 고치기보다 `reservoir_sae/`에서 wrapper와 entrypoint를 만들어 붙이는 방식으로 시작한다. 원본 내부 변경이 꼭 필요한 경우에는 작은 patch 파일이나 명확한 adapter hook만 둔다.

## Step 1. Download ReservoirTTA Code

Repository:

```text
https://github.com/LTS5/ReservoirTTA
```

작업:

1. `third_party/ReservoirTTA`에 clone한다.
2. 원본 dependency, config, dataset loader, style feature extractor, reservoir routing 코드 위치를 확인한다.
3. 원본 실행이 가능한 최소 command를 찾는다.
4. 원본 코드는 baseline 재현용으로 가능한 보존한다.

산출물:

- `third_party/ReservoirTTA`
- 원본 실행 command 메모
- routing descriptor를 교체해야 하는 파일/함수 목록

## Step 2. Identify the StyleVec Interface

확인할 것:

1. StyleVec descriptor의 shape
2. descriptor normalization 방식
3. reservoir prototype update 방식
4. routing distance metric
5. batch 단위인지 sample 단위인지
6. descriptor extraction이 GPU/CPU 어디에서 수행되는지

목표는 다음 공통 인터페이스를 정의하는 것이다.

```python
class DescriptorExtractor:
    def extract(self, images) -> torch.Tensor:
        """Return [batch, descriptor_dim] routing descriptors."""
```

StyleVec baseline과 SAE routing이 모두 이 인터페이스를 따르게 만든다.

## Step 3. Build SAE Descriptor Extractor

현재 프로젝트에는 `Model/SAE.py`의 `VanillaL1SAE`와 `SAE_validation.py`의 ViT hook/token 수집 흐름이 있다. 이를 ReservoirTTA routing에 맞춰 다음처럼 얇게 감싼다.

입력:

```text
images: [B, 3, H, W]
```

처리:

1. Track A에서는 ReservoirTTA와 같은 frozen VGG-19 routing extractor에서 target layer feature를 hook으로 수집한다.
2. Track B에서는 adaptation classifier와 같은 backbone, 우선 ResNet-50 또는 ViT-B/16의 target layer feature를 hook으로 수집한다.
3. feature를 해당 SAE normalizer로 정규화한다.
4. `VanillaL1SAE.encode()`로 sparse latent `z`를 얻는다.
5. spatial/token-level latent를 image-level descriptor로 pooling한다.

초기 pooling 후보:

```text
mean activation: z.mean(dim=tokens)
frequency: (z > threshold).float().mean(dim=tokens)
top-k binary: image별 상위 k latent만 1
```

첫 구현은 가장 안정적인 `frequency` 또는 `mean activation` 하나로 시작하고, config에서 바꿀 수 있게 한다.

필요한 체크포인트:

1. SAE checkpoint path
2. SAE input dimension / hidden dimension
3. token normalizer mean/std 또는 기존 저장 포맷
4. target ViT block index
5. token scope: `cls`, `patch`, `all`

만약 학습된 SAE checkpoint가 아직 없거나 저장 포맷이 불명확하면, 첫 단계는 small ImageNet subset 또는 현재 보유한 Imagenette 기반 SAE로 pipeline smoke test를 먼저 한다.

## Step 4. Minimal Dataset Scope

처음에는 ImageNet과 ImageNet-C만 사용한다.

### Clean Source / Reference

```text
ImageNet validation
```

용도:

- clean accuracy 확인
- optional SAE descriptor prototype sanity check
- source model baseline

### Test-Time Stream

```text
ImageNet-C validation
```

최소 corruption set:

```text
gaussian_noise
shot_noise
motion_blur
snow
contrast
jpeg_compression
```

최소 severity:

```text
severity 3 only
```

처음부터 15 corruptions x 5 severities 전체를 돌리지 않는다. 위 subset으로 routing과 adaptation loop가 제대로 도는지 확인한 뒤 확장한다.

### Stream Construction

첫 실험 stream:

```text
clean -> gaussian_noise -> motion_blur -> gaussian_noise -> contrast -> jpeg_compression
```

이렇게 recurring domain을 하나 넣어 reservoir가 이전 specialist를 재사용하는지 확인한다.

## Step 5. Experiments

최소 비교군:

1. No adaptation
2. Single-model TTA
3. ReservoirTTA with StyleVec routing
4. ReservoirTTA with SAE latent routing
5. Random specialist routing

처음에는 원본 ReservoirTTA의 default adaptation loss를 그대로 사용한다. StyleVec을 SAE로 바꾸는 효과만 보려면 adaptation objective를 동시에 바꾸지 않는 것이 좋다.

## Step 6. Metrics

필수:

```text
online accuracy
mean corruption error or accuracy by corruption
routing assignment histogram
specialist usage count
recurring domain reassignment consistency
```

가능하면 추가:

```text
prototype cosine distance
SAE top-k latent overlap by corruption
per-specialist clean accuracy after stream
runtime / descriptor extraction time
```

결과 저장:

```text
outputs/reservoir_sae/{run_name}/
  config.yaml
  metrics.csv
  routing_log.csv
  summary.json
```

## Step 7. Implementation Order

1. ReservoirTTA repo clone 및 원본 구조 파악
2. StyleVec extractor와 routing 호출부 찾기
3. 공통 descriptor interface 추가
4. StyleVec adapter로 원본 baseline이 그대로 도는지 확인
5. SAE adapter 구현
6. ImageNet-C minimal config 작성
7. no-adapt / random / StyleVec / SAE routing smoke test
8. corruption subset에서 짧은 online run
9. 결과 csv/json 저장 및 간단한 분석 스크립트 작성

## Risks and Decisions

### SAE Backbone Mismatch

ReservoirTTA 원본 모델과 현재 SAE가 학습된 backbone이 다르면 descriptor가 classifier 내부 feature reliance를 직접 반영하지 못할 수 있다. 첫 구현에서는 pipeline 작동을 우선하고, 이후 ReservoirTTA backbone feature에 맞춰 SAE를 다시 학습하는 방향을 둔다.

### Dataset Availability

ImageNet과 ImageNet-C는 로컬 경로가 필요하다. 코드에서는 dataset root를 config로 받고, 실제 데이터가 없으면 명확한 에러 메시지를 내도록 한다. smoke test에는 작은 subset index를 지원한다.

### Compute Cost

Reservoir specialist 여러 개와 SAE descriptor extraction을 함께 돌리면 비용이 커진다. 첫 실험은 corruption subset, severity 3, sample limit, small batch로 시작한다.

### Original Code Intrusion

원본 ReservoirTTA를 많이 수정하면 baseline 비교가 흐려진다. 가능한 adapter/wrapper 방식으로 붙이고, 꼭 필요한 변경만 patch한다.

## First Milestone

첫 번째 완료 기준:

```text
python reservoir_sae/experiments/run_imagenet_c_minimal.py \
  --config reservoir_sae/experiments/configs/imagenet_c_minimal.yaml \
  --routing sae \
  --max-samples 1024
```

위 command가 ImageNet-C subset에서 실행되고, `metrics.csv`, `routing_log.csv`, `summary.json`을 저장한다.

두 번째 완료 기준:

```text
--routing stylevec
--routing sae
--routing random
```

세 routing mode가 같은 stream에서 실행되어 비교 가능한 결과 파일을 만든다.
