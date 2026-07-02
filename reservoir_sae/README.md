# Minimal ReservoirTTA + SAE Routing

This folder contains compact ImageNet-C routing experiments for comparing
ReservoirTTA-style routing descriptors with SAE latent descriptors.

The current config uses clean ImageNet from the Hugging Face cache under
`data/hf_cache`, and uses ReservoirTTA-style local ImageNet-C domain segments
for the target stream.

## What Runs

`experiments/run_imagenet_c_minimal.py` runs a small online ImageNet-C stream
and writes:

- `metrics.csv`
- `routing_log.csv`
- `domain_summary.csv`
- `domain_sequence.csv`
- `summary.json`
- `config.json`

Supported routing modes:

- `stylevec`: frozen VGG-19 StyleVec baseline.
- `vgg_sae`: frozen VGG-19 pooled features encoded by a VGG-feature SAE.
- `sae`: alias for `vgg_sae`, kept for backward compatibility.
- `vit_sae`: ViT-B/16 token SAE routing while the classifier can remain the
  default ResNet-50.
- `classifier_vit_sae`: classifier is ViT-B/16 and the descriptor hooks the
  same ViT-B/16 model instance used for classification.
- `random`: random descriptor routing smoke baseline.

Default config:

- Dataset cache: `data/hf_cache`
- Clean source: `ILSVRC/imagenet-1k`, `validation`
- ImageNet-C stream: `data/imagenet-c/<corruption>/<severity>/<class>/*`
- Classifier: `resnet50`
- Output root: `outputs/reservoir_sae/minimal`

## Files

- `experiments/run_imagenet_c_minimal.py`
  - Main ReservoirTTA-style runner.

- `experiments/train_vgg_sae_descriptor.py`
  - Legacy controlled-baseline helper for VGG-feature SAE routing.
  - This is not the main ViT-B SAE training path.

- `experiments/train_vit_sae_descriptor.py`
  - Uses the existing `SAE_validation` training pipeline to train or package a
    ViT-B/16 token SAE checkpoint for `vit_sae` and `classifier_vit_sae`.

- `descriptors/stylevec_adapter.py`
  - Frozen VGG-19 StyleVec descriptor.

- `descriptors/sae_adapter.py`
  - Frozen VGG-19 feature SAE descriptor.

- `descriptors/vit_sae_adapter.py`
  - ViT-B/16 block-token SAE descriptor.

## Setup

Activate the project environment:

```bash
conda activate feature_reliance_cu126
```

All commands below can also be run without activating by prefixing them with:

```bash
conda run -n feature_reliance_cu126
```

## Dataset Setup

Clean ImageNet is loaded from the local HF cache. If the cache is missing,
download/cache it:

```bash
conda run -n feature_reliance_cu126 huggingface-cli login

conda run -n feature_reliance_cu126 python scripts/download_hf_imagenet1k.py \
  --split validation \
  --cache-dir data/hf_cache
```

For the target stream, use local ImageNet-C in ReservoirTTA/RobustBench layout:

```text
data/imagenet-c/
  gaussian_noise/
    5/
      n01440764/
        *.JPEG
      ...
    4/
    3/
    2/
    1/
  shot_noise/
  ...
```

Download the official ImageNet-C archives from Zenodo:

```bash
conda run -n feature_reliance_cu126 python scripts/download_hf_imagenet_c.py \
  --output-dir data/imagenet-c \
  --download-dir data/zenodo_cache/imagenet-c
```

The default downloads the standard 15-corruption ImageNet-C archives:
`noise.tar`, `blur.tar`, `weather.tar`, and `digital.tar`.

Archive contents:

- `noise.tar`: `gaussian_noise`, `shot_noise`, `impulse_noise`
- `blur.tar`: `defocus_blur`, `glass_blur`, `motion_blur`, `zoom_blur`
- `weather.tar`: `frost`, `snow`, `fog`, `brightness`
- `digital.tar`: `contrast`, `elastic_transform`, `pixelate`, `jpeg_compression`

Zenodo also provides `extra.tar`, but it is not part of the usual 15-corruption
ImageNet-C / ReservoirTTA protocol. It contains `speckle_noise`, `spatter`,
`gaussian_blur`, and `saturate`. Download it only when explicitly needed:

```bash
conda run -n feature_reliance_cu126 python scripts/download_hf_imagenet_c.py \
  --output-dir data/imagenet-c \
  --download-dir data/zenodo_cache/imagenet-c \
  --include-extra
```

The runner follows the original ReservoirTTA-style domain sequence:

```text
15 corruptions x severities [5, 4, 3, 2, 1]
```

`--max-samples` means examples per corruption/severity domain in
`reservoirtta` mode. The default config uses `num_examples_per_domain: 5000`.
The cached HF `ang9867/ImageNet-C` mirror has only `image,label` columns, so it
cannot recover true corruption/severity domains. Use `--stream-mode mixed_hf`
only for sanity checks.

## Routing Runs

Run all routing modes in one command:

```bash
bash reservoir_sae/run_reservoir_sae_all.sh \
  --stream-mode reservoirtta \
  --imagenet-c-root data/imagenet-c \
  --model-name vit_base_patch16_224 \
  --output-dir outputs/reservoir_sae/all_routings_vit_b \
  --max-samples 5000
```

Default routing list:

```text
stylevec, vgg_sae, vit_sae, classifier_vit_sae, random
```

For a quick cached-HF sanity check:

```bash
bash reservoir_sae/run_reservoir_sae_all.sh \
  --stream-mode mixed_hf \
  --routings random,stylevec \
  --max-samples 16 \
  --skip-source-calibration \
  --output-dir outputs/reservoir_sae/smoke_all
```

Run the VGG StyleVec baseline:

```bash
conda run -n feature_reliance_cu126 python reservoir_sae/experiments/run_imagenet_c_minimal.py \
  --config reservoir_sae/experiments/configs/imagenet_c_minimal.yaml \
  --routing stylevec \
  --imagenet-c-root data/imagenet-c \
  --offline \
  --max-samples 5000
```

Run random routing:

```bash
conda run -n feature_reliance_cu126 python reservoir_sae/experiments/run_imagenet_c_minimal.py \
  --config reservoir_sae/experiments/configs/imagenet_c_minimal.yaml \
  --routing random \
  --imagenet-c-root data/imagenet-c \
  --offline \
  --max-samples 5000
```

If clean ImageNet source calibration is not available, add
`--skip-source-calibration`. That uses a fixed novelty threshold.

## VGG-SAE Routing

This is a controlled baseline against VGG StyleVec. It uses frozen VGG-19
features, not the ViT-B SAE training setup used for the main SAE routing runs.

Train the legacy VGG-SAE descriptor:

```bash
conda run -n feature_reliance_cu126 python reservoir_sae/experiments/train_vgg_sae_descriptor.py \
  --cache-dir data/hf_cache \
  --offline \
  --output outputs/reservoir_sae/vgg_sae.pt \
  --max-samples 8192 \
  --epochs 50
```

Run VGG-SAE routing:

```bash
conda run -n feature_reliance_cu126 python reservoir_sae/experiments/run_imagenet_c_minimal.py \
  --config reservoir_sae/experiments/configs/imagenet_c_minimal.yaml \
  --routing vgg_sae \
  --sae-checkpoint outputs/reservoir_sae/vgg_sae.pt \
  --imagenet-c-root data/imagenet-c \
  --offline \
  --max-samples 5000
```

`--routing sae` currently does the same thing as `--routing vgg_sae`.

## ViT-B SAE Routing

This is the main SAE routing path. SAE training here reuses the existing
`SAE_validation` pipeline through `Utils.SAE_utils`; it does not use a separate
hand-written training loop.

The runner expects a packaged ViT-SAE checkpoint containing:

- SAE weights
- `token_mean`
- `token_std`
- ViT model name
- target block and token scope

Train a ViT-B SAE descriptor using the existing pipeline:

- streaming token normalizer: `fit_token_normalizer_streaming`
- geometric decoder-bias initialization: `compute_b_dec_init_streaming`
- validation-token collection: `collect_tokens_with_hook`
- SAE training: `train_sae_auto`
- validation metrics: `evaluate_sae_tokens`
- AMP, `torch.compile`, finite checks, cache/stream token source selection,
  and early stopping from the existing utilities

Default SAE hyperparameters:

```json
{
  "expansion": 32,
  "dec_bias_mode": "geom",
  "active_threshold": 0.2,
  "l1_reg": 3e-5,
  "batch_size": 7096,
  "bias_init_geom_max_iter": 100,
  "bias_init_geom_tol": 1e-5,
  "model_compile": true,
  "amp_dtype": "bfloat16",
  "check_finite": true,
  "matmul_precision": "high"
}
```

```bash
conda run -n feature_reliance_cu126 python reservoir_sae/experiments/train_vit_sae_descriptor.py \
  --cache-dir data/hf_cache \
  --offline \
  --output outputs/reservoir_sae/vit_b_sae.pt \
  --max-samples 8192 \
  --max-train-tokens 1000000 \
  --max-val-tokens 200000 \
  --epochs 350
```

Package an existing `SAE_validation` checkpoint for routing. This recomputes
the ViT token normalizer from `data/hf_cache`, then stores it with the existing
SAE weights:

```bash
conda run -n feature_reliance_cu126 python reservoir_sae/experiments/train_vit_sae_descriptor.py \
  --cache-dir data/hf_cache \
  --offline \
  --init-checkpoint outputs/SAE_validation/grid_search/trial_0000/best_sae_state.pt \
  --output outputs/reservoir_sae/vit_b_sae.pt \
  --max-samples 8192 \
  --max-train-tokens 1000000 \
  --max-val-tokens 200000 \
  --epochs 0
```

Use this packaging path when you already trust an SAE trained by
`SAE_validation.py` and only need a routing checkpoint with token statistics.

Run ViT-B SAE routing with the default ResNet-50 classifier:

```bash
conda run -n feature_reliance_cu126 python reservoir_sae/experiments/run_imagenet_c_minimal.py \
  --config reservoir_sae/experiments/configs/imagenet_c_minimal.yaml \
  --routing vit_sae \
  --vit-sae-checkpoint outputs/reservoir_sae/vit_b_sae.pt \
  --imagenet-c-root data/imagenet-c \
  --offline \
  --max-samples 5000
```

Run classifier-aligned ViT-B SAE routing. In this mode the classifier and the
routing extractor are the same ViT-B/16 model instance:

```bash
conda run -n feature_reliance_cu126 python reservoir_sae/experiments/run_imagenet_c_minimal.py \
  --config reservoir_sae/experiments/configs/imagenet_c_minimal.yaml \
  --routing classifier_vit_sae \
  --model-name vit_base_patch16_224 \
  --vit-sae-checkpoint outputs/reservoir_sae/vit_b_sae.pt \
  --imagenet-c-root data/imagenet-c \
  --device cuda \
  --offline \
  --max-samples 5000
```

## Useful Runner Overrides

- `--routing`: one of `stylevec`, `sae`, `vgg_sae`, `vit_sae`,
  `classifier_vit_sae`, `random`.
- `--stream-mode`: `reservoirtta` for domain-segmented ImageNet-C, or
  `mixed_hf` for the old mixed HF sanity stream.
- `--imagenet-c-root`: local ImageNet-C root for ReservoirTTA-style runs.
- `--max-samples`: examples per corruption/severity domain in `reservoirtta`
  mode; total mixed-stream examples in `mixed_hf` mode.
- `--max-domains`: limit domain segments for smoke tests.
- `--cache-dir`: override HF cache directory.
- `--offline`: use only cached HF datasets.
- `--output-dir`: override output root.
- `--sae-checkpoint`: VGG-SAE checkpoint path. Also works as a ViT-SAE
  checkpoint override when `--routing vit_sae` or `--routing classifier_vit_sae`.
- `--vit-sae-checkpoint`: ViT-SAE checkpoint path.
- `--model-name`: override classifier model, required for classifier-aligned
  ViT-B runs unless the config already sets `model.name: vit_base_patch16_224`.
- `--batch-size`: override stream batch size.
- `--num-workers`: DataLoader workers.
- `--skip-source-calibration`: use fixed reservoir threshold.
