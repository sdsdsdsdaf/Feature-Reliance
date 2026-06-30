# Minimal ReservoirTTA + SAE Routing

This folder contains the first compact ImageNet-C experiment described in
`plan.md`.

## Files

- `experiments/run_imagenet_c_minimal.py`
  - Runs compact ImageNet-C with ReservoirTTA-style routing.
  - Supports `--routing stylevec`, `--routing sae`, and `--routing random`.
  - Stores `metrics.csv`, `routing_log.csv`, `summary.json`, and `config.json`.

- `experiments/train_vgg_sae_descriptor.py`
  - Trains a small SAE on frozen VGG-19 pooled features from ImageNet validation.
  - Produces the checkpoint needed by `--routing sae`.

- `descriptors/stylevec_adapter.py`
  - ReservoirTTA-style VGG StyleVec descriptor.

- `descriptors/sae_adapter.py`
  - VGG feature SAE latent descriptor.

## Suggested Order

Activate the project conda environment:

```bash
conda activate feature_reliance_cu126
```

All commands below can also be run without activating by prefixing them with
`conda run -n feature_reliance_cu126`.

Download/cache the datasets:

```bash
conda run -n feature_reliance_cu126 huggingface-cli login

conda run -n feature_reliance_cu126 python scripts/download_hf_imagenet1k.py \
  --split validation \
  --cache-dir data/hf_cache

conda run -n feature_reliance_cu126 python scripts/download_hf_mini_imagenet_c.py \
  --cache-dir data/hf_cache
```

Run the StyleVec baseline:

```bash
conda run -n feature_reliance_cu126 python reservoir_sae/experiments/run_imagenet_c_minimal.py \
  --config reservoir_sae/experiments/configs/imagenet_c_minimal.yaml \
  --routing stylevec \
  --max-samples 1024
```

Train the VGG-SAE descriptor:

```bash
conda run -n feature_reliance_cu126 python reservoir_sae/experiments/train_vgg_sae_descriptor.py \
  --cache-dir data/hf_cache \
  --output outputs/reservoir_sae/vgg_sae.pt \
  --max-samples 8192 \
  --epochs 50
```

Run SAE routing:

```bash
conda run -n feature_reliance_cu126 python reservoir_sae/experiments/run_imagenet_c_minimal.py \
  --config reservoir_sae/experiments/configs/imagenet_c_minimal.yaml \
  --routing sae \
  --sae-checkpoint outputs/reservoir_sae/vgg_sae.pt \
  --max-samples 1024
```

Run random routing:

```bash
conda run -n feature_reliance_cu126 python reservoir_sae/experiments/run_imagenet_c_minimal.py \
  --config reservoir_sae/experiments/configs/imagenet_c_minimal.yaml \
  --routing random \
  --max-samples 1024
```

If ImageNet-1K HF access is not ready yet, add
`--skip-source-calibration` to the runner. That uses a fixed novelty threshold
instead of calibrating it from clean ImageNet validation descriptors.
