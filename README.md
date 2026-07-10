# SAE-Broden Alignment

This repo includes `broden.py`, a script for checking whether SAE patch latents align with Broden concepts at mask level.

The script produces two kinds of evidence:

- a Markdown/CSV table mapping each SAE latent to its best Broden concept by IoU
- visual checks showing Broden images, Broden label/mask previews, and SAE latent activation overlays

## Quick Start

Run a small dry run first. This only loads Broden metadata and writes image/mask previews, so it is the fastest way to catch path or label decoding issues.

```bash
conda run -n feature_reliance_cu126 python broden.py \
  --latent-ids 37,102,188,421 \
  --subset=16 \
  --dry-run
```

Then run the full alignment.

```bash
conda run -n feature_reliance_cu126 python broden.py \
  --latent-ids 37,102,188,421 \
  --activation-percentile 99 \
  --sensitivity-percentiles 95,98,99 \
  --topk-overlays 3
```

## Path Arguments

| argument | default | meaning |
| --- | --- | --- |
| `--broden-root` | `data/broden1_227` | Broden dataset root containing `index.csv`, `c_*.csv`, images, and label/mask files. |
| `--sae-checkpoint` | `outputs/reservoir_sae/vit_b_sae.pt` | SAE checkpoint file, or a trial directory containing `best_sae_state.pt`. |
| `--output-dir` | `Cache/sae_broden` | Directory where tables, previews, overlays, and config logs are written. |

## Common Optional Arguments

| argument | default | meaning |
| --- | --- | --- |
| `--model-name` | `vit_base_patch16_224` | timm model used to extract patch tokens. |
| `--model-checkpoint` | none | Optional backbone checkpoint. |
| `--hook-layer` | checkpoint value or `10` | ViT block used for SAE input activations. |
| `--token-scope` | checkpoint value or `patch` | Must be `patch` or `all` for spatial overlay. |
| `--latent-ids` | `all` | Comma-separated latent ids. Use a subset for quick iteration. |
| `--categories` | `object,part,color,material,texture,scene` | Broden categories to evaluate. |
| `--subset` | none | Image cap applied independently to every enabled category, e.g. `--subset=16`. |
| `--category-subsets` | none | Optional per-category overrides such as `object=32,part=16`; specified values override `--subset`. |
| `--max-images` | none | Limit Broden images for debugging. |
| `--num-activation-shards` | `4` | Number of activation cache shard files. |
| `--activation-percentile` | `99` | Latent-wise threshold. Activations at or above this percentile are active. |
| `--sensitivity-percentiles` | `95,98,99` | Additional thresholds for sensitivity checking. |
| `--preview-per-category` | `1` | Broden image/mask previews saved per category. |
| `--topk-overlays` | `3` | Overlay plots saved per category. |
| `--device` | CUDA if available | Torch device. |
| `--dry-run` | false | Write Broden previews and run config without SAE IoU computation. |

## Outputs

All outputs are written under `--output-dir`.

| output | content |
| --- | --- |
| `sae_broden_alignment.md` | Final table: `SAE latent`, `best Broden concept`, `category`, `IoU`. |
| `sae_broden_alignment.csv` | CSV version of the final table. |
| `latent_concept_iou_long.csv` | Long-form IoU for every evaluated latent-concept pair. |
| `latent_category_iou.csv` | Category-level IoU for each latent after unioning all concept masks within each category. |
| `latent_category_iou.md` | Markdown version of category-level IoU. |
| `activation_thresholds.csv` | Per-latent activation threshold used for binary masks. |
| `threshold_sensitivity.csv` | Best concept per latent under each sensitivity percentile. |
| `broden_data_preview/` | Initial Broden image and label/mask sanity-check plots. |
| `broden_category_overlays/` | Category-level Broden mask and SAE activation overlay plots. |
| `broden_category_overlays/index.md` | Overlay plot index with concept, latent, IoU, and image id. |
| `run_config.json` | Actual arguments and run metadata for reproducibility. |

## Threshold Policy

The default threshold is latent-wise `99th percentile`. For each latent, the script collects patch activations over the evaluated Broden images and marks patches active when:

```text
A_k(x, patch) >= percentile_99(A_k)
```

This is preferable to a single global threshold because SAE latent scales can differ. The script also writes `threshold_sensitivity.csv` for the percentiles listed in `--sensitivity-percentiles`, so unstable latent-concept assignments can be flagged.

## Broden Format Assumptions

The loader expects the Network Dissection-style Broden structure when available:

- `index.csv` with an image column and category columns
- `c_object.csv`, `c_part.csv`, `c_color.csv`, `c_material.csv`, `c_texture.csv`, `c_scene.csv`
- image and label/mask paths stored relative to the `images/` directory
- category entries that are either concept ids or label/mask image paths

For scene and texture labels without pixel masks, the script treats the concept as image-level and uses a full-image mask. For pixel-level categories, label images are decoded by concept id and downsampled to the ViT patch grid.

Concept IoU is computed for each `latent_id x concept_id`. Category IoU is computed separately for each `latent_id x category` by unioning all available concept masks in that category for each image, then applying the same dataset-level intersection-over-union aggregation.

## Validation

The code was syntax-checked with:

```bash
conda run -n feature_reliance_cu126 python -m py_compile broden.py Utils/broden_utils.py
```
