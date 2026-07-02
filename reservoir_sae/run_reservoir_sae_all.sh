#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="reservoir_sae/experiments/configs/imagenet_c_minimal.yaml"
OUTPUT_DIR="outputs/reservoir_sae/all_routings_vit_b"
IMAGENET_C_ROOT="data/imagenet-c"
STREAM_MODE="reservoirtta"
MODEL_NAME="vit_base_patch16_224"
DEVICE="cuda"
BATCH_SIZE="64"
NUM_WORKERS="2"
MAX_SAMPLES="5000"
MAX_DOMAINS=""
CONDA_ENV=""
SAE_CHECKPOINT="outputs/reservoir_sae/vgg_sae.pt"
VIT_SAE_CHECKPOINT="outputs/reservoir_sae/vit_b_sae.pt"
SKIP_SOURCE_CALIBRATION="0"
OFFLINE="1"
ROUTINGS=("stylevec" "vgg_sae" "vit_sae" "classifier_vit_sae" "random")

usage() {
  cat <<'EOF'
Run all reservoir_sae routing modes with one command.

Usage:
  bash reservoir_sae/run_reservoir_sae_all.sh [options]

Options:
  --config PATH                 Config yaml.
  --output-dir PATH             Output root. Each routing writes under PATH/<routing>.
  --imagenet-c-root PATH        Local ImageNet-C root for ReservoirTTA stream.
  --stream-mode MODE            reservoirtta or mixed_hf.
  --model-name NAME             Classifier model. Default: vit_base_patch16_224.
  --device cuda|cpu             Default: cuda, runner falls back if unavailable.
  --batch-size N                Default: 64.
  --num-workers N               Default: 2.
  --max-samples N               ReservoirTTA mode: examples per domain. mixed_hf: total examples.
  --max-domains N               Optional smoke-test limit on corruption/severity domains.
  --sae-checkpoint PATH         VGG-SAE checkpoint.
  --vit-sae-checkpoint PATH     ViT-SAE checkpoint.
  --routings "a,b,c"            Comma-separated routing list.
  --skip-source-calibration     Use fixed threshold instead of clean ImageNet calibration.
  --no-offline                  Allow HF network access.
  --conda-env NAME              Use conda run -n NAME instead of current Python.
  --no-conda                    Use current Python instead of conda run.
  -h, --help                    Show this help.

Default routings:
  stylevec,vgg_sae,vit_sae,classifier_vit_sae,random
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --imagenet-c-root) IMAGENET_C_ROOT="$2"; shift 2 ;;
    --stream-mode) STREAM_MODE="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --max-domains) MAX_DOMAINS="$2"; shift 2 ;;
    --sae-checkpoint) SAE_CHECKPOINT="$2"; shift 2 ;;
    --vit-sae-checkpoint) VIT_SAE_CHECKPOINT="$2"; shift 2 ;;
    --routings)
      IFS=',' read -r -a ROUTINGS <<< "$2"
      shift 2
      ;;
    --skip-source-calibration) SKIP_SOURCE_CALIBRATION="1"; shift ;;
    --no-offline) OFFLINE="0"; shift ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --no-conda) CONDA_ENV=""; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "$CONDA_ENV" ]]; then
  PYTHON_CMD=(conda run -n "$CONDA_ENV" python)
else
  PYTHON_CMD=(python)
fi

if [[ "$STREAM_MODE" == "reservoirtta" && ! -d "$IMAGENET_C_ROOT" ]]; then
  cat >&2 <<EOF
[warning] ImageNet-C root not found: $IMAGENET_C_ROOT
          ReservoirTTA mode requires local layout:
          root/corruption/severity/class/image
          The default root is repo-local data/imagenet-c.
          The runner will fail clearly if no HF domain metadata is available.
EOF
fi

echo "Project root:  $PROJECT_ROOT"
echo "Config:       $CONFIG"
echo "Output dir:   $OUTPUT_DIR"
echo "Stream mode:  $STREAM_MODE"
echo "ImageNet-C:   $IMAGENET_C_ROOT"
echo "Python cmd:   ${PYTHON_CMD[*]}"
echo "Model:        $MODEL_NAME"
echo "Device:       $DEVICE"
echo "Batch size:   $BATCH_SIZE"
echo "Workers:      $NUM_WORKERS"
echo "Max samples:  $MAX_SAMPLES"
echo "Max domains:  ${MAX_DOMAINS:-all}"
echo "Routings:     ${ROUTINGS[*]}"
echo

for routing in "${ROUTINGS[@]}"; do
  echo "===== RUN routing=${routing} model=${MODEL_NAME} ====="

  cmd=(
    "${PYTHON_CMD[@]}"
    reservoir_sae/experiments/run_imagenet_c_minimal.py
    --config "$CONFIG"
    --routing "$routing"
    --model-name "$MODEL_NAME"
    --output-dir "$OUTPUT_DIR"
    --stream-mode "$STREAM_MODE"
    --imagenet-c-root "$IMAGENET_C_ROOT"
    --device "$DEVICE"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --max-samples "$MAX_SAMPLES"
  )

  if [[ -n "$MAX_DOMAINS" ]]; then
    cmd+=(--max-domains "$MAX_DOMAINS")
  fi
  if [[ "$OFFLINE" == "1" ]]; then
    cmd+=(--offline)
  fi
  if [[ "$SKIP_SOURCE_CALIBRATION" == "1" ]]; then
    cmd+=(--skip-source-calibration)
  fi

  case "$routing" in
    sae|vgg_sae)
      if [[ ! -f "$SAE_CHECKPOINT" ]]; then
        echo "Missing VGG-SAE checkpoint for routing=${routing}: $SAE_CHECKPOINT" >&2
        exit 1
      fi
      cmd+=(--sae-checkpoint "$SAE_CHECKPOINT")
      ;;
    vit_sae|classifier_vit_sae)
      if [[ ! -f "$VIT_SAE_CHECKPOINT" ]]; then
        echo "Missing ViT-SAE checkpoint for routing=${routing}: $VIT_SAE_CHECKPOINT" >&2
        exit 1
      fi
      cmd+=(--vit-sae-checkpoint "$VIT_SAE_CHECKPOINT")
      ;;
  esac

  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
  echo
done

echo "All routing runs finished."
echo "Results: $OUTPUT_DIR"
