import copy
import csv
import json
from datetime import timedelta
import itertools
import os
import traceback
from pathlib import Path

import time
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
import torch
import timm
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader
from torchvision.datasets import Imagenette

from Utils.Config import SAEExperimentConfig
from Utils.datasets import build_dataset
from Utils.early_stopping import unwrap_compiled_model
from Utils.SAE_plot_utils import (
    _plot_overlay,
    collect_patch_latents_for_batch,
    model_tensor_to_image,
    save_plot,
    save_trial_plots,
)
from Utils.SAE_utils import (
    TransformDataset,
    collect_tokens_with_hook,
    compute_b_dec_init_streaming,
    describe_dataset,
    evaluate_sae_tokens,
    fit_token_normalizer_streaming,
    get_torch_dtype,
    jsonable,
    normalize_tokens_chunked,
    save_json,
    train_sae_auto,
)


# =============================================================================
# Experiment Settings
# =============================================================================
# Edit this block for normal runs. The dataclasses in Utils/Config.py provide
# defaults, but this script-level block is the visible experiment entry point.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_ROOT = "outputs/SAE_validation"
# trial 재사용은 trial_NNNN 디렉터리 이름만 보고 판정한다(override 값은 안 본다). grid를
# 바꿀 때 이 이름을 그대로 두면 이전 grid의 완료된 trial_0000이 새 override 라벨을 달고
# 재사용된다 — 조용히 틀린 결과가 나온다. GRID_SPACE를 바꾸면 여기도 반드시 바꾼다.
GRID_DIR_NAME = "grid_search"

# Dataset / backbone
# "imagenet" = HF 캐시(data/hf_cache)의 ImageNet-1k, 1000 클래스. PatchSAE와 같은 학습 분포다.
# "imagenette" = 10 클래스 서브셋(이전 기본값). 라벨 매핑이 달라지므로 intervention 쪽
# (Utils.SAE_plot_utils.imagenet_label_map_for)이 이 값을 보고 분기한다.
DATASET_TYPE = "imagenet"
DATA_ROOT = "data"
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
MODEL_NAME = "vit_base_patch16_224"
PRETRAINED = True
TARGET_BLOCK = 10
TOKEN_SCOPE = "patch"  # "cls", "patch", "all"

# Dataloader / token extraction
DATALOADER_BATCH_SIZE = 64
DATALOADER_NUM_WORKERS = 12
DATALOADER_PIN_MEMORY = True
MAX_TRAIN_TOKENS = 1_000_000
# ImageNet-1k val은 5만 장 x 196 = 980만 토큰이다. 그 절반인 490만(= 25,000장)을 쓴다.
# 이전 값 80만(4,082장 / 전체의 8.2%)에서 올렸다 — 클래스 커버리지는 그때도 989/1000으로
# 나쁘지 않았지만, trial 간·실행 간 비교의 절대 수준 신뢰도를 올리기 위한 것이다.
#
# 주의: 회차 간 val_nmse 흔들림(실측 1.9%)은 이걸 늘려도 안 줄어든다. 검증셋이 고정이라
# 표본 잡음이 아니라 파라미터가 실제로 움직여서 나는 값이기 때문이다(→ lr 스케줄 문제).
#
# 메모리: 490만 x 768 x 2B(fp16) = 7.0 GiB. normalize를 청크로 돌려 fp16 저장을 유지한다
# (Utils/SAE_utils.normalize_tokens_chunked). fp32로 통째 올리면 21 GiB 피크라 터진다.
MAX_VAL_TOKENS = 4_900_000
# 정규화 통계 / b_dec 초기화용 상한. 학습 예산과 분리한다 — 통계적으로는 부분표본이면
# 충분한데, b_dec는 Weiszfeld 반복마다 ViT 전체 패스를 다시 돌아서 여기가 커지면
# 준비 단계가 학습보다 오래 걸린다. None이면 MAX_TRAIN_TOKENS를 따른다.
MAX_NORMALIZER_TOKENS = 2_000_000
MAX_BDEC_TOKENS = 500_000
# 활성 정규화 규약. "scalar"가 기본이다 — 차원별 나눗셈은 축마다 배율이 달라
# raw 공간의 방향을 뒤트는데, 실측 std 범위가 1.38~12.16(8.8배)이라 왜곡이 작지 않다.
# SAE가 찾는 게 활성 공간의 '방향'(개념)이므로 축을 제각기 늘리면 안 된다.
# 중심화(평균 빼기)는 두 모드 모두 차원별로 한다 — 평행이동이라 기하를 보존한다.
#
# Anthropic/SAELens/OpenAI TopK와 같은 규약이라 외부 수치와 비교가 깨끗해진다.
# 이전 실행들은 전부 "per_dim"이므로 그것들과 직접 비교하려면 여기를 되돌린다.
TOKEN_NORM_MODE = "scalar"  # "scalar" | "per_dim"
TOKEN_SOURCE_MODE = "auto"  # "auto", "cache", "stream"
TOKEN_CACHE_DTYPE = "float16"
TOKEN_CACHE_MAX_CPU_GIB = 8.0

# SAE training
# TRAIN_SCHEDULE_MODE:
#   "epoch"        — 고정 토큰 집합을 EPOCHS번 반복. 같은 활성을 반복해서 쓴다.
#   "token_budget" — 스트림을 한 번만 흘리며 TOTAL_TRAIN_TOKENS를 채운다. 활성을 한 번만
#                    쓰므로 ViT forward가 토큰당 1회이고 RAM 상한이 없다. 검증은 epoch이
#                    아니라 EVAL_EVERY_STEPS 스텝마다. PatchSAE 참조 구현과 같은 형태다.
#                    이 모드에서는 EPOCHS와 MAX_TRAIN_TOKENS를 안 쓴다.
TRAIN_SCHEDULE_MODE = "token_budget"  # "epoch", "token_budget"
TOTAL_TRAIN_TOKENS = 502_217_464
# 검증 1회 비용은 val 토큰 수에 비례한다 — 실측 80만 토큰 = 11.1초이므로 490만이면 약 68초다.
# MAX_VAL_TOKENS를 6배로 올렸으니 검증 빈도를 절반으로 낮춰 총 검증 비용을 맞춘다:
#   1000 스텝마다 -> 70회 x 68초 = 79분/trial  (trial 1개의 28%)
#   2000 스텝마다 -> 35회 x 68초 = 40분/trial  (trial 1개의 14%)
# EARLY_STOPPING_PATIENCE는 "검증 횟수"로 세므로 이 값과 함께 움직여야 한다 — 아래 참조.
EVAL_EVERY_STEPS = 1500
EPOCHS = 120
# PatchSAE 참조 구현 기본값(lr 4e-4 + constant-with-warmup 500 step). 이전 값은 1e-4 / warmup 없음.
SAE_LR = 4e-4
SAE_LR_WARMUP_STEPS = 500
# 후반부 lr 감쇠. PatchSAE는 warmup 뒤 끝까지 고정인데, 그러면 분지에 들어간 뒤에도
# 갱신 폭이 그대로라 파라미터가 최소점 주변을 계속 배회한다.
#
# 실측 근거(2026-08-07, grid_search): 고정 검증셋에서 재는 val_nmse가 회차마다
# 1.2~1.9% 흔들렸다. 검증셋이 고정이고 평가가 결정적이므로 표본 잡음이 아니라
# 전부 파라미터 이동이다. 평탄구간 전체 개선폭이 회차간 표준편차의 1.2~1.7배에
# 그쳐서, 체크포인트 선정이 사실상 운으로 결정됐다:
#   trial_0000: is_best 가 ev1~9 에서만 켜지고 나머지 62 eval 은 한 번도 갱신 못 함
#   trial_0001: ev32 이후 19 eval 연속 무개선
SAE_LR_DECAY = "cosine"        # "none" | "linear" | "cosine"
SAE_LR_DECAY_START_FRAC = 0.6  # 마지막 40% 구간에서만 줄인다
SAE_LR_FINAL_FRAC = 0.0        # 끝에서 정확히 0
SAE_WEIGHT_DECAY = 0.0
EXPANSION = 64
B_DEC_INIT_MODE = "geom"  # "zero", "mean", "geom"
B_DEC_INIT_GRID = [B_DEC_INIT_MODE]  # Set multiple modes here for grid search.
# Weiszfeld 반복 1회마다 ViT forward 전체 패스가 다시 돈다. 학습보다 이 초기화가 더
# 오래 걸리지 않도록 낮게 잡는다 (b_dec는 학습되는 파라미터라 초기값일 뿐이다).
B_DEC_INIT_GEOM_MAX_ITER = 10
SAE_ACTIVE_THRESHOLD = 0.2
# PatchSAE의 8e-5를 우리 규약으로 환산: ||x - b_dec||2 실측 30.25 x 8e-5 = 2.4e-3.
# 여기에 SAE_RECON_SPACE="raw" 보정 3배를 곱한 값이다 (recon 항이 mean(sigma^2)만큼
# 커지므로 같은 sparsity 압력을 유지하려면 lambda도 같이 커져야 한다).
# grid가 켜져 있으면 GRID_SPACE가 이 값을 덮어쓴다.
L1_REG = 7.2e-3
SAE_BATCH_SIZE = 7096
MODEL_COMPILE = True
# 재구성 손실/지표를 어느 공간에서 잴 것인가.
#   "raw"  — raw z -> norm -> SAE -> denorm -> raw z 와 비교. 배포 경로와 같은 공간이다
#            (intervention이 recon*sigma+mu 로 되돌려 ViT에 다시 꽂는다).
#   "norm" — 정규화 공간(이전 동작). 최적화 대상과 배포 현실이 다른 공간에 있게 된다.
# 주의: "raw"면 recon 항이 mean(sigma^2)배(실측 약 2.5~3.2) 커져 sparsity 압력이 그만큼
# 약해진다. 같은 L0를 유지하려면 L1_REG를 대략 3배 올려야 한다.
SAE_RECON_SPACE = "raw"

# --- ghost gradients (PatchSAE src/sae_training/sparse_autoencoder.py 이식) ---
# 죽은 latent만 골라 ReLU 대신 exp()를 태워 재구성 잔차를 설명하게 한다. exp()는
# pre-activation이 음수여도 기울기가 0이 아니라서 ReLU+L1의 흡수 상태를 빠져나온다.
#
# 켜는 근거: trial_0000의 dead_latent_frac이 64.8%(z>0.2) / 61.9%(z>0)로 나와,
# 합의한 판정 기준(40% 초과면 구조 변경)을 넘겼다. TopK 전환 전에 더 가벼운 수단을
# 먼저 시도한다.
#
# 실측 A/B (2026-08-07, trial_0000 체크포인트에서 이어 400스텝, window=0):
#   ghost ON  — 미발화 latent 33,330 -> 24,054(50스텝) -> 27,818(400스텝)
#   ghost OFF — 33,297 -> 33,158 -> 33,165 (내내 평평)
# 부활은 확실히 일어나지만 L1이 다시 죽여서 일부만 유지된다. 위 테스트는 window=0이라
# 아래 운용값(1000)보다 훨씬 공격적이었다는 점을 감안할 것.
#
# 비용: SAE 스텝만 보면 2.82x, ViT forward까지 포함한 실제 벽시계로는 1.52x.
SAE_USE_GHOST_GRADS = True
# PatchSAE 기본값과 동일. 이 스텝 수 동안 한 번도 발화하지 않으면 dead로 본다.
SAE_DEAD_FEATURE_WINDOW = 1000
# PatchSAE 기본값과 동일(사실상 z>0). 보고용 active_threshold(0.2)와 별개다.
SAE_DEAD_FEATURE_THRESHOLD = 1e-8
# ghost 항에 쓸 배치 행 수. None이면 배치 전체이고 그게 PatchSAE와 같다.
#
# 2048은 **의도적인 이탈**이다. 배치 7096의 29%만 ghost에 쓴다. 속도 손잡이지 VRAM
# 손잡이가 아니라서(실측 peak 5.79 GiB로 rows와 무관) 얻는 건 시간뿐이다:
#   None -> 306 ms/step (SAE 스텝 2.82x), 트라이얼당 약 5시간 50분
#   2048 -> 175 ms/step (SAE 스텝 1.61x), 트라이얼당 약 4시간 40분
# 기울기가 닿는 latent 집합은 rows와 무관하게 같고 추정 표본만 줄어든다
# (tests/test_ghost_grads.py::test_max_rows_subsamples_without_changing_which_latents_get_gradient).
# PatchSAE와 엄밀히 대조하려면 None으로 되돌릴 것.
SAE_GHOST_GRAD_MAX_ROWS = 2048

# AMP / safety checks
USE_AMP = torch.cuda.is_available()
SAE_AMP_DTYPE = "bfloat16"
SAE_CHECK_FINITE = True
MATMUL_PRECISION = "high"

# Early stopping
# None = 끔. token_budget 모드에서는 예산을 끝까지 쓴다 — PatchSAE 참조 구현도 고정 토큰
# 예산으로 돌리고 조기 종료가 없다.
#
# 왜 껐나: patience는 val_nmse만 보는데, 학습 중 SAE는 "nmse는 평평한 채 L0만 하강"하는
# 구간을 길게 지난다. val_nmse의 노이즈가 +-0.001인 데 반해 EARLY_STOPPING_EPS가 5e-5라
# 20배 작았고, 개선이 노이즈에 묻히면 patience가 그 하강 도중에 걸려버렸다. 실측(2026-08-06
# grid_lambda_low)에서 세 trial이 예산의 57% / 18% / 33% 지점에서 멈췄고 셋 다 멈추는
# 순간까지 active가 단조 감소 중이었다 — 평형에 도달한 trial이 하나도 없었다. 그 탓에
# l0_raw가 lambda에 대해 비단조로 나왔다(422 / 575 / 176). lambda의 성질이 아니라 각
# trial이 하강 곡선의 다른 지점에서 잘린 결과다.
#
# patience=None이어도 best checkpoint 선정(L0 제약 하)은 그대로 돈다 — 조기 종료 카운팅과
# 체크포인트 선정은 분리돼 있다(Utils/early_stopping.py).
EARLY_STOPPING_PATIENCE = None
# 노이즈 스케일로 맞춘다. eps는 두 곳에서 "같다"의 기준으로 쓰인다:
#   (1) patience의 개선 판정        — tier_score < best - eps
#   (2) 체크포인트 동점 판정        — |tier_score - lowest| <= eps 면 active가 적은 쪽을 남긴다
# 5e-5는 val_nmse 노이즈(+-0.001)의 1/20이라 (2)가 사실상 절대 참이 안 됐다. 그래서 "동점이면
# 더 희소한 쪽" 규칙이 죽어 있었고, best는 그냥 가장 낮게 찍힌 노이즈 draw가 됐다. 실측
# (2026-08-07 grid_raw_loss/trial_0000): eval 9의 nmse=0.0850/active=188.08이 끝까지 best로
# 남고, 그보다 훨씬 희소한 eval 62(nmse=0.0852/active=139.29)가 0.0002 차이로 계속 밀렸다.
# 1e-3이면 그 창 안에서 희소한 쪽으로 단조 이동한다. 창의 기준점은 lowest_score(지금까지의
# 최소)라 accept될 때마다 떠내려가지 않는다 — Utils/early_stopping.py:139.
EARLY_STOPPING_EPS = 1e-3
EARLY_STOPPING_VERBOSE = False

# Grid search
RUN_GRID_SEARCH = True
GRID_MAX_TRIALS = None
GRID_METRIC = "val_nmse"
GRID_MODE = "min"
# best 선정에 거는 희소성 제약. 학습 손실(L1)은 그대로고 "무엇을 남길지"만 바꾼다.
# 근거: PatchSAE(ICLR2025) Table 2의 B/16 + all tokens + layer11 + expansion64
# + lambda_l1 8e-5 실측 L0=148.56. metric은 z>0 기준이어야 한다 — active_threshold
# 초과 개수(mean_l0)로 재면 활성 크기만 작아진 dense SAE가 그대로 통과한다.
# None으로 두면 제약이 꺼지고 기존 동작(val_nmse 단독 선정)으로 돌아간다.
SPARSITY_L0_METRIC = "l0_raw"
# 150은 PatchSAE 운용점(L0 148.56)이고, lambda 1.2e-3에서 실측 l0_raw 146.5로 딱 걸렸다.
# lambda를 그 아래로 내리면 l0_raw가 150을 넘어 **모든 trial이 infeasible**이 되고
# select_best_trial이 selected=None을 반환한다 — grid가 통째로 무의미해진다. 그래서
# 아래 GRID_SPACE와 함께 올린다. 800은 49,152개 중 1.6%로, 이 제약을 도입하게 만든
# 원래 실패(l0_raw 4,550 / 12,288 = 37% dense)와는 여전히 한참 멀다.
SPARSITY_L0_MAX = 800.0
# lambda 하향 스윕 — 재구성 품질을 올리는 게 목적이다. 개입 진단의 바닥이 너무 높다:
# alpha=1.0(latent 무개입)에서 재구성만으로 정확도 20.3점이 날아가는데 cue 효과는 4점이라
# 신호 대 잡음이 나쁘다. 1.2e-3은 앵커로 남겨 이전 실행과 대조한다.
#
# 실측 (lambda -> l0_raw / FVU), 200스텝 eval + patience 30 시절:
#   1.2e-3 -> 146.5 / 0.138,  2.4e-3 -> 44.9 / 0.189,  4.8e-3 -> 14.3 / 0.268
#
# 이 세 점으로 한 번 돌렸으나(2026-08-06 grid_lambda_low) early stopping이 예산의
# 57% / 18% / 33% 지점에서 잘라서 **셋 다 미수렴**이었다: 0.6e-3 -> 422.6 / 0.0987,
# 0.85e-3 -> 575.5 / 0.1173, 1.2e-3 -> 175.9 / 0.1403. FVU 예측은 거의 맞았지만
# l0_raw가 lambda에 대해 비단조로 나왔는데, 그건 lambda의 성질이 아니라 각 trial이
# 자기 하강 곡선의 다른 지점에서 잘린 결과다(멈추는 순간까지 active가 단조 감소 중이었다).
# EARLY_STOPPING_PATIENCE=None으로 끄고 다시 돌린다 — 위 숫자는 하한으로만 읽을 것.
#
# **SAE_RECON_SPACE="raw"로 바뀌면서 lambda를 3배 올렸다.** recon 항이 mean(sigma^2)배
# (실측 약 2.5~3.2) 커지므로 sparsity 압력 lambda*L1/recon 이 그만큼 약해진다. 3배는
# 그 보정이지 정밀 측정이 아니다 — 첫 trial의 l0_raw를 보고 조정할 것.
#   0.6e-3 -> 1.8e-3,  0.85e-3 -> 2.55e-3,  1.2e-3 -> 3.6e-3
# 위 실측표의 lambda 값들은 전부 norm-space 손실 기준이라 직접 비교하면 안 된다.
#
# 셋 다 feasible할 전망이라 선정은 사실상 val_nmse가 하고 가장 낮은 lambda가 뽑힌다.
# 그건 의도한 것이다 — 이번 grid의 목적은 제약 안에서 고르는 게 아니라 희소성/재구성
# 트레이드오프 곡선을 세 점으로 재는 것이다. l0_max는 폭주 방지용으로만 남는다.
GRID_SPACE = {
    "sae.l1_reg": [3.6e-3, 2.4e-3, 1.5e-3, 1.0e-3, 0.6e-3, 0.3e-3,],
}
_UNUSED_GRID_SPACE_FULL = {
    "sae.expansion": [16, 32, 64],
    "sae.l1_reg": [3e-5, 1e-4, 3e-4],
    "optim_config.lr": [5e-5, 1e-4],
    "sae.dec_bias_mode": B_DEC_INIT_GRID,
    "sae.active_threshold": [0.1, 0.2],
}

# Perturbation 랭킹 / latent intervention 곡선
# 두 단계는 val의 같은 부분집합을 쓴다 — 여기서 고른 cue latent를 같은 이미지에서 검증한다.
# 예전에는 오버레이 그림용 12장에 묶여 있어서 acc_drop 분해능이 1/12(0.083)였고 cue와
# random의 차이가 이미지 한 장 단위로만 나왔다. None이면 val 전체(5만 장)를 쓰는데,
# 비용이 (spec 15 x alpha 7) x 이미지 수에 선형이라 trial당 수 시간이 된다.
INTERVENTION_EVAL_IMAGES = 2000
INTERVENTION_EVAL_BATCH_SIZE = 64
PERTURBATION_RANKING_BATCH_SIZE = 8

# Latent masking / feature-alignment validation
RUN_LATENT_MASKING_ALIGNMENT = True
LATENT_MASK_NUM_LATENTS = 16
LATENT_MASK_NUM_TOKENS = 512
LATENT_MASK_TOP_FEATURES = 12
LATENT_MASK_SEED = 0
LATENT_MASK_ACTIVE_ONLY = True
LATENT_MASK_SAVE_TENSORS = False
RUN_LATENT_OVERLAY_VISUALIZATION = True
LATENT_OVERLAY_NUM_IMAGES = 4
LATENT_OVERLAY_LATENTS_PER_IMAGE = 4
LATENT_OVERLAY_SEED = 0


def build_default_config():
    config = SAEExperimentConfig().validate()

    config.extraction_config.device = DEVICE
    config.output.root_dir = OUTPUT_ROOT
    config.output.grid_dir_name = GRID_DIR_NAME

    num_classes = 1000 if DATASET_TYPE == "imagenet" else 10
    for spec, split in ((config.train_dataset_spec, TRAIN_SPLIT), (config.val_dataset_spec, VAL_SPLIT)):
        spec.name = f"{DATASET_TYPE}_{split}"
        spec.dataset_type = DATASET_TYPE
        spec.root = DATA_ROOT
        spec.split = split
        spec.num_classes = num_classes

    if config.model_spec is not None:
        config.model_spec.model_name = MODEL_NAME
        config.model_spec.pretrained_weight = "pretrained" if PRETRAINED else ""

    config.hook.target_block = TARGET_BLOCK
    config.hook.token_scope = TOKEN_SCOPE

    config.data_config.batch_size = DATALOADER_BATCH_SIZE
    config.data_config.num_workers = DATALOADER_NUM_WORKERS
    config.data_config.pin_memory = DATALOADER_PIN_MEMORY
    config.token.max_train_tokens = MAX_TRAIN_TOKENS
    config.token.max_val_tokens = MAX_VAL_TOKENS
    config.token.max_normalizer_tokens = MAX_NORMALIZER_TOKENS
    config.token.max_bdec_tokens = MAX_BDEC_TOKENS

    config.schedule.mode = TRAIN_SCHEDULE_MODE
    config.schedule.total_train_tokens = TOTAL_TRAIN_TOKENS
    config.schedule.eval_every_steps = EVAL_EVERY_STEPS
    config.token.norm_mode = TOKEN_NORM_MODE
    config.token.source_mode = TOKEN_SOURCE_MODE
    config.token.cache_dtype = TOKEN_CACHE_DTYPE
    config.token.cache_max_cpu_gib = TOKEN_CACHE_MAX_CPU_GIB

    config.optim_config.epochs = EPOCHS
    config.optim_config.lr = SAE_LR
    config.optim_config.lr_warmup_steps = SAE_LR_WARMUP_STEPS
    config.optim_config.lr_decay = SAE_LR_DECAY
    config.optim_config.lr_decay_start_frac = SAE_LR_DECAY_START_FRAC
    config.optim_config.lr_final_frac = SAE_LR_FINAL_FRAC
    config.optim_config.weight_decay = SAE_WEIGHT_DECAY
    config.optim_config.use_amp = USE_AMP

    config.sae.expansion = EXPANSION
    config.sae.dec_bias_mode = B_DEC_INIT_MODE
    config.sae.bias_init_geom_max_iter = B_DEC_INIT_GEOM_MAX_ITER
    config.sae.active_threshold = SAE_ACTIVE_THRESHOLD
    config.sae.l1_reg = L1_REG
    config.sae.batch_size = SAE_BATCH_SIZE
    config.sae.model_compile = MODEL_COMPILE
    config.sae.amp_dtype = SAE_AMP_DTYPE
    config.sae.check_finite = SAE_CHECK_FINITE
    config.sae.matmul_precision = MATMUL_PRECISION
    config.sae.recon_space = SAE_RECON_SPACE
    config.sae.use_ghost_grads = SAE_USE_GHOST_GRADS
    config.sae.dead_feature_window = SAE_DEAD_FEATURE_WINDOW
    config.sae.dead_feature_threshold = SAE_DEAD_FEATURE_THRESHOLD
    config.sae.ghost_grad_max_rows = SAE_GHOST_GRAD_MAX_ROWS

    config.early_stopping.patience = EARLY_STOPPING_PATIENCE
    config.early_stopping.eps = EARLY_STOPPING_EPS
    config.early_stopping.save_verbose = EARLY_STOPPING_VERBOSE

    config.grid.enabled = RUN_GRID_SEARCH
    config.grid.max_trials = GRID_MAX_TRIALS
    config.grid.metric = GRID_METRIC
    config.grid.mode = GRID_MODE
    config.grid.space = GRID_SPACE

    config.sparsity.metric = SPARSITY_L0_METRIC
    config.sparsity.l0_max = SPARSITY_L0_MAX

    config.diagnostics.eval_images = INTERVENTION_EVAL_IMAGES
    config.diagnostics.eval_batch_size = INTERVENTION_EVAL_BATCH_SIZE
    config.diagnostics.ranking_batch_size = PERTURBATION_RANKING_BATCH_SIZE

    config.validate()
    if config.extraction_config.device == "cuda" and not torch.cuda.is_available():
        config.extraction_config.device = "cpu"
        config.optim_config.use_amp = False
    return config


def build_backbone_and_transform(config: SAEExperimentConfig):
    model_name = MODEL_NAME
    pretrained = PRETRAINED
    if config.model_spec is not None:
        model_name = config.model_spec.model_name
        pretrained = bool(config.model_spec.pretrained_weight)

    model = timm.create_model(model_name, pretrained=pretrained)
    model.eval().to(config.extraction_config.device)
    data_config = resolve_model_data_config(model)
    transform = create_transform(**data_config, is_training=False)
    mean = list(data_config.get("mean", (0.485, 0.456, 0.406)))
    std = list(data_config.get("std", (0.229, 0.224, 0.225)))
    return model, transform, mean, std


def build_sae_dataset(dataset_spec, transform):
    """dataset_spec 하나로 imagenette / ImageNet-1k 를 같은 인터페이스로 연다.

    ImageNet-1k는 HF 캐시(data/hf_cache)에서 읽는다. transform은 두 경우 모두
    TransformDataset이 씌운다 — HF 로더가 주는 np.ndarray는 거기서 PIL로 맞춘다."""
    kind = str(dataset_spec.dataset_type).lower()
    if kind == "imagenette":
        dataset = Imagenette(
            root=dataset_spec.root or "data",
            split=dataset_spec.split,
            download=False,
        )
    elif kind == "imagenet":
        dataset, _meta = build_dataset("imagenet", split=dataset_spec.split, root=dataset_spec.root or "data")
    else:
        raise ValueError(
            f"SAE_validation supports dataset_type in {{'imagenette', 'imagenet'}}, got {dataset_spec.dataset_type!r}."
        )
    return TransformDataset(dataset, transform=transform)


def build_train_val_loaders(config, transform):
    train_dataset = build_sae_dataset(config.train_dataset_spec, transform)
    val_dataset = build_sae_dataset(config.val_dataset_spec, transform)
    pin_memory = bool(config.data_config.pin_memory and str(config.extraction_config.device).startswith("cuda"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data_config.batch_size,
        shuffle=True,
        num_workers=config.data_config.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data_config.batch_size,
        shuffle=False,
        num_workers=config.data_config.num_workers,
        pin_memory=pin_memory,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def tokens_per_image_for_scope(token_scope, grid=14):
    """token_scope별 이미지 1장당 토큰 수. cls=1, patch=grid^2, all=grid^2+1."""
    scope = str(token_scope).lower()
    if scope == "cls":
        return 1
    if scope == "patch":
        return grid * grid
    return grid * grid + 1


def print_config_summary(config, train_dataset=None):
    """실행 설정 요약. schedule.mode에 따라 학습량 관련 줄만 다르게 찍는다 —
    쓰이지 않는 값(예: token_budget 모드의 epochs/max_train_tokens)을 같이 찍으면
    무엇이 실제로 학습량을 정하는지 오해하게 된다."""
    print("\nSAE experiment config")
    print("=" * 32)
    print(f"model_name             : {MODEL_NAME}")
    print(f"pretrained             : {PRETRAINED}")
    print(f"device                 : {config.extraction_config.device}")
    print(f"target_block           : {config.hook.target_block}")
    print(f"token_scope            : {config.hook.token_scope}")
    print(f"max_val_tokens         : {config.token.max_val_tokens}")
    print(f"expansion              : {config.sae.expansion}")
    print(f"dec_bias_mode          : {config.sae.dec_bias_mode}")
    print(f"active_threshold       : {config.sae.active_threshold}")
    print(f"lr                     : {config.optim_config.lr}")
    print(f"lr_warmup_steps        : {config.optim_config.lr_warmup_steps}")
    print(f"l1_reg                 : {config.sae.l1_reg}")
    print(f"sae_batch_size         : {config.sae.batch_size}")
    print(f"use_amp                : {config.optim_config.use_amp}")
    print(f"amp_dtype              : {config.sae.amp_dtype}")
    print(f"early_stop_patience    : {config.early_stopping.patience}")

    print("-" * 32)
    mode = str(config.schedule.mode).lower()
    print(f"schedule_mode          : {mode}")
    if mode == "token_budget":
        budget = int(config.schedule.total_train_tokens or 0)
        tpi = tokens_per_image_for_scope(config.hook.token_scope)
        print(f"total_train_tokens     : {budget:,}")
        print(f"eval_every_steps       : {config.schedule.eval_every_steps}")
        print(f"→ SAE steps            : {budget // max(1, config.sae.batch_size):,}")
        print(f"→ evaluations          : {budget // max(1, config.sae.batch_size) // max(1, config.schedule.eval_every_steps):,}")
        if train_dataset is not None:
            n_img = len(train_dataset)
            per_epoch = n_img * tpi
            print(f"→ train set            : {n_img:,} images x {tpi} tokens = {per_epoch:,} tokens / epoch")
            print(f"→ epochs equivalent    : {budget / max(1, per_epoch):.3f}")
            print(f"→ images processed     : {budget // max(1, tpi):,}")
        print("  (epochs / max_train_tokens / token_source_mode 는 이 모드에서 쓰이지 않는다)")
    else:
        print(f"epochs                 : {config.optim_config.epochs}")
        print(f"max_train_tokens       : {config.token.max_train_tokens}")
        print(f"token_source_mode      : {config.token.source_mode}")
        if train_dataset is not None and config.token.max_train_tokens:
            tpi = tokens_per_image_for_scope(config.hook.token_scope)
            used = int(config.token.max_train_tokens) // tpi
            print(f"→ unique images used   : {used:,} / {len(train_dataset):,} "
                  f"({100.0 * used / max(1, len(train_dataset)):.2f}% of train set), reused every epoch")
        print("  (total_train_tokens / eval_every_steps 는 이 모드에서 쓰이지 않는다)")


def run_sae_trial(config, trial_dir, trial_id=None):
    config = copy.deepcopy(config).validate()
    trial_dir = Path(trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)
    save_json(trial_dir / "config.json", config)

    torch.set_float32_matmul_precision(config.sae.matmul_precision)
    model, transform, mean, std = build_backbone_and_transform(config)
    train_dataset, val_dataset, train_loader, val_loader = build_train_val_loaders(config, transform)

    # 데이터셋을 만든 뒤에 찍는다 — 학습량을 epoch 환산으로 보여주려면 train set 크기가 필요하다.
    print_config_summary(config, train_dataset=train_dataset)

    describe_dataset("train_dataset", train_dataset)
    describe_dataset("val_dataset", val_dataset)

    # 준비 단계는 학습 예산과 분리한다. 특히 b_dec의 Weiszfeld 반복은 반복마다 ViT
    # 전체 패스를 다시 도므로, 학습 토큰이 커질 때 같이 커지면 학습보다 비싸진다.
    normalizer_tokens = config.token.max_normalizer_tokens or config.token.max_train_tokens
    bdec_tokens = config.token.max_bdec_tokens or config.token.max_train_tokens

    print("\nFitting train-token normalizer from streaming train tokens...")
    token_stats, train_token_count = fit_token_normalizer_streaming(
        model,
        train_loader,
        max_tokens=normalizer_tokens,
        target_block=config.hook.target_block,
        token_scope=config.hook.token_scope,
        device=config.extraction_config.device,
        norm_mode=config.token.norm_mode,
    )
    print(
        f"train tokens seen for normalizer: {train_token_count:,} "
        f"(norm_mode={config.token.norm_mode}, std {token_stats['std'].min():.4f}~{token_stats['std'].max():.4f})"
    )

    print("\nFitting b_dec init from the same train-token stream...")
    b_dec_init = compute_b_dec_init_streaming(
        model,
        train_loader,
        token_stats,
        max_tokens=bdec_tokens,
        target_block=config.hook.target_block,
        token_scope=config.hook.token_scope,
        device=config.extraction_config.device,
        sae_config=config.sae,
    )
    print(f"b_dec init: mode={config.sae.dec_bias_mode}, shape={tuple(b_dec_init.shape)}")

    print("\nCollecting cached validation tokens...")
    val_tokens, val_labels = collect_tokens_with_hook(
        model,
        val_loader,
        max_tokens=config.token.max_val_tokens,
        target_block=config.hook.target_block,
        token_scope=config.hook.token_scope,
        device=config.extraction_config.device,
        cache_dtype=get_torch_dtype(config.token.cache_dtype),
        return_labels=True,
    )
    # .float()로 통째 올리면 원본(fp16)과 사본(fp32)이 동시에 살아 490만 토큰 기준
    # 21 GiB 피크가 난다. 청크만 fp32로 올려 계산하고 저장 dtype은 유지한다.
    val_tokens = normalize_tokens_chunked(val_tokens, token_stats, config.token.normalize_chunk_size)
    print(
        f"val tokens cached: {tuple(val_tokens.shape)} {val_tokens.dtype} "
        f"({val_tokens.numel() * val_tokens.element_size() / 2**30:.2f} GiB)"
    )

    input_dim = int(token_stats["mean"].shape[1])
    hidden_dim = int(input_dim * config.sae.expansion)
    checkpoint_path = trial_dir / "best_sae_state.pt"

    trained_sae, history = train_sae_auto(
        model,
        train_loader,
        val_tokens,
        token_stats,
        input_dim,
        hidden_dim,
        b_dec_init,
        config,
        checkpoint_path=checkpoint_path,
        # 학습 1 epoch이 소비할 토큰 수여야 한다. train_token_count는 정규화 통계가 본
        # 토큰 수라 max_normalizer_tokens를 따로 키우면 여기가 같이 커지고, 그러면
        # cache/stream 판정과 진행 바 총량이 둘 다 어긋난다(cache로 들어갈 크기인데
        # stream을 골라 epoch마다 ViT 전체 패스를 다시 돈다).
        expected_tokens=config.token.max_train_tokens,
    )

    validation_metrics = evaluate_sae_tokens(
        trained_sae,
        val_tokens,
        batch_size=config.sae.batch_size,
        threshold=config.sae.active_threshold,
        device=config.extraction_config.device,
        # recon_space='raw'면 학습 손실과 같은 공간에서 재야 한다. 그래야 summary의
        # val_nmse가 체크포인트 선정 기준과 같은 값이 된다.
        token_stats=token_stats if str(config.sae.recon_space).lower() == "raw" else None,
    )

    latent_masking_alignment = run_latent_masking_alignment_validation(
        trained_sae,
        val_tokens,
        val_labels,
        trial_dir,
        config,
    )
    latent_overlay_visualizations = run_latent_overlay_visualizations(
        model,
        val_dataset,
        trained_sae,
        token_stats,
        trial_dir,
        config,
        mean,
        std,
        latent_ids=latent_masking_alignment.get("selected_latents", []),
    )

    diagnostics = save_trial_plots(
        trial_dir=trial_dir,
        model=model,
        train_history=history,
        sae=trained_sae,
        token_stats=token_stats,
        val_tokens=val_tokens,
        val_labels=val_labels,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        mean=mean,
        std=std,
    )

    best_rows = [row for row in history if row.get("is_best")]
    best_row = best_rows[-1] if best_rows else (min(history, key=lambda row: row["normalized_mse"]) if history else {})
    summary = {
        "trial_id": trial_id,
        "status": "completed",
        "trial_dir": str(trial_dir),
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "train_token_count": train_token_count,
        "val_token_count": int(val_tokens.shape[0]),
        "best_epoch": best_row.get("epoch"),
        "best_val_nmse": best_row.get("normalized_mse"),
        "best_active_mean_count": best_row.get("active_mean_count"),
        "best_l0_raw": best_row.get("l0_raw"),
        "best_l0_feasible": best_row.get("l0_feasible"),
        "sparsity_constraint": {"metric": config.sparsity.metric, "max": config.sparsity.l0_max},
        "final_validation_metrics": validation_metrics,
        "latent_masking_alignment": latent_masking_alignment,
        "latent_overlay_visualizations": latent_overlay_visualizations,
        "checkpoint_path": str(checkpoint_path),
        "diagnostics": diagnostics,
    }
    save_json(trial_dir / "history.json", history)
    save_json(trial_dir / "summary.json", summary)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("trial summary:", jsonable(summary))
    return summary


def iter_grid_items(space):
    keys = list(space.keys())
    values = [space[key] for key in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def set_nested_attr(obj, dotted_key, value):
    parts = dotted_key.split(".")
    target = obj
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


def apply_grid_overrides(config, overrides):
    config = copy.deepcopy(config)
    for key, value in overrides.items():
        set_nested_attr(config, key, value)
    return config.validate()


def format_tuning_hp(overrides):
    return ", ".join(f"{key}={value}" for key, value in overrides.items())


def flatten_dict(data, prefix=""):
    flat = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, name))
        elif isinstance(value, (list, tuple)):
            flat[name] = jsonable(value)
        else:
            flat[name] = jsonable(value)
    return flat


def write_grid_summary_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    flat_rows = [flatten_dict(row) for row in rows]
    fieldnames = sorted({key for row in flat_rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in flat_rows:
            writer.writerow(row)


def _mse_mean(x):
    return float(torch.mean(x.float().pow(2)).item())


def _top_label_counts(labels, mask, k=5):
    if labels is None or labels.numel() == 0 or not bool(mask.any()):
        return []
    active_labels = labels[mask.detach().cpu()]
    if active_labels.numel() == 0:
        return []
    unique, counts = torch.unique(active_labels, return_counts=True)
    order = torch.argsort(counts, descending=True)[: int(k)]
    return [
        {"label": int(unique[idx].item()), "count": int(counts[idx].item())}
        for idx in order
    ]


def _write_latent_alignment_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "latent_id",
        "mean_activation",
        "activation_frequency",
        "active_mean_activation",
        "delta_nmse",
        "masked_nmse",
        "recon_shift_mse",
        "decoder_norm",
        "encoder_norm",
        "encoder_decoder_cosine",
        "top_feature_indices",
        "top_feature_weights",
        "top_feature_activation_corrs",
        "top_active_labels",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (list, tuple, dict)):
                    out[key] = jsonable(value)
                else:
                    out[key] = value
            writer.writerow(out)


@torch.no_grad()
def run_latent_masking_alignment_validation(sae, val_tokens, val_labels, trial_dir, config):
    """Mask random SAE latents and align their reconstruction effect to input features.

    The validation tokens are normalized ViT block features. Each latent is
    aligned to the input feature dimensions where its decoder vector has the
    largest absolute weights. Masking uses the linear decoder identity:
    x_hat_without_j = x_hat - z_j * W_dec[j].
    """

    if not RUN_LATENT_MASKING_ALIGNMENT:
        return {"enabled": False}
    if val_tokens is None or int(val_tokens.shape[0]) == 0:
        return {"enabled": True, "status": "skipped", "reason": "empty val_tokens"}

    start_time = time.perf_counter()
    trial_dir = Path(trial_dir)
    out_dir = trial_dir / "latent_masking_alignment"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_sae = unwrap_compiled_model(sae).eval().to(config.extraction_config.device)
    num_tokens = min(int(LATENT_MASK_NUM_TOKENS), int(val_tokens.shape[0]))
    generator = torch.Generator(device="cpu").manual_seed(int(LATENT_MASK_SEED))
    sample_idx = torch.randperm(int(val_tokens.shape[0]), generator=generator)[:num_tokens]
    sample_tokens_cpu = val_tokens[sample_idx].float().cpu()
    sample_labels = val_labels[sample_idx].detach().cpu() if val_labels is not None else None
    sample_tokens = sample_tokens_cpu.to(config.extraction_config.device)

    x_hat, z = base_sae(sample_tokens)
    z = z.float()
    x_hat = x_hat.float()
    hidden_dim = int(z.shape[1])
    threshold = float(config.sae.active_threshold)
    active_freq = (z > threshold).float().mean(dim=0).cpu()
    mean_activation = z.mean(dim=0).cpu()

    if LATENT_MASK_ACTIVE_ONLY:
        candidates = torch.nonzero(active_freq > 0, as_tuple=False).flatten()
        if candidates.numel() == 0:
            candidates = torch.arange(hidden_dim)
    else:
        candidates = torch.arange(hidden_dim)
    shuffled = candidates[torch.randperm(candidates.numel(), generator=generator)]
    selected = shuffled[: min(int(LATENT_MASK_NUM_LATENTS), int(shuffled.numel()))].tolist()

    centered = sample_tokens.float()
    token_var = torch.mean((centered - centered.mean(dim=0, keepdim=True)).pow(2)).clamp_min(1e-12)
    full_mse = torch.mean((x_hat - centered).pow(2))
    full_nmse = float((full_mse / token_var).item())

    w_dec = base_sae.W_dec.detach().float()
    w_enc = base_sae.W_enc.detach().float()
    rows = []
    details = []
    top_k = min(int(LATENT_MASK_TOP_FEATURES), int(w_dec.shape[1]))

    for latent_id in selected:
        latent_id = int(latent_id)
        activation = z[:, latent_id].detach()
        decoder_vec = w_dec[latent_id].to(x_hat.device)
        contribution = activation[:, None] * decoder_vec[None, :]
        masked_recon = x_hat - contribution
        masked_mse = torch.mean((masked_recon - centered).pow(2))
        masked_nmse = float((masked_mse / token_var).item())
        recon_shift_mse = _mse_mean(contribution)
        active_mask = (activation > threshold).detach().cpu()
        active_values = activation.detach().cpu()[active_mask]

        top_abs, top_idx = torch.topk(w_dec[latent_id].abs().cpu(), k=top_k)
        top_idx = top_idx.tolist()
        top_weights = [float(w_dec[latent_id, idx].item()) for idx in top_idx]
        feature_corrs = []
        activation_cpu = activation.detach().cpu()
        activation_std = activation_cpu.std().clamp_min(1e-8)
        for feature_idx in top_idx:
            feature_values = sample_tokens_cpu[:, feature_idx]
            feature_std = feature_values.std().clamp_min(1e-8)
            corr = torch.mean(
                ((activation_cpu - activation_cpu.mean()) / activation_std)
                * ((feature_values - feature_values.mean()) / feature_std)
            )
            feature_corrs.append(float(corr.item()))

        enc_vec = w_enc[:, latent_id].detach().cpu()
        dec_vec = w_dec[latent_id].detach().cpu()
        enc_dec_cos = torch.nn.functional.cosine_similarity(enc_vec, dec_vec, dim=0).item()
        label_counts = _top_label_counts(sample_labels, active_mask, k=5)
        row = {
            "latent_id": latent_id,
            "mean_activation": float(mean_activation[latent_id].item()),
            "activation_frequency": float(active_freq[latent_id].item()),
            "active_mean_activation": float(active_values.mean().item()) if active_values.numel() else 0.0,
            "delta_nmse": float(masked_nmse - full_nmse),
            "masked_nmse": masked_nmse,
            "recon_shift_mse": recon_shift_mse,
            "decoder_norm": float(dec_vec.norm().item()),
            "encoder_norm": float(enc_vec.norm().item()),
            "encoder_decoder_cosine": float(enc_dec_cos),
            "top_feature_indices": top_idx,
            "top_feature_weights": top_weights,
            "top_feature_activation_corrs": feature_corrs,
            "top_active_labels": label_counts,
        }
        rows.append(row)
        details.append(
            {
                **row,
                "top_feature_abs_weights": [float(v.item()) for v in top_abs],
            }
        )

    if selected:
        selected_contribution = z[:, selected] @ w_dec[selected].to(z.device)
        all_masked_recon = x_hat - selected_contribution
        all_masked_nmse = float((torch.mean((all_masked_recon - centered).pow(2)) / token_var).item())
        all_masked_shift_mse = _mse_mean(selected_contribution)
    else:
        all_masked_nmse = full_nmse
        all_masked_shift_mse = 0.0

    _write_latent_alignment_csv(out_dir / "latent_alignment.csv", rows)
    save_json(out_dir / "latent_alignment.json", details)
    summary = {
        "enabled": True,
        "status": "completed",
        "num_sample_tokens": int(num_tokens),
        "hidden_dim": int(hidden_dim),
        "candidate_pool_size": int(candidates.numel()),
        "selected_latents": [int(x) for x in selected],
        "active_only": bool(LATENT_MASK_ACTIVE_ONLY),
        "active_threshold": threshold,
        "full_reconstruction_nmse": full_nmse,
        "all_selected_masked_nmse": all_masked_nmse,
        "all_selected_delta_nmse": float(all_masked_nmse - full_nmse),
        "all_selected_recon_shift_mse": all_masked_shift_mse,
        "csv": str(out_dir / "latent_alignment.csv"),
        "json": str(out_dir / "latent_alignment.json"),
        "elapsed_seconds": float(time.perf_counter() - start_time),
    }
    if LATENT_MASK_SAVE_TENSORS:
        tensor_path = out_dir / "latent_masking_tensors.pt"
        torch.save(
            {
                "sample_indices": sample_idx,
                "sample_tokens": sample_tokens_cpu,
                "sample_labels": sample_labels,
                "selected_latents": torch.tensor(selected, dtype=torch.long),
                "z_selected": z[:, selected].detach().cpu() if selected else torch.empty(num_tokens, 0),
                "full_reconstruction": x_hat.detach().cpu(),
            },
            tensor_path,
        )
        summary["tensors"] = str(tensor_path)
    save_json(out_dir / "summary.json", summary)
    print(
        "latent masking/alignment:",
        f"sample_tokens={num_tokens}",
        f"selected_latents={len(selected)}",
        f"full_nmse={full_nmse:.6f}",
        f"all_masked_delta_nmse={summary['all_selected_delta_nmse']:.6f}",
    )
    return summary


@torch.no_grad()
def run_latent_overlay_visualizations(
    model,
    val_dataset,
    sae,
    token_stats,
    trial_dir,
    config,
    mean,
    std,
    latent_ids=None,
):
    """Save image-level SAE latent activation overlays for a few validation images."""

    if not RUN_LATENT_OVERLAY_VISUALIZATION:
        return {"enabled": False}
    token_scope = str(config.hook.token_scope).lower()
    if token_scope == "cls":
        return {
            "enabled": True,
            "status": "skipped",
            "reason": "CLS-token SAE has no patch grid to overlay on the image.",
        }
    if not hasattr(model, "blocks"):
        return {
            "enabled": True,
            "status": "skipped",
            "reason": "Overlay visualization requires a ViT-style model with .blocks.",
        }

    start_time = time.perf_counter()
    trial_dir = Path(trial_dir)
    out_dir = trial_dir / "latent_masking_alignment" / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device="cpu").manual_seed(int(LATENT_OVERLAY_SEED))
    num_images = min(int(LATENT_OVERLAY_NUM_IMAGES), len(val_dataset))
    image_indices = torch.randperm(len(val_dataset), generator=generator)[:num_images].tolist()
    candidate_latents = [int(x) for x in latent_ids or []]

    records = []
    for image_idx in image_indices:
        image_tensor, label = val_dataset[int(image_idx)]
        images = image_tensor.unsqueeze(0)
        z = collect_patch_latents_for_batch(
            images,
            model,
            sae,
            token_stats,
            config.hook.target_block,
            config.hook.token_scope,
            config.extraction_config.device,
        )
        z_image = z[0]
        if candidate_latents:
            valid_ids = [latent_id for latent_id in candidate_latents if 0 <= latent_id < z_image.shape[1]]
            if valid_ids:
                scores = torch.tensor([float(z_image[:, latent_id].max().item()) for latent_id in valid_ids])
                order = torch.argsort(scores, descending=True)[: int(LATENT_OVERLAY_LATENTS_PER_IMAGE)]
                selected = [valid_ids[int(i)] for i in order.tolist()]
            else:
                selected = []
        else:
            selected = []
        if not selected:
            selected = z_image.pow(2).mean(dim=0).topk(min(int(LATENT_OVERLAY_LATENTS_PER_IMAGE), z_image.shape[1])).indices.tolist()
            selected = [int(x) for x in selected]

        image_np = model_tensor_to_image(image_tensor, mean, std)
        fig, axes = plt.subplots(1, len(selected) + 1, figsize=(3.2 * (len(selected) + 1), 3.4))
        if len(selected) == 0:
            axes = [axes]
        axes[0].imshow(image_np)
        axes[0].set_title(f"original\nidx={int(image_idx)}, label={int(label)}", fontsize=9)
        axes[0].axis("off")
        latent_records = []
        for ax, latent_id in zip(axes[1:], selected):
            patch_map = z_image[:, int(latent_id)]
            _plot_overlay(ax, image_np, patch_map)
            active_count = int((patch_map > float(config.sae.active_threshold)).sum().item())
            peak = float(patch_map.max().item())
            ax.set_title(f"latent={int(latent_id)}\nactive={active_count}, peak={peak:.2f}", fontsize=9)
            latent_records.append(
                {
                    "latent_id": int(latent_id),
                    "active_patches": active_count,
                    "peak_activation": peak,
                    "mean_activation": float(patch_map.mean().item()),
                }
            )
        fig.suptitle("Random masked SAE latent overlays on original image", fontsize=11)
        fig.tight_layout()
        path = save_plot(fig, out_dir, f"latent_overlay_image_{int(image_idx):05d}", dpi=config.output.plot_dpi)
        records.append(
            {
                "image_idx": int(image_idx),
                "label": int(label),
                "path": str(path),
                "latents": latent_records,
            }
        )

    summary = {
        "enabled": True,
        "status": "completed",
        "num_images": len(records),
        "latents_per_image": int(LATENT_OVERLAY_LATENTS_PER_IMAGE),
        "image_indices": [row["image_idx"] for row in records],
        "candidate_latents": candidate_latents,
        "records": records,
        "output_dir": str(out_dir),
        "elapsed_seconds": float(time.perf_counter() - start_time),
    }
    save_json(out_dir / "summary.json", summary)
    print(
        "latent overlay visualization:",
        f"images={len(records)}",
        f"latents_per_image={LATENT_OVERLAY_LATENTS_PER_IMAGE}",
        f"output_dir={out_dir}",
    )
    return summary


def select_best_trial(summaries, metric, mode, l0_metric=None, l0_max=None):
    """완료된 trial 중 best를 고른다.

    l0_max가 주어지면 그 제약을 만족하는 trial만 후보다. 후보가 하나도 없으면 임의로
    하나를 고르지 않고 `selected=None`과 함께 왜 못 골랐는지를 담아 돌려준다 —
    dense한 SAE가 조용히 best로 승격되는 걸 막는 게 이 함수의 요점이다.

    반환: {"selected": <trial row | None>, "selection": {...판정 근거...}} 또는
    완료된 trial이 아예 없으면 None."""
    completed = [row for row in summaries if row.get("status") == "completed" and row.get(metric) is not None]
    if not completed:
        return None
    reverse = mode == "max"
    ranked = sorted(completed, key=lambda row: row[metric], reverse=reverse)

    if l0_max is None:
        return {
            "selected": ranked[0],
            "selection": {"constraint": None, "total_trials": len(completed)},
        }

    scored = [row for row in completed if row.get(l0_metric) is not None]
    feasible = [row for row in ranked if row.get(l0_metric) is not None and float(row[l0_metric]) <= float(l0_max)]
    closest = min(scored, key=lambda row: float(row[l0_metric])) if scored else None
    selection = {
        "constraint": {"metric": l0_metric, "max": float(l0_max)},
        "total_trials": len(completed),
        "scored_trials": len(scored),
        "feasible_trials": len(feasible),
        "closest_trial": (closest or {}).get("trial_id"),
        "closest_l0": None if closest is None else float(closest[l0_metric]),
    }
    if feasible:
        return {"selected": feasible[0], "selection": {**selection, "feasible": True}}
    return {
        "selected": None,
        "selection": {
            **selection,
            "feasible": False,
            "reason": f"{l0_metric} <= {l0_max} 를 만족하는 trial이 없다",
        },
    }


def _grid_row_from_summary(trial_id, trial_dir, overrides, summary):
    """trial summary.json 하나를 grid_summary 행으로 정규화한다.

    새로 학습한 trial과 이미 있는 trial 디렉터리를 재사용할 때가 같은 스키마를 쓰도록
    한 곳에 모아둔다(재실행 시 선정만 다시 돌릴 수 있어야 한다)."""
    metrics = summary.get("final_validation_metrics") or {}
    return {
        "trial_id": trial_id,
        "status": "completed",
        "trial_dir": str(trial_dir),
        "val_nmse": metrics.get("normalized_mse"),
        "cosine": metrics.get("cosine"),
        "mean_l0": metrics.get("mean_l0"),
        "l0_raw": metrics.get("l0_raw"),
        "l0_ratio_raw": metrics.get("l0_ratio_raw"),
        "best_epoch": summary.get("best_epoch"),
        "best_val_nmse": summary.get("best_val_nmse"),
        "best_active_mean_count": summary.get("best_active_mean_count"),
        "best_l0_raw": summary.get("best_l0_raw"),
        "l0_feasible": summary.get("best_l0_feasible"),
        "overrides": overrides,
    }


def run_sae_grid_search(base_config: SAEExperimentConfig):
    base_config = copy.deepcopy(base_config).validate()
    root_dir = Path(base_config.output.root_dir) / base_config.output.grid_dir_name
    root_dir.mkdir(parents=True, exist_ok=True)

    trial_summaries = []
    grid_items = list(iter_grid_items(base_config.grid.space))
    if base_config.grid.max_trials is not None:
        grid_items = grid_items[: int(base_config.grid.max_trials)]

    total_trials = len(grid_items)
    for trial_idx, overrides in enumerate(grid_items):
        trial_id = f"trial_{trial_idx:04d}"
        trial_dir = root_dir / trial_id
        if trial_dir.exists():
            # 완료된 trial만 재사용한다. 예전에는 디렉터리가 있으면 무조건 continue라
            # trial_summaries가 비었고 best_trial.json이 null로 덮였다.
            existing = trial_dir / "summary.json"
            done_summary = None
            if existing.exists():
                try:
                    candidate = json.loads(existing.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    candidate = None
                if candidate is not None and candidate.get("status") == "completed":
                    done_summary = candidate
            if done_summary is not None:
                trial_summaries.append(_grid_row_from_summary(trial_id, trial_dir, overrides, done_summary))
                print(f"Trial {trial_idx + 1}/{total_trials} already exists, reusing summary.json")
                continue
            # 중단되거나 실패한 trial이다. 건너뛰면 그 조합은 영원히 학습되지 않으므로
            # 다시 돌린다(run_sae_trial이 같은 디렉터리에 덮어쓴다).
            print(f"Trial {trial_idx + 1}/{total_trials} exists but is incomplete; retraining.")

        trial_start_time = time.perf_counter()
        trial_dir = root_dir / trial_id
        print(f"\n[{trial_idx + 1}/{total_trials} trial] tuning HP: {format_tuning_hp(overrides)}")
        print(f"===== {trial_id} =====")
        config = apply_grid_overrides(base_config, overrides)

        try:
            summary = run_sae_trial(config, trial_dir=trial_dir, trial_id=trial_id)
            row = _grid_row_from_summary(trial_id, trial_dir, overrides, summary)
        except Exception as exc:
            error_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            trial_dir.mkdir(parents=True, exist_ok=True)
            save_json(
                trial_dir / "summary.json",
                {
                    "trial_id": trial_id,
                    "status": "failed",
                    "trial_dir": str(trial_dir),
                    "overrides": overrides,
                    "error": error_text,
                },
            )
            row = {
                "trial_id": trial_id,
                "status": "failed",
                "trial_dir": str(trial_dir),
                "val_nmse": None,
                "cosine": None,
                "mean_l0": None,
                "l0_raw": None,
                "l0_ratio_raw": None,
                "best_epoch": None,
                "best_val_nmse": None,
                "best_active_mean_count": None,
                "best_l0_raw": None,
                "l0_feasible": None,
                "overrides": overrides,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(error_text)
        trial_summaries.append(row)
        save_json(root_dir / "grid_summary.json", trial_summaries)
        write_grid_summary_csv(root_dir / "grid_summary.csv", trial_summaries)
        trial_end_time = time.perf_counter()
        elapsed_time = trial_end_time - trial_start_time
        elapsed_str = str(timedelta(seconds=int(elapsed_time)))
        print(f"Trial {trial_id} completed in {elapsed_str}.")

    best = select_best_trial(
        trial_summaries,
        metric=base_config.grid.metric,
        mode=base_config.grid.mode,
        l0_metric=base_config.sparsity.metric,
        l0_max=base_config.sparsity.l0_max,
    )
    save_json(root_dir / "best_trial.json", best)
    if best is None:
        print("\n[WARN] 완료된 trial이 하나도 없어 best를 고르지 못했다.")
    elif best.get("selection", {}).get("feasible") is False:
        print(f"\n[WARN] 희소성 제약을 만족하는 trial이 없다: {jsonable(best['selection'])}")
    else:
        print(f"\nBest trial: {jsonable(best)}")
    return trial_summaries, best


def main():
    config = build_default_config()
    if config.grid.enabled:
        return run_sae_grid_search(config)

    trial_dir = Path(config.output.root_dir) / "single_trial"
    return run_sae_trial(config, trial_dir=trial_dir, trial_id="single_trial")


if __name__ == "__main__":
    print("Starting SAE validation script...")
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed_time)))
    print(f"SAE validation script completed in {elapsed_str}.")
