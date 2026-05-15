# %% [markdown]
# # Feature Disentanglement Experiment
# 
# This notebook follows `feature_disentangle_cell_pseudocode.md`.
# It is intentionally an orchestrator: reusable pieces should later move into `Utils/`, `Model/`, or an experiment package.
# 
# Execution order:
# 
# 1. Phase 0: setup, subset construction, perturbation preview/calibration.
# 2. Phase 1: ResNet and ViT activation/token ranking.
# 3. Phase 2: Vanilla L1 SAE smoke training.
# 4. Phase 3: SAE latent validation and perturbation sensitivity.
# 5. Phase 4: latent intervention with reconstruction/random/matched baselines.
# 
# Important interpretation rule: activation maps, attention, and Grad-CAM are only feasibility/localization aids. Causal claims belong only to Phase 4 intervention results.
# 

# %%
# Cell 1. Imports

from pathlib import Path
from dataclasses import asdict
from collections import defaultdict
import gc
import json
import math
import random
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import timm
from timm.data import resolve_model_data_config

from Utils.Config import (
    TransformHyperParams,
    DataConfig,
    DatasetSpec,
    ExtractionConfig,
    ModelSpec,
    EvalScenario,
)
from Utils.Dataset import (
    ImageNetValFlatDataset,
    ImageNetValSubsetDataset,
    ImageFolderDS,
    build_sample_indices_from_targets,
)
from Utils.utils import (
    IMAGENET_R_CLASS_IDS,
    set_seed,
    get_system_info,
    ensure_dir,
    save_json,
    build_transform,
    build_dataset,
    build_scenario_config,
    run_perturbation_validation,
    cal_accuracy,
)
from Utils.metric import (
    compute_dataset_metrics,
    evaluate_feature_metrics,
    linear_cka as repo_linear_cka,
    js_divergence,
)


# %%
# Cell 2. Experiment Settings

# 실행/재현성
SEED = 42
RUN_SYSTEM_INFO = True
DTYPE_FOR_CACHE = torch.float16
USE_PRETRAINED_WEIGHTS = True

# pretrained model 선택
RESNET_TIMM_MODEL_NAME = 'resnet50'
RESNET_PRETRAINED_WEIGHT_LABEL = 'in1k'
VIT_TIMM_MODEL_NAME = 'vit_base_patch16_224.augreg_in1k'
VIT_PRETRAINED_WEIGHT_LABEL = 'augreg_in1k'

# 경로
DATA_ROOT = 'Data'
IMAGENET_R_ROOT = 'Data/imagenet-r'
CACHE_ROOT = 'Cache'
OUTPUT_ROOT = 'outputs/feature_disentangle'

# perturbation 구성
PERTURBATIONS = ['original', 'grayscale', 'bilateral', 'patchshuffle', 'patchrotation', 'localwarp']
PERTURBATION_TO_CUE = {'original': 'none', 'grayscale': 'color', 'bilateral': 'texture', 'patchshuffle': 'shape', 'patchrotation': 'shape', 'localwarp': 'shape'}

# transform 기본값
TRANSFORM_KWARGS = dict(p=1.0, prefix='resizecrop', resize_size=256, gray_alpha=1.0, bilateral_d=11, sigma_color=170, sigma_space=75, grid_size=7, alpha_localwarp=35, sigma_localwarp=3.5)

# dataset / dataloader
SMOKE_MODE = True
SMOKE_N = 128
MAIN_N = None
DATA_BATCH_SIZE = 32 if torch.cuda.is_available() else 16
NUM_WORKERS = 0  # WSL에서 worker 복제 RAM 사용을 피하기 위해 기본 0
DATA_SHUFFLE = False
DATA_PIN_MEMORY = None  # None이면 CUDA 여부에 맞춤

# Phase 0 preview / calibration
PREVIEW_K = 4
RUN_PREVIEW = True
RUN_PERTURBATION_VALIDATION = False
VALIDATION_MAX_SAMPLES = None  # None이면 smoke subset 크기 사용
RUN_MODEL_LEVEL_CALIBRATION = False
MODEL_CALIBRATION_MAX_BATCHES = None

# Phase 1 ResNet ranking
RUN_RESNET_PHASE1 = True
RESNET_MAX_BATCHES = None
RESNET_TOP_K_CHANNELS = 20
RESNET_TARGET_LAYER_NAMES = ['layer4']  # conv1/layer1은 RAM 사용량이 큼

# Phase 1 ViT ranking
RUN_VIT_PHASE1 = True
VIT_CANDIDATE_BLOCKS = [6, 9, 11]  # 필요한 block만 늘려서 사용
VIT_MAX_BATCHES = None
VIT_TOP_K_BLOCKS = 5

# Phase 1 memory control
PHASE1_KEEP_OUTPUTS_IN_MEMORY = False  # True면 activation/token 전체를 RAM에 보관하므로 OOM 위험
PHASE1_STREAMING_RANKING = True  # True면 batch 단위로 metric만 누적

# Phase 1 visualization
RUN_RESNET_VIS = True
RUN_VIT_VIS = False
VIS_IMAGE_POSITIONS = [0, 1, 2]

# Phase 2 SAE training
RUN_SAE_TRAINING = True
SAE_SOURCE_TYPE = 'vit'  # 'vit' 또는 'resnet'
SAE_TARGET_BLOCK = 6
SAE_TARGET_LAYER = 'layer4'
SAE_USE_STD = True
SAE_MAX_TRAIN_TOKENS = 10_000
SAE_MAX_VAL_TOKENS = 5_000
SAE_BATCH_SIZE = 4096
SAE_EPOCHS = 3
SAE_SWEEP = [{'expansion_factor': 4, 'lambda_l1': 1e-4, 'lr': 3e-4}]

# Phase 3 SAE validation
RUN_SAE_VALIDATION = True
SELECTED_SAE_RUN_DIR = None  # None이면 Phase 2 선택 run 사용
LATENT_TOP_K = 20
LATENT_ACTIVE_THRESHOLD = 1e-6
HIGH_FREQ_THRESHOLD = 0.20
DEAD_FREQ_THRESHOLD = 1e-6

# Phase 4 intervention
RUN_INTERVENTION = True
INTERVENTION_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
INTERVENTION_MAX_BATCHES = None
RANDOM_BASELINE_REPEATS = 3

set_seed(SEED)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_root = Path(DATA_ROOT)
imagenet_r_root = Path(IMAGENET_R_ROOT)
cache_root = Path(CACHE_ROOT)
output_root = Path(OUTPUT_ROOT)

phase0_dir = output_root / 'phase0_perturbation_calibration'
phase1_dir = output_root / 'phase1_activation_maps'
phase2_dir = output_root / 'phase2_sae'
phase3_dir = output_root / 'phase3_latent_validation'
phase4_dir = output_root / 'phase4_intervention'

for d in [output_root, phase0_dir, phase1_dir, phase2_dir, phase3_dir, phase4_dir]:
    ensure_dir(d)

perturbations = list(PERTURBATIONS)
perturbation_to_cue = dict(PERTURBATION_TO_CUE)
shape_perturbations = [p for p, cue in perturbation_to_cue.items() if cue == 'shape']
transform_hparams = TransformHyperParams(**TRANSFORM_KWARGS)

run_config = {
    'seed': SEED,
    'device': str(DEVICE),
    'dtype_for_cache': str(DTYPE_FOR_CACHE),
    'use_pretrained_weights': USE_PRETRAINED_WEIGHTS,
    'models': {'resnet': RESNET_TIMM_MODEL_NAME, 'vit': VIT_TIMM_MODEL_NAME},
    'paths': {'data_root': str(data_root), 'imagenet_r_root': str(imagenet_r_root), 'cache_root': str(cache_root), 'output_root': str(output_root)},
    'perturbations': perturbations,
    'perturbation_to_cue': perturbation_to_cue,
    'transform_hparams': asdict(transform_hparams),
}
save_json(run_config, output_root / 'run_config.json')

if RUN_SYSTEM_INFO:
    get_system_info()

print(f'Using device: {DEVICE}')
print(f'Outputs: {output_root.resolve()}')


# %% [markdown]
# ## Phase 0. Dataset And Perturbation Calibration
# 
# The subset is ImageNet validation filtered to ImageNet-R aligned 200 classes. The same selected image indices are reused across clean, perturbed, model, and intervention comparisons.
# 

# %%
# Cell 3. Dataset Subset Builder


base_ds = ImageNetValFlatDataset(root=str(data_root), transform=None)
imagenet_200_indices = build_sample_indices_from_targets(
    targets=base_ds.targets,
    class_ids=IMAGENET_R_CLASS_IDS,
)
print(f'ImageNet-R aligned ImageNet val images: {len(imagenet_200_indices):,}')

if SMOKE_MODE:
    rng = np.random.default_rng(SEED)
    n = min(SMOKE_N, len(imagenet_200_indices))
    selected_indices = sorted(int(i) for i in rng.choice(imagenet_200_indices, size=n, replace=False))
elif MAIN_N is not None:
    rng = np.random.default_rng(SEED)
    n = min(MAIN_N, len(imagenet_200_indices))
    selected_indices = sorted(int(i) for i in rng.choice(imagenet_200_indices, size=n, replace=False))
else:
    selected_indices = [int(i) for i in imagenet_200_indices]

imagenet_200_spec = DatasetSpec(
    name='imagenet_200',
    dataset_type='imagenet_val_subset',
    root=str(data_root),
    split='val',
    num_classes=200,
    class_map_name='imagenet_r_subset_map',
    sample_indices=selected_indices,
    labels_map=[int(x) for x in IMAGENET_R_CLASS_IDS],
    id_dataset_name='imagenet_200',
)

imagenet_r_spec = DatasetSpec(
    name='imagenet_r',
    dataset_type='imagenet_r',
    root=str(imagenet_r_root),
    split='val',
    num_classes=200,
    domain_type='natural_ood',
    shift_type='style',
    class_map_name='imagenet_r_subset_map',
    eval_protocol_name='imagenet_r_eval',
    id_dataset_name='imagenet_200',
)

data_config = DataConfig(
    batch_size=DATA_BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available() if DATA_PIN_MEMORY is None else DATA_PIN_MEMORY,
    shuffle=DATA_SHUFFLE,
    datasets=[imagenet_200_spec, imagenet_r_spec],
)

save_json(
    {
        'smoke_mode': SMOKE_MODE,
        'smoke_n': SMOKE_N,
        'main_n': MAIN_N,
        'num_selected': len(selected_indices),
        'selected_indices': selected_indices,
    },
    phase0_dir / 'selected_indices.json',
)


def build_loader(dataset_spec, perturbation, model_spec, normalize=True, batch_size=None, shuffle=False):
    transform = build_transform(
        perturbation=perturbation,
        mean=model_spec.mean,
        std=model_spec.std,
        resize_size=model_spec.resize_size,
        hparams=transform_hparams,
        normalize=normalize,
    )
    dataset = build_dataset(dataset_spec, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size or data_config.batch_size,
        shuffle=shuffle,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
    )



def resize_size_from_timm_config(data_cfg):
    input_size = data_cfg.get('input_size', (3, 224, 224))
    image_size = int(input_size[-1])
    crop_pct = float(data_cfg.get('crop_pct') or 1.0)
    return int(math.floor(image_size / crop_pct)) if crop_pct > 0 else image_size


def make_timm_model_spec(
    timm_model_name,
    notebook_model_name,
    pretrained_weight_label,
    pretrained=USE_PRETRAINED_WEIGHTS,
):
    model = timm.create_model(timm_model_name, pretrained=pretrained)
    data_cfg = resolve_model_data_config(model)
    input_size = data_cfg.get('input_size', (3, 224, 224))
    if int(input_size[-1]) != 224:
        raise ValueError(
            f'The current perturbation transform center-crops to 224, but {timm_model_name} '
            f'expects input_size={input_size}. Update Utils.transfrom.get_transform first.'
        )
    return ModelSpec(
        model_name=notebook_model_name,
        pretrained_weight=pretrained_weight_label if pretrained else 'random',
        model=model,
        mean=list(data_cfg['mean']),
        std=list(data_cfg['std']),
        resize_size=resize_size_from_timm_config(data_cfg),
    )


def make_resnet50_spec(pretrained=USE_PRETRAINED_WEIGHTS):
    return make_timm_model_spec(
        timm_model_name=RESNET_TIMM_MODEL_NAME,
        notebook_model_name='resnet50',
        pretrained_weight_label=RESNET_PRETRAINED_WEIGHT_LABEL,
        pretrained=pretrained,
    )


def make_vit_b16_spec(pretrained=USE_PRETRAINED_WEIGHTS):
    return make_timm_model_spec(
        timm_model_name=VIT_TIMM_MODEL_NAME,
        notebook_model_name='vit-b',
        pretrained_weight_label=VIT_PRETRAINED_WEIGHT_LABEL,
        pretrained=pretrained,
    )

print(f'Selected subset size: {len(selected_indices):,}')
print(f'Data batch size: {data_config.batch_size}')


# %%
# Shared metric helpers for Phase 1+


def safe_linear_cka(X, Y, eps=1e-12, max_rows=None):
    X = X.detach().float().reshape(X.shape[0], -1).cpu()
    Y = Y.detach().float().reshape(Y.shape[0], -1).cpu()
    if max_rows is not None and X.shape[0] > max_rows:
        idx = torch.linspace(0, X.shape[0] - 1, steps=max_rows).long()
        X = X[idx]
        Y = Y[idx]
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Gram-form linear CKA avoids building a feature-by-feature matrix.
    xy = X @ Y.T
    xx = X @ X.T
    yy = Y @ Y.T
    hsic_xy = xy.pow(2).sum()
    hsic_xx = xx.pow(2).sum()
    hsic_yy = yy.pow(2).sum()
    return float((hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + eps)).item())


def activation_to_metric_matrix(A, pool_hw=14, max_images=256):
    A = A.detach().float().cpu()
    if max_images is not None:
        A = A[:max_images]
    if A.ndim == 4 and min(A.shape[-2:]) > pool_hw:
        A = F.adaptive_avg_pool2d(A, output_size=(pool_hw, pool_hw))
    return A.flatten(1)


def cosine_distance_from_matrices(X, Y):
    return float((1.0 - F.cosine_similarity(X.float(), Y.float(), dim=1).mean()).item())


def l2_distance_from_matrices(X, Y):
    return float((X.float() - Y.float()).norm(dim=1).mean().item())


def specificity_columns(df, prefix='sensitivity'):
    value_cols = [c for c in df.columns if c.startswith(prefix + '_')]
    perturb_cols = {c.removeprefix(prefix + '_'): c for c in value_cols}
    return value_cols, perturb_cols


# %%
# Cell 4. Perturbation Preview and Validation

validation_max_samples = VALIDATION_MAX_SAMPLES if VALIDATION_MAX_SAMPLES is not None else (min(SMOKE_N, len(selected_indices)) if SMOKE_MODE else None)


def to_display_image(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = x.permute(1, 2, 0).numpy()
        else:
            x = x.numpy()
    x = np.asarray(x)
    if x.dtype != np.uint8:
        if x.max() <= 1.5:
            x = np.clip(x * 255.0, 0, 255)
        else:
            x = np.clip(x, 0, 255)
        x = x.astype(np.uint8)
    return x


def build_visual_transforms(perturbation_names):
    return {
        p: build_transform(
            perturbation=p,
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            resize_size=256,
            hparams=transform_hparams,
            normalize=False,
        )
        for p in perturbation_names
    }


def save_perturbation_preview(sample_global_idx, visual_transforms, save_dir):
    raw_image, label = base_ds[sample_global_idx]
    fig, axes = plt.subplots(1, len(perturbations), figsize=(3.0 * len(perturbations), 3.4))
    if len(perturbations) == 1:
        axes = [axes]

    for ax, perturbation in zip(axes, perturbations):
        transformed = visual_transforms[perturbation](raw_image.copy())
        ax.imshow(to_display_image(transformed))
        ax.set_title(f'{perturbation}\n{perturbation_to_cue[perturbation]}')
        ax.axis('off')

    fig.suptitle(f'global index={sample_global_idx}, label={label}', y=1.04)
    fig.tight_layout()
    save_path = Path(save_dir) / f'perturbation_preview_{sample_global_idx}.png'
    fig.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    return save_path


preview_paths = []
if RUN_PREVIEW:
    visual_transforms = build_visual_transforms(perturbations)
    for sample_global_idx in selected_indices[:PREVIEW_K]:
        preview_paths.append(save_perturbation_preview(sample_global_idx, visual_transforms, phase0_dir))
    print('Saved preview images:')
    for path in preview_paths:
        print(f'  {path}')

validation_df = pd.DataFrame()
if RUN_PERTURBATION_VALIDATION:
    validation_result = run_perturbation_validation(
        transform_hparams=transform_hparams,
        perturbations=perturbations,
        max_samples=validation_max_samples,
        verbose_image=False,
        max_workers=max(1, min(len(perturbations) - 1, NUM_WORKERS)),
    )
    rows = []
    for perturbation, record in validation_result.results.items():
        rows.append({
            'perturbation': perturbation,
            'cue': perturbation_to_cue.get(perturbation, 'unknown'),
            'config_hash': record.config_hash,
            **record.metrics,
        })
    validation_df = pd.DataFrame(rows).sort_values(['cue', 'perturbation'])
    validation_df.to_csv(phase0_dir / 'perturbation_input_metrics.csv', index=False)
    display(validation_df)
else:
    print('RUN_PERTURBATION_VALIDATION=False, skipped input-level calibration run.')


@torch.no_grad()
def extract_logits_and_reps(model, loader, device=DEVICE, max_batches=None):
    model.eval().to(device)
    logits_list, reps_list, labels_list = [], [], []

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc='extract logits/reps')):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)

        features = model.forward_features(images)
        try:
            reps = model.forward_head(features, pre_logits=True)
            logits = model.forward_head(features, pre_logits=False)
        except TypeError:
            logits = model(images)
            reps = features.flatten(1) if features.ndim > 2 else features

        logits_list.append(logits.detach().float().cpu())
        reps_list.append(reps.detach().float().cpu())
        labels_list.append(labels.detach().cpu())

    return {
        'logits': torch.cat(logits_list, dim=0),
        'representations': torch.cat(reps_list, dim=0),
        'labels': torch.cat(labels_list, dim=0),
    }


def run_model_level_calibration(model_specs, dataset_spec, perturbation_names, max_batches=None):
    rows = []
    for model_spec in model_specs:
        outputs = {}
        for perturbation in perturbation_names:
            loader = build_loader(dataset_spec, perturbation, model_spec)
            outputs[perturbation] = extract_logits_and_reps(model_spec.model, loader, max_batches=max_batches)

        clean = outputs['original']
        clean_acc = cal_accuracy(clean['logits'], clean['labels'], class_map_name=dataset_spec.class_map_name)

        for perturbation in perturbation_names:
            current = outputs[perturbation]
            acc = cal_accuracy(current['logits'], current['labels'], class_map_name=dataset_spec.class_map_name)
            row = {
                'model': model_spec.model_name,
                'pretrained_weight': model_spec.pretrained_weight,
                'perturbation': perturbation,
                'cue': perturbation_to_cue.get(perturbation, 'unknown'),
                'accuracy': acc,
                'accuracy_drop_vs_clean': clean_acc - acc,
            }
            if perturbation != 'original':
                row['js_divergence'] = js_divergence(clean['logits'], current['logits'], return_float=True)
                row['cka'] = safe_linear_cka(clean['representations'], current['representations'])
            rows.append(row)

        del outputs
        model_spec.model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(phase0_dir / 'perturbation_model_metrics.csv', index=False)
    return df

if RUN_MODEL_LEVEL_CALIBRATION:
    calib_specs = [make_resnet50_spec(), make_vit_b16_spec()]
    model_calibration_df = run_model_level_calibration(
        calib_specs,
        imagenet_200_spec,
        perturbations,
        max_batches=MODEL_CALIBRATION_MAX_BATCHES,
    )
    display(model_calibration_df)
else:
    print('RUN_MODEL_LEVEL_CALIBRATION=False, skipped model-level calibration run.')


# %% [markdown]
# ## Phase 1. Activation And Token Ranking
# 
# Phase 1 only checks whether perturbations produce stable, localized representation changes. Do not treat these rankings as causal evidence.
# 

# %%
# Cell 5. ResNet Activation Extraction and Ranking

resnet_outputs = globals().get('resnet_outputs', {})
resnet_layer_df = globals().get('resnet_layer_df', pd.DataFrame())
resnet_channel_df = globals().get('resnet_channel_df', pd.DataFrame())
resnet_top_channels = globals().get('resnet_top_channels', {})


def get_resnet_target_layers(model):
    all_layers = {
        'conv1': model.conv1,
        'layer1': model.layer1,
        'layer2': model.layer2,
        'layer3': model.layer3,
        'layer4': model.layer4,
    }
    return {name: all_layers[name] for name in RESNET_TARGET_LAYER_NAMES}


def get_resnet_layers_by_name(model, layer_names):
    all_layers = {
        'conv1': model.conv1,
        'layer1': model.layer1,
        'layer2': model.layer2,
        'layer3': model.layer3,
        'layer4': model.layer4,
    }
    return {name: all_layers[name] for name in layer_names}


def register_activation_hooks(target_layers, activation_cache):
    handles = []
    for layer_name, module in target_layers.items():
        def make_hook(name):
            def hook(_module, _inputs, output):
                activation_cache[name] = output.detach().float().cpu()
            return hook
        handles.append(module.register_forward_hook(make_hook(layer_name)))
    return handles


@torch.no_grad()
def forward_with_resnet_activations(model, images, target_layers, device=DEVICE):
    temp_activations = {}
    handles = register_activation_hooks(target_layers, temp_activations)
    try:
        logits = model(images.to(device, non_blocking=True))
        activations = {name: temp_activations[name] for name in target_layers}
    finally:
        for handle in handles:
            handle.remove()
    return logits.detach().float().cpu(), activations


def init_metric_accumulator():
    return {'n_batches': 0, 'cka': 0.0, 'cosine_distance': 0.0, 'l2_distance': 0.0, 'delta_mean': 0.0}


def update_metric_accumulator(acc, A_o, A_p):
    X = activation_to_metric_matrix(A_o, max_images=None)
    Y = activation_to_metric_matrix(A_p, max_images=None)
    acc['n_batches'] += 1
    acc['cka'] += safe_linear_cka(X, Y)
    acc['cosine_distance'] += cosine_distance_from_matrices(X, Y)
    acc['l2_distance'] += l2_distance_from_matrices(X, Y)
    acc['delta_mean'] += float((X - Y).abs().mean().item())


def finalize_metric_accumulator(acc):
    n = max(1, acc['n_batches'])
    return {k: (v / n if k != 'n_batches' else v) for k, v in acc.items()}


def standardize_spatial_channels(A, eps=1e-6):
    A = A.detach().float()
    mean = A.mean(dim=(2, 3), keepdim=True)
    std = A.std(dim=(2, 3), keepdim=True).clamp_min(eps)
    return (A - mean) / std


@torch.no_grad()
def run_resnet_streaming_phase1(model, model_spec, dataset_spec, perturbation_names, max_batches=None):
    model.eval().to(DEVICE)
    target_layers = get_resnet_target_layers(model)
    layer_metrics = {
        (perturbation, layer_name): init_metric_accumulator()
        for perturbation in perturbation_names if perturbation != 'original'
        for layer_name in target_layers
    }
    channel_sums = defaultdict(dict)
    channel_counts = defaultdict(dict)

    for perturbation in perturbation_names:
        if perturbation == 'original':
            continue

        clean_loader = build_loader(dataset_spec, 'original', model_spec)
        pert_loader = build_loader(dataset_spec, perturbation, model_spec)
        desc = f'ResNet streaming original vs {perturbation}'

        for batch_idx, ((clean_images, _), (pert_images, _)) in enumerate(tqdm(zip(clean_loader, pert_loader), total=len(clean_loader), desc=desc)):
            if max_batches is not None and batch_idx >= max_batches:
                break

            _, A_clean = forward_with_resnet_activations(model, clean_images, target_layers)
            _, A_pert = forward_with_resnet_activations(model, pert_images, target_layers)

            for layer_name in target_layers:
                A_o = A_clean[layer_name]
                A_p = A_pert[layer_name]
                update_metric_accumulator(layer_metrics[(perturbation, layer_name)], A_o, A_p)

                sens = (standardize_spatial_channels(A_o) - standardize_spatial_channels(A_p)).abs().mean(dim=(0, 2, 3)).cpu()
                if perturbation not in channel_sums[layer_name]:
                    channel_sums[layer_name][perturbation] = torch.zeros_like(sens)
                    channel_counts[layer_name][perturbation] = 0
                channel_sums[layer_name][perturbation] += sens * A_o.shape[0]
                channel_counts[layer_name][perturbation] += A_o.shape[0]

            del A_clean, A_pert, clean_images, pert_images
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    layer_rows = []
    for (perturbation, layer_name), acc in layer_metrics.items():
        values = finalize_metric_accumulator(acc)
        layer_rows.append({
            'model': 'resnet50',
            'layer': layer_name,
            'perturbation': perturbation,
            'cue': perturbation_to_cue[perturbation],
            **values,
        })
    layer_df = pd.DataFrame(layer_rows).sort_values(['cue', 'perturbation', 'cka'])

    channel_rows = []
    for layer_name, by_perturbation in channel_sums.items():
        mean_sens = {
            p: channel_sums[layer_name][p] / max(1, channel_counts[layer_name][p])
            for p in by_perturbation
        }
        n_channels = next(iter(mean_sens.values())).numel()
        for channel in range(n_channels):
            row = {'model': 'resnet50', 'layer': layer_name, 'channel': channel}
            for perturbation, values in mean_sens.items():
                row[f'sensitivity_{perturbation}'] = float(values[channel].item())

            non_gray = [p for p in mean_sens if p != 'grayscale']
            non_bilat = [p for p in mean_sens if p != 'bilateral']
            shape_ps = [p for p in shape_perturbations if p in mean_sens]
            color_texture_ps = [p for p in ['grayscale', 'bilateral'] if p in mean_sens]
            row['color_specificity'] = row.get('sensitivity_grayscale', 0.0) - float(np.mean([row[f'sensitivity_{p}'] for p in non_gray]))
            row['texture_specificity'] = row.get('sensitivity_bilateral', 0.0) - float(np.mean([row[f'sensitivity_{p}'] for p in non_bilat]))
            row['shape_specificity'] = float(np.mean([row[f'sensitivity_{p}'] for p in shape_ps])) - float(np.mean([row[f'sensitivity_{p}'] for p in color_texture_ps]))
            channel_rows.append(row)

    channel_df = pd.DataFrame(channel_rows)
    return layer_df, channel_df


def select_top_resnet_channels(channel_df, top_k=20):
    top = {}
    if channel_df.empty:
        return top
    for cue in ['color', 'texture', 'shape']:
        score_col = f'{cue}_specificity'
        cols = ['model', 'layer', 'channel', score_col]
        top[cue] = (
            channel_df.sort_values(score_col, ascending=False)
            .head(top_k)[cols]
            .to_dict(orient='records')
        )
    return top


def unique_in_order(values):
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def make_position_subset_spec(dataset_spec, image_positions, name_suffix='vis_subset'):
    invalid_positions = [pos for pos in image_positions if pos < 0 or pos >= len(selected_indices)]
    if invalid_positions:
        raise IndexError(
            f'VIS_IMAGE_POSITIONS contains out-of-range positions {invalid_positions}; '
            f'valid range is 0..{len(selected_indices) - 1}.'
        )

    subset_indices = [int(selected_indices[pos]) for pos in image_positions]
    return DatasetSpec(
        name=f'{dataset_spec.name}_{name_suffix}',
        dataset_type=dataset_spec.dataset_type,
        root=dataset_spec.root,
        split=dataset_spec.split,
        num_classes=dataset_spec.num_classes,
        sample_indices=subset_indices,
        labels_map=list(dataset_spec.labels_map),
        id_dataset_name=dataset_spec.id_dataset_name,
        domain_type=dataset_spec.domain_type,
        shift_type=dataset_spec.shift_type,
        class_map_name=dataset_spec.class_map_name,
        eval_protocol_name=dataset_spec.eval_protocol_name,
    )


def required_resnet_visualization_inputs(top_channels, image_positions, records_per_cue=3):
    layers = []
    perturbation_names = ['original']

    for cue, records in top_channels.items():
        perturbation = next((p for p, c in perturbation_to_cue.items() if c == cue), None)
        if perturbation is not None:
            perturbation_names.append(perturbation)
        for record in records[:records_per_cue]:
            if 'layer' in record:
                layers.append(record['layer'])

    if not layers:
        layers = list(RESNET_TARGET_LAYER_NAMES)

    return {
        'layers': unique_in_order(layers),
        'perturbations': unique_in_order(perturbation_names),
        'image_positions': unique_in_order([int(pos) for pos in image_positions]),
    }


def resnet_outputs_cover_visualization(outputs, requirements):
    if not outputs:
        return False
    for perturbation in requirements['perturbations']:
        if perturbation not in outputs:
            return False
        activations = outputs[perturbation].get('activations', {})
        if any(layer not in activations for layer in requirements['layers']):
            return False
        cached_positions = outputs[perturbation].get('image_positions')
        if cached_positions is not None:
            if any(pos not in cached_positions for pos in requirements['image_positions']):
                return False
        else:
            max_pos = max(requirements['image_positions'], default=-1)
            first_layer = requirements['layers'][0]
            if first_layer not in activations or activations[first_layer].shape[0] <= max_pos:
                return False
    return True


@torch.no_grad()
def extract_resnet_outputs_for_visualization(
    model,
    model_spec,
    dataset_spec,
    perturbation_names,
    layer_names,
    image_positions,
):
    image_positions = unique_in_order([int(pos) for pos in image_positions])
    vis_spec = make_position_subset_spec(dataset_spec, image_positions, name_suffix='resnet_vis')
    target_layers = get_resnet_layers_by_name(model, layer_names)
    outputs = {}

    model.eval().to(DEVICE)
    for perturbation in perturbation_names:
        loader = build_loader(
            vis_spec,
            perturbation,
            model_spec,
            batch_size=max(1, min(len(image_positions), data_config.batch_size)),
            shuffle=False,
        )
        logits_list = []
        labels_list = []
        activation_lists = {layer_name: [] for layer_name in layer_names}

        for images, labels in tqdm(loader, desc=f'ResNet vis cache {perturbation}'):
            logits, activations = forward_with_resnet_activations(model, images, target_layers)
            logits_list.append(logits.to(dtype=DTYPE_FOR_CACHE))
            labels_list.append(labels.detach().cpu())
            for layer_name in layer_names:
                activation_lists[layer_name].append(activations[layer_name].to(dtype=DTYPE_FOR_CACHE))

        outputs[perturbation] = {
            'logits': torch.cat(logits_list, dim=0),
            'labels': torch.cat(labels_list, dim=0),
            'activations': {
                layer_name: torch.cat(chunks, dim=0)
                for layer_name, chunks in activation_lists.items()
            },
            'image_positions': image_positions,
            'selected_indices': [int(selected_indices[pos]) for pos in image_positions],
            'cache_scope': 'visualization',
        }

    outputs['_meta'] = {
        'cache_scope': 'visualization',
        'image_positions': image_positions,
        'selected_indices': [int(selected_indices[pos]) for pos in image_positions],
        'layers': list(layer_names),
        'perturbations': list(perturbation_names),
    }
    return outputs


if RUN_RESNET_PHASE1:
    if PHASE1_KEEP_OUTPUTS_IN_MEMORY:
        print('[WARN] PHASE1_KEEP_OUTPUTS_IN_MEMORY=True can use a lot of RAM. Streaming ranking is still used here.')

    resnet_spec = make_resnet50_spec()
    resnet = resnet_spec.model
    resnet_layer_df, resnet_channel_df = run_resnet_streaming_phase1(
        resnet,
        resnet_spec,
        imagenet_200_spec,
        perturbations,
        max_batches=RESNET_MAX_BATCHES,
    )

    resnet_layer_df.to_csv(phase1_dir / 'phase1_resnet_layer_ranking.csv', index=False)
    resnet_channel_df.to_csv(phase1_dir / 'phase1_resnet_channel_ranking.csv', index=False)
    resnet_top_channels = select_top_resnet_channels(resnet_channel_df, top_k=RESNET_TOP_K_CHANNELS)
    save_json(resnet_top_channels, phase1_dir / 'phase1_resnet_top_channels.json')

    if RUN_RESNET_VIS:
        resnet_vis_requirements = required_resnet_visualization_inputs(resnet_top_channels, VIS_IMAGE_POSITIONS)
        if not resnet_outputs_cover_visualization(resnet_outputs, resnet_vis_requirements):
            print(
                'Caching small ResNet outputs for visualization: '
                f"images={resnet_vis_requirements['image_positions']}, "
                f"layers={resnet_vis_requirements['layers']}, "
                f"perturbations={resnet_vis_requirements['perturbations']}"
            )
            resnet_outputs = extract_resnet_outputs_for_visualization(
                resnet,
                resnet_spec,
                imagenet_200_spec,
                resnet_vis_requirements['perturbations'],
                resnet_vis_requirements['layers'],
                resnet_vis_requirements['image_positions'],
            )

    display(resnet_layer_df)
    display(resnet_channel_df.head())

    resnet.cpu()
    del resnet
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
else:
    print('RUN_RESNET_PHASE1=False, skipped ResNet activation extraction.')


# %%
# Cell 6. ViT Patch Token Extraction and Ranking

vit_outputs = globals().get('vit_outputs', {})
vit_block_df = globals().get('vit_block_df', pd.DataFrame())
vit_top_blocks = globals().get('vit_top_blocks', {})


def register_vit_block_hooks(model, block_indices, hidden_cache):
    handles = []
    for block_idx in block_indices:
        def make_hook(idx):
            def hook(_module, _inputs, output):
                hidden_cache[idx] = output.detach().float().cpu()
            return hook
        handles.append(model.blocks[block_idx].register_forward_hook(make_hook(block_idx)))
    return handles


@torch.no_grad()
def forward_with_vit_tokens(model, images, block_indices, device=DEVICE):
    temp_hidden = {}
    handles = register_vit_block_hooks(model, block_indices, temp_hidden)
    try:
        features = model.forward_features(images.to(device, non_blocking=True))
        logits = model.forward_head(features)
        patch_tokens = {idx: temp_hidden[idx][:, 1:, :].cpu() for idx in block_indices}
        cls_tokens = {idx: temp_hidden[idx][:, 0, :].cpu() for idx in block_indices}
    finally:
        for handle in handles:
            handle.remove()
    return logits.detach().float().cpu(), patch_tokens, cls_tokens


@torch.no_grad()
def run_vit_streaming_phase1(model, model_spec, dataset_spec, perturbation_names, block_indices, max_batches=None):
    model.eval().to(DEVICE)
    metric_sums = {
        (perturbation, block_idx): init_metric_accumulator()
        for perturbation in perturbation_names if perturbation != 'original'
        for block_idx in block_indices
    }

    for perturbation in perturbation_names:
        if perturbation == 'original':
            continue

        clean_loader = build_loader(dataset_spec, 'original', model_spec)
        pert_loader = build_loader(dataset_spec, perturbation, model_spec)
        desc = f'ViT streaming original vs {perturbation}'

        for batch_idx, ((clean_images, _), (pert_images, _)) in enumerate(tqdm(zip(clean_loader, pert_loader), total=len(clean_loader), desc=desc)):
            if max_batches is not None and batch_idx >= max_batches:
                break

            _, H_clean, _ = forward_with_vit_tokens(model, clean_images, block_indices)
            _, H_pert, _ = forward_with_vit_tokens(model, pert_images, block_indices)

            for block_idx in block_indices:
                H_o = H_clean[block_idx].float()
                H_p = H_pert[block_idx].float()
                acc = metric_sums[(perturbation, block_idx)]
                X = H_o.flatten(1)
                Y = H_p.flatten(1)
                token_X = H_o.reshape(-1, H_o.shape[-1])
                token_Y = H_p.reshape(-1, H_p.shape[-1])
                acc['n_batches'] += 1
                acc['cka'] += safe_linear_cka(X, Y)
                acc['cosine_distance'] += cosine_distance_from_matrices(token_X, token_Y)
                acc['l2_distance'] += l2_distance_from_matrices(token_X, token_Y)
                acc['delta_mean'] += float((H_o - H_p).norm(dim=-1).mean().item())

            del H_clean, H_pert, clean_images, pert_images
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    rows = []
    for (perturbation, block_idx), acc in metric_sums.items():
        values = finalize_metric_accumulator(acc)
        rows.append({
            'model': 'vit-b',
            'block': block_idx,
            'perturbation': perturbation,
            'cue': perturbation_to_cue[perturbation],
            'block_score': values.pop('delta_mean'),
            **values,
        })
    return pd.DataFrame(rows).sort_values(['cue', 'perturbation', 'cka'])


def select_top_vit_blocks(block_df, top_k=5):
    top = {}
    if block_df.empty:
        return top
    for cue in ['color', 'texture', 'shape']:
        cue_df = block_df[block_df['cue'] == cue].copy()
        if cue_df.empty:
            top[cue] = []
            continue
        summary = (
            cue_df.groupby('block', as_index=False)
            .agg(mean_delta=('block_score', 'mean'), mean_cka=('cka', 'mean'))
        )
        summary['rank_score'] = summary['mean_delta'] - summary['mean_cka']
        top[cue] = (
            summary.sort_values('rank_score', ascending=False)
            .head(top_k)
            .to_dict(orient='records')
        )
    return top


def vit_patch_delta_maps(outputs, block_idx, perturbation, image_pos):
    H_o = outputs['original']['patch_tokens'][block_idx][image_pos].float()
    H_p = outputs[perturbation]['patch_tokens'][block_idx][image_pos].float()
    patch_norm_clean = H_o.norm(dim=-1)
    patch_norm_perturbed = H_p.norm(dim=-1)
    patch_delta = (H_o - H_p).norm(dim=-1)
    n = patch_delta.numel()
    grid = int(math.sqrt(n))
    if grid * grid != n:
        raise ValueError(f'Patch count is not square: {n}')
    return {
        'patch_norm_clean': patch_norm_clean.reshape(grid, grid),
        'patch_norm_perturbed': patch_norm_perturbed.reshape(grid, grid),
        'patch_delta': patch_delta.reshape(grid, grid),
    }


if RUN_VIT_PHASE1:
    if PHASE1_KEEP_OUTPUTS_IN_MEMORY:
        print('[WARN] PHASE1_KEEP_OUTPUTS_IN_MEMORY=True can use a lot of RAM. Streaming ranking is still used here.')

    vit_spec = make_vit_b16_spec()
    vit = vit_spec.model
    vit_block_df = run_vit_streaming_phase1(
        vit,
        vit_spec,
        imagenet_200_spec,
        perturbations,
        VIT_CANDIDATE_BLOCKS,
        max_batches=VIT_MAX_BATCHES,
    )
    vit_block_df.to_csv(phase1_dir / 'phase1_vit_block_ranking.csv', index=False)
    vit_top_blocks = select_top_vit_blocks(vit_block_df, top_k=VIT_TOP_K_BLOCKS)
    save_json(vit_top_blocks, phase1_dir / 'phase1_vit_top_blocks.json')

    display(vit_block_df)
    print(vit_top_blocks)

    vit.cpu()
    del vit
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
else:
    print('RUN_VIT_PHASE1=False, skipped ViT token extraction.')


# %%
# Cell 7. Visualization Utilities



def normalize_map(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    x = x - float(np.nanmin(x))
    denom = float(np.nanmax(x)) + eps
    return x / denom


def overlay_heatmap(image, heatmap, alpha=0.45, cmap_name='magma'):
    image = to_display_image(image).astype(np.float32) / 255.0
    heatmap = normalize_map(heatmap)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    colored = plt.get_cmap(cmap_name)(heatmap)[..., :3]
    overlay = (1.0 - alpha) * image + alpha * colored
    return np.clip(overlay, 0.0, 1.0)


def plot_resnet_channel_comparison(raw_image, A_original, A_perturbed, layer, channel, perturbation, save_path):
    original_map = A_original[channel].detach().float().cpu().numpy()
    perturbed_map = A_perturbed[channel].detach().float().cpu().numpy()
    diff_map = np.abs(original_map - perturbed_map)

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.3))
    axes[0].imshow(to_display_image(raw_image))
    axes[0].set_title('image')
    axes[1].imshow(normalize_map(original_map), cmap='magma')
    axes[1].set_title('clean act')
    axes[2].imshow(normalize_map(perturbed_map), cmap='magma')
    axes[2].set_title(f'{perturbation} act')
    axes[3].imshow(normalize_map(diff_map), cmap='magma')
    axes[3].set_title('abs diff')
    axes[4].imshow(overlay_heatmap(raw_image, original_map))
    axes[4].set_title('clean overlay')

    for ax in axes:
        ax.axis('off')
    fig.suptitle(f'ResNet {layer} channel={channel} cue={perturbation_to_cue[perturbation]}')
    fig.tight_layout()
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    return save_path


def plot_vit_patch_delta(raw_image, patch_norm_clean, patch_norm_perturbed, patch_delta, block, perturbation, save_path):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.3))
    axes[0].imshow(to_display_image(raw_image))
    axes[0].set_title('image')
    axes[1].imshow(normalize_map(patch_norm_clean), cmap='magma')
    axes[1].set_title('clean norm')
    axes[2].imshow(normalize_map(patch_norm_perturbed), cmap='magma')
    axes[2].set_title(f'{perturbation} norm')
    axes[3].imshow(normalize_map(patch_delta), cmap='magma')
    axes[3].set_title('delta')
    axes[4].imshow(overlay_heatmap(raw_image, patch_delta))
    axes[4].set_title('delta overlay')

    for ax in axes:
        ax.axis('off')
    fig.suptitle(f'ViT block={block} cue={perturbation_to_cue[perturbation]}')
    fig.tight_layout()
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    return save_path


def load_json_if_exists(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def unique_in_order(values):
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def make_position_subset_spec(dataset_spec, image_positions, name_suffix='vis_subset'):
    invalid_positions = [pos for pos in image_positions if pos < 0 or pos >= len(selected_indices)]
    if invalid_positions:
        raise IndexError(
            f'VIS_IMAGE_POSITIONS contains out-of-range positions {invalid_positions}; '
            f'valid range is 0..{len(selected_indices) - 1}.'
        )

    subset_indices = [int(selected_indices[pos]) for pos in image_positions]
    return DatasetSpec(
        name=f'{dataset_spec.name}_{name_suffix}',
        dataset_type=dataset_spec.dataset_type,
        root=dataset_spec.root,
        split=dataset_spec.split,
        num_classes=dataset_spec.num_classes,
        sample_indices=subset_indices,
        labels_map=list(dataset_spec.labels_map),
        id_dataset_name=dataset_spec.id_dataset_name,
        domain_type=dataset_spec.domain_type,
        shift_type=dataset_spec.shift_type,
        class_map_name=dataset_spec.class_map_name,
        eval_protocol_name=dataset_spec.eval_protocol_name,
    )


def required_resnet_visualization_inputs(top_channels, image_positions, records_per_cue=3):
    layers = []
    perturbation_names = ['original']

    for cue, records in top_channels.items():
        perturbation = next((p for p, c in perturbation_to_cue.items() if c == cue), None)
        if perturbation is not None:
            perturbation_names.append(perturbation)
        for record in records[:records_per_cue]:
            if 'layer' in record:
                layers.append(record['layer'])

    if not layers:
        layers = list(RESNET_TARGET_LAYER_NAMES)

    return {
        'layers': unique_in_order(layers),
        'perturbations': unique_in_order(perturbation_names),
        'image_positions': unique_in_order([int(pos) for pos in image_positions]),
    }


def resnet_outputs_cover_visualization(outputs, requirements):
    if not outputs:
        return False
    for perturbation in requirements['perturbations']:
        if perturbation not in outputs:
            return False
        activations = outputs[perturbation].get('activations', {})
        if any(layer not in activations for layer in requirements['layers']):
            return False
        cached_positions = outputs[perturbation].get('image_positions')
        if cached_positions is not None:
            if any(pos not in cached_positions for pos in requirements['image_positions']):
                return False
        else:
            max_pos = max(requirements['image_positions'], default=-1)
            first_layer = requirements['layers'][0]
            if first_layer not in activations or activations[first_layer].shape[0] <= max_pos:
                return False
    return True


@torch.no_grad()
def extract_resnet_outputs_for_visualization(
    model,
    model_spec,
    dataset_spec,
    perturbation_names,
    layer_names,
    image_positions,
):
    image_positions = unique_in_order([int(pos) for pos in image_positions])
    vis_spec = make_position_subset_spec(dataset_spec, image_positions, name_suffix='resnet_vis')
    all_layers = {
        'conv1': model.conv1,
        'layer1': model.layer1,
        'layer2': model.layer2,
        'layer3': model.layer3,
        'layer4': model.layer4,
    }
    target_layers = {name: all_layers[name] for name in layer_names}
    outputs = {}

    model.eval().to(DEVICE)
    for perturbation in perturbation_names:
        loader = build_loader(
            vis_spec,
            perturbation,
            model_spec,
            batch_size=max(1, min(len(image_positions), data_config.batch_size)),
            shuffle=False,
        )
        logits_list = []
        labels_list = []
        activation_lists = {layer_name: [] for layer_name in layer_names}

        for images, labels in tqdm(loader, desc=f'ResNet vis cache {perturbation}'):
            logits, activations = forward_with_resnet_activations(model, images, target_layers)
            logits_list.append(logits.to(dtype=DTYPE_FOR_CACHE))
            labels_list.append(labels.detach().cpu())
            for layer_name in layer_names:
                activation_lists[layer_name].append(activations[layer_name].to(dtype=DTYPE_FOR_CACHE))

        outputs[perturbation] = {
            'logits': torch.cat(logits_list, dim=0),
            'labels': torch.cat(labels_list, dim=0),
            'activations': {
                layer_name: torch.cat(chunks, dim=0)
                for layer_name, chunks in activation_lists.items()
            },
            'image_positions': image_positions,
            'selected_indices': [int(selected_indices[pos]) for pos in image_positions],
            'cache_scope': 'visualization',
        }

    outputs['_meta'] = {
        'cache_scope': 'visualization',
        'image_positions': image_positions,
        'selected_indices': [int(selected_indices[pos]) for pos in image_positions],
        'layers': list(layer_names),
        'perturbations': list(perturbation_names),
    }
    return outputs


def resnet_cached_image_index(outputs, perturbation, image_pos):
    cached_positions = outputs[perturbation].get('image_positions')
    if cached_positions is None:
        return int(image_pos)
    if int(image_pos) not in cached_positions:
        raise IndexError(f'image_pos={image_pos} is not in the ResNet visualization cache.')
    return cached_positions.index(int(image_pos))


if RUN_RESNET_VIS:
    resnet_top_channels = resnet_top_channels or load_json_if_exists(phase1_dir / 'phase1_resnet_top_channels.json', {})
    resnet_vis_requirements = required_resnet_visualization_inputs(resnet_top_channels, VIS_IMAGE_POSITIONS)
    if not resnet_outputs_cover_visualization(resnet_outputs, resnet_vis_requirements):
        print(
            'Building small ResNet visualization cache: '
            f"images={resnet_vis_requirements['image_positions']}, "
            f"layers={resnet_vis_requirements['layers']}, "
            f"perturbations={resnet_vis_requirements['perturbations']}"
        )
        resnet_vis_spec = make_resnet50_spec()
        resnet_outputs = extract_resnet_outputs_for_visualization(
            resnet_vis_spec.model,
            resnet_vis_spec,
            imagenet_200_spec,
            resnet_vis_requirements['perturbations'],
            resnet_vis_requirements['layers'],
            resnet_vis_requirements['image_positions'],
        )
        resnet_vis_spec.model.cpu()
        del resnet_vis_spec
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for cue, records in resnet_top_channels.items():
        perturbation = next((p for p, c in perturbation_to_cue.items() if c == cue), None)
        if perturbation is None:
            continue
        for record in records[:3]:
            layer = record['layer']
            channel = int(record['channel'])
            for image_pos in VIS_IMAGE_POSITIONS:
                raw_image, _ = base_ds[selected_indices[image_pos]]
                save_path = phase1_dir / 'figures' / f'resnet_{layer}_channel_{channel}_{perturbation}_image_{image_pos}.png'
                clean_cache_idx = resnet_cached_image_index(resnet_outputs, 'original', image_pos)
                pert_cache_idx = resnet_cached_image_index(resnet_outputs, perturbation, image_pos)
                plot_resnet_channel_comparison(
                    raw_image,
                    resnet_outputs['original']['activations'][layer][clean_cache_idx],
                    resnet_outputs[perturbation]['activations'][layer][pert_cache_idx],
                    layer,
                    channel,
                    perturbation,
                    save_path,
                )

if RUN_VIT_VIS:
    if not vit_outputs:
        raise RuntimeError('Run Cell 5 first or load vit_outputs.')
    vit_top_blocks = vit_top_blocks or load_json_if_exists(phase1_dir / 'phase1_vit_top_blocks.json', {})
    for cue, records in vit_top_blocks.items():
        perturbation = next((p for p, c in perturbation_to_cue.items() if c == cue), None)
        if perturbation is None:
            continue
        for record in records[:3]:
            block = int(record['block'])
            for image_pos in VIS_IMAGE_POSITIONS:
                raw_image, _ = base_ds[selected_indices[image_pos]]
                maps = vit_patch_delta_maps(vit_outputs, block, perturbation, image_pos)
                save_path = phase1_dir / 'figures' / f'vit_block_{block}_{perturbation}_image_{image_pos}_patch_delta.png'
                plot_vit_patch_delta(raw_image, maps['patch_norm_clean'], maps['patch_norm_perturbed'], maps['patch_delta'], block, perturbation, save_path)

print('Visualization helpers ready.')


# %% [markdown]
# ## Phase 2. SAE Prototype
# 
# This starts with a Vanilla L1 SAE smoke test. The point is to validate activation cache, reconstruction quality, and future replacement/intervention wiring before moving to PatchSAE or other variants.
# 

# %%
# Cell 8. SAE Prototype Training


sae_train_results = globals().get('sae_train_results', [])
selected_sae_run_dir = globals().get('selected_sae_run_dir', None)


class VanillaL1SAE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.encoder = nn.Linear(self.input_dim, self.hidden_dim)
        self.decoder = nn.Linear(self.hidden_dim, self.input_dim)
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight, a=math.sqrt(5))
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x):
        return F.relu(self.encoder(x))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


def vit_tokens_to_matrix(tokens):
    B, N, D = tokens.shape
    X = tokens.reshape(B * N, D).float()
    metadata = pd.DataFrame({
        'image_pos': np.repeat(np.arange(B), N),
        'patch_id': np.tile(np.arange(N), B),
    })
    grid = int(math.sqrt(N))
    if grid * grid == N:
        metadata['patch_y'] = metadata['patch_id'] // grid
        metadata['patch_x'] = metadata['patch_id'] % grid
    return X, metadata, {'num_images': B, 'tokens_per_image': N, 'input_dim': D, 'grid': grid}


def resnet_activations_to_matrix(activations):
    B, C, H, W = activations.shape
    X = activations.permute(0, 2, 3, 1).reshape(B * H * W, C).float()
    metadata = pd.DataFrame({
        'image_pos': np.repeat(np.arange(B), H * W),
        'spatial_id': np.tile(np.arange(H * W), B),
    })
    metadata['spatial_y'] = metadata['spatial_id'] // W
    metadata['spatial_x'] = metadata['spatial_id'] % W
    return X, metadata, {'num_images': B, 'tokens_per_image': H * W, 'input_dim': C, 'height': H, 'width': W}


def build_sae_activation_matrix(source_type, perturbation='original', target_block=None, target_layer=None):
    if source_type == 'vit':
        if not vit_outputs:
            raise RuntimeError('Run Cell 5 first; vit_outputs is empty.')
        block = SAE_TARGET_BLOCK if target_block is None else target_block
        tokens = vit_outputs[perturbation]['patch_tokens'][block]
        X, metadata, meta = vit_tokens_to_matrix(tokens)
        meta.update({'source_type': 'vit', 'target_block': block, 'perturbation': perturbation})
        return X, metadata, meta

    if source_type == 'resnet':
        if not resnet_outputs:
            raise RuntimeError('Run Cell 4 first; resnet_outputs is empty.')
        layer = SAE_TARGET_LAYER if target_layer is None else target_layer
        activations = resnet_outputs[perturbation]['activations'][layer]
        X, metadata, meta = resnet_activations_to_matrix(activations)
        meta.update({'source_type': 'resnet', 'target_layer': layer, 'perturbation': perturbation})
        return X, metadata, meta

    raise ValueError(f'Unknown source_type: {source_type}')


def split_activation_matrix(X, metadata, max_train_tokens, max_val_tokens, seed=SEED):
    n = X.shape[0]
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng)
    n_train = min(max_train_tokens, n)
    n_val = min(max_val_tokens, max(0, n - n_train))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    return X[train_idx], X[val_idx], metadata.iloc[train_idx.numpy()].reset_index(drop=True), metadata.iloc[val_idx.numpy()].reset_index(drop=True)


def compute_preprocess_stats(X, use_std=True, eps=1e-6):
    mean = X.mean(dim=0)
    if use_std:
        std = X.std(dim=0).clamp_min(eps)
    else:
        std = torch.ones_like(mean)
    return {'mean': mean.cpu(), 'std': std.cpu(), 'use_std': bool(use_std), 'eps': eps}


def apply_preprocess(X, stats):
    mean = stats['mean'].to(X.device)
    std = stats['std'].to(X.device)
    return (X - mean) / std


def undo_preprocess(X_norm, stats):
    mean = stats['mean'].to(X_norm.device)
    std = stats['std'].to(X_norm.device)
    return X_norm * std + mean


@torch.no_grad()
def evaluate_sae_reconstruction(sae, X_norm, batch_size=8192, device=DEVICE, threshold=1e-6):
    sae.eval().to(device)
    x_hats, zs = [], []
    for start in range(0, X_norm.shape[0], batch_size):
        xb = X_norm[start:start + batch_size].to(device)
        x_hat, z = sae(xb)
        x_hats.append(x_hat.detach().cpu())
        zs.append(z.detach().cpu())
    x_hat = torch.cat(x_hats, dim=0)
    z = torch.cat(zs, dim=0)
    X_norm = X_norm.cpu()
    mse = F.mse_loss(x_hat, X_norm).item()
    var = X_norm.var().item()
    nmse = mse / (var + 1e-12)
    ss_res = (X_norm - x_hat).pow(2).sum()
    ss_tot = (X_norm - X_norm.mean(dim=0, keepdim=True)).pow(2).sum().clamp_min(1e-12)
    r2 = float((1.0 - ss_res / ss_tot).item())
    cosine = float(F.cosine_similarity(X_norm, x_hat, dim=1).mean().item())
    l0 = float((z > threshold).float().sum(dim=1).mean().item())
    freq = (z > threshold).float().mean(dim=0)
    dead_ratio = float((freq < 1e-6).float().mean().item())
    return {
        'mse': mse,
        'variance': var,
        'nmse': nmse,
        'r2': r2,
        'cosine': cosine,
        'l0': l0,
        'dead_latent_ratio': dead_ratio,
    }


def train_vanilla_l1_sae(X_train_norm, X_val_norm, config, run_dir):
    input_dim = X_train_norm.shape[1]
    hidden_dim = int(config['expansion_factor'] * input_dim)
    sae = VanillaL1SAE(input_dim=input_dim, hidden_dim=hidden_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(sae.parameters(), lr=config['lr'])
    run_dir = ensure_dir(run_dir)

    logs = []
    for epoch in range(config.get('epochs', SAE_EPOCHS)):
        sae.train()
        perm = torch.randperm(X_train_norm.shape[0])
        pbar = tqdm(range(0, X_train_norm.shape[0], config.get('batch_size', SAE_BATCH_SIZE)), desc=f'SAE epoch {epoch + 1}')
        for start in pbar:
            idx = perm[start:start + config.get('batch_size', SAE_BATCH_SIZE)]
            xb = X_train_norm[idx].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            x_hat, z = sae(xb)
            mse = F.mse_loss(x_hat, xb)
            l1 = z.abs().mean()
            loss = mse + config['lambda_l1'] * l1
            loss.backward()
            optimizer.step()
            logs.append({'epoch': epoch + 1, 'loss': float(loss.item()), 'mse': float(mse.item()), 'l1': float(l1.item())})
            pbar.set_postfix(loss=f'{loss.item():.4g}', mse=f'{mse.item():.4g}', l1=f'{l1.item():.4g}')

    metrics = evaluate_sae_reconstruction(sae, X_val_norm, batch_size=config.get('batch_size', SAE_BATCH_SIZE))
    torch.save(
        {
            'state_dict': sae.state_dict(),
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'config': config,
            'metrics': metrics,
        },
        run_dir / 'sae.pt',
    )
    pd.DataFrame(logs).to_csv(run_dir / 'train_log.csv', index=False)
    save_json(config, run_dir / 'sae_train_config.json')
    save_json(metrics, run_dir / 'sae_metrics.json')
    return sae, metrics


if RUN_SAE_TRAINING:
    X, token_metadata, source_meta = build_sae_activation_matrix(
        SAE_SOURCE_TYPE,
        perturbation='original',
        target_block=SAE_TARGET_BLOCK,
        target_layer=SAE_TARGET_LAYER,
    )
    X_train, X_val, train_metadata, val_metadata = split_activation_matrix(
        X,
        token_metadata,
        SAE_MAX_TRAIN_TOKENS,
        SAE_MAX_VAL_TOKENS,
    )
    preprocess_stats = compute_preprocess_stats(X_train, use_std=SAE_USE_STD)
    X_train_norm = apply_preprocess(X_train, preprocess_stats)
    X_val_norm = apply_preprocess(X_val, preprocess_stats)

    sae_train_results = []
    for sweep_idx, base_config in enumerate(SAE_SWEEP):
        config = {
            **base_config,
            'epochs': SAE_EPOCHS,
            'batch_size': SAE_BATCH_SIZE,
            'source_type': SAE_SOURCE_TYPE,
            'target_block': SAE_TARGET_BLOCK if SAE_SOURCE_TYPE == 'vit' else None,
            'target_layer': SAE_TARGET_LAYER if SAE_SOURCE_TYPE == 'resnet' else None,
            'max_train_tokens': int(X_train.shape[0]),
            'max_val_tokens': int(X_val.shape[0]),
            'use_std': SAE_USE_STD,
        }
        run_name = f"{SAE_SOURCE_TYPE}_smoke_{sweep_idx:03d}_exp{config['expansion_factor']}_l1{config['lambda_l1']:.0e}"
        run_dir = phase2_dir / run_name
        torch.save(preprocess_stats, run_dir / 'preprocess_stats.pt')
        save_json(source_meta, run_dir / 'source_meta.json')
        train_metadata.to_csv(run_dir / 'train_token_metadata.csv', index=False)
        val_metadata.to_csv(run_dir / 'val_token_metadata.csv', index=False)
        sae, metrics = train_vanilla_l1_sae(X_train_norm, X_val_norm, config, run_dir)
        sae_train_results.append({'run_dir': str(run_dir), **config, **metrics})

    sae_results_df = pd.DataFrame(sae_train_results).sort_values(['r2', 'nmse'], ascending=[False, True])
    sae_results_df.to_csv(phase2_dir / 'sae_sweep_results.csv', index=False)
    selected_sae_run_dir = Path(sae_results_df.iloc[0]['run_dir'])
    print(f'Selected SAE run: {selected_sae_run_dir}')
    display(sae_results_df)
else:
    print('RUN_SAE_TRAINING=False, skipped SAE smoke training.')


# %%
# Cell 9. SAE Validation Plots and Latent Groups

selected_sae_path = SELECTED_SAE_RUN_DIR or selected_sae_run_dir

latent_frequency_df = globals().get('latent_frequency_df', pd.DataFrame())
latent_sensitivity_df = globals().get('latent_sensitivity_df', pd.DataFrame())
latent_groups = globals().get('latent_groups', {})


def load_sae_run(run_dir):
    run_dir = Path(run_dir)
    checkpoint = torch.load(run_dir / 'sae.pt', map_location='cpu')
    sae = VanillaL1SAE(checkpoint['input_dim'], checkpoint['hidden_dim'])
    sae.load_state_dict(checkpoint['state_dict'])
    sae.eval()
    stats = torch.load(run_dir / 'preprocess_stats.pt', map_location='cpu')
    with open(run_dir / 'source_meta.json', 'r', encoding='utf-8') as f:
        source_meta = json.load(f)
    return sae, stats, source_meta, checkpoint


@torch.no_grad()
def encode_matrix_in_chunks(sae, X_norm, batch_size=8192, device=DEVICE):
    sae.eval().to(device)
    z_chunks, xhat_chunks = [], []
    for start in range(0, X_norm.shape[0], batch_size):
        xb = X_norm[start:start + batch_size].to(device)
        x_hat, z = sae(xb)
        z_chunks.append(z.detach().cpu())
        xhat_chunks.append(x_hat.detach().cpu())
    return torch.cat(z_chunks, dim=0), torch.cat(xhat_chunks, dim=0)


def build_matrix_from_source_outputs(source_meta, perturbation):
    source_type = source_meta['source_type']
    if source_type == 'vit':
        block = int(source_meta['target_block'])
        tokens = vit_outputs[perturbation]['patch_tokens'][block]
        return vit_tokens_to_matrix(tokens)
    if source_type == 'resnet':
        layer = source_meta['target_layer']
        activations = resnet_outputs[perturbation]['activations'][layer]
        return resnet_activations_to_matrix(activations)
    raise ValueError(f'Unknown source_type: {source_type}')


def encode_source_outputs(sae, stats, source_meta, perturbation):
    X, metadata, matrix_meta = build_matrix_from_source_outputs(source_meta, perturbation)
    X_norm = apply_preprocess(X, stats)
    z, x_hat_norm = encode_matrix_in_chunks(sae, X_norm)
    x_hat = undo_preprocess(x_hat_norm, stats)
    return X, z, x_hat, metadata, matrix_meta


def latent_frequency_table(z, threshold=LATENT_ACTIVE_THRESHOLD):
    active = z > threshold
    freq = active.float().mean(dim=0)
    mean_when_active = torch.zeros_like(freq)
    for latent_id in range(z.shape[1]):
        mask = active[:, latent_id]
        if mask.any():
            mean_when_active[latent_id] = z[mask, latent_id].mean()
    df = pd.DataFrame({
        'latent_id': np.arange(z.shape[1]),
        'frequency': freq.numpy(),
        'is_dead': (freq < DEAD_FREQ_THRESHOLD).numpy(),
        'is_high_frequency': (freq > HIGH_FREQ_THRESHOLD).numpy(),
        'mean_activation_when_active': mean_when_active.numpy(),
        'max_activation': z.max(dim=0).values.numpy(),
    })
    return df


def image_scores_from_z(z, matrix_meta):
    B = int(matrix_meta['num_images'])
    T = int(matrix_meta['tokens_per_image'])
    return z.reshape(B, T, z.shape[1]).amax(dim=1)


def compute_latent_suppression_sensitivity(sae, stats, source_meta, perturbation_names):
    X_original, z_original, xhat_original, metadata, matrix_meta = encode_source_outputs(sae, stats, source_meta, 'original')
    original_scores = image_scores_from_z(z_original, matrix_meta)
    original_mean = original_scores.mean(dim=0)

    rows = []
    per_perturb_mean = {}
    for perturbation in perturbation_names:
        if perturbation == 'original':
            continue
        _, z_p, _, _, matrix_meta_p = encode_source_outputs(sae, stats, source_meta, perturbation)
        scores_p = image_scores_from_z(z_p, matrix_meta_p)
        per_perturb_mean[perturbation] = scores_p.mean(dim=0)

    for latent_id in range(z_original.shape[1]):
        row = {
            'latent_id': latent_id,
            'mean_original_score': float(original_mean[latent_id].item()),
        }
        for perturbation, mean_scores in per_perturb_mean.items():
            delta = original_mean[latent_id] - mean_scores[latent_id]
            rel_delta = delta / (original_mean[latent_id].abs() + 1e-8)
            row[f'delta_{perturbation}'] = float(delta.item())
            row[f'relative_delta_{perturbation}'] = float(rel_delta.item())

        non_gray = [p for p in per_perturb_mean if p != 'grayscale']
        non_bilat = [p for p in per_perturb_mean if p != 'bilateral']
        shape_ps = [p for p in shape_perturbations if p in per_perturb_mean]
        color_texture_ps = [p for p in ['grayscale', 'bilateral'] if p in per_perturb_mean]

        row['color_specificity'] = row.get('relative_delta_grayscale', 0.0) - float(np.mean([row[f'relative_delta_{p}'] for p in non_gray]))
        row['texture_specificity'] = row.get('relative_delta_bilateral', 0.0) - float(np.mean([row[f'relative_delta_{p}'] for p in non_bilat]))
        row['shape_specificity'] = float(np.mean([row[f'relative_delta_{p}'] for p in shape_ps])) - float(np.mean([row[f'relative_delta_{p}'] for p in color_texture_ps]))
        rows.append(row)

    return pd.DataFrame(rows), z_original, metadata, matrix_meta


def select_latent_groups(sensitivity_df, frequency_df, top_k=20):
    merged = sensitivity_df.merge(frequency_df, on='latent_id', how='left')
    usable = merged[(~merged['is_dead']) & (~merged['is_high_frequency'])].copy()
    groups = {}
    for cue in ['color', 'texture', 'shape']:
        col = f'{cue}_specificity'
        groups[f'{cue}_latents'] = usable.sort_values(col, ascending=False).head(top_k)['latent_id'].astype(int).tolist()
    return groups


def plot_latent_frequency(frequency_df, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].hist(frequency_df['frequency'], bins=60)
    axes[0].set_title('latent frequency')
    axes[0].set_xlabel('P(z > threshold)')
    axes[0].set_ylabel('count')
    axes[1].bar(['dead', 'high freq'], [frequency_df['is_dead'].mean(), frequency_df['is_high_frequency'].mean()])
    axes[1].set_ylim(0, 1)
    axes[1].set_title('latent quality flags')
    fig.tight_layout()
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_latent_top_images(z, metadata, matrix_meta, latent_id, save_path, top_k=8):
    scores = z[:, latent_id]
    top_idx = torch.topk(scores, k=min(top_k, scores.numel())).indices.numpy()
    cols = min(4, len(top_idx))
    rows = int(math.ceil(len(top_idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.1 * rows))
    axes = np.asarray(axes).reshape(-1)

    for ax, token_idx in zip(axes, top_idx):
        row = metadata.iloc[int(token_idx)]
        image_pos = int(row['image_pos'])
        raw_image, _ = base_ds[selected_indices[image_pos]]
        ax.imshow(to_display_image(raw_image))
        ax.set_title(f"img={image_pos}, score={float(scores[token_idx]):.3g}")
        ax.axis('off')
    for ax in axes[len(top_idx):]:
        ax.axis('off')
    fig.suptitle(f'latent {latent_id} top activations')
    fig.tight_layout()
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


if RUN_SAE_VALIDATION:
    if selected_sae_path is None:
        raise RuntimeError('Set SELECTED_SAE_RUN_DIR to a Phase 2 run directory.')
    sae, preprocess_stats, source_meta, checkpoint = load_sae_run(selected_sae_path)
    X_original, z_original, xhat_original, token_metadata, matrix_meta = encode_source_outputs(sae, preprocess_stats, source_meta, 'original')

    X_norm = apply_preprocess(X_original, preprocess_stats)
    xhat_norm = apply_preprocess(xhat_original, preprocess_stats)
    metrics = evaluate_sae_reconstruction(sae, X_norm)
    metrics_row = {
        'source_type': source_meta['source_type'],
        'target_block': source_meta.get('target_block'),
        'target_layer': source_meta.get('target_layer'),
        'sae_run_dir': str(selected_sae_path),
        'dict_size': checkpoint['hidden_dim'],
        **metrics,
    }
    pd.DataFrame([metrics_row]).to_csv(phase3_dir / 'sae_metrics.csv', index=False)

    latent_frequency_df = latent_frequency_table(z_original)
    latent_frequency_df.to_csv(phase3_dir / 'latent_frequency.csv', index=False)
    plot_latent_frequency(latent_frequency_df, phase3_dir / 'latent_frequency_histogram.png')

    latent_sensitivity_df, z_original, token_metadata, matrix_meta = compute_latent_suppression_sensitivity(
        sae,
        preprocess_stats,
        source_meta,
        perturbations,
    )
    latent_sensitivity_df.to_csv(phase3_dir / 'latent_suppression_sensitivity.csv', index=False)

    latent_groups = select_latent_groups(latent_sensitivity_df, latent_frequency_df, top_k=LATENT_TOP_K)
    save_json(latent_groups, phase3_dir / 'latent_groups.json')

    for group_name, latent_ids in latent_groups.items():
        for latent_id in latent_ids[:5]:
            plot_latent_top_images(z_original, token_metadata, matrix_meta, latent_id, phase3_dir / 'top_images' / f'latent_{latent_id}_top_images.png')

    display(pd.DataFrame([metrics_row]))
    display(latent_frequency_df.head())
    display(latent_sensitivity_df.head())
    print(latent_groups)
else:
    print('RUN_SAE_VALIDATION=False, skipped SAE latent validation.')


# %% [markdown]
# ## Phase 4. Latent Intervention
# 
# Only this phase can support causal language. Reconstruction-only drift must be small, and cue-specific latent intervention must separate from random and matched baselines.
# 

# %%
# Cell 10. Latent Intervention



def reconstruct_sae_from_checkpoint(checkpoint):
    sae = VanillaL1SAE(checkpoint['input_dim'], checkpoint['hidden_dim'])
    sae.load_state_dict(checkpoint['state_dict'])
    return sae


def intervene_activation_tensor(x, sae, stats, source_type, latent_ids, alpha):
    original_dtype = x.dtype
    x_float = x.float()
    latent_ids = [int(i) for i in latent_ids]

    if source_type == 'vit':
        cls_token = x_float[:, :1, :]
        patch_tokens = x_float[:, 1:, :]
        B, N, D = patch_tokens.shape
        X = patch_tokens.reshape(B * N, D)
    elif source_type == 'resnet':
        B, C, H, W = x_float.shape
        X = x_float.permute(0, 2, 3, 1).reshape(B * H * W, C)
    else:
        raise ValueError(f'Unknown source_type: {source_type}')

    mean = stats['mean'].to(X.device)
    std = stats['std'].to(X.device)
    X_norm = (X - mean) / std

    with torch.no_grad():
        z = sae.encode(X_norm)
        if latent_ids:
            z[:, latent_ids] = alpha * z[:, latent_ids]
        X_hat_norm = sae.decode(z)
        X_hat = X_hat_norm * std + mean

    if source_type == 'vit':
        patch_hat = X_hat.reshape(B, N, D)
        out = torch.cat([cls_token, patch_hat], dim=1)
    else:
        out = X_hat.reshape(B, H, W, C).permute(0, 3, 1, 2)

    return out.to(dtype=original_dtype)


def register_intervention_hook(model, source_meta, sae, stats, latent_ids, alpha):
    source_type = source_meta['source_type']
    sae = sae.to(DEVICE).eval()

    if source_type == 'vit':
        block_idx = int(source_meta['target_block'])
        module = model.blocks[block_idx]
    elif source_type == 'resnet':
        layer_name = source_meta['target_layer']
        module = dict(get_resnet_target_layers(model))[layer_name]
    else:
        raise ValueError(f'Unknown source_type: {source_type}')

    def hook(_module, _inputs, output):
        return intervene_activation_tensor(output, sae, stats, source_type, latent_ids, alpha)

    return module.register_forward_hook(hook)


@torch.no_grad()
def collect_outputs_for_eval(model, loader, max_batches=None, device=DEVICE):
    model.eval().to(device)
    logits_list, reps_list, labels_list = [], [], []
    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc='intervention eval')):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        features = model.forward_features(images)
        try:
            reps = model.forward_head(features, pre_logits=True)
            logits = model.forward_head(features, pre_logits=False)
        except TypeError:
            logits = model(images)
            reps = features.flatten(1) if features.ndim > 2 else features
        logits_list.append(logits.detach().float().cpu())
        reps_list.append(reps.detach().float().cpu())
        labels_list.append(labels.detach().cpu())
    return {
        'logits': torch.cat(logits_list, dim=0),
        'representations': torch.cat(reps_list, dim=0),
        'labels': torch.cat(labels_list, dim=0),
    }


def compare_intervention_outputs(clean_outputs, intervened_outputs, dataset_spec):
    clean_logits = clean_outputs['logits']
    int_logits = intervened_outputs['logits']
    labels = clean_outputs['labels']
    if not torch.equal(labels, intervened_outputs['labels']):
        raise ValueError('Label mismatch between clean and intervened outputs.')

    clean_acc = cal_accuracy(clean_logits, labels, class_map_name=dataset_spec.class_map_name)
    int_acc = cal_accuracy(int_logits, labels, class_map_name=dataset_spec.class_map_name)
    clean_top1 = clean_logits.argmax(dim=1)
    int_top1 = int_logits.argmax(dim=1)
    target_logit_before = clean_logits.gather(1, clean_top1[:, None]).squeeze(1)
    target_logit_after = int_logits.gather(1, clean_top1[:, None]).squeeze(1)

    return {
        'clean_accuracy': clean_acc,
        'intervened_accuracy': int_acc,
        'accuracy_drop': clean_acc - int_acc,
        'target_logit_drop': float((target_logit_before - target_logit_after).mean().item()),
        'top1_flip_rate': float((clean_top1 != int_top1).float().mean().item()),
        'js_divergence': js_divergence(clean_logits, int_logits, return_float=True),
        'representation_cka': safe_linear_cka(clean_outputs['representations'], intervened_outputs['representations'], max_rows=512),
    }


def sample_random_latents(frequency_df, k, seed=SEED):
    usable = frequency_df[(~frequency_df['is_dead']) & (~frequency_df['is_high_frequency'])]['latent_id'].astype(int).tolist()
    rng = np.random.default_rng(seed)
    if k > len(usable):
        k = len(usable)
    return rng.choice(usable, size=k, replace=False).astype(int).tolist()


def sample_matched_latents(frequency_df, target_latents, exclude=None):
    exclude = set(int(x) for x in (exclude or [])) | set(int(x) for x in target_latents)
    freq = frequency_df.set_index('latent_id')
    candidates = freq[(~freq['is_dead']) & (~freq['is_high_frequency'])].copy()
    candidates = candidates.drop(index=[i for i in exclude if i in candidates.index], errors='ignore')
    matched = []
    for latent_id in target_latents:
        if candidates.empty:
            break
        target_freq = float(freq.loc[int(latent_id), 'frequency'])
        target_mean = float(freq.loc[int(latent_id), 'mean_activation_when_active'])
        dist = (candidates['frequency'] - target_freq).abs() + 0.1 * (candidates['mean_activation_when_active'] - target_mean).abs()
        chosen = int(dist.idxmin())
        matched.append(chosen)
        candidates = candidates.drop(index=chosen)
    return matched


def run_intervention_eval(model, loader, dataset_spec, source_meta, sae, stats, clean_outputs, latent_ids, alpha, group_name, baseline_type):
    handle = register_intervention_hook(model, source_meta, sae, stats, latent_ids=latent_ids, alpha=alpha)
    try:
        intervened_outputs = collect_outputs_for_eval(model, loader, max_batches=INTERVENTION_MAX_BATCHES)
    finally:
        handle.remove()
    metrics = compare_intervention_outputs(clean_outputs, intervened_outputs, dataset_spec)
    return {
        'latent_group': group_name,
        'baseline_type': baseline_type,
        'num_latents': len(latent_ids),
        'alpha': alpha,
        **metrics,
    }


if RUN_INTERVENTION:
    selected_sae_path = SELECTED_SAE_RUN_DIR or selected_sae_run_dir
    if selected_sae_path is None:
        raise RuntimeError('Set SELECTED_SAE_RUN_DIR to a Phase 2 run directory.')
    sae, stats, source_meta, checkpoint = load_sae_run(selected_sae_path)
    latent_groups = latent_groups or load_json_if_exists(phase3_dir / 'latent_groups.json', {})
    latent_frequency_df = latent_frequency_df if not latent_frequency_df.empty else pd.read_csv(phase3_dir / 'latent_frequency.csv')

    if source_meta['source_type'] == 'vit':
        model_spec = make_vit_b16_spec()
    elif source_meta['source_type'] == 'resnet':
        model_spec = make_resnet50_spec()
    else:
        raise ValueError(source_meta['source_type'])

    model = model_spec.model.to(DEVICE).eval()
    clean_loader = build_loader(imagenet_200_spec, 'original', model_spec)
    clean_outputs = collect_outputs_for_eval(model, clean_loader, max_batches=INTERVENTION_MAX_BATCHES)

    rows = []
    rows.append(run_intervention_eval(
        model,
        clean_loader,
        imagenet_200_spec,
        source_meta,
        sae,
        stats,
        clean_outputs,
        latent_ids=[],
        alpha=1.0,
        group_name='reconstruction_only',
        baseline_type='reconstruction_only',
    ))

    for group_name, latent_ids in latent_groups.items():
        latent_ids = [int(x) for x in latent_ids]
        if not latent_ids:
            continue
        matched_ids = sample_matched_latents(latent_frequency_df, latent_ids)
        for alpha in INTERVENTION_ALPHAS:
            rows.append(run_intervention_eval(model, clean_loader, imagenet_200_spec, source_meta, sae, stats, clean_outputs, latent_ids, alpha, group_name, 'cue_latents'))
            if matched_ids:
                rows.append(run_intervention_eval(model, clean_loader, imagenet_200_spec, source_meta, sae, stats, clean_outputs, matched_ids, alpha, group_name, 'matched_magnitude'))
            for repeat in range(RANDOM_BASELINE_REPEATS):
                random_ids = sample_random_latents(latent_frequency_df, len(latent_ids), seed=SEED + repeat)
                rows.append(run_intervention_eval(model, clean_loader, imagenet_200_spec, source_meta, sae, stats, clean_outputs, random_ids, alpha, group_name, f'random_{repeat}'))

    intervention_df = pd.DataFrame(rows)
    intervention_df.to_csv(phase4_dir / 'intervention_results.csv', index=False)
    display(intervention_df)

    for metric in ['accuracy_drop', 'js_divergence', 'top1_flip_rate']:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for (group_name, baseline_type), sub in intervention_df.groupby(['latent_group', 'baseline_type']):
            if group_name == 'reconstruction_only':
                continue
            summary = sub.groupby('alpha', as_index=False)[metric].mean()
            ax.plot(summary['alpha'], summary[metric], marker='o', label=f'{group_name}/{baseline_type}')
        ax.set_xlabel('alpha')
        ax.set_ylabel(metric)
        ax.legend(fontsize=8)
        ax.set_title(f'Intervention curve: {metric}')
        fig.tight_layout()
        fig.savefig(phase4_dir / f'intervention_{metric}.png', dpi=160, bbox_inches='tight')
        plt.close(fig)

    model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
else:
    print('RUN_INTERVENTION=False, skipped latent intervention.')


# %% [markdown]
# ## Stop Conditions
# 
# - Phase 1 fail: activation/token response is unstable or dominated by one broken perturbation. Tune perturbation HP, layer/block, or subset first.
# - Phase 2 fail: SAE reconstruction is poor. Do not interpret latents.
# - Phase 3 fail: top activating images are incoherent or perturbation sensitivity is unstable. Do not form cue-specific latent groups.
# - Phase 4 fail: cue-specific intervention is indistinguishable from random/matched baselines, or reconstruction-only drift is large. Do not claim causality.
# 
