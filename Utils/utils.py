import hashlib
import json
from typing import List, Tuple, Union, Optional

import cv2
import numpy as np

import albumentations as A
import torch.nn as nn
from torchvision import transforms
import os, platform, sys, psutil, torch
from torch import Tensor
from pathlib import Path
from typing import Any, Dict
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from pprint import pprint



try:
    from Utils.Config import (
        TransformHyperParams,
        DataConfig,
        ExtractionConfig,
        ExtractedTensors,
        ModelSpec,
        PerturbationRecord,
        ModelRunResult,
        PerturbationMetricResult,
        PerturbationValidationResult,
        ExperimentResult,
    )
    from Utils.transfrom import get_transform
    from Utils.Dataset import ImageNetValFlatDataset
    from Utils.metric import relative_accuracy
    from Utils.metric import compute_dataset_metrics
    from Utils.metric import linear_cka, js_divergence
except:
    from Utils.Config import (
        TransformHyperParams,
        DataConfig,
        ExtractionConfig,
        ExtractedTensors,
        ModelSpec,
        PerturbationRecord,
        ModelRunResult,
        PerturbationMetricResult,
        PerturbationValidationResult,
        ExperimentResult,
    )
    from transfrom import get_transform
    from Dataset import ImageNetValFlatDataset
    from metric import relative_accuracy
    from metric import compute_dataset_metrics
    from Utils.metric import linear_cka, js_divergence

# TODO Hash 기반으로 리팩토링

def get_system_info():
    print("===== System Info =====")
    
    # OS
    print(f"OS: {platform.system()} {platform.release()}")
    
    # Python
    print(f"Python Version: {sys.version.split()[0]}")
    
    # CPU
    print(f"CPU: {platform.processor()}")
    
    # RAM
    ram = psutil.virtual_memory().total / (1024 ** 3)
    print(f"RAM: {ram:.2f} GB")
    
    print("\n===== PyTorch Info =====")
    
    # PyTorch
    print(f"PyTorch Version: {torch.__version__}")
    
    # CUDA
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Version (PyTorch): {torch.version.cuda}")
    
    # GPU + VRAM
    if torch.cuda.is_available():
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram = props.total_memory / (1024 ** 3)
            print(f"GPU {i}: {props.name}, {vram:.2f} GB VRAM")
    else:
        print("GPU: Not available")

def load_imagenet(normalize=True):
    pass

def load_saved_tensors(save_dir: str | Path) -> tuple[Tensor, Tensor, Tensor]:
    save_dir = Path(save_dir)
    reps = torch.load(save_dir / "representations.pt", weights_only=True)
    logits = torch.load(save_dir / "logits.pt", weights_only=True)
    labels = torch.load(save_dir / "labels.pt", weights_only=True)
    return reps, logits, labels

def get_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def cal_accuracy(logits: Tensor, labels: Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()

def maybe_save_tensors(
    save_dir: Path,
    extracted: ExtractedTensors,
) -> None:
    ensure_dir(save_dir)
    torch.save(extracted.representations, save_dir / "representations.pt")
    torch.save(extracted.logits, save_dir / "logits.pt")
    torch.save(extracted.labels, save_dir / "labels.pt")
    
    del extracted.representations, extracted.logits, extracted.labels
    print(f"[Saved] {save_dir}")
    
def get_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return mapping[dtype_str]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def cal_accuracy(logit: Tensor, label: Tensor) -> float:
    pred = torch.argmax(logit, dim=1)
    return (pred == label).float().mean().item()


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================================================
# Hash / Config
# =========================================================

def make_config_hash(config: Dict[str, Any], hash_len: int = 12) -> str:
    canonical_str = json.dumps(
        config,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )   
    
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()[:hash_len]


def build_perturbation_config(
    perturbation: str,
    hparams: TransformHyperParams,
) -> Dict[str, Any]:
    config = {
        "perturbation": perturbation,
        "prefix": hparams.prefix,
        "resize_size": hparams.resize_size,
        "p": hparams.p,
    }

    if perturbation == "original":
        return config

    if perturbation == "grayscale":
        config.update({
            "gray_alpha": hparams.gray_alpha,
        })
    elif perturbation == "bilateral":
        config.update({
            "bilateral_d": hparams.bilateral_d,
            "sigma_color": hparams.sigma_color,
            "sigma_space": hparams.sigma_space,
        })
    elif perturbation == "gaussianblur":
        config.update({
            "gaussian_k": hparams.gaussian_k,
            "gaussian_sigma": hparams.gaussian_sigma,
        })
    elif perturbation == "patchshuffle":
        config.update({
            "grid_size": hparams.grid_size,
        })
    elif perturbation == "patchrotation":
        config.update({
            "grid_size": hparams.grid_size,
        })
    elif perturbation == "localwarp":
        config.update({
            "alpha_localwarp": hparams.alpha_localwarp,
            "sigma_localwarp": hparams.sigma_localwarp,
        })
    else:
        raise ValueError(f"Unsupported perturbation: {perturbation}")

    return config


def make_save_dir(
    root_dir: str,
    model_name: str,
    pretrained_weight: str,
    perturbation: str,
    perturbation_config: Dict[str, Any],
) -> Path:
    config_hash = make_config_hash(perturbation_config)
    save_dir = Path(root_dir) / model_name / pretrained_weight / perturbation / config_hash
    ensure_dir(save_dir)

    config_path = save_dir / "config.json"
    if not config_path.exists():
        save_json(perturbation_config, config_path)

    return save_dir


# =========================================================
# Transform Builder
# =========================================================

def build_transform(
    perturbation: str,
    mean: List[float],
    std: List[float],
    hparams: TransformHyperParams,
    normalize: bool = True,
):
    prefix = hparams.prefix

    if perturbation == "original":
        if normalize:
            return get_transform(
                test_augmentations=prefix,
                resize_size=hparams.resize_size,
                mean=mean,
                std=std,
                split="test",
                normalize=True,
            )
        else:
            return get_transform(
                train_augmentations=prefix,
                resize_size=hparams.resize_size,
                split="train",
                normalize=False,
            )

    common_kwargs = dict(
        p=hparams.p,
        resize_size=hparams.resize_size,
        split="train",
        normalize=normalize,
    )

    if normalize:
        common_kwargs["mean"] = mean
        common_kwargs["std"] = std

    if perturbation == "grayscale":
        return get_transform(
            train_augmentations=f"{prefix}_grayscale",
            gray_alpha=hparams.gray_alpha,
            **common_kwargs,
        )

    if perturbation == "bilateral":
        return get_transform(
            train_augmentations=f"{prefix}_bilateral",
            bilateral_d=hparams.bilateral_d,
            sigma_color=hparams.sigma_color,
            sigma_space=hparams.sigma_space,
            **common_kwargs,
        )

    if perturbation == "gaussianblur":
        return get_transform(
            train_augmentations=f"{prefix}_gaussianblur",
            gaussian_k=hparams.gaussian_k,
            gaussian_sigma=hparams.gaussian_sigma,
            **common_kwargs,
        )

    if perturbation == "patchshuffle":
        return get_transform(
            train_augmentations=f"{prefix}_patchshuffle",
            grid_size=hparams.grid_size,
            **common_kwargs,
        )

    if perturbation == "patchrotation":
        return get_transform(
            train_augmentations=f"{prefix}_patchrotation",
            grid_size=hparams.grid_size,
            **common_kwargs,
        )

    if perturbation == "localwarp":
        return get_transform(
            train_augmentations=f"{prefix}_localwarp",
            alpha_localwarp=hparams.alpha_localwarp,
            sigma_localwarp=hparams.sigma_localwarp,
            **common_kwargs,
        )

    raise ValueError(f"Unsupported perturbation: {perturbation}")


def build_transform_dict(
    mean: List[float],
    std: List[float],
    hparams: TransformHyperParams,
    perturbations: List[str],
    normalize: bool = True,
) -> Dict[str, Any]:
    return {
        p: build_transform(
            perturbation=p,
            mean=mean,
            std=std,
            hparams=hparams,
            normalize=normalize,
        )
        for p in perturbations
    }


# =========================================================
# Visualization
# =========================================================

def visualize_perturbations(
    dataset_root: Optional[str],
    perturbations: List[str],
    hparams: TransformHyperParams,
    sample_index: int = 3,
    run_dir: Optional[str] = None,
    show_seconds: int = 10,
) -> None:
    ds = (
        ImageNetValFlatDataset(dataset_root, transform=None)
        if dataset_root is not None
        else ImageNetValFlatDataset(transform=None)
    )
    image, _ = ds[sample_index]

    vis_transforms = build_transform_dict(
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        hparams=hparams,
        perturbations=perturbations,
        normalize=False,
    )

    cols = 4
    rows = int(np.ceil(len(perturbations) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, perturbation in zip(axes, perturbations):
        t = vis_transforms[perturbation]
        img_t = t(image.copy())

        if isinstance(img_t, torch.Tensor):
            img_t = img_t.detach().cpu()
            if img_t.ndim == 3 and img_t.shape[0] in (1, 3):
                img_t = img_t.permute(1, 2, 0).numpy()
            else:
                img_t = img_t.numpy()

        ax.imshow(np.clip(img_t, 0, 255).astype(np.uint8) if img_t.dtype != np.uint8 else img_t)
        ax.set_title(perturbation)
        ax.axis("off")
        ax.set_aspect("equal")

    for ax in axes[len(perturbations):]:
        ax.axis("off")

    plt.tight_layout()
    
    # Save
    if run_dir is not None:
        ensure_dir(run_dir)
        save_path = Path(run_dir) / "visualization.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[Saved] {save_path}")
    
    plt.show(block=False)
    plt.pause(show_seconds)
    plt.close(fig)


# =========================================================
# Extraction
# =========================================================

@torch.no_grad()
def extract_logit_and_representation(
    model: nn.Module,
    dataloader: DataLoader,
    model_name: str,
    pretrained_weight: str,
    perturbation: str,
    perturbation_config: Dict[str, Any],
    extraction_config: ExtractionConfig,
) -> None:
    
    device = get_device(extraction_config.device)
    dtype = get_torch_dtype(extraction_config.dtype)

    if dataloader is None:
        raise ValueError("dataloader must not be None")

    save_dir = make_save_dir(
        root_dir=extraction_config.root_dir,
        model_name=model_name,
        pretrained_weight=pretrained_weight,
        perturbation=perturbation,
        perturbation_config=perturbation_config,
    )

    logits_path = save_dir / "logits.pt"
    reps_path = save_dir / "representations.pt"
    labels_path = save_dir / "labels.pt"

    if (
        not extraction_config.overwrite
        and logits_path.exists()
        and reps_path.exists()
        and labels_path.exists()
    ):
        print(f"[Skip] Already exists: {save_dir}")
        return None

    model = model.to(device)
    model.eval()

    representations = []
    logits = []
    labels = []

    for i, (image, label) in enumerate(tqdm(dataloader, desc=f"Extracting [{model_name}/{perturbation}]")):
        image = image.to(device, non_blocking=True)

        feat = model.forward_features(image)
        rep: Tensor = model.forward_head(feat, pre_logits=True)
        logit: Tensor = model.forward_head(feat, pre_logits=False)

        if extraction_config.debug_first_batch and i == 0:
            print("rep shape  :", rep.shape)
            print("logit shape:", logit.shape)
            print("rep dtype  :", rep.dtype)
            print("logit dtype:", logit.dtype)
            print("rep mean/std  :", rep.mean().item(), rep.std().item())
            print("logit mean/std:", logit.mean().item(), logit.std().item())

            if rep.shape == logit.shape:
                print("allclose?:", torch.allclose(rep, logit, atol=1e-6))
            else:
                print("allclose?: skipped (different shapes)")

        representations.append(rep.detach().float().cpu())
        logits.append(logit.detach().to(dtype).cpu())
        labels.append(label.detach().cpu())

    representations = torch.cat(representations, dim=0).float()
    logits = torch.cat(logits, dim=0).to(dtype)
    labels = torch.cat(labels, dim=0)

    torch.save(representations, reps_path)
    torch.save(logits, logits_path)
    torch.save(labels, labels_path)

    print(f"[Saved] {save_dir}")
    
    del representations, logits, labels
    
    return None


# =========================================================
# Metric Evaluation from Saved Files
# =========================================================

def evaluate_saved_perturbation(
    root_dir: str,
    model_name: str,
    pretrained_weight: str,
    perturbation: str,
    perturbation_config: Dict[str, Any],
    original_config: Dict[str, Any],
    original_accuracy: Optional[float] = None,
    num_class: int = 1000,
) -> PerturbationRecord:

    save_dir = make_save_dir(
        root_dir=root_dir,
        model_name=model_name,
        pretrained_weight=pretrained_weight,
        perturbation=perturbation,
        perturbation_config=perturbation_config,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    
    reps, logits, labels = load_saved_tensors(save_dir)
    acc = cal_accuracy(logits, labels)

    rel_acc = None
    js_div = None
    cka_score = None

    if perturbation != "original":
        if original_accuracy is None:
            raise ValueError("original_accuracy is required for non-original perturbations")

        original_save_dir = make_save_dir(
            root_dir=root_dir,
            model_name=model_name,
            pretrained_weight=pretrained_weight,
            perturbation="original",
            perturbation_config=original_config,
        )

        orig_reps, orig_logits, orig_labels = load_saved_tensors(original_save_dir)
        orig_reps, orig_logits, orig_labels = orig_reps.to(device), orig_logits.to(device), orig_labels
        reps, logits, labels = reps.to(device), logits.to(device), labels

        if not torch.equal(labels, orig_labels):
            raise ValueError("Label mismatch between original and perturbation runs.")

        rel_acc = relative_accuracy(acc, original_accuracy, num_class=num_class)
        js_div = js_divergence(orig_logits.float(), logits.float()).item()
        cka_score = linear_cka(orig_reps.float(), reps.float()).item()

    return PerturbationRecord(
        perturbation=perturbation,
        config_hash=make_config_hash(perturbation_config),
        save_dir=str(save_dir),
        accuracy=acc,
        relative_accuracy_score=rel_acc,
        js_divergence=js_div,
        cka=cka_score,
    )


# =========================================================
# Single Model Runner
# =========================================================

def run_single_model_experiment(
    model_spec: ModelSpec,
    transform_hparams: TransformHyperParams,
    data_config: DataConfig,
    extraction_config: ExtractionConfig,
    perturbations: List[str],
    verbose_image: bool = False,
    plot_seconds: int = 10,
    run_dir: Optional[str] = None,
) -> ModelRunResult:
    
    if "original" not in perturbations:
        perturbations = ["original"] + perturbations

    if verbose_image:
        visualize_perturbations(
            dataset_root=data_config.dataset_root,
            perturbations=perturbations,
            hparams=transform_hparams,
            run_dir=run_dir,
            show_seconds=plot_seconds,
        )

    transforms = build_transform_dict(
        mean=model_spec.mean,
        std=model_spec.std,
        hparams=transform_hparams,
        perturbations=perturbations,
        normalize=True,
    )

    result = ModelRunResult(
        model_name=model_spec.model_name,
        pretrained_weight=model_spec.pretrained_weight,
    )

    # -----------------------------------------------------
    # 1. Extract and save tensors for all perturbations
    # -----------------------------------------------------
    for perturbation in perturbations:
        ds = (
            ImageNetValFlatDataset(data_config.dataset_root, transform=transforms[perturbation])
            if data_config.dataset_root is not None
            else ImageNetValFlatDataset(transform=transforms[perturbation])
        )

        dataloader = DataLoader(
            ds,
            batch_size=data_config.batch_size,
            shuffle=data_config.shuffle,
            num_workers=data_config.num_workers,
            pin_memory=data_config.pin_memory,
        )

        perturbation_config = build_perturbation_config(perturbation, transform_hparams)

        extract_logit_and_representation(
            model=model_spec.model,
            dataloader=dataloader,
            model_name=model_spec.model_name,
            pretrained_weight=model_spec.pretrained_weight,
            perturbation=perturbation,
            perturbation_config=perturbation_config,
            extraction_config=extraction_config,
        )

        del ds, dataloader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----------------------------------------------------
    # 2. Evaluate original first
    # -----------------------------------------------------
    original_config = build_perturbation_config("original", transform_hparams)

    original_record = evaluate_saved_perturbation(
        root_dir=extraction_config.root_dir,
        model_name=model_spec.model_name,
        pretrained_weight=model_spec.pretrained_weight,
        perturbation="original",
        perturbation_config=original_config,
        original_config=original_config,
        original_accuracy=None,
    )

    result.original_accuracy = original_record.accuracy
    result.perturbations["original"] = original_record

    # -----------------------------------------------------
    # 3. Evaluate perturbed runs against original
    # -----------------------------------------------------
    for perturbation in perturbations:
        if perturbation == "original":
            continue

        perturbation_config = build_perturbation_config(perturbation, transform_hparams)

        record = evaluate_saved_perturbation(
            root_dir=extraction_config.root_dir,
            model_name=model_spec.model_name,
            pretrained_weight=model_spec.pretrained_weight,
            perturbation=perturbation,
            perturbation_config=perturbation_config,
            original_config=original_config,
            original_accuracy=result.original_accuracy,
        )

        result.perturbations[perturbation] = record

    return result



# =========================================================
# Multi Model Runner
# =========================================================

def get_next_trial_id(root):
    existing = sorted([p.name for p in root.glob("trial_*")])
    if not existing:
        return 0
    last = max(int(name.split("_")[1]) for name in existing)
    return last + 1

def run_experiments(
    model_specs: List[ModelSpec],
    transform_hparams: TransformHyperParams,
    data_config: DataConfig,
    extraction_config: ExtractionConfig,
    perturbations: Optional[List[str]] = None,
    verbose_image: bool = False,
    plot_seconds: int = 10,
    save_summary: bool = True,
    summary_name: str = "experiment_summary.json",
    run_validation: bool = True,
    validation_max_samples: Optional[int] = None,
) -> ExperimentResult:
    
    if perturbations is None:
        perturbations = [
            "original",
            "grayscale",
            "bilateral",
            "gaussianblur",
            "patchshuffle",
            "patchrotation",
            "localwarp",
        ]

    output = ExperimentResult(
        transform_hparams=transform_hparams,
        data_config=data_config,
        extraction_config=extraction_config,
    )
    
    root = Path(extraction_config.root_dir) / "MetaData"
    trial_id = get_next_trial_id(root)

    run_dir = root / f"trial_{trial_id:04d}"
    
    # =====================================================
    # 1) perturbation validation 먼저 수행
    # =====================================================
    if run_validation:
        print(f"\n{'='*20} Perturbation Validation {'='*20}")
        output.perturbation_validation = run_perturbation_validation(
            transform_hparams=transform_hparams,
            perturbations=perturbations,
            max_samples=validation_max_samples,
        )
        
    for model_spec in model_specs:
        print(f"\n{'='*20} {model_spec.model_name} / {model_spec.pretrained_weight} {'='*20}")

        model_result = run_single_model_experiment(
            model_spec=model_spec,
            transform_hparams=transform_hparams,
            data_config=data_config,
            extraction_config=extraction_config,
            perturbations=perturbations,
            verbose_image=verbose_image,
            plot_seconds = plot_seconds,
            run_dir=run_dir
        )

        output.model_results[f"{model_spec.model_name}__{model_spec.pretrained_weight}"] = model_result

        print()
        print(f"Original Acc: {model_result.original_accuracy:.6f}")

        for name, record in model_result.perturbations.items():
            if name == "original":
                continue

            acc = f"{record.accuracy:.6f}" if record.accuracy is not None else "None"
            rel = f"{record.relative_accuracy_score:.6f}" if record.relative_accuracy_score is not None else "None"
            js  = f"{record.js_divergence:.6f}" if record.js_divergence is not None else "None"
            cka = f"{record.cka:.6f}" if record.cka is not None else "None"

            print(
                f"{name:14s} | "
                f"acc={acc} | "
                f"rel_acc={rel} | "
                f"js={js} | "
                f"cka={cka} | "
                f"hash={record.config_hash}"
            )
            
    # TODO Trial{id}\Experiment구조로 리팩토링
    if save_summary:
        summary_path = Path(run_dir) / summary_name
        save_json(output.to_jsonable(), summary_path)
        print(f"\n[Saved Summary] {summary_path}")

    return output

def run_perturbation_validation(
    transform_hparams: TransformHyperParams,
    perturbations: List[str],
    max_samples: Optional[int] = None,
    verbose_image: bool = False,
) -> PerturbationValidationResult:
    
    base_ds = ImageNetValFlatDataset()
    base_transform = get_transform(
        test_augmentations=transform_hparams.prefix,
        resize_size=transform_hparams.resize_size,
        split="test",
        normalize=False,
    )
    
    transform_name_feature_supp_map = {
        "original": None,
        "grayscale": "color",
        "bilateral": "texture",
        "gaussianblur": "texture",
        "patchshuffle": "shape",
        "patchrotation": "shape",
        "localwarp": "shape",
    }

    result = PerturbationValidationResult()

    if verbose_image:
        visualize_perturbations(
            dataset_root=None,
            perturbations=perturbations,
            hparams=transform_hparams,
        )

    for perturbation in perturbations:
        if perturbation == "original":
            continue
        if perturbation not in list(transform_name_feature_supp_map.keys()):
             raise ValueError(
                f"Unknown perturbation: '{perturbation}'. "
                f"Available: {list(transform_name_feature_supp_map.keys())}"
            )
        
        desc = f"Measuring [{perturbation}/{transform_name_feature_supp_map[perturbation]}]" 
        perturbation_config = build_perturbation_config(perturbation, transform_hparams)
        config_hash = make_config_hash(perturbation_config)

        pert_transform = build_transform(
            perturbation=perturbation,
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            hparams=transform_hparams,
            normalize=False,
        )

        metrics = compute_dataset_metrics(
            dataset=base_ds,
            base_transform=base_transform,
            transform=pert_transform,
            max_samples=max_samples,
            desc=desc
        )

        result.results[perturbation] = PerturbationMetricResult(
            perturbation=perturbation,
            config_hash=config_hash,
            metrics=metrics,
        )
        
        def pretty_print_validation(results: dict):
            print("\n" + "="*60)
            print(" Perturbation Validation Summary ".center(60))
            print("="*60)

            header = (
                f"{'Perturb':<14} | {'Feature':<8} | "
                f"{'LV':>6} {'HFE':>6} {'ESSIM':>6} {'GC':>6} | "
                f"{'Texture':>7} {'Shape':>7}"
            )
            print(header)
            print("-"*60)

            for name, record in results.items():
                m = record.metrics

                feature = transform_name_feature_supp_map.get(name, "N/A")

                print(
                    f"{name:<14} | {feature:<8} | "
                    f"{m['LV_ratio']:6.3f} {m['HFE_ratio']:6.3f} {m['ESSIM']:6.3f} {m['GC']:6.3f} | "
                    f"{m['texture_score']:7.3f} {m['shape_score']:7.3f}"
                )

            print("="*60 + "\n")
    
    pretty_print_validation(result.results)
    return result