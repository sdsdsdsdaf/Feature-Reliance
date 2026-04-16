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
        ModelRunResult,
        PerturbationMetricResult,
        PerturbationValidationResult,
        ExperimentResult,
        DatasetSpec, 
        EvalScenario,
        ScenarioRecord
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
        ModelRunResult,
        PerturbationMetricResult,
        PerturbationValidationResult,
        ExperimentResult,
        DatasetSpec, 
        EvalScenario,
        ScenarioRecord
    )
    from transfrom import get_transform
    from Dataset import ImageNetValFlatDataset
    from metric import relative_accuracy
    from metric import compute_dataset_metrics
    from Utils.metric import linear_cka, js_divergence

# =========================================================
# Evaluation Protocol / Class Mapping
# =========================================================

IMAGENET_R_CLASS_IDS = [
    # TODO: fill with the official 1k-class indices used by ImageNet-R
]

CLASS_MAPPING_REGISTRY = {
    "imagenet_r_subset_map": {
        "subset_class_ids": IMAGENET_R_CLASS_IDS,
    },
}


def get_class_mapping(class_map_name: Optional[str]) -> Optional[Dict[str, Any]]:
    if class_map_name is None:
        return None
    if class_map_name not in CLASS_MAPPING_REGISTRY:
        raise ValueError(f"Unknown class_map_name: {class_map_name}")
    return CLASS_MAPPING_REGISTRY[class_map_name]

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

def load_saved_scenario_tensors(
    root_dir: str,
    model_name: str,
    pretrained_weight: str,
    dataset_spec: DatasetSpec,
    perturbation: str,
    scenario_config: Dict[str, Any],
):
    save_dir = make_save_dir(
        root_dir=root_dir,
        model_name=model_name,
        pretrained_weight=pretrained_weight,
        dataset_name=dataset_spec.name,
        split=dataset_spec.split,
        perturbation=perturbation,
        scenario_config=scenario_config,
    )
    reps, logits, labels = load_saved_tensors(save_dir)
    return save_dir, reps, logits, labels

def get_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def cal_accuracy(
    logits: Tensor,
    labels: Tensor,
    class_map_name: Optional[str] = None,
) -> float:
    mapping = get_class_mapping(class_map_name)

    if mapping is None:
        preds = torch.argmax(logits, dim=1)
        return (preds == labels).float().mean().item()

    # Example: subset evaluation such as ImageNet-R
    if "subset_class_ids" in mapping:
        subset_class_ids = mapping["subset_class_ids"]
        subset_idx = torch.tensor(subset_class_ids, device=logits.device, dtype=torch.long)

        logits_subset = logits.index_select(dim=1, index=subset_idx)
        preds_subset_local = torch.argmax(logits_subset, dim=1)
        preds_global = subset_idx[preds_subset_local]

        valid_mask = torch.isin(labels.to(logits.device), subset_idx)

        if valid_mask.sum().item() == 0:
            raise ValueError("No valid labels found for subset evaluation.")

        correct = (preds_global[valid_mask] == labels.to(logits.device)[valid_mask]).float().mean().item()
        return correct

    raise ValueError(f"Unsupported mapping format for {class_map_name}")

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


def build_scenario_config(
    dataset_spec: DatasetSpec,
    perturbation: str,
    hparams: TransformHyperParams,
) -> Dict[str, Any]:
    config = {
        "dataset_name": dataset_spec.name,
        "dataset_type": dataset_spec.dataset_type,
        "split": dataset_spec.split,
        "num_classes": dataset_spec.num_classes,
        "class_map_name": dataset_spec.class_map_name,
        "eval_protocol_name": dataset_spec.eval_protocol_name,
        "domain_type": dataset_spec.domain_type,
        "shift_type": dataset_spec.shift_type,
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

def build_dataset(dataset_spec: DatasetSpec, transform):
    if dataset_spec.dataset_type == "imagenet_val_flat":
        return (
            ImageNetValFlatDataset(dataset_spec.root, transform=transform)
            if dataset_spec.root is not None
            else ImageNetValFlatDataset(transform=transform)
        )

    # TODO: implement actual dataset classes
    elif dataset_spec.dataset_type == "imagenet_r":
        return NotImplementedError("ImageNet-R is Not Impelemented")

    elif dataset_spec.dataset_type == "stylized_imagenet":
        return NotImplementedError("ImageNet-C Not Implemeted   ")

    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_spec.dataset_type}")


def make_save_dir(
    root_dir: str,
    model_name: str,
    pretrained_weight: str,
    dataset_name: str,
    split: str,
    perturbation: str,
    scenario_config: Dict[str, Any],
) -> Path:
    config_hash = make_config_hash(scenario_config)
    save_dir = (
        Path(root_dir)
        / model_name
        / pretrained_weight
        / dataset_name
        / split
        / perturbation
        / config_hash
    )
    ensure_dir(save_dir)

    config_path = save_dir / "config.json"
    if not config_path.exists():
        save_json(scenario_config, config_path)

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


def get_scenario_name(dataset_name: str, perturbation: str) -> str:
    return f"{dataset_name}__{perturbation}"


def get_dataset_spec_by_name(data_config: DataConfig, dataset_name: str) -> DatasetSpec:
    for ds in data_config.datasets:
        if ds.name == dataset_name:
            return ds
    raise ValueError(f"Dataset '{dataset_name}' not found in data_config.datasets")


def build_default_scenarios(
    data_config: DataConfig,
    perturbations: List[str],
) -> List[EvalScenario]:
    scenarios = []
    for ds in data_config.datasets:
        for p in perturbations:
            scenarios.append(
                EvalScenario(
                    dataset_name=ds.name,
                    perturbation=p,
                    scenario_name=get_scenario_name(ds.name, p),
                )
            )
    return scenarios

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
    dataset_spec: DatasetSpec,
    perturbation: str,
    scenario_config: Dict[str, Any],
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
        dataset_name=dataset_spec.name,
        split=dataset_spec.split,
        perturbation=perturbation,
        scenario_config=scenario_config,
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

    for i, (image, label) in enumerate(
        tqdm(dataloader, desc=f"Extracting [{model_name}/{dataset_spec.name}/{perturbation}]")
    ):
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

def evaluate_saved_scenario(
    root_dir: str,
    model_name: str,
    pretrained_weight: str,
    dataset_spec: DatasetSpec,
    perturbation: str,
    scenario_config: Dict[str, Any],
    same_dataset_clean_config: Optional[Dict[str, Any]] = None,
    id_clean_dataset_spec: Optional[DatasetSpec] = None,
    id_clean_config: Optional[Dict[str, Any]] = None,
) -> ScenarioRecord:

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    save_dir, reps, logits, labels = load_saved_scenario_tensors(
        root_dir=root_dir,
        model_name=model_name,
        pretrained_weight=pretrained_weight,
        dataset_spec=dataset_spec,
        perturbation=perturbation,
        scenario_config=scenario_config,
    )

    reps, logits, labels = reps.to(device), logits.to(device), labels.to(device)
    acc = cal_accuracy(logits, labels, class_map_name=dataset_spec.class_map_name,)

    scenario_name = get_scenario_name(dataset_spec.name, perturbation)

    record = ScenarioRecord(
        scenario_name=scenario_name,
        dataset_name=dataset_spec.name,
        perturbation=perturbation,
        config_hash=make_config_hash(scenario_config),
        save_dir=str(save_dir),
        accuracy=acc,
    )

    # Compare against same-dataset clean
    if same_dataset_clean_config is not None and perturbation != "original":
        _, clean_reps, clean_logits, clean_labels = load_saved_scenario_tensors(
            root_dir=root_dir,
            model_name=model_name,
            pretrained_weight=pretrained_weight,
            dataset_spec=dataset_spec,
            perturbation="original",
            scenario_config=same_dataset_clean_config,
        )

        clean_reps, clean_logits, clean_labels = clean_reps.to(device), clean_logits.to(device), clean_labels.to(device)

        if not torch.equal(labels, clean_labels):
            raise ValueError(f"Label mismatch in same-dataset clean comparison: {scenario_name}")

        clean_acc = cal_accuracy(
            clean_logits,
            clean_labels,
            class_map_name=dataset_spec.class_map_name,
        )
        record.accuracy_drop_vs_same_dataset_clean = clean_acc - acc
        record.intervention_gain_vs_same_dataset_clean = acc - clean_acc
        record.relative_accuracy_score = relative_accuracy(
            acc, clean_acc, num_class=dataset_spec.num_classes
        )
        record.js_divergence = js_divergence(clean_logits.float(), logits.float()).item()
        record.cka = linear_cka(clean_reps.float(), reps.float()).item()

    # Compare against ID clean
    if id_clean_dataset_spec is not None and id_clean_config is not None:
        _, id_reps, id_logits, id_labels = load_saved_scenario_tensors(
            root_dir=root_dir,
            model_name=model_name,
            pretrained_weight=pretrained_weight,
            dataset_spec=id_clean_dataset_spec,
            perturbation="original",
            scenario_config=id_clean_config,
        )

        
        
        id_logits, id_labels = id_logits.to(device), id_labels.to(device)
        id_acc = cal_accuracy(
            id_logits,
            id_labels,
            class_map_name=id_clean_dataset_spec.class_map_name,
        )

        record.accuracy_drop_vs_id_clean = id_acc - acc
        record.ood_gap_vs_id_clean = acc - id_acc

    return record


# =========================================================
# Single Model Runner
# =========================================================

def run_single_model_experiment(
    model_spec: ModelSpec,
    transform_hparams: TransformHyperParams,
    data_config: DataConfig,
    extraction_config: ExtractionConfig,
    scenarios: List[EvalScenario],
    verbose_image: bool = False,
    plot_seconds: int = 10,
    run_dir: Optional[str] = None,
    id_dataset_name: str = "imagenet",
) -> ModelRunResult:

    result = ModelRunResult(
        model_name=model_spec.model_name,
        pretrained_weight=model_spec.pretrained_weight,
    )

    # Optional visualization only for the first ID dataset
    if verbose_image:
        first_dataset_spec = get_dataset_spec_by_name(data_config, scenarios[0].dataset_name)
        vis_perturbations = sorted(list({s.perturbation for s in scenarios if s.dataset_name == first_dataset_spec.name}))
        visualize_perturbations(
            dataset_root=first_dataset_spec.root,
            perturbations=vis_perturbations,
            hparams=transform_hparams,
            run_dir=run_dir,
            show_seconds=plot_seconds,
        )

    # 1. Extract for all scenarios
    for scenario in scenarios:
        dataset_spec = get_dataset_spec_by_name(data_config, scenario.dataset_name)

        transform = build_transform(
            perturbation=scenario.perturbation,
            mean=model_spec.mean,
            std=model_spec.std,
            hparams=transform_hparams,
            normalize=scenario.normalize,
        )

        ds = build_dataset(dataset_spec, transform=transform)

        dataloader = DataLoader(
            ds,
            batch_size=data_config.batch_size,
            shuffle=data_config.shuffle,
            num_workers=data_config.num_workers,
            pin_memory=data_config.pin_memory,
        )

        scenario_config = build_scenario_config(
            dataset_spec=dataset_spec,
            perturbation=scenario.perturbation,
            hparams=transform_hparams,
        )

        extract_logit_and_representation(
            model=model_spec.model,
            dataloader=dataloader,
            model_name=model_spec.model_name,
            pretrained_weight=model_spec.pretrained_weight,
            dataset_spec=dataset_spec,
            perturbation=scenario.perturbation,
            scenario_config=scenario_config,
            extraction_config=extraction_config,
        )

        del ds, dataloader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 2. Evaluate all scenarios
    id_dataset_spec = get_dataset_spec_by_name(data_config, id_dataset_name)
    id_clean_config = build_scenario_config(
        dataset_spec=id_dataset_spec,
        perturbation="original",
        hparams=transform_hparams,
    )

    for scenario in scenarios:
        dataset_spec = get_dataset_spec_by_name(data_config, scenario.dataset_name)

        scenario_config = build_scenario_config(
            dataset_spec=dataset_spec,
            perturbation=scenario.perturbation,
            hparams=transform_hparams,
        )

        same_dataset_clean_config = build_scenario_config(
            dataset_spec=dataset_spec,
            perturbation="original",
            hparams=transform_hparams,
        )

        record = evaluate_saved_scenario(
            root_dir=extraction_config.root_dir,
            model_name=model_spec.model_name,
            pretrained_weight=model_spec.pretrained_weight,
            dataset_spec=dataset_spec,
            perturbation=scenario.perturbation,
            scenario_config=scenario_config,
            same_dataset_clean_config=same_dataset_clean_config,
            id_clean_dataset_spec=id_dataset_spec,
            id_clean_config=id_clean_config,
        )

        scenario_name = scenario.scenario_name or get_scenario_name(
            scenario.dataset_name,
            scenario.perturbation,
        )
        result.scenario_results[scenario_name] = record

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
    scenarios: Optional[List[EvalScenario]] = None,
    perturbations: Optional[List[str]] = None,
    verbose_image: bool = False,
    plot_seconds: int = 10,
    save_summary: bool = True,
    summary_name: str = "experiment_summary.json",
    run_validation: bool = True,
    validation_max_samples: Optional[int] = None,
    id_dataset_name: str = "imagenet",
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

    if scenarios is None:
        scenarios = build_default_scenarios(data_config, perturbations)

    output = ExperimentResult(
        transform_hparams=transform_hparams,
        data_config=data_config,
        extraction_config=extraction_config,
        scenarios=scenarios,
    )

    root = Path(extraction_config.root_dir) / "MetaData"
    trial_id = get_next_trial_id(root)
    run_dir = root / f"trial_{trial_id:04d}"

    # Perturbation validation: run only on ID dataset assumption
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
            scenarios=scenarios,
            verbose_image=verbose_image,
            plot_seconds=plot_seconds,
            run_dir=run_dir,
            id_dataset_name=id_dataset_name,
        )

        output.model_results[f"{model_spec.model_name}__{model_spec.pretrained_weight}"] = model_result
        
        print()
        print(
            f"{'scenario':^24s} | "
            f"{'Accuracy':^13s} | "
            f"{'Rel Accuracy':^13s} | "
            f"{'js':^13s} | "
            f"{'cka':^13s} | "
            f"{'drop_same':^13s} | "
            f"{'drop_id':^13s}"
        )
        print("-" * 120)
        for name, record in model_result.scenario_results.items():
            acc = f"{record.accuracy:.6f}" if record.accuracy is not None else "None"
            rel = f"{record.relative_accuracy_score:.6f}" if record.relative_accuracy_score is not None else "None"
            js = f"{record.js_divergence:.6f}" if record.js_divergence is not None else "None"
            cka = f"{record.cka:.6f}" if record.cka is not None else "None"
            drop_same = (
                f"{record.accuracy_drop_vs_same_dataset_clean:.6f}"
                if record.accuracy_drop_vs_same_dataset_clean is not None else "None"
            )
            drop_id = (
                f"{record.accuracy_drop_vs_id_clean:.6f}"
                if record.accuracy_drop_vs_id_clean is not None else "None"
            )

            print(
                f"{name:24s} | "
                f"{acc:>13s} | "
                f"{rel:>13s} | "
                f"{js:>13s} | "
                f"{cka:>13s} | "
                f"{drop_same:>13s} | "
                f"{drop_id:>13s}"
            )

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
