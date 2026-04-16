from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Dict, List, Optional, Any
from torch import Tensor
import torch.nn as nn   

@dataclass
class TransformHyperParams:
    resize_size: int = 224
    p: float = 1.0
    prefix: str = "resizecrop"

    # Texture
    bilateral_d: int = 11
    sigma_color: float = 170.0
    sigma_space: float = 75.0

    gaussian_k: int = 11
    gaussian_sigma: float = 2.0
    nlmeans_h: int = 20
    template_window_size: int = 11
    search_window_size: int = 11

    # Color
    gray_alpha: float = 1.0

    # Shape
    grid_size: int = 6
    alpha_localwarp: float = 35.0
    sigma_localwarp: float = 2.5


@dataclass
class DataConfig:
    batch_size: int = 512
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = False
    dataset_root: Optional[str] = None


@dataclass
class ExtractionConfig:
    root_dir: str = "Cache"
    device: str = "cuda"
    dtype: str = "float16"
    overwrite: bool = False
    debug_first_batch: bool = False


@dataclass
class ModelSpec:
    model_name: str
    pretrained_weight: str
    model: nn.Module
    mean: List[float]
    std: List[float]


@dataclass
class PerturbationRecord:
    perturbation: str
    config_hash: str
    save_dir: str
    accuracy: Optional[float] = None
    relative_accuracy_score: Optional[float] = None

    js_divergence: Optional[float] = None
    cka: Optional[float] = None

@dataclass
class ExtractedTensors:
    representations: Tensor
    logits: Tensor
    labels: Tensor

@dataclass
class ModelRunResult:
    model_name: str
    pretrained_weight: str
    original_accuracy: Optional[float] = None
    perturbations: Dict[str, PerturbationRecord] = field(default_factory=dict)


@dataclass
class PerturbationMetricResult:
    perturbation: str
    config_hash: str
    metrics: Dict[str, float]


@dataclass
class PerturbationValidationResult:
    results: Dict[str, PerturbationMetricResult] = field(default_factory=dict)

@dataclass
class ExperimentResult:
    transform_hparams: TransformHyperParams
    data_config: DataConfig
    extraction_config: ExtractionConfig
    perturbation_validation: Optional[PerturbationValidationResult] = None
    model_results: Dict[str, ModelRunResult] = field(default_factory=dict)

    def to_jsonable(self) -> Dict[str, Any]:
        import numpy as np
        import torch

        def safe_cast(obj):
            # Dataclass -> dict
            if is_dataclass(obj):
                return {k: safe_cast(v) for k, v in asdict(obj).items()}

            # Dict
            if isinstance(obj, dict):
                return {k: safe_cast(v) for k, v in obj.items()}

            # List / Tuple
            if isinstance(obj, (list, tuple)):
                return [safe_cast(v) for v in obj]

            # Torch tensor
            if isinstance(obj, torch.Tensor):
                if obj.numel() == 1:
                    return obj.item()
                return obj.detach().cpu().tolist()

            # Numpy array
            if isinstance(obj, np.ndarray):
                return obj.tolist()

            # Numpy scalar
            if isinstance(obj, np.generic):
                return obj.item()

            # Path 같은 것도 혹시 있으면 문자열화
            try:
                from pathlib import Path
                if isinstance(obj, Path):
                    return str(obj)
            except Exception:
                pass

            return obj

        return {
            "transform_hparams": safe_cast(self.transform_hparams),
            "data_config": safe_cast(self.data_config),
            "extraction_config": safe_cast(self.extraction_config),
            "model_results": safe_cast(self.model_results),
            "perturbation_validation": (
                safe_cast(self.perturbation_validation.results)
                if self.perturbation_validation is not None
                else None
            ),
        }
        
        
