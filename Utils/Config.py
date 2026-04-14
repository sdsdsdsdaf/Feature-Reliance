from dataclasses import dataclass, field, asdict
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
        return {
            "transform_hparams": asdict(self.transform_hparams),
            "data_config": asdict(self.data_config),
            "extraction_config": asdict(self.extraction_config),
            "model_results": {
                k: {
                    "model_name": v.model_name,
                    "pretrained_weight": v.pretrained_weight,
                    "original_accuracy": v.original_accuracy,
                    "perturbations": {
                        pk: asdict(pv) for pk, pv in v.perturbations.items()
                    },
                }
                for k, v in self.model_results.items()
            },
            "perturbation_validation": (
                {
                    k: asdict(v) if hasattr(v, "__dataclass_fields__") else v
                    for k, v in self.perturbation_validation.results.items()
                }
                if self.perturbation_validation is not None
                else None
            ),
                
        }
        
        
