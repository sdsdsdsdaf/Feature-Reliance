from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Dict, List, Optional, Any, Literal
from torch import Tensor
import torch.nn as nn
  
    
@dataclass
class TransformHyperParams:
    p: float = 1.0
    prefix: str = "resizecrop"
    resize_size: int = 256

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
class DatasetSpec:
    name: str
    dataset_type: str                    # e.g. "imagenet_val_flat", "imagenet_r"
    root: Optional[str] = None
    split: str = "val"
    num_classes: int = 1000
    sample_indices: List[int] = field(default_factory=list)
    labels_map: List[int] = field(default_factory=list)
    id_dataset_name: str = None

    # Metadata for analysis
    domain_type: str = "id"             # "id", "natural_ood", "synthetic_ood"
    shift_type: Optional[str] = None    # "texture", "style", "shape", "mixed"
    class_map_name: Optional[str] = None
    eval_protocol_name: Optional[str] = None



@dataclass
class DataConfig:
    batch_size: int = 512
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = False
    datasets: List[DatasetSpec] = field(default_factory=list)


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
    resize_size: int


@dataclass
class EvalScenario:
    dataset_name: str
    perturbation: str = "original"
    scenario_name: Optional[str] = None
    normalize: bool = True



@dataclass
class ScenarioRecord:
    scenario_name: str
    dataset_name: str
    perturbation: str
    config_hash: str
    save_dir: str

    accuracy: Optional[float] = None
    relative_accuracy_score: Optional[float] = None
    js_divergence: Optional[float] = None
    cka: Optional[float] = None

    # OOD-aware
    accuracy_drop_vs_same_dataset_clean: Optional[float] = None
    accuracy_drop_vs_id_clean: Optional[float] = None
    ood_gap_vs_id_clean: Optional[float] = None
    intervention_gain_vs_same_dataset_clean: Optional[float] = None


@dataclass
class ExtractedTensors:
    representations: Tensor
    logits: Tensor
    labels: Tensor



@dataclass
class ModelRunResult:
    model_name: str
    pretrained_weight: str
    scenario_results: Dict[str, ScenarioRecord] = field(default_factory=dict)


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
    scenarios: List[EvalScenario] = field(default_factory=list)
    perturbation_validation: Optional[PerturbationValidationResult] = None
    model_results: Dict[str, ModelRunResult] = field(default_factory=dict)

    def to_jsonable(self) -> Dict[str, Any]:
        import numpy as np
        import torch

        def safe_cast(obj):
            if is_dataclass(obj):
                return {k: safe_cast(v) for k, v in asdict(obj).items()}

            if isinstance(obj, dict):
                return {k: safe_cast(v) for k, v in obj.items()}

            if isinstance(obj, (list, tuple)):
                return [safe_cast(v) for v in obj]

            if isinstance(obj, torch.Tensor):
                if obj.numel() == 1:
                    return obj.item()
                return obj.detach().cpu().tolist()

            if isinstance(obj, np.ndarray):
                return obj.tolist()

            if isinstance(obj, np.generic):
                return obj.item()

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
            "scenarios": safe_cast(self.scenarios),
            "model_results": safe_cast(self.model_results),
            "perturbation_validation": (
                safe_cast(self.perturbation_validation.results)
                if self.perturbation_validation is not None
                else None
            ),
        }
        
    @classmethod
    def from_jsonable(cls, d: Dict[str, Any]) -> "ExperimentResult":
        return cls(
            transform_hparams=TransformHyperParams(**d["transform_hparams"]),
            data_config=DataConfig(
                batch_size=d["data_config"].get("batch_size", 512),
                num_workers=d["data_config"].get("num_workers", 4),
                pin_memory=d["data_config"].get("pin_memory", True),
                shuffle=d["data_config"].get("shuffle", False),
                datasets=[
                    DatasetSpec(**x)
                    for x in d["data_config"].get("datasets", [])
                ],
            ),
            extraction_config=ExtractionConfig(**d["extraction_config"]),
            scenarios=[
                EvalScenario(**x)
                for x in d.get("scenarios", [])
            ],
            perturbation_validation=(
                PerturbationValidationResult(
                    results={
                        k: PerturbationMetricResult(**v)
                        for k, v in d["perturbation_validation"].items()
                    }
                )
                if d.get("perturbation_validation") is not None
                else None
            ),
            model_results={
                k: ModelRunResult(
                    model_name=v["model_name"],
                    pretrained_weight=v["pretrained_weight"],
                    scenario_results={
                        sk: ScenarioRecord(**sv)
                        for sk, sv in v.get("scenario_results", {}).items()
                    },
                )
                for k, v in d.get("model_results", {}).items()
            },
        )
            


# ------ Train Config ------
@dataclass
class LossConfig:
    mode: str = "feature"  # "kl", "feature", "both", "none"
    feature_loss_type: str = "consine" # "cosine", "mse", "mse_sum"
    lambda_kl: float = 1.0
    lambda_feat: float = 1.0
    lambda_scale: float = 0.0
    lambda_delta: float = 0.0
    temperature: float = 1.0
    detach_teacher: bool = True
    normalize_feature: bool = True
    ce_clean_weight: float = 1.0
    ce_pert_weight: float = 1.0
    eps: float = 1e-6
    
@dataclass
class OptimConfig:
    epochs: int = 10
    lr: float = 1e-4
    adaptor_lr: Optional[float] = None
    head_lr: Optional[float] = None
    weight_decay: float = 1e-4
    use_amp: bool = False
    
@dataclass
class LoggingConfig:
    use_wandb: bool = False
    project_name: str = "feature-reliance"
    run_name: Optional[str] = None
    verbose_epoch: int = 1
    
@dataclass
class AdaptorConfig:
    reduction: int = 16 
    use_norm: bool = False 
    use_trainable_scale: bool = False
    init_scale: float = 1e-3
    dropout: float = 0.0
    target_layers: str|int|List[str]|List[int] ="last1"

@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cuda"
    verbose_model: bool = False

    # 기존 Config 재사용
    model_spec: ModelSpec = None
    data_config: DataConfig = None
    transform_hparams: TransformHyperParams = None

    # 학습 전용 설정만 새로 정의
    perturbation: str = "localwarp"
    train_dataset_spec: DatasetSpec = None
    val_dataset_spec: Optional[DatasetSpec] = None

    class_map_name: str = "imagenet_r_subset_map"

    model_type: Literal["timm_cnn", "timm_vit", "hf_dinov2_cls"] = "timm_cnn"
    freeze_backbone: bool = True
    freeze_linear_head: bool = True

    loss_config: LossConfig = None
    optim_config: OptimConfig = None
    logging_config: LoggingConfig = None
    adpator_config: AdaptorConfig = None

    def __post_init__(self):
        if self.loss_config is None:
            self.loss_config = LossConfig()

        if self.optim_config is None:
            self.optim_config = OptimConfig()

        if self.logging_config is None:
            self.logging_config = LoggingConfig()  
            
        if self.adpator_config is None:
            self.adpator_config = AdaptorConfig()
