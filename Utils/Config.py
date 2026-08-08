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
    lambda_clean_preserve: float = 0.0
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
    activation_function: str = "gelu"
    # constant-with-warmup 스텝 수. 0이면 스케줄러를 안 만들고 기존 동작(고정 lr) 그대로다.
    # PatchSAE 참조 구현(src/sae_training/utils.get_scheduler)의 'constantwithwarmup'과
    # 같은 형태: lr_scale = min(1.0, (step + 1) / lr_warmup_steps).
    lr_warmup_steps: int = 0

    # --- 후반부 lr 감쇠 ---
    # PatchSAE는 warmup 이후 lr을 끝까지 고정한다. 그런데 그러면 분지에 들어간 뒤에도
    # 갱신 폭이 그대로라 파라미터가 최소점 주변을 계속 배회한다.
    #
    # 실측 근거(2026-08-07, grid_search): 고정된 검증셋에서 재는 val_nmse가 회차마다
    # 1.2~1.9% 흔들렸다. 검증셋이 고정이고 평가가 결정적이므로 이 흔들림은 표본 잡음이
    # 아니라 전부 파라미터 이동이다. 그 결과 평탄구간 전체의 개선폭이 회차간 표준편차의
    # 1.2~1.7배에 그쳐(신호/잡음), 체크포인트 선정이 실력이 아니라 운으로 결정됐다.
    #
    #   "none"   — 감쇠 없음(기존 동작, PatchSAE 그대로)
    #   "linear" — 시작 지점부터 최종값까지 선형 감쇠
    #   "cosine" — 같은 구간을 코사인으로 감쇠
    lr_decay: str = "none"
    # 전체 스텝의 이 비율 지점부터 감쇠를 시작한다. 0.8이면 마지막 20%에서만 줄인다.
    lr_decay_start_frac: float = 0.8
    # 최종 lr = lr * lr_final_frac. 0.0이면 끝에서 정확히 0이 된다.
    lr_final_frac: float = 0.0
    
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
    freeze_anchor: bool = False

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


# ------ SAE Validation / Grid Search Config ------
@dataclass
class SAEBackboneHookConfig:
    target_block: int = 10
    token_scope: str = "all"


@dataclass
class SAETrainScheduleConfig:
    """학습량과 검증 주기를 무엇으로 세는지 정한다.

    mode="epoch"   : 고정 토큰 집합을 epochs번 반복하고 epoch마다 검증한다(기존 동작).
    mode="token_budget": 스트림을 한 번만 흘리면서 total_train_tokens를 채울 때까지
                    학습하고, eval_every_steps마다 검증한다. 같은 활성을 두 번 쓰지
                    않으므로 ViT forward 비용이 토큰 수에 선형이고 RAM 상한이 없다
                    (PatchSAE 참조 구현 src/sae_training/sae_trainer.py:202와 같은 형태).

    epoch 모드에서는 total_train_tokens / eval_every_steps를 무시한다."""

    mode: str = "epoch"  # "epoch" | "token_budget"
    total_train_tokens: Optional[int] = None
    eval_every_steps: int = 200


@dataclass
class SAETokenConfig:
    max_train_tokens: Optional[int] = 1_000_000
    max_val_tokens: Optional[int] = None
    # 정규화 통계와 b_dec 초기화는 학습량과 무관하게 부분표본이면 충분하다. 학습 예산이
    # 커질 때 이 둘이 같이 커지면(특히 b_dec는 Weiszfeld 반복마다 전체 패스를 다시 돈다)
    # 준비 단계가 학습보다 비싸진다. None이면 max_train_tokens를 따른다.
    max_normalizer_tokens: Optional[int] = None
    max_bdec_tokens: Optional[int] = None
    # latent 통계 히스토그램(save_trial_plots)이 쓸 토큰 상한. **max_val_tokens 와 분리한다.**
    #
    # 왜 나눴나: 그 경로가 val_tokens 와 별개로 토큰을 한 벌 더 모은다. 예전에는 둘 다
    # 80만이라 합쳐 4.6 GiB 였는데, max_val_tokens 를 490만으로 올리자 이쪽도 같이 490만이
    # 되면서 fp32 로 14 GiB 를 잡았다. val_tokens 7 GiB 와 합쳐 21 GiB → 학습을 다 끝내고
    # 진단 단계에서 SIGKILL(exit 137). 2026-08-07 실측.
    #
    # 이 값은 latent 발화 히스토그램용이라 검증 지표만큼 표본이 필요하지 않다.
    max_latent_stats_tokens: Optional[int] = 800_000
    cache_dtype: str = "float16"
    # 활성을 어떤 규약으로 정규화할 것인가.
    #   "per_dim" — 차원마다 따로 표준화. (x - mean_d) / std_d, std가 [1, 768] 벡터.
    #   "scalar"  — 중심화는 차원별로 하되 **배율은 전역 스칼라 하나**.
    #               std = sqrt(mean_d(var_d)) 로 잡아 E[||x_norm||2] = sqrt(d) 가 되게 한다
    #               (SAELens 의 'expected_average_only_in' 과 같은 규약).
    #
    # 왜 나누는가: 차원별 나눗셈은 축마다 배율이 달라 **raw 공간의 방향을 뒤튼다.**
    # 실측(2026-08-07, block10 patch 토큰 20만개) std 범위가 1.38~12.16 으로 8.8배라
    # 왜곡이 작지 않다. SAE 가 찾으려는 게 활성 공간의 '방향'(개념)인데 축을 제각기
    # 늘리면 그 방향이 보존되지 않는다. 반면 차원별 **평균 빼기**는 평행이동이라
    # 점들 사이의 거리·방향을 보존하므로 두 모드 모두 유지한다(잔여 offset 은 어차피
    # 학습되는 b_dec 가 흡수한다).
    #
    # 문헌 관례: PatchSAE 는 정규화를 아예 안 하고 손실을 토큰별 ||x||2 로 나눈다.
    # Anthropic/SAELens/OpenAI TopK 는 전역 스칼라를 쓴다. "per_dim" 은 이 저장소 고유
    # 규약이라 외부 수치와 직접 비교할 때 걸림돌이 된다.
    #
    # 저장 형태는 두 모드 모두 [1, D] 로 맞춘다 — "scalar" 면 전 원소가 같은 값이다.
    # FrozenSAE 가 token_std.shape == (1, input_dim) 을 검증하므로 형태를 바꾸지 않는다.
    norm_mode: str = "per_dim"  # "per_dim" | "scalar"
    normalize_chunk_size: int = 65_536
    source_mode: str = "auto"  # "auto", "cache", "stream"
    cache_max_cpu_gib: float = 8.0
    cache_build_peak_factor: float = 3.0
    cache_min_free_cpu_gib_after_build: float = 4.0
    cache_num_workers: int = 0


@dataclass
class SAEConfig:
    expansion: int = 64
    dec_bias_mode: str = "zero"  # "zero", "mean", "geom"
    active_threshold: float = 0.2
    l1_reg: float = 1e-4
    batch_size: int = 7096
    # Weiszfeld 반복 1회 = ViT forward 전체 패스 1회 + 100만 토큰 float64 CPU 연산이라
    # (Utils/SAE_utils.compute_b_dec_init_streaming), 100회면 학습 자체보다 오래 걸린다.
    # b_dec는 학습되는 파라미터라 이건 초기값일 뿐이고, PatchSAE Table 2의 Dec. bias
    # ablation도 geom(L0 29.48) vs mean(32.47)으로 초기화 방식의 영향이 작다고 본다.
    bias_init_geom_max_iter: int = 10
    bias_init_geom_tol: float = 1e-5
    model_compile: bool = True
    amp_dtype: str = "bfloat16"
    check_finite: bool = True
    matmul_precision: str = "high"
    # 재구성 손실과 검증 지표를 어느 공간에서 잴 것인가.
    #   "norm" — 정규화 공간. F.mse_loss(x_hat, x_norm). 모든 차원을 균등 가중한다.
    #   "raw"  — denorm 공간. denorm(x_hat)과 raw x를 비교한다(기본).
    #
    # SAE는 정규화된 입력으로 학습하지만 실제 배포에서는 denorm된 재구성이 ViT에 다시
    # 꽂힌다(Utils/SAE_plot_utils.reconstruct_tokens_with_latent_scaling). "norm"이면
    # 최적화 대상과 배포 현실이 다른 공간에 있게 된다.
    #
    # mu는 상쇄되므로 (x_hat*sigma + mu) - (x*sigma + mu) = (x_hat - x)*sigma 이고,
    # 결국 sigma로 가중한 MSE다. denorm을 실제로 계산하면 큰 값끼리 빼면서 정밀도만 잃는다.
    #
    # 주의: recon 항의 스케일이 mean(sigma^2)배 커진다(실측 약 2.5~3.2). sparsity 압력이
    # lambda*L1/recon 이므로 lambda를 그만큼 올려야 같은 L0가 나온다.
    recon_space: str = "raw"  # "norm" | "raw"

    # --- ghost gradients (PatchSAE src/sae_training/sparse_autoencoder.py 이식) ---
    # 죽은 latent만 골라 ReLU 대신 exp()를 태워 재구성 잔차를 설명하게 하고, 그 손실을
    # recon 항과 같은 크기로 재스케일해 더한다. exp()는 pre-activation이 음수여도
    # 기울기가 0이 아니라서 ReLU+L1이 만든 흡수 상태(한 번 죽으면 못 돌아옴)를 빠져나온다.
    #
    # 기본값 False — 켜면 학습 손실의 정의가 바뀌므로 기존 실행과 직접 비교할 수 없다.
    # PatchSAE 참조 구현의 기본값은 True다.
    use_ghost_grads: bool = False
    # 이 스텝 수 동안 한 번도 발화하지 않은 latent를 dead로 본다
    # (PatchSAE n_forward_passes_since_fired > dead_feature_window, 기본 1000).
    dead_feature_window: int = 1000
    # 발화 판정 임계. active_threshold(0.2, 보고용)와 별개다 — PatchSAE는 1e-8, 즉 z>0이다.
    # 실측(2026-08-07, trial_0000 체크포인트): z>0 기준 dead 61.9%, z>0.2 기준 65.7%.
    dead_feature_threshold: float = 1e-8
    # ghost 항에 쓸 배치 행 수 상한. None이면 배치 전체(PatchSAE와 동일).
    #
    # 실측 (2026-08-07, batch 7096 / hidden 49,152 / dead 30,415 / RTX 4070):
    #   ghost OFF          peak 4.34 GiB   109 ms/step
    #   ghost ON  (전체)    peak 5.78 GiB   306 ms/step  (2.82x)
    #   ghost ON  (2048)   peak 5.79 GiB   175 ms/step  (1.61x)
    #   ghost ON  (1024)   peak 5.79 GiB   149 ms/step  (1.37x)
    #
    # 즉 이건 **속도 손잡이지 VRAM 손잡이가 아니다.** peak는 ghost 슬라이스가 아니라
    # 배치 전체 hidden_pre를 스텝 내내 들고 있는 데서 나오므로 rows를 줄여도 안 준다.
    # 기울기가 닿는 latent 집합은 rows와 무관하게 같고 표본만 줄어든다.
    ghost_grad_max_rows: Optional[int] = None


@dataclass
class SAEEarlyStoppingConfig:
    patience: Optional[int] = 5
    eps: float = 1e-5
    metric_name: str = "val_nmse"
    save_verbose: bool = False


@dataclass
class SAEOutputConfig:
    root_dir: str = "outputs/SAE_validation"
    grid_dir_name: str = "grid_search"
    plot_display_seconds: int = 0
    plot_dpi: int = 200


@dataclass
class SAESparsityConstraintConfig:
    """best checkpoint / best trial 선정에 거는 희소성 제약.

    학습 손실은 건드리지 않는다 — L1 압력은 `SAEConfig.l1_reg` 그대로다. 이 제약은
    "어느 epoch / 어느 trial을 남길 것인가"에만 관여한다(문헌 관행: L1로 누르고 L0로 고른다).

    metric은 `l0_raw`(z>0, 임계 없는 진짜 L0)를 기본으로 한다. `mean_l0`는
    `active_threshold`(기본 0.2) 초과 개수라 활성 크기가 작아지면 실제 밀도와 무관하게
    작아진다 — 제약 지표로 쓰면 dense한 SAE가 통과한다.

    l0_max=None이면 제약 없이 기존 동작 그대로다."""

    l0_max: Optional[float] = None
    metric: str = "l0_raw"  # "l0_raw"(z>0) | "mean_l0"(z>active_threshold)


@dataclass
class SAEDiagnosticsConfig:
    """perturbation latent 랭킹과 latent intervention 곡선이 함께 쓰는 평가 예산.

    두 단계는 val의 **같은 부분집합**을 쓴다 — 랭킹에서 고른 cue latent를 같은 이미지에서
    검증한다. 예전에는 이 값이 오버레이 그림용 12장에 묶여 있어서 intervention의
    acc_drop 분해능이 1/12(0.083)밖에 안 나왔고, cue와 random의 차이가 이미지 한 장
    단위로만 보였다.

    eval_images=None이면 val 전체를 쓴다. intervention 비용은
    (spec 수 x alpha 수) x 이미지 수에 정확히 선형이므로(기본 15 x 7 = 105 구성) alpha를
    줄이면 같은 비용으로 이미지를 늘릴 수 있다 — val 전체(5만 장)는 trial당 수 시간이다."""

    eval_images: Optional[int] = 2000
    # intervention forward의 배치. SAE encode/decode는 내부에서 청크로 쪼개지므로
    # 이 값이 커져도 GPU 피크는 거의 안 오른다.
    eval_batch_size: int = 64
    # 랭킹은 이미지당 [B, patch, hidden] latent 텐서를 CPU에 들고 있어서 배치에 선형으로
    # RAM을 먹는다(B=8, hidden 49152 기준 텐서당 약 0.3 GiB).
    ranking_batch_size: int = 8
    # intervention 곡선의 alpha 격자. None이면 SAE_plot_utils의 기본 7점을 쓴다.
    # alpha=1.0은 latent를 안 건드리는 항등이라 대조군 역할을 하니 항상 넣는 게 좋다.
    intervention_alphas: Optional[List[float]] = None
    # cue latent 랭킹 점수. "absolute"(기존) | "relative"(상대화) | "specific"(특이도) |
    # "relative_specific"(둘 다). absolute는 절대 변화량이라 항상 크게 켜지는 latent가
    # 섭동 민감도와 무관하게 이긴다 — 실측에서 발화 빈도 1~3위가 세 kind 전부를 점령했다.
    perturbation_score_mode: str = "absolute"
    # 상대화의 분모 폭주 방지 하한(발화 빈도). relative 계열을 쓸 때만 의미가 있고,
    # 0이면 하한 없음 — absolute가 기존과 정확히 같게 유지된다.
    perturbation_min_frequency: float = 0.0


@dataclass
class SAEGridSearchConfig:
    enabled: bool = True
    max_trials: Optional[int] = None
    metric: str = "val_nmse"
    mode: Literal["min", "max"] = "min"
    space: Dict[str, List[Any]] = field(default_factory=lambda: {
        "sae.expansion": [16, 32, 64],
        "sae.l1_reg": [3e-5, 1e-4, 3e-4],
        "optim_config.lr": [5e-5, 1e-4],
        "sae.dec_bias_mode": ["zero", "mean"],
        "sae.active_threshold": [0.1, 0.2],
    })


@dataclass
class SAEExperimentConfig:
    model_spec: Optional[ModelSpec] = None
    data_config: DataConfig = field(default_factory=lambda: DataConfig(
        batch_size=64,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
    ))
    train_dataset_spec: DatasetSpec = field(default_factory=lambda: DatasetSpec(
        name="imagenette_train",
        dataset_type="imagenette",
        root="data",
        split="train",
        num_classes=10,
    ))
    val_dataset_spec: DatasetSpec = field(default_factory=lambda: DatasetSpec(
        name="imagenette_val",
        dataset_type="imagenette",
        root="data",
        split="val",
        num_classes=10,
    ))
    extraction_config: ExtractionConfig = field(default_factory=lambda: ExtractionConfig(
        root_dir="Cache",
        device="cuda",
        dtype="float16",
    ))
    optim_config: OptimConfig = field(default_factory=lambda: OptimConfig(
        epochs=350,
        lr=1e-4,
        weight_decay=0.0,
        use_amp=True,
    ))
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)
    hook: SAEBackboneHookConfig = field(default_factory=SAEBackboneHookConfig)
    token: SAETokenConfig = field(default_factory=SAETokenConfig)
    schedule: SAETrainScheduleConfig = field(default_factory=SAETrainScheduleConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)
    early_stopping: SAEEarlyStoppingConfig = field(default_factory=SAEEarlyStoppingConfig)
    output: SAEOutputConfig = field(default_factory=SAEOutputConfig)
    grid: SAEGridSearchConfig = field(default_factory=SAEGridSearchConfig)
    sparsity: SAESparsityConstraintConfig = field(default_factory=SAESparsityConstraintConfig)
    diagnostics: SAEDiagnosticsConfig = field(default_factory=SAEDiagnosticsConfig)

    def validate(self):
        if self.hook.target_block != 11 and self.hook.token_scope.lower() != "patch":
            self.hook.token_scope = "patch"
        return self
