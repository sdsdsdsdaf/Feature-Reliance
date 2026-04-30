import gc
from pathlib import Path
import time
import warnings
import os

from Experiment import format_seconds
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

warnings.filterwarnings(
    "ignore",
    message=r".*Metadata Warning, tag 274 had too many entries.*",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=r".*Corrupt EXIF data.*",
    category=UserWarning,
    module=r"PIL\.TiffImagePlugin",
)




"""
A. clean only
   CE(clean)

B. naive perturb aug
   CE(clean) + CE(perturbed)

C. ours-feature
   CE(clean) + CE(perturbed) + λ_feat * feature consistency

D. ours-both
   CE(clean) + CE(perturbed) + λ_feat * feature consistency + λ_kl * KL

"""



import torch

from Utils.Config import DataConfig, DatasetSpec, EvalScenario, ExtractionConfig, LoggingConfig, LossConfig, ModelSpec, OptimConfig, TrainConfig, TransformHyperParams, AdaptorConfig
from Utils.Dataset import ImageNetValFlatDataset, build_sample_indices_from_targets
from Utils.train_utils import train
from Utils.utils import IMAGENET_R_CLASS_IDS, run_experiments
import timm

if __name__ == "__main__":
    
    
    resnet = timm.create_model("resnet50", pretrained=True)
    
    base_ds = ImageNetValFlatDataset(root="Data", transform=None)
    imagenet_200_indices = build_sample_indices_from_targets(
        targets=base_ds.targets,
        class_ids=IMAGENET_R_CLASS_IDS,
    )
    del base_ds
    
    resnet_model_spec = ModelSpec(
        model_name="resnet50",
        pretrained_weight="in1k",
        model=resnet,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        resize_size=235,
    )
    del resnet

    
    transform_hparams = TransformHyperParams(
        p=1.0,
        prefix="resizecrop",
        resize_size=256,

        bilateral_d=11,
        sigma_color=170.0,
        sigma_space=75.0,

        gaussian_k=11,
        gaussian_sigma=2.0,

        gray_alpha=1.0,

        grid_size=7,
        alpha_localwarp=35.0,
        sigma_localwarp=3.5,
    )


    DATA_SCOPE = "subset"  # "full" or "subset"

    if DATA_SCOPE == "subset":
        train_dataset_spec = DatasetSpec(
            name="imagenet_train_200",
            dataset_type="hf_imagenet_train_subset",
            root="Data/ILSVRC2012/train",
            split="train",
            num_classes=1000,
            labels_map=IMAGENET_R_CLASS_IDS,
            class_map_name="imagenet_r_subset_map",
            domain_type="id",
            id_dataset_name="imagenet_train_200",
        )

        val_dataset_spec = DatasetSpec(
            name="imagenet_200",
            dataset_type="imagenet_val_subset",
            root="Data",
            split="val",
            num_classes=200,
            domain_type="id",
            class_map_name="imagenet_r_subset_map",
            sample_indices=imagenet_200_indices,
            labels_map=IMAGENET_R_CLASS_IDS,
            id_dataset_name="imagenet_200",
        )

    elif DATA_SCOPE == "full":
        train_dataset_spec = DatasetSpec(
            name="imagenet_train",
            dataset_type="hf_imagenet_train_subset",
            root="Data/ILSVRC2012/train",
            split="train",
            num_classes=1000,
            labels_map=list(range(1000)),
            domain_type="id",
            id_dataset_name="imagenet",
        )

        val_dataset_spec = DatasetSpec(
            name="imagenet",
            dataset_type="imagenet_val_flat",
            root="Data",
            split="val",
            num_classes=1000,
            domain_type="id",
            id_dataset_name="imagenet",
        )

    else:
        raise ValueError(f"Unknown DATA_SCOPE: {DATA_SCOPE}")


    data_config = DataConfig(
        batch_size=256,
        num_workers=4,
        pin_memory=False,
        shuffle=True,
        datasets=[
            train_dataset_spec,
            val_dataset_spec,
        ],
    )
    
    

    train_config = TrainConfig(
        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",

        model_spec=resnet_model_spec,
        data_config=data_config,
        transform_hparams=transform_hparams,
        verbose_model=True,

        train_dataset_spec=train_dataset_spec,
        val_dataset_spec=val_dataset_spec,

        perturbation="localwarp",

        model_type="timm_cnn",
        freeze_backbone=True,
        freeze_linear_head=True,

        loss_config=LossConfig(
            mode="feature",
            feature_loss_type="cosine",
            lambda_feat=1.0,
            lambda_kl=0.1,
            lambda_scale=1e-2,
            lambda_delta=1.0,
            temperature=2.0,
            detach_teacher=True,
            normalize_feature=True,
            ce_clean_weight=0.5,
            ce_pert_weight=1.0,
        ),

        optim_config=OptimConfig(
            epochs=5,
            lr=1e-4,
            weight_decay=1e-4,
            use_amp=True,
        ),

        logging_config=LoggingConfig(
            use_wandb=True,
            run_name="resnet50_localwarp_feature_ce_clean=0.3",
            verbose_epoch=1,
        ),
        
        adpator_config = AdaptorConfig(
            reduction=16,
            use_norm=True,
            use_trainable_scale=True,
            init_scale=1e-3,
            target_layers="last1"
        )
    )
        
        
    model, history = train(train_config)
    gc.collect()
    
    # InterVention OOD Check
    data_config = DataConfig(
        batch_size=512,
        num_workers=4,
        pin_memory=False,
        shuffle=False,
        datasets=[
            DatasetSpec(
                name="imagenet",
                dataset_type="imagenet_val_flat",
                root="Data",
                split="val",
                num_classes=1000,
                domain_type="id",
                id_dataset_name="imagenet"
            ),
            
            DatasetSpec(
                name="imagenet_200",
                dataset_type="imagenet_val_subset",
                root="Data",
                split="val",
                num_classes=200,
                domain_type="id",
                class_map_name="imagenet_r_subset_map",
                sample_indices=imagenet_200_indices,
                labels_map=IMAGENET_R_CLASS_IDS,
                id_dataset_name="imagenet_200"
            ),
            
            # OOD dataset 추가 시
            DatasetSpec(
                name="imagenet_r",
                dataset_type="imagenet_r",
                root="Data/imagenet-r",
                split="val",
                num_classes=200,
                domain_type="natural_ood",
                shift_type="style",
                class_map_name="imagenet_r_subset_map",
                eval_protocol_name="imagenet_r_eval",
                id_dataset_name="imagenet_200"
            ),
        ],
    )

    extraction_config = ExtractionConfig(
        root_dir="Cache",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype="float16",
        overwrite=False,
        debug_first_batch=False,
    )
    
    metadata_root = Path(extraction_config.root_dir) / "Intervention_MetaData"
    resnet = timm.create_model("resnet50", pretrained=True)
    
    start_time = time.time()
    
    model_specs = [
        ModelSpec(
            model_name="localwarp_intervention_resnet50",
            pretrained_weight="in1k",
            model=model.backbone,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            resize_size=235, # ResNet50의 경우 256으로 resize 후 center crop 224 진행 (timm의 기본 전처리 방식)
        ),
        ModelSpec(
            model_name="resnet50",
            pretrained_weight="in1k",
            model=resnet,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            resize_size=235, # ResNet50의 경우 256으로 resize 후 center crop 224 진행 (timm의 기본 전처리 방식)
        ),
    ]
    
    # 기본값
    hparams_kwargs = dict(
        resize_size=256,
        p=1.0,
        prefix="resizecrop",

        bilateral_d=11,
        sigma_color=170,
        sigma_space=75,

        gaussian_k=11,
        gaussian_sigma=2.0,
        gray_alpha=1.0,

        grid_size=7,
        alpha_localwarp=35,
        sigma_localwarp=3.5,
    )

    # 각 실험 조건에 따라 수기로 변경
 

    transform_hparams = TransformHyperParams(**hparams_kwargs)

    
    try:
        result = run_experiments(
            model_specs=model_specs,
            transform_hparams=transform_hparams,
            data_config=data_config,
            extraction_config=extraction_config,
            metadata_root=metadata_root,
            perturbations=[
                "original",
                "grayscale",
                "bilateral",
                "patchshuffle",
                "patchrotation",
                "localwarp",
            ],
            verbose_image=False,
            run_validation=False,
            validation_max_samples=None,
            max_workers=5,
        )

    finally:
        del model_specs

        if "result" in locals():
            del result

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elapsed = time.time() - start_time
    print(f"Elapsed time: {format_seconds(elapsed)}")
    print()
