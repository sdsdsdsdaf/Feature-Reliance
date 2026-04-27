import timm
import torch

from Utils.Config import *
from Utils.Dataset import ImageNetValFlatDataset, build_sample_indices_from_targets
from Utils.utils import IMAGENET_R_CLASS_IDS, run_experiments, set_seed, get_system_info

from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    TransferSpeedColumn,
)
from itertools import product
from pathlib import Path
import gc

alpha_grid = [0, 5, 10, 20, 35, 50, 80] # LocalWarp의 alpha 값 후보군
sigma_grid = [0.5, 1.0, 2.0, 3.5, 6.0]

sigma_color_grid = [10, 30, 60, 100, 150, 170] # Bilateral Filter의 sigma_color 값 후보군
sigma_space_grid = [3, 5, 10, 20, 40, 75]

grid_size_grid = [3, 4, 5, 6, 7, 8, 9, 10, 11, 15] # PatchShuffle의 grid_size 값 후보군
temp = [1]



if __name__ == "__main__":
    set_seed(42)
    
    # 1. pair grid 정의
    localwarp_pairs = list(product(alpha_grid, sigma_grid))
    bilateral_pairs = list(product(sigma_color_grid, sigma_space_grid))
    patch_pairs = [(g,) for g in grid_size_grid]

    # 2. 실험 대상 선택
    experiment_pairs = [
        ("localwarp", pair) for pair in localwarp_pairs
    ] + [
        ("bilateral", pair) for pair in bilateral_pairs
    ] + [
        ("patchshuffle", pair) for pair in patch_pairs
    ]
    total_i = len(experiment_pairs)
    
    get_system_info()
    base_ds = ImageNetValFlatDataset(root="Data", transform=None)
    imagenet_200_indices = build_sample_indices_from_targets(
        targets=base_ds.targets,
        class_ids=IMAGENET_R_CLASS_IDS,
    )
    del base_ds
    print()

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
    
    scenarios = [
        EvalScenario(dataset_name="imagenet", perturbation="original"),
        EvalScenario(dataset_name="imagenet", perturbation="grayscale"),
        EvalScenario(dataset_name="imagenet", perturbation="patchshuffle"),

        EvalScenario(dataset_name="imagenet_r", perturbation="original"),
        EvalScenario(dataset_name="imagenet_r", perturbation="grayscale"),
        EvalScenario(dataset_name="imagenet_r", perturbation="patchshuffle"),
    ] # 개별 실험 내역
    
    metadata_root = Path(extraction_config.root_dir) / "MetaData"
    for i, (perturb_name, pair) in enumerate(experiment_pairs, start=1):
        
        expected_trial_id = i - 1
        expected_summary_path = (
            metadata_root
            / f"trial_{expected_trial_id:04d}"
            / "experiment_summary.json"
        )

        if expected_summary_path.exists():
            print(
                f"[{i}/{total_i}] Skip {perturb_name} {pair} "
                f"because summary exists: {expected_summary_path}"
            )
            continue
        
        print(f"[{i}/{total_i}] Running {perturb_name} with {pair}")
        
        resnet = timm.create_model("resnet50", pretrained=True)
        vit = timm.create_model("vit_base_patch16_224.augreg_in1k", pretrained=True)
        
        model_specs = [
            ModelSpec(
                model_name="resnet50",
                pretrained_weight="in1k",
                model=resnet,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                resize_size=235, # ResNet50의 경우 256으로 resize 후 center crop 224 진행 (timm의 기본 전처리 방식)
            ),
            ModelSpec(
                model_name="vit-b",
                pretrained_weight="augreg_in1k",
                model=vit,
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
                resize_size=248, # ViT-B/16의 경우 224보다 약간 큰 248로 resize 후 center crop 224 진행 (timm의 기본 전처리 방식)
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

         # pair에 따라 해당 perturbation 파라미터만 변경
        if perturb_name == "localwarp":
            alpha, sigma = pair
            hparams_kwargs["alpha_localwarp"] = alpha
            hparams_kwargs["sigma_localwarp"] = sigma

        elif perturb_name == "bilateral":
            sigma_color, sigma_space = pair
            hparams_kwargs["sigma_color"] = sigma_color
            hparams_kwargs["sigma_space"] = sigma_space

        elif perturb_name == "patchshuffle":
            (grid_size,) = pair
            hparams_kwargs["grid_size"] = grid_size

        transform_hparams = TransformHyperParams(**hparams_kwargs)

        
        try:
            result = run_experiments(
                model_specs=model_specs,
                transform_hparams=transform_hparams,
                data_config=data_config,
                extraction_config=extraction_config,
                perturbations=[
                    "original",
                    "grayscale",
                    "bilateral",
                    "patchshuffle",
                    "patchrotation",
                    "localwarp",
                ],
                verbose_image=True,
                run_validation=True,
                validation_max_samples=None,
                max_workers=5,
            )

        finally:
            del resnet, vit, model_specs

            if "result" in locals():
                del result

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print()
