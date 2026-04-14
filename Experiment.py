import timm
import torch

from Utils.Config import *
from Utils.utils import run_experiments


if __name__ == "__main__":

    resnet = timm.create_model("resnet50", pretrained=True)
    vit = timm.create_model("vit_base_patch16_224.augreg_in1k", pretrained=True)

    model_specs = [
        ModelSpec(
            model_name="resnet50",
            pretrained_weight="in1k",
            model=resnet,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ModelSpec(
            model_name="vit-b",
            pretrained_weight="augreg_in1k",
            model=vit,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ]

    transform_hparams = TransformHyperParams(
        resize_size=224,
        p=1.0,
        prefix="resizecrop",
        bilateral_d=11,
        sigma_color=170,
        sigma_space=75,
        gaussian_k=11,
        gaussian_sigma=2.0,
        gray_alpha=1.0,
        grid_size=6,
        alpha_localwarp=35.0,
        sigma_localwarp=2.5,
    )

    data_config = DataConfig(
        batch_size=256,
        num_workers=4,
        pin_memory=False,
        shuffle=False,
        dataset_root="Data",   # 필요 없으면 None
    )

    extraction_config = ExtractionConfig(
        root_dir="Cache",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype="float16",
        overwrite=False,
        debug_first_batch=False,
    )

    result:ExperimentResult = run_experiments(
        model_specs=model_specs,
        transform_hparams=transform_hparams,
        data_config=data_config,
        extraction_config=extraction_config,
        perturbations=[
            "original",
            "bilateral",
            "grayscale",
            "patchshuffle",
            "patchrotation",
            "localwarp",
        ],
        verbose_image=False,
        save_summary=True,
    )

    print(result.model_results["resnet50__in1k"].original_accuracy)
    print(result.model_results["resnet50__in1k"].perturbations)
    
    print(result.model_results["vit-b__in1k"].original_accuracy)
    print(result.model_results["vit__in1k"].perturbations)