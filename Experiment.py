import timm
import torch

from Utils.Config import *
from Utils.Dataset import ImageNetValFlatDataset, build_sample_indices_from_targets
from Utils.utils import IMAGENET_R_CLASS_IDS, run_experiments, set_seed, get_system_info

alpha_grid = [0, 5, 10, 20, 35, 50, 80]
sigma_grid = [0.5, 1.0, 2.0, 3.5, 6.0]


if __name__ == "__main__":
    set_seed(42)
    
    get_system_info()
    base_ds = ImageNetValFlatDataset(root="Data", transform=None)
    imagenet_200_indices = build_sample_indices_from_targets(
        targets=base_ds.targets,
        class_ids=IMAGENET_R_CLASS_IDS,
    )
    del base_ds
    
    resnet = timm.create_model("resnet50", pretrained=True)
    vit = timm.create_model("vit_base_patch16_224.augreg_in1k", pretrained=True)
    i = 0
    print()
    for alpha in alpha_grid:
        for sigma in sigma_grid:
            i += 1
            total_i = len(sigma_grid) * len(alpha_grid)
            print(f"[{i}/{total_i}] alpha: {alpha}, sigma: {sigma}")
    
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
                resize_size=256,
                p=1.0,
                prefix="resizecrop",
                bilateral_d=11,
                sigma_color=170,
                sigma_space=75,
                gaussian_k=11,
                gaussian_sigma=2.0,
                gray_alpha=1.0,
                grid_size=6,
                alpha_localwarp=alpha,
                sigma_localwarp=sigma,
            )

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
                run_validation= True,
                validation_max_samples = 10000, # 최종 실험시에는 None으로
            )
            print()
