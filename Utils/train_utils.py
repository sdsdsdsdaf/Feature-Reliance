import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torch import Tensor
from torch.amp import GradScaler
from Utils.Config import DatasetSpec

import wandb
import timm
from transformers import AutoModelForImageClassification

from Model.model import UnifiedModel
from Model.Adaptor import inject_adaptors

try:
    from Utils.Dataset import ImageNetValFlatDataset, build_sample_indices_from_targets
    from Utils.utils import CLASS_MAPPING_REGISTRY, build_transform, build_dataset, IMAGENET_R_CLASS_IDS, set_seed
    from Utils.Dataset import HFImageNetTrainSubsetDataset
    from Utils.Config import (
        TrainConfig, 
        ModelSpec, 
        DataConfig, 
        TransformHyperParams,
        LossConfig,
        OptimConfig,
        LoggingConfig,
    )
    from Model.Loss import ConsistencyLoss
    from Model.model import UnifiedModel
    from Model.Adaptor import inject_adaptors
except ImportError:
    from Dataset import ImageNetValFlatDataset, build_sample_indices_from_targets
    from utils import CLASS_MAPPING_REGISTRY, build_transform, build_dataset, IMAGENET_R_CLASS_IDS, set_seed
    from Dataset import HFImageNetTrainSubsetDataset
    from Utils.Config import (
        TrainConfig, 
        ModelSpec, 
        DataConfig, 
        TransformHyperParams,
        LossConfig,
        OptimConfig,
        LoggingConfig,
    )
    from Model.Loss import ConsistencyLoss
    from Model.model import UnifiedModel
    from Model.Adaptor import inject_adaptors

def set_trainable_params(model:nn.Module, freeze_backbone: bool = True, freeze_linear_head: bool = True):
    if not freeze_backbone:
        for p in model.parameters():
            p.requires_grad = True
        return model

    for name, p in model.named_parameters():
        name_lower = name.lower()

        is_adapter = (
            "adapter" in name_lower
            or "adaptor" in name_lower
        )

        is_linear_head = (
            "classifier" in name_lower
            or "fc" in name_lower
            or "head" in name_lower
        )

        if is_adapter:
            p.requires_grad = True

        elif is_linear_head:
            p.requires_grad = not freeze_linear_head

        else:
            p.requires_grad = False

    return model

def compute_acc(logits:Tensor, labels:Tensor):
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct, total

def build_train_val_dataloaders(config: TrainConfig):
    model_spec = config.model_spec
    data_config = config.data_config
    hparams = config.transform_hparams
    class_ids = CLASS_MAPPING_REGISTRY[config.class_map_name]["subset_class_ids"]
    
    train_dataset_spec = config.train_dataset_spec
    val_dataset_spec = config.val_dataset_spec
    
    clean_transform = build_transform(
        perturbation="original",
        mean=model_spec.mean,
        std=model_spec.std,
        resize_size=model_spec.resize_size,
        hparams=hparams,
        normalize=True,
    )
    perturb_transform = build_transform(
        perturbation=config.perturbation,
        mean=model_spec.mean,
        std=model_spec.std,
        resize_size=model_spec.resize_size,
        hparams=hparams,
        normalize=True,
    )
    
    train_ds = build_dataset(
        dataset_spec=train_dataset_spec,
        clean_transform=clean_transform,
        perturb_transform=perturb_transform,
    )

    
    train_loader = DataLoader(
        train_ds,
        batch_size=data_config.batch_size,
        shuffle=data_config.shuffle,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        drop_last=False,
    )
    
    val_loader = None
    if val_dataset_spec is not None:

        val_ds = build_dataset(
            dataset_spec=val_dataset_spec,
            clean_transform=clean_transform,
            perturb_transform=perturb_transform,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=data_config.batch_size,
            shuffle=False,
            num_workers=data_config.num_workers,
            pin_memory=data_config.pin_memory,
            drop_last=False,
        )

    return train_loader, val_loader

def build_train_model(config: TrainConfig):
    

    model_spec = config.model_spec
    timm_model_name_map = {
        ("vit-b", "augreg_in1k"): "vit_base_patch16_224.augreg_in1k",
    }
    hf_model_name_map = {
        ("dinov2_vit-b", "imagenet1k-1-layer"): "facebook/dinov2-base-imagenet1k-1-layer",
    }

    if config.model_type in ["timm_cnn", "timm_vit"]:
        backbone_source_name = timm_model_name_map.get(
            (model_spec.model_name, model_spec.pretrained_weight),
            model_spec.model_name,
        )

        backbone = timm.create_model(
            backbone_source_name,
            pretrained=True,
        )

        backbone = inject_adaptors(
            backbone,
            model_spec.model_name,
        )

    elif config.model_type == "hf_dinov2_cls":
        backbone_source_name = hf_model_name_map.get(
            (model_spec.model_name, model_spec.pretrained_weight),
            model_spec.model_name,
        )

        backbone = AutoModelForImageClassification.from_pretrained(
            backbone_source_name,
        )

        backbone = inject_adaptors(
            backbone,
            model_spec.model_name,
        )

    else:
        raise ValueError(f"Unsupported model_type: {config.model_type}")

    model = UnifiedModel(
        backbone=backbone,
        model_type=config.model_type,
    )

    model = set_trainable_params(
        model,
        freeze_backbone=config.freeze_backbone,
        freeze_linear_head=config.freeze_linear_head,
    )

    return model
    
def train_one_epoch(
    model: nn.Module,
    train_dataloader,
    val_dataloader,
    optimizer,
    criterion,
    device,
    scaler=None,
    epoch: int = 0,
    verbose_epoch: int = 1,
    use_wandb: bool = False,
    global_step: int = 0,
):
    model.train()

    use_amp = scaler is not None
    device_type = "cuda" if torch.device(device).type == "cuda" else "cpu"

    train_loss = 0.0
    train_ce_clean = 0.0
    train_ce_pert = 0.0
    train_cons = 0.0

    train_clean_correct = 0
    train_pert_correct = 0
    train_total = 0

    for batch in tqdm(train_dataloader, desc=f"Train Epoch {epoch + 1}"):
        x_clean = batch["clean"].to(device, non_blocking=True)
        x_pert = batch["perturbed"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device_type, enabled=use_amp):
            logits_clean, feat_clean = model(x_clean, return_features=True)
            logits_pert, feat_pert = model(x_pert, return_features=True)

            loss, loss_dict = criterion(
                original_logits=logits_clean,
                perturbed_logits=logits_pert,
                labels=y,
                original_features=feat_clean,
                perturbed_features=feat_pert,
            )

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        clean_correct, batch_total = compute_acc(logits_clean.detach(), y)
        pert_correct, _ = compute_acc(logits_pert.detach(), y)

        train_clean_correct += clean_correct
        train_pert_correct += pert_correct
        train_total += batch_total

        train_loss += loss.item()
        train_ce_clean += loss_dict["loss_ce_clean"].item()
        train_ce_pert += loss_dict["loss_ce_pert"].item()
        train_cons += loss_dict["loss_consistency"].item()

        step_metrics = {
            "step/train_loss": loss.item(),
            "step/train_ce_clean": loss_dict["loss_ce_clean"].item(),
            "step/train_ce_pert": loss_dict["loss_ce_pert"].item(),
            "step/train_consistency": loss_dict["loss_consistency"].item(),
            "step/train_acc_clean": clean_correct / batch_total,
            "step/train_acc_pert": pert_correct / batch_total,
            "epoch": epoch + 1,
        }

        if use_wandb:
            import wandb
            wandb.log(step_metrics, step=global_step)

        global_step += 1

    n_train = len(train_dataloader)

    metrics = {
        "train/loss": train_loss / n_train,
        "train/ce_clean": train_ce_clean / n_train,
        "train/ce_pert": train_ce_pert / n_train,
        "train/consistency": train_cons / n_train,
        "train/acc_clean": train_clean_correct / train_total,
        "train/acc_pert": train_pert_correct / train_total,
        "epoch": epoch + 1,
        "global_step": global_step,
    }

    if val_dataloader is not None and (epoch + 1) % verbose_epoch == 0:
        model.eval()

        val_loss = 0.0
        val_ce_clean = 0.0
        val_ce_pert = 0.0
        val_cons = 0.0

        val_clean_correct = 0
        val_pert_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Val Epoch {epoch + 1}"):
                x_clean = batch["clean"].to(device, non_blocking=True)
                x_pert = batch["perturbed"].to(device, non_blocking=True)
                y = batch["label"].to(device, non_blocking=True)

                with autocast(device_type=device_type, enabled=use_amp):
                    logits_clean, feat_clean = model(x_clean, return_features=True)
                    logits_pert, feat_pert = model(x_pert, return_features=True)

                    loss, loss_dict = criterion(
                        original_logits=logits_clean,
                        perturbed_logits=logits_pert,
                        labels=y,
                        original_features=feat_clean,
                        perturbed_features=feat_pert,
                    )

                clean_correct, batch_total = compute_acc(logits_clean, y)
                pert_correct, _ = compute_acc(logits_pert, y)

                val_clean_correct += clean_correct
                val_pert_correct += pert_correct
                val_total += batch_total

                val_loss += loss.item()
                val_ce_clean += loss_dict["loss_ce_clean"].item()
                val_ce_pert += loss_dict["loss_ce_pert"].item()
                val_cons += loss_dict["loss_consistency"].item()

        n_val = len(val_dataloader)

        val_metrics = {
            "val/loss": val_loss / n_val,
            "val/ce_clean": val_ce_clean / n_val,
            "val/ce_pert": val_ce_pert / n_val,
            "val/consistency": val_cons / n_val,
            "val/acc_clean": val_clean_correct / val_total,
            "val/acc_pert": val_pert_correct / val_total,
        }

        metrics.update(val_metrics)

    if use_wandb and ((epoch + 1) % verbose_epoch == 0):
        import wandb
        wandb.log(metrics, step=global_step)

    return metrics, global_step

def train(config: TrainConfig):

    set_seed(config.seed)
    
    if config.model_spec is None:
        raise ValueError("model_spec must be provided in TrainConfig")
    if config.data_config is None:
        raise ValueError("data_config must be provided in TrainConfig")
    if config.transform_hparams is None:
        raise ValueError("transform_hparams must be provided in TrainConfig")

    device_obj = torch.device(config.device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        device_obj = torch.device("cpu")
    device = str(device_obj)

    if config.logging_config.use_wandb:
        wandb.init(
            project=config.logging_config.project_name,
            name=config.logging_config.run_name,
            config={
                "seed": config.seed,
                "device": config.device,
                "model_spec": config.model_spec.__dict__,
                "data_config": config.data_config.__dict__,
                "transform_hparams": config.transform_hparams.__dict__,
                "loss_config": config.loss_config.__dict__,
                "optim_config": config.optim_config.__dict__,
                "perturbation": config.perturbation,
                "model_type": config.model_type,
                "freeze_backbone": config.freeze_backbone,
            },
        )

    train_loader, val_loader = build_train_val_dataloaders(config)

    model = build_train_model(config).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.optim_config.lr,
        weight_decay=config.optim_config.weight_decay,
    )

    criterion = ConsistencyLoss(
        mode=config.loss_config.mode,
        lambda_kl=config.loss_config.lambda_kl,
        lambda_feat=config.loss_config.lambda_feat,
        temperature=config.loss_config.temperature,
        detach_teacher=config.loss_config.detach_teacher,
        normalize_feature=config.loss_config.normalize_feature,
        ce_clean_weight=config.loss_config.ce_clean_weight,
        ce_pert_weight=config.loss_config.ce_pert_weight,
    ).to(device)
    
    scaler = None
    if config.optim_config.use_amp and device_obj.type == "cuda":
        scaler = GradScaler("cuda")

    global_step = 0
    history = []

    for epoch in range(config.optim_config.epochs):
        metrics, global_step = train_one_epoch(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            epoch=epoch,
            verbose_epoch=config.logging_config.verbose_epoch,
            use_wandb=config.logging_config.use_wandb,
            global_step=global_step,
        )

        history.append(metrics)
        print(metrics)

    if config.logging_config.use_wandb:
        
        wandb.finish()

    return model, history


if __name__ == "__main__":
    
    import timm
    
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
        num_classes=1000,
        domain_type="id",
        class_map_name="imagenet_r_subset_map",
        sample_indices=imagenet_200_indices,
        labels_map=IMAGENET_R_CLASS_IDS,
        id_dataset_name="imagenet_200"
    )

    data_config = DataConfig(
        batch_size=128,
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

        train_dataset_spec=train_dataset_spec,
        val_dataset_spec=val_dataset_spec,

        perturbation="localwarp",

        model_type="timm_cnn",
        freeze_backbone=True,
        freeze_linear_head=False,

        loss_config=LossConfig(
            mode="feature",
            lambda_feat=0.1,
            lambda_kl=0.1,
            temperature=2.0,
        ),

        optim_config=OptimConfig(
            epochs=10,
            lr=1e-4,
            weight_decay=1e-4,
            use_amp=True,
        ),

        logging_config=LoggingConfig(
            use_wandb=False,
            run_name="resnet50_localwarp_feature",
            verbose_epoch=1,
        ),
    )
        
        
    model, history = train(train_config)
