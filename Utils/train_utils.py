import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torch import Tensor
from torch.amp import GradScaler

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
    
    from Utils.Config import DatasetSpec
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
    
    from Config import DatasetSpec
    from Model.Loss import ConsistencyLoss
    from Model.model import UnifiedModel
    from Model.Adaptor import inject_adaptors
    
from collections import defaultdict
import torch.nn as nn


import torch.nn as nn
# ============================================================
# Parameter counting utilities
# ============================================================

def count_params(module: nn.Module, trainable_only: bool = False) -> int:
    """
    Count parameters in a module.
    """
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    return sum(p.numel() for p in module.parameters())


def count_unique_params_from_modules(modules):
    """
    Count unique parameters from a list of modules.
    Avoid double-counting shared or nested parameters.
    """
    seen = set()
    total = 0
    trainable = 0

    for module in modules:
        for p in module.parameters():
            pid = id(p)

            if pid in seen:
                continue

            seen.add(pid)
            total += p.numel()

            if p.requires_grad:
                trainable += p.numel()

    return total, trainable


def get_module_param_ids(modules):
    """
    Return unique parameter ids belonging to the given modules.
    """
    param_ids = set()

    for module in modules:
        for p in module.parameters():
            param_ids.add(id(p))

    return param_ids


# ============================================================
# Module detection
# ============================================================

def get_adaptor_modules(model: nn.Module):
    """
    Find actual adaptor modules only.

    This avoids counting child modules such as:
    - layer4.adaptor.down
    - layer4.adaptor.act
    - layer4.adaptor.up

    Expected:
    - layer4.adaptor -> ConvAdaptor
    """
    adaptor_class_names = {
        "LinearAdaptor",
        "ConvAdaptor",
    }

    adaptor_modules = []

    for name, module in model.named_modules():
        if name == "":
            continue

        name_lower = name.lower()
        last_name = name_lower.split(".")[-1]
        class_name = module.__class__.__name__

        is_adaptor_module = (
            class_name in adaptor_class_names
            or last_name in {"adaptor", "adapter"}
        )

        if is_adaptor_module:
            adaptor_modules.append(module)

    return adaptor_modules

def get_head_modules(model: nn.Module):
    """
    Find likely classifier/head modules from nested module paths.

    This is safer than checking only top-level attributes like model.head.
    """
    head_modules = []

    head_names = {
        "head",
        "heads",
        "classifier",
        "classifiers",
        "fc",
    }

    for name, module in model.named_modules():
        if name == "":
            continue

        name_lower = name.lower()
        last_name = name_lower.split(".")[-1]

        is_head_module = last_name in head_names

        if is_head_module and isinstance(module, nn.Module):
            head_modules.append(module)

    return head_modules


# ============================================================
# Summary printing
# ============================================================

def print_model_param_summary(model: nn.Module):
    """
    Print parameter summary using unique parameter ids.

    Priority:
    - adaptor parameters are counted as Adaptor
    - remaining head parameters are counted as Head
    - all others are counted as Other / Backbone
    """
    adaptor_modules = get_adaptor_modules(model)
    head_modules = get_head_modules(model)

    adaptor_param_ids = get_module_param_ids(adaptor_modules)
    head_param_ids = get_module_param_ids(head_modules)

    total_params = 0
    trainable_params = 0

    adaptor_total = 0
    adaptor_trainable = 0

    head_total = 0
    head_trainable = 0

    other_total = 0
    other_trainable = 0

    for name, p in model.named_parameters():
        n = p.numel()
        pid = id(p)
        is_trainable = p.requires_grad

        total_params += n

        if is_trainable:
            trainable_params += n

        # Adaptor has priority over head if names/modules overlap
        if pid in adaptor_param_ids:
            adaptor_total += n
            if is_trainable:
                adaptor_trainable += n

        elif pid in head_param_ids:
            head_total += n
            if is_trainable:
                head_trainable += n

        else:
            other_total += n
            if is_trainable:
                other_trainable += n

    frozen_params = total_params - trainable_params

    rows = [
        ("Total Model", total_params, trainable_params),
        ("Head", head_total, head_trainable),
        ("Adaptor", adaptor_total, adaptor_trainable),
        ("Other / Backbone", other_total, other_trainable),
        ("Frozen Params", frozen_params, 0),
    ]

    print("\n" + "=" * 80)
    print("Model Parameter Summary")
    print("=" * 80)
    print(
        f"{'Module':<20} "
        f"{'Total Params':>18} "
        f"{'Trainable Params':>20} "
        f"{'Trainable %':>14}"
    )
    print("-" * 80)

    for row_name, total, trainable in rows:
        ratio = (trainable / total * 100) if total > 0 else 0.0

        print(
            f"{row_name:<20} "
            f"{total:>18,} "
            f"{trainable:>20,} "
            f"{ratio:>13.2f}%"
        )

    print("-" * 80)
    print(f"Number of adaptor modules: {len(adaptor_modules)}")
    print(f"Number of head modules:    {len(head_modules)}")
    print("=" * 80 + "\n")


# ============================================================
# Debug utilities
# ============================================================

def print_trainable_params(model: nn.Module):
    """
    Print all currently trainable parameters.
    """
    total = 0

    print("=" * 100)
    print("Trainable Parameters")
    print("=" * 100)

    for name, p in model.named_parameters():
        if p.requires_grad:
            n = p.numel()
            total += n
            print(f"{name:<80} {str(tuple(p.shape)):<25} {n:,}")

    print("-" * 100)
    print(f"Total trainable params: {total:,}")
    print("=" * 100)


def print_trainable_non_adaptor_params(model: nn.Module):
    """
    Print trainable parameters excluding adaptor/adapter parameters.
    This should be empty when freeze_backbone=True and freeze_linear_head=True.
    """
    total = 0

    print("=" * 100)
    print("Trainable Parameters Excluding Adaptor")
    print("=" * 100)

    for name, p in model.named_parameters():
        name_lower = name.lower()

        is_adaptor = (
            "adaptor" in name_lower
            or "adapter" in name_lower
        )

        if p.requires_grad and not is_adaptor:
            n = p.numel()
            total += n
            print(f"{name:<80} {str(tuple(p.shape)):<25} {n:,}")

    print("-" * 100)
    print(f"Total non-adaptor trainable params: {total:,}")
    print("=" * 100)


def print_detected_adaptor_modules(model: nn.Module):
    """
    Print detected adaptor modules.
    """
    adaptor_modules = get_adaptor_modules(model)

    print("=" * 100)
    print("Detected Adaptor Modules")
    print("=" * 100)

    for target_module in adaptor_modules:
        for name, module in model.named_modules():
            if module is target_module:
                total = sum(p.numel() for p in module.parameters())
                trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)

                print(
                    f"{name:<70} "
                    f"{module.__class__.__name__:<25} "
                    f"total={total:,} "
                    f"trainable={trainable:,}"
                )
                break

    print("=" * 100)


def print_detected_head_modules(model: nn.Module):
    """
    Print detected head modules.
    """
    head_modules = get_head_modules(model)

    print("=" * 100)
    print("Detected Head Modules")
    print("=" * 100)

    for target_module in head_modules:
        for name, module in model.named_modules():
            if module is target_module:
                total = sum(p.numel() for p in module.parameters())
                trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)

                print(
                    f"{name:<70} "
                    f"{module.__class__.__name__:<25} "
                    f"total={total:,} "
                    f"trainable={trainable:,}"
                )
                break

    print("=" * 100)


def assert_only_adaptor_trainable(model: nn.Module):
    """
    Assert that only adaptor/adapter parameters are trainable.
    Use this when freeze_backbone=True and freeze_linear_head=True.
    """
    bad = []

    for name, p in model.named_parameters():
        name_lower = name.lower()

        is_adaptor = (
            "adaptor" in name_lower
            or "adapter" in name_lower
        )

        if p.requires_grad and not is_adaptor:
            bad.append((name, tuple(p.shape), p.numel()))

    if bad:
        print("=" * 100)
        print("Unexpected Trainable Parameters")
        print("=" * 100)

        for name, shape, n in bad:
            print(f"{name:<80} {str(shape):<25} {n:,}")

        print("=" * 100)

        total = sum(x[2] for x in bad)

        raise RuntimeError(
            f"Found {len(bad)} non-adaptor trainable parameters. "
            f"Total = {total:,}"
        )
    
def collect_adaptor_summary_metrics(model: nn.Module, prefix: str = "step/adaptor_summary"):
    grad_norms = []
    weight_norms = []
    scale_values = []

    for module in model.modules():
        if module.__class__.__name__ not in ["LinearAdaptor", "ConvAdaptor"]:
            continue

        for name, p in module.named_parameters():
            if "weight" in name:
                weight_norms.append(p.detach().norm())

            if p.grad is not None:
                grad_norms.append(p.grad.detach().norm())

        if hasattr(module, "scale") and isinstance(module.scale, nn.Parameter):
            scale_values.append(module.scale.detach().mean())

    metrics = {}

    if grad_norms:
        grad_norms = torch.stack(grad_norms)
        metrics[f"{prefix}/grad_norm_mean"] = grad_norms.mean().item()
        metrics[f"{prefix}/grad_norm_max"] = grad_norms.max().item()

    if weight_norms:
        weight_norms = torch.stack(weight_norms)
        metrics[f"{prefix}/weight_norm_mean"] = weight_norms.mean().item()
        metrics[f"{prefix}/weight_norm_max"] = weight_norms.max().item()

    if scale_values:
        scale_values = torch.stack(scale_values)
        metrics[f"{prefix}/scale_mean"] = scale_values.mean().item()
        metrics[f"{prefix}/scale_max"] = scale_values.max().item()
        metrics[f"{prefix}/scale_min"] = scale_values.min().item()

    return metrics

def get_adaptor_scale_reg(model: nn.Module):
    scale_reg = None

    for module in model.modules():
        if hasattr(module, "scale") and isinstance(module.scale, nn.Parameter):
            term = module.scale.pow(2).sum()
            scale_reg = term if scale_reg is None else scale_reg + term

    if scale_reg is None:
        device = next(model.parameters()).device
        scale_reg = torch.zeros((), device=device)

    return scale_reg


def get_adaptor_delta_reg(model: nn.Module):
    regs = []

    for module in model.modules():
        if hasattr(module, "last_delta") and module.last_delta is not None:
            regs.append(module.last_delta.pow(2).mean())

    if not regs:
        device = next(model.parameters()).device
        return torch.zeros((), device=device)

    return torch.stack(regs).mean()


def collect_adaptor_delta_ratios(model: nn.Module, prefix: str = "step/adaptor_delta_ratio"):
    ratios = []

    for module in model.modules():
        if hasattr(module, "last_delta_ratio") and module.last_delta_ratio is not None:
            ratios.append(float(module.last_delta_ratio))

    if not ratios:
        return {}

    return {
        f"{prefix}/mean": sum(ratios) / len(ratios),
        f"{prefix}/max": max(ratios),
        f"{prefix}/min": min(ratios),
    }


def build_optimizer_param_groups(model: nn.Module, optim_config: OptimConfig):
    adaptor_modules = get_adaptor_modules(model)
    head_modules = get_head_modules(model)
    adaptor_param_ids = get_module_param_ids(adaptor_modules)
    head_param_ids = get_module_param_ids(head_modules)

    adaptor_decay_params = []
    adaptor_no_decay_params = []
    head_decay_params = []
    head_no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        name_lower = name.lower()
        is_no_decay = (
            name_lower.endswith(".bias")
            or "norm" in name_lower
            or name_lower.endswith(".scale")
            or ".scale" in name_lower
        )

        param_id = id(param)
        is_head = param_id in head_param_ids
        is_adaptor = param_id in adaptor_param_ids or not is_head

        if is_adaptor:
            if is_no_decay:
                adaptor_no_decay_params.append(param)
            else:
                adaptor_decay_params.append(param)
        elif is_head:
            if is_no_decay:
                head_no_decay_params.append(param)
            else:
                head_decay_params.append(param)

    adaptor_lr = optim_config.adaptor_lr or optim_config.lr
    head_lr = optim_config.head_lr or optim_config.lr

    groups = []
    if adaptor_decay_params:
        groups.append({
            "params": adaptor_decay_params,
            "lr": adaptor_lr,
            "weight_decay": optim_config.weight_decay,
        })
    if adaptor_no_decay_params:
        groups.append({
            "params": adaptor_no_decay_params,
            "lr": adaptor_lr,
            "weight_decay": 0.0,
        })
    if head_decay_params:
        groups.append({
            "params": head_decay_params,
            "lr": head_lr,
            "weight_decay": optim_config.weight_decay,
        })
    if head_no_decay_params:
        groups.append({
            "params": head_no_decay_params,
            "lr": head_lr,
            "weight_decay": 0.0,
        })

    if not groups:
        raise ValueError("No trainable parameters found for optimizer.")

    return groups

def set_trainable_params(
    model: nn.Module,
    freeze_backbone: bool = True,
    freeze_linear_head: bool = True,
):
    """
    Keep the original function signature.

    Behavior:
    - freeze_backbone=False:
        train everything.
    - freeze_backbone=True:
        freeze everything first,
        then unfreeze adaptor/adapter modules,
        and optionally unfreeze linear head.
    """

    # Original behavior: train all parameters
    if not freeze_backbone:
        for p in model.parameters():
            p.requires_grad = True
        return model

    # 1. Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # 2. Unfreeze adaptor/adapter modules
    for name, module in model.named_modules():
        name_lower = name.lower()

        is_adaptor_module = (
            "adapter" in name_lower
            or "adaptor" in name_lower
        )

        if is_adaptor_module:
            for p in module.parameters(recurse=True):
                p.requires_grad = True

    # 3. Optionally unfreeze classifier/head modules
    if not freeze_linear_head:
        for name, module in model.named_modules():
            name_lower = name.lower()

            is_linear_head_module = (
                name_lower == "classifier"
                or name_lower.endswith(".classifier")
                or name_lower == "fc"
                or name_lower.endswith(".fc")
                or name_lower == "head"
                or name_lower.endswith(".head")
            )

            if is_linear_head_module:
                for p in module.parameters(recurse=True):
                    p.requires_grad = True

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
        drop_last=True,
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
    adaptor_config = config.adpator_config
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
            target=adaptor_config.target_layers,
            reduction=adaptor_config.reduction,
            use_norm=adaptor_config.use_norm,
            use_trainable_scale=adaptor_config.use_trainable_scale,
            init_scale=adaptor_config.init_scale,
            dropout=adaptor_config.dropout,
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
            target=adaptor_config.target_layers,
            reduction=adaptor_config.reduction,
            use_norm=adaptor_config.use_norm,
            use_trainable_scale=adaptor_config.use_trainable_scale,
            init_scale=adaptor_config.init_scale,
            dropout=adaptor_config.dropout,
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
    
    if config.verbose_model:
        print(model)
        print_model_param_summary(model)
    

    return model
    
def train_one_epoch(
    model: nn.Module,
    train_dataloader,
    val_dataloader,
    optimizer,
    criterion,
    loss_config,
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
    train_scale_reg = 0.0
    train_delta_reg = 0.0

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

        scale_reg = get_adaptor_scale_reg(model)
        delta_reg = get_adaptor_delta_reg(model)
        loss = (
            loss
            + loss_config.lambda_scale * scale_reg
            + loss_config.lambda_delta * delta_reg
        )
        loss_dict["loss_scale_reg"] = scale_reg.detach()
        loss_dict["loss_delta_reg"] = delta_reg.detach()
        loss_dict["loss_total_with_reg"] = loss.detach()

        assert_finite(
            stage="train",
            epoch=epoch,
            step=global_step,
            tensors={
                "logits_clean": logits_clean,
                "logits_pert": logits_pert,
                "feat_clean": feat_clean,
                "feat_pert": feat_pert,
                "loss": loss,
                **loss_dict,
            },
        )

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            assert_finite_grads(
                model,
                stage="train_backward",
                epoch=epoch,
                step=global_step,
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            assert_finite_grads(
                model,
                stage="train_backward",
                epoch=epoch,
                step=global_step,
            )
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
        train_scale_reg += scale_reg.item()
        train_delta_reg += delta_reg.item()
        adaptor_metrics = collect_adaptor_summary_metrics(model)
        adaptor_delta_metrics = collect_adaptor_delta_ratios(model)

        step_metrics = {
            "step/train_loss": loss.item(),
            "step/train_ce_clean": loss_dict["loss_ce_clean"].item(),
            "step/train_ce_pert": loss_dict["loss_ce_pert"].item(),
            "step/train_consistency": loss_dict["loss_consistency"].item(),
            "reg/scale_reg": scale_reg.item(),
            "reg/delta_reg": delta_reg.item(),
            "reg/lambda_scale": loss_config.lambda_scale,
            "reg/lambda_delta": loss_config.lambda_delta,
            "step/train_acc_clean": clean_correct / batch_total,
            "step/train_acc_pert": pert_correct / batch_total,
            "epoch": epoch + 1,
        }
        step_metrics.update(adaptor_metrics)
        step_metrics.update(adaptor_delta_metrics)

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
        "train/scale_reg": train_scale_reg / n_train,
        "train/delta_reg": train_delta_reg / n_train,
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

                assert_finite(
                    stage="val",
                    epoch=epoch,
                    step=global_step,
                    tensors={
                        "logits_clean": logits_clean,
                        "logits_pert": logits_pert,
                        "feat_clean": feat_clean,
                        "feat_pert": feat_pert,
                        "loss": loss,
                        **loss_dict,
                    },
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


def assert_finite(stage: str, epoch: int, step: int, tensors: dict):
    for name, tensor in tensors.items():
        if tensor is None or not torch.is_tensor(tensor):
            continue

        finite_mask = torch.isfinite(tensor)
        if finite_mask.all():
            continue

        bad_count = finite_mask.numel() - finite_mask.sum().item()
        bad_values = (
            tensor.detach()
            .masked_select(~finite_mask)
            .flatten()[:5]
            .cpu()
            .tolist()
        )
        raise FloatingPointError(
            f"Non-finite tensor detected during {stage} "
            f"(epoch={epoch + 1}, step={step}, tensor={name}, "
            f"bad={bad_count}/{finite_mask.numel()}, sample={bad_values})"
        )


def assert_finite_grads(model: nn.Module, stage: str, epoch: int, step: int):
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            continue

        finite_mask = torch.isfinite(grad)
        if finite_mask.all():
            continue

        bad_count = finite_mask.numel() - finite_mask.sum().item()
        bad_values = (
            grad.detach()
            .masked_select(~finite_mask)
            .flatten()[:5]
            .cpu()
            .tolist()
        )
        raise FloatingPointError(
            f"Non-finite gradient detected during {stage} "
            f"(epoch={epoch + 1}, step={step}, param={name}, "
            f"bad={bad_count}/{finite_mask.numel()}, sample={bad_values})"
        )


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
                "adaptor_config": config.adpator_config.__dict__,
                "perturbation": config.perturbation,
                "model_type": config.model_type,
                "freeze_backbone": config.freeze_backbone,
            },
        )

    train_loader, val_loader = build_train_val_dataloaders(config)

    model = build_train_model(config).to(device)

    optimizer = torch.optim.AdamW(
        build_optimizer_param_groups(model, config.optim_config),
        betas=(0.9, 0.999),
    )

    criterion = ConsistencyLoss(
        mode=config.loss_config.mode,
        feature_loss_type=config.loss_config.feature_loss_type,
        lambda_kl=config.loss_config.lambda_kl,
        lambda_feat=config.loss_config.lambda_feat,
        temperature=config.loss_config.temperature,
        detach_teacher=config.loss_config.detach_teacher,
        normalize_feature=config.loss_config.normalize_feature,
        ce_clean_weight=config.loss_config.ce_clean_weight,
        ce_pert_weight=config.loss_config.ce_pert_weight,
        eps=config.loss_config.eps,
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
            loss_config=config.loss_config,
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
