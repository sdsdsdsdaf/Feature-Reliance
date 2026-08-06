import json
import math
import os
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

from torch import nn

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.auto import tqdm

from Model.SAE import VanillaL1SAE
from Utils.early_stopping import EarlyStopper, unwrap_compiled_model

TQDM_KW = {
    "file": sys.stdout,
    "disable": not sys.stdout.isatty(),
    "dynamic_ncols": True,
}


class TransformDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.ds = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        x, y = item[0], item[1]
        if self.transform is not None:
            # HF ImageNet 로더는 np.ndarray를 준다. timm transform은 PIL/Tensor만 받으므로
            # 여기서 맞춘다 (imagenette는 이미 PIL이라 그대로 통과).
            if isinstance(x, np.ndarray):
                x = Image.fromarray(x)
            x = self.transform(x)
        return x, int(y)


def get_torch_dtype(dtype):
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = str(dtype).replace("torch.", "").lower()
    if key not in aliases:
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    return aliases[key]


def jsonable(obj):
    if is_dataclass(obj):
        return jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.dtype):
        return str(obj).replace("torch.", "")
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, nn.Module):
        return obj.__class__.__name__
    return obj


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(data), f, indent=2, ensure_ascii=False)


def save_figure(fig, path, dpi=200):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def describe_dataset(name, dataset, batch_size=8):
    base_ds = dataset.ds if hasattr(dataset, "ds") else dataset
    print(f"## {name}")
    print(f"num_samples: {len(dataset):,}")
    if hasattr(base_ds, "classes"):
        print(f"num_classes: {len(base_ds.classes)}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    xb, yb = next(iter(loader))
    print(f"first batch xb: {tuple(xb.shape)}, yb: {tuple(yb.shape)}")


@torch.no_grad()
def compute_mean_bias(tokens):
    return tokens.float().mean(dim=0)


@torch.no_grad()
def compute_geometric_median(tokens, max_iter=100, tol=1e-5, eps=1e-8):
    tokens = tokens.float()
    y = tokens.mean(dim=0)
    for _ in tqdm(range(int(max_iter)), desc="geometric median", **TQDM_KW):
        distances = torch.norm(tokens - y, dim=1).clamp_min(eps)
        weights = 1.0 / distances
        y_next = (tokens * weights[:, None]).sum(dim=0) / weights.sum()
        if torch.norm(y_next - y).item() < float(tol):
            return y_next
        y = y_next
    return y


def select_block_tokens(block_output, token_scope="patch", cpu=False):
    if isinstance(block_output, (tuple, list)):
        block_output = block_output[0]
    if block_output.ndim != 3:
        raise ValueError(f"Expected block output [B, tokens, dim], got {tuple(block_output.shape)}.")
    token_scope = str(token_scope).lower()
    if token_scope == "cls":
        tokens = block_output[:, :1]
    elif token_scope == "patch":
        tokens = block_output[:, 1:]
    elif token_scope in {"all", "clspatch"}:
        tokens = block_output
    else:
        raise ValueError("token_scope must be one of: 'cls', 'patch', 'all', 'clspatch'.")
    tokens = tokens.reshape(-1, tokens.shape[-1]).detach().float()
    return tokens.cpu() if cpu else tokens


def normalize_tokens(tokens, stats):
    return (tokens - stats["mean"]) / stats["std"]


def normalize_tokens_inplace(tokens, stats):
    tokens.sub_(stats["mean"])
    tokens.div_(stats["std"])
    return tokens


@torch.no_grad()
def collect_tokens_with_hook(
    model,
    loader,
    max_tokens,
    target_block,
    token_scope,
    device,
    cache_dtype=torch.float16,
    return_labels=False,
):
    model.eval().to(device)
    if target_block < 0 or target_block >= len(model.blocks):
        raise ValueError(f"target_block={target_block} is outside model.blocks length {len(model.blocks)}.")

    token_chunks = []
    label_chunks = []
    captured = {}
    total = 0

    def hook(_module, _inputs, output):
        captured["tokens"] = select_block_tokens(output, token_scope=token_scope, cpu=True)

    handle = model.blocks[target_block].register_forward_hook(hook)
    try:
        for images, labels in tqdm(loader, desc="collecting ViT block tokens", **TQDM_KW):
            captured.clear()
            _ = model(images.to(device))
            tokens = captured["tokens"]
            tokens_per_image = max(1, tokens.shape[0] // images.shape[0])

            if max_tokens is not None:
                remaining = int(max_tokens) - total
                if remaining <= 0:
                    break
                tokens = tokens[:remaining]

            token_chunks.append(tokens.to(cache_dtype if cache_dtype is not None else torch.float32))
            if return_labels:
                expanded_labels = labels.repeat_interleave(tokens_per_image)[: tokens.shape[0]].detach().cpu()
                label_chunks.append(expanded_labels)

            total += tokens.shape[0]
            if max_tokens is not None and total >= int(max_tokens):
                break
    finally:
        handle.remove()

    if not token_chunks:
        raise RuntimeError("No tokens were collected.")

    tokens = torch.cat(token_chunks, dim=0)
    if return_labels:
        return tokens, torch.cat(label_chunks, dim=0)
    return tokens


@torch.no_grad()
def iter_token_batches_with_hook(model, loader, max_tokens, target_block, token_scope, device):
    model.eval().to(device)
    captured = {}
    emitted = 0

    def hook(_module, _inputs, output):
        captured["tokens"] = select_block_tokens(output, token_scope=token_scope, cpu=True)

    handle = model.blocks[target_block].register_forward_hook(hook)
    try:
        for images, _labels in loader:
            captured.clear()
            _ = model(images.to(device))
            tokens = captured["tokens"]
            if max_tokens is not None:
                remaining = int(max_tokens) - emitted
                if remaining <= 0:
                    break
                tokens = tokens[:remaining]
            emitted += tokens.shape[0]
            yield tokens
            if max_tokens is not None and emitted >= int(max_tokens):
                break
    finally:
        handle.remove()


@torch.no_grad()
def fit_token_normalizer_streaming(model, loader, max_tokens, target_block, token_scope, device, eps=1e-6):
    token_sum = None
    token_sq_sum = None
    total = 0

    for tokens in tqdm(
        iter_token_batches_with_hook(model, loader, max_tokens, target_block, token_scope, device),
        desc="fitting SAE token normalizer",
        **TQDM_KW,
    ):
        tokens = tokens.float()
        if token_sum is None:
            token_sum = torch.zeros(1, tokens.shape[-1], dtype=torch.float64)
            token_sq_sum = torch.zeros(1, tokens.shape[-1], dtype=torch.float64)
        token_sum += tokens.double().sum(dim=0, keepdim=True)
        token_sq_sum += tokens.double().pow(2).sum(dim=0, keepdim=True)
        total += tokens.shape[0]
        del tokens

    if total == 0:
        raise RuntimeError("No tokens were seen while fitting normalizer.")
    mean = (token_sum / total).float()
    var = (token_sq_sum / total - mean.double().pow(2)).clamp_min(float(eps) ** 2).float()
    return {"mean": mean, "std": var.sqrt().clamp_min(eps)}, total


@torch.no_grad()
def compute_b_dec_init_streaming(model, loader, token_stats, max_tokens, target_block, token_scope, device, sae_config):
    mode = str(sae_config.dec_bias_mode).lower()
    dim = token_stats["mean"].shape[-1]

    if mode == "zero":
        return torch.zeros(dim)

    if mode == "mean":
        token_sum = torch.zeros(dim, dtype=torch.float64)
        total = 0
        for tokens in tqdm(
            iter_token_batches_with_hook(model, loader, max_tokens, target_block, token_scope, device),
            desc="streaming b_dec mean",
            **TQDM_KW,
        ):
            tokens = normalize_tokens_inplace(tokens.float(), token_stats)
            token_sum += tokens.double().sum(dim=0)
            total += tokens.shape[0]
        if total == 0:
            raise RuntimeError("No tokens were seen while fitting b_dec mean.")
        return (token_sum / total).float()

    if mode != "geom":
        raise ValueError(f"Unknown dec_bias_mode: {sae_config.dec_bias_mode}")

    y = torch.zeros(dim, dtype=torch.float32)
    for _ in tqdm(
        range(int(sae_config.bias_init_geom_max_iter)),
        desc="streaming b_dec geometric median",
        **TQDM_KW,
    ):
        numerator = torch.zeros(dim, dtype=torch.float64)
        denominator = torch.zeros((), dtype=torch.float64)
        for tokens in iter_token_batches_with_hook(model, loader, max_tokens, target_block, token_scope, device):
            tokens = normalize_tokens_inplace(tokens.float(), token_stats)
            distances = torch.norm(tokens.double() - y.double().unsqueeze(0), dim=1).clamp_min(1e-8)
            weights = 1.0 / distances
            numerator += (tokens.double() * weights[:, None]).sum(dim=0)
            denominator += weights.sum()
        y_next = (numerator / denominator.clamp_min(1e-8)).float()
        if torch.norm(y_next - y).item() < float(sae_config.bias_init_geom_tol):
            return y_next
        y = y_next
    return y


def _amp_enabled(device, use_amp):
    return bool(use_amp and str(device).startswith("cuda") and torch.cuda.is_available())


def _make_grad_scaler(device, use_amp, amp_dtype):
    enabled = _amp_enabled(device, use_amp) and amp_dtype == torch.float16
    return torch.amp.GradScaler("cuda", enabled=enabled)


def _assert_finite_tensor(name, tensor):
    if tensor is not None and not torch.isfinite(tensor).all().item():
        raise FloatingPointError(f"Non-finite value detected in {name}.")


def _assert_finite_grads(model):
    for name, param in model.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all().item():
            raise FloatingPointError(f"Non-finite gradient detected in {name}.")


def _run_sae_step(sae, optimizer, xb, scaler, sae_config, optim_config, device, scheduler=None, token_std=None):
    """SGD 한 스텝. xb는 정규화된 활성이다.

    sae_config.recon_space="raw"이면 재구성 오차를 denorm 공간에서 잰다. denorm을 명시적으로
    계산할 필요는 없다 — mu가 상쇄되므로
        (x_hat*sigma + mu) - (x*sigma + mu) = (x_hat - x)*sigma
    이고, 큰 값끼리 빼면서 정밀도를 잃지도 않는다."""
    amp_dtype = get_torch_dtype(sae_config.amp_dtype)
    amp_enabled = _amp_enabled(device, optim_config.use_amp)
    autocast_device = "cuda" if str(device).startswith("cuda") else "cpu"
    recon_space = str(getattr(sae_config, "recon_space", "norm")).lower()
    if recon_space not in ("norm", "raw"):
        raise ValueError(f"Unknown sae.recon_space: {recon_space!r}. 'norm' 또는 'raw'.")
    if recon_space == "raw" and token_std is None:
        raise ValueError("sae.recon_space='raw'는 token_std가 필요하다 (token_stats['std']를 device에 올려 넘길 것).")

    with torch.autocast(device_type=autocast_device, dtype=amp_dtype, enabled=amp_enabled):
        x_hat, z = sae(xb)
        if recon_space == "raw":
            recon_loss = ((x_hat - xb) * token_std).pow(2).mean()
        else:
            recon_loss = F.mse_loss(x_hat, xb)
        l1_loss = z.abs().sum(dim=-1).mean()
        loss = recon_loss + float(sae_config.l1_reg) * l1_loss

    if sae_config.check_finite:
        _assert_finite_tensor("SAE forward loss", loss)
        _assert_finite_tensor("SAE reconstruction loss", recon_loss)
        _assert_finite_tensor("SAE L1 loss", l1_loss)

    optimizer.zero_grad(set_to_none=True)
    if scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()

    if sae_config.check_finite:
        _assert_finite_grads(sae)

    unwrap_compiled_model(sae).remove_gradient_parallel_to_decoder_directions()
    if scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    unwrap_compiled_model(sae).set_decoder_norm_to_unit_norm()
    return loss, recon_loss, l1_loss, x_hat, z


def format_sae_epoch_log(row, hidden_dim=None, threshold=0.2, patience=None):
    active_threshold = row.get("active_threshold", threshold)
    active_mean_count = row.get("active_mean_count", row["mean_l0"])
    active_total = row.get("active_total", hidden_dim)
    active_ratio = row.get("active_ratio")
    if active_ratio is None and active_total is not None:
        active_ratio = active_mean_count / max(1, int(active_total))
    active_total_text = str(int(active_total)) if active_total is not None else "?"
    active_ratio_text = f"{active_ratio * 100:.2f}%" if active_ratio is not None else "n/a"
    time_text = f" | epoch={float(row['epoch_seconds']):.2f}s" if "epoch_seconds" in row else ""
    early_stop_text = ""
    if "best_val_nmse" in row and "early_stop_bad_epochs" in row:
        best_active_text = ""
        if "best_active_mean_count" in row:
            best_active_text = f" | best_active={float(row['best_active_mean_count']):.2f}"
        early_stop_text = (
            f" | best_val_nmse={float(row['best_val_nmse']):.4f}"
            f"{best_active_text}"
            f" | no_improve={int(row['early_stop_bad_epochs'])}/{patience}"
        )
    return (
        f"epoch {int(row['epoch']):03d} | "
        f"train_loss={row['train_loss']:.6f} | "
        f"train_mse={row['train_mse']:.6f} | "
        f"train_l1={row['train_l1']:.6f} | "
        f"val_mse={row['mse']:.6f} | "
        f"val_nmse={row['normalized_mse']:.4f} | "
        f"active>{active_threshold:g}={active_mean_count:.2f}/{active_total_text} ({active_ratio_text})"
        f"{time_text}{early_stop_text}"
    )


def _dtype_element_size(dtype):
    dtype = dtype or torch.float32
    return torch.empty((), dtype=dtype).element_size()


def estimate_token_cache_bytes(num_tokens, input_dim, dtype):
    return int(num_tokens) * int(input_dim) * _dtype_element_size(dtype)


def get_available_cpu_memory_bytes():
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def format_gib(num_bytes):
    if num_bytes is None:
        return "unknown"
    return f"{float(num_bytes) / (1024 ** 3):.2f} GiB"


def choose_token_source_mode(token_config, expected_tokens, input_dim):
    requested_mode = str(token_config.source_mode).lower()
    if requested_mode not in {"auto", "cache", "stream"}:
        raise ValueError("token.source_mode must be one of: 'auto', 'cache', 'stream'.")
    if requested_mode in {"cache", "stream"}:
        return requested_mode
    if expected_tokens is None:
        print("Token source auto decision: expected_tokens is unknown; using stream mode.")
        return "stream"

    dtype = get_torch_dtype(token_config.cache_dtype) or torch.float32
    cache_bytes = estimate_token_cache_bytes(expected_tokens, input_dim, dtype)
    peak_bytes = int(cache_bytes * float(token_config.cache_build_peak_factor))
    max_cpu_bytes = int(float(token_config.cache_max_cpu_gib) * (1024 ** 3))
    min_free_after_bytes = int(float(token_config.cache_min_free_cpu_gib_after_build) * (1024 ** 3))
    available_cpu_bytes = get_available_cpu_memory_bytes()
    fits_policy = peak_bytes <= max_cpu_bytes
    fits_available = available_cpu_bytes is None or peak_bytes + min_free_after_bytes <= available_cpu_bytes

    print("Token source auto decision:")
    print(f"  expected_tokens              : {int(expected_tokens):,}")
    print(f"  input_dim                    : {int(input_dim)}")
    print(f"  cache dtype                  : {dtype}")
    print(f"  estimated cache              : {format_gib(cache_bytes)}")
    print(f"  estimated build peak         : {format_gib(peak_bytes)}")
    print(f"  max allowed build peak       : {float(token_config.cache_max_cpu_gib):.2f} GiB")
    print(f"  min free CPU after build     : {float(token_config.cache_min_free_cpu_gib_after_build):.2f} GiB")
    print(f"  available CPU memory         : {format_gib(available_cpu_bytes)}")
    print(f"  fits policy / available      : {fits_policy} / {fits_available}")
    return "cache" if fits_policy and fits_available else "stream"


def _new_sae(input_dim, hidden_dim, b_dec_init, config, device):
    return VanillaL1SAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        b_dec_init=b_dec_init,
        dec_bias_mode=config.sae.dec_bias_mode,
    ).to(device)


def _compile_sae_if_needed(sae, sae_config):
    if sae_config.model_compile and hasattr(torch, "compile"):
        print("Compiling SAE model...")
        return torch.compile(sae)
    return sae


def _make_optimizer(sae, optim_config):
    return torch.optim.AdamW(sae.parameters(), lr=float(optim_config.lr), weight_decay=float(optim_config.weight_decay))


def _make_scheduler(optimizer, optim_config):
    """constant-with-warmup 스케줄러. warmup 구간에서 lr을 선형으로 올리고 이후 고정한다.

    PatchSAE 참조 구현의 'constantwithwarmup'과 같은 식이다
    (`min(1.0, (step + 1) / warmup_steps)`). lr_warmup_steps가 0이면 None을 돌려
    스케줄러 없이 기존 동작을 유지한다."""
    warmup_steps = int(getattr(optim_config, "lr_warmup_steps", 0) or 0)
    if warmup_steps <= 0:
        return None
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps)
    )


def _raw_space_stats(config, token_stats, device):
    """recon_space='raw'일 때 학습/검증이 쓸 device 위 통계를 한 번만 만든다.

    반환 (token_std, eval_stats). 'norm'이면 (None, None)이고 그러면 손실도 지표도
    기존 정규화 공간 동작 그대로다. 스텝마다 .to(device)를 다시 하지 않으려고 학습 함수
    시작 지점에서 한 번 호출한다."""
    if str(getattr(config.sae, "recon_space", "norm")).lower() != "raw":
        return None, None
    stats = {
        "mean": token_stats["mean"].to(device).float(),
        "std": token_stats["std"].to(device).float(),
    }
    return stats["std"], stats


def _build_eval_row(sae, val_tokens, hidden_dim, eval_index, step, tokens_seen, train_loss_sum, train_mse_sum, train_l1_sum, train_rows, elapsed, config, token_stats=None):
    """스텝 기준 검증 결과 한 줄. epoch 모드의 row와 같은 스키마를 유지한다.

    `epoch` 키에는 검증 회차를 넣는다 — EarlyStopper와 plot_sae_training_history가
    그 키를 x축/식별자로 쓰고 있어서 이름을 바꾸면 둘 다 깨진다. 실제 진행도는
    step/tokens_seen에 따로 담는다."""
    row = _build_epoch_row(
        sae, val_tokens, hidden_dim, eval_index,
        train_loss_sum, train_mse_sum, train_l1_sum, train_rows, elapsed, config,
        token_stats=token_stats,
    )
    row["step"] = int(step)
    row["tokens_seen"] = int(tokens_seen)
    return row


def train_sae_token_budget(
    model,
    loader,
    val_tokens,
    token_stats,
    input_dim,
    hidden_dim,
    b_dec_init,
    config,
    checkpoint_path,
):
    """스트림을 한 번만 흘리며 total_train_tokens를 채울 때까지 학습한다.

    같은 활성을 두 번 쓰지 않는다 — 배치를 뽑아 SGD 한 스텝 하고 버린다. 그래서
    ViT forward가 토큰당 정확히 1회이고(에폭 반복으로 인한 재계산 없음) 활성을
    보관하지 않으므로 RAM 상한도 없다. 검증은 epoch이 아니라 eval_every_steps마다 한다.

    예산이 데이터셋 한 바퀴보다 크면 loader를 다시 돈다."""
    device = config.extraction_config.device
    budget = config.schedule.total_train_tokens
    if budget is None:
        raise ValueError("schedule.mode='token_budget'이면 schedule.total_train_tokens가 필요하다.")
    budget = int(budget)
    eval_every = max(1, int(config.schedule.eval_every_steps))

    sae = _compile_sae_if_needed(_new_sae(input_dim, hidden_dim, b_dec_init, config, device), config.sae)
    optimizer = _make_optimizer(sae, config.optim_config)
    scaler = _make_grad_scaler(device, config.optim_config.use_amp, get_torch_dtype(config.sae.amp_dtype))
    scheduler = _make_scheduler(optimizer, config.optim_config)
    token_std, eval_stats = _raw_space_stats(config, token_stats, device)
    history = []
    early_stopper = EarlyStopper(
        patience=config.early_stopping.patience,
        eps=config.early_stopping.eps,
        checkpoint_path=checkpoint_path,
        metric_name=config.early_stopping.metric_name,
        l0_metric=config.sparsity.metric,
        l0_max=config.sparsity.l0_max,
        verbose=config.early_stopping.save_verbose,
    )

    print(f"SAE token-budget training: {budget:,} tokens, eval every {eval_every} steps")
    n_tokens = n_steps = eval_index = 0
    train_loss_sum = train_mse_sum = train_l1_sum = 0.0
    train_rows = 0
    window_start = time.perf_counter()
    carry = None
    should_stop = done = False

    with tqdm(total=budget, desc="SAE token budget", unit="tok", **TQDM_KW) as pbar:
        while not done:
            sae.train()
            for token_batch in iter_token_batches_with_hook(
                model, loader, None, config.hook.target_block, config.hook.token_scope, device
            ):
                token_batch = normalize_tokens_inplace(token_batch.float(), token_stats)
                if carry is not None:
                    token_batch = torch.cat([carry, token_batch], dim=0)
                    carry = None
                token_batch = token_batch[torch.randperm(token_batch.shape[0])]
                full_count = (token_batch.shape[0] // config.sae.batch_size) * config.sae.batch_size

                for start in range(0, full_count, config.sae.batch_size):
                    xb = token_batch[start : start + config.sae.batch_size].float().to(device)
                    loss, recon_loss, l1_loss, x_hat, z = _run_sae_step(
                        sae, optimizer, xb, scaler, config.sae, config.optim_config, device, scheduler, token_std
                    )
                    train_loss_sum += float(loss.item()) * xb.shape[0]
                    train_mse_sum += float(recon_loss.item()) * xb.shape[0]
                    train_l1_sum += float(l1_loss.item()) * xb.shape[0]
                    train_rows += xb.shape[0]
                    n_tokens += xb.shape[0]
                    n_steps += 1
                    pbar.update(xb.shape[0])
                    del xb, x_hat, z, loss, recon_loss, l1_loss

                    if n_steps % eval_every == 0 or n_tokens >= budget:
                        eval_index += 1
                        row = _build_eval_row(
                            sae, val_tokens, hidden_dim, eval_index, n_steps, n_tokens,
                            train_loss_sum, train_mse_sum, train_l1_sum, train_rows,
                            time.perf_counter() - window_start, config, eval_stats,
                        )
                        _, should_stop = early_stopper.step(row["normalized_mse"], sae, row)
                        history.append(row)
                        print(format_sae_epoch_log(row, hidden_dim=hidden_dim, threshold=config.sae.active_threshold, patience=config.early_stopping.patience))
                        train_loss_sum = train_mse_sum = train_l1_sum = 0.0
                        train_rows = 0
                        window_start = time.perf_counter()
                        sae.train()

                    if should_stop or n_tokens >= budget:
                        done = True
                        break

                if full_count < token_batch.shape[0]:
                    carry = token_batch[full_count:].detach().cpu()
                del token_batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if done:
                    break

            if not done and n_tokens == 0:
                raise RuntimeError("토큰이 하나도 나오지 않았다 — loader가 비어 있는지 확인할 것.")

    if should_stop:
        print(
            f"Early stopping at step {n_steps} ({n_tokens:,} tokens): val_nmse did not improve by at least "
            f"{config.early_stopping.eps:g} for {early_stopper.bad_epochs} evaluations."
        )

    best_checkpoint = early_stopper.load_best(sae, map_location=device)
    print(
        f"Loaded best SAE checkpoint for downstream validation: "
        f"eval={best_checkpoint['epoch']}, step={best_checkpoint['row'].get('step')}, "
        f"val_nmse={best_checkpoint[config.early_stopping.metric_name]:.6f}"
    )
    return sae, history


def train_sae_streaming(
    model,
    loader,
    val_tokens,
    token_stats,
    input_dim,
    hidden_dim,
    b_dec_init,
    config,
    checkpoint_path,
    expected_tokens=None,
):
    device = config.extraction_config.device
    sae = _compile_sae_if_needed(_new_sae(input_dim, hidden_dim, b_dec_init, config, device), config.sae)
    optimizer = _make_optimizer(sae, config.optim_config)
    scaler = _make_grad_scaler(device, config.optim_config.use_amp, get_torch_dtype(config.sae.amp_dtype))
    scheduler = _make_scheduler(optimizer, config.optim_config)
    token_std, eval_stats = _raw_space_stats(config, token_stats, device)
    history = []
    early_stopper = EarlyStopper(
        patience=config.early_stopping.patience,
        eps=config.early_stopping.eps,
        checkpoint_path=checkpoint_path,
        metric_name=config.early_stopping.metric_name,
        l0_metric=config.sparsity.metric,
        l0_max=config.sparsity.l0_max,
        verbose=config.early_stopping.save_verbose,
    )
    expected_steps = math.ceil(int(expected_tokens) / config.sae.batch_size) if expected_tokens is not None else None
    print(f"SAE train tokens per epoch: {expected_tokens if expected_tokens is not None else 'unknown'}")

    for epoch in range(1, int(config.optim_config.epochs) + 1):
        epoch_start_time = time.perf_counter()
        sae.train()
        train_loss_sum = train_mse_sum = train_l1_sum = 0.0
        train_rows = 0
        carry = None

        with tqdm(
            total=expected_steps,
            desc=f"SAE epoch {epoch}/{config.optim_config.epochs}",
            leave=False,
            **TQDM_KW,
        ) as pbar:
            for token_batch in iter_token_batches_with_hook(
                model,
                loader,
                config.token.max_train_tokens,
                config.hook.target_block,
                config.hook.token_scope,
                device,
            ):
                token_batch = normalize_tokens_inplace(token_batch.float(), token_stats)
                if carry is not None:
                    token_batch = torch.cat([carry, token_batch], dim=0)
                    carry = None
                perm = torch.randperm(token_batch.shape[0])
                token_batch = token_batch[perm]
                full_count = (token_batch.shape[0] // config.sae.batch_size) * config.sae.batch_size

                for start in range(0, full_count, config.sae.batch_size):
                    xb = token_batch[start : start + config.sae.batch_size].float().to(device)
                    loss, recon_loss, l1_loss, x_hat, z = _run_sae_step(
                        sae, optimizer, xb, scaler, config.sae, config.optim_config, device, scheduler, token_std
                    )
                    train_loss_sum += float(loss.item()) * xb.shape[0]
                    train_mse_sum += float(recon_loss.item()) * xb.shape[0]
                    train_l1_sum += float(l1_loss.item()) * xb.shape[0]
                    train_rows += xb.shape[0]
                    if pbar.total is not None:
                        pbar.update(1)
                    del xb, x_hat, z, loss, recon_loss, l1_loss

                if full_count < token_batch.shape[0]:
                    carry = token_batch[full_count:].detach().cpu()
                del token_batch, perm
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if carry is not None and carry.numel() > 0:
                xb = carry.float().to(device)
                loss, recon_loss, l1_loss, x_hat, z = _run_sae_step(
                    sae, optimizer, xb, scaler, config.sae, config.optim_config, device, scheduler, token_std
                )
                train_loss_sum += float(loss.item()) * xb.shape[0]
                train_mse_sum += float(recon_loss.item()) * xb.shape[0]
                train_l1_sum += float(l1_loss.item()) * xb.shape[0]
                train_rows += xb.shape[0]
                if pbar.total is not None:
                    pbar.update(1)
                del xb, x_hat, z, loss, recon_loss, l1_loss, carry

        row = _build_epoch_row(
            sae,
            val_tokens,
            hidden_dim,
            epoch,
            train_loss_sum,
            train_mse_sum,
            train_l1_sum,
            train_rows,
            time.perf_counter() - epoch_start_time,
            config,
            token_stats=eval_stats,
        )
        _, should_stop = early_stopper.step(row["normalized_mse"], sae, row)
        history.append(row)
        print(format_sae_epoch_log(row, hidden_dim=hidden_dim, threshold=config.sae.active_threshold, patience=config.early_stopping.patience))
        if should_stop:
            print(
                f"Early stopping at epoch {epoch}: val_nmse did not improve by at least "
                f"{config.early_stopping.eps:g} for {early_stopper.bad_epochs} epochs."
            )
            break

    best_checkpoint = early_stopper.load_best(sae, map_location=device)
    print(
        f"Loaded best SAE checkpoint for downstream validation: "
        f"epoch={best_checkpoint['epoch']}, val_nmse={best_checkpoint[config.early_stopping.metric_name]:.6f}"
    )
    return sae, history


def train_sae_cached(
    model,
    loader,
    val_tokens,
    token_stats,
    input_dim,
    hidden_dim,
    b_dec_init,
    config,
    checkpoint_path,
):
    device = config.extraction_config.device
    print("Collecting train tokens once for cached SAE training...")
    train_tokens = collect_tokens_with_hook(
        model,
        loader,
        config.token.max_train_tokens,
        config.hook.target_block,
        config.hook.token_scope,
        device,
        cache_dtype=get_torch_dtype(config.token.cache_dtype),
    )
    train_tokens = normalize_tokens_inplace(train_tokens.float(), token_stats)
    cache_dtype = get_torch_dtype(config.token.cache_dtype)
    if cache_dtype is not None:
        train_tokens = train_tokens.to(cache_dtype)
    train_tokens = train_tokens.cpu()

    token_loader = DataLoader(
        TensorDataset(train_tokens),
        batch_size=config.sae.batch_size,
        shuffle=True,
        num_workers=config.token.cache_num_workers,
        pin_memory=str(device).startswith("cuda"),
        drop_last=False,
    )

    sae = _compile_sae_if_needed(_new_sae(input_dim, hidden_dim, b_dec_init, config, device), config.sae)
    optimizer = _make_optimizer(sae, config.optim_config)
    scaler = _make_grad_scaler(device, config.optim_config.use_amp, get_torch_dtype(config.sae.amp_dtype))
    scheduler = _make_scheduler(optimizer, config.optim_config)
    token_std, eval_stats = _raw_space_stats(config, token_stats, device)
    history = []
    early_stopper = EarlyStopper(
        patience=config.early_stopping.patience,
        eps=config.early_stopping.eps,
        checkpoint_path=checkpoint_path,
        metric_name=config.early_stopping.metric_name,
        l0_metric=config.sparsity.metric,
        l0_max=config.sparsity.l0_max,
        verbose=config.early_stopping.save_verbose,
    )
    print(f"SAE cached train tokens per epoch: {len(train_tokens):,}")

    for epoch in range(1, int(config.optim_config.epochs) + 1):
        epoch_start_time = time.perf_counter()
        sae.train()
        train_loss_sum = train_mse_sum = train_l1_sum = 0.0
        train_rows = 0

        for (xb_cpu,) in tqdm(
            token_loader,
            desc=f"SAE cached epoch {epoch}/{config.optim_config.epochs}",
            leave=False,
            **TQDM_KW,
        ):
            xb = xb_cpu.to(device, non_blocking=True).float()
            loss, recon_loss, l1_loss, x_hat, z = _run_sae_step(
                sae, optimizer, xb, scaler, config.sae, config.optim_config, device, scheduler, token_std
            )
            train_loss_sum += float(loss.item()) * xb.shape[0]
            train_mse_sum += float(recon_loss.item()) * xb.shape[0]
            train_l1_sum += float(l1_loss.item()) * xb.shape[0]
            train_rows += xb.shape[0]
            del xb, x_hat, z, loss, recon_loss, l1_loss

        row = _build_epoch_row(
            sae,
            val_tokens,
            hidden_dim,
            epoch,
            train_loss_sum,
            train_mse_sum,
            train_l1_sum,
            train_rows,
            time.perf_counter() - epoch_start_time,
            config,
            token_stats=eval_stats,
        )
        _, should_stop = early_stopper.step(row["normalized_mse"], sae, row)
        history.append(row)
        print(format_sae_epoch_log(row, hidden_dim=hidden_dim, threshold=config.sae.active_threshold, patience=config.early_stopping.patience))
        if should_stop:
            print(
                f"Early stopping at epoch {epoch}: val_nmse did not improve by at least "
                f"{config.early_stopping.eps:g} for {early_stopper.bad_epochs} epochs."
            )
            break

    best_checkpoint = early_stopper.load_best(sae, map_location=device)
    print(
        f"Loaded best SAE checkpoint for downstream validation: "
        f"epoch={best_checkpoint['epoch']}, val_nmse={best_checkpoint[config.early_stopping.metric_name]:.6f}"
    )
    return sae, history


def train_sae_auto(model, loader, val_tokens, token_stats, input_dim, hidden_dim, b_dec_init, config, checkpoint_path, expected_tokens=None):
    schedule_mode = str(getattr(config, "schedule", None) and config.schedule.mode or "epoch").lower()
    if schedule_mode == "token_budget":
        # 단일 통과라 활성을 보관하지 않는다 — cache/stream 선택 자체가 필요 없다.
        return train_sae_token_budget(
            model, loader, val_tokens, token_stats, input_dim, hidden_dim, b_dec_init, config, checkpoint_path
        )
    if schedule_mode != "epoch":
        raise ValueError(f"schedule.mode must be 'epoch' or 'token_budget', got {schedule_mode!r}.")

    token_count_for_decision = expected_tokens if expected_tokens is not None else config.token.max_train_tokens
    mode = choose_token_source_mode(config.token, token_count_for_decision, input_dim)
    print(f"Token source mode selected: {mode}")
    if mode == "cache":
        return train_sae_cached(model, loader, val_tokens, token_stats, input_dim, hidden_dim, b_dec_init, config, checkpoint_path)
    return train_sae_streaming(
        model,
        loader,
        val_tokens,
        token_stats,
        input_dim,
        hidden_dim,
        b_dec_init,
        config,
        checkpoint_path,
        expected_tokens=expected_tokens,
    )


def _build_epoch_row(sae, val_tokens, hidden_dim, epoch, train_loss_sum, train_mse_sum, train_l1_sum, train_rows, epoch_seconds, config, token_stats=None):
    val_metrics = evaluate_sae_tokens(
        sae,
        val_tokens,
        batch_size=config.sae.batch_size,
        threshold=config.sae.active_threshold,
        device=config.extraction_config.device,
        token_stats=token_stats,
    )
    active_mean_count = val_metrics["mean_l0"]
    active_total = int(hidden_dim)
    return {
        "epoch": epoch,
        "train_loss": train_loss_sum / max(1, train_rows),
        "train_mse": train_mse_sum / max(1, train_rows),
        "train_l1": train_l1_sum / max(1, train_rows),
        "train_rows": train_rows,
        **val_metrics,
        "active_threshold": config.sae.active_threshold,
        "active_mean_count": active_mean_count,
        "active_total": active_total,
        "active_ratio": active_mean_count / max(1, active_total),
        "epoch_seconds": epoch_seconds,
    }


@torch.no_grad()
def evaluate_sae_tokens(sae, tokens_norm, batch_size, threshold, device, token_stats=None):
    """검증 지표. tokens_norm은 정규화된 활성이다.

    token_stats(mean/std, device 위)를 주면 재구성 오차와 cosine을 **denorm 공간**에서
    잰다 — 학습 손실이 recon_space='raw'일 때 지표도 같은 공간이어야 체크포인트 선정과
    early stopping이 최적화 대상과 어긋나지 않는다. None이면 기존대로 정규화 공간이다.

    raw 분산은 따로 한 바퀴 돌지 않고 스트리밍으로 모은다: tokens_norm은 차원별 평균이 0이라
    (x_norm*sigma)가 곧 차원별 중심화된 raw 편차다."""
    sae.eval().to(device)
    total_sse = total_cosine = total_l0 = total_l0_raw = total_rows = total_elements = 0.0
    total_var_raw = 0.0
    raw_space = token_stats is not None
    if raw_space:
        sd = token_stats["std"].to(device).float()
        mu = token_stats["mean"].to(device).float()
    hidden_dim = None
    for start in tqdm(range(0, tokens_norm.shape[0], batch_size), leave=False, desc="Eval", **TQDM_KW):
        xb = tokens_norm[start : start + batch_size].float().to(device, non_blocking=True)
        x_hat, z = sae(xb)
        if raw_space:
            xr = xb * sd + mu
            xhr = x_hat * sd + mu
            total_sse += float((xr - xhr).square().sum().item())
            total_var_raw += float((xb * sd).square().sum().item())
            total_cosine += float(F.cosine_similarity(xr, xhr, dim=1).sum().item())
            del xr, xhr
        else:
            total_sse += float((xb - x_hat).square().sum().item())
            total_cosine += float(F.cosine_similarity(xb, x_hat, dim=1).sum().item())
        total_l0 += float((z > threshold).sum().item())
        # ReLU 출력이라 z >= 0 이고, z > 0 은 z != 0 과 같다. count_nonzero는 bool 임시
        # 텐서를 만들지 않고 바로 리듀스하므로 (z > 0) 대비 검증 피크 VRAM이 늘지 않는다
        # (hidden 49152, batch 7096이면 그 임시 하나가 0.3 GiB다).
        total_l0_raw += float(torch.count_nonzero(z).item())
        if hidden_dim is None:
            hidden_dim = int(z.shape[-1])
        total_rows += xb.shape[0]
        total_elements += xb.numel()
        del xb, x_hat, z

    mse = total_sse / max(1, total_elements)
    if raw_space:
        # 분모도 raw 공간의 (차원별 중심화된) 분산이라야 FVU가 된다
        variance = total_var_raw / max(1, total_elements)
    else:
        variance = float(tokens_norm.float().var(unbiased=False).item())
    l0_raw = total_l0_raw / max(1, total_rows)
    return {
        "mse": mse,
        "normalized_mse": mse / max(variance, 1e-12),
        "val_nmse": mse / max(variance, 1e-12),
        "recon_space": "raw" if raw_space else "norm",
        "cosine": total_cosine / max(1, total_rows),
        "mean_l0": total_l0 / max(1, total_rows),
        "l0_raw": l0_raw,
        "l0_ratio_raw": l0_raw / max(1, hidden_dim or 1),
    }


@torch.no_grad()
def latent_frequency(sae, tokens_norm, batch_size, threshold, device):
    sae.eval().to(device)
    base_sae = unwrap_compiled_model(sae)
    hidden_dim = base_sae.hidden_dim
    counts = torch.zeros(hidden_dim)
    total = 0
    for start in tqdm(
        range(0, tokens_norm.shape[0], batch_size),
        leave=False,
        desc="latent frequency",
        **TQDM_KW,
    ):
        xb = tokens_norm[start : start + batch_size].float().to(device)
        z = base_sae.encode(xb).detach().cpu()
        counts += (z > threshold).float().sum(dim=0)
        total += z.shape[0]
        del xb, z
    return counts / max(1, total)


def plot_sae_training_history(history, hidden_dim=None, active_threshold=0.2, figsize=(12, 8)):
    if not history:
        raise ValueError('history is empty.')

    epochs = np.array([row['epoch'] for row in history], dtype=float)
    train_loss = np.array([row['train_loss'] for row in history], dtype=float)
    train_mse = np.array([row['train_mse'] for row in history], dtype=float)
    val_mse = np.array([row['mse'] for row in history], dtype=float)
    train_l1 = np.array([row['train_l1'] for row in history], dtype=float)
    val_nmse = np.array([row['normalized_mse'] for row in history], dtype=float)
    one_minus_val_cosine = 1.0 - np.array([row['cosine'] for row in history], dtype=float)
    mean_l0 = np.array([row['mean_l0'] for row in history], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax_loss, ax_l1, ax_quality, ax_active = axes.reshape(-1)

    ax_loss.plot(epochs, train_loss, label='train_loss')
    ax_loss.plot(epochs, train_mse, label='train_mse')
    ax_loss.plot(epochs, val_mse, label='val_mse')
    ax_loss.set_title('Loss / MSE')
    ax_loss.set_xlabel('epoch')
    ax_loss.set_ylabel('value')
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    ax_l1.plot(epochs, train_l1)
    ax_l1.set_title('L1 activation penalty term')
    ax_l1.set_xlabel('epoch')
    ax_l1.set_ylabel('mean |z|')
    ax_l1.grid(True, alpha=0.3)

    ax_quality.plot(epochs, val_nmse, label='val_nmse')
    ax_quality.plot(epochs, one_minus_val_cosine, label='1 - val_cosine')
    ax_quality.set_title('Validation quality')
    ax_quality.set_xlabel('epoch')
    ax_quality.set_ylabel('value')
    ax_quality.grid(True, alpha=0.3)
    ax_quality.legend()

    if hidden_dim is not None:
        active_values = mean_l0 / max(1, int(hidden_dim))
        active_ylabel = 'active latent ratio'
    else:
        active_values = mean_l0
        active_ylabel = 'mean active latents'
    ax_active.plot(epochs, active_values)
    ax_active.set_title(f'Active latents > {active_threshold:g}')
    ax_active.set_xlabel('epoch')
    ax_active.set_ylabel(active_ylabel)
    ax_active.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
