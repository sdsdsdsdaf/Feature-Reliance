"""L0 궤적 probe — 학습 도중 `mean_l0`(z>active_threshold)가 U자로 떨어질 때
진짜 L0(`l0_raw`, z>0)도 같이 떨어지는지 확인한다.

배경: grid의 sae_train_log.png는 `mean_l0`가 epoch 30~100 부근에서 바닥을 찍고
다시 올라가는 U자를 보여준다(trial_0013: 31.1@78 -> 278.2@349). 그런데 저장된
checkpoint들(U자 바닥 근처 epoch)을 `z>0`으로 재보면 7,787~13,519라서, 그 바닥이
진짜 희소성인지 임계 카운팅의 착시인지 아직 모른다. epoch별 weight가 남아 있지
않아 소급 측정이 불가능하므로 한 trial을 다시 돌려 궤적을 직접 찍는다.

설정은 trial_0013과 동일하다(expansion 32, l1_reg 3e-5, lr 5e-5, threshold 0.2,
dec_bias_mode geom). 바닥을 충분히 지나도록 120 epoch만 돌리고, 궤적 전체가
필요하므로 early stopping은 끈다. 학습 손실은 건드리지 않는다 — L1 그대로다.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from SAE_validation import build_backbone_and_transform, build_default_config, build_train_val_loaders
from Utils.SAE_utils import (
    collect_tokens_with_hook,
    compute_b_dec_init_streaming,
    fit_token_normalizer_streaming,
    get_torch_dtype,
    normalize_tokens_inplace,
    save_json,
    train_sae_auto,
)

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=120,
                        help="학습 epoch 수. U자 바닥(원 trial 기준 ep 60~80)을 충분히 지나야 한다")
    parser.add_argument("--geom-iters", type=int, default=10,
                        help="b_dec geometric median 반복 횟수. 반복마다 ViT 전체 패스 + "
                             "100만 토큰 float64 CPU 연산이라 기본값 100은 학습보다 오래 걸린다")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="DataLoader 워커 수. 기본 설정은 0이라 JPEG 디코드가 메인 스레드에서 "
                             "순차 처리돼 ViT 패스가 디코드 바운드가 된다")
    parser.add_argument("--expansion", type=int, default=32, help="SAE 확장 배수 (trial_0013 = 32)")
    parser.add_argument("--l1-reg", type=float, default=3e-5, help="L1 계수 (trial_0013 = 3e-5)")
    parser.add_argument("--lr", type=float, default=5e-5, help="학습률 (trial_0013 = 5e-5)")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/experiments/probe_l0"))
    return parser.parse_args()


def build_probe_config(args):
    """trial_0013의 하이퍼파라미터를 복원하되 epoch 수를 줄이고 early stopping을 끈다."""
    config = build_default_config()
    config.sae.expansion = args.expansion
    config.sae.l1_reg = args.l1_reg
    config.sae.active_threshold = 0.2
    config.sae.dec_bias_mode = "geom"
    config.sae.bias_init_geom_max_iter = args.geom_iters
    config.data_config.num_workers = args.num_workers
    config.optim_config.lr = args.lr
    config.optim_config.epochs = args.epochs
    config.early_stopping.patience = None  # 궤적 전체가 필요하다
    config.early_stopping.save_verbose = True
    config.grid.enabled = False
    # 이 실험은 궤적을 보는 게 목적이라 제약 선정을 끈다(제약이 켜지면 checkpoint가
    # feasible epoch으로 튀어 로그 해석이 헷갈린다). history에는 l0_raw가 그대로 남는다.
    config.sparsity.l0_max = None
    return config.validate()


def plot_trajectory(history, path):
    """mean_l0 / l0_raw / val_nmse 궤적을 한 장에 그린다."""
    epochs = [r["epoch"] for r in history]
    fig, (ax_l0, ax_q) = plt.subplots(1, 2, figsize=(13, 5))

    ax_l0.plot(epochs, [r["l0_raw"] for r in history], label="l0_raw (z>0)", color="crimson")
    ax_l0.plot(epochs, [r["mean_l0"] for r in history], label="mean_l0 (z>0.2)", color="steelblue")
    ax_l0.axhline(150, ls="--", lw=1, color="gray", label="target L0=150 (PatchSAE)")
    ax_l0.set_yscale("log")
    ax_l0.set_xlabel("epoch")
    ax_l0.set_ylabel("active latents / token")
    ax_l0.set_title(f"L0 trajectory (hidden={history[0]['active_total']})")
    ax_l0.legend()
    ax_l0.grid(alpha=0.3)

    ax_q.plot(epochs, [r["normalized_mse"] for r in history], label="val_nmse", color="darkgreen")
    ax_q.plot(epochs, [r["train_l1"] for r in history], label="train_l1", color="darkorange")
    ax_q.set_yscale("log")
    ax_q.set_xlabel("epoch")
    ax_q.set_title("reconstruction / L1 penalty term")
    ax_q.legend()
    ax_q.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    config = build_probe_config(args)
    print(f"probe: expansion={config.sae.expansion} l1_reg={config.sae.l1_reg} lr={config.optim_config.lr} "
          f"epochs={config.optim_config.epochs} geom_iters={config.sae.bias_init_geom_max_iter}")
    torch.set_float32_matmul_precision(config.sae.matmul_precision)

    model, transform, _, _ = build_backbone_and_transform(config)
    _, _, train_loader, val_loader = build_train_val_loaders(config, transform)

    print("Fitting train-token normalizer...")
    token_stats, train_token_count = fit_token_normalizer_streaming(
        model,
        train_loader,
        max_tokens=config.token.max_train_tokens,
        target_block=config.hook.target_block,
        token_scope=config.hook.token_scope,
        device=config.extraction_config.device,
    )
    b_dec_init = compute_b_dec_init_streaming(
        model,
        train_loader,
        token_stats,
        max_tokens=config.token.max_train_tokens,
        target_block=config.hook.target_block,
        token_scope=config.hook.token_scope,
        device=config.extraction_config.device,
        sae_config=config.sae,
    )
    val_tokens, _ = collect_tokens_with_hook(
        model,
        val_loader,
        max_tokens=config.token.max_val_tokens,
        target_block=config.hook.target_block,
        token_scope=config.hook.token_scope,
        device=config.extraction_config.device,
        cache_dtype=get_torch_dtype(config.token.cache_dtype),
        return_labels=True,
    )
    val_tokens = normalize_tokens_inplace(val_tokens.float(), token_stats)
    print(f"val tokens: {tuple(val_tokens.shape)}")

    input_dim = int(token_stats["mean"].shape[1])
    hidden_dim = int(input_dim * config.sae.expansion)
    _, history = train_sae_auto(
        model,
        train_loader,
        val_tokens,
        token_stats,
        input_dim,
        hidden_dim,
        b_dec_init,
        config,
        checkpoint_path=out_dir / "probe_sae_state.pt",
        expected_tokens=train_token_count,
    )

    save_json(out_dir / "history.json", history)
    plot_trajectory(history, out_dir / "l0_trajectory.png")

    lo_raw = min(history, key=lambda r: r["l0_raw"])
    lo_thr = min(history, key=lambda r: r["mean_l0"])
    verdict = {
        "config": {"expansion": config.sae.expansion, "l1_reg": config.sae.l1_reg,
                   "lr": config.optim_config.lr, "epochs": config.optim_config.epochs,
                   "geom_iters": config.sae.bias_init_geom_max_iter, "hidden_dim": hidden_dim},
        "min_l0_raw": {"epoch": lo_raw["epoch"], "l0_raw": lo_raw["l0_raw"],
                       "l0_ratio_raw": lo_raw["l0_ratio_raw"], "val_nmse": lo_raw["normalized_mse"]},
        "min_mean_l0": {"epoch": lo_thr["epoch"], "mean_l0": lo_thr["mean_l0"],
                        "l0_raw": lo_thr["l0_raw"], "val_nmse": lo_thr["normalized_mse"]},
        "last": {"epoch": history[-1]["epoch"], "mean_l0": history[-1]["mean_l0"],
                 "l0_raw": history[-1]["l0_raw"], "val_nmse": history[-1]["normalized_mse"]},
        "any_epoch_under_150_raw": any(r["l0_raw"] <= 150 for r in history),
    }
    save_json(out_dir / "verdict.json", verdict)
    print("\n" + json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
