"""학습된 trial 체크포인트로 진단(랭킹 + intervention + 그림)만 다시 돌린다.

학습을 건너뛰므로 intervention 표본 수나 alpha 격자를 바꿔서 재평가할 때 trial당
4시간이 아니라 수십 분이면 된다. 체크포인트/history는 그대로 두고 결과만 새 디렉터리에
쓴다 — 원본 trial은 건드리지 않는다.

    python3 scripts/reeval_trial_diagnostics.py \
        --trial-dir outputs/SAE_validation/grid_search/trial_0000 \
        --eval-images 2000 \
        --out-dir outputs/SAE_validation/reeval/trial_0000_n2000

주의: token normalizer는 저장돼 있지 않아 train 스트림에서 다시 맞춘다. train_loader가
shuffle=True라 매번 정확히 같은 토큰을 보진 않지만, 200만 토큰 평균/표준편차의 표본오차는
무시할 수준이다(같은 분포에서 다시 추정하는 것이지 다른 통계를 쓰는 게 아니다).
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Model.SAE import VanillaL1SAE
from SAE_validation import (
    build_backbone_and_transform,
    build_default_config,
    build_train_val_loaders,
)
from Utils.SAE_plot_utils import PERTURBATION_SCORE_MODES, save_trial_plots
from Utils.SAE_utils import (
    collect_tokens_with_hook,
    evaluate_sae_tokens,
    fit_token_normalizer_streaming,
    get_torch_dtype,
    jsonable,
    normalize_tokens_inplace,
    save_json,
)


def apply_saved_config(config, saved: dict):
    """trial_dir/config.json의 값을 현재 dataclass 위에 덮어쓴다.

    저장 시점에 없던 필드(예: diagnostics)는 saved에 없으므로 현재 기본값이 남는다.
    반대로 현재 dataclass에 없는 옛 키는 조용히 무시한다."""
    for key, value in saved.items():
        if not hasattr(config, key):
            continue
        current = getattr(config, key)
        if isinstance(value, dict) and hasattr(current, "__dataclass_fields__"):
            apply_saved_config(current, value)
        elif not isinstance(value, dict):
            setattr(config, key, value)
    return config


def load_sae(checkpoint_path, device):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"]
    # torch.compile로 학습된 체크포인트는 키에 _orig_mod. 접두사가 붙는다
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    input_dim, hidden_dim = state["W_enc"].shape
    sae = VanillaL1SAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        b_dec_init=torch.zeros(input_dim),
        dec_bias_mode="geom",
    )
    sae.load_state_dict(state)
    sae.eval().to(device)
    meta = {k: payload.get(k) for k in ("epoch", "val_nmse", "l0_raw", "l0_feasible", "l0_constraint")}
    return sae, input_dim, hidden_dim, meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trial-dir", required=True, help="best_sae_state.pt / config.json / history.json 이 있는 디렉터리")
    p.add_argument("--out-dir", default=None, help="결과를 쓸 디렉터리 (기본: <trial-dir>/reeval)")
    p.add_argument("--eval-images", type=int, default=2000, help="랭킹과 intervention이 함께 쓰는 val 이미지 수. 0이면 val 전체")
    p.add_argument("--alphas", default=None, help="쉼표로 구분한 alpha 격자 (예: 0,0.5,1.0,1.5). 생략하면 기본 7점")
    p.add_argument("--score-mode", default="absolute", choices=list(PERTURBATION_SCORE_MODES),
                   help="cue latent 랭킹 점수. absolute=기존, relative=상대화, specific=특이도, relative_specific=둘 다")
    p.add_argument("--min-frequency", type=float, default=0.0,
                   help="상대화의 분모 폭주 방지 하한(발화 빈도). relative 계열에서 필요하다")
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--ranking-batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=None, help="생략하면 trial config 값")
    args = p.parse_args()

    trial_dir = Path(args.trial_dir)
    out_dir = Path(args.out_dir) if args.out_dir else trial_dir / "reeval"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = build_default_config()
    saved_path = trial_dir / "config.json"
    if saved_path.exists():
        apply_saved_config(config, json.loads(saved_path.read_text()))
        print(f"config loaded from {saved_path}")
    else:
        print(f"WARNING: {saved_path} 없음 — build_default_config() 기본값으로 진행한다")

    config.diagnostics.eval_images = None if args.eval_images == 0 else int(args.eval_images)
    config.diagnostics.eval_batch_size = args.eval_batch_size
    config.diagnostics.ranking_batch_size = args.ranking_batch_size
    config.diagnostics.perturbation_score_mode = args.score_mode
    config.diagnostics.perturbation_min_frequency = args.min_frequency
    if args.alphas:
        config.diagnostics.intervention_alphas = [float(x) for x in args.alphas.split(",")]
    if args.num_workers is not None:
        config.data_config.num_workers = args.num_workers
    config.output.root_dir = str(out_dir)
    config = copy.deepcopy(config).validate()

    torch.set_float32_matmul_precision(config.sae.matmul_precision)
    device = config.extraction_config.device
    t_start = time.time()

    model, transform, mean, std = build_backbone_and_transform(config)
    train_dataset, val_dataset, train_loader, val_loader = build_train_val_loaders(config, transform)

    sae, input_dim, hidden_dim, ckpt_meta = load_sae(trial_dir / "best_sae_state.pt", device)
    print(f"SAE loaded: {input_dim} -> {hidden_dim}   checkpoint {ckpt_meta}")

    print("\nRefitting train-token normalizer...")
    normalizer_tokens = config.token.max_normalizer_tokens or config.token.max_train_tokens
    token_stats, seen = fit_token_normalizer_streaming(
        model, train_loader, max_tokens=normalizer_tokens,
        target_block=config.hook.target_block, token_scope=config.hook.token_scope, device=device,
    )
    print(f"normalizer fit on {seen:,} tokens")

    print("\nCollecting validation tokens...")
    val_tokens, val_labels = collect_tokens_with_hook(
        model, val_loader, max_tokens=config.token.max_val_tokens,
        target_block=config.hook.target_block, token_scope=config.hook.token_scope, device=device,
        cache_dtype=get_torch_dtype(config.token.cache_dtype), return_labels=True,
    )
    val_tokens = normalize_tokens_inplace(val_tokens.float(), token_stats)
    print(f"val tokens: {tuple(val_tokens.shape)}")

    metrics = evaluate_sae_tokens(sae, val_tokens, batch_size=config.sae.batch_size,
                                  threshold=config.sae.active_threshold, device=device)
    print("reconstruction:", jsonable(metrics))

    history_path = trial_dir / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []

    diagnostics = save_trial_plots(
        trial_dir=out_dir, model=model, train_history=history, sae=sae, token_stats=token_stats,
        val_tokens=val_tokens, val_labels=val_labels, train_dataset=train_dataset,
        val_dataset=val_dataset, config=config, mean=mean, std=std,
    )

    summary = {
        "source_trial_dir": str(trial_dir),
        "checkpoint_meta": ckpt_meta,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "diagnostics_config": {
            "eval_images": config.diagnostics.eval_images,
            "intervention_alphas": config.diagnostics.intervention_alphas,
            "perturbation_score_mode": config.diagnostics.perturbation_score_mode,
            "perturbation_min_frequency": config.diagnostics.perturbation_min_frequency,
        },
        "final_validation_metrics": metrics,
        "diagnostics": diagnostics,
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    save_json(out_dir / "reeval_summary.json", summary)
    print(f"\nwrote {out_dir / 'reeval_summary.json'}  ({summary['elapsed_seconds'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
