"""네 가지 채점 모드가 각각 어떤 cue latent를 고르는지 나란히 본다.

섭동 통계 누적은 모드와 무관하므로 **한 번만** 돌리고 네 번 채점한다 — 모드 하나당
랭킹을 다시 도는 것보다 4배 싸다.

보려는 것 두 가지:
- 발화 빈도 상위 latent가 계속 상위를 점령하는가 (absolute의 실측 문제)
- 세 kind가 서로 다른 latent를 고르는가 (color/texture/shape를 구분하려면 필수)

    python3 scripts/compare_perturbation_scoring.py \
        --trial-dir outputs/SAE_validation/grid_search/trial_0000 \
        --eval-images 512 --min-frequency 1e-4
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from SAE_validation import build_backbone_and_transform, build_default_config, build_train_val_loaders
from Utils.SAE_plot_utils import (
    INTERVENTION_TOP_K,
    PERTURBATION_KINDS,
    PERTURBATION_SCORE_MODES,
    accumulate_perturbation_deltas,
    score_perturbation_accum,
)
from Utils.SAE_utils import fit_token_normalizer_streaming, save_json
from scripts.reeval_trial_diagnostics import apply_saved_config, load_sae


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trial-dir", required=True)
    p.add_argument("--eval-images", type=int, default=512, help="랭킹 겹침이 포화하는 지점이 약 512다")
    p.add_argument("--min-frequency", type=float, default=1e-4)
    p.add_argument("--ranking-batch-size", type=int, default=8)
    p.add_argument("--top-k", type=int, default=INTERVENTION_TOP_K, help="cue 집합 크기")
    p.add_argument("--out", default=None, help="결과 JSON 경로 (기본: <trial-dir>/scoring_comparison.json)")
    args = p.parse_args()

    trial_dir = Path(args.trial_dir)
    config = build_default_config()
    saved = trial_dir / "config.json"
    if saved.exists():
        apply_saved_config(config, json.loads(saved.read_text()))
    config.validate()
    device = config.extraction_config.device

    torch.set_float32_matmul_precision(config.sae.matmul_precision)
    model, transform, mean, std = build_backbone_and_transform(config)
    _train_ds, val_dataset, train_loader, _val_loader = build_train_val_loaders(config, transform)
    sae, _in_dim, hidden_dim, _meta = load_sae(trial_dir / "best_sae_state.pt", device)

    token_stats, _ = fit_token_normalizer_streaming(
        model, train_loader, max_tokens=config.token.max_normalizer_tokens or config.token.max_train_tokens,
        target_block=config.hook.target_block, token_scope=config.hook.token_scope, device=device,
    )

    n = min(int(args.eval_images), len(val_dataset))
    print(f"\naccumulating perturbation deltas on {n:,} images (한 번만 돈다)")
    accum, count = accumulate_perturbation_deltas(
        model, val_dataset, sae, token_stats, config, mean, std,
        image_indices=list(range(n)), batch_size=args.ranking_batch_size,
    )

    freq = accum[PERTURBATION_KINDS[0]]["freq_sum"] / max(1, count)
    alive = freq > 0
    q = torch.tensor([0.5, 0.75, 0.9, 0.99])
    print(f"\nfrequency(z > {config.sae.active_threshold}) — 살아있는 latent {int(alive.sum()):,} / {hidden_dim:,}")
    print("  살아있는 것들의 분위수:", {f"p{int(x*100)}": f"{v:.2e}" for x, v in zip(q.tolist(), torch.quantile(freq[alive], q).tolist())})
    n_pass = int((freq >= args.min_frequency).sum())
    print(f"  min_frequency={args.min_frequency:g} 통과: {n_pass:,} ({100*n_pass/hidden_dim:.1f}%)")

    top_freq = set(torch.topk(freq, k=20).indices.tolist())

    out = {"eval_images": n, "min_frequency": args.min_frequency, "top_k": args.top_k, "modes": {}}
    for mode in PERTURBATION_SCORE_MODES:
        # absolute는 하한 없이 채점해야 기존 동작과 같다
        floor = 0.0 if mode == "absolute" else args.min_frequency
        ranking = score_perturbation_accum(accum, count, kinds=PERTURBATION_KINDS,
                                           top_k=max(args.top_k, 20), score_mode=mode, min_frequency=floor)
        cues = {k: ranking[k]["latent_ids"][:args.top_k] for k in PERTURBATION_KINDS}
        out["modes"][mode] = {"min_frequency": floor, "cue_latents": cues}

        print(f"\n{'='*78}\n### {mode}   (min_frequency={floor:g})")
        for k in PERTURBATION_KINDS:
            hits = len(set(cues[k]) & top_freq)
            sel = torch.tensor(cues[k], dtype=torch.long)
            f_sel = freq[sel]
            print(f"  {k:14s} {cues[k]}")
            # 고른 latent가 실제로 얼마나 켜지는지 — 너무 희소하면 지워도 아무 일이 안 일어난다
            print(f"  {'':14s} 발화빈도 top-20과 겹침 {hits}/{args.top_k} | 고른 latent의 빈도 "
                  f"min {f_sel.min():.2e} med {f_sel.median():.2e} max {f_sel.max():.2e}")
            out["modes"][mode].setdefault("cue_frequency", {})[k] = {
                "min": float(f_sel.min()), "median": float(f_sel.median()), "max": float(f_sel.max()),
            }
        pairs = [(a, b) for i, a in enumerate(PERTURBATION_KINDS) for b in PERTURBATION_KINDS[i + 1:]]
        overlaps = {f"{a}∩{b}": len(set(cues[a]) & set(cues[b])) for a, b in pairs}
        shared_all = set(cues[PERTURBATION_KINDS[0]])
        for k in PERTURBATION_KINDS[1:]:
            shared_all &= set(cues[k])
        print(f"  kind 간 겹침: {overlaps}   세 kind 공통: {len(shared_all)} {sorted(shared_all)}")
        out["modes"][mode]["pairwise_overlap"] = overlaps
        out["modes"][mode]["shared_by_all_kinds"] = sorted(shared_all)
        out["modes"][mode]["top_freq_overlap"] = {k: len(set(cues[k]) & top_freq) for k in PERTURBATION_KINDS}

    dest = Path(args.out) if args.out else trial_dir / "scoring_comparison.json"
    save_json(dest, out)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
