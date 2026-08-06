from pathlib import Path

import torch


def unwrap_compiled_model(model):
    return getattr(model, "_orig_mod", model)


class EarlyStopper:
    # Sentinel worse than every real tier: "no checkpoint has been saved yet".
    NO_TIER = 2

    def __init__(
        self,
        patience,
        eps,
        checkpoint_path,
        metric_name="val_nmse",
        tie_breaker_name="active_mean_count",
        l0_metric=None,
        l0_max=None,
        verbose=True,
    ):
        self.patience = patience
        self.eps = float(eps)
        self.checkpoint_path = Path(checkpoint_path)
        self.metric_name = metric_name
        self.tie_breaker_name = tie_breaker_name
        self.l0_metric = l0_metric
        self.l0_max = None if l0_max is None else float(l0_max)
        self.best_score = float("inf")
        self.early_stop_best_score = float("inf")
        self.lowest_score = float("inf")
        self.best_tie_breaker = float("inf")
        self.bad_epochs = 0
        self.best_epoch = None
        self.best_tier = self.NO_TIER
        self.early_stop_best_tier = self.NO_TIER
        self.best_val_nmse = float("inf")
        self.best_is_feasible = None
        self.verbose = verbose

    def _tier_and_score(self, score, row):
        """(tier, 그 tier에서 최소화할 값)을 돌려준다.

        tier 0 = L0 제약을 만족하는 epoch → 그 안에서 val_nmse를 줄인다.
        tier 1 = 제약을 위반한 epoch → 아직 후보가 아니므로 L0를 줄이는 것 자체가 개선이다.
        제약이 없으면 항상 tier 0이라 기존 동작과 완전히 동일하다."""
        if self.l0_max is None:
            return 0, score
        l0 = row.get(self.l0_metric)
        if l0 is None:
            raise KeyError(
                f"희소성 제약이 켜져 있는데 epoch row에 L0 지표 {self.l0_metric!r}가 없다."
            )
        return (0, score) if float(l0) <= self.l0_max else (1, float(l0))

    def _save_checkpoint(self, model, row):
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        base_model = unwrap_compiled_model(model)
        torch.save(
            {
                "model_state_dict": base_model.state_dict(),
                "epoch": int(row["epoch"]),
                self.metric_name: float(row["normalized_mse"]),
                self.tie_breaker_name: float(row.get(self.tie_breaker_name, float("nan"))),
                "l0_raw": row.get("l0_raw"),
                "l0_feasible": row.get("l0_feasible"),
                "l0_constraint": {"metric": self.l0_metric, "max": self.l0_max},
                "row": dict(row),
            },
            self.checkpoint_path,
        )

    def step(self, score, model, row):
        score = float(score)
        tier, tier_score = self._tier_and_score(score, row)
        tie_breaker = float(row.get(self.tie_breaker_name, float("inf")))
        has_checkpoint = self.best_epoch is not None
        row["l0_feasible"] = tier == 0

        # Early stopping: a better tier always counts as improvement, so patience never
        # runs out while the run is still working its way down to the L0 constraint.
        if tier < self.early_stop_best_tier:
            self.early_stop_best_tier = tier
            self.early_stop_best_score = tier_score
            metric_improved = True
        elif tier > self.early_stop_best_tier:
            metric_improved = False
        else:
            # Within a tier, only changes larger than eps count as improvement.
            metric_improved = tier_score < (self.early_stop_best_score - self.eps)
            if metric_improved:
                self.early_stop_best_score = tier_score

        if metric_improved:
            self.bad_epochs = 0
        elif has_checkpoint:
            self.bad_epochs += 1

        # Checkpoint selection is separate from patience counting.
        if tier < self.best_tier:
            # First epoch to satisfy the constraint: the violating epochs are no longer
            # comparable candidates, so drop their bookkeeping and restart within this tier.
            self.best_tier = tier
            self.best_score = float("inf")
            self.lowest_score = float("inf")
            self.best_tie_breaker = float("inf")
            checkpoint_updated = True
        elif tier > self.best_tier:
            checkpoint_updated = False
        else:
            # Same tier: keep the lowest score seen, or within-eps alternatives with lower activity.
            checkpoint_score_decreased = tier_score < self.best_score
            metric_tied = has_checkpoint and abs(tier_score - self.lowest_score) <= self.eps
            tie_breaker_improved = tie_breaker < self.best_tie_breaker
            checkpoint_updated = (
                (not has_checkpoint)
                or checkpoint_score_decreased
                or (metric_tied and tie_breaker_improved)
            )

        if checkpoint_updated:
            self.best_score = tier_score
            self.best_tie_breaker = tie_breaker
            self.best_epoch = int(row["epoch"])
            self.best_val_nmse = float(row["normalized_mse"])
            self.best_is_feasible = tier == 0
            self._save_checkpoint(model, row)
            if self.verbose:
                feasible_note = "" if self.l0_max is None else f", l0_feasible={self.best_is_feasible}"
                print(
                    f"saved best SAE checkpoint: {self.checkpoint_path} "
                    f"(epoch={self.best_epoch}, {self.metric_name}={score:.6f}, "
                    f"{self.tie_breaker_name}={tie_breaker:.4f}{feasible_note})"
                )

        if tier == self.best_tier:
            self.lowest_score = min(self.lowest_score, tier_score)
        row["best_val_nmse"] = self.best_val_nmse
        row["early_stop_best_val_nmse"] = self.early_stop_best_score
        row["lowest_val_nmse"] = self.lowest_score
        row["best_active_mean_count"] = self.best_tie_breaker
        row["best_l0_feasible"] = self.best_is_feasible
        row["early_stop_bad_epochs"] = self.bad_epochs
        row["is_best"] = checkpoint_updated
        row["metric_improved"] = metric_improved
        return checkpoint_updated, self.should_stop

    @property
    def should_stop(self):
        return self.patience is not None and self.bad_epochs >= int(self.patience)

    def load_best(self, model, map_location=None):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"SAE checkpoint does not exist: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=map_location)
        base_model = unwrap_compiled_model(model)
        base_model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint
