from pathlib import Path

import torch


def unwrap_compiled_model(model):
    return getattr(model, "_orig_mod", model)


class EarlyStopper:
    def __init__(
        self,
        patience,
        eps,
        checkpoint_path,
        metric_name="val_nmse",
        tie_breaker_name="active_mean_count",
        verbose=True,
    ):
        self.patience = patience
        self.eps = float(eps)
        self.checkpoint_path = Path(checkpoint_path)
        self.metric_name = metric_name
        self.tie_breaker_name = tie_breaker_name
        self.best_score = float("inf")
        self.early_stop_best_score = float("inf")
        self.lowest_score = float("inf")
        self.best_tie_breaker = float("inf")
        self.bad_epochs = 0
        self.best_epoch = None
        self.verbose = verbose

    def _save_checkpoint(self, model, row):
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        base_model = unwrap_compiled_model(model)
        torch.save(
            {
                "model_state_dict": base_model.state_dict(),
                "epoch": int(row["epoch"]),
                self.metric_name: float(row["normalized_mse"]),
                self.tie_breaker_name: float(row[self.tie_breaker_name]),
                "row": dict(row),
            },
            self.checkpoint_path,
        )

    def step(self, score, model, row):
        score = float(score)
        tie_breaker = float(row.get(self.tie_breaker_name, float("inf")))
        has_checkpoint = self.best_epoch is not None

        # Early stopping only treats changes larger than eps as improvement.
        metric_improved = score < (self.early_stop_best_score - self.eps)

        # Checkpoint selection is separate from patience counting:
        # keep the lowest score seen, or within-eps alternatives with lower activity.
        checkpoint_score_decreased = score < self.best_score
        metric_tied = has_checkpoint and abs(score - self.lowest_score) <= self.eps
        tie_breaker_improved = tie_breaker < self.best_tie_breaker
        checkpoint_updated = (
            (not has_checkpoint)
            or checkpoint_score_decreased
            or (metric_tied and tie_breaker_improved)
        )

        if metric_improved:
            self.early_stop_best_score = score
            self.bad_epochs = 0
        elif has_checkpoint:
            self.bad_epochs += 1

        if checkpoint_updated:
            self.best_score = score
            self.best_tie_breaker = tie_breaker
            self.best_epoch = int(row["epoch"])
            self._save_checkpoint(model, row)
            if self.verbose:
                print(
                    f"saved best SAE checkpoint: {self.checkpoint_path} "
                    f"(epoch={self.best_epoch}, {self.metric_name}={score:.6f}, "
                    f"{self.tie_breaker_name}={tie_breaker:.4f})"
                )

        self.lowest_score = min(self.lowest_score, score)
        row["best_val_nmse"] = self.best_score
        row["early_stop_best_val_nmse"] = self.early_stop_best_score
        row["lowest_val_nmse"] = self.lowest_score
        row["best_active_mean_count"] = self.best_tie_breaker
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
