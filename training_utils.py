"""Small, deterministic helpers for stable and adaptive training schedules."""

import math


class ExactDistributedEvalSampler:
    """Shard evaluation indices across ranks without padding or duplication."""

    def __init__(self, dataset, num_replicas, rank):
        if int(num_replicas) < 1:
            raise ValueError("num_replicas must be positive")
        if int(rank) < 0 or int(rank) >= int(num_replicas):
            raise ValueError("rank must be in [0, num_replicas)")
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        return max(0, (len(self.dataset) - self.rank + self.num_replicas - 1)
                   // self.num_replicas)


class EarlyStopping:
    """Stop after validation fails to improve by min_delta for patience checks."""

    def __init__(self, patience=8, min_delta=1e-4):
        if int(patience) < 1:
            raise ValueError("patience must be at least one")
        if min_delta < 0 or not math.isfinite(float(min_delta)):
            raise ValueError("min_delta must be finite and non-negative")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = float("inf")
        self.bad_evals = 0

    def observe(self, metric):
        metric = float(metric)
        if math.isfinite(metric) and metric < self.best - self.min_delta:
            self.best = metric
            self.bad_evals = 0
            return False
        self.bad_evals += 1
        return self.bad_evals >= self.patience

    def state_dict(self):
        return {"patience": self.patience, "min_delta": self.min_delta,
                "best": self.best, "bad_evals": self.bad_evals}

    def load_state_dict(self, state):
        if int(state["patience"]) != self.patience or float(state["min_delta"]) != self.min_delta:
            raise ValueError("early stopping configuration mismatch")
        self.best = float(state["best"])
        self.bad_evals = int(state["bad_evals"])
        if self.bad_evals < 0:
            raise ValueError("invalid early stopping state")



def warmup_cosine_lr(step, base_lr, warmup, total_steps, min_ratio=0.1):
    """Linear warmup followed by cosine decay, returning an absolute LR."""
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if base_lr <= 0 or not math.isfinite(base_lr):
        raise ValueError("base_lr must be finite and positive")
    if not 0 <= min_ratio <= 1:
        raise ValueError("min_ratio must be between zero and one")
    warmup = max(0, int(warmup))
    step = max(0, int(step))
    if warmup and step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(total_steps - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * cosine)


class PlateauScale:
    """Reduce the scheduled learning rate when validation loss stalls."""

    def __init__(self, patience=2, factor=0.5, min_scale=0.125, min_delta=1e-4):
        if patience < 1:
            raise ValueError("patience must be at least one")
        if not 0 < factor < 1:
            raise ValueError("factor must be between zero and one")
        if not 0 < min_scale <= 1:
            raise ValueError("min_scale must be in (0, 1]")
        if min_delta < 0:
            raise ValueError("min_delta cannot be negative")
        self.patience = int(patience)
        self.factor = float(factor)
        self.min_scale = float(min_scale)
        self.min_delta = float(min_delta)
        self.scale = 1.0
        self.best = float("inf")
        self.bad_evals = 0

    def observe(self, metric):
        """Observe validation loss and return the current LR multiplier."""
        metric = float(metric)
        if math.isfinite(metric) and metric < self.best - self.min_delta:
            self.best = metric
            self.bad_evals = 0
            return self.scale
        self.bad_evals += 1
        if self.bad_evals >= self.patience:
            self.scale = max(self.min_scale, self.scale * self.factor)
            self.bad_evals = 0
        return self.scale

    def state_dict(self):
        return {
            "patience": self.patience,
            "factor": self.factor,
            "min_scale": self.min_scale,
            "min_delta": self.min_delta,
            "scale": self.scale,
            "best": self.best,
            "bad_evals": self.bad_evals,
        }

    def load_state_dict(self, state):
        for key in ("patience", "factor", "min_scale", "min_delta"):
            if key in state and state[key] != getattr(self, key):
                raise ValueError(f"plateau scheduler configuration mismatch: {key}")
        self.scale = float(state["scale"])
        self.best = float(state["best"])
        self.bad_evals = int(state["bad_evals"])
        if not 0 < self.scale <= 1 or self.bad_evals < 0:
            raise ValueError("invalid plateau scheduler state")


def perplexity_from_loss(loss):
    """Convert mean cross-entropy to perplexity; the theoretical floor is 1."""
    loss = float(loss)
    if not math.isfinite(loss) or loss < 0:
        raise ValueError("loss must be finite and non-negative")
    return math.exp(min(loss, 80.0))


def split_2022_periods(sample_timestamps):
    """Split 2022 into Jan-Jun calibration, Jul-Sep threshold, Oct-Dec horizon masks."""
    july = 1656633600  # 2022-07-01 00:00:00 UTC
    october = 1664582400  # 2022-10-01 00:00:00 UTC
    timestamps = [int(value) for value in sample_timestamps]
    return (
        [value < july for value in timestamps],
        [july <= value < october for value in timestamps],
        [value >= october for value in timestamps],
    )
