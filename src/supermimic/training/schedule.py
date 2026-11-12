"""Learning-rate schedules: "cosine" (paper Table A.5: decay to `final_ratio` over `decay_fraction` of training,
then constant) or "constant" (the official SuperMimic code). `start` holds the base rate until that iteration and
decays over the rest (a constant-LR run resumed for a decay phase)."""

from __future__ import annotations

import math


def cosine_lr(step: int, total_steps: int, base_lr: float, final_ratio: float = 0.03, decay_fraction: float = 250_000 / 320_000, start: int = 0) -> float:
    decay_steps = max(1, int((total_steps - start) * decay_fraction))
    step = max(0, step - start)
    if step >= decay_steps:
        return base_lr * final_ratio
    cos = 0.5 * (1.0 + math.cos(math.pi * step / decay_steps))
    return base_lr * (final_ratio + (1.0 - final_ratio) * cos)


def scheduled_lr(schedule: str, step: int, total_steps: int, base_lr: float, final_ratio: float = 0.03, decay_fraction: float = 250_000 / 320_000, start: int = 0) -> float:
    if schedule == "constant":
        return base_lr
    assert schedule == "cosine", schedule
    return cosine_lr(step, total_steps, base_lr, final_ratio, decay_fraction, start)
