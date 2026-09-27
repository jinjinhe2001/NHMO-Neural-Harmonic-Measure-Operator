"""LR + loss-weight schedules per v2 plan §7.2–§7.3."""
from __future__ import annotations

import math


def cosine_with_warmup(
    step: int,
    total_steps: int,
    warmup_steps: int = 1000,
    lr_max: float = 3e-4,
    lr_min: float = 1e-5,
) -> float:
    """Linear warmup for `warmup_steps`, then cosine decay to `lr_min`.

    Args:
        step: current training step (0-indexed).
        total_steps: total training length (for cosine period).
        warmup_steps: number of linear warmup steps (LR ramps 0 → lr_max).
        lr_max: peak learning rate after warmup.
        lr_min: final learning rate.

    Returns:
        Learning rate at this step (float).
    """
    if step < warmup_steps:
        return lr_max * step / max(warmup_steps, 1)
    # Cosine decay from lr_max → lr_min over [warmup_steps, total_steps)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min + (lr_max - lr_min) * cosine


def loss_weight_schedule(step: int, cfg: dict | None = None) -> dict:
    """Piecewise loss weights per v2 plan §7.2.

    Stage 0 (0 → stage0_end):        nll=1.0, mv=0.1, bl=0.5, z=1.0
    Stage 1 (stage0_end → stage1_end): nll=1.0, mv=1.0, bl=0.5, z=1.0
    Stage 2 (stage1_end → end):       nll=0.5, mv=2.0, bl=0.5, z=1.0

    Defaults (per §7.2):
        stage0_end = 10000
        stage1_end = 80000
    """
    cfg = cfg or {}
    stage0_end = int(cfg.get("stage0_end", 10000))
    stage1_end = int(cfg.get("stage1_end", 80000))

    if step < stage0_end:
        return {"lambda_nll": 1.0, "lambda_mv": 0.1, "lambda_bl": 0.5, "lambda_z": 1.0}
    if step < stage1_end:
        return {"lambda_nll": 1.0, "lambda_mv": 1.0, "lambda_bl": 0.5, "lambda_z": 1.0}
    return {"lambda_nll": 0.5, "lambda_mv": 2.0, "lambda_bl": 0.5, "lambda_z": 1.0}
