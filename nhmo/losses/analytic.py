"""L_analytic: direct log-space supervision against a known analytic kernel.

Per v2 plan §8.2:
    L_analytic = E_{p, ζ} [ (log K_θ(p, ζ) − log K_true(p, ζ))² ]

Phase 6 uses this for sphere pretraining with `SphereHarmonicMeasure` as
K_true. Log-space MSE preserves the Poisson kernel's 10⁴–10⁶ dynamic
range as a uniform relative-error target (linear-space MSE would be
dominated by near-boundary samples where K is huge).

A7 clarification: A7 bans L2 on K as a supervised-density-matching loss;
L2 on log K against ground-truth log K is the CORRECT way to match two
log-densities and does NOT reintroduce the prohibition.
"""
from __future__ import annotations

import torch
from torch import Tensor


def l_analytic(
    log_K_theta: Tensor,     # (B, N) learned log K (normalized)
    log_K_true: Tensor,      # (B, N) analytic log K at the same (p, ζ)
    mask: Tensor | None = None,   # (B, N) optional valid-sample mask
) -> Tensor:                 # scalar
    diff = log_K_theta - log_K_true
    if mask is not None:
        m = mask.to(diff.dtype)
        n_valid = m.sum().clamp(min=1.0)
        return ((diff ** 2) * m).sum() / n_valid
    return (diff ** 2).mean()
