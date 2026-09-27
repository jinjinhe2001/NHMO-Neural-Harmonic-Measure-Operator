"""L_Z: normalization loss for the "soft" regime (§6.4).

Three regimes of the kernel's `normalize` attribute:
  "soft":  K_θ = exp(log K̃_θ). Add L_Z = E_p [ huber(log ∫ w K̃, δ) ].
  "hard":  K_θ(p, ζ) = exp(log K̃_θ) / Z_θ(p). L_Z ≡ 0 (kernel is exactly
           normalized; see nhmo/core/kernel.py's `kernel()` method).
  "none":  Debugging. L_Z ≡ 0.

The LossRegistry gates which regime is active. This module implements
only the "soft" branch.

# ---- PHASE 7.1.0 REFORMULATION (H5b fix, 2026-04-24) ----
# Was: (Z − 1)² where Z = exp(logsumexp(log K̃ + log w)).
# Now: log_Z² where log_Z = logsumexp(log K̃ + log w).
#
# Root cause: materializing Z = exp(log_Z) and squaring (Z − 1) makes the
# loss quadratic in an exponential of log K̃. When MCB training produces
# momentary log K̃ spikes (seen to 36.88 at step 1432 of the h5fix warmup
# run), Z becomes ≈ 1e16 and (Z − 1)² ≈ 1e32. Gradient goes Inf in one
# step, corrupts weights on the next, ~40 consecutive NaN steps follow.
#
# log_Z² preserves the penalty's structural role:
#   - Same minimum: log_Z² = 0 ⟺ Z = 1 (unchanged target).
#   - Taylor-equivalent near optimum: (Z − 1)² ≈ (log Z)² for Z ≈ 1.
#   - Bounded gradient: d L / d(log K̃_j) = 2·log_Z · K_θ_j · w_j where
#     K_θ = exp(log K̃ − log_Z) is the normalized density (bounded by N
#     in logsumexp, not by exp(log_K̃_max)). Grows polynomially in
#     log K̃, not exponentially.
# ---------------------------------------------------------

# ---- PHASE 7.1.1 HUBER FIX (2026-04-24 second pass) ----
# The log_Z² form is still quadratic in log_Z, so its gradient magnitude
# 2·|log_Z| grows without bound. MCB runs on 1000+ shapes showed log_Z
# transiently drifting to 40+, producing gradient scales of 80+ which
# destabilized training after ~1300 steps (7 unrecovered NLL spikes).
#
# Huber piecewise loss matches log_Z² and its gradient near the origin
# (the regime of healthy training) and saturates to LINEAR beyond |log_Z|
# = δ. Gradient is bounded by 2·δ everywhere. Matches value AND derivative
# of the quadratic form at the transition:
#
#     H_δ(x) =  x²                          if |x| ≤ δ
#           =  2δ|x| − δ²                  if |x| > δ
#
#     H_δ'(x) = 2x                          if |x| ≤ δ
#           =  2δ·sign(x)                  if |x| > δ      ← BOUNDED by 2δ
#
# Setting huber_delta=None recovers the original log_Z² behavior for
# backward compatibility (analytic pretraining uses this since it
# already trains stably).
# ---------------------------------------------------------
"""
from __future__ import annotations

import torch
from torch import Tensor


def l_z_soft(
    log_K_tilde_at_zeta: Tensor,   # (B, N_surface) UNNORMALIZED log K̃
    surface_area_weights: Tensor,  # (B, N_surface)
    huber_delta: float | None = None,
) -> Tensor:                       # scalar
    """Huber-on-log_Z normalization penalty.

    When `huber_delta is None`, reduces to log_Z² (the Phase 7.1.0 form).
    When `huber_delta` is a positive float, uses the piecewise Huber that
    matches log_Z² within |log_Z| ≤ δ and saturates to linear beyond.
    Gradient is bounded by 2δ everywhere, so isolated log_Z spikes can
    never produce the 587× NLL jumps observed in the Phase 7.1.0 retry.
    """
    log_w = torch.log(surface_area_weights + 1e-30)
    log_Z = torch.logsumexp(log_K_tilde_at_zeta + log_w, dim=-1)   # (B,)
    if huber_delta is None:
        return (log_Z ** 2).mean()
    delta = float(huber_delta)
    if delta <= 0.0:
        raise ValueError(f"huber_delta must be positive; got {delta!r}")
    abs_lz = log_Z.abs()
    quadratic = log_Z ** 2
    linear = 2.0 * delta * abs_lz - delta ** 2
    return torch.where(abs_lz <= delta, quadratic, linear).mean()
