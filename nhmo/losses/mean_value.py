"""L_MV: mean-value martingale loss (the physics anchor, §6.2).

Harmonic functions satisfy, for any sphere B_r(p) ⊂ Ω:
    u(p) = E_{ζ ~ uniform(∂B_r(p))} [u(ζ)]

Enforced DIRECTLY on the kernel (no probe h needed) via §6.2's "elegant"
formulation:
    K_θ(p, ζ) = E_{y ~ uniform(∂B_r(p))} [K_θ(y, ζ)]

    L_MV = E_{p, Ω, ζ, r} [ (K_θ(p, ζ) − (1/S) Σ_s K_θ(y_s, ζ))² ]

This is the unsupervised physics anchor — without it, v2 is a supervised
WoS mimic (§12 A5).

Structural constraints:

1. S is a CONFIG parameter. Phase 4 default S=8 (correctness validation,
   8× cheaper than §6.2's production 64). Phase 5 raises via override.
2. shape_latent is encoded ONCE per (p, Ω) pair. The sphere-sample loop
   reuses it. A runtime assertion catches the most likely perf bug:
   re-encoding per y_s.
3. Loop body is strictly functional: no in-place ops, no train/eval
   toggles, no detach_. Loss accumulation uses the running-sum pattern
   (tensor + tensor) NOT torch.stack([losses]).mean() which keeps S
   autograd graphs alive.
4. backward() is called ONCE on the returned scalar, OUTSIDE the loop.
   Never retain_graph=True.

r sampling:
- Log-uniform: r = sdf_p · exp(U · (log r_max − log r_min) + log r_min),
  U ~ Uniform(0, 1). Log-uniform samples each scale decade equally — the
  natural measure for a scale-covariant Poisson kernel.
- r_frac_min = 0.2, r_frac_max = 0.9. The lower 0.2 (not 0.1) is because
  at 0.1 · sdf_p the sphere is so small K(y_s, ·) is nearly constant
  across s → MC variance dominates the martingale signal.
- Floor guard: if sdf_p · r_frac_max < r_floor (= 1e-3 · bbox_diag), skip
  L_MV for that batch entry. L_BL is the correct loss for the near-
  boundary regime; the two losses partition the domain without overlap.

# ---- PHASE 5 VECTORIZATION NOTE ----
# The sphere loop is L_MV's dominant forward cost. Phase 5 vectorizes by
# expanding shape_latent to (B*S, M, d_model) and treating y_s as an
# extended batch dimension. Estimated memory at B=8, S=64, M=64, d=192,
# N_surface=2000:
#   shape_latent expansion:      B*S*M*d*4 bytes = 25 MiB   (OK)
#   cross-attn activation peak:  B*S*N*d*4 bytes = 780 MiB × 4 layers
#                                              ≈ 3.1 GiB   (needs checkpoint)
# Phase 5 plan: gradient checkpointing on the kernel head's cross-attn
# stack. Fits in 40 GiB.
# -------------------------------------

# ---- PHASE 7.1.0 NORMALIZATION NOTE (2026-04-24) ----
# L_MV now normalizes K_θ INTERNALLY on every evaluation (at p and each
# y_s), independent of model.normalize. The martingale property is a
# property of the harmonic measure's DENSITY (normalized K_θ), not of
# the raw log-kernel output log K̃_θ. Evaluating martingale residual on
# unnormalized exp(log K̃) was the root cause of the Phase 7.1.0
# soft-warmup divergence (H5): raw exp(log K̃) can reach 10^6 for
# momentary log K̃ spikes, making residuals (K_tilde_p − mean K_tilde_y)²
# = O(10^12) and gradients O(10^13). Normalizing-internally keeps K_θ
# bounded by the surface-quadrature normalization condition regardless
# of intermediate log K̃ fluctuation.
#
# Computational cost: now (S + 1) logsumexp operations over N_surface
# per L_MV call (one for p, one for each y_s). At S = 4, that's 5× the
# normalization cost of the old path — small compared to the S cross-
# attn forwards already in the loop. No additional memory.
# -----------------------------------------------------

# ---- PHASE 5+ REVISIT: r-sampling strategy ----
# Current: log-uniform(0.2, 0.9) · sdf(p). Motivation: scale-covariant
# sampling over the martingale's informative range. After Phase 5
# measurements, consider:
#   (a) importance-weighting r toward regimes where martingale residual
#       is largest (adaptive sampling)
#   (b) stratified r sampling (deterministic grid in log-space) for lower
#       gradient variance at the cost of correlation across training steps
#   (c) tightening r_frac_max on shapes with many thin features, where
#       0.9 · sdf_p frequently produces spheres that touch ∂Ω
# Do NOT implement these in Phase 4. Measure first.
# -------------------------------------
"""
from __future__ import annotations

import math

import torch
from torch import Tensor


def l_mv(
    model,                                # HarmonicMeasureField (duck-typed; has .log_kernel())
    p: Tensor,                            # (B, 3) interior query points
    shape_latent,                         # ShapeLatent (tokens: (B, M, d_model))
    zeta_probe: Tensor,                   # (B, N_probe, D) boundary samples to evaluate K at
    zeta_probe_normal: Tensor,            # (B, N_probe, D)
    sdf_at_p: Tensor,                     # (B,) distance from each p to ∂Ω
    *,
    S: int,                               # REQUIRED: # sphere samples; Phase 4 default 8, Phase 5 64
    r_frac_min: float = 0.2,              # LOWER bound fraction of sdf_p
    r_frac_max: float = 0.9,              # upper bound fraction of sdf_p
    r_floor_rel_bbox: float = 1.0e-3,     # absolute floor, as fraction of bbox diag
    sample_log_uniform: bool = True,      # True → log-uniform; False → uniform (ablation)
    bbox_diag: float = 2.0 * math.sqrt(3),  # default: diagonal of [-1,1]^3
    surface_area_weights: Tensor = None,  # REQUIRED (B, N_probe) — l_mv always normalizes internally
) -> Tensor:                              # scalar
    """L_MV enforces the martingale property on the NORMALIZED harmonic
    measure density K_θ. Internally normalizes K at p and at every
    sphere sample y_s, INDEPENDENT of the model's .normalize mode.

    This is required for numerical stability: evaluating the martingale
    residual on unnormalized exp(log K̃) lets momentary spikes in log K̃
    (which are bounded-ish before division by Z but unbounded raw)
    produce gradient-exploding residuals. Phase 7.1.0 soft-warmup
    divergence (2026-04-24) traced to this exact failure mode (H5);
    always-normalize fixes it structurally.

    surface_area_weights is REQUIRED (not optional) post Phase 7.1.0
    fix. Callers must pass the same w that the kernel head sees at
    test time; a small constant w everywhere is fine if the domain's
    quadrature is not explicitly area-weighted.
    """
    if surface_area_weights is None:
        raise ValueError(
            "l_mv requires surface_area_weights (post Phase 7.1.0 fix). "
            "l_mv always normalizes K internally; pass the same weights "
            "the kernel head expects at test time."
        )
    B, N_probe, _ = zeta_probe.shape
    d_model = shape_latent.tokens.shape[-1]
    M_slices = shape_latent.tokens.shape[-2]
    log_w = torch.log(surface_area_weights + 1e-30)  # (B, N_probe)

    # Floor guard: mask out batch entries whose sdf_p is too small.
    r_floor = r_floor_rel_bbox * bbox_diag
    r_max_candidate = sdf_at_p * r_frac_max
    active_mask = (r_max_candidate >= r_floor).to(p.dtype)  # (B,)
    n_active = active_mask.sum()

    if n_active.item() == 0:
        # All batch entries in the near-boundary regime; L_MV is not
        # meaningful here. Return a zero-valued tensor that is
        # gradient-connected to model parameters (autograd needs a graph
        # edge for the upstream backward). Multiply by any model param's
        # sum times 0 — legal, produces zero gradient, connects graph.
        any_param = next(model.parameters())
        return 0.0 * any_param.sum()

    # Sample r per batch entry (log-uniform or uniform).
    U = torch.rand(B, device=p.device, dtype=p.dtype)
    if sample_log_uniform:
        log_frac = U * (math.log(r_frac_max) - math.log(r_frac_min)) + math.log(r_frac_min)
        frac = torch.exp(log_frac)
    else:
        frac = U * (r_frac_max - r_frac_min) + r_frac_min
    r = frac * sdf_at_p                                  # (B,)

    # Sample S unit-direction vectors on the (D-1)-sphere uniformly
    # (Gaussian + normalize). D = p.shape[-1] is 2 for 2D Laplace
    # (→ direction on S¹, i.e. a circle) or 3 for 3D (→ S²).
    spatial_dim = p.shape[-1]
    dirs = torch.randn(B, S, spatial_dim, device=p.device, dtype=p.dtype)
    dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-6)
    # y_s = p + r · dir  →  (B, S, D)
    y_all = p.unsqueeze(1) + r.view(B, 1, 1) * dirs

    # K_θ(p, ζ): normalize internally via logsumexp (works identically in
    # hard/soft/none modes because we bypass model.kernel's mode-switch).
    log_K_tilde_p = model.log_kernel(
        p, zeta_probe, zeta_probe_normal, shape_latent,
    )                                                    # (B, N_probe)
    log_Z_p = torch.logsumexp(log_K_tilde_p + log_w, dim=-1, keepdim=True)  # (B, 1)
    log_K_p = log_K_tilde_p - log_Z_p                    # (B, N_probe)
    K_at_p = torch.exp(log_K_p)                          # (B, N_probe)

    # ---- Sphere-sample loop (Phase 4 functional style) ----
    # shape_latent is encoded ONCE per (p, Ω) and reused across S samples.
    # Running-sum accumulation (NOT torch.stack([...]).mean()) to avoid
    # keeping S autograd graphs alive.
    assert shape_latent.tokens.shape == (B, M_slices, d_model), (
        f"shape_latent re-encoded per sphere sample — bug. "
        f"Got {shape_latent.tokens.shape}, expected ({B}, {M_slices}, {d_model})."
    )

    K_at_y_sum = torch.zeros(B, N_probe, device=p.device, dtype=K_at_p.dtype)
    for s in range(S):
        y_s = y_all[:, s, :]                              # (B, D)
        # Normalize K at y_s internally (Phase 7.1.0 fix — H5).
        log_K_tilde_y = model.log_kernel(
            y_s, zeta_probe, zeta_probe_normal, shape_latent,
        )                                                 # (B, N_probe)
        log_Z_y = torch.logsumexp(log_K_tilde_y + log_w, dim=-1, keepdim=True)
        log_K_y = log_K_tilde_y - log_Z_y                 # (B, N_probe)
        K_s = torch.exp(log_K_y)                          # (B, N_probe)
        K_at_y_sum = K_at_y_sum + K_s                     # running sum
    K_at_y_mean = K_at_y_sum / float(S)                   # (B, N_probe)

    # Per-batch-entry martingale residual, then mask out inactive entries,
    # then average over active entries only.
    sq_residual = (K_at_p - K_at_y_mean) ** 2             # (B, N_probe)
    per_entry_residual = sq_residual.mean(dim=-1)         # (B,)
    masked = per_entry_residual * active_mask             # (B,)
    return masked.sum() / n_active.clamp(min=1.0)
