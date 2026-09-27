"""L_BL: boundary-limit loss enforcing ω_p → δ_ζ as p → ζ_0 (§6.3).

For p very close to ∂Ω at boundary point ζ_0, the harmonic measure
concentrates on ζ_0:
    ω(p, dζ) → δ_{ζ_0}(dζ)  as p → ζ_0

Operationally:
    - pick ζ_0 ∈ ∂Ω; pick p_ε = ζ_0 − ε · n(ζ_0), with ε small
    - K_θ(p_ε, ·) should be highly peaked at ζ ≈ ζ_0
    - loss: −log K_θ(p_ε, ζ_0) + γ · (mass away from ζ_0)

# normal convention: OUTWARD (mesh-native, matches Phase 2 preprocessing).
# p_ε = ζ_0 − ε · n_outward puts p_ε inside Ω for small enough ε.
# See PHILOSOPHY.md §"Convention: Normals".

Adaptive ε:
  A naive "ε = 0.01 · bbox_diagonal" penetrates thin features on MCB
  shapes (gear teeth, thread crests) and produces p_ε outside Ω. The
  adaptive variant shrinks ε until sdf(p_ε) < −eps/2 (strictly inside Ω
  by at least eps/2 from any part of ∂Ω).

  Phase 4 ships with an OPTIONAL sdf_fn param:
    - sdf_fn=None   → fixed eps_target (suitable for tests and convex Ω)
    - sdf_fn given  → adaptive ε via _compute_epsilon_adaptive

  Phase 5's trainer passes the dataset's SDF as sdf_fn automatically.

SDF convention: negative inside Ω, positive outside (standard mesh-to-sdf
convention). Adjust your preprocessing if it uses the opposite sign.

Unsupervised. Prevents the common failure mode of learning a smooth,
non-peaked kernel.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor


def _compute_epsilon_adaptive(
    zeta_0: Tensor,             # (B, 3) boundary points
    normal_out: Tensor,         # (B, 3) outward normals
    sdf_fn: Callable[[Tensor], Tensor],   # (B, 3) → (B,) negative inside
    bbox_diag: float,
    eps_target: float = 0.01,
    max_iter: int = 10,
    eps_floor: float = 1e-5,
) -> Tensor:
    """Shrink ε per boundary point until p_ε = ζ_0 − ε · n_out is strictly
    inside Ω (sdf(p_ε) < −ε/2). Returns a per-entry ε tensor of shape (B,).

    O(max_iter) SDF evaluations, vectorized across the batch.
    """
    eps = torch.full(
        zeta_0.shape[:1], eps_target * bbox_diag,
        device=zeta_0.device, dtype=zeta_0.dtype,
    )                                                     # (B,)
    for _ in range(max_iter):
        p_candidate = zeta_0 - eps.unsqueeze(-1) * normal_out
        sdf_at_p = sdf_fn(p_candidate)                    # (B,) negative inside
        # "too close to another part of ∂Ω" iff sdf_at_p > -eps/2.
        too_close = sdf_at_p > -eps / 2
        # Shrink violators by half; leave others alone.
        eps = torch.where(too_close, eps * 0.5, eps)
        # Clamp to floor; further iterations do nothing once all are < floor.
        eps = eps.clamp(min=eps_floor)
        if not too_close.any():
            break
    return eps


def l_bl(
    model,                         # HarmonicMeasureField (.log_kernel)
    shape_latent,                  # ShapeLatent
    zeta_0: Tensor,                # (B, 3) boundary points
    normal_at_0: Tensor,           # (B, 3) OUTWARD normals
    zeta_surface: Tensor,          # (B, N_surface, 3) surface quadrature
    normal_surface: Tensor,        # (B, N_surface, 3)
    surface_area_weights: Tensor,  # (B, N_surface)
    *,
    epsilon: float = 0.02,         # in normalized [-1,1]^3 coords
    gamma: float = 1.0,            # away-mass regularizer weight
    delta: float = 0.1,            # away-mass distance threshold
    sdf_fn: Callable[[Tensor], Tensor] | None = None,  # optional; enables adaptive ε
    bbox_diag: float = 2.0 * math.sqrt(3),
) -> Tensor:                       # scalar
    B, N_surface, _ = zeta_surface.shape

    # Compute ε: adaptive if sdf_fn provided, else scalar eps_target.
    if sdf_fn is not None:
        eps = _compute_epsilon_adaptive(
            zeta_0, normal_at_0, sdf_fn, bbox_diag, eps_target=epsilon,
        )                                                 # (B,)
    else:
        eps = torch.full(
            zeta_0.shape[:1], epsilon,
            device=zeta_0.device, dtype=zeta_0.dtype,
        )                                                 # (B,)

    # p_ε = ζ_0 − ε · n_outward (places p_ε inside Ω)
    p_eps = zeta_0 - eps.unsqueeze(-1) * normal_at_0      # (B, 3)

    # Combine ζ_0 with zeta_surface for a single forward pass.
    zeta_combined = torch.cat([zeta_0.unsqueeze(1), zeta_surface], dim=1)       # (B, 1+N, 3)
    normal_combined = torch.cat([normal_at_0.unsqueeze(1), normal_surface], dim=1)  # (B, 1+N, 3)
    # Weight for the ζ_0 entry is 0 — it doesn't carry quadrature mass.
    w_target = torch.zeros(B, 1, device=zeta_0.device, dtype=zeta_0.dtype)
    weights_combined = torch.cat([w_target, surface_area_weights], dim=1)       # (B, 1+N)

    log_K_tilde_all = model.log_kernel(
        p_eps, zeta_combined, normal_combined, shape_latent,
    )                                                     # (B, 1+N)

    # Normalize using only the surface samples (skip target).
    log_K_tilde_surf = log_K_tilde_all[:, 1:]             # (B, N)
    log_w_surf = torch.log(surface_area_weights + 1e-30)
    log_Z = torch.logsumexp(log_K_tilde_surf + log_w_surf, dim=-1)              # (B,)

    # Log K at the target (peaked-at-ζ_0 term).
    log_K_target = log_K_tilde_all[:, 0] - log_Z          # (B,)
    peaked_term = -log_K_target.mean()

    # Away-mass regularizer: mass at points far from ζ_0.
    dists = (zeta_surface - zeta_0.unsqueeze(1)).norm(dim=-1)                   # (B, N)
    mask_away = (dists > delta).to(zeta_surface.dtype)
    log_K_surf = log_K_tilde_surf - log_Z.unsqueeze(-1)   # (B, N)
    K_surf = torch.exp(log_K_surf)
    away_mass = (mask_away * K_surf * surface_area_weights).sum(dim=-1).mean()  # scalar

    return peaked_term + gamma * away_mass
