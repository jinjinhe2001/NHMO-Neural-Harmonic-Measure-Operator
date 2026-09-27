"""Quadrature helpers for post-hoc solution integration.

    u(p) = Σ_j  w_j · K_θ(p, ζ_j) · h(ζ_j)

This is the only place where h (Dirichlet data) appears in the v2
pipeline — consistent with commitment C3 (BC-agnostic by construction).
The kernel module never sees h; solve() calls self.kernel(...) first and
passes the resulting K through this function.

Precision policy:
  INTERNAL COMPUTATION IS ALWAYS FLOAT64.  `dtype_out` only controls the
  return-type convenience for callers that need to match an existing
  fp32 pipeline; the sum itself happens in fp64 regardless. Rationale:
    - Poisson kernels are singular at ∂Ω with dynamic range 10^4–10^6
      on sharp features; fp32's 23-bit mantissa is insufficient for the
      cancellation in Σ w K h.
    - post_hoc_solve is not on any training hot path (called at eval,
      for MV residual monitoring, and for optional L2(u) aux loss), so
      the fp32→fp64 cost is negligible.
    - A flag for internal precision creates a bug class where training
      loss and eval numbers silently disagree depending on flag state.
      Eliminate the flag; eliminate the bug class.
"""
from __future__ import annotations

import torch
from torch import Tensor


def post_hoc_solve(
    K: Tensor,
    w: Tensor,
    h: Tensor,
    *,
    dtype_out: torch.dtype = torch.float64,
) -> Tensor:
    """u(p) = Σ_j  w_j · K(p, ζ_j) · h(ζ_j).

    Internal computation is ALWAYS float64. `dtype_out` only sets the
    return dtype for caller convenience; it does NOT make the internal
    sum single-precision.

    Args:
        K: (B, N_surface) normalized kernel values.
        w: (B, N_surface) surface-area quadrature weights (Σ_j w_j ≈ |∂Ω|).
        h: (B, N_surface) boundary-condition values at the surface samples.
        dtype_out: return dtype; default float64.

    Returns:
        (B,) tensor of scalar solution values.
    """
    K64 = K.to(torch.float64)
    w64 = w.to(torch.float64)
    h64 = h.to(torch.float64)
    u64 = (w64 * K64 * h64).sum(dim=-1)
    return u64.to(dtype_out)
