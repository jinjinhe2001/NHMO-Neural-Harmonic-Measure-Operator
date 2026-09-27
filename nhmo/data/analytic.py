"""Analytic Poisson kernels for Stage-0 pretraining and S-4 sanity.

Per v2 plan §8.1. Phase 6 ships:
  - SphereHarmonicMeasure: full impl with (radius, center, rotation)
    parameterization so the §8.1 curriculum (random rotations + scales ∈
    [0.5, 2.0] + translations) can use a single class.
  - HalfSpaceHarmonicMeasure: formula shipped; domain generator + dataset
    wiring deferred to Phase 6.1 (see TODO_POST_PHASE_6.md).
  - CubeHarmonicMeasure: stubbed (Phase 6.1 / Phase 7; see TODO_POST_PHASE_6.md).

LossRegistry-compatibility:
  - .normalize = "hard" on all three classes (they are exactly normalized
    on their respective domains; L_Z = 0).
  - .log_kernel and .kernel signatures match HarmonicMeasureField's
    (duck-typed). `shape_latent` and `surface_area_weights` args are
    accepted and ignored (analytic kernels are closed-form).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class SphereHarmonicMeasure(nn.Module):
    """Analytic Poisson kernel for a ball Ω = {x : |R^T (x - c)| < r}.

    Parameters:
      radius: r (ball radius, scalar or per-batch tensor)
      center: c (ball center in model frame, (3,) or (B, 3))
      rotation: R (3×3 rotation matrix or (B, 3, 3); optional, defaults to identity)

    For the SPHERE the rotation is mathematically a no-op (the ball is
    isotropic), but the shape_latent encoder sees distinct surface-point
    configurations — §8.1 curriculum applies rotation to test
    encoder rotation-equivariance.

    Formula (in the ball's canonical frame, after un-translating and
    un-rotating to the unit ball):
        P(p̂, ζ̂) = (r² − |p̂|²) / (4π · r · |p̂ − ζ̂|³)
    where p̂ = R^T (p − c), ζ̂ = R^T (ζ − c). The kernel is
    intrinsically scalar-valued, so we only need to compute distances
    post-transform; |p̂ − ζ̂| = |p − ζ| (isometry).

    Thus for a rotated/translated ball:
        P(p, ζ) = (r² − |R^T(p − c)|²) / (4π · r · |p − ζ|³)

    normalize = "hard" (analytic kernel is exactly normalized on ∂Ω).
    """
    normalize: str = "hard"

    def __init__(
        self,
        radius: float | Tensor = 1.0,
        center: Tensor | None = None,
        rotation: Tensor | None = None,
    ) -> None:
        super().__init__()
        # One trainable dummy so LossRegistry's any_param.sum() trick works
        # when this kernel is used in tests. Zero gradient by construction.
        self._dummy = nn.Parameter(torch.zeros(1))
        self.radius = radius
        self.center = center
        self.rotation = rotation

    def _canonical_p(self, p: Tensor) -> Tensor:
        """p (B, 3) → p̂ = R^T (p − c) for analytic formula."""
        if self.center is not None:
            c = self.center
            if c.dim() == 1:
                c = c.unsqueeze(0).expand_as(p)
            p = p - c
        if self.rotation is not None:
            R = self.rotation
            if R.dim() == 2:
                # Same rotation for all batch entries
                p = p @ R        # R^T @ col-vec == row-vec @ R
            else:
                # per-batch rotation: (B, 3, 3)
                p = torch.einsum("bij,bi->bj", R, p)
        return p

    def log_kernel(
        self,
        p: Tensor,                   # (B, 3)
        zeta: Tensor,                # (B, N, 3)
        zeta_normal: Tensor,         # (B, N, 3) — unused for sphere
        shape_latent=None,
    ) -> Tensor:
        p_canon = self._canonical_p(p)
        if isinstance(self.radius, Tensor):
            r = self.radius.view(-1, 1)          # (B, 1)
        else:
            r = torch.full(
                (p.shape[0], 1), float(self.radius),
                device=p.device, dtype=p.dtype,
            )
        r2 = r * r
        p_norm_sq = (p_canon * p_canon).sum(dim=-1, keepdim=True).clamp(max=r2 - 1e-8)
        # |p − ζ| in world frame == |p̂ − ζ̂|.
        diff = zeta - p.unsqueeze(1)             # (B, N, 3)
        dist = diff.norm(dim=-1).clamp(min=1e-6)  # (B, N)
        num = r2 - p_norm_sq                     # (B, 1)
        denom = 4.0 * math.pi * r * dist.pow(3)  # (B, N)
        K = num / denom                          # (B, N)
        return torch.log(K.clamp(min=1e-30)) + 0.0 * self._dummy.sum()

    def kernel(
        self,
        p: Tensor,
        zeta: Tensor,
        zeta_normal: Tensor,
        shape_latent=None,
        surface_area_weights: Tensor | None = None,
    ) -> Tensor:
        return torch.exp(self.log_kernel(p, zeta, zeta_normal, shape_latent))


class HalfSpaceHarmonicMeasure(nn.Module):
    """Analytic Poisson kernel for upper half-space Ω = {z > 0}.

    P(p, ζ) = z_p / (2π · |p − ζ|³)
    with p = (x_p, y_p, z_p) for z_p > 0 and ζ = (x_ζ, y_ζ, 0).

    Phase 6 ships the FORMULA ONLY. Domain generator + dataset wiring +
    training integration are deferred to Phase 6.1 because the half-space
    is infinite in x/y/upper-z and bounding it (slab truncation) injects
    an incorrect inductive bias. See TODO_POST_PHASE_6.md for revisit
    criteria.

    normalize = "hard".
    """
    normalize: str = "hard"

    def __init__(self) -> None:
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1))

    def log_kernel(
        self, p: Tensor, zeta: Tensor, zeta_normal: Tensor, shape_latent=None,
    ) -> Tensor:
        z_p = p[..., 2:3].clamp(min=1e-8)        # (B, 1)
        diff = zeta - p.unsqueeze(1)
        dist = diff.norm(dim=-1).clamp(min=1e-6)
        K = z_p / (2.0 * math.pi * dist.pow(3))
        return torch.log(K.clamp(min=1e-30)) + 0.0 * self._dummy.sum()

    def kernel(
        self, p: Tensor, zeta: Tensor, zeta_normal: Tensor,
        shape_latent=None, surface_area_weights: Tensor | None = None,
    ) -> Tensor:
        return torch.exp(self.log_kernel(p, zeta, zeta_normal, shape_latent))


class CubeHarmonicMeasure(nn.Module):
    """Series-form Poisson kernel for the unit cube. Phase 6.1 / Phase 7.

    3D cube Poisson kernel is a triple Fourier/Dirichlet series whose
    truncation error is non-uniform (worst at corners). Deferred; see
    TODO_POST_PHASE_6.md for revisit criteria and an L-shaped-2D-via-
    Schwarz-Christoffel alternative if corner-sharpness pretraining is
    later motivated.
    """
    normalize: str = "hard"

    def __init__(self, n_harmonics: int = 20) -> None:
        super().__init__()
        self.n_harmonics = n_harmonics

    def log_kernel(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError(
            "CubeHarmonicMeasure: deferred to Phase 6.1 / Phase 7. See "
            "TODO_POST_PHASE_6.md for revisit criteria."
        )
