"""Synthetic on-the-fly analytic shapes for Phase 5 smoke training.

Phase 5 trains on a single domain: the unit ball Ω = { x ∈ ℝ³ : |x| < R }
with R = 0.9 (chosen slightly < 1 to keep boundary points inside the
[-1, 1]³ FourierFeatures range). The ball has an analytic SDF and
surface sampling is a normalized Gaussian — no mesh preprocessing.

Coordinate system:
  - Model inputs are in [-1, 1]³ (Phase 2's FourierFeatures range).
  - Warp WoS operates in [0, 1]³ (legacy convention). This module
    provides both views: `*_model_frame` functions return [-1, 1]³;
    `*_wos_frame` functions return [0, 1]³. Conversion: wos = (model + 1) / 2.

Phase 6 (§8) extends to half-space + cube + random rotations/scales.
Phase 5 is ball-only.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor


BALL_RADIUS = 0.9  # in [-1, 1]³ model frame; Fourier range allows ~1.05


# ----------------------------------------------------------------- SDF

def ball_sdf(pts: Tensor, radius: float = BALL_RADIUS) -> Tensor:
    """SDF of the unit ball in the MODEL frame [-1, 1]³.

    Convention: negative inside Ω (interior), positive outside.
    `sdf(p) = |p| - R`.
    """
    return pts.norm(dim=-1) - radius


def model_frame_to_wos_frame(pts_model: Tensor) -> Tensor:
    """[-1, 1]³ → [0, 1]³. `wos = (model + 1) / 2`."""
    return (pts_model + 1.0) * 0.5


def wos_frame_to_model_frame(pts_wos: Tensor) -> Tensor:
    """[0, 1]³ → [-1, 1]³. `model = 2 * wos - 1`."""
    return pts_wos * 2.0 - 1.0


def rasterize_ball_sdf_grid(G: int, radius: float = BALL_RADIUS, device: str = "cpu") -> Tensor:
    """Rasterize the unit-ball SDF onto a G³ grid in the WoS frame [0, 1]³.

    Returns: (G, G, G) tensor where grid[iz, iy, ix] = sdf at
    (ix / (G-1), iy / (G-1), iz / (G-1)) in WoS frame.

    The grid is in WoS frame (for Warp kernel compatibility). SDF values
    are also in WoS frame units: sdf_wos(p_wos) = 2 * sdf_model(2·p_wos − 1),
    but we just compute it directly to avoid scale confusion:
    sdf(p_wos) = | 2·p_wos − 1 | − radius.
    """
    coords = torch.linspace(0.0, 1.0, G, device=device)
    iz, iy, ix = torch.meshgrid(coords, coords, coords, indexing="ij")
    pts_wos = torch.stack([ix, iy, iz], dim=-1)                     # (G, G, G, 3)
    pts_model = wos_frame_to_model_frame(pts_wos)
    return ball_sdf(pts_model, radius)


# ----------------------------------------------------------------- surface sampling

def ball_surface_samples(n: int, radius: float = BALL_RADIUS, device: str = "cpu") -> tuple[Tensor, Tensor]:
    """Uniform samples on the ball surface (in MODEL frame) + outward normals.

    Returns:
        points: (n, 3) on ∂B(0, radius)
        normals: (n, 3) outward unit normals (= points / |points|)
    """
    v = torch.randn(n, 3, device=device)
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
    points = v * radius
    normals = v                                                     # outward unit normal
    return points, normals


def ball_interior_samples(
    n: int, radius: float = BALL_RADIUS, device: str = "cpu",
    band: tuple[float, float] | None = None,
) -> Tensor:
    """Uniform samples inside the ball (in MODEL frame).

    Optional band restricts |p| to [band_low, band_high] × radius.
    """
    low = 0.0 if band is None else band[0] * radius
    high = radius if band is None else band[1] * radius
    v = torch.randn(n, 3, device=device)
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
    u = torch.rand(n, 1, device=device)
    r = (low ** 3 + u * (high ** 3 - low ** 3)) ** (1.0 / 3.0)
    return v * r


# ----------------------------------------------------------------- shape context

def unit_ball_shape_ctx(
    n_surface: int = 512,
    n_interior_anchors: int = 512,
    device: str = "cpu",
) -> dict:
    """Build the `shape_ctx` dict consumed by TransolverEncoder.

    Returns a dict with (1, *, 3) tensors — single-shape batch.
    """
    surf, normals = ball_surface_samples(n_surface, device=device)
    interior = ball_interior_samples(n_interior_anchors, device=device)
    return {
        "surface_points": surf.unsqueeze(0),
        "surface_normals": normals.unsqueeze(0),
        "interior_points": interior.unsqueeze(0),
    }


def ball_bbox_diag_model_frame(radius: float = BALL_RADIUS) -> float:
    """Bounding-box diagonal for L_BL ε scaling."""
    return 2.0 * radius * math.sqrt(3.0)
