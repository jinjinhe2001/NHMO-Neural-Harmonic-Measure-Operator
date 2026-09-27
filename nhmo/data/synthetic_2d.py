"""2D synthetic domains for Phase 7.0.

Disk (2D analog of ball) and annulus. Provides:
  - Closed-form SDFs (via nhmo/geometry/sdf_2d.py)
  - Surface + interior sampling
  - SDF grid rasterization in WoS frame [0, 1]²

Coordinate system (matches 3D `synthetic.py`):
  - Model frame: [-1, 1]² — matches FourierFeatures range.
  - WoS frame:   [0, 1]²  — legacy Warp convention.
  - Transform:   wos = (model + 1) / 2.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from nhmo.geometry.sdf_2d import annulus_sdf, disk_sdf


DISK_RADIUS = 0.9           # keeps boundary inside [-1, 1]² Fourier range


# --------------------------------------------------------- SDF convenience

def disk_sdf_model_frame(pts: Tensor, radius: float = DISK_RADIUS) -> Tensor:
    return disk_sdf(pts, radius=radius)


def model_frame_to_wos_frame(pts_model: Tensor) -> Tensor:
    return (pts_model + 1.0) * 0.5


def wos_frame_to_model_frame(pts_wos: Tensor) -> Tensor:
    return pts_wos * 2.0 - 1.0


def rasterize_disk_sdf_grid(
    G: int, radius: float = DISK_RADIUS, device: str = "cpu"
) -> Tensor:
    """SDF of the disk on a G×G grid in WoS frame [0, 1]²."""
    coords = torch.linspace(0.0, 1.0, G, device=device)
    iy, ix = torch.meshgrid(coords, coords, indexing="ij")
    pts_wos = torch.stack([ix, iy], dim=-1)                    # (G, G, 2)
    pts_model = wos_frame_to_model_frame(pts_wos)
    return disk_sdf(pts_model, radius=radius)                  # (G, G)


def rasterize_annulus_sdf_grid(
    G: int, r_in: float, r_out: float, device: str = "cpu"
) -> Tensor:
    coords = torch.linspace(0.0, 1.0, G, device=device)
    iy, ix = torch.meshgrid(coords, coords, indexing="ij")
    pts_wos = torch.stack([ix, iy], dim=-1)
    pts_model = wos_frame_to_model_frame(pts_wos)
    return annulus_sdf(pts_model, r_in=r_in, r_out=r_out)


# --------------------------------------------------------- surface sampling

def disk_surface_samples(
    n: int, radius: float = DISK_RADIUS, device: str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Uniform arc-length samples on the disk boundary + outward normals.

    Returns:
        points:  (n, 2) on ∂D(0, radius) — model frame
        normals: (n, 2) outward unit normals = points / |points|
    """
    theta = torch.rand(n, device=device) * 2.0 * math.pi
    v = torch.stack([theta.cos(), theta.sin()], dim=-1)         # (n, 2) unit circle
    points = v * radius
    normals = v
    return points, normals


def annulus_surface_samples(
    n: int, r_in: float, r_out: float, device: str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Stratified samples on the annulus boundary. Proportionally allocates
    n_out = n · r_out / (r_in + r_out) to the outer circle (arc-length
    proportional), remainder to the inner circle. Outer normal = radial
    outward; inner normal = radial INWARD (outward w.r.t. annulus domain).
    """
    total = r_in + r_out
    n_out = int(round(n * r_out / total))
    n_in = n - n_out
    theta_out = torch.rand(n_out, device=device) * 2.0 * math.pi
    v_out = torch.stack([theta_out.cos(), theta_out.sin()], dim=-1)
    p_out = v_out * r_out
    theta_in = torch.rand(n_in, device=device) * 2.0 * math.pi
    v_in = torch.stack([theta_in.cos(), theta_in.sin()], dim=-1)
    p_in = v_in * r_in
    points = torch.cat([p_out, p_in], dim=0)
    normals = torch.cat([v_out, -v_in], dim=0)                  # outer: +r, inner: -r
    return points, normals


# --------------------------------------------------------- interior sampling

def disk_interior_samples(
    n: int,
    radius: float = DISK_RADIUS,
    device: str = "cpu",
    band: tuple[float, float] | None = None,
) -> Tensor:
    """Uniform samples inside the disk (model frame). Optional ``band``
    restricts |p| to [band_lo · r, band_hi · r]."""
    lo = 0.0 if band is None else band[0] * radius
    hi = radius if band is None else band[1] * radius
    theta = torch.rand(n, device=device) * 2.0 * math.pi
    v = torch.stack([theta.cos(), theta.sin()], dim=-1)
    # Uniform in area within annular shell: r = sqrt(lo² + u*(hi² - lo²))
    u = torch.rand(n, device=device)
    r = (lo ** 2 + u * (hi ** 2 - lo ** 2)).sqrt()
    return v * r.unsqueeze(-1)


def annulus_interior_samples(
    n: int, r_in: float, r_out: float, device: str = "cpu",
) -> Tensor:
    """Uniform samples inside the annulus (r_in < |p| < r_out)."""
    theta = torch.rand(n, device=device) * 2.0 * math.pi
    v = torch.stack([theta.cos(), theta.sin()], dim=-1)
    u = torch.rand(n, device=device)
    r = (r_in ** 2 + u * (r_out ** 2 - r_in ** 2)).sqrt()
    return v * r.unsqueeze(-1)


# --------------------------------------------------------- shape_ctx helpers

def disk_shape_ctx(
    n_surface: int = 256,
    n_interior_anchors: int = 128,
    radius: float = DISK_RADIUS,
    device: str = "cpu",
) -> dict:
    surf, normals = disk_surface_samples(n_surface, radius, device)
    interior = disk_interior_samples(n_interior_anchors, radius, device=device)
    return {
        "surface_points": surf.unsqueeze(0),
        "surface_normals": normals.unsqueeze(0),
        "interior_points": interior.unsqueeze(0),
    }


def disk_bbox_diag_model_frame(radius: float = DISK_RADIUS) -> float:
    return 2.0 * radius * math.sqrt(2.0)
