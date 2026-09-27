"""2D signed distance functions for Phase 7.0 analytic domains.

Conventions (match 3D `sdf.py`):
  - SDF is negative INSIDE the domain Ω, positive outside, zero on ∂Ω.
  - Inputs are arbitrary-rank tensors with last dim = 2.

Implemented shapes:
  - `disk_sdf(p, radius, center)`
  - `annulus_sdf(p, r_in, r_out, center)`
  - `polygon_sdf(p, polygon)`  — for MNIST-digit contours; accepts a
    (V, 2) vertex array and computes signed distance via distance-to-
    nearest-segment plus inside-outside test via winding number.

No `polygon_sdf` batched-vectorization over multiple polygons; that's
an optimization left for when MNIST training shows it's the bottleneck.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def disk_sdf(p: Tensor, radius: float = 1.0, center: Optional[Tensor] = None) -> Tensor:
    """SDF of a disk {x ∈ ℝ² : |x - c| < r}. Negative inside."""
    if center is not None:
        p = p - center
    return p.norm(dim=-1) - float(radius)


def annulus_sdf(
    p: Tensor,
    r_in: float,
    r_out: float,
    center: Optional[Tensor] = None,
) -> Tensor:
    """SDF of an annulus {x : r_in < |x - c| < r_out}.

    Interior points satisfy r_in < |x| < r_out; outside is the union of
    the inner disk and the exterior of the outer disk.
    """
    if not (0.0 < r_in < r_out):
        raise ValueError(f"Need 0 < r_in < r_out; got r_in={r_in}, r_out={r_out}")
    if center is not None:
        p = p - center
    r = p.norm(dim=-1)
    # Distance to the nearest boundary; sign: inside annulus if r_in < r < r_out.
    # d_outer = r - r_out: positive outside the outer disk
    # d_inner = r_in - r:  positive inside the inner disk
    # Annulus SDF = max(d_outer, d_inner) — negative iff r_in < r < r_out.
    return torch.maximum(r - float(r_out), float(r_in) - r)


def polygon_sdf(p: Tensor, polygon: Tensor) -> Tensor:
    """Signed distance from each p to a simple 2D polygon.

    Args:
        p:        (..., 2) query points
        polygon:  (V, 2) vertices in order (closed implicitly — last vertex
                  connects back to the first). Must be a simple polygon
                  (no self-intersection); orientation can be CW or CCW.

    Returns:
        sdf: (...) signed distance; negative inside the polygon.

    Uses (distance-to-nearest-edge) + (inside/outside test via ray-cast
    parity count). For MNIST digit contours (~50 vertices) this is fast
    enough even unvectorized across polygons.
    """
    if polygon.dim() != 2 or polygon.shape[-1] != 2:
        raise ValueError(f"polygon must be (V, 2); got shape {tuple(polygon.shape)}")
    V = polygon.shape[0]
    if V < 3:
        raise ValueError(f"polygon needs at least 3 vertices; got {V}")

    orig_shape = p.shape[:-1]
    p_flat = p.reshape(-1, 2)                         # (P, 2)
    P = p_flat.shape[0]

    # Edges: (V, 2) start-points a, (V, 2) end-points b. edge i = (a_i, b_i).
    a = polygon
    b = torch.roll(polygon, shifts=-1, dims=0)        # (V, 2)
    edge = b - a                                      # (V, 2)

    # For each (query p, edge i): project p onto segment (a, b), clamp
    # t ∈ [0, 1]. closest = a + t * edge. Distance = |p - closest|.
    p_exp = p_flat.unsqueeze(1)                       # (P, 1, 2)
    a_exp = a.unsqueeze(0)                            # (1, V, 2)
    edge_exp = edge.unsqueeze(0)                      # (1, V, 2)
    edge_len_sq = (edge_exp * edge_exp).sum(dim=-1)   # (1, V)
    edge_len_sq = edge_len_sq.clamp(min=1e-12)
    t = ((p_exp - a_exp) * edge_exp).sum(dim=-1) / edge_len_sq   # (P, V)
    t = t.clamp(0.0, 1.0)
    closest = a_exp + t.unsqueeze(-1) * edge_exp      # (P, V, 2)
    dist_sq = ((p_exp - closest) ** 2).sum(dim=-1)    # (P, V)
    min_dist_sq, _ = dist_sq.min(dim=-1)              # (P,)
    abs_dist = min_dist_sq.sqrt()                     # (P,)

    # Inside/outside via ray-cast parity (horizontal ray from p to +∞).
    # Edge i crosses the horizontal line at p.y iff (a_y > p_y) XOR (b_y > p_y).
    # The x-coord at the crossing: a_x + t * edge_x where
    # t = (p_y - a_y) / (b_y - a_y).
    py = p_flat[:, 1].unsqueeze(1)                     # (P, 1)
    ay = a[:, 1].unsqueeze(0)                          # (1, V)
    by = b[:, 1].unsqueeze(0)                          # (1, V)
    ax = a[:, 0].unsqueeze(0)                          # (1, V)
    bx = b[:, 0].unsqueeze(0)                          # (1, V)
    condition = (ay > py) ^ (by > py)                  # (P, V) bool
    # Guard divide-by-zero on horizontal edges (ay == by): the XOR above
    # is False for horizontal edges regardless of py, so they don't contribute.
    denom = (by - ay).clamp(min=1e-30)  # positive clamp to avoid 0-div; sign
                                        # may flip but combined with condition
                                        # checks, the parity test stays correct.
    # Need signed denom for the x-coord formula; but clamping min to
    # positive epsilon is wrong when by < ay. Use masked compute:
    safe_denom = torch.where(
        (by - ay).abs() > 1e-30,
        by - ay,
        torch.ones_like(by - ay),
    )
    x_at = ax + (py - ay) / safe_denom * (bx - ax)     # (P, V)
    crosses_ray = condition & (x_at > p_flat[:, 0].unsqueeze(1))
    # Parity = number of ray crossings mod 2
    parity = crosses_ray.sum(dim=-1) % 2               # (P,) int
    inside = parity == 1                                # (P,) bool

    sdf = torch.where(inside, -abs_dist, abs_dist)
    return sdf.reshape(orig_shape)
