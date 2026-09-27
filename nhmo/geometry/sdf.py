"""SDF utilities: trilinear interpolation on grid-backed SDFs.

Used by Phase 5's WoSDataset to query SDF values at arbitrary points
inside a shape's grid (for adaptive ε in L_BL, for MV's sdf_at_p).
Phase 5 smoke test uses analytic SDF (nhmo.data.synthetic.ball_sdf);
Phase 7's MCB loader uses grid SDFs and relies on this module.
"""
from __future__ import annotations

import torch
from torch import Tensor


def trilinear_interp(
    grid: Tensor,                  # (B, G, G, G) or (G, G, G) SDF grid
    pts_01: Tensor,                # (B, N, 3) or (N, 3) in [0, 1]³
) -> Tensor:
    """Sample `grid` at continuous [0, 1]³ coordinates via trilinear interp.

    Grid convention: `grid[..., iz, iy, ix]` with axes (depth=iz, row=iy,
    col=ix). Caller supplies `pts_01` in (x, y, z) order matching
    (ix, iy, iz).

    Returns a tensor one rank lower than `pts_01`: if pts is (B, N, 3)
    output is (B, N); if pts is (N, 3) output is (N,).
    """
    pts_batched = pts_01.dim() == 3
    grid_batched = grid.dim() == 4

    if not pts_batched:
        pts_01 = pts_01.unsqueeze(0)    # (1, N, 3)
    if not grid_batched:
        grid = grid.unsqueeze(0)        # (1, G, G, G)

    B, N, _ = pts_01.shape
    G = grid.shape[-1]
    assert grid.shape == (B, G, G, G), f"grid shape {grid.shape}"

    # Map [0, 1] → [0, G-1]
    f = (pts_01 * (G - 1)).clamp(0.0, float(G - 1))             # (B, N, 3)
    i = f.floor().clamp(max=G - 2).long()                       # (B, N, 3)
    frac = (f - i.to(f.dtype)).clamp(0.0, 1.0)                  # (B, N, 3)

    ix = i[..., 0]; iy = i[..., 1]; iz = i[..., 2]
    fx = frac[..., 0]; fy = frac[..., 1]; fz = frac[..., 2]

    b_idx = torch.arange(B, device=pts_01.device).unsqueeze(-1).expand(B, N)

    def g(z, y, x):
        return grid[b_idx, z, y, x]  # (B, N)

    v000 = g(iz,     iy,     ix    )
    v100 = g(iz,     iy,     ix + 1)
    v010 = g(iz,     iy + 1, ix    )
    v110 = g(iz,     iy + 1, ix + 1)
    v001 = g(iz + 1, iy,     ix    )
    v101 = g(iz + 1, iy,     ix + 1)
    v011 = g(iz + 1, iy + 1, ix    )
    v111 = g(iz + 1, iy + 1, ix + 1)

    c00 = v000 * (1 - fx) + v100 * fx
    c10 = v010 * (1 - fx) + v110 * fx
    c01 = v001 * (1 - fx) + v101 * fx
    c11 = v011 * (1 - fx) + v111 * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    out = c0 * (1 - fz) + c1 * fz                               # (B, N)

    if not pts_batched:
        out = out.squeeze(0)
    return out
