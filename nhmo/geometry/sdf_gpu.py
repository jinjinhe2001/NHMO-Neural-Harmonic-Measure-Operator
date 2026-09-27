"""GPU signed-distance rasterization for triangle meshes.

Inference-only replacement for the CPU `trimesh.proximity.signed_distance`
call in `MCBShapeGenerator._rasterize_sdf_grid`. Two parts:

  - Unsigned distance: closest point on each triangle (Ericson, Real-Time
    Collision Detection, Sec. 5.1.5), vectorized over (point x face) chunks.
  - Sign: ray-crossing parity along one fixed, slightly jittered direction
    (Moller-Trumbore intersection). For watertight meshes this matches the
    inside test of `trimesh.contains` and is robust near thin features, where
    a nearest-face-normal sign flips.

Sign convention follows nhmo: negative inside the mesh, positive outside.

Validation (MCB-B nut/motor, 64^3 grids): max sign disagreement with trimesh of
at most 1 grid point out of 262k and a 99th-percentile absolute difference of
about 1.6e-7 (see tests/test_fast_inference_parity.py).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

# Fixed, slightly jittered ray direction for the parity test. Axis-aligned
# rays hit mesh edges and vertices of CAD parts far more often.
_RAY_DIR = (0.57735027, 0.57738264, 0.57729704)


def closest_point_triangle(p: Tensor, a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """Closest point on triangles (a, b, c) to points p.

    p: (n, 1, 3); a, b, c: (1, m, 3). Returns (n, m, 3).
    """
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = (ab * ap).sum(-1)
    d2 = (ac * ap).sum(-1)
    bp = p - b
    d3 = (ab * bp).sum(-1)
    d4 = (ac * bp).sum(-1)
    cp = p - c
    d5 = (ab * cp).sum(-1)
    d6 = (ac * cp).sum(-1)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    denom_abc = va + vb + vc
    v_abc = vb / denom_abc.clamp(min=1e-30)
    w_abc = vc / denom_abc.clamp(min=1e-30)
    pt_face = a + ab * v_abc.unsqueeze(-1) + ac * w_abc.unsqueeze(-1)

    v_ab = (d1 / (d1 - d3).clamp(min=1e-30)).clamp(0, 1)
    pt_ab = a + ab * v_ab.unsqueeze(-1)
    w_ac = (d2 / (d2 - d6).clamp(min=1e-30)).clamp(0, 1)
    pt_ac = a + ac * w_ac.unsqueeze(-1)
    num_bc = d4 - d3
    den_bc = ((d4 - d3) + (d5 - d6)).clamp(min=1e-30)
    w_bc = (num_bc / den_bc).clamp(0, 1)
    pt_bc = b + (c - b) * w_bc.unsqueeze(-1)

    out = pt_face
    reg_bc = (va <= 0) & (num_bc >= 0) & ((d5 - d6) >= 0)
    out = torch.where(reg_bc.unsqueeze(-1), pt_bc, out)
    reg_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    out = torch.where(reg_ac.unsqueeze(-1), pt_ac, out)
    reg_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    out = torch.where(reg_ab.unsqueeze(-1), pt_ab, out)
    out = torch.where(((d1 <= 0) & (d2 <= 0)).unsqueeze(-1), a.expand_as(out), out)
    out = torch.where(((d3 >= 0) & (d4 <= d3)).unsqueeze(-1), b.expand_as(out), out)
    out = torch.where(((d6 >= 0) & (d5 <= d6)).unsqueeze(-1), c.expand_as(out), out)
    return out


def gpu_signed_distance(
    pts: Tensor,
    v0: Tensor,
    v1: Tensor,
    v2: Tensor,
    pt_chunk: int = 2048,
    face_chunk: int = 8192,
) -> Tensor:
    """Signed distance from pts (N, 3) to the triangle soup (v0, v1, v2), each (F, 3).

    Returns (N,) with the nhmo convention: negative inside, positive outside.
    """
    dev = pts.device
    N = pts.shape[0]
    F = v0.shape[0]
    d = torch.tensor(_RAY_DIR, device=dev, dtype=pts.dtype)
    d = d / d.norm()
    e1 = v1 - v0
    e2 = v2 - v0
    h = torch.linalg.cross(d.expand(F, 3), e2)
    a = (e1 * h).sum(-1)
    valid = a.abs() > 1e-10
    inv_a = torch.where(valid, 1.0 / a, torch.zeros_like(a))

    out = torch.empty(N, device=dev, dtype=pts.dtype)
    for s in range(0, N, pt_chunk):
        e = min(N, s + pt_chunk)
        n = e - s
        p = pts[s:e].unsqueeze(1)                                       # (n, 1, 3)
        bd2 = torch.full((n,), float("inf"), device=dev, dtype=pts.dtype)
        hits = torch.zeros(n, dtype=torch.long, device=dev)
        for fs in range(0, F, face_chunk):
            fe = min(F, fs + face_chunk)
            cp = closest_point_triangle(p, v0[fs:fe].unsqueeze(0),
                                        v1[fs:fe].unsqueeze(0), v2[fs:fe].unsqueeze(0))
            d2 = ((p - cp) ** 2).sum(-1)                                # (n, m)
            bd2 = torch.minimum(bd2, d2.min(dim=1).values)
            # Ray-parity contribution of this face block.
            sv = p - v0[fs:fe].unsqueeze(0)                             # (n, m, 3)
            u = (sv * h[fs:fe].unsqueeze(0)).sum(-1) * inv_a[fs:fe].unsqueeze(0)
            q = torch.linalg.cross(sv, e1[fs:fe].unsqueeze(0).expand_as(sv))
            vv = (q * d.view(1, 1, 3)).sum(-1) * inv_a[fs:fe].unsqueeze(0)
            tt = (q * e2[fs:fe].unsqueeze(0)).sum(-1) * inv_a[fs:fe].unsqueeze(0)
            hit = (valid[fs:fe].unsqueeze(0)
                   & (u >= 0) & (u <= 1) & (vv >= 0) & ((u + vv) <= 1) & (tt > 1e-9))
            hits += hit.sum(dim=1)
        dist = bd2.clamp(min=0).sqrt()
        inside = (hits % 2) == 1
        out[s:e] = torch.where(inside, -dist, dist)
    return out


def wos_grid_points(grid_size: int, device, dtype=torch.float32) -> Tensor:
    """Model-frame coordinates of the WoS-frame SDF grid, flattened to (G^3, 3).

    Same point ordering as `MCBShapeGenerator._rasterize_sdf_grid`: the grid is
    indexed [z, y, x] and point (ix, iy, iz) lives at 2 * (x, y, z) - 1.
    """
    coords = torch.linspace(0.0, 1.0, grid_size, device=device, dtype=dtype)
    iz, iy, ix = torch.meshgrid(coords, coords, coords, indexing="ij")
    return torch.stack([ix, iy, iz], dim=-1).reshape(-1, 3) * 2.0 - 1.0


def rasterize_sdf_grid_gpu(mesh, grid_size: int, device) -> Tensor:
    """GPU counterpart of `MCBShapeGenerator._rasterize_sdf_grid`.

    mesh: trimesh.Trimesh in the model frame. Returns a (G, G, G) float32 grid
    in the WoS frame, negative inside.
    """
    device = torch.device(device)
    V = torch.from_numpy(np.asarray(mesh.vertices, dtype=np.float32)).to(device)
    Fc = torch.from_numpy(np.asarray(mesh.faces, dtype=np.int64)).to(device)
    pts = wos_grid_points(grid_size, device)
    sd = gpu_signed_distance(pts, V[Fc[:, 0]], V[Fc[:, 1]], V[Fc[:, 2]])
    return sd.reshape(grid_size, grid_size, grid_size).contiguous()
