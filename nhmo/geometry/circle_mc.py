"""Walk-on-Spheres Monte Carlo in 2D (circle-based), Warp-accelerated.

Parallel of `nhmo/geometry/sphere_mc.py` for 2D Laplace problems. Each
walk steps by a 2D circle of radius = min(dist-to-box, |sdf|) until
termination at the ε-shell.

Returns (hit_points, hit_mask) exactly as the 3D sampler does; only the
last dim changes (2 vs 3).

Module-level `import warp as wp`: this module requires Warp at import
time, same as sphere_mc.py. Tests skip on non-Warp machines.

Authorization: v2 plan §11.1 authorizes Warp only in
`nhmo/geometry/sphere_mc.py`. Phase 7.0 extends that authorization to
this 2D sibling via the same mechanism — a single-module dep, clearly
scoped.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import Tensor

import warp as wp

wp.init()


# ============================================================ Warp kernel

@wp.func
def _bilinear_sdf(
    sdf: wp.array3d(dtype=float),
    x: float, y: float,
    depth_offset: int, grid_size: int,
) -> float:
    """Bilinear interp at (x, y) ∈ [0, 1]² on a 2D SDF grid stacked per-shape."""
    ix_f = x * float(grid_size - 1)
    iy_f = y * float(grid_size - 1)
    ix_f = wp.clamp(ix_f, 0.0, float(grid_size - 2))
    iy_f = wp.clamp(iy_f, 0.0, float(grid_size - 2))
    ix = int(ix_f)
    iy = int(iy_f)
    fx = ix_f - float(ix)
    fy = iy_f - float(iy)
    iy_off = depth_offset + iy

    v00 = sdf[iy_off,     ix,     0]
    v10 = sdf[iy_off,     ix + 1, 0]
    v01 = sdf[iy_off + 1, ix,     0]
    v11 = sdf[iy_off + 1, ix + 1, 0]

    c0 = v00 * (1.0 - fx) + v10 * fx
    c1 = v01 * (1.0 - fx) + v11 * fx
    return c0 * (1.0 - fy) + c1 * fy


@wp.kernel
def _wos_kernel_2d(
    sdf_flat: wp.array3d(dtype=float),      # (B*G, G, 1) — per-shape 2D slices
    query_points: wp.array(dtype=wp.vec2),
    hits: wp.array(dtype=wp.vec2),
    mask: wp.array(dtype=wp.int32),
    n_walks: int,
    grid_size: int,
    epsilon: float,
    max_steps: int,
    seed: int,
):
    tid = wp.tid()
    shape_idx = tid // n_walks
    depth_offset = shape_idx * grid_size

    qp = query_points[shape_idx]
    px = qp[0]
    py = qp[1]

    state = wp.rand_init(seed, tid)
    terminated = int(0)

    for step in range(max_steps):
        dist_box = wp.min(
            wp.min(px, py),
            wp.min(1.0 - px, 1.0 - py),
        )
        sdf_val = _bilinear_sdf(sdf_flat, px, py, depth_offset, grid_size)
        radius = wp.min(dist_box, wp.abs(sdf_val))

        if radius < epsilon:
            terminated = 1
            break

        phi = 6.28318530718 * wp.randf(state)       # uniform angle on unit circle
        dx = wp.cos(phi)
        dy = wp.sin(phi)

        px = px + radius * dx
        py = py + radius * dy

        px = wp.clamp(px, 0.0, 1.0)
        py = wp.clamp(py, 0.0, 1.0)

    hits[tid] = wp.vec2(px, py)
    mask[tid] = terminated


# ======================================================= Python wrapper

class WoSHitSampler2D:
    """GPU 2D WoS sampler via Warp. Parallel of WoSHitSampler (3D).

    Duck-typed to share the `sample_hits` signature with the 3D sampler
    modulo the last dim of p / hits being 2 instead of 3.
    """

    def __init__(self, epsilon: float = 1e-4, max_steps: int = 256) -> None:
        self.epsilon = epsilon
        self.max_steps = max_steps

    def sample_hits(
        self,
        p: Tensor,                 # (B, 2) query points in [0, 1]²
        sdf_grid: Tensor,          # (B, G, G) SDF grid
        n_walks: int,
        seed: int | None = None,
    ) -> Tuple[Tensor, Tensor]:
        """Run 2D WoS from each query; return raw hits + mask."""
        assert p.dim() == 2 and p.shape[1] == 2, f"p shape {p.shape} != (B, 2)"
        assert sdf_grid.dim() == 3, f"sdf_grid must be (B, G, G); got {sdf_grid.shape}"
        B, G, _ = sdf_grid.shape
        assert sdf_grid.shape == (B, G, G), "2D SDF grid must be square"
        assert p.shape[0] == B
        if p.device.type != "cuda":
            raise RuntimeError(
                "WoSHitSampler2D requires CUDA tensors. For CPU tests, use "
                "tests/fixtures/mock_wos_2d.py::MockWoSHitSampler2D."
            )

        device_str = str(p.device)

        # Stack per-shape slices along depth: (B*G, G, 1)
        sdf_flat_torch = (
            sdf_grid.contiguous().reshape(B * G, G, 1).to(torch.float32)
        )
        sdf_flat_wp = wp.from_torch(sdf_flat_torch, dtype=wp.float32)
        p_wp = wp.from_torch(
            p.contiguous().to(torch.float32).reshape(B, 2), dtype=wp.vec2
        )

        total = B * n_walks
        hits_wp = wp.zeros(total, dtype=wp.vec2, device=device_str)
        mask_wp = wp.zeros(total, dtype=wp.int32, device=device_str)

        if seed is None:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

        wp.launch(
            kernel=_wos_kernel_2d,
            dim=total,
            inputs=[
                sdf_flat_wp, p_wp, hits_wp, mask_wp,
                n_walks, G, float(self.epsilon), int(self.max_steps), int(seed),
            ],
            device=device_str,
        )
        wp.synchronize_device(device_str)

        hits_torch = wp.to_torch(hits_wp).reshape(B, n_walks, 2)
        mask_torch = wp.to_torch(mask_wp).reshape(B, n_walks).to(torch.bool)
        return hits_torch, mask_torch
