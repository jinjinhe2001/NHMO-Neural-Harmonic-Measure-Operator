"""Walk-on-Spheres (WoS) Monte Carlo, GPU-accelerated via Warp.

Ports the WoS math from `legacy/v1/nhm/wos/laplace_3d.py` (reference only;
NOT imported — see v2 plan §12 A3) and changes the return type from
histogram-over-K-boundary-points to raw (hit_points, hit_mask).

Key differences from legacy:
  - Returns `(hit_points: (B, n_walks, 3), hit_mask: (B, n_walks))` instead
    of a histogram. No atomicAdd into boundary-point buckets — we record
    the raw termination coordinate.
  - `hit_mask[b, i]` is True if the walk terminated at the ε-shell, False
    if max_steps exceeded (rare with reasonable max_steps).

v2 plan amendment (§11.1): Warp is authorized only in this module; no
other Warp use in `nhmo/`.

Coordinate convention: the Warp kernel operates in **[0, 1]³** (matches
the legacy convention). The caller (dataset / trainer) is responsible
for translating between the [0, 1] WoS frame and the [-1, 1] model frame.

CPU path: this module requires Warp; attempting to import it on a machine
without Warp installed raises ImportError at module load. Tests that need
a CPU-runnable hit sampler should import `MockWoSHitSampler` from
`tests/fixtures/mock_wos.py` instead. `WoSHitSampler` and `MockWoSHitSampler`
are duck-typed (share the `sample_hits` signature) but are NOT
interchangeable in Trainer — Trainer hardcodes `WoSHitSampler` and raises
`EnvironmentError` at init on non-Warp machines.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import Tensor

import warp as wp

wp.init()


# ======================================================================
# Warp kernel: ported from legacy/v1/nhm/wos/laplace_3d.py:82–135
# with histogram output replaced by raw hit coordinates.
# ======================================================================

@wp.func
def _trilinear_sdf(
    sdf: wp.array3d(dtype=float),
    x: float, y: float, z: float,
    depth_offset: int, grid_size: int,
) -> float:
    """Trilinear interp at (x, y, z) ∈ [0, 1]³ with per-shape depth offset."""
    ix_f = x * float(grid_size - 1)
    iy_f = y * float(grid_size - 1)
    iz_f = z * float(grid_size - 1)

    ix_f = wp.clamp(ix_f, 0.0, float(grid_size - 2))
    iy_f = wp.clamp(iy_f, 0.0, float(grid_size - 2))
    iz_f = wp.clamp(iz_f, 0.0, float(grid_size - 2))

    ix = int(ix_f)
    iy = int(iy_f)
    iz = int(iz_f)

    fx = ix_f - float(ix)
    fy = iy_f - float(iy)
    fz = iz_f - float(iz)

    iz_off = depth_offset + iz

    v000 = sdf[iz_off,     iy,     ix]
    v100 = sdf[iz_off,     iy,     ix + 1]
    v010 = sdf[iz_off,     iy + 1, ix]
    v110 = sdf[iz_off,     iy + 1, ix + 1]
    v001 = sdf[iz_off + 1, iy,     ix]
    v101 = sdf[iz_off + 1, iy,     ix + 1]
    v011 = sdf[iz_off + 1, iy + 1, ix]
    v111 = sdf[iz_off + 1, iy + 1, ix + 1]

    c00 = v000 * (1.0 - fx) + v100 * fx
    c10 = v010 * (1.0 - fx) + v110 * fx
    c01 = v001 * (1.0 - fx) + v101 * fx
    c11 = v011 * (1.0 - fx) + v111 * fx

    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy

    return c0 * (1.0 - fz) + c1 * fz


@wp.kernel
def _wos_kernel(
    sdf_flat: wp.array3d(dtype=float),        # (B*G, G, G) — per-shape depth slices
    query_points: wp.array(dtype=wp.vec3),    # (B,)
    hits: wp.array(dtype=wp.vec3),            # (B * n_walks,) — output hit coords
    mask: wp.array(dtype=wp.int32),           # (B * n_walks,) — 1=terminated, 0=max_steps
    n_walks: int,
    grid_size: int,
    epsilon: float,
    max_steps: int,
    seed: int,
):
    """Each thread runs one WoS walk and writes the termination coordinate."""
    tid = wp.tid()
    shape_idx = tid // n_walks
    depth_offset = shape_idx * grid_size

    qp = query_points[shape_idx]
    px = qp[0]
    py = qp[1]
    pz = qp[2]

    state = wp.rand_init(seed, tid)
    terminated = int(0)

    for step in range(max_steps):
        # Distance to box [0, 1]³
        dist_box = wp.min(
            wp.min(wp.min(px, py), pz),
            wp.min(wp.min(1.0 - px, 1.0 - py), 1.0 - pz),
        )
        sdf_val = _trilinear_sdf(sdf_flat, px, py, pz, depth_offset, grid_size)
        radius = wp.min(dist_box, wp.abs(sdf_val))

        if radius < epsilon:
            terminated = 1
            break

        u = wp.randf(state)
        v = wp.randf(state)
        cos_theta = 2.0 * u - 1.0
        sin_theta_sq = 1.0 - cos_theta * cos_theta
        if sin_theta_sq < 0.0:
            sin_theta_sq = 0.0
        sin_theta = wp.sqrt(sin_theta_sq)
        phi = 6.28318530718 * v

        dx = sin_theta * wp.cos(phi)
        dy = sin_theta * wp.sin(phi)
        dz = cos_theta

        px = px + radius * dx
        py = py + radius * dy
        pz = pz + radius * dz

        px = wp.clamp(px, 0.0, 1.0)
        py = wp.clamp(py, 0.0, 1.0)
        pz = wp.clamp(pz, 0.0, 1.0)

    hits[tid] = wp.vec3(px, py, pz)
    mask[tid] = terminated


# ======================================================================
# Python-side sampler class
# ======================================================================

class WoSHitSampler:
    """GPU WoS hit sampler via Warp. Requires CUDA + warp-lang.

    Trainer hardcodes this class (no mock fallback at runtime); construct
    `MockWoSHitSampler` explicitly for CPU unit tests.
    """

    def __init__(self, epsilon: float = 1e-4, max_steps: int = 256) -> None:
        self.epsilon = epsilon
        self.max_steps = max_steps

    def sample_hits(
        self,
        p: Tensor,                 # (B, 3) query points in [0, 1]³
        sdf_grid: Tensor,          # (B, G, G, G) SDF grid
        n_walks: int,
        seed: int | None = None,
    ) -> Tuple[Tensor, Tensor]:
        """Run WoS from each query point; return raw hit coords + mask.

        Returns:
            hits: (B, n_walks, 3) hit coordinates in [0, 1]³
            mask: (B, n_walks) bool — True where walk terminated at ε-shell
        """
        assert p.dim() == 2 and p.shape[1] == 3, f"p shape {p.shape} != (B, 3)"
        assert sdf_grid.dim() == 4, f"sdf_grid must be (B, G, G, G); got {sdf_grid.shape}"
        B, G, _, _ = sdf_grid.shape
        assert sdf_grid.shape == (B, G, G, G), "SDF grid must be cubic"
        assert p.shape[0] == B
        if p.device.type != "cuda":
            raise RuntimeError(
                "WoSHitSampler requires CUDA tensors. For CPU tests, use "
                "tests/fixtures/mock_wos.py::MockWoSHitSampler."
            )

        device_str = str(p.device)

        # Flatten along depth for the per-shape depth-offset trick.
        sdf_flat_torch = sdf_grid.contiguous().reshape(B * G, G, G).to(torch.float32)
        sdf_flat_wp = wp.from_torch(sdf_flat_torch, dtype=wp.float32)
        p_wp = wp.from_torch(p.contiguous().to(torch.float32).reshape(B, 3), dtype=wp.vec3)

        total = B * n_walks
        hits_wp = wp.zeros(total, dtype=wp.vec3, device=device_str)
        mask_wp = wp.zeros(total, dtype=wp.int32, device=device_str)

        if seed is None:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

        wp.launch(
            kernel=_wos_kernel,
            dim=total,
            inputs=[
                sdf_flat_wp, p_wp, hits_wp, mask_wp,
                n_walks, G, float(self.epsilon), int(self.max_steps), int(seed),
            ],
            device=device_str,
        )
        wp.synchronize_device(device_str)

        hits_torch = wp.to_torch(hits_wp).reshape(B, n_walks, 3)
        mask_torch = wp.to_torch(mask_wp).reshape(B, n_walks).to(torch.bool)
        return hits_torch, mask_torch
