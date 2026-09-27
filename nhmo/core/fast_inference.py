"""Fast inference path for the 3D pipeline: per-shape precompute + cached solves.

At inference time NHMO factorizes the work into a per-shape precompute and a
per-problem solve:

    precompute(Omega):  shape latent, SDF grid, K_eff[q, j] = w_j K(p_q, zeta_j)
    solve(h, f):        u(p_q) = sum_j K_eff[q, j] h(zeta_j) + v_phi(p_q; Omega, f)

The reference implementations (`build_keff_reference`, `lift_forward_reference`)
reproduce the arithmetic of the evaluation code used for the paper tables. The
fast implementations below are mathematically identical and only change how
the work is batched:

1. `build_keff_folded`: `HarmonicMeasureField.log_kernel` re-embeds the same
   boundary samples (Fourier features + two projections) for every query of a
   chunk. The folded build embeds them once per shape and broadcasts the
   embedded tokens. In fp32 this reproduces the reference K_eff bit for bit.
2. `lift_forward_folded`: the lift's cross-attention treats each query as a
   batch entry of sequence length 1 attending to an expanded copy of the same
   context, so the context K/V projections are recomputed once per query.
   `_CrossAttnBlock` guarantees that each output token depends only on
   (x_j, context), so folding all queries into the sequence dimension with one
   context row is exact in exact arithmetic. In fp32 the two paths run
   different GEMM/attention kernel shapes and agree to rounding level
   (relative difference ~1e-7), not bitwise.
3. GPU SDF rasterization (`nhmo.geometry.sdf_gpu`) replaces the CPU trimesh
   signed-distance grid.
4. `precompute_shape` combines the GPU SDF grid, large-chunk interior-anchor
   sampling, the shape encoder and the folded K_eff build.

Optional bf16 autocast (`amp_dtype=torch.bfloat16`) gives a further speedup at
a ~1% change of K_eff; the rebuttal build times used bf16 for K_eff and fp32
for the lift.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from nhmo.core.kernel import HarmonicMeasureField, ShapeLatent
from nhmo.core.source_lift import PoissonLiftModule


def stable_seed(name: str, seed: int) -> int:
    """Process-independent 31-bit seed from a string key and an integer seed.

    (Python's built-in `hash(str)` is randomized per process.)
    """
    return int((zlib.crc32(name.encode("utf-8")) + 1_000_003 * int(seed)) % (2 ** 31))


# --------------------------------------------------------------- K_eff build

def build_keff_reference(
    kernel: HarmonicMeasureField,
    p_model: Tensor,               # (Q, 3)
    surface: Tensor,               # (1, N_s, 3)
    surface_normals: Tensor,       # (1, N_s, 3)
    shape_latent: ShapeLatent,     # tokens (1, M, d)
    total_area: float,
    q_batch: int = 1024,
) -> Tensor:                       # (Q, N_s)
    """K_eff[q, j] = w_j * exp(log K(p_q, zeta_j) - log Z(p_q)), original path."""
    device = p_model.device
    N_s = surface.shape[1]
    Q = p_model.shape[0]
    K_rows = torch.empty(Q, N_s, device=device)
    w = torch.full((N_s,), total_area / N_s, device=device)
    log_w = torch.log(w + 1e-30)
    for s in range(0, Q, q_batch):
        e = min(Q, s + q_batch)
        qb = e - s
        log_K = kernel.log_kernel(p_model[s:e], surface.expand(qb, -1, -1),
                                  surface_normals.expand(qb, -1, -1),
                                  ShapeLatent(tokens=shape_latent.tokens.expand(qb, -1, -1)))
        log_Z = torch.logsumexp(log_K + log_w.unsqueeze(0), dim=-1, keepdim=True)
        K_rows[s:e] = torch.exp(log_K - log_Z) * w.unsqueeze(0)
    return K_rows


def build_keff_folded(
    kernel: HarmonicMeasureField,
    p_model: Tensor,
    surface: Tensor,
    surface_normals: Tensor,
    shape_latent: ShapeLatent,
    total_area: float,
    q_batch: int = 1024,
    amp_dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Same as `build_keff_reference`, with the zeta tokens embedded once per shape."""
    device = p_model.device
    N_s = surface.shape[1]
    Q = p_model.shape[0]
    K_rows = torch.empty(Q, N_s, device=device)
    w = torch.full((N_s,), total_area / N_s, device=device)
    log_w = torch.log(w + 1e-30)
    with torch.autocast(device.type, dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None):
        z_flat = surface.reshape(N_s, 3)
        n_flat = surface_normals.reshape(N_s, 3)
        zeta_tokens = (kernel.proj_zeta(kernel.fourier_zeta(z_flat))
                       + kernel.proj_normal(kernel.fourier_normal(n_flat))).unsqueeze(0)   # (1, N_s, d)
        for s in range(0, Q, q_batch):
            e = min(Q, s + q_batch)
            qb = e - s
            p_token = kernel.proj_p(kernel.fourier_p(p_model[s:e])).unsqueeze(1)          # (qb, 1, d)
            context = torch.cat([shape_latent.tokens.expand(qb, -1, -1), p_token], dim=1)
            x = zeta_tokens.expand(qb, -1, -1)
            for layer in kernel.cross_layers:
                x = layer(x, context)
            log_K = kernel.head_mlp(x).squeeze(-1).float()
            if kernel.log_K_max is not None:
                cap = kernel.log_K_max
                log_K = cap * torch.tanh(log_K / cap)
            log_Z = torch.logsumexp(log_K + log_w.unsqueeze(0), dim=-1, keepdim=True)
            K_rows[s:e] = torch.exp(log_K - log_Z) * w.unsqueeze(0)
    return K_rows


# ------------------------------------------------------------- lift forward

def lift_forward_reference(
    lift: PoissonLiftModule,
    p_model: Tensor,               # (Q, 3)
    shape_tokens: Tensor,          # (1, M_shape, d)
    source_tokens: Tensor,         # (1, M_src, d)
    sdf_at_p: Tensor,              # (Q,)
    q_batch: int = 4096,
) -> Tensor:                       # (Q,)
    """Original per-query lift forward (each query a batch entry of length 1)."""
    Q = p_model.shape[0]
    chunks = []
    for s in range(0, Q, q_batch):
        e = min(Q, s + q_batch)
        qb = e - s
        chunks.append(lift(p_model[s:e], shape_tokens.expand(qb, -1, -1),
                           source_tokens.expand(qb, -1, -1), sdf_at_p[s:e]))
    return torch.cat(chunks)


def lift_forward_folded(
    lift: PoissonLiftModule,
    p_model: Tensor,
    shape_tokens: Tensor,
    source_tokens: Tensor,
    sdf_at_p: Tensor,
    chunk: int = 65536,
    amp_dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Query-folded lift forward: all queries share one context row."""
    head = lift.lift_head
    if head.use_uh_cond:
        raise NotImplementedError("query folding is implemented for use_uh_cond=False")
    device = p_model.device
    context = torch.cat([shape_tokens, source_tokens], dim=1)                 # (1, M, d)
    Q = p_model.shape[0]
    outs = []
    with torch.autocast(device.type, dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None):
        for s in range(0, Q, chunk):
            e = min(Q, s + chunk)
            x = head.proj_p(head.fourier_p(p_model[s:e])).unsqueeze(0)       # (1, q, d)
            for layer in head.cross_layers:
                x = layer(x, context)
            tilde = head.head_mlp(x.squeeze(0)).squeeze(-1).float()
            if head.gauge_kind == "tanh":
                depth = torch.tanh((-sdf_at_p[s:e]).clamp(min=0.0) / head.gauge_eps)
            else:
                depth = (-sdf_at_p[s:e]).clamp(min=0.0)
            outs.append(depth.float() * tilde)
    return torch.cat(outs)


# --------------------------------------------------------- shape precompute

@dataclass
class ShapeCache:
    """Everything a cached per-problem solve needs for one tet-meshed shape."""
    shape_id: str
    surface: Tensor               # (1, N_s, 3) model frame
    surface_normals: Tensor       # (1, N_s, 3)
    sdf_grid_wos: Tensor          # (G, G, G)
    total_area: float
    shape_latent: ShapeLatent
    v_tet: np.ndarray             # (N_v, 3) original frame
    interior_inds: np.ndarray     # (Q,)
    center: np.ndarray
    scale: float
    p_model: Optional[Tensor] = None      # (Q, 3) interior tet vertices, model frame
    sdf_at_p: Optional[Tensor] = None     # (Q,)
    keff: Optional[Tensor] = None         # (Q, N_s)
    bd_inds: Optional[np.ndarray] = None
    nearest: Optional[Tensor] = None      # (N_s,) index into bd vertices


def encode_shape(
    kernel: HarmonicMeasureField,
    sol_path: str | Path,
    grid_size: int,
    n_surface: int,
    n_interior: int,
    device: torch.device,
    seed: Optional[int] = None,
    sdf_backend: str = "trimesh",
    interior_chunk: Optional[int] = None,
) -> ShapeCache:
    """Shape encode shared by the reference and the fast evaluation paths.

    With `sdf_backend="trimesh"` and `interior_chunk=None` this is exactly the
    encode used for the paper's Table 2 (plus the optional seed).
    """
    from nhmo.data.mcb_loader import MCBSolShapeGenerator, load_mcb_eval_fields

    sol_path = Path(sol_path)
    shape_id = sol_path.parent.parent.name
    gen = MCBSolShapeGenerator(
        sol_npz_path=sol_path, grid_size=grid_size,
        n_surface=n_surface, n_interior_anchors=n_interior, device=str(device),
        seed=None if seed is None else stable_seed(shape_id, seed),
        sdf_backend=sdf_backend, interior_chunk=interior_chunk,
    )
    spec = gen()
    surface = spec["shape_ctx"]["surface_points"].to(device)
    surface_normals = spec["shape_ctx"]["surface_normals"].to(device)
    interior_pool = spec["shape_ctx"]["interior_points"].to(device)
    sdf_grid_wos = spec["sdf_grid_wos"].to(device)
    total_area = float(spec.get("total_area_model_frame", 4.0))

    f = load_mcb_eval_fields(str(sol_path))
    v_tet = np.asarray(f.v_tet)
    bd_inds = np.asarray(f.bd_v_inds)
    bbox_min = v_tet.min(axis=0).astype(np.float64)
    bbox_max = v_tet.max(axis=0).astype(np.float64)
    center = 0.5 * (bbox_min + bbox_max)
    half = 0.5 * (bbox_max - bbox_min)
    scale = 0.9 / max(float(np.max(half)), 1e-12)
    bd_mask = np.zeros(len(v_tet), dtype=bool)
    bd_mask[bd_inds] = True
    interior_inds = np.where(~bd_mask)[0]

    with torch.no_grad():
        sl = kernel.encode({
            "surface_points": surface,
            "surface_normals": surface_normals,
            "interior_points": interior_pool,
        })
    return ShapeCache(shape_id=shape_id, surface=surface, surface_normals=surface_normals,
                      sdf_grid_wos=sdf_grid_wos, total_area=total_area, shape_latent=sl,
                      v_tet=v_tet, interior_inds=interior_inds, center=center, scale=scale,
                      bd_inds=bd_inds)


def nearest_boundary_vertex(cache: ShapeCache, bd_inds: np.ndarray) -> Tensor:
    """For each surface sample, the index of the nearest boundary tet vertex."""
    bd_pts_model = (cache.v_tet[bd_inds] - cache.center) * cache.scale
    surf_np = cache.surface.squeeze(0).cpu().numpy()
    diffs = surf_np[:, None, :] - bd_pts_model[None, :, :]
    nearest = np.argmin((diffs * diffs).sum(axis=-1), axis=-1)
    return torch.from_numpy(nearest).to(cache.surface.device)


def precompute_shape(
    kernel: HarmonicMeasureField,
    sol_path: str | Path,
    grid_size: int = 32,
    n_surface: int = 2000,
    n_interior: int = 512,
    device: torch.device | str = "cuda",
    seed: Optional[int] = 0,
    fast: bool = True,
    keff_amp_dtype: Optional[torch.dtype] = None,
    q_batch_keff: int = 1024,
) -> ShapeCache:
    """Per-shape precompute: encode + SDF at queries + K_eff over all interior vertices.

    fast=True: GPU SDF grid, 4096-candidate interior-anchor chunks, folded K_eff.
    fast=False: the original trimesh path and reference K_eff.
    """
    from nhmo.geometry.sdf import trilinear_interp

    device = torch.device(device)
    cache = encode_shape(kernel, sol_path, grid_size, n_surface, n_interior, device,
                         seed=seed, sdf_backend="gpu" if fast else "trimesh",
                         interior_chunk=4096 if fast else None)
    p_model = torch.from_numpy(((cache.v_tet[cache.interior_inds] - cache.center)
                                * cache.scale).astype(np.float32)).to(device)
    p_wos = (p_model + 1.0) * 0.5
    cache.p_model = p_model
    cache.sdf_at_p = trilinear_interp(cache.sdf_grid_wos.unsqueeze(0), p_wos.unsqueeze(0)).squeeze(0)
    with torch.no_grad():
        if fast:
            cache.keff = build_keff_folded(kernel, p_model, cache.surface, cache.surface_normals,
                                           cache.shape_latent, cache.total_area,
                                           q_batch=q_batch_keff, amp_dtype=keff_amp_dtype)
        else:
            cache.keff = build_keff_reference(kernel, p_model, cache.surface, cache.surface_normals,
                                              cache.shape_latent, cache.total_area, q_batch=q_batch_keff)
    cache.nearest = nearest_boundary_vertex(cache, cache.bd_inds)
    return cache


def source_probe(cache: ShapeCache, source_term: np.ndarray, n_source_probe: int,
                 seed: Optional[int]) -> tuple[Tensor, Tensor]:
    """Subsample (position, value) pairs of f over all tet vertices for the source encoder."""
    device = cache.surface.device
    v_tet = cache.v_tet
    n_probe = min(n_source_probe, len(v_tet))
    # seed=None reproduces the unseeded behavior of the original runs.
    rng = np.random.RandomState() if seed is None else np.random.RandomState(stable_seed(cache.shape_id, seed))
    probe_idx = rng.choice(len(v_tet), n_probe, replace=False)
    v_tet_model = (v_tet - cache.center) * cache.scale
    f_probe_pos = torch.from_numpy(v_tet_model[probe_idx].astype(np.float32)).to(device)
    f_probe = torch.from_numpy(np.asarray(source_term).reshape(-1)[probe_idx].astype(np.float32)).to(device)
    return f_probe_pos, f_probe


def solve_cached(
    lift: PoissonLiftModule,
    cache: ShapeCache,
    bd_v_inds: np.ndarray,
    bd_v_vals: np.ndarray,
    source_term: np.ndarray,
    n_source_probe: int = 1024,
    seed: Optional[int] = 0,
    fast: bool = True,
    lift_amp_dtype: Optional[torch.dtype] = None,
) -> tuple[Tensor, Tensor]:
    """Per-problem solve from a precomputed cache. Returns (u_h, v_phi) at interior vertices."""
    device = cache.surface.device
    if cache.bd_inds is None or len(bd_v_inds) != len(cache.bd_inds) or \
            not np.array_equal(np.asarray(bd_v_inds), cache.bd_inds):
        cache.bd_inds = np.asarray(bd_v_inds)
        cache.nearest = nearest_boundary_vertex(cache, cache.bd_inds)
    bd_vals = torch.from_numpy(np.asarray(bd_v_vals, dtype=np.float32)).to(device)
    f_pos, f_val = source_probe(cache, source_term, n_source_probe, seed)
    with torch.no_grad():
        u_h = cache.keff @ bd_vals[cache.nearest]
        src_tokens = lift.encode_source(f_pos.unsqueeze(0), f_val.unsqueeze(0))
        if fast:
            v_phi = lift_forward_folded(lift, cache.p_model, cache.shape_latent.tokens,
                                        src_tokens, cache.sdf_at_p, amp_dtype=lift_amp_dtype)
        else:
            v_phi = lift_forward_reference(lift, cache.p_model, cache.shape_latent.tokens,
                                           src_tokens, cache.sdf_at_p)
    return u_h, v_phi
