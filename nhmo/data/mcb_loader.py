"""MCB dataset loader — NGF-compatible split.

Goal: reproduce the MCB-B split used by NGF (Yoo et al., Neural Green's
Functions, NeurIPS 2025) for apples-to-apples S-1 evaluation.

Phase 6.9 B2 shipped MCBShapeGenerator (works on any trimesh.Trimesh)
and a stubbed MCBDataset. Phase 7.1 fills in the sol.npz path:

  load_mcb_geometry_only(path)
      Training-safe: reads ONLY v_tet, f_tet from sol.npz. Never touches
      sol / source_term / bd_* fields. Enforces R_7.2 at the API level.

  MCBSolShapeGenerator(sol_path, ...)
      Composes load_mcb_geometry_only + extract_tet_boundary +
      MCBShapeGenerator's surface/interior sampling.

  MCBSolDataset(root, split_files, ...)
      Enumerates (category, shape_id, bc_config) from NGF split files
      under `$MCB_ROOT/splits/{category}/{split_name}.txt`. Each
      __getitem__(idx) returns a shape_ctx dict without reading sol.

R_7.2 / R_7.3 enforcement:
  - This module MUST NOT read sol / source_term / bd_* fields from
    sol.npz in its training-path functions. `load_mcb_geometry_only`
    opens the npz and extracts ONLY v_tet + f_tet.
  - Eval-only field access goes through a SEPARATE function
    `load_mcb_eval_fields` (ready for Phase 7.2 head_to_head_ngf.py;
    never called from nhmo/train/ or from the dataset's __getitem__).
  - Tests: tests/test_training_never_reads_sol_field.py (AST scan) and
    tests/test_mcb_training_uses_own_bc.py (no mcb_bc_parser import in
    nhmo/train/*).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from nhmo.data.mcb_boundary import extract_tet_boundary


class MCBShapeGenerator:
    """Shape generator producing `shape_ctx` dicts from an arbitrary
    trimesh.Trimesh. Matches `UnitBallShapeGenerator`'s interface so it
    can feed `WoSDataset` directly.

    `seed=None` (training default) draws fresh surface samples and interior
    anchors on every call. An integer seed makes every call return the same
    samples, which is what the released evaluators use for reproducibility.
    `sdf_backend="gpu"` replaces the CPU trimesh SDF rasterization by
    `nhmo.geometry.sdf_gpu.rasterize_sdf_grid_gpu` (inference only).
    """

    def __init__(
        self,
        mesh,                                  # trimesh.Trimesh
        grid_size: int = 64,
        n_surface: int = 512,
        n_interior_anchors: int = 512,
        device: str = "cpu",
        seed: Optional[int] = None,
        sdf_backend: str = "trimesh",
        interior_chunk: Optional[int] = None,
    ) -> None:
        import trimesh

        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"expected trimesh.Trimesh, got {type(mesh).__name__}")
        if sdf_backend not in ("trimesh", "gpu"):
            raise ValueError(f"sdf_backend must be 'trimesh' or 'gpu', got {sdf_backend!r}")
        self.mesh = mesh
        self.grid_size = grid_size
        self.n_surface = n_surface
        self.n_interior_anchors = n_interior_anchors
        self.device = device
        self.seed = seed
        self.sdf_backend = sdf_backend
        self.interior_chunk = interior_chunk

        # Precompute bbox diag in model frame; mesh is assumed to be normalized
        # to roughly fit inside [-1, 1]³ — caller's responsibility.
        self.bbox_min = np.asarray(mesh.bounds[0], dtype=np.float32)
        self.bbox_max = np.asarray(mesh.bounds[1], dtype=np.float32)
        self.bbox_diag_model_frame = float(np.linalg.norm(self.bbox_max - self.bbox_min))

        # Rasterize SDF once (mesh is fixed for this generator).
        self._sdf_grid_wos = self._rasterize_sdf_grid()

    def _rasterize_sdf_grid(self) -> Tensor:
        """Rasterize a signed-distance grid in WoS frame [0, 1]³.

        Uses trimesh.proximity.signed_distance (slow for large G; O(G³·F)),
        or the GPU rasterizer when `sdf_backend == "gpu"`.
        """
        if self.sdf_backend == "gpu":
            from nhmo.geometry.sdf_gpu import rasterize_sdf_grid_gpu
            dev = self.device if str(self.device).startswith("cuda") else "cuda"
            return rasterize_sdf_grid_gpu(self.mesh, self.grid_size, dev).to(self.device)

        import trimesh.proximity

        G = self.grid_size
        coords = np.linspace(0.0, 1.0, G, dtype=np.float32)
        # (G, G, G, 3) in WoS frame
        iz, iy, ix = np.meshgrid(coords, coords, coords, indexing="ij")
        pts_wos = np.stack([ix, iy, iz], axis=-1)
        # Map WoS → model frame (mesh lives in model frame).
        pts_model = 2.0 * pts_wos - 1.0
        flat = pts_model.reshape(-1, 3).astype(np.float64)
        sd = trimesh.proximity.signed_distance(self.mesh, flat)       # inside > 0
        # Align sign convention with nhmo: negative inside.
        sd = -sd.astype(np.float32)
        grid = sd.reshape(G, G, G)
        return torch.from_numpy(grid).to(self.device)

    def __call__(self) -> dict:
        """Produce a batch's worth of shape_ctx + metadata."""
        from nhmo.geometry.surface import sample_surface_mesh

        points, normals, _face_idx, _area = sample_surface_mesh(
            self.mesh, self.n_surface, seed=self.seed,
        )
        surf = torch.from_numpy(points).to(self.device)
        surf_n = torch.from_numpy(normals).to(self.device)

        # Interior samples: rejection-sample from the mesh bbox.
        interior = self._sample_interior()
        interior = torch.from_numpy(interior).to(self.device)

        return {
            "shape_ctx": {
                "surface_points": surf.unsqueeze(0),                  # (1, N_s, 3)
                "surface_normals": surf_n.unsqueeze(0),
                "interior_points": interior.unsqueeze(0),
            },
            "sdf_grid_wos": self._sdf_grid_wos,
            "radius_model_frame": self.bbox_diag_model_frame / 2.0,
            "bbox_diag_model_frame": self.bbox_diag_model_frame,
        }

    def _sample_interior(self) -> np.ndarray:
        """Rejection-sample n_interior points inside the mesh.

        Uses `trimesh.proximity.signed_distance > 0` for interior test.
        For thin shells this may reject most candidates; Phase 7 should
        cache interior samples if MCB shapes are consistent-complex.
        """
        import trimesh.proximity

        n = self.n_interior_anchors
        accumulated = []
        attempts = 0
        max_attempts = n * 50
        # Seeded evaluators get a private RNG so results do not depend on
        # the global numpy state; training (seed=None) keeps np.random.
        rng = np.random if self.seed is None else np.random.RandomState(self.seed)

        while sum(len(a) for a in accumulated) < n and attempts < max_attempts:
            if self.interior_chunk is None:
                chunk_size = max(n, 2 * n - sum(len(a) for a in accumulated))
            else:
                chunk_size = int(self.interior_chunk)
            candidates = rng.uniform(
                self.bbox_min, self.bbox_max, size=(chunk_size, 3)
            ).astype(np.float32)
            sd = trimesh.proximity.signed_distance(self.mesh, candidates.astype(np.float64))
            inside = candidates[sd > 0]                              # trimesh: inside > 0
            if len(inside) > 0:
                accumulated.append(inside)
            attempts += chunk_size

        if sum(len(a) for a in accumulated) < n:
            raise RuntimeError(
                f"MCBShapeGenerator: interior rejection sampler exhausted "
                f"{attempts} candidates but only found "
                f"{sum(len(a) for a in accumulated)}/{n} interior points."
            )

        interior = np.concatenate(accumulated, axis=0)[:n]
        return interior


# ====================================================== sol.npz loaders

# R_7.2 discipline: this is the ONLY training-path reader of sol.npz.
# Exactly two fields are extracted: `v_tet` and `f_tet`. Any other field
# access from training code must go through a separate function
# (load_mcb_eval_fields) that EVAL-path code may import but training
# code must not. The split is enforced at the file level, not just by
# convention — adding a third field read to `load_mcb_geometry_only` is
# itself a C3 violation.

def load_mcb_geometry_only(
    sol_npz_path: str | Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Read ONLY v_tet and f_tet from a NGF sol.npz. Never touches sol,
    source_term, bd_v_inds, bd_v_vals, bd_A, bd_B, bd_C, bd_D.

    Returns:
        v_tet: (N_v, 3) float64 vertex positions
        f_tet: (N_t, 4) int32 tet indices
    """
    with np.load(sol_npz_path, allow_pickle=True) as z:
        # Strict two-field read. Adding a third line that reads any other
        # field constitutes a C3 violation (see PHILOSOPHY.md §C3 and R_7.2).
        v_tet = np.asarray(z["v_tet"])
        f_tet = np.asarray(z["f_tet"])
    return v_tet, f_tet


@dataclass(frozen=True)
class _MCBEvalFields:
    """Eval-only bundle of sol.npz fields. Frozen."""
    v_tet: np.ndarray
    f_tet: np.ndarray
    sol: np.ndarray
    source_term: np.ndarray
    bd_v_inds: np.ndarray
    bd_v_vals: np.ndarray
    bd_A: float
    bd_B: float
    bd_C: float
    bd_D: float


def load_mcb_eval_fields(sol_npz_path: str | Path) -> _MCBEvalFields:
    """Read the full set of sol.npz fields for head-to-head evaluation.

    R_7.2: this function must NEVER be called from `nhmo/train/*` or from
    `nhmo/data/wos_dataset*`. An AST scan in
    `tests/test_training_never_reads_sol_field.py` enforces that
    `load_mcb_eval_fields` and the string `"sol"` are absent from those
    modules.
    """
    with np.load(sol_npz_path, allow_pickle=True) as z:
        return _MCBEvalFields(
            v_tet=np.asarray(z["v_tet"]),
            f_tet=np.asarray(z["f_tet"]),
            sol=np.asarray(z["sol"]),
            source_term=np.asarray(z["source_term"]),
            bd_v_inds=np.asarray(z["bd_v_inds"]),
            bd_v_vals=np.asarray(z["bd_v_vals"]),
            bd_A=float(z["bd_A"]),
            bd_B=float(z["bd_B"]),
            bd_C=float(z["bd_C"]),
            bd_D=float(z["bd_D"]),
        )


# ====================================================== shape generator

class MCBSolShapeGenerator:
    """Shape generator that reads ONE sol.npz and produces a shape_ctx dict.

    Geometry-only path: calls `load_mcb_geometry_only` → extracts
    triangle boundary via `extract_tet_boundary` → wraps as a
    trimesh.Trimesh → uses `MCBShapeGenerator` internals for surface
    sampling + SDF rasterization.

    Unlike `RandomDiskShapeGenerator`, this one is STATIC — it serves
    the same sol.npz on every call, because the geometry is fixed. The
    Dataset layer above varies across shapes / BC configs.
    """

    def __init__(
        self,
        sol_npz_path: str | Path,
        grid_size: int = 64,
        n_surface: int = 2048,
        n_interior_anchors: int = 512,
        device: str = "cpu",
        seed: Optional[int] = None,
        sdf_backend: str = "trimesh",
        interior_chunk: Optional[int] = None,
    ) -> None:
        import trimesh
        v_tet, f_tet = load_mcb_geometry_only(sol_npz_path)
        v_boundary, f_boundary, _ = extract_tet_boundary(v_tet, f_tet)
        # Normalize into [-1, 1]³ model frame: center on bbox midpoint,
        # scale so the largest half-extent fits in [-0.9, 0.9] (leave
        # margin for FourierFeatures' [-1, 1] assertion).
        bbox_min = v_boundary.min(axis=0)
        bbox_max = v_boundary.max(axis=0)
        center = 0.5 * (bbox_min + bbox_max)
        half_extent = 0.5 * (bbox_max - bbox_min)
        max_half = float(np.maximum.reduce(half_extent))
        scale = 0.9 / max(max_half, 1e-12)
        v_normalized = (v_boundary - center) * scale
        self._mesh = trimesh.Trimesh(
            vertices=v_normalized.astype(np.float64),
            faces=f_boundary.astype(np.int64),
            process=False,
        )
        # Delegate to MCBShapeGenerator for sampling + SDF raster
        self._inner = MCBShapeGenerator(
            mesh=self._mesh,
            grid_size=grid_size,
            n_surface=n_surface,
            n_interior_anchors=n_interior_anchors,
            device=device,
            seed=seed,
            sdf_backend=sdf_backend,
            interior_chunk=interior_chunk,
        )
        self._sol_path = str(sol_npz_path)

    def __call__(self) -> dict:
        record = self._inner()
        # Expose total mesh area so MCBWoSDataset can compute quadrature
        # weights without re-sampling.
        record["total_area_model_frame"] = float(self._mesh.area)
        return record


# ====================================================== dataset

class MCBSolDataset(Dataset):
    """Enumerate NGF-format split files and serve shape_ctx dicts for training.

    Root layout expected (matches Phase 6.9 D3 acquisition):
        $MCB_ROOT/splits/{category}/{split_name}.txt
        $MCB_ROOT/ngf_solutions/{category}/{shape_id}/{bc_config}/sol.npz

    Each split file contains lines of the form:
        ../data/mcb_b/{category}/{shape_id}/{bc_config}/sol.npz

    The dataset iterates over UNIQUE shape_ids (not over shape × BC),
    since the geometry is invariant across BC configs — training sees
    each shape at most once per epoch. If multiple BC configs are
    present the FIRST encountered is used for geometry (they are
    identical by construction).

    R_7.2: __getitem__ uses `MCBSolShapeGenerator` which calls
    `load_mcb_geometry_only`. No sol-field access anywhere in the
    training data path.
    """

    def __init__(
        self,
        mcb_root: str | Path,
        categories: List[str],
        split_name: str = "known_shape_known_prob",
        grid_size: int = 64,
        n_surface: int = 2048,
        n_interior_anchors: int = 512,
        device: str = "cpu",
    ) -> None:
        self.mcb_root = Path(mcb_root)
        self.categories = list(categories)
        self.split_name = split_name
        self.grid_size = grid_size
        self.n_surface = n_surface
        self.n_interior_anchors = n_interior_anchors
        self.device = device
        self._entries = self._enumerate_entries()

    def _enumerate_entries(self) -> List[Tuple[str, str, Path]]:
        """Return (category, shape_id, sol_npz_path) tuples — one per unique shape."""
        out: List[Tuple[str, str, Path]] = []
        seen: set[Tuple[str, str]] = set()
        for cat in self.categories:
            split_file = self.mcb_root / "splits" / cat / f"{self.split_name}.txt"
            if not split_file.exists():
                raise FileNotFoundError(
                    f"Split file missing: {split_file}. Expected NGF-format "
                    f"splits at $MCB_ROOT/splits/{cat}/{self.split_name}.txt."
                )
            lines = split_file.read_text().strip().splitlines()
            for line in lines:
                parts = Path(line).parts
                # Expected: ../data/mcb_b/{cat}/{shape_id}/{bc_config}/sol.npz
                if len(parts) < 5 or parts[-1] != "sol.npz":
                    continue
                shape_id = parts[-3]
                bc_config = parts[-2]
                key = (cat, shape_id)
                if key in seen:
                    continue
                seen.add(key)
                sol_path = self.mcb_root / "ngf_solutions" / cat / shape_id / bc_config / "sol.npz"
                if not sol_path.exists():
                    # Missing file (e.g. one of Phase 6.9's 3 missing sol.npz in
                    # screws_and_bolts/00057159). Skip; log count.
                    continue
                out.append((cat, shape_id, sol_path))
        return out

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> dict:
        cat, shape_id, sol_path = self._entries[idx]
        gen = MCBSolShapeGenerator(
            sol_npz_path=sol_path,
            grid_size=self.grid_size,
            n_surface=self.n_surface,
            n_interior_anchors=self.n_interior_anchors,
            device=self.device,
        )
        record = gen()
        record["category"] = cat
        record["shape_id"] = shape_id
        return record


# ====================================================== WoS batching

class MCBWoSDataset(Dataset):
    """On-the-fly WoS batcher for MCB shapes. Parallel of WoSDataset (3D
    ball) adapted to arbitrary triangle meshes.

    Differences from WoSDataset (3D ball):
      - Interior queries sample from `shape_ctx["interior_points"]` (the
        generator-provided interior point cloud), not from a ball.
      - Surface quadrature uses `shape_ctx["surface_points"]` and the
        per-sample area weight = mesh.area / N (uniform-area sampling
        invariant of `sample_surface_mesh`).
      - SDF at query points: approximated from the mesh's bounding-box
        half-extent minus |p - center|. For detailed SDF use
        `trimesh.proximity.signed_distance` (slow; not used here).

    R_7.2: reads only shape_ctx + sdf_grid_wos from the spec dict; never
    touches sol / source_term / bd_* fields (the generator doesn't load
    them).
    """

    def __init__(
        self,
        shape_generator: Callable[[], dict],
        wos_sampler,
        n_queries: int = 16,
        n_wos_hits: int = 4,
        n_surface: int = 256,
        virtual_length: int = 1000,
    ) -> None:
        self.shape_generator = shape_generator
        self.wos_sampler = wos_sampler
        self.n_queries = n_queries
        self.n_wos_hits = n_wos_hits
        self.n_surface = n_surface
        self.virtual_length = virtual_length

    def __len__(self) -> int:
        return self.virtual_length

    @torch.no_grad()
    def __getitem__(self, idx: int) -> dict:
        from nhmo.data.synthetic import model_frame_to_wos_frame, wos_frame_to_model_frame

        spec = self.shape_generator()
        shape_ctx = spec["shape_ctx"]
        sdf_grid_wos = spec["sdf_grid_wos"]
        radius = float(spec["radius_model_frame"])     # bbox half-diag (approx)
        bbox_diag = float(spec["bbox_diag_model_frame"])
        device = shape_ctx["surface_points"].device

        interior_full = shape_ctx["interior_points"].squeeze(0)   # (N_int, 3)
        N_int = interior_full.shape[0]
        if N_int < self.n_queries:
            raise ValueError(
                f"shape has only {N_int} interior points; need at least "
                f"n_queries={self.n_queries}."
            )
        # Pick B_q interior points at random (with replacement OK for small batches)
        perm = torch.randperm(N_int, device=device)[: self.n_queries]
        p_mv = interior_full[perm]                                # (B_q, 3)
        p_nll = p_mv.clone()

        # SDF at p_mv: lookup the rasterized SDF at each query point.
        # The grid stores model-frame signed distance at WoS-frame grid
        # positions (negative inside per nhmo convention). Interior p → sdf < 0.
        # Dist-to-boundary = −sdf_model. Cap max so L_MV y_s samples stay
        # within the FourierFeatures [-1, 1]³ range (which is the tight
        # invariant — any L_MV sphere extending past the bbox breaks the
        # Fourier assertion).
        from nhmo.geometry.sdf import trilinear_interp
        p_mv_wos = model_frame_to_wos_frame(p_mv)                  # (B_q, 3) in [0,1]³
        sdf_model_at_p = trilinear_interp(
            sdf_grid_wos.unsqueeze(0), p_mv_wos.unsqueeze(0),
        ).squeeze(0)                                                # (B_q,) model-frame sdf, <0 inside
        # Conservative bound: additionally cap by distance-to-model-bbox [-1, 1]³
        # so L_MV spheres stay within the Fourier range even if the SDF grid is
        # slightly wrong near corners.
        dist_to_model_bbox = (1.0 - p_mv.abs().max(dim=-1).values).clamp(min=0.0)
        sdf_at_p_mv = torch.minimum(
            (-sdf_model_at_p).clamp(min=1e-3),
            dist_to_model_bbox,
        ).clamp(min=1e-3)                                            # (B_q,)

        # WoS
        p_nll_wos = model_frame_to_wos_frame(p_nll)
        B_q = p_nll_wos.shape[0]
        sdf_grid_batched = sdf_grid_wos.unsqueeze(0).expand(B_q, -1, -1, -1).contiguous()
        hits_wos, hit_mask = self.wos_sampler.sample_hits(
            p_nll_wos, sdf_grid_batched, self.n_wos_hits, seed=idx + 1,
        )
        hits_model = wos_frame_to_model_frame(hits_wos)

        # Hit normals: look up the closest mesh surface normal per hit.
        # For speed in dry-run, approximate: normal = (hit - mesh_centroid)
        # normalized. For production, use trimesh.proximity.closest_point.
        surface_points_flat = shape_ctx["surface_points"].squeeze(0)     # (N_s, 3)
        surface_normals_flat = shape_ctx["surface_normals"].squeeze(0)
        # Approximate: for each hit, find the nearest surface point in the
        # ctx sample; use its face normal. Cheap O(N_hit × N_s).
        N_h = hits_model.shape[1]
        diff = hits_model.unsqueeze(2) - surface_points_flat.unsqueeze(0).unsqueeze(0)  # (B_q, N_h, N_s, 3)
        d2 = (diff * diff).sum(dim=-1)                                                   # (B_q, N_h, N_s)
        nearest = d2.argmin(dim=-1)                                                       # (B_q, N_h)
        hit_normals = surface_normals_flat[nearest]                                       # (B_q, N_h, 3)

        # Surface quadrature: use shape_ctx's surface points directly.
        N_surf_available = surface_points_flat.shape[0]
        n_surf = min(self.n_surface, N_surf_available)
        idx_surf = torch.randperm(N_surf_available, device=device)[:n_surf]
        surf_pts = surface_points_flat[idx_surf].unsqueeze(0).expand(B_q, -1, -1).contiguous()
        surf_normals = surface_normals_flat[idx_surf].unsqueeze(0).expand(B_q, -1, -1).contiguous()

        # Area weights: need total mesh area. The shape_generator doesn't
        # currently expose it; compute from surface samples as
        # total_area ≈ N_surf_available × per_sample_area (area_weights
        # returned by sample_surface_mesh equals total_area / N).
        # For dry run, use the approximate total_area_estimate stored in spec.
        total_area = float(spec.get("total_area_model_frame", 4.0))  # safe default
        area_weight_each = total_area / n_surf
        surface_area_weights = torch.full(
            (B_q, n_surf), area_weight_each, device=device, dtype=p_mv.dtype,
        )

        # L_BL target: sample one surface point per query
        bl_idx = torch.randperm(N_surf_available, device=device)[:B_q]
        bl_pts = surface_points_flat[bl_idx]
        bl_normals = surface_normals_flat[bl_idx]

        return {
            "shape_ctx": shape_ctx,
            "p_nll": p_nll,
            "p_mv": p_mv,
            "sdf_at_p_mv": sdf_at_p_mv,
            "zeta_hits": hits_model,
            "hit_normals": hit_normals,
            "hit_mask": hit_mask,
            "zeta_bl_0": bl_pts,
            "normal_bl_0": bl_normals,
            "zeta_surface": surf_pts,
            "normal_surface": surf_normals,
            "surface_area_weights": surface_area_weights,
            "bbox_diag_model_frame": bbox_diag,
        }
