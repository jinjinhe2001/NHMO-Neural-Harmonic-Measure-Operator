"""WoSDataset: on-the-fly WoS batcher for training.

Per v2 plan §7.1: WoS runs ON THE FLY in `__getitem__`. No pre-computed
histograms. The dataset consumes a `shape_generator` callable that returns
a shape context + auxiliary metadata per call; each call produces one
ready-to-train batch dict consumable by `LossRegistry`.

Phase 5 smoke path:
  - `shape_generator = lambda: unit_ball_shape_spec(G, n_surface, device)`
  - Produces the unit ball domain on every batch (re-sampled surface
    points + interior points for variety).

Phase 7 extension:
  - `shape_generator` returns MCB shape records loaded from disk cache.

Coordinate system: the LossRegistry-consumable output is in MODEL frame
[-1, 1]³ (Phase 2 Fourier convention). WoS is run internally in [0, 1]³
via the linear map `wos = (model + 1) / 2`.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor
from torch.utils.data import Dataset

from nhmo.core.kernel import ShapeLatent  # noqa: F401 — used downstream
from nhmo.data.synthetic import (
    BALL_RADIUS,
    ball_bbox_diag_model_frame,
    ball_sdf,
    ball_interior_samples,
    ball_surface_samples,
    model_frame_to_wos_frame,
    rasterize_ball_sdf_grid,
    unit_ball_shape_ctx,
    wos_frame_to_model_frame,
)
from nhmo.data.analytic import SphereHarmonicMeasure


class UnitBallShapeGenerator:
    """Generator that produces the unit-ball shape on demand.

    Stored grid is in WoS frame [0, 1]³; surface/interior samples returned
    in MODEL frame [-1, 1]³. The WoS-frame grid is shared (radius is fixed),
    just re-sampled surface + interior per call.
    """

    def __init__(
        self,
        grid_size: int = 64,
        n_surface: int = 512,
        n_interior_anchors: int = 512,
        radius: float = BALL_RADIUS,
        device: str = "cpu",
    ) -> None:
        self.grid_size = grid_size
        self.n_surface = n_surface
        self.n_interior_anchors = n_interior_anchors
        self.radius = radius
        self.device = device
        # Rasterize once; it's shape-invariant for the ball.
        self._sdf_grid_wos = rasterize_ball_sdf_grid(grid_size, radius, device)

    def __call__(self) -> dict:
        surf, normals = ball_surface_samples(self.n_surface, self.radius, self.device)
        interior = ball_interior_samples(self.n_interior_anchors, self.radius, self.device)
        return {
            "shape_ctx": {
                "surface_points": surf.unsqueeze(0),       # (1, N_s, 3) model frame
                "surface_normals": normals.unsqueeze(0),
                "interior_points": interior.unsqueeze(0),
            },
            "sdf_grid_wos": self._sdf_grid_wos,            # (G, G, G) wos frame
            "radius_model_frame": self.radius,
            "bbox_diag_model_frame": ball_bbox_diag_model_frame(self.radius),
        }


def _sample_query_points_model_frame(
    radius: float, n: int, device: str,
    band: tuple[float, float] | None = None,
) -> Tensor:
    return ball_interior_samples(n, radius, device, band=band)


class WoSDataset(Dataset):
    """Yields ready-to-train batches assembled from a shape_generator + WoS hits.

    Each `__getitem__(idx)` call:
      1. Obtains a shape spec from `shape_generator()`.
      2. Samples `n_queries` interior points as (p_nll, p_mv).
      3. Runs WoS from each query to get `n_wos_hits` hits.
      4. Samples `n_surface` quadrature points (re-using the shape's surface).
      5. Samples one boundary point (ζ_bl_0) + normal for L_BL.
      6. Returns a dict in MODEL frame [-1, 1]³.

    `__len__` returns `virtual_length` — the dataset is synthetic
    (infinite), so this is just the nominal epoch size.
    """

    def __init__(
        self,
        shape_generator: Callable[[], dict],
        wos_sampler,                            # WoSHitSampler or MockWoSHitSampler
        n_queries: int = 32,
        n_wos_hits: int = 4,
        n_surface: int = 2000,
        virtual_length: int = 10000,
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
        spec = self.shape_generator()
        shape_ctx = spec["shape_ctx"]
        sdf_grid_wos = spec["sdf_grid_wos"]
        radius = float(spec["radius_model_frame"])
        bbox_diag = float(spec["bbox_diag_model_frame"])
        device = shape_ctx["surface_points"].device

        # --- Query points (p_nll, p_mv) in model frame ---
        # p_mv: interior points well away from boundary (for L_MV radius).
        p_mv = _sample_query_points_model_frame(
            radius, self.n_queries, device, band=(0.2, 0.8),
        )
        # p_nll: any interior (can be same as p_mv for the smoke test).
        p_nll = p_mv.clone()

        # SDF at p_mv: |p_mv| - radius (negative inside → take absolute for MV).
        sdf_at_p_mv = (radius - p_mv.norm(dim=-1)).clamp(min=1e-6)  # positive = dist to boundary

        # --- WoS hits from p_nll ---
        # Convert p_nll → WoS frame, run WoS, convert hits back to model frame.
        p_nll_wos = model_frame_to_wos_frame(p_nll)
        # Batch across queries: treat each query as its own "shape" for the
        # WoS call (single grid shared, one query each).
        B_q = p_nll_wos.shape[0]
        sdf_grid_batched = sdf_grid_wos.unsqueeze(0).expand(B_q, -1, -1, -1).contiguous()

        # Deterministic WoS seed tied to idx — lets checkpoint-resume test
        # reproduce the same hit sequence.
        hits_wos, hit_mask = self.wos_sampler.sample_hits(
            p_nll_wos, sdf_grid_batched, self.n_wos_hits, seed=idx + 1,
        )
        # hits_wos: (B_q, n_walks, 3) in [0, 1]³; convert to model frame.
        hits_model = wos_frame_to_model_frame(hits_wos)
        # hit_normals: for the unit ball the outward normal equals the
        # normalized hit position (since hits lie approximately on the sphere).
        hit_normals = hits_model / (hits_model.norm(dim=-1, keepdim=True) + 1e-6)

        # --- Surface quadrature samples (ζ_surface) ---
        surf_pts, surf_normals = ball_surface_samples(self.n_surface, radius, device)
        surf_pts = surf_pts.unsqueeze(0).expand(B_q, -1, -1)
        surf_normals = surf_normals.unsqueeze(0).expand(B_q, -1, -1)
        area_weight_each = 4.0 * math.pi * (radius ** 2) / self.n_surface
        surface_area_weights = torch.full(
            (B_q, self.n_surface), area_weight_each, device=device, dtype=p_mv.dtype,
        )

        # --- L_BL target: one boundary point per query ---
        bl_pts, bl_normals = ball_surface_samples(B_q, radius, device)

        # --- Assemble batch: flatten query dim into batch dim (B = B_q). ---
        # The LossRegistry expects shape_latent with dim B; we'll use B_q batches
        # of the same shape_latent. Trainer encodes once and expands.
        batch = {
            "shape_ctx": shape_ctx,               # keep the (1, *, 3) shape_ctx; trainer expands
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
        return batch


class RandomSphereShapeGenerator:
    """§8.1 curriculum: random ball with scale s ∈ [0.5, 2.0], translation
    t ∈ [-t_max, t_max]³, rotation R ∈ SO(3). Returns a shape spec with
    grid-rasterized SDF in WoS frame + analytic kernel instance.

    Constraints:
      - radius = s * radius_base (radius_base=0.3 by default)
      - translation keeps ball inside [-1, 1]³ model frame:
        |c| + radius ≤ 1 per axis. With radius_max = 0.6 and
        t_max = 0.3 along each axis we stay inside with slight margin.

    Rotation is mathematically a no-op for an isotropic ball, but the
    encoder's shape_latent receives rotated surface samples; the
    kernel head sees the rotated sample configuration. Analytic
    `SphereHarmonicMeasure(radius, center, rotation)` accounts for this
    when computing K_true.
    """

    def __init__(
        self,
        grid_size: int = 64,
        n_surface: int = 512,
        n_interior_anchors: int = 512,
        radius_base: float = 0.3,
        scale_min: float = 0.5,
        scale_max: float = 2.0,
        translate_max: float = 0.3,
        enable_rotation: bool = True,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        self.grid_size = grid_size
        self.n_surface = n_surface
        self.n_interior_anchors = n_interior_anchors
        self.radius_base = radius_base
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.translate_max = translate_max
        self.enable_rotation = enable_rotation
        self.device = device
        self.rng = torch.Generator(device=device)
        if seed is not None:
            self.rng.manual_seed(int(seed))

    def _random_rotation(self) -> Tensor:
        """Random 3D rotation via QR on a Gaussian matrix (3, 3)."""
        A = torch.randn(3, 3, generator=self.rng, device=self.device)
        Q, R_up = torch.linalg.qr(A)
        # Fix sign to ensure det(Q) = +1 (SO(3), not O(3)).
        d = torch.sign(torch.diag(R_up))
        d = torch.where(d == 0, torch.ones_like(d), d)
        Q = Q * d.unsqueeze(0)
        if torch.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        return Q

    def _rasterize_translated_ball(self, radius: float, center: Tensor) -> Tensor:
        """Rasterize SDF grid in WoS frame for a translated/scaled ball.

        WoS-frame distance: d_wos(p_wos, q_wos) = 2 · d_model(p_model, q_model).
        For WoS kernel to step correctly in [0, 1]³, store WoS-frame SDF:
            sdf_wos = |p_wos − c_wos| − r_wos
        where c_wos = (c_model + 1) / 2, r_wos = radius / 2.

        (Rotation is sphere-invariant, not stored in the grid.)
        """
        G = self.grid_size
        coords = torch.linspace(0.0, 1.0, G, device=self.device)
        iz, iy, ix = torch.meshgrid(coords, coords, coords, indexing="ij")
        pts_wos = torch.stack([ix, iy, iz], dim=-1)           # (G, G, G, 3)
        c_wos = (center + 1.0) * 0.5                           # (3,)
        r_wos = radius * 0.5
        return (pts_wos - c_wos).norm(dim=-1) - r_wos          # (G, G, G)

    def __call__(self) -> dict:
        s = (self.scale_min
             + (self.scale_max - self.scale_min)
             * torch.rand(1, generator=self.rng, device=self.device).item())
        radius = s * self.radius_base
        # Keep the ball in [-1, 1]³ model frame with margin.
        max_c = max(0.0, 1.0 - radius - 0.05)
        t_max = min(self.translate_max, max_c)
        center = (
            2 * torch.rand(3, generator=self.rng, device=self.device) - 1.0
        ) * t_max                                               # (3,)
        rotation = self._random_rotation() if self.enable_rotation else None

        # Sample surface + normals in a canonical unit-sphere frame, then
        # scale, rotate, translate into the domain.
        canonical_surf, canonical_norm = ball_surface_samples(
            self.n_surface, radius=1.0, device=self.device,
        )
        if rotation is not None:
            # Rotate; the sphere is isotropic so geometrically identical
            # but the encoder sees distinct configurations.
            canonical_surf = canonical_surf @ rotation.T
            canonical_norm = canonical_norm @ rotation.T
        surf = canonical_surf * radius + center                 # (N_s, 3) model frame
        normals = canonical_norm                                 # outward unit (already)

        canonical_int = ball_interior_samples(
            self.n_interior_anchors, radius=1.0, device=self.device,
        )
        if rotation is not None:
            canonical_int = canonical_int @ rotation.T
        interior = canonical_int * radius + center               # (N_i, 3) model frame

        sdf_grid_wos = self._rasterize_translated_ball(radius, center)

        analytic_kernel = SphereHarmonicMeasure(
            radius=radius,
            center=center.clone(),
            rotation=rotation.clone() if rotation is not None else None,
        )

        bbox_diag = 2.0 * math.sqrt(3.0)    # still [-1, 1]³ bounding box

        return {
            "shape_ctx": {
                "surface_points": surf.unsqueeze(0),
                "surface_normals": normals.unsqueeze(0),
                "interior_points": interior.unsqueeze(0),
            },
            "sdf_grid_wos": sdf_grid_wos,
            "radius_model_frame": radius,
            "center_model_frame": center,
            "rotation_model_frame": rotation,
            "analytic_kernel": analytic_kernel,
            "bbox_diag_model_frame": bbox_diag,
        }


class AnalyticBallDataset(WoSDataset):
    """Phase 6 pretraining dataset: random sphere + precomputed analytic K_true.

    Inherits all of Phase 5's WoSDataset plumbing (query sampling, WoS hits,
    surface quadrature) and adds `log_K_true_at_surface` to the batch dict
    for LossRegistry's L_analytic computation.

    Assumes `shape_generator()` returns an additional `analytic_kernel`
    entry (a SphereHarmonicMeasure instance configured for the same domain).
    """

    @torch.no_grad()
    def __getitem__(self, idx: int) -> dict:
        spec = self.shape_generator()
        shape_ctx = spec["shape_ctx"]
        sdf_grid_wos = spec["sdf_grid_wos"]
        radius = float(spec["radius_model_frame"])
        center = spec["center_model_frame"]
        bbox_diag = float(spec["bbox_diag_model_frame"])
        analytic_kernel = spec["analytic_kernel"]
        device = shape_ctx["surface_points"].device

        # Interior query points in this domain: |p - c| < radius.
        # Reuse ball_interior_samples at radius 1 and then scale+translate.
        canonical_p = ball_interior_samples(
            self.n_queries, radius=1.0, device=device, band=(0.2, 0.8),
        )
        rotation = spec.get("rotation_model_frame")
        if rotation is not None:
            canonical_p = canonical_p @ rotation.T
        p_mv = canonical_p * radius + center.unsqueeze(0)
        p_nll = p_mv.clone()

        # SDF at p_mv (distance to boundary = radius - |canonical_p|)
        sdf_at_p_mv = (radius * (1.0 - canonical_p.norm(dim=-1))).clamp(min=1e-6)

        # WoS from p_nll
        p_nll_wos = model_frame_to_wos_frame(p_nll)
        B_q = p_nll_wos.shape[0]
        sdf_grid_batched = sdf_grid_wos.unsqueeze(0).expand(B_q, -1, -1, -1).contiguous()
        hits_wos, hit_mask = self.wos_sampler.sample_hits(
            p_nll_wos, sdf_grid_batched, self.n_wos_hits, seed=idx + 1,
        )
        hits_model = wos_frame_to_model_frame(hits_wos)
        # Hit normals: for ball, outward normal at ζ is (ζ - center) / radius.
        hit_offsets = hits_model - center.unsqueeze(0).unsqueeze(0)
        hit_normals = hit_offsets / (hit_offsets.norm(dim=-1, keepdim=True) + 1e-6)

        # Surface quadrature samples: use shape_ctx's surface points, expanded
        # across the query batch.
        surf_pts_single = shape_ctx["surface_points"].squeeze(0)      # (N, 3)
        surf_normals_single = shape_ctx["surface_normals"].squeeze(0)
        N_surf = surf_pts_single.shape[0]
        surf_pts = surf_pts_single.unsqueeze(0).expand(B_q, N_surf, 3)
        surf_normals = surf_normals_single.unsqueeze(0).expand(B_q, N_surf, 3)
        area_weight_each = 4.0 * math.pi * (radius ** 2) / N_surf
        surface_area_weights = torch.full(
            (B_q, N_surf), area_weight_each, device=device, dtype=p_mv.dtype,
        )

        # Boundary point for L_BL (not actually used in pretraining cfg since
        # lambda_bl is null, but we include it for dict-shape consistency).
        bl_offsets = ball_surface_samples(B_q, radius=1.0, device=device)[0]
        if rotation is not None:
            bl_offsets = bl_offsets @ rotation.T
        bl_pts = bl_offsets * radius + center.unsqueeze(0)
        bl_normals = bl_offsets                                        # outward

        # L_analytic supervision: log K_true at surface samples
        log_K_true_at_surface = analytic_kernel.log_kernel(
            p_nll, surf_pts, surf_normals,
        )

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
            "log_K_true_at_surface": log_K_true_at_surface,
            "bbox_diag_model_frame": bbox_diag,
        }


def collate_wos_batches(batch_list: list[dict]) -> dict:
    """Identity collator for Phase 5 single-batch smoke test (B=1 list).

    Phase 5 uses `shuffle=False, batch_size=1`; the dataset already emits
    a pre-batched dict (query dim = effective batch dim). This collate
    unpacks the length-1 list.
    """
    assert len(batch_list) == 1, (
        f"Phase 5 smoke test uses batch_size=1; got {len(batch_list)}. "
        f"Phase 7's MCB loader may support multi-shape batching."
    )
    return batch_list[0]
