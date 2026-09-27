"""MNIST digit-shaped 2D domain generator.

Treats each MNIST digit's outermost contour as a simply-connected domain
boundary in [-1, 1]². Multiply-connected (digit-with-holes) is left for
a follow-up; the simply-connected path is sufficient to validate v2's
2D pipeline on real MNIST data.

Pipeline per digit:
  1. Read 28×28 image, binarize, extract contours.
  2. Pick the LARGEST closed contour as the outer boundary (drops
     internal "0" / "8" / "9" hole boundaries — those are Phase 7.0.3).
  3. Resample to N_surface points (uniform arc-length).
  4. Rasterize signed distance on a G×G grid (uses
     `nhmo.geometry.sdf_2d.polygon_sdf`).
  5. Sample interior points via SDF rejection.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor

from nhmo.data.mnist_domains import (
    MnistDigitDomain,
    extract_digit_contours,
    binarize_digit,
    pixel_to_model_frame,
    load_mnist_split,
)
from nhmo.data.synthetic_2d import model_frame_to_wos_frame, wos_frame_to_model_frame
from nhmo.geometry.sdf_2d import polygon_sdf


def _resample_arc_length(polyline: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed 2D polyline to n points uniformly in arc length.

    polyline: (V, 2). Returns (n, 2). The closed-loop edge from last to
    first vertex is included in the arc-length parameterization.
    """
    pts = np.asarray(polyline, dtype=np.float64)
    closed = np.concatenate([pts, pts[:1]], axis=0)              # (V+1, 2)
    seg = np.diff(closed, axis=0)                                # (V, 2)
    seg_len = np.linalg.norm(seg, axis=1)                        # (V,)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])             # (V+1,)
    total = float(cum[-1])
    if total <= 1e-12:
        raise ValueError("degenerate polyline (zero arc length)")
    # Sample n target arc lengths in [0, total) — note half-open to keep
    # the polyline closed without duplicating the seam vertex.
    s = np.linspace(0.0, total, n + 1)[:-1]
    out = np.empty((n, 2), dtype=np.float32)
    for j in range(n):
        target = s[j]
        idx = np.searchsorted(cum, target, side="right") - 1
        idx = max(0, min(idx, len(seg_len) - 1))
        seg_t = (target - cum[idx]) / max(seg_len[idx], 1e-12)
        out[j] = (closed[idx] + seg_t * seg[idx]).astype(np.float32)
    return out


def _polyline_outward_normals(polyline_ccw: np.ndarray) -> np.ndarray:
    """Compute outward unit normals at each vertex of a CCW polyline.

    For a CCW outer boundary, the outward normal at edge (a, b) is
    rotate-(b - a) by -90° = (dy, -dx).
    """
    n = len(polyline_ccw)
    edges = np.roll(polyline_ccw, -1, axis=0) - polyline_ccw      # (n, 2)
    # Vertex-normal = average of adjacent-edge normals.
    edge_normals = np.stack([edges[:, 1], -edges[:, 0]], axis=-1)  # (n, 2)
    edge_normals /= (np.linalg.norm(edge_normals, axis=-1, keepdims=True) + 1e-12)
    vert_normals = 0.5 * (edge_normals + np.roll(edge_normals, 1, axis=0))
    vert_normals /= (np.linalg.norm(vert_normals, axis=-1, keepdims=True) + 1e-12)
    return vert_normals.astype(np.float32)


def _signed_area(poly: np.ndarray) -> float:
    """Signed area: positive for CCW, negative for CW."""
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _ensure_ccw(poly: np.ndarray) -> np.ndarray:
    if _signed_area(poly) < 0:
        return poly[::-1].copy()
    return poly


class MNISTDigitShapeGenerator:
    """Static generator for one MNIST digit's outer contour as ∂Ω.

    On each __call__ returns the same shape_ctx + SDF grid (the digit
    geometry doesn't change). Round-robin across multiple digits is
    handled at the trainer level.
    """

    def __init__(
        self,
        digit_image: np.ndarray,                       # (28, 28) uint8
        digit_index: int,
        label: int,
        grid_size: int = 64,
        n_surface: int = 256,
        n_interior_anchors: int = 256,
        device: str = "cpu",
        n_polyline_vertices: int = 200,
    ) -> None:
        self.digit_index = digit_index
        self.label = label
        self.grid_size = grid_size
        self.n_surface = n_surface
        self.n_interior_anchors = n_interior_anchors
        self.device = device

        bin_img = binarize_digit(digit_image)
        contours = extract_digit_contours(bin_img, n_polyline_vertices)
        if len(contours) == 0:
            raise ValueError(f"digit {digit_index} produced no contours")
        # Pick outermost (largest perimeter) — simply-connected domain.
        sizes = [
            float(np.sum(np.linalg.norm(np.diff(np.concatenate([c, c[:1]], axis=0), axis=0), axis=1)))
            for c in contours
        ]
        outer_pixel = contours[int(np.argmax(sizes))]
        outer_model = pixel_to_model_frame(outer_pixel)               # (V, 2) in [-1, 1]²
        outer_model = _ensure_ccw(outer_model)
        self._polyline_model = _resample_arc_length(outer_model, n_surface)
        self._polyline_normals = _polyline_outward_normals(self._polyline_model)
        # Total perimeter (for surface quadrature weight)
        edges = np.roll(self._polyline_model, -1, axis=0) - self._polyline_model
        self._perimeter = float(np.sum(np.linalg.norm(edges, axis=1)))
        # Bbox diag in model frame (digits are within [-0.8, 0.8]² roughly)
        bbox_min = self._polyline_model.min(axis=0)
        bbox_max = self._polyline_model.max(axis=0)
        self._bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))

        # Rasterize SDF on G×G grid in WoS frame [0, 1]²
        G = grid_size
        coords = torch.linspace(0.0, 1.0, G)
        iy, ix = torch.meshgrid(coords, coords, indexing="ij")
        pts_wos = torch.stack([ix, iy], dim=-1)                       # (G, G, 2)
        pts_model = wos_frame_to_model_frame(pts_wos)                 # (G, G, 2)
        poly_t = torch.from_numpy(self._polyline_model)
        sdf = polygon_sdf(pts_model.reshape(-1, 2), poly_t).reshape(G, G)
        self._sdf_grid_wos = sdf.to(device).contiguous()

        # Pre-sample interior points (used as anchors; per-step queries
        # subsample from these).
        self._interior_pool = self._sample_interior_pool(n_interior_anchors * 4)

    def _sample_interior_pool(self, n: int) -> Tensor:
        bbox_min = self._polyline_model.min(axis=0).astype(np.float32)
        bbox_max = self._polyline_model.max(axis=0).astype(np.float32)
        out = []
        attempts = 0
        max_attempts = n * 50
        rng = np.random.RandomState(self.digit_index)
        while sum(len(a) for a in out) < n and attempts < max_attempts:
            chunk = rng.uniform(bbox_min, bbox_max, size=(n, 2)).astype(np.float32)
            sd = polygon_sdf(torch.from_numpy(chunk), torch.from_numpy(self._polyline_model)).numpy()
            inside = chunk[sd < 0]
            if len(inside) > 0:
                out.append(inside)
            attempts += n
        if sum(len(a) for a in out) < n:
            raise RuntimeError(
                f"digit {self.digit_index}: interior rejection sampler exhausted "
                f"{attempts}, found only {sum(len(a) for a in out)}/{n}"
            )
        pool = np.concatenate(out, axis=0)[:n]
        return torch.from_numpy(pool).to(self.device)

    def __call__(self) -> dict:
        # Surface points + normals — fixed (they're just the polyline)
        surf = torch.from_numpy(self._polyline_model).to(self.device)
        surf_n = torch.from_numpy(self._polyline_normals).to(self.device)
        # Interior anchors: pick n_interior from the pool
        N_pool = self._interior_pool.shape[0]
        idx = torch.randperm(N_pool, device=self.device)[: self.n_interior_anchors]
        interior = self._interior_pool[idx]

        return {
            "shape_ctx": {
                "surface_points": surf.unsqueeze(0),                  # (1, N_s, 2)
                "surface_normals": surf_n.unsqueeze(0),
                "interior_points": interior.unsqueeze(0),
            },
            "sdf_grid_wos": self._sdf_grid_wos,
            "perimeter_model_frame": self._perimeter,
            "bbox_diag_model_frame": self._bbox_diag,
            "radius_model_frame": 0.5 * self._bbox_diag,              # rough proxy
        }


def _square_polyline(n_per_edge: int = 64) -> np.ndarray:
    """CCW unit square [-1, 1]² polyline with n_per_edge points per edge.

    Returned in CCW order so its outward normal points OUT of [-1,1]² (and
    out of Ω in the multiply-connected MNIST setup).
    """
    t = np.linspace(-1.0, 1.0, n_per_edge + 1, dtype=np.float32)[:-1]
    bottom = np.stack([t, -np.ones_like(t)], axis=-1)
    right = np.stack([np.ones_like(t), t], axis=-1)
    top = np.stack([-t, np.ones_like(t)], axis=-1)
    left = np.stack([-np.ones_like(t), -t], axis=-1)
    return np.concatenate([bottom, right, top, left], axis=0)


def _square_sdf(p: Tensor, half: float = 1.0) -> Tensor:
    """Signed distance to the axis-aligned square [-half, half]². Negative inside."""
    q = p.abs() - half
    outside = q.clamp(min=0.0).norm(dim=-1)
    inside = q.max(dim=-1).values.clamp(max=0.0)
    return outside + inside


class MNISTMultiplyConnectedShapeGenerator:
    """Multiply-connected MNIST domain: Ω = [-1, 1]² \\ digit_interior(s).

    ∂Ω = outer unit square ∪ ALL digit contours. The digit contours act as
    holes in the unit-square domain (matches v1 MNIST setup; the v1 paper's
    3.6% rel-L2 baseline assumed this topology).

    Outward-normal convention (PHILOSOPHY.md "outward = away from Ω"):
      - Outer square: outward = away from origin (outward of [-1,1]²).
      - Each digit contour: outward = INTO digit (away from background, i.e.,
        away from Ω). For CCW digit polylines, this is the INWARD normal of
        the polygon, i.e., (-tangent_y, tangent_x).
    """

    def __init__(
        self,
        digit_image: np.ndarray,
        digit_index: int,
        label: int,
        grid_size: int = 64,
        n_surface_per_curve: int = 128,   # samples per ∂Ω component (legacy mode)
        n_square_edge: int = 64,          # samples per side of outer square (legacy mode)
        n_interior_anchors: int = 256,
        device: str = "cpu",
        n_polyline_vertices: int = 200,
        keep_top_k_digit_contours: int = 4,
        n_total_samples: int | None = None,  # if set, allocate ∝ perimeter (v1 mode)
    ) -> None:
        self.digit_index = digit_index
        self.label = label
        self.grid_size = grid_size
        self.device = device
        self.n_interior_anchors = n_interior_anchors

        # --- Extract digit contours, keep up to top-K by perimeter ---
        bin_img = binarize_digit(digit_image)
        contours = extract_digit_contours(bin_img, n_polyline_vertices)
        if len(contours) == 0:
            raise ValueError(f"digit {digit_index} produced no contours")
        sized = sorted(
            contours,
            key=lambda c: float(np.sum(np.linalg.norm(
                np.diff(np.concatenate([c, c[:1]], axis=0), axis=0), axis=1
            ))),
            reverse=True,
        )
        kept = sized[:keep_top_k_digit_contours]

        # Per-curve sample budget: v1-style proportional allocation if n_total_samples set,
        # else legacy fixed-per-component allocation.
        if n_total_samples is not None:
            # Compute raw arc-lengths in model frame (after pixel→model conversion).
            kept_model_raw = [_ensure_ccw(pixel_to_model_frame(c)) for c in kept]
            def _arclen(poly):
                return float(np.sum(np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)))
            digit_arclens = [_arclen(p) for p in kept_model_raw]
            square_arclen = 4.0 * 2.0  # outer square in [-1,1]² has perimeter 8
            total_arc = square_arclen + sum(digit_arclens)
            # Allocate samples ∝ arclength, ensure ≥ 4 per curve.
            n_square = max(8, int(round(n_total_samples * square_arclen / total_arc)))
            digit_n = [
                max(4, int(round(n_total_samples * a / total_arc)))
                for a in digit_arclens
            ]
            # Build resampled polylines.
            digit_polylines: List[np.ndarray] = []
            for cm_raw, n_d in zip(kept_model_raw, digit_n):
                digit_polylines.append(_resample_arc_length(cm_raw, n_d).astype(np.float32))
            self._digit_polylines = digit_polylines
            # Square: split n_square across 4 sides as evenly as possible.
            self._square_polyline = _square_polyline(max(2, n_square // 4)).astype(np.float32)
        else:
            digit_polylines = []
            for c in kept:
                cm = pixel_to_model_frame(c)
                cm = _ensure_ccw(cm)
                cm = _resample_arc_length(cm, n_surface_per_curve)
                digit_polylines.append(cm.astype(np.float32))
            self._digit_polylines = digit_polylines
            self._square_polyline = _square_polyline(n_square_edge).astype(np.float32)
        self._square_normals = _polyline_outward_normals(self._square_polyline)

        # --- Digit-contour normals: flip polygon outward → into digit ---
        digit_normals: List[np.ndarray] = []
        for poly in self._digit_polylines:
            poly_normals = _polyline_outward_normals(poly)            # CCW outward = INTO Ω
            digit_normals.append(-poly_normals)                        # flip → INTO digit (outward of Ω)
        self._digit_normals = digit_normals

        # --- Concatenate into one (N_s, 2) surface buffer ---
        all_pts = [self._square_polyline] + self._digit_polylines
        all_norms = [self._square_normals] + self._digit_normals
        self._surface_points = np.concatenate(all_pts, axis=0)
        self._surface_normals = np.concatenate(all_norms, axis=0)

        # --- Per-component arc lengths for quadrature weights ---
        def _perim(poly: np.ndarray) -> float:
            edges = np.roll(poly, -1, axis=0) - poly
            return float(np.sum(np.linalg.norm(edges, axis=1)))
        self._segment_lengths = [_perim(p) for p in all_pts]            # one per curve
        # Per-vertex weight = curve_perimeter / N_curve_samples (uniform arc-length)
        weights: List[float] = []
        for poly, perim in zip(all_pts, self._segment_lengths):
            weights.extend([perim / len(poly)] * len(poly))
        self._surface_weights = np.array(weights, dtype=np.float32)
        self._total_perimeter = float(np.sum(self._segment_lengths))

        # --- Rasterize SDF on G×G grid in WoS frame [0, 1]² ---
        # sdf_Ω(p) = max(sdf_square(p), -sdf_digit_i(p) for all i)
        G = grid_size
        coords = torch.linspace(0.0, 1.0, G)
        iy, ix = torch.meshgrid(coords, coords, indexing="ij")
        pts_wos = torch.stack([ix, iy], dim=-1).reshape(-1, 2)         # (G², 2)
        pts_model = wos_frame_to_model_frame(pts_wos)                  # (G², 2)
        sdf = _square_sdf(pts_model, half=1.0)                          # (G²,) negative inside square
        for poly in self._digit_polylines:
            sdf_d = polygon_sdf(pts_model, torch.from_numpy(poly))     # negative inside digit
            sdf = torch.maximum(sdf, -sdf_d)                            # subtract digit hole
        self._sdf_grid_wos = sdf.reshape(G, G).to(device).contiguous()

        # --- Bounding box of Ω: just the unit square ---
        self._bbox_diag = float(np.sqrt(2.0 * (2.0 ** 2)))             # diag of [-1,1]² = 2√2

        # --- Pre-sample interior points (Ω = inside square AND outside all digits) ---
        self._interior_pool = self._sample_interior_pool(n_interior_anchors * 4)

    def _sample_interior_pool(self, n: int) -> Tensor:
        out: List[np.ndarray] = []
        attempts = 0
        max_attempts = n * 100
        rng = np.random.RandomState(self.digit_index)
        # Stay inside [-0.95, 0.95]² to leave epsilon-margin from outer boundary.
        lo, hi = np.array([-0.95, -0.95], dtype=np.float32), np.array([0.95, 0.95], dtype=np.float32)
        while sum(len(a) for a in out) < n and attempts < max_attempts:
            chunk = rng.uniform(lo, hi, size=(n * 2, 2)).astype(np.float32)
            chunk_t = torch.from_numpy(chunk)
            # Reject anything inside any digit
            inside_any_digit = torch.zeros(chunk_t.shape[0], dtype=torch.bool)
            for poly in self._digit_polylines:
                sd = polygon_sdf(chunk_t, torch.from_numpy(poly))
                inside_any_digit = inside_any_digit | (sd < 0)
            keep_mask = (~inside_any_digit).numpy()
            kept = chunk[keep_mask]
            if len(kept) > 0:
                out.append(kept)
            attempts += n * 2
        total = sum(len(a) for a in out)
        if total < n:
            raise RuntimeError(
                f"digit {self.digit_index}: multiply-connected interior sampler "
                f"exhausted {attempts}, found only {total}/{n}"
            )
        pool = np.concatenate(out, axis=0)[:n]
        return torch.from_numpy(pool).to(self.device)

    def __call__(self) -> dict:
        surf = torch.from_numpy(self._surface_points).to(self.device)
        surf_n = torch.from_numpy(self._surface_normals).to(self.device)
        N_pool = self._interior_pool.shape[0]
        idx = torch.randperm(N_pool, device=self.device)[: self.n_interior_anchors]
        interior = self._interior_pool[idx]
        return {
            "shape_ctx": {
                "surface_points": surf.unsqueeze(0),
                "surface_normals": surf_n.unsqueeze(0),
                "interior_points": interior.unsqueeze(0),
            },
            "sdf_grid_wos": self._sdf_grid_wos,
            "perimeter_model_frame": self._total_perimeter,
            "bbox_diag_model_frame": self._bbox_diag,
            "radius_model_frame": 0.5 * self._bbox_diag,
            # Optional: per-vertex quadrature weights (variable per curve).
            "surface_weights_per_vertex": torch.from_numpy(self._surface_weights).to(self.device),
        }


class MNISTDigitDataset:
    """Loads MNIST images from `mnist_local_root/raw/` and exposes a list
    of (digit_index, label, image_bytes) tuples.
    """

    def __init__(
        self,
        mnist_local_root: str | Path,
        split: str = "train",
        max_digits: Optional[int] = None,
    ) -> None:
        self.local_root = Path(mnist_local_root)
        if not (self.local_root / "raw").exists():
            raise FileNotFoundError(
                f"MNIST raw/ dir missing under {self.local_root}. "
                "Expected: raw/{train|t10k}-images-idx3-ubyte.gz etc."
            )
        # load_mnist_split takes the directory containing the idx files
        # directly. Standard MNIST layout: $local_root/raw/{train|t10k}-*.
        raw_dir = self.local_root / "raw"
        imgs, labels = load_mnist_split(str(raw_dir), split=split)
        if max_digits is not None:
            imgs = imgs[:max_digits]
            labels = labels[:max_digits]
        self.imgs = imgs
        self.labels = labels

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int) -> tuple[int, int, np.ndarray]:
        return idx, int(self.labels[idx]), self.imgs[idx]
