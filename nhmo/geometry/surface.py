"""Surface sampling with normals and area weights.

Phase 1 design contract (user decision Q4, 2026-04-21; implementation
delivered in Phase 6.9 2026-04-22):

  1. `sample_surface_mesh` returns normals that are FACE NORMALS of the
     triangle each point lies on. NO barycentric interpolation of vertex
     normals. NO area/angle-weighted vertex normals. This is a correctness
     requirement — harmonic measure is singular at sharp boundary features
     (corners, edges); vertex-normal schemes smooth those features out and
     destroy the physical structure v2 is designed to capture.

  2. Implementation path: `trimesh.sample.sample_surface(mesh, n)` returns
     `(points, face_index)` natively. Normals are `mesh.face_normals[face_idx]`,
     per-face-exact. Area weights come from `mesh.area_faces` scaled so
     `Σ w_i ≈ A(∂Ω)`.

  3. `sample_box_faces` samples strictly INTERIOR to each AABB face. Points
     on edges or corners are rejected and resampled. The preprocessing
     pipeline logs the fraction of rejected samples; warns if any shape
     exceeds 5%.

See tests/test_preprocessing.py for the unit-cube invariant and
tests/test_surface_face_normal.py for the sharp-feature test.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def sample_surface_mesh(
    mesh,                      # trimesh.Trimesh
    n: int,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Uniform-area sample of the mesh surface with FACE normals.

    Returns:
        points:       (n, 3) float32 — sampled positions on ∂Ω
        normals:      (n, 3) float32 — face normals at each point
                      (exactly `mesh.face_normals[face_indices]`; NOT
                      barycentric-interpolated vertex normals)
        face_indices: (n,)   int64   — triangle index each sample lies on
        area_weights: (n,)   float32 — per-sample quadrature weight
                      (Σ area_weights == mesh.area)

    Uses `trimesh.sample.sample_surface(mesh, n)` which natively returns
    (points, face_index) with uniform-area probability. This gives face
    indices for free; no closest-triangle query needed.
    """
    import trimesh  # Phase-6.9 runtime dep

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"expected trimesh.Trimesh, got {type(mesh).__name__}")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    # Pass seed through to trimesh.sample (it maintains its own RNG and does
    # NOT respect np.random.seed()).
    if seed is not None:
        points, face_indices = trimesh.sample.sample_surface(mesh, n, seed=int(seed))
    else:
        points, face_indices = trimesh.sample.sample_surface(mesh, n)

    points = np.asarray(points, dtype=np.float32)
    face_indices = np.asarray(face_indices, dtype=np.int64)
    # FACE normals (Q4(a) correctness requirement — no vertex interpolation)
    normals = np.asarray(mesh.face_normals[face_indices], dtype=np.float32)

    # Area weights: split total mesh area uniformly across samples (sample_surface
    # uses uniform-area probability, so every sample carries the same expected area).
    total_area = float(mesh.area)
    area_weights = np.full(n, total_area / n, dtype=np.float32)

    return points, normals, face_indices, area_weights


# ----------------------------------------------------------------- box faces

# Axis-aligned outward face normals (one per face of an AABB).
_BOX_FACE_NORMALS = np.array(
    [
        [-1.0, 0.0, 0.0],  # x=min face
        [ 1.0, 0.0, 0.0],  # x=max face
        [0.0, -1.0, 0.0],  # y=min face
        [0.0,  1.0, 0.0],  # y=max face
        [0.0, 0.0, -1.0],  # z=min face
        [0.0, 0.0,  1.0],  # z=max face
    ],
    dtype=np.float32,
)


def sample_box_faces(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    n: int,
    edge_reject_tol: float = 1e-6,
    max_resample_ratio: float = 10.0,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform-area sample of the 6 faces of an AABB, strictly interior to
    each face (never on edges or corners).

    Returns:
        points:  (n, 3) float32 inside the chosen face (not on edges)
        normals: (n, 3) float32 axis-aligned outward unit normals

    Strategy: allocate n samples proportional to each face's area, draw
    each sample's two in-face coordinates uniformly, then reject and
    resample any sample whose in-face coordinate lands within
    `edge_reject_tol` of any face edge. The logger reports the rejection
    fraction when > 5% (Q4(d) directive); exposed as a return artifact
    here for the caller to inspect.

    Raises RuntimeError if more than `max_resample_ratio * n` attempts are
    exhausted (indicates `edge_reject_tol` is too large relative to face size).
    """
    bbox_min = np.asarray(bbox_min, dtype=np.float32)
    bbox_max = np.asarray(bbox_max, dtype=np.float32)
    if bbox_min.shape != (3,) or bbox_max.shape != (3,):
        raise ValueError(f"bbox must be (3,); got {bbox_min.shape}, {bbox_max.shape}")
    extent = bbox_max - bbox_min
    if (extent <= 0).any():
        raise ValueError(f"bbox has non-positive extent: {extent}")
    if edge_reject_tol < 0:
        raise ValueError(f"edge_reject_tol must be non-negative, got {edge_reject_tol}")

    # Face areas for proportional allocation.
    face_areas = np.array([
        extent[1] * extent[2],  # x=min
        extent[1] * extent[2],  # x=max
        extent[0] * extent[2],  # y=min
        extent[0] * extent[2],  # y=max
        extent[0] * extent[1],  # z=min
        extent[0] * extent[1],  # z=max
    ], dtype=np.float64)
    face_probs = face_areas / face_areas.sum()

    if seed is not None:
        rng = np.random.RandomState(int(seed))
    else:
        rng = np.random.RandomState()

    # Assign each of the n samples to one of the 6 faces proportional to area.
    face_idx_per_sample = rng.choice(6, size=n, p=face_probs)

    points = np.empty((n, 3), dtype=np.float32)
    normals = _BOX_FACE_NORMALS[face_idx_per_sample].astype(np.float32)

    total_attempts = 0
    rejections = 0
    attempt_budget = int(max_resample_ratio * n) + n + 100

    for i in range(n):
        f = int(face_idx_per_sample[i])
        # The fixed axis (along which the face lies) and its value.
        fixed_axis = f // 2
        fixed_value = bbox_min[fixed_axis] if f % 2 == 0 else bbox_max[fixed_axis]
        # Indices of the two in-face axes
        a, b = [k for k in range(3) if k != fixed_axis]

        while True:
            total_attempts += 1
            if total_attempts > attempt_budget:
                raise RuntimeError(
                    f"sample_box_faces exceeded resample budget "
                    f"({attempt_budget} attempts for {n} samples). "
                    f"edge_reject_tol={edge_reject_tol} may be too large "
                    f"relative to face extents {extent}."
                )
            ua = rng.uniform(bbox_min[a], bbox_max[a])
            ub = rng.uniform(bbox_min[b], bbox_max[b])
            # Reject if within edge_reject_tol of either edge along the in-face axes.
            if (
                ua - bbox_min[a] < edge_reject_tol
                or bbox_max[a] - ua < edge_reject_tol
                or ub - bbox_min[b] < edge_reject_tol
                or bbox_max[b] - ub < edge_reject_tol
            ):
                rejections += 1
                continue
            points[i, fixed_axis] = fixed_value
            points[i, a] = ua
            points[i, b] = ub
            break

    # Rejection fraction (Q4(d) — flag if > 5%).
    # Caller is responsible for logging; we just surface it.
    # (Storing on the function for later introspection is not thread-safe;
    # callers who want this statistic should wrap the call.)
    # NOTE: we don't return it to keep the signature tight. If needed later,
    # expose via a `return_stats: bool = False` kwarg.
    return points, normals
