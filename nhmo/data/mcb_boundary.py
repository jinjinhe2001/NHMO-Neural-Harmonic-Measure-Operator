"""Tet-mesh → triangle-mesh boundary extraction for Phase 7.1 MCB pipeline.

NGF's `sol.npz` files ship the MCB geometry as a tetrahedral mesh
(`v_tet`: (N_v, 3), `f_tet`: (N_t, 4)). NHMO v2's surface-sampling
machinery in `nhmo/geometry/surface.py` expects a trimesh.Trimesh
(triangle mesh). This module bridges the two: for each tet, enumerate
its 4 faces; keep faces that appear in EXACTLY ONE tet (boundary
triangles); group by which vertices they touch.

R_7.1 discipline (Phase 7 plan risk item): the extracted boundary uses
FACE-normals per `surface.py`'s Q4(a) requirement — no vertex-normal
interpolation anywhere. Boundary normals are oriented outward from Ω
via the sign of the tet's fourth vertex (the one opposite the triangle).

Pure numpy; no torch, no trimesh dependency for the core function. A
convenience wrapper `extract_boundary_trimesh()` returns a
trimesh.Trimesh for downstream use.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def extract_tet_boundary(
    v_tet: np.ndarray,           # (N_v, 3)
    f_tet: np.ndarray,           # (N_t, 4) — tet vertex indices
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract the triangle boundary of a tet mesh.

    Algorithm: for each tet, enumerate its 4 faces as sorted-index
    triples (so opposite-orientation duplicates match). Count occurrences.
    Faces with count == 1 are boundary. Orient each boundary triangle's
    winding so its normal points AWAY from the tet's opposite vertex.

    Args:
        v_tet: (N_v, 3) vertex positions
        f_tet: (N_t, 4) int tet indices

    Returns:
        v_boundary:  (N_vb, 3) positions of vertices that appear on any
                     boundary triangle (subset of v_tet)
        f_boundary:  (N_fb, 3) triangle indices INTO v_boundary (0-based
                     in the reduced vertex set)
        bd_v_inds:   (N_vb,) indices of each boundary vertex back into
                     the original v_tet array
    """
    if v_tet.ndim != 2 or v_tet.shape[-1] != 3:
        raise ValueError(f"v_tet must be (N_v, 3); got {v_tet.shape}")
    if f_tet.ndim != 2 or f_tet.shape[-1] != 4:
        raise ValueError(f"f_tet must be (N_t, 4); got {f_tet.shape}")

    N_t = f_tet.shape[0]
    # Four face combinations per tet; the 4th vertex is OPPOSITE each face.
    # Vertex-index ordering for the 4 faces of a tet with vertices (a, b, c, d):
    #   face 0 opposite d : (a, b, c)
    #   face 1 opposite c : (a, b, d)
    #   face 2 opposite b : (a, c, d)
    #   face 3 opposite a : (b, c, d)
    oppose = np.array([3, 2, 1, 0], dtype=np.int64)      # index of the opposite vertex per face
    face_vs = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)

    tri_all = f_tet[:, face_vs].reshape(N_t * 4, 3)            # (N_t*4, 3) unsorted tris
    tri_sorted = np.sort(tri_all, axis=-1)                     # canonical key
    opp_all = f_tet[np.arange(N_t).repeat(4), np.tile(oppose, N_t)]  # (N_t*4,) opposite vertex per face

    # Count occurrences of each sorted-triple key.
    # Use a lexicographic sort + run-length trick for O(N log N) without hash.
    order = np.lexsort(tri_sorted.T)
    tri_sorted_ord = tri_sorted[order]
    opp_ord = opp_all[order]
    tri_unsort_ord = tri_all[order]

    # Equality with the previous row → part of a duplicate run.
    same_as_prev = np.concatenate([
        [False],
        np.all(tri_sorted_ord[1:] == tri_sorted_ord[:-1], axis=-1),
    ])

    # A triangle is on the boundary iff it appears exactly once → not a
    # member of any duplicate run. Implementation: after the lexsort, a
    # run of consecutive identical rows has same_as_prev True for every
    # row except the first in the run. Boundary rows are those where
    # BOTH this row and the next row differ (or it's the last row).
    is_run_start = ~same_as_prev                               # (N_t*4,)
    next_is_same = np.concatenate([same_as_prev[1:], [False]]) # True if next row shares this key
    boundary_mask = is_run_start & ~next_is_same               # unique rows

    bd_tri_raw = tri_unsort_ord[boundary_mask]                 # (N_b, 3) — unsorted (oriented) tris
    bd_opp = opp_ord[boundary_mask]                            # (N_b,) — opposite vertex per bd tri

    # Orient each boundary triangle so its normal points AWAY from bd_opp.
    # Current normal from (a, b, c) is (b - a) × (c - a). Sign check:
    # outward normal has positive dot product with (a - opposite).
    a = v_tet[bd_tri_raw[:, 0]]
    b = v_tet[bd_tri_raw[:, 1]]
    c = v_tet[bd_tri_raw[:, 2]]
    opp = v_tet[bd_opp]
    edge1 = b - a
    edge2 = c - a
    normal = np.cross(edge1, edge2)                            # (N_b, 3)
    outward_dir = a - opp                                       # (N_b, 3) points away from opp
    sign = (normal * outward_dir).sum(axis=-1)                  # (N_b,)
    flip = sign < 0
    # Swap vertex 1 and 2 where flip is True to invert the winding.
    bd_tri_oriented = bd_tri_raw.copy()
    bd_tri_oriented[flip] = bd_tri_raw[flip][:, [0, 2, 1]]

    # Reduce to the boundary vertex set.
    bd_v_inds = np.unique(bd_tri_oriented)                      # sorted
    # Renumber: old index → position in bd_v_inds
    mapping = np.full(v_tet.shape[0], -1, dtype=np.int64)
    mapping[bd_v_inds] = np.arange(bd_v_inds.shape[0])
    f_boundary = mapping[bd_tri_oriented]                       # (N_b, 3)
    v_boundary = v_tet[bd_v_inds]                               # (N_vb, 3)
    return v_boundary, f_boundary, bd_v_inds


def extract_boundary_trimesh(v_tet: np.ndarray, f_tet: np.ndarray):
    """Convenience: return a trimesh.Trimesh of the boundary. Requires trimesh."""
    import trimesh

    v_b, f_b, _ = extract_tet_boundary(v_tet, f_tet)
    return trimesh.Trimesh(vertices=v_b, faces=f_b, process=False)
