"""Mask-registered kernel geometry for the 2D MNIST benchmark (default 2D geometry).

Every geometric quantity the 2D kernel sees is derived from the benchmark's own domain mask
(`make_domain_mask`, identical to tools/gen_mnist_pde_data_param_bc.py), in the frame of the
benchmark grid:

  grid frame   pixel (row i, col j) of an R x R benchmark grid sits at
               x = -1 + 2 j / (R - 1),  y = -1 + 2 i / (R - 1)
               (the generator builds h, f with np.linspace(-1, 1, R) and indexing='xy', so
               x = column, y = row, row 0 at y = -1). This is also the frame in which
               F.grid_sample(..., align_corners=True) reads a grid, and the frame of any grid
               bilinearly resampled with align_corners=True (for example 256 -> 128).
  nodes        512 points on the 0.5-level contours of the mask (skimage.find_contours,
               sub-pixel): the outer boundary of the domain and every digit contour. Nodes are
               allocated to contours in proportion to contour length (at least 4 per contour,
               total exactly n_nodes) and placed uniformly in arclength on each closed contour.
  weights      per-node arclength quadrature weight L_c / k_c of its contour c.
  normals      unit normals from the periodic central-difference tangent, oriented out of the
               domain (per contour, by majority vote of an SDF test).
  SDF          signed distance to the 0.5-level contour from the Euclidean distance transform of
               the mask, domain positive: (edt_in - 0.5) px inside, -(edt_out - 0.5) px outside,
               converted to model units (x 2/(R-1)) or WoS units (x 1/(R-1)).
  interior     points drawn uniformly in [-1, 1]^2 and kept if the bilinear SDF is at least
               `min_depth` model units (default 0.04 = 5 px at 256^2); used as KDE queries and as
               the encoder's interior anchors.

`shape_geometry` is used for the kernel training targets (tools/gen_mask_corpus.py), for the u_h
fields used to train the lifts, and by the evaluator, so the kernel geometry always coincides with
the mask of the shape. (Earlier versions read the kernel geometry from
`MNISTMultiplyConnectedShapeGenerator`, `--geometry polyline`, which does not coincide with the mask.)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as _edt
from scipy.ndimage import zoom
from skimage import measure

MIN_DEPTH_DEFAULT = 0.04   # model units


def make_domain_mask(digit_28: np.ndarray, R: int = 256, threshold: float = 0.5) -> np.ndarray:
    """Benchmark domain mask (identical to tools/gen_mnist_pde_data_param_bc.make_domain_mask)."""
    img = np.asarray(digit_28).astype(np.float32) / 255.0
    img = zoom(img, R / 28.0, order=1)
    img = img[:R, :R]
    if img.shape != (R, R):
        out = np.zeros((R, R), dtype=np.float32)
        out[: img.shape[0], : img.shape[1]] = img
        img = out
    domain = ~(img > threshold)
    domain[0, :] = False
    domain[-1, :] = False
    domain[:, 0] = False
    domain[:, -1] = False
    return domain


def pixel_points(mask_bool: torch.Tensor):
    """Model-frame coordinates of the True pixels of an R x R mask (grid frame)."""
    R = mask_bool.shape[-1]
    ys, xs = torch.where(mask_bool)
    s = 2.0 / (R - 1)
    p = torch.stack([xs.float() * s - 1.0, ys.float() * s - 1.0], dim=-1)
    return p, ys, xs


def mask_sdf_px(mask: np.ndarray) -> np.ndarray:
    """Signed distance (pixels) to the 0.5-level contour, domain positive."""
    m = mask.astype(bool)
    d_in = _edt(m)
    d_out = _edt(~m)
    return np.where(m, d_in - 0.5, -(d_out - 0.5)).astype(np.float32)


def mask_sdf_model(mask: np.ndarray) -> np.ndarray:
    R = mask.shape[0]
    return mask_sdf_px(mask) * np.float32(2.0 / (R - 1))


def mask_sdf_wos(mask: np.ndarray) -> np.ndarray:
    R = mask.shape[0]
    return mask_sdf_px(mask) * np.float32(1.0 / (R - 1))


def sample_grid(grid: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Bilinear lookup of an (R, R) grid at model-frame points (N, 2), grid frame."""
    g = grid[None, None].float()
    out = F.grid_sample(g, pts.view(1, 1, -1, 2).float(), mode='bilinear',
                        padding_mode='border', align_corners=True)
    return out.view(-1)


def _allocate(lengths, n_nodes, k_min=4):
    L = np.asarray(lengths, dtype=np.float64)
    ideal = n_nodes * L / L.sum()
    k = np.maximum(k_min, np.floor(ideal)).astype(int)
    rem = n_nodes - int(k.sum())
    if rem > 0:
        order = np.argsort(-(ideal - k))
        for i in range(rem):
            k[order[i % len(k)]] += 1
    while rem < 0:
        i = int(np.argmax(k))
        if k[i] <= k_min:
            raise ValueError('too many contours for n_nodes')
        k[i] -= 1
        rem += 1
    assert int(k.sum()) == n_nodes
    return k


def contour_nodes(mask: np.ndarray, n_nodes: int = 512, sdf_model: np.ndarray | None = None):
    """Nodes, outward normals, weights (numpy, model frame) and diagnostics."""
    R = mask.shape[0]
    s = 2.0 / (R - 1)
    if sdf_model is None:
        sdf_model = mask_sdf_model(mask)
    cs = measure.find_contours(mask.astype(np.float64), 0.5)
    loops = []
    for c in cs:
        xy = np.stack([c[:, 1], c[:, 0]], axis=1) * s - 1.0      # x = col, y = row
        if np.linalg.norm(xy[0] - xy[-1]) > 1e-9:                  # close it (not expected)
            xy = np.concatenate([xy, xy[:1]], axis=0)
        if len(xy) < 4:
            continue
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        if seg.sum() <= 0:
            continue
        loops.append((xy, seg))
    lengths = [seg.sum() for _, seg in loops]
    ks = _allocate(lengths, n_nodes)
    sdf_t = torch.from_numpy(sdf_model)
    pts_all, nrm_all, w_all, comp, agree = [], [], [], [], []
    for ci, ((xy, seg), k, Lc) in enumerate(zip(loops, ks, lengths)):
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        t = np.arange(k) * (Lc / k)
        px = np.interp(t, cum, xy[:, 0])
        py = np.interp(t, cum, xy[:, 1])
        P = np.stack([px, py], axis=1)
        T = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)          # periodic tangent
        N = np.stack([T[:, 1], -T[:, 0]], axis=1)
        N /= (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)
        d = 1.0 * s                                                  # one pixel
        sp = sample_grid(sdf_t, torch.from_numpy((P + d * N).astype(np.float32))).numpy()
        sm = sample_grid(sdf_t, torch.from_numpy((P - d * N).astype(np.float32))).numpy()
        outward = sp < sm                                            # SDF decreases outward
        frac = float(outward.mean())
        if frac < 0.5:
            N = -N
            frac = 1.0 - frac
        agree.append(frac)
        pts_all.append(P)
        nrm_all.append(N)
        w_all.append(np.full(k, Lc / k))
        comp.append(np.full(k, ci))
    pts = np.concatenate(pts_all).astype(np.float32)
    nrm = np.concatenate(nrm_all).astype(np.float32)
    w = np.concatenate(w_all).astype(np.float32)
    info = {'n_contours': len(loops), 'contour_lengths': [float(x) for x in lengths],
            'nodes_per_contour': [int(x) for x in ks], 'perimeter': float(sum(lengths)),
            'normal_vote_agreement_min': float(min(agree)), 'component': np.concatenate(comp)}
    return pts, nrm, w, info


def interior_points(sdf_model: np.ndarray, n: int, rng: np.random.RandomState,
                    min_depth: float = MIN_DEPTH_DEFAULT) -> np.ndarray:
    sdf_t = torch.from_numpy(sdf_model)
    acc = []
    tries = 0
    while sum(len(a) for a in acc) < n:
        cand = rng.uniform(-1.0, 1.0, (8192, 2)).astype(np.float32)
        v = sample_grid(sdf_t, torch.from_numpy(cand)).numpy()
        keep = cand[v >= min_depth]
        if len(keep):
            acc.append(keep)
        tries += 1
        if tries > 1000:
            raise RuntimeError('interior sampler exhausted')
    return np.concatenate(acc)[:n].astype(np.float32)


def anchor_seed(mnist_idx: int) -> int:
    return (1_000_003 * int(mnist_idx) + 17) % (2 ** 32)


def shape_geometry(mask: np.ndarray, mnist_idx: int, n_nodes: int = 512, n_anchors: int = 256,
                   min_depth: float = MIN_DEPTH_DEFAULT) -> dict:
    """Kernel geometry of one benchmark shape. Deterministic given (mask, mnist_idx)."""
    mask = np.asarray(mask).astype(bool)
    sdf_m = mask_sdf_model(mask)
    pts, nrm, w, info = contour_nodes(mask, n_nodes, sdf_m)
    anchors = interior_points(sdf_m, n_anchors, np.random.RandomState(anchor_seed(mnist_idx)),
                              min_depth)
    return {'surface_points': pts, 'surface_normals': nrm, 'surface_weights': w,
            'interior_points': anchors, 'sdf_model': sdf_m, 'info': info}


def geometry_to_torch(geo: dict, device) -> dict:
    return {k: torch.from_numpy(geo[k]).to(device)
            for k in ('surface_points', 'surface_normals', 'surface_weights', 'interior_points')}
