"""Legacy (polyline geometry) KDE targets; the released 2D kernel uses tools/gen_mask_corpus.py.

Precompute omega_gt for 2D MNIST multiply-connected domains (v1-aligned).

For each shape:
  1. Build the same MNISTMultiplyConnectedShapeGenerator the kernel will see at training time
  2. Sample K_QUERIES random interior anchor points
  3. For each anchor, run N_WALKS WoS exit simulations
  4. Aggregate exits over the N_SURFACE boundary samples via Gaussian KDE
     with bandwidth σ in arclength units.
  5. Normalize so omega_gt[shape, query, :] sums to 1.

Two shape sources:
  --data-root DIR --split S     shapes of a generated PDE split (dir names encode the MNIST index)
  --mnist-start A --mnist-end B MNIST training images A..B-1 directly, shape ids raw_<i>_<label>.
                                The polyline-geometry kernel of earlier versions (K3) was trained
                                on raw_0 .. raw_4999
                                (walks 1e4, KDE sigma 0.2% of the perimeter), built in three
                                shards [0,2000), [2000,3500), [3500,5000) and merged with
                                tools/merge_omega_gt.py.

Output: a single .pt with stacked tensors:
    {
      'shape_ids':  (N_shapes,) — strings (digit indices)
      'surface_points':  (N_shapes, N_surface, 2) in model frame
      'surface_normals': (N_shapes, N_surface, 2)
      'surface_weights': (N_shapes, N_surface)
      'sdf_grids':       (N_shapes, G, G) in [0,1]² WoS frame
      'queries':         (N_shapes, K_QUERIES, 2) in model frame
      'omega_gt':        (N_shapes, K_QUERIES, N_surface) — normalized density
      'config': dict
    }
"""
from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nhmo.data.mnist_loader import MNISTDigitDataset, MNISTMultiplyConnectedShapeGenerator
from nhmo.data.synthetic_2d import model_frame_to_wos_frame, wos_frame_to_model_frame
from nhmo.geometry.circle_mc import WoSHitSampler2D


_IDX_RE = re.compile(r'idx(\d+)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=str, default=None)
    ap.add_argument('--split', type=str, default='train', choices=['train', 'test', 'test_ood'])
    ap.add_argument('--mnist-root', type=str, default=os.environ.get('MNIST_ROOT', 'data/mnist'),
                    help='directory containing raw/ MNIST idx files')
    ap.add_argument('--mnist-start', type=int, default=None)
    ap.add_argument('--mnist-end', type=int, default=None)
    ap.add_argument('--n-shapes', type=int, default=-1, help='-1 = all')
    ap.add_argument('--n-queries', type=int, default=32)
    ap.add_argument('--n-walks', type=int, default=10000)
    ap.add_argument('--n-surface', type=int, default=512)
    ap.add_argument('--grid-size', type=int, default=64)
    ap.add_argument('--n-interior-anchors', type=int, default=256)
    ap.add_argument('--kde-sigma-frac', type=float, default=0.02,
                    help='KDE bandwidth as fraction of total perimeter')
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--device', type=str, default='cuda')
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Iterate shapes: (shape_id, MNIST index) pairs
    if args.mnist_start is not None:
        mnist = MNISTDigitDataset(mnist_local_root=args.mnist_root, split='train', max_digits=None)
        items = [(None, i) for i in range(args.mnist_start, args.mnist_end)]
    else:
        split_dir = Path(args.data_root) / args.split
        shape_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        if args.n_shapes > 0:
            shape_dirs = shape_dirs[: args.n_shapes]
        mnist = MNISTDigitDataset(
            mnist_local_root=args.mnist_root,
            split='train' if args.split == 'train' else 'test',
            max_digits=None,
        )
        items = [(sd.name, int(_IDX_RE.search(sd.name).group(1))) for sd in shape_dirs]
    print(f'[precompute] {len(items)} shapes', flush=True)

    sampler = WoSHitSampler2D(epsilon=1e-3, max_steps=128)

    out = {
        'shape_ids': [],
        'surface_points': [],
        'surface_normals': [],
        'surface_weights': [],
        'sdf_grids': [],
        'queries': [],
        'omega_gt': [],
    }

    rng = np.random.RandomState(42)
    t_start = time.time()
    skipped = 0

    for sx, (name, idx) in enumerate(items):
        try:
            di, label, img = mnist[idx]
            shape_id = name if name is not None else f'raw_{idx}_{label}'
            gen = MNISTMultiplyConnectedShapeGenerator(
                digit_image=img, digit_index=di, label=label,
                grid_size=args.grid_size,
                n_total_samples=args.n_surface,
                n_interior_anchors=args.n_interior_anchors,
                device=str(device),
                n_polyline_vertices=200,
                keep_top_k_digit_contours=4,
            )
            spec = gen()
            surf = spec['shape_ctx']['surface_points'].squeeze(0).to(device)              # (N_s, 2)
            surf_n = spec['shape_ctx']['surface_normals'].squeeze(0).to(device)
            interior_pool = spec['shape_ctx']['interior_points'].squeeze(0).to(device)
            surface_weights = torch.tensor(gen._surface_weights, device=device)
            sdf_grid = spec['sdf_grid_wos'].to(device)                                     # (G, G)

            # Get N_s for THIS shape (proportional sampling can vary slightly per shape)
            N_s = surf.shape[0]

            # Pad/truncate surface buffers to args.n_surface for uniform stacking.
            if N_s != args.n_surface:
                # Simple resample: linear interp index resampling
                idx_old = np.arange(N_s)
                idx_new = np.linspace(0, N_s - 1, args.n_surface)
                surf_np = surf.cpu().numpy()
                surf_n_np = surf_n.cpu().numpy()
                w_np = surface_weights.cpu().numpy()
                surf_resamp = np.array([np.interp(idx_new, idx_old, surf_np[:, d]) for d in (0, 1)]).T
                surf_n_resamp = np.array([np.interp(idx_new, idx_old, surf_n_np[:, d]) for d in (0, 1)]).T
                w_resamp = np.interp(idx_new, idx_old, w_np)
                # Renormalize w to keep total perim constant
                w_resamp = w_resamp * w_np.sum() / w_resamp.sum()
                surf = torch.from_numpy(surf_resamp.astype(np.float32)).to(device)
                surf_n = torch.from_numpy(surf_n_resamp.astype(np.float32)).to(device)
                surface_weights = torch.from_numpy(w_resamp.astype(np.float32)).to(device)
                N_s = args.n_surface

            # Pick K random query points from interior pool
            K = min(args.n_queries, interior_pool.shape[0])
            qi = torch.from_numpy(rng.choice(interior_pool.shape[0], K, replace=False)).to(device)
            queries_model = interior_pool[qi]                                              # (K, 2)
            queries_wos = model_frame_to_wos_frame(queries_model)                           # (K, 2)

            # Run n_walks per query
            sdf_batched = sdf_grid.unsqueeze(0).expand(K, -1, -1).contiguous()
            hits_wos, hit_mask = sampler.sample_hits(
                queries_wos, sdf_batched, args.n_walks, seed=int(rng.randint(1, 2**30)),
            )
            # hits_wos: (K * n_walks, 2)
            hits_model = wos_frame_to_model_frame(hits_wos)
            hits_model = hits_model.reshape(K, args.n_walks, 2)
            hit_mask = hit_mask.reshape(K, args.n_walks).bool()

            # KDE: for each (query, surface_point), sum Gaussian contributions from valid hits
            # We approximate via L2 distance between hit and surface points (in model frame).
            # KDE: omega_gt[k, i] ∝ Σ_w exp(-||hit_w - surf_i||² / (2σ²))
            total_perim = float(surface_weights.sum().item())
            sigma = args.kde_sigma_frac * total_perim
            sigma2 = sigma * sigma

            omega_gt_shape = torch.zeros((K, N_s), device=device, dtype=torch.float32)

            # Memory-careful: process each query's hits separately
            for k in range(K):
                hits_k = hits_model[k][hit_mask[k]]          # (W_valid, 2)
                if hits_k.shape[0] == 0:
                    continue
                # diffs: (W_valid, N_s)
                d2 = ((hits_k.unsqueeze(1) - surf.unsqueeze(0)) ** 2).sum(-1)
                kde = torch.exp(-d2 / (2.0 * sigma2))         # (W_valid, N_s)
                # Weight each hit's mass by its arclength element at the surface, normalize
                # We want a density: omega(ζ_i) ∝ Σ_w kernel(hit_w, ζ_i) / (n_walks * sigma)
                # Then normalize over surface (sum = 1 in discrete weighted sum).
                density = kde.sum(0) / max(1, hits_k.shape[0])           # (N_s,)
                Z = float((density * surface_weights).sum().item()) + 1e-12
                omega_gt_shape[k] = density / Z

            # Stack into output
            out['shape_ids'].append(shape_id)
            out['surface_points'].append(surf.cpu())
            out['surface_normals'].append(surf_n.cpu())
            out['surface_weights'].append(surface_weights.cpu())
            out['sdf_grids'].append(sdf_grid.cpu())
            out['queries'].append(queries_model.cpu())
            out['omega_gt'].append(omega_gt_shape.cpu())

            if (sx + 1) % 10 == 0 or sx == len(items) - 1:
                el = time.time() - t_start
                print(f'[precompute] {sx + 1}/{len(items)} elapsed={el:.0f}s '
                      f'omega_gt range=[{omega_gt_shape.min().item():.3f}, '
                      f'{omega_gt_shape.max().item():.3f}]', flush=True)
        except Exception as e:
            skipped += 1
            print(f'[precompute] skip {idx}: {type(e).__name__} {e!s:.100}', flush=True)

    print(f'\n[precompute] DONE: {len(out["shape_ids"])} shapes  skipped={skipped}  '
          f'elapsed={time.time()-t_start:.0f}s', flush=True)

    # Stack tensors
    out_t = {
        'shape_ids': out['shape_ids'],
        'surface_points': torch.stack(out['surface_points']),
        'surface_normals': torch.stack(out['surface_normals']),
        'surface_weights': torch.stack(out['surface_weights']),
        'sdf_grids': torch.stack(out['sdf_grids']),
        'queries': torch.stack(out['queries']),
        'omega_gt': torch.stack(out['omega_gt']),
        'config': {
            'split': args.split, 'n_walks': args.n_walks,
            'mnist_range': [args.mnist_start, args.mnist_end],
            'n_surface': args.n_surface, 'n_queries': args.n_queries,
            'grid_size': args.grid_size, 'kde_sigma_frac': args.kde_sigma_frac,
        },
    }
    torch.save(out_t, out_p)
    print(f'[precompute] wrote {out_p}', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
