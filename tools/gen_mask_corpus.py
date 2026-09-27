"""Kernel training targets for the released 2D kernel (mask-registered geometry).

Corpus definition: the first 5000 MNIST *training* images (train-images-idx3-ubyte, index order)
after removing every MNIST index used by a test or test_ood shape of the 2D benchmark (the idx<N>
in the shape directory names). Benchmark train-split shapes stay in the corpus. The released list
is configs/mnist_kernel_corpus_mask.json (`--write-index-list` regenerates it).

Per shape (geometry conventions of nhmo.data.mask_geometry, grid frame):
  mask        make_domain_mask(train image, 256) (the benchmark generator's function)
  nodes       512 mask-contour nodes, outward normals, arclength weights
  queries     32 interior points (depth >= 0.04 model units), RandomState((1_000_003 * idx + 29) mod 2^32)
  walks       10^4 WoS walks per query (WoSHitSampler2D, epsilon 1e-3, 128-step cap, WoS frame),
              on the 256^2 mask SDF in WoS units; Warp seed (1_000_003 * idx + 555) mod 2^30
  target      Gaussian KDE of the valid exits at the nodes, sigma = 0.002 x total boundary length,
              normalized so that sum_i w_i omega_i = 1.
Output schema = the K3 training targets (consumed unchanged by nhmo.train.mnist_kde_main).

    python tools/gen_mask_corpus.py --write-index-list configs/mnist_kernel_corpus_mask.json
    python tools/gen_mask_corpus.py --index-list configs/mnist_kernel_corpus_mask.json \
        --shard-start 0 --shard-end 1000 --out data/omega_mask_shard0.pt     # ... 5 shards
    python tools/merge_omega_gt.py data/omega_mask_5k.pt data/omega_mask_shard{0,1,2,3,4}.pt
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nhmo.data import mask_geometry as mg  # noqa: E402
from nhmo.data.mnist_loader import MNISTDigitDataset  # noqa: E402


def query_seed(idx):
    return (1_000_003 * int(idx) + 29) % (2 ** 32)


def walk_seed(idx):
    return (1_000_003 * int(idx) + 555) % (2 ** 30)


def index_list(data_root: Path, n: int = 5000) -> dict:
    excl = {}
    for split in ('test', 'test_ood'):
        for d in sorted((data_root / split).iterdir()):
            if d.is_dir():
                excl.setdefault(int(re.search(r'idx(\d+)', d.name).group(1)), []).append(f'{split}/{d.name}')
    train_idx = sorted(int(re.search(r'idx(\d+)', d.name).group(1))
                       for d in (data_root / 'train').iterdir() if d.is_dir())
    keep, i = [], 0
    while len(keep) < n:
        if i not in excl:
            keep.append(i)
        i += 1
    return {'definition': 'first 5000 MNIST training-image indices (train-images-idx3-ubyte order) '
                          'after removing every index used by a test or test_ood shape of '
                          'mnist_pde_2d_paramBC_lf',
            'n': len(keep), 'last_index': keep[-1], 'excluded': sorted(excl),
            'excluded_shapes': {str(k): v for k, v in sorted(excl.items())},
            'n_excluded': len(excl), 'n_benchmark_train_shapes_in_corpus': len(set(keep) & set(train_idx)),
            'indices': keep}


def main() -> int:
    pa = argparse.ArgumentParser()
    pa.add_argument('--data-root', default=os.environ.get('NHMO_DATA_2D', 'data/mnist_pde_2d_paramBC_lf'))
    pa.add_argument('--mnist-root', default=os.environ.get('MNIST_ROOT', 'data/mnist'))
    pa.add_argument('--write-index-list', default=None)
    pa.add_argument('--index-list', default=None)
    pa.add_argument('--shard-start', type=int, default=0)
    pa.add_argument('--shard-end', type=int, default=-1)
    pa.add_argument('--n-walks', type=int, default=10000)
    pa.add_argument('--n-queries', type=int, default=32)
    pa.add_argument('--n-nodes', type=int, default=512)
    pa.add_argument('--kde-sigma-frac', type=float, default=0.002)
    pa.add_argument('--resolution', type=int, default=256)
    pa.add_argument('--out', default=None)
    args = pa.parse_args()
    if args.write_index_list:
        L = index_list(Path(args.data_root))
        json.dump(L, open(args.write_index_list, 'w'), indent=1)
        print({k: v for k, v in L.items() if k not in ('indices', 'excluded', 'excluded_shapes')})
        return 0

    from nhmo.geometry.circle_mc import WoSHitSampler2D
    dev = torch.device('cuda')
    L = json.load(open(args.index_list))
    ids = L['indices']
    end = len(ids) if args.shard_end < 0 else args.shard_end
    ids = ids[args.shard_start:end]
    mn = MNISTDigitDataset(mnist_local_root=Path(args.mnist_root), split='train', max_digits=None)
    sampler = WoSHitSampler2D(epsilon=1e-3, max_steps=128)
    out = {k: [] for k in ('shape_ids', 'surface_points', 'surface_normals', 'surface_weights',
                           'sdf_grids', 'queries', 'omega_gt', 'mnist_indices', 'wos_masked_fraction',
                           'n_contours', 'perimeter')}
    t0 = time.time()
    for sx, idx in enumerate(ids):
        _, label, img = mn[idx]
        mask = mg.make_domain_mask(img, args.resolution)
        sdf_m = mg.mask_sdf_model(mask)
        pts, nrm, w, info = mg.contour_nodes(mask, args.n_nodes, sdf_m)
        q = mg.interior_points(sdf_m, args.n_queries, np.random.RandomState(query_seed(idx)))
        sdf_w = torch.from_numpy(mg.mask_sdf_wos(mask)).to(dev)
        K = q.shape[0]
        q_wos = (torch.from_numpy(q).to(dev) + 1.0) * 0.5
        hits_wos, hmask = sampler.sample_hits(q_wos, sdf_w.unsqueeze(0).expand(K, -1, -1).contiguous(),
                                              args.n_walks, seed=walk_seed(idx))
        hits = hits_wos * 2.0 - 1.0
        surf = torch.from_numpy(pts).to(dev)
        w_t = torch.from_numpy(w).to(dev)
        perim = float(info['perimeter'])
        sigma = args.kde_sigma_frac * perim
        om = torch.zeros(K, args.n_nodes, device=dev)
        for k in range(K):
            hk = hits[k][hmask[k]]
            if hk.shape[0] == 0:
                continue
            d2 = ((hk.unsqueeze(1) - surf.unsqueeze(0)) ** 2).sum(-1)
            dens = torch.exp(-d2 / (2.0 * sigma * sigma)).sum(0) / hk.shape[0]
            om[k] = dens / ((dens * w_t).sum() + 1e-12)
        out['shape_ids'].append(f'raw_{idx}_{label}')
        out['surface_points'].append(torch.from_numpy(pts))
        out['surface_normals'].append(torch.from_numpy(nrm))
        out['surface_weights'].append(torch.from_numpy(w))
        out['sdf_grids'].append(F.interpolate(sdf_w[None, None], size=(64, 64), mode='bilinear',
                                              align_corners=True)[0, 0].cpu())
        out['queries'].append(torch.from_numpy(q))
        out['omega_gt'].append(om.cpu())
        out['mnist_indices'].append(int(idx))
        out['wos_masked_fraction'].append(float(1.0 - hmask.float().mean().item()))
        out['n_contours'].append(int(info['n_contours']))
        out['perimeter'].append(perim)
        if (sx + 1) % 100 == 0:
            print(f'[corpus] {args.shard_start}+{sx + 1}/{len(ids)} ({time.time() - t0:.0f}s)', flush=True)
    res = {k: (torch.stack(v) if isinstance(v[0], torch.Tensor) else v) for k, v in out.items()}
    res['config'] = {'geometry': 'nhmo.data.mask_geometry (make_domain_mask 256^2, 0.5-level contours, grid frame)',
                     'n_walks': args.n_walks, 'n_surface': args.n_nodes, 'n_queries': args.n_queries,
                     'kde_sigma_frac': args.kde_sigma_frac,
                     'wos': {'epsilon': 1e-3, 'max_steps': 128, 'frame': 'wos [0,1]^2', 'sdf_res': args.resolution},
                     'query_min_depth_model': mg.MIN_DEPTH_DEFAULT,
                     'seeds': 'queries RandomState((1000003*idx+29) mod 2^32); walks (1000003*idx+555) mod 2^30',
                     'shard': [args.shard_start, end], 'index_list': args.index_list}
    torch.save(res, args.out)
    print(f'[corpus] wrote {args.out} ({len(ids)} shapes, {time.time() - t0:.0f}s)', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
