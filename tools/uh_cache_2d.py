"""Precompute the train-split u_h fields at the lift resolution for a frozen 2D kernel.

<out>/<shape_dir>.pt = {problem name: u_h (R_in x R_in, fp32, CPU)} for all poisson_* and
laplace_* problems of the shape. The kernel matrix is evaluated once per shape (with the chunking
and arithmetic of the trainers' on-the-fly path), so the cached fields are bitwise identical to
what `uh_at_res` computes during training; pass the directory to the lift / residual-head
trainers with --uh-cache-dir. Shards (--shard i --nshards n) can run on several GPUs.
Implemented for the mask-registered geometry (the legacy polyline geometry draws random encoder
anchors per record, so its u_h is not a fixed function of the shape).

    python tools/uh_cache_2d.py --kernel-ckpt $CKPT/2d/kernel_2d_mask.pt \
        --out data/uh_cache_mask_train128
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nhmo.train.poisson_lift_field_train_2d import (  # noqa: E402
    _ShapeRecord, _downsample, _parse_idx, encode_latents, keff_chunks, uh_from_keff_chunks,
)
from nhmo.train.residual_head_train_2d import load_kernel  # noqa: E402


class _LowRes:
    """The fields of a record that keff_chunks reads, with the mask at the lift resolution."""

    def __init__(self, rec, R_in, normalize_mode):
        self.R = R_in
        self.mask = _downsample(rec.mask.float(), R_in) > 0.5
        self.surface_points, self.surface_normals = rec.surface_points, rec.surface_normals
        self.surface_weights, self.shape_latent = rec.surface_weights, rec.shape_latent
        self.pixel_conv, self.normalize_mode = rec.pixel_conv, normalize_mode


def main() -> int:
    pa = argparse.ArgumentParser()
    pa.add_argument('--kernel-ckpt', required=True)
    pa.add_argument('--data-root', default=os.environ.get('NHMO_DATA_2D', 'data/mnist_pde_2d_paramBC_lf'))
    pa.add_argument('--split', default='train')
    pa.add_argument('--input-res', type=int, default=128)
    pa.add_argument('--shard', type=int, default=0)
    pa.add_argument('--nshards', type=int, default=1)
    pa.add_argument('--out', required=True)
    args = pa.parse_args()
    dev = torch.device('cuda')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kernel, kcfg = load_kernel(args.kernel_ckpt, dev)
    shape_dirs = sorted(d for d in (Path(args.data_root) / args.split).iterdir() if d.is_dir())
    mine = shape_dirs[args.shard::args.nshards]
    t0 = time.time()
    n_done = 0
    for sd in mine:
        fn = out / f'{sd.name}.pt'
        if fn.exists():
            continue
        rec = _ShapeRecord(sd, None, _parse_idx(sd.name), -1, {}, dev,
                           problem_glob=('poisson_*.npz', 'laplace_*.npz'), geometry='mask')
        encode_latents(kernel, [rec])
        kc = keff_chunks(kernel, _LowRes(rec, args.input_res, kcfg['kernel']['normalize']))
        d = {name: uh_from_keff_chunks(kc, rec.surface_points, _downsample(h, args.input_res)).cpu()
             for name, h, _, _ in rec.problems}
        torch.save(d, str(fn) + '.tmp')
        os.replace(str(fn) + '.tmp', fn)
        n_done += 1
        if n_done % 20 == 0:
            print(f'[uh-cache] shard {args.shard}: {n_done}/{len(mine)} shapes ({time.time() - t0:.0f}s)', flush=True)
    print(f'[uh-cache] shard {args.shard} done: {n_done} shapes in {time.time() - t0:.0f}s', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
