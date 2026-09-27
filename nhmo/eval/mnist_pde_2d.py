"""2D MNIST benchmark evaluation (paper Table 1).

Protocol:
  u_h is computed with the frozen kernel at every interior pixel of the native
  256^2 mask, then u_h, mask, h, f and the reference u are bilinearly resampled
  to 128^2; the lift runs at 128^2 and the metric is the relative L2 error over
  the interior pixels (mask = 1) at 128^2, one value per (shape, problem) pair.

Model variants, selected by the checkpoints passed:
  --lift-ckpt omitted            kernel only:  u = u_h
  lift_inputs = mhfu (L17c)      NHMO (kernel + lift):  u = u_h + v(mask, h, f, u_h)
  lift_inputs = msf              source-only lift:  u = u_h + v(mask, sdf, f)
  lift_inputs = msf + --r-ckpt   variant with residual head:  u = u_h + v(mask, sdf, f) + r(mask, h, u_h)

Kernel boundary geometry (`--geometry mask`, default): boundary nodes, normals,
quadrature weights and interior anchors are computed from each shape's own
mask.npy in the grid frame (nhmo.data.mask_geometry), and u_h is evaluated at the
grid-frame pixel coordinates. `--geometry polyline` instead builds the kernel
geometry with `MNISTMultiplyConnectedShapeGenerator` from an MNIST image
(`--mnist-images`), the convention of earlier versions of this code; it does not
coincide with the benchmark mask and is kept only for checkpoints trained with it.

    python -m nhmo.eval.mnist_pde_2d --split test \\
        --kernel-ckpt $CKPT/2d/kernel_2d_mask.pt --lift-ckpt $CKPT/2d/lift_2d_l17c_mask.pt \\
        --out results/table1_nhmo_test.json
"""
import argparse
import json
import os
import re
import statistics as st
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

from nhmo.core.field_lift import FieldLift2D
from nhmo.core.kernel import ShapeLatent
from nhmo.core.kernel_2d import HarmonicMeasureField2D
from nhmo.data import mask_geometry as mg
from nhmo.data.mnist_loader import MNISTDigitDataset, MNISTMultiplyConnectedShapeGenerator

_IDX_RE = re.compile(r'idx(\d+)')
GEN_KWARGS = dict(n_total_samples=512, n_polyline_vertices=200,
                  keep_top_k_digit_contours=4, n_interior_anchors=256)


def load_kernel_2d(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ck['cfg']
    kernel = HarmonicMeasureField2D(encoder_cfg=cfg['encoder'], kernel_head_cfg=cfg['kernel_head'],
                                    normalize=cfg['kernel']['normalize']).to(device)
    kernel.load_state_dict(ck['model_state_dict'])
    kernel.eval()
    return kernel, cfg


def load_field_net(path, device, state_key='lift_state_dict', cfg_key='lift_cfg'):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    c = ck[cfg_key]
    net = FieldLift2D(in_channels=c['in_channels'], base_channels=c['base_channels'],
                      depth=c['depth'], gauge_mode=c['gauge_mode']).to(device)
    net.load_state_dict(ck[state_key])
    net.eval()
    return net, c


def downsample(field, target_res):
    if field.shape[-1] == target_res:
        return field
    return F.interpolate(field.unsqueeze(0).unsqueeze(0).float(), size=(target_res, target_res),
                         mode='bilinear', align_corners=True).squeeze(0).squeeze(0)


def pixel_coords(mask, pixel_conv='legacy'):
    """Model-frame coordinates of the True pixels of an R x R mask.

    legacy: ((j + 0.5) / R * 2 - 1, (i + 0.5) / R * 2 - 1), used with the polyline geometry;
    grid:   (-1 + 2 j / (R - 1), -1 + 2 i / (R - 1)), the frame of the data and of grid_sample
            with align_corners=True (used with --geometry mask).
    """
    R = mask.shape[-1]
    ys, xs = torch.where(mask)
    if pixel_conv == 'grid':
        s = 2.0 / (R - 1)
        return torch.stack([xs.float() * s - 1.0, ys.float() * s - 1.0], dim=-1), ys, xs
    return (torch.stack([(xs.float() + 0.5) / R * 2.0 - 1.0, (ys.float() + 0.5) / R * 2.0 - 1.0], dim=-1),
            ys, xs)


def compute_uh_field(kernel, mask, surf, surf_n, weights, shape_latent, h_grid, normalize_mode,
                     batch=4096, pixel_conv='legacy'):
    """u_h(p) = sum_j w_j K(p, zeta_j) h(zeta_j) at every interior pixel of `mask`."""
    device = mask.device
    R = mask.shape[-1]
    p_all, ys, xs = pixel_coords(mask, pixel_conv)
    if len(ys) == 0:
        return torch.zeros(R, R, device=device)
    h_b = F.grid_sample(h_grid.unsqueeze(0).unsqueeze(0), surf.unsqueeze(0).unsqueeze(0),
                        mode='bilinear', padding_mode='border', align_corners=True).squeeze()
    Q = p_all.shape[0]
    u_h_pixels = torch.empty(Q, device=device)
    with torch.no_grad():
        for s in range(0, Q, batch):
            e = min(Q, s + batch)
            p_b = p_all[s:e]
            n = p_b.shape[0]
            sl_b = ShapeLatent(tokens=shape_latent.tokens.expand(n, -1, -1))
            w_b = weights.unsqueeze(0).expand(n, -1)
            log_K = kernel.log_kernel(p_b, surf.unsqueeze(0).expand(n, -1, -1),
                                      surf_n.unsqueeze(0).expand(n, -1, -1), sl_b)
            if normalize_mode == 'hard':
                K_eff = torch.exp(log_K)
            else:
                logZ = torch.logsumexp(log_K + torch.log(w_b + 1e-30), dim=-1, keepdim=True)
                K_eff = torch.exp(log_K - logZ)
            u_h_pixels[s:e] = (w_b * K_eff * h_b.unsqueeze(0).expand(n, -1)).sum(dim=-1)
    u_h_field = torch.zeros(R, R, device=device)
    u_h_field[ys, xs] = u_h_pixels
    return u_h_field


def compute_keff_pixels(kernel, mask, surf, surf_n, weights, shape_latent, normalize_mode, batch=4096,
                        pixel_conv='legacy'):
    """Per-shape kernel matrix K_eff[q, j] = w_j K(p_q, zeta_j) over the interior pixels of `mask`.

    u_h = K_eff @ h(zeta) then equals `compute_uh_field` up to fp32 rounding, while the kernel is
    evaluated once per shape instead of once per problem.
    """
    device = mask.device
    pts, ys, xs = pixel_coords(mask, pixel_conv)
    K = torch.empty(pts.shape[0], surf.shape[0], device=device)
    with torch.no_grad():
        for s in range(0, pts.shape[0], batch):
            e = min(pts.shape[0], s + batch)
            n = e - s
            log_K = kernel.log_kernel(pts[s:e], surf.unsqueeze(0).expand(n, -1, -1),
                                      surf_n.unsqueeze(0).expand(n, -1, -1),
                                      ShapeLatent(tokens=shape_latent.tokens.expand(n, -1, -1)))
            w_b = weights.unsqueeze(0).expand(n, -1)
            if normalize_mode == 'hard':
                K[s:e] = w_b * torch.exp(log_K)
            else:
                logZ = torch.logsumexp(log_K + torch.log(w_b + 1e-30), dim=-1, keepdim=True)
                K[s:e] = w_b * torch.exp(log_K - logZ)
    return ys, xs, K


def uh_from_keff(ys, xs, K, surf, h_grid, R):
    h_b = F.grid_sample(h_grid.unsqueeze(0).unsqueeze(0), surf.unsqueeze(0).unsqueeze(0),
                        mode='bilinear', padding_mode='border', align_corners=True).reshape(-1)
    u = torch.zeros(R, R, device=K.device)
    u[ys, xs] = K @ h_b
    return u


def stats(v):
    if not v:
        return {}
    s = sorted(v)
    return {'n': len(v), 'mean': sum(v) / len(v), 'median': st.median(v),
            'p95': s[int(0.95 * len(v))], 'max': max(v)}


def main():
    ap = argparse.ArgumentParser(description='2D MNIST benchmark evaluation')
    ap.add_argument('--kernel-ckpt', required=True)
    ap.add_argument('--lift-ckpt', default=None, help='omit for the kernel-only variant')
    ap.add_argument('--r-ckpt', default=None, help='residual head (variant with residual head)')
    ap.add_argument('--data-root', default=os.environ.get('NHMO_DATA_2D', 'data/mnist_pde_2d_paramBC_lf'))
    ap.add_argument('--geometry', choices=['mask', 'polyline'], default='mask',
                    help='kernel geometry: mask (default) = nhmo.data.mask_geometry of each shape mask; '
                         'polyline = MNIST-polyline geometry of earlier versions (legacy checkpoints only)')
    ap.add_argument('--mnist-root', default=os.environ.get('MNIST_ROOT', 'data/mnist'),
                    help='polyline geometry only: directory containing raw/{train,t10k}-images-idx3-ubyte.gz')
    ap.add_argument('--mnist-images', choices=['auto', 'train'], default='auto',
                    help='polyline geometry only: auto = t10k images for test splits (legacy protocol), '
                         'train = the images the benchmark was generated from')
    ap.add_argument('--split', default='test', choices=['test', 'test_ood', 'train'])
    ap.add_argument('--n-shapes', type=int, default=-1)
    ap.add_argument('--input-res', type=int, default=128)
    ap.add_argument('--seed', type=int, default=0,
                    help='seed of the random choices in the evaluation (default 0; -1 = unseeded)')
    ap.add_argument('--uh-batch', type=int, default=4096,
                    help='pixels per kernel batch (4096 needs ~26 GB of GPU memory)')
    ap.add_argument('--no-keff-cache', action='store_true',
                    help='recompute the kernel for every problem (the arithmetic of the original '
                         'evaluator); by default the per-shape kernel matrix is computed once and '
                         'reused, which gives the same u_h up to fp32 rounding and is ~8x faster')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.seed >= 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    kernel, cfg = load_kernel_2d(args.kernel_ckpt, device)
    lift, lift_cfg, vf_inputs = None, None, None
    if args.lift_ckpt:
        lift, lift_cfg = load_field_net(args.lift_ckpt, device)
        vf_inputs = lift_cfg.get('lift_inputs', 'mhfu' if lift_cfg['in_channels'] == 4 else None)
    r_head = None
    if args.r_ckpt:
        assert vf_inputs in ('msf', 'mf'), 'the residual head composes with a source-only lift'
        r_head, _ = load_field_net(args.r_ckpt, device, 'r_state_dict', 'r_cfg')
    variant = ('kernel_only' if lift is None else
               f'{vf_inputs}+r' if r_head is not None else vf_inputs)
    print(f'[2d-eval] variant={variant} split={args.split} geometry={args.geometry} '
          f'mnist_images={args.mnist_images}', flush=True)

    R_in = args.input_res
    split_dir = Path(args.data_root) / args.split
    shape_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    if args.n_shapes > 0:
        shape_dirs = shape_dirs[: args.n_shapes]
    mnist_split = 'train' if (args.mnist_images == 'train' or args.split == 'train') else 'test'
    mnist = (None if args.geometry == 'mask' else
             MNISTDigitDataset(mnist_local_root=Path(args.mnist_root), split=mnist_split, max_digits=None))
    pixel_conv = 'grid' if args.geometry == 'mask' else 'legacy'

    per_pair = []
    t0 = time.time()
    for si, sd in enumerate(shape_dirs):
        idx = int(_IDX_RE.search(sd.name).group(1))
        mask_np = np.load(sd / 'mask.npy').astype(bool)
        if args.geometry == 'mask':
            geo = mg.geometry_to_torch(mg.shape_geometry(mask_np, idx), device)
            surf, surf_n = geo['surface_points'], geo['surface_normals']
            interior_pool, weights = geo['interior_points'], geo['surface_weights']
        else:
            di, label, img = mnist[idx]
            try:
                gen = MNISTMultiplyConnectedShapeGenerator(digit_image=img, digit_index=di, label=label,
                                                           **GEN_KWARGS)
                spec = gen()
            except Exception as e:
                print(f'  skip {sd.name}: {e}')
                continue
            surf = spec['shape_ctx']['surface_points'].to(device).squeeze(0)
            surf_n = spec['shape_ctx']['surface_normals'].to(device).squeeze(0)
            interior_pool = spec['shape_ctx']['interior_points'].to(device).squeeze(0)
            weights = torch.tensor(gen._surface_weights, device=device)
        mask = torch.from_numpy(mask_np).to(device)
        R = mask.shape[0]
        m_np = mask.cpu().numpy()
        sdf_grid = torch.from_numpy((-distance_transform_edt(m_np) + distance_transform_edt(~m_np))
                                    .astype(np.float32) * (2.0 / R)).to(device)
        with torch.no_grad():
            sl = kernel.encode({'surface_points': surf.unsqueeze(0),
                                'surface_normals': surf_n.unsqueeze(0),
                                'interior_points': interior_pool.unsqueeze(0)})
        if not args.no_keff_cache:
            keff = compute_keff_pixels(kernel, mask, surf, surf_n, weights, sl,
                                       cfg['kernel']['normalize'], batch=args.uh_batch,
                                       pixel_conv=pixel_conv)

        for npz in sorted(sd.glob('*.npz')):
            prob_name = npz.stem
            d = np.load(npz)
            h_grid = torch.from_numpy(d['h'].astype(np.float32)).to(device)
            is_poisson = 'poisson_' in prob_name
            f_grid = (torch.from_numpy(d['f'].astype(np.float32)).to(device) if 'f' in d.files
                      else torch.zeros_like(h_grid))
            u_grid = torch.from_numpy(d['u_true'].astype(np.float32)).to(device)

            if args.no_keff_cache:
                u_h = compute_uh_field(kernel, mask, surf, surf_n, weights, sl, h_grid,
                                       cfg['kernel']['normalize'], batch=args.uh_batch,
                                       pixel_conv=pixel_conv)
            else:
                u_h = uh_from_keff(*keff, surf, h_grid, R)
            m_lo = downsample(mask.float(), R_in)
            h_lo, f_lo = downsample(h_grid, R_in), downsample(f_grid, R_in)
            uh_lo, u_lo = downsample(u_h, R_in), downsample(u_grid, R_in)
            m_b = (m_lo > 0.5).unsqueeze(0).float()
            with torch.no_grad():
                u_pred = uh_lo.unsqueeze(0)
                if lift is not None:
                    if vf_inputs == 'mhfu':
                        x_in = torch.stack([m_lo, h_lo, f_lo, uh_lo], dim=0)
                    elif vf_inputs == 'msf':
                        x_in = torch.stack([m_lo, downsample(sdf_grid, R_in), f_lo], dim=0)
                    else:  # 'mf'
                        x_in = torch.stack([m_lo, f_lo], dim=0)
                    u_pred = u_pred + lift(x_in.unsqueeze(0), m_b)
                if r_head is not None:
                    u_pred = u_pred + r_head(torch.stack([m_lo, h_lo, uh_lo], dim=0).unsqueeze(0), m_b)
                u_pred = u_pred * m_b
                u_target = u_lo.unsqueeze(0) * m_b
                mb = m_b.bool().squeeze(0)
                diff = (u_pred[0] - u_target[0])[mb]
                true = u_target[0][mb]
                rl2 = (diff.norm() / (true.norm() + 1e-12)).item()
            if prob_name.startswith('laplace_'):
                fam = prob_name.split('_')[1]
            else:
                parts = prob_name.split('_')
                fam = parts[3] if len(parts) > 3 else parts[2]
            per_pair.append({'shape_id': sd.name, 'problem': prob_name, 'is_poisson': bool(is_poisson),
                             'family': fam, 'rel_l2': rl2})
        if (si + 1) % 10 == 0:
            print(f'[2d-eval] {si + 1}/{len(shape_dirs)} shapes ({time.time() - t0:.0f}s)', flush=True)

    lap = [p['rel_l2'] for p in per_pair if not p['is_poisson']]
    poi = [p['rel_l2'] for p in per_pair if p['is_poisson']]
    agg = {'Laplace': stats(lap), 'Poisson': stats(poi), 'Mixed': stats(lap + poi)}
    by_fam = defaultdict(list)
    for p in per_pair:
        by_fam[(p['is_poisson'], p['family'])].append(p['rel_l2'])
    per_fam = {f'{"Poisson" if k[0] else "Laplace"}_{k[1]}': stats(v) for k, v in by_fam.items()}
    for k, v in agg.items():
        if v:
            print(f'  {k}: n={v["n"]} mean={v["mean"]:.4f} median={v["median"]:.4f} p95={v["p95"]:.4f}')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as fp:
        json.dump({'aggregate': agg, 'per_family': per_fam, 'per_pair': per_pair, 'variant': variant,
                   'kernel_ckpt': args.kernel_ckpt, 'lift_ckpt': args.lift_ckpt, 'r_ckpt': args.r_ckpt,
                   'split': args.split, 'mnist_images': args.mnist_images, 'seed': args.seed,
                   'keff_cache': not args.no_keff_cache, 'geometry': args.geometry,
                   'protocol': 'u_h at 256^2 -> bilinear to 128^2 -> rel-L2 over mask at 128^2'},
                  fp, indent=1)
    print(f'[2d-eval] wrote {args.out}')


if __name__ == '__main__':
    main()
