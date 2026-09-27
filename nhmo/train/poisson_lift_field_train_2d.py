"""Field-to-field 2D Poisson lift trainer (frozen 2D kernel, U-Net lift).

    u_pred = u_h + v_phi(inputs),   u_h(p) = <h, K_theta(p, .; Omega)> (frozen kernel)

Loss: masked pixel-wise MSE at --input-res (128^2), optionally y-normalized.
u_h is computed on the fly at --input-res and memoized per problem (it is a
deterministic function of the problem).

--lift-inputs selects what v_phi sees:
  mhfu : (mask, h, f, u_h), the NHMO model of Table 1 (L17c recipe;
         trained on poisson_* problems only)
  msf  : (mask, sdf, f), the source-only lift of the variant with residual head
         (trained with --include-laplace)
  mf   : (mask, f), ablation

--geometry mask (default) uses the mask-registered kernel geometry of nhmo.data.mask_geometry and
grid-frame u_h pixel coordinates; `polyline` is the MNIST-polyline geometry of earlier versions.
--uh-cache-dir reads u_h fields precomputed by tools/uh_cache_2d.py (bitwise identical to the
on-the-fly values, one kernel evaluation per shape instead of per problem). --seed fixes the lift
initialization, the data order and the y-norm subsample.

Recipes of the released checkpoints (all with --uh-cache-dir <cache> --seed 0, see README):
  NHMO     (lift_2d_l17c_mask.pt): --lift-inputs mhfu --base-channels 48 --depth 4 --total-steps 10000 --use-y-norm
  source   (lift_2d_msf_mask.pt) : --lift-inputs msf --include-laplace --base-channels 64 --depth 4 --total-steps 30000 --use-y-norm
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

from nhmo.core.kernel_2d import HarmonicMeasureField2D
from nhmo.core.kernel import ShapeLatent
from nhmo.core.field_lift import FieldLift2D
from nhmo.data import mask_geometry as mg
from nhmo.data.mnist_loader import (
    MNISTDigitDataset,
    MNISTMultiplyConnectedShapeGenerator,
)

_IDX_RE = re.compile(r'idx(\d+)')


def _parse_idx(s):
    m = _IDX_RE.search(s)
    if m is None: raise ValueError(f"bad: {s}")
    return int(m.group(1))


class _ShapeRecord:
    def __init__(self, shape_dir, digit_image, digit_idx, label, gen_kwargs, device,
                 problem_glob=('poisson_*.npz',), geometry='mask'):
        # Default matches the ORIGINAL trainer (poisson_* only; the canonical
        # L17c lift never saw the laplace_* files). Evaluation/analysis code
        # passes ('poisson_*.npz', 'laplace_*.npz') to load the full 408-pair
        # mixed set the paper metric uses.
        self.shape_dir = shape_dir
        self.geometry = geometry
        self.pixel_conv = 'grid' if geometry == 'mask' else 'legacy'
        mask = np.load(shape_dir / 'mask.npy').astype(bool)
        if geometry == 'mask':
            geo = mg.geometry_to_torch(mg.shape_geometry(mask, _parse_idx(shape_dir.name)), device)
            self.surface_points = geo['surface_points']
            self.surface_normals = geo['surface_normals']
            self.interior_pool = geo['interior_points']
            self.surface_weights = geo['surface_weights']
        else:
            self.gen = MNISTMultiplyConnectedShapeGenerator(
                digit_image=digit_image, digit_index=digit_idx, label=label, **gen_kwargs,
            )
            spec = self.gen()
            self.surface_points = spec['shape_ctx']['surface_points'].to(device).squeeze(0)
            self.surface_normals = spec['shape_ctx']['surface_normals'].to(device).squeeze(0)
            self.interior_pool = spec['shape_ctx']['interior_points'].to(device).squeeze(0)
            self.surface_weights = torch.tensor(self.gen._surface_weights, device=device)
        self.R = mask.shape[0]
        self.mask = torch.from_numpy(mask).to(device)
        sdf_pos = distance_transform_edt(mask)
        sdf_neg = distance_transform_edt(~mask)
        sdf_grid = (-sdf_pos + sdf_neg).astype(np.float32) * (2.0 / self.R)
        self.sdf_grid = torch.from_numpy(sdf_grid).to(device)
        self.shape_latent = None
        self.problems = []
        paths = []
        for pat in problem_glob:
            paths += list(shape_dir.glob(pat))
        for npz in sorted(paths):
            d = np.load(npz)
            h = torch.from_numpy(d['h'].astype(np.float32)).to(device)
            # laplace_*.npz files carry no 'f' key (f == 0 implicitly)
            if 'f' in d.files:
                f = torch.from_numpy(d['f'].astype(np.float32)).to(device)
            else:
                f = torch.zeros_like(h)
            u = torch.from_numpy(d['u_true'].astype(np.float32)).to(device)
            self.problems.append((npz.stem, h, f, u))
        self.uh_memo = {}


def _build_records(data_root, split, n_shapes, device, gen_kwargs,
                   problem_glob=('poisson_*.npz',), geometry='mask'):
    split_dir = Path(data_root) / split
    shape_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])[:n_shapes]
    mnist = None
    if geometry == 'polyline':
        mnist = MNISTDigitDataset(
            mnist_local_root=Path(os.environ.get('MNIST_ROOT', 'data/mnist')),
            split='train' if split == 'train' else 'test',
            max_digits=None,
        )
    print(f'[abl] loading {len(shape_dirs)} shapes (split={split})', flush=True)
    records = []
    t0 = time.time()
    for sd in shape_dirs:
        idx = _parse_idx(sd.name)
        di, label, img = mnist[idx] if mnist is not None else (idx, -1, None)
        try:
            rec = _ShapeRecord(sd, img, di, label, gen_kwargs, device,
                               problem_glob=problem_glob, geometry=geometry)
            records.append(rec)
        except Exception as e:
            print(f'  skip {sd.name}: {e}', flush=True)
        if (len(records) + 1) % 200 == 0:
            print(f'  built {len(records)+1}/{len(shape_dirs)} elapsed={int(time.time()-t0)}s', flush=True)
    print(f'[abl] built {len(records)} records in {int(time.time()-t0)}s', flush=True)
    return records


def _pixel_points(mask_bool, pixel_conv):
    R = mask_bool.shape[-1]
    ys, xs = torch.where(mask_bool)
    if pixel_conv == 'grid':
        s = 2.0 / (R - 1)
        return torch.stack([xs.float() * s - 1.0, ys.float() * s - 1.0], dim=-1), ys, xs
    return (torch.stack([(xs.float() + 0.5) / R * 2.0 - 1.0, (ys.float() + 0.5) / R * 2.0 - 1.0], dim=-1),
            ys, xs)


def keff_chunks(kernel, rec, BS=4096):
    """Per-chunk K_eff for all interior pixels of rec.mask (the h-independent part of u_h)."""
    R = rec.R
    p_all, ys, xs = _pixel_points(rec.mask, getattr(rec, 'pixel_conv', 'legacy'))
    chunks = []
    Q = p_all.shape[0]
    surf = rec.surface_points
    with torch.no_grad():
        for s in range(0, Q, BS):
            e = min(Q, s + BS)
            p_b = p_all[s:e]
            n = p_b.shape[0]
            sl_b = ShapeLatent(tokens=rec.shape_latent.tokens.expand(n, -1, -1))
            surf_b = surf.unsqueeze(0).expand(n, -1, -1)
            surf_n_b = rec.surface_normals.unsqueeze(0).expand(n, -1, -1)
            w_b = rec.surface_weights.unsqueeze(0).expand(n, -1)
            log_K = kernel.log_kernel(p_b, surf_b, surf_n_b, sl_b)
            if rec.normalize_mode == 'hard':
                K_eff = torch.exp(log_K)
            else:
                logZ = torch.logsumexp(log_K + torch.log(w_b + 1e-30), dim=-1, keepdim=True)
                K_eff = torch.exp(log_K - logZ)
            chunks.append((s, e, K_eff, w_b))
    return R, ys, xs, chunks


def uh_from_keff_chunks(kc, surf, h_grid):
    """u_h field from `keff_chunks`; same elementwise expression as the original trainer."""
    R, ys, xs, chunks = kc
    device = h_grid.device
    if len(ys) == 0:
        return torch.zeros(R, R, device=device)
    h_b = F.grid_sample(h_grid.unsqueeze(0).unsqueeze(0), surf.unsqueeze(0).unsqueeze(0),
                        mode='bilinear', padding_mode='border', align_corners=True).squeeze()
    u = torch.empty(ys.shape[0], device=device)
    with torch.no_grad():
        for s, e, K_eff, w_b in chunks:
            u[s:e] = (w_b * K_eff * h_b.unsqueeze(0).expand(e - s, -1)).sum(dim=-1)
    out = torch.zeros(R, R, device=device)
    out[ys, xs] = u
    return out


def _compute_uh_field(kernel, rec, h_grid, normalize_mode):
    """u_h = <h, K_theta(p, .)> at every interior pixel of rec.mask."""
    rec.normalize_mode = normalize_mode
    return uh_from_keff_chunks(keff_chunks(kernel, rec), rec.surface_points, h_grid)


def encode_latents(kernel, records):
    # The canonical K3 kernel uses the point-cloud encoder, which REJECTS
    # 'sdf_grid' in shape_ctx (encoder_2d.py raises). Only the SDF-CNN encoder
    # variant (J.8 ablation) consumes it — include it conditionally.
    use_sdf = hasattr(kernel, '_encoder') and hasattr(kernel._encoder, 'sdf_stem')
    sdf_target = int(getattr(getattr(kernel, '_encoder', None), 'sdf_grid_size', 64)) if use_sdf else 0
    with torch.no_grad():
        for rec in records:
            ctx = {
                'surface_points': rec.surface_points.unsqueeze(0),
                'surface_normals': rec.surface_normals.unsqueeze(0),
                'interior_points': rec.interior_pool.unsqueeze(0),
            }
            if use_sdf:
                if rec.sdf_grid.shape[-1] != sdf_target:
                    sg = F.interpolate(rec.sdf_grid.unsqueeze(0).unsqueeze(0).float(),
                                       size=(sdf_target, sdf_target),
                                       mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
                else:
                    sg = rec.sdf_grid
                ctx['sdf_grid'] = sg.unsqueeze(0)
            rec.shape_latent = kernel.encode(ctx)


def _downsample(field, target_res):
    if field.shape[-1] == target_res:
        return field
    return F.interpolate(field.unsqueeze(0).unsqueeze(0).float(), size=(target_res, target_res),
                          mode='bilinear', align_corners=True).squeeze(0).squeeze(0)


def uh_at_res(kernel, rec, h_grid, R_in, normalize_mode, prob_name):
    """On-the-fly u_h at R_in, memoized per problem (u_h is deterministic)."""
    if prob_name in rec.uh_memo:
        return rec.uh_memo[prob_name].to(rec.mask.device)
    class _LowResRec: pass
    lr = _LowResRec()
    lr.R = R_in
    lr.mask = (_downsample(rec.mask.float(), R_in) > 0.5)
    lr.surface_points = rec.surface_points
    lr.surface_normals = rec.surface_normals
    lr.surface_weights = rec.surface_weights
    lr.shape_latent = rec.shape_latent
    lr.pixel_conv = getattr(rec, 'pixel_conv', 'legacy')
    h_grid_lo = _downsample(h_grid, R_in)
    uh = _compute_uh_field(kernel, lr, h_grid_lo, normalize_mode)
    rec.uh_memo[prob_name] = uh.cpu()
    return uh


def load_uh_cache(records, cache_dir):
    """Pre-fill each record's u_h memo from <cache_dir>/<shape>.pt (tools/uh_cache_2d.py)."""
    n_hit = n_miss = 0
    for rec in records:
        fn = Path(cache_dir) / f'{rec.shape_dir.name}.pt'
        if not fn.exists():
            n_miss += len(rec.problems)
            continue
        d = torch.load(fn, map_location='cpu')
        for prob_name, _, _, _ in rec.problems:
            if prob_name in d:
                rec.uh_memo[prob_name] = d[prob_name]
                n_hit += 1
            else:
                n_miss += 1
    print(f'[abl] u_h cache: {n_hit} hits, {n_miss} misses (misses computed on the fly)', flush=True)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--kernel-ckpt', type=str, required=True)
    pa.add_argument('--data-root', type=str,
                    default=os.environ.get('NHMO_DATA_2D', 'data/mnist_pde_2d_paramBC_lf'))
    pa.add_argument('--mnist-root', type=str, default=None,
                    help='directory containing raw/ MNIST idx files (env MNIST_ROOT)')
    pa.add_argument('--n-shapes-train', type=int, default=1000)
    pa.add_argument('--total-steps', type=int, default=10000)
    pa.add_argument('--lr', type=float, default=3e-4)
    pa.add_argument('--lr-min', type=float, default=1e-5)
    pa.add_argument('--warmup-steps', type=int, default=300)
    pa.add_argument('--checkpoint-every', type=int, default=2000)
    pa.add_argument('--log-every', type=int, default=200)
    pa.add_argument('--base-channels', type=int, default=48)
    pa.add_argument('--depth', type=int, default=4)
    pa.add_argument('--input-res', type=int, default=128)
    pa.add_argument('--gauge-mode', choices=['mask', 'none'], default='mask')
    pa.add_argument('--use-y-norm', action='store_true')
    pa.add_argument('--lift-inputs', choices=['mhfu', 'mf', 'msf'], default='mhfu',
                    help='mhfu = canonical (mask,h,f,u_h); mf = pure-balayage (mask,f); '
                         'msf = tuned pure-balayage (mask,sdf,f) — sdf is geometry-only, '
                         'so the lift stays strictly h-independent')
    pa.add_argument('--include-laplace', action='store_true',
                    help='include laplace_*.npz pairs in TRAINING (teaches the pure lift '
                         'that f==0 -> ~0; the canonical recipe trained on poisson_* only)')
    pa.add_argument('--geometry', choices=['mask', 'polyline'], default='mask',
                    help='kernel geometry: mask (default) or polyline (earlier versions)')
    pa.add_argument('--uh-cache-dir', type=str, default=None,
                    help='directory of per-shape u_h caches written by tools/uh_cache_2d.py')
    pa.add_argument('--seed', type=int, default=None)
    pa.add_argument('--out', type=str, required=True)
    pa.add_argument('--device', type=str, default='cuda')
    args = pa.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if args.mnist_root:
        os.environ['MNIST_ROOT'] = args.mnist_root
    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = torch.load(args.kernel_ckpt, map_location='cpu', weights_only=False)['cfg']
    kernel = HarmonicMeasureField2D(
        encoder_cfg=cfg['encoder'], kernel_head_cfg=cfg['kernel_head'],
        normalize=cfg['kernel']['normalize'],
    ).to(device)
    sd = torch.load(args.kernel_ckpt, map_location='cpu', weights_only=False)
    kernel.load_state_dict(sd['model_state_dict'])
    for p in kernel.parameters(): p.requires_grad_(False)
    kernel.eval()
    print(f'[abl] kernel loaded; lift_inputs={args.lift_inputs}', flush=True)

    data_root = Path(args.data_root)
    gen_kwargs = dict(n_total_samples=512, n_polyline_vertices=200,
                      keep_top_k_digit_contours=4, n_interior_anchors=256)
    train_glob = (('poisson_*.npz', 'laplace_*.npz') if args.include_laplace
                  else ('poisson_*.npz',))
    train_records = _build_records(data_root, 'train', args.n_shapes_train, device, gen_kwargs,
                                   problem_glob=train_glob, geometry=args.geometry)
    print(f'[abl] pre-encoding shape latents', flush=True)
    encode_latents(kernel, train_records)
    if args.uh_cache_dir:
        load_uh_cache(train_records, args.uh_cache_dir)

    in_channels = {'mhfu': 4, 'mf': 2, 'msf': 3}[args.lift_inputs]
    lift = FieldLift2D(in_channels=in_channels, base_channels=args.base_channels,
                       depth=args.depth, gauge_mode=args.gauge_mode).to(device)
    print(f'[abl] lift params = {sum(p.numel() for p in lift.parameters()):,} '
          f'(in_channels={in_channels})', flush=True)

    optim = torch.optim.AdamW(lift.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=args.lr, total_steps=args.total_steps,
        pct_start=args.warmup_steps / max(args.total_steps, 1),
        anneal_strategy='cos', final_div_factor=args.lr / args.lr_min)

    metrics_path = out_dir / 'metrics.jsonl'
    np_rng = np.random.RandomState(123 if args.seed is None else args.seed)
    R_in = args.input_res

    y_std = 1.0
    if args.use_y_norm:
        u_samples = []
        for rec in train_records[:200]:
            for _, _, _, u_g in rec.problems:
                idx = torch.where(rec.mask)
                pick = np_rng.choice(len(idx[0]), min(200, len(idx[0])), replace=False)
                u_samples.append(u_g[idx[0][pick], idx[1][pick]].flatten())
        y_std = float(torch.cat(u_samples).std().clamp(min=1e-3).item())
        print(f'[abl] y_std={y_std:.4f}', flush=True)

    t_start = time.time()
    for step in range(args.total_steps):
        rec = train_records[np_rng.randint(len(train_records))]
        problem_idx = np_rng.randint(len(rec.problems))
        prob_name, h_grid, f_grid, u_grid = rec.problems[problem_idx]

        u_h_field = uh_at_res(kernel, rec, h_grid, R_in, cfg['kernel']['normalize'], prob_name)

        m_lo = _downsample(rec.mask.float(), R_in)
        f_lo = _downsample(f_grid, R_in)
        uh_lo = u_h_field  # already at R_in
        u_lo = _downsample(u_grid, R_in)
        if args.lift_inputs == 'mhfu':
            h_lo = _downsample(h_grid, R_in)
            x_in = torch.stack([m_lo, h_lo, f_lo, uh_lo], dim=0).unsqueeze(0)
        elif args.lift_inputs == 'msf':  # pure balayage + SDF geometry channel
            sdf_lo = _downsample(rec.sdf_grid, R_in)
            x_in = torch.stack([m_lo, sdf_lo, f_lo], dim=0).unsqueeze(0)
        else:  # mf — pure balayage: v_phi sees geometry and source only
            x_in = torch.stack([m_lo, f_lo], dim=0).unsqueeze(0)
        m_b = (m_lo > 0.5).unsqueeze(0).float()

        v_phi = lift(x_in, m_b)
        u_pred = (uh_lo.unsqueeze(0) + v_phi) * m_b
        u_target = u_lo.unsqueeze(0) * m_b
        if args.use_y_norm:
            loss = (((u_pred - u_target) / y_std) ** 2 * m_b).sum() / m_b.sum().clamp(min=1)
        else:
            loss = ((u_pred - u_target) ** 2 * m_b).sum() / m_b.sum().clamp(min=1)
        rel_l2 = (u_pred - u_target).norm() / (u_target.norm() + 1e-12)

        optim.zero_grad(); loss.backward(); optim.step(); sched.step()

        if step % args.log_every == 0:
            log = {'step': step, 'lr': float(optim.param_groups[0]['lr']),
                   'loss_mse': float(loss.item()), 'rel_l2': float(rel_l2.item()),
                   'lift_inputs': args.lift_inputs, 'elapsed_s': time.time() - t_start}
            with open(metrics_path, 'a') as fh:
                fh.write(json.dumps(log) + '\n')
            print(f'[abl] step={step} lr={log["lr"]:.2e} loss={loss.item():.4e} '
                  f'rel_l2={rel_l2.item():.3f}', flush=True)

        if (step + 1) % args.checkpoint_every == 0 or step + 1 == args.total_steps:
            torch.save({'step': step + 1, 'lift_state_dict': lift.state_dict(),
                        'lift_cfg': dict(in_channels=in_channels, base_channels=args.base_channels,
                                         depth=args.depth, gauge_mode=args.gauge_mode,
                                         input_res=R_in, lift_inputs=args.lift_inputs),
                        'kernel_ckpt': args.kernel_ckpt, 'y_std': y_std, 'seed': args.seed,
                        'geometry': args.geometry},
                       out_dir / f'checkpoint_step_{step + 1}.pt')

    print(f'[abl] DONE total_elapsed={int(time.time()-t_start)}s', flush=True)


if __name__ == '__main__':
    main()
