"""Residual head of the 2D variant with residual head (paper Section 6 / Appendix K.1).

    u(p) = <h, K_theta>  +  v_f(Omega, f)  +  r(Omega, h, u_h)
            frozen kernel   frozen source-only lift   trainable residual head

The source term v_f (inputs mask, sdf, f) stays independent of h; the
h-dependence that the kernel term misses is carried by the residual head r
with inputs (mask, h, u_h) and no f. Recipe identical to the original lift
(U-Net base 48 / depth 4, y-normalized masked MSE at 128^2, 10k steps,
AdamW + one-cycle cosine schedule), trained on the poisson_* problems.

    python -m nhmo.train.residual_head_train_2d \\
        --kernel-ckpt $CKPT/2d/kernel_2d_mask.pt --frozen-lift-ckpt $CKPT/2d/lift_2d_msf_mask.pt \\
        --uh-cache-dir data/uh_cache_mask_train128 --seed 0 --out runs/rhead_2d_mfr_mask
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from nhmo.core.field_lift import FieldLift2D
from nhmo.core.kernel_2d import HarmonicMeasureField2D
from nhmo.train.poisson_lift_field_train_2d import (
    _build_records, _downsample, encode_latents, load_uh_cache, uh_at_res,
)


def load_kernel(ckpt_path, device):
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = sd['cfg']
    k = HarmonicMeasureField2D(encoder_cfg=cfg['encoder'], kernel_head_cfg=cfg['kernel_head'],
                               normalize=cfg['kernel']['normalize']).to(device)
    k.load_state_dict(sd['model_state_dict'])
    for p in k.parameters():
        p.requires_grad_(False)
    k.eval()
    return k, cfg


def load_lift(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    lc = ck['lift_cfg']
    lift = FieldLift2D(in_channels=lc['in_channels'], base_channels=lc['base_channels'],
                       depth=lc['depth'], gauge_mode=lc['gauge_mode']).to(device)
    lift.load_state_dict(ck['lift_state_dict'])
    for p in lift.parameters():
        p.requires_grad_(False)
    lift.eval()
    return lift, lc


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--kernel-ckpt', required=True)
    pa.add_argument('--frozen-lift-ckpt', required=True, help='source-only (msf or mf) lift')
    pa.add_argument('--data-root', default=os.environ.get('NHMO_DATA_2D', 'data/mnist_pde_2d_paramBC_lf'))
    pa.add_argument('--mnist-root', default=None, help='directory containing raw/ (env MNIST_ROOT)')
    pa.add_argument('--n-shapes-train', type=int, default=1000)
    pa.add_argument('--total-steps', type=int, default=10000)
    pa.add_argument('--lr', type=float, default=3e-4)
    pa.add_argument('--lr-min', type=float, default=1e-5)
    pa.add_argument('--warmup-steps', type=int, default=300)
    pa.add_argument('--base-channels', type=int, default=48)
    pa.add_argument('--depth', type=int, default=4)
    pa.add_argument('--input-res', type=int, default=128)
    pa.add_argument('--checkpoint-every', type=int, default=5000)
    pa.add_argument('--geometry', choices=['mask', 'polyline'], default='mask',
                    help='kernel geometry: mask (default) or polyline (earlier versions)')
    pa.add_argument('--uh-cache-dir', default=None, help='u_h caches from tools/uh_cache_2d.py')
    pa.add_argument('--seed', type=int, default=None)
    pa.add_argument('--out', required=True)
    pa.add_argument('--device', default='cuda')
    args = pa.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    if args.mnist_root:
        os.environ['MNIST_ROOT'] = args.mnist_root
    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    R_in = args.input_res

    kernel, kcfg = load_kernel(args.kernel_ckpt, device)
    v_f, vf_lc = load_lift(args.frozen_lift_ckpt, device)
    vf_inputs = vf_lc.get('lift_inputs', 'mf')
    print(f'[rhead] kernel + frozen {vf_inputs} lift loaded', flush=True)

    gen_kwargs = dict(n_total_samples=512, n_polyline_vertices=200,
                      keep_top_k_digit_contours=4, n_interior_anchors=256)
    # Training distribution mirrors the original lift: poisson_* problems only.
    tr = _build_records(Path(args.data_root), 'train', args.n_shapes_train, device, gen_kwargs,
                        geometry=args.geometry)
    encode_latents(kernel, tr)
    if args.uh_cache_dir:
        load_uh_cache(tr, args.uh_cache_dir)

    r_head = FieldLift2D(in_channels=3, base_channels=args.base_channels,
                         depth=args.depth, gauge_mode='mask').to(device)
    print(f'[rhead] params={sum(p.numel() for p in r_head.parameters()):,}', flush=True)
    optim = torch.optim.AdamW(r_head.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=args.lr, total_steps=args.total_steps,
        pct_start=args.warmup_steps / args.total_steps,
        anneal_strategy='cos', final_div_factor=args.lr / args.lr_min)

    np_rng = np.random.RandomState(123 if args.seed is None else args.seed)
    u_samples = []
    for rec in tr[:200]:
        for _, _, _, u_g in rec.problems:
            idx = torch.where(rec.mask)
            pick = np_rng.choice(len(idx[0]), min(200, len(idx[0])), replace=False)
            u_samples.append(u_g[idx[0][pick], idx[1][pick]].flatten())
    y_std = float(torch.cat(u_samples).std().clamp(min=1e-3).item())
    print(f'[rhead] y_std={y_std:.4f}', flush=True)

    t0 = time.time()
    for step in range(args.total_steps):
        rec = tr[np_rng.randint(len(tr))]
        prob_name, h_grid, f_grid, u_grid = rec.problems[np_rng.randint(len(rec.problems))]
        uh = uh_at_res(kernel, rec, h_grid, R_in, kcfg['kernel']['normalize'], prob_name)
        m_lo = _downsample(rec.mask.float(), R_in)
        m_b = (m_lo > 0.5).unsqueeze(0).float()
        f_lo = _downsample(f_grid, R_in)
        h_lo = _downsample(h_grid, R_in)
        u_lo = _downsample(u_grid, R_in)
        with torch.no_grad():
            if vf_inputs == 'msf':
                sdf_lo = _downsample(rec.sdf_grid, R_in)
                vf = v_f(torch.stack([m_lo, sdf_lo, f_lo], 0).unsqueeze(0), m_b)
            else:
                vf = v_f(torch.stack([m_lo, f_lo], 0).unsqueeze(0), m_b)
        rr = r_head(torch.stack([m_lo, h_lo, uh], 0).unsqueeze(0), m_b)
        u_pred = (uh.unsqueeze(0) + vf + rr) * m_b
        u_tgt = u_lo.unsqueeze(0) * m_b
        loss = (((u_pred - u_tgt) / y_std) ** 2 * m_b).sum() / m_b.sum().clamp(min=1)
        optim.zero_grad()
        loss.backward()
        optim.step()
        sched.step()
        if step % 200 == 0:
            rel = (u_pred - u_tgt).norm() / (u_tgt.norm() + 1e-12)
            print(f'[rhead] step={step} loss={loss.item():.4e} rel_l2={rel.item():.3f} '
                  f'({int(time.time() - t0)}s)', flush=True)
        if (step + 1) % args.checkpoint_every == 0 or step + 1 == args.total_steps:
            torch.save({'step': step + 1, 'r_state_dict': r_head.state_dict(),
                        'frozen_mf_ckpt': args.frozen_lift_ckpt, 'y_std': y_std,
                        'seed': args.seed, 'geometry': args.geometry,
                        'r_cfg': dict(in_channels=3, base_channels=args.base_channels,
                                      depth=args.depth, gauge_mode='mask', input_res=R_in)},
                       out_dir / f'rhead_step_{step + 1}.pt')
    print(f'[rhead] DONE total_elapsed={int(time.time() - t0)}s', flush=True)


if __name__ == '__main__':
    main()
