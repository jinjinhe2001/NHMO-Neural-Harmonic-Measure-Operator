"""2D MNIST kernel training against precomputed Walk-on-Spheres KDE targets.

NLL on individual WoS exit samples collapses the 2D kernel to delta spikes on
the short digit boundaries. Instead, the kernel is fit to a kernel-density
estimate of 10^4 WoS exits per query (tools/precompute_omega_gt_2d.py):
    loss = KL(P_theta || P_gt) + lambda_l1 * |P_theta - P_gt|_1
over a random subset of --n-bdry-sample boundary nodes per step.

Released kernel (checkpoints/2d/kernel_2d_mask.pt; targets from tools/gen_mask_corpus.py):
    python -m nhmo.train.mnist_kde_main --config configs/mnist_kernel.yaml \\
        --omega-gt-train data/omega_mask_5k.pt --total-steps 60000 \\
        --checkpoint-every 5000 --log-every 200 --seed 0 --out runs/kernel_2d_mask
(defaults: lr 3e-4 one-cycle with 500 warm-up steps, lr_min 1e-5, 64 boundary
nodes per step, lambda_l1 0.5, 4 shapes per step.)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nhmo.core.kernel_2d import HarmonicMeasureField2D
from nhmo.core.kernel import ShapeLatent
from nhmo.train.config import load_config


def _bilinear_interp(grid: Tensor, points_xy: Tensor) -> Tensor:
    g = grid.unsqueeze(0).unsqueeze(0)
    pts = points_xy.reshape(-1, 1, 1, 2)
    out = F.grid_sample(g.expand(pts.shape[0], -1, -1, -1), pts,
                        mode='bilinear', padding_mode='border', align_corners=True)
    return out.reshape(points_xy.shape[:-1])


def main() -> int:
    pa = argparse.ArgumentParser()
    pa.add_argument('--omega-gt-train', type=str, required=True)
    pa.add_argument('--omega-gt-test', type=str, default=None)
    pa.add_argument('--config', type=str, required=True)
    pa.add_argument('--total-steps', type=int, default=30000)
    pa.add_argument('--lr', type=float, default=3e-4)
    pa.add_argument('--lr-min', type=float, default=1e-5)
    pa.add_argument('--warmup-steps', type=int, default=500)
    pa.add_argument('--checkpoint-every', type=int, default=2000)
    pa.add_argument('--log-every', type=int, default=100)
    pa.add_argument('--n-bdry-sample', type=int, default=64,
                    help='Sub-sample N_s boundary points per training step (v1 used 64)')
    pa.add_argument('--lambda-l1', type=float, default=0.5)
    pa.add_argument('--warm-start', type=str, default=None,
                    help='Path to a previous mnist_kde checkpoint to load weights from.')
    pa.add_argument('--out', type=str, required=True)
    pa.add_argument('--device', type=str, default='cuda')
    pa.add_argument('--seed', type=int, default=None,
                    help='torch seed for the kernel initialization (the released kernel used 0)')
    args = pa.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load precomputed omega_gt
    print(f'[kde-train] loading {args.omega_gt_train}', flush=True)
    db = torch.load(args.omega_gt_train, map_location='cpu', weights_only=False)
    surface_points = db['surface_points'].to(device)        # (N_shapes, N_s, 2)
    surface_normals = db['surface_normals'].to(device)
    surface_weights = db['surface_weights'].to(device)
    queries = db['queries'].to(device)                       # (N_shapes, K, 2)
    omega_gt = db['omega_gt'].to(device)                     # (N_shapes, K, N_s)
    sdf_grids = db['sdf_grids'].to(device)                   # (N_shapes, G, G) WoS frame
    N_shapes, K_per_shape, N_s = omega_gt.shape
    print(f'[kde-train] {N_shapes} shapes x {K_per_shape} queries x {N_s} surface pts', flush=True)

    # Build kernel
    cfg = load_config(args.config)
    kernel = HarmonicMeasureField2D(
        encoder_cfg=cfg['encoder'], kernel_head_cfg=cfg['kernel_head'],
        normalize=cfg['kernel']['normalize'],
    ).to(device)
    print(f'[kde-train] kernel params = {sum(p.numel() for p in kernel.parameters()):,}', flush=True)

    if args.warm_start:
        prev = torch.load(args.warm_start, map_location='cpu', weights_only=False)
        kernel.load_state_dict(prev['model_state_dict'])
        print(f'[kde-train] warm-started from {args.warm_start} (src step {prev.get("step")})', flush=True)

    optim = torch.optim.AdamW(kernel.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=args.lr, total_steps=args.total_steps,
        pct_start=args.warmup_steps / max(args.total_steps, 1),
        anneal_strategy='cos', final_div_factor=args.lr / args.lr_min,
    )

    metrics_path = out_dir / 'metrics.jsonl'
    rng = np.random.RandomState(42)
    t0 = time.time()
    BATCH_SHAPES = 4

    for step in range(args.total_steps):
        # Sample mini-batch of (shape, query) pairs
        shape_ids = rng.randint(0, N_shapes, BATCH_SHAPES)
        query_ids = rng.randint(0, K_per_shape, BATCH_SHAPES)

        surf = surface_points[shape_ids]            # (B, N_s, 2)
        surf_n = surface_normals[shape_ids]
        weights = surface_weights[shape_ids]         # (B, N_s)
        p = queries[shape_ids, query_ids]            # (B, 2)
        omega = omega_gt[shape_ids, query_ids]       # (B, N_s)

        # Sub-sample n_bdry_sample boundary indices per shape
        K = args.n_bdry_sample
        bs = torch.from_numpy(
            rng.randint(0, N_s, (BATCH_SHAPES, K))
        ).to(device).long()
        # Gather
        surf_sub = torch.gather(surf, 1, bs.unsqueeze(-1).expand(-1, -1, 2))     # (B, K, 2)
        surf_n_sub = torch.gather(surf_n, 1, bs.unsqueeze(-1).expand(-1, -1, 2))
        w_sub = torch.gather(weights, 1, bs)                                      # (B, K)
        omega_sub = torch.gather(omega, 1, bs)                                    # (B, K)

        # Encode shape latent
        shape_ctx = {
            'surface_points': surf, 'surface_normals': surf_n,
            'interior_points': p.unsqueeze(1),
        }
        if cfg.get('encoder', {}).get('type') == 'sdf_cnn':
            shape_ctx['sdf_grid'] = sdf_grids[shape_ids]
        sl = kernel.encode(shape_ctx)

        # Forward log_K at sub-sampled boundary points
        log_K = kernel.log_kernel(p, surf_sub, surf_n_sub, sl)                    # (B, K)
        # Soft-norm: divide by Z_sub computed over the sub-sampled K
        if cfg['kernel']['normalize'] == 'hard':
            P_pred = torch.exp(log_K)
        else:
            log_w = torch.log(w_sub + 1e-30)
            log_Z = torch.logsumexp(log_K + log_w, dim=-1, keepdim=True)
            P_pred = torch.exp(log_K - log_Z)                                     # (B, K) — density

        # Normalize omega_gt over the same sub-set
        P_gt_un = omega_sub                                                       # (B, K) — density
        Z_gt = (P_gt_un * w_sub).sum(dim=-1, keepdim=True) + 1e-12
        P_gt = P_gt_un / Z_gt

        # KL(P_pred || P_gt) with weights as a quadrature measure
        # discrete KL: Σ w_i P_pred_i log(P_pred_i / P_gt_i)
        eps = 1e-12
        kl = (w_sub * P_pred * (torch.log(P_pred + eps) - torch.log(P_gt + eps))).sum(dim=-1).mean()
        l1 = (w_sub * (P_pred - P_gt).abs()).sum(dim=-1).mean()
        loss = kl + args.lambda_l1 * l1

        # Diagnostic: rel-L2 of densities
        rel_l2 = (P_pred - P_gt).pow(2).sum(dim=-1).sqrt() / (P_gt.pow(2).sum(dim=-1).sqrt() + eps)
        rel_l2 = rel_l2.mean().item()

        optim.zero_grad()
        loss.backward()
        optim.step()
        sched.step()

        if step % args.log_every == 0:
            log = {
                'step': step,
                'lr': float(optim.param_groups[0]['lr']),
                'loss_total': float(loss.item()),
                'loss_kl': float(kl.item()),
                'loss_l1': float(l1.item()),
                'rel_l2_density': rel_l2,
                'elapsed_s': time.time() - t0,
            }
            with open(metrics_path, 'a') as fh:
                fh.write(json.dumps(log) + '\n')
            print(f'[kde-train] step={step} lr={log["lr"]:.2e} '
                  f'loss={loss.item():.4e} kl={kl.item():.4e} l1={l1.item():.4e} '
                  f'rel_l2_dens={rel_l2:.3f}', flush=True)

        if step > 0 and step % args.checkpoint_every == 0:
            torch.save({
                'step': step,
                'cfg': cfg,
                'model_state_dict': kernel.state_dict(),
            }, out_dir / f'checkpoint_step_{step}.pt')

    torch.save({
        'step': args.total_steps,
        'cfg': cfg,
        'model_state_dict': kernel.state_dict(),
    }, out_dir / f'checkpoint_step_{args.total_steps}.pt')
    print(f'[kde-train] DONE total={time.time()-t0:.0f}s', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
