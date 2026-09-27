"""2D inference timing: per-shape precompute and per-problem cached solve at 128^2.

per shape  : mask-registered kernel geometry (nhmo.data.mask_geometry), shape encode, and
             K_eff[q, j] = w_j K(p_q, zeta_j) for every interior pixel q of the 128^2 mask
per problem: h at the boundary nodes, u_h = K_eff @ h, input resampling to 128^2, lift forward
             (default: NHMO lift lift_2d_l17c_mask.pt; for the variant with residual head pass
             --lift lift_2d_msf_mask.pt --r-ckpt rhead_2d_mfr_mask.pt)

Defaults: first 10 shapes of test_ood. The first shape is a warm-up.

    python tools/bench_2d_inference.py --out results/bench2d.json
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
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nhmo.core.kernel import ShapeLatent  # noqa: E402
from nhmo.data import mask_geometry as mg  # noqa: E402
from nhmo.eval.mnist_pde_2d import downsample, load_field_net, load_kernel_2d  # noqa: E402

_IDX_RE = re.compile(r'idx(\d+)')


def main() -> int:
    pa = argparse.ArgumentParser()
    pa.add_argument('--ckpt-dir', default=os.environ.get('NHMO_CKPT_DIR', 'checkpoints'))
    pa.add_argument('--data-root', default=os.environ.get('NHMO_DATA_2D', 'data/mnist_pde_2d_paramBC_lf'))
    pa.add_argument('--split', default='test_ood')
    pa.add_argument('--n-shapes', type=int, default=10)
    pa.add_argument('--kernel', default='kernel_2d_mask.pt')
    pa.add_argument('--lift', default='lift_2d_l17c_mask.pt')
    pa.add_argument('--r-ckpt', default='none', help='residual head of the variant (default: none)')
    pa.add_argument('--res', type=int, default=128)
    pa.add_argument('--out', default=None)
    a = pa.parse_args()
    device = torch.device('cuda')
    ck = Path(a.ckpt_dir) / '2d'
    kernel, cfg = load_kernel_2d(ck / a.kernel, device)
    lift, lc = load_field_net(ck / a.lift, device)
    vf_inputs = lc.get('lift_inputs', 'mhfu')
    r_head = (load_field_net(ck / a.r_ckpt, device, 'r_state_dict', 'r_cfg')[0]
              if a.r_ckpt and a.r_ckpt != 'none' else None)
    shape_dirs = sorted(d for d in (Path(a.data_root) / a.split).iterdir() if d.is_dir())[: a.n_shapes]
    R = a.res

    def sync():
        torch.cuda.synchronize()

    pre_ms, solve_ms, n_pairs = [], [], 0
    with torch.no_grad():
        for si, sd in enumerate([shape_dirs[0]] + shape_dirs):          # first entry = warm-up
            idx = int(_IDX_RE.search(sd.name).group(1))
            mask_np = np.load(sd / 'mask.npy').astype(bool)
            mask_full = torch.from_numpy(mask_np).to(device)
            probs = [np.load(p) for p in sorted(sd.glob('*.npz'))]
            sync()
            t0 = time.perf_counter()
            geo = mg.geometry_to_torch(mg.shape_geometry(mask_np, idx), device)
            surf, surf_n, w = geo['surface_points'], geo['surface_normals'], geo['surface_weights']
            sl = kernel.encode({'surface_points': surf.unsqueeze(0), 'surface_normals': surf_n.unsqueeze(0),
                                'interior_points': geo['interior_points'].unsqueeze(0)})
            m_lo = downsample(mask_full.float(), R)
            m_b = (m_lo > 0.5).unsqueeze(0).float()
            p, ys, xs = mg.pixel_points(m_b[0] > 0.5)
            keff = torch.empty(p.shape[0], surf.shape[0], device=device)
            for s in range(0, p.shape[0], 4096):
                e = min(p.shape[0], s + 4096)
                n = e - s
                log_K = kernel.log_kernel(p[s:e], surf.unsqueeze(0).expand(n, -1, -1),
                                          surf_n.unsqueeze(0).expand(n, -1, -1),
                                          ShapeLatent(tokens=sl.tokens.expand(n, -1, -1)))
                logZ = torch.logsumexp(log_K + torch.log(w + 1e-30), dim=-1, keepdim=True)
                keff[s:e] = torch.exp(log_K - logZ) * w
            if vf_inputs == 'msf':
                sdf = torch.from_numpy((-distance_transform_edt(mask_np) + distance_transform_edt(~mask_np))
                                       .astype(np.float32) * (2.0 / mask_np.shape[0])).to(device)
                sdf_lo = downsample(sdf, R)
            sync()
            if si > 0:
                pre_ms.append((time.perf_counter() - t0) * 1e3)
            for d in probs:
                h = torch.from_numpy(d['h'].astype(np.float32)).to(device)
                f = (torch.from_numpy(d['f'].astype(np.float32)).to(device) if 'f' in d.files
                     else torch.zeros_like(h))
                sync()
                t0 = time.perf_counter()
                h_b = F.grid_sample(h[None, None], surf[None, None], mode='bilinear',
                                    padding_mode='border', align_corners=True).reshape(-1)
                uh = torch.zeros(R, R, device=device)
                uh[ys, xs] = keff @ h_b
                h_lo, f_lo = downsample(h, R), downsample(f, R)
                if vf_inputs == 'mhfu':
                    x = torch.stack([m_lo, h_lo, f_lo, uh], 0)
                else:
                    x = torch.stack([m_lo, sdf_lo, f_lo], 0)
                u = uh[None] + lift(x[None], m_b)
                if r_head is not None:
                    u = u + r_head(torch.stack([m_lo, h_lo, uh], 0)[None], m_b)
                u = u * m_b
                sync()
                if si > 0:
                    solve_ms.append((time.perf_counter() - t0) * 1e3)
                    n_pairs += 1
    summ = {'per_shape_precompute_ms_median': float(np.median(pre_ms)),
            'per_problem_ms_median': float(np.median(solve_ms)),
            'per_problem_ms_mean': float(np.mean(solve_ms)), 'n_shapes': len(pre_ms), 'n_pairs': n_pairs,
            'gpu': torch.cuda.get_device_name(0)}
    print(json.dumps(summ, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({'summary': summ, 'precompute_ms': pre_ms, 'solve_ms': solve_ms,
                                           'args': vars(a)}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
