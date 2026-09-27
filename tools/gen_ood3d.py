"""Generate 3D coefficient-OOD Poisson problems on MCB-B test shapes with lapy FEM references.

For each category, take the first --n-shapes shapes of the
unknown_shape_unknown_prob split, and generate --k-per-shape OOD problems with
a DOCUMENTED family (both methods see these zero-retraining; neither trained on
this family, mirroring the 2D OOD protocol):

  h_ood(v) = s_h * (a*x + b*x*y + c*(x^2 - y^2) + d*exp(x)*cos(y)),  a..d ~ U[1,2]
  f_ood(v) = s_f * (alpha*sin(2 pi x)*cos(2 pi y)*(1+z) + beta*exp(-|v-v0|^2/(2*0.15^2)))
             alpha,beta ~ U[1,2], v0 = random interior vertex
  s_h = 0.15 (released h range ~[-0.05, 0.36]), s_f = 50 (released |f| <= 100)
  coordinates in the released mesh frame ([-0.5, 0.5]^3).

GT via lapy (TetMesh + Solver.poisson with Dirichlet dtup) — the same solver the
NGF repo uses to verify its released solutions. PARITY GATE: per category we
first re-solve one RELEASED problem and compare to its released sol; the sign
convention is auto-detected there and the gate must pass (<2% rel) before any
OOD problem is written.

Output tree (released-schema sol.npz):
  <out>/ngf_solutions/<cat>/<sid>/ood-<k>/sol.npz   + manifest.json

--s-f 0 gives the Laplace-only variant (f == 0) that isolates boundary-data
extrapolation. Evaluate with
  python -m nhmo.eval.mcb_lift --problems-root <out>/ngf_solutions --category nut       --n-shapes 10 --bcs-per-shape 4 --kernel-ckpt ... --lift-ckpt ...

Requires `lapy` (pip install lapy). The coefficients of each problem are drawn
from a seed derived from (category, shape, k, --seed). The rebuttal problem set
was drawn with process-dependent seeds; its coefficients are recorded in its
manifest.json, and the exact problems are part of the data release.
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np

import zlib

from lapy import TetMesh, Solver

CATS = ['nut', 'gear', 'motor', 'fitting', 'screws_and_bolts']


def shape_list(mcb_root, cat, n_shapes):
    split = Path(mcb_root) / 'splits' / cat / 'unknown_shape_unknown_prob.txt'
    sids, seen = [], set()
    for line in split.read_text().strip().splitlines():
        parts = Path(line).parts
        if len(parts) < 5 or parts[-1] != 'sol.npz':
            continue
        sid, bc = parts[-3], parts[-2]
        if sid not in seen:
            seen.add(sid); sids.append((sid, bc))
        if len(sids) >= n_shapes:
            break
    return sids


def h_fn(v, a, b, c, d):
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    return a * x + b * x * y + c * (x ** 2 - y ** 2) + d * np.exp(x) * np.cos(y)


def f_fn(v, alpha, beta, v0):
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    bump = np.exp(-((v - v0) ** 2).sum(1) / (2 * 0.15 ** 2))
    return alpha * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) * (1 + z) + beta * bump


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--mcb-root', default=os.environ.get('MCB_ROOT', 'data/MCB_benchmark'))
    pa.add_argument('--out', required=True)
    pa.add_argument('--seed', type=int, default=0)
    pa.add_argument('--n-shapes', type=int, default=10)
    pa.add_argument('--k-per-shape', type=int, default=4)
    pa.add_argument('--s-h', type=float, default=0.15)
    pa.add_argument('--s-f', type=float, default=50.0)
    args = pa.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    manifest = {'family': 'h = s_h*(a x + b xy + c (x^2-y^2) + d e^x cos y), '
                          'f = s_f*(alpha sin2pix cos2piy (1+z) + beta gauss(v0,0.15)); '
                          'a..d,alpha,beta ~ U[1,2]',
                's_h': args.s_h, 's_f': args.s_f, 'problems': [], 'parity': {}}

    for cat in CATS:
        sids = shape_list(args.mcb_root, cat, args.n_shapes)
        print(f'== {cat}: {len(sids)} shapes ==', flush=True)
        sign = None
        for si, (sid, bc0) in enumerate(sids):
            base = Path(args.mcb_root) / 'ngf_solutions' / cat / sid / bc0 / 'sol.npz'
            d = np.load(base)
            v = np.asarray(d['v_tet'], np.float64)
            t = np.asarray(d['f_tet'], np.int64)
            bd = np.asarray(d['bd_v_inds'], np.int64)
            t0 = time.time()
            mesh = TetMesh(v, t)
            solver = Solver(mesh)

            if sign is None:
                # parity gate on the released problem (auto-detect sign)
                src0 = np.asarray(d['source_term'], np.float64).reshape(-1)
                bdv0 = np.asarray(d['bd_v_vals'], np.float64)
                sol0 = np.asarray(d['sol'], np.float64).reshape(-1)
                best = None
                for sgn in (+1.0, -1.0):
                    u = solver.poisson(sgn * src0, dtup=(bd, bdv0))
                    err = np.linalg.norm(u - sol0) / (np.linalg.norm(sol0) + 1e-12)
                    if best is None or err < best[1]:
                        best = (sgn, err)
                sign, perr = best
                manifest['parity'][cat] = {'shape': sid, 'sign': sign, 'rel_err': float(perr)}
                print(f'  [parity] {cat}/{sid}: sign={sign:+.0f} rel_err={perr:.4f}', flush=True)
                if perr > 0.02:
                    raise RuntimeError(f'PARITY FAIL {cat}: rel_err={perr:.4f} — investigate before generating OOD GT')

            interior = np.setdiff1d(np.arange(len(v)), bd)
            for k in range(args.k_per_shape):
                key = f'{cat}/{sid}/{k}'.encode()
                rng = np.random.RandomState((zlib.crc32(key) + 1_000_003 * args.seed) % (2 ** 31))
                a, b, c, dd, alpha, beta = 1.0 + rng.rand(6)
                v0 = v[interior[rng.randint(len(interior))]]
                h_all = args.s_h * h_fn(v, a, b, c, dd)
                f_all = args.s_f * f_fn(v, alpha, beta, v0)
                u = solver.poisson(sign * f_all, dtup=(bd, h_all[bd]))
                pdir = out / 'ngf_solutions' / cat / sid / f'ood-{k}'
                pdir.mkdir(parents=True, exist_ok=True)
                np.savez(pdir / 'sol.npz',
                         v_tet=v, f_tet=np.asarray(d['f_tet'], np.int32),
                         sol=np.asarray(u, np.float64),
                         source_term=f_all.astype(np.float32).reshape(-1, 1),
                         bd_v_inds=bd.astype(np.int32),
                         bd_v_vals=h_all[bd].astype(np.float64),
                         bd_A=np.float64(a), bd_B=np.float64(b),
                         bd_C=np.float64(c), bd_D=np.float64(dd))
                manifest['problems'].append({'cat': cat, 'sid': sid, 'k': k,
                                             'v0': v0.tolist(),
                                             'a': a, 'b': b, 'c': c, 'd': dd,
                                             'alpha': alpha, 'beta': beta,
                                             'sol_range': [float(u.min()), float(u.max())]})
            print(f'  {sid}: {args.k_per_shape} OOD problems ({time.time()-t0:.1f}s)', flush=True)

    with open(out / 'manifest.json', 'w') as fp:
        json.dump(manifest, fp, indent=1)
    print('done:', len(manifest['problems']), 'problems')


if __name__ == '__main__':
    main()
