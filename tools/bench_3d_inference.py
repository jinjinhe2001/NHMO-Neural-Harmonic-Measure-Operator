"""3D inference timing on MCB-B: per-shape build and per-problem cached solve.

Measures the optimized path of `nhmo.core.fast_inference`:
  build  = geometry load + surface sampling + GPU SDF grid + interior anchors
           + shape encode + folded K_eff over all interior tet vertices
           + nearest-boundary lookup
  solve  = boundary matvec K_eff @ h + source encode + query-folded lift
           over all interior tet vertices
With `--reference` it also times the original path (trimesh SDF, reference
K_eff, per-query lift) on the same shapes.

Defaults match the timing configuration quoted in the rebuttal (64^3 SDF grid,
2048 surface samples, 2048 source probes, bf16 K_eff, fp32 lift). The first
shape is a warm-up and the first problem of each shape is not timed.

    python tools/bench_3d_inference.py --category nut --out results/bench_nut.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nhmo.core.fast_inference import precompute_shape, solve_cached  # noqa: E402
from nhmo.data.mcb_loader import load_mcb_eval_fields  # noqa: E402
from nhmo.eval.mcb_lift import load_models, split_pairs  # noqa: E402


def main() -> int:
    pa = argparse.ArgumentParser()
    pa.add_argument("--category", default="nut")
    pa.add_argument("--ckpt-dir", default=os.environ.get("NHMO_CKPT_DIR", "checkpoints"))
    pa.add_argument("--mcb-root", default=os.environ.get("MCB_ROOT", "data/MCB_benchmark"))
    pa.add_argument("--n-shapes", type=int, default=5)
    pa.add_argument("--per-shape", type=int, default=4)
    pa.add_argument("--grid-size", type=int, default=64)
    pa.add_argument("--n-surface", type=int, default=2048)
    pa.add_argument("--n-interior", type=int, default=512)
    pa.add_argument("--n-source-probe", type=int, default=2048)
    pa.add_argument("--keff-amp", choices=["bf16", "fp32"], default="bf16")
    pa.add_argument("--lift-amp", choices=["bf16", "fp32"], default="fp32")
    pa.add_argument("--reference", action="store_true", help="also time the original path")
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--out", default=None)
    args = pa.parse_args()

    device = torch.device("cuda")
    ck = Path(args.ckpt_dir) / "3d"
    kernel, lift, _, _ = load_models(str(ck / f"kernel_{args.category}.pt"),
                                     str(ck / f"lift_{args.category}.pt"), device)
    by_shape = split_pairs(Path(args.mcb_root), args.category, "unknown_shape_unknown_prob")
    shapes = [(s, by_shape[s][: args.per_shape]) for s in list(by_shape)[: args.n_shapes]]
    keff_amp = torch.bfloat16 if args.keff_amp == "bf16" else None
    lift_amp = torch.bfloat16 if args.lift_amp == "bf16" else None

    def sync():
        torch.cuda.synchronize()

    modes = [("fast", True)] + ([("reference", False)] if args.reference else [])
    res = {m: {"build_s": [], "solve_ms": [], "n_queries": []} for m, _ in modes}
    with torch.no_grad():
        # warm-up (CUDA context, kernels, allocator)
        c = precompute_shape(kernel, shapes[0][1][0], args.grid_size, args.n_surface, args.n_interior,
                             device, seed=args.seed, fast=True, keff_amp_dtype=keff_amp)
        f = load_mcb_eval_fields(str(shapes[0][1][0]))
        solve_cached(lift, c, f.bd_v_inds, f.bd_v_vals, f.source_term, args.n_source_probe,
                     seed=args.seed, fast=True, lift_amp_dtype=lift_amp)
        del c
        for sid, probs in shapes:
            for m, fast in modes:
                torch.cuda.empty_cache()
                sync()
                t0 = time.perf_counter()
                cache = precompute_shape(kernel, probs[0], args.grid_size, args.n_surface, args.n_interior,
                                         device, seed=args.seed, fast=fast,
                                         keff_amp_dtype=keff_amp if fast else None)
                sync()
                res[m]["build_s"].append(time.perf_counter() - t0)
                res[m]["n_queries"].append(int(cache.p_model.shape[0]))
                for j, p in enumerate(probs):
                    f = load_mcb_eval_fields(str(p))
                    sync()
                    t0 = time.perf_counter()
                    solve_cached(lift, cache, f.bd_v_inds, f.bd_v_vals, f.source_term,
                                 args.n_source_probe, seed=args.seed, fast=fast,
                                 lift_amp_dtype=lift_amp if fast else None)
                    sync()
                    if j > 0:
                        res[m]["solve_ms"].append((time.perf_counter() - t0) * 1e3)
                del cache
            print(f"[bench] {sid} done: " + ", ".join(
                f"{m} build {res[m]['build_s'][-1]:.2f}s" for m, _ in modes), flush=True)

    summary = {m: {"build_s_median": float(np.median(r["build_s"])),
                   "solve_ms_median": float(np.median(r["solve_ms"])),
                   "n_queries_median": float(np.median(r["n_queries"]))} for m, r in res.items()}
    print(json.dumps(summary, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"summary": summary, "raw": res, "args": vars(args),
                                              "gpu": torch.cuda.get_device_name(0)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
