"""Synthetic Laplace probe of the 3D kernel (paper Table 4).

Same trick as tools.eval_mnist_u_check but in 3D: pick functions h with
Δh = 0 in 3D so the Dirichlet solution on any domain Ω is u_true(p) = h(p).
This isolates KERNEL quality from any source-term modeling, since source
is forced to zero by construction.

3D harmonic test BCs (all satisfy Δh = 0 in 3D):
    h(x,y,z) = x                        (linear)
    h(x,y,z) = x*y                      (off-diagonal quadratic)
    h(x,y,z) = x*x - y*y                (real harmonic poly)
    h(x,y,z) = 2*z*z - x*x - y*y        (Y_2^0 spherical harmonic)
    h(x,y,z) = exp(x) * cos(y)          (3D harmonic)
    h(x,y,z) = exp(x) * sin(y)          (3D harmonic)

Run on server:
    python3 -m tools.eval_mcb_laplace \\
        --ckpt checkpoints/3d/kernel_nut.pt --category nut \\
        --split known_shape_known_prob --n-shapes 10 --n-queries 256
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from nhmo.core.fast_inference import stable_seed
from nhmo.core.kernel import HarmonicMeasureField, ShapeLatent
from nhmo.data.mcb_loader import MCBSolDataset, MCBSolShapeGenerator


BC_KINDS = ("x", "xy", "x2_minus_y2", "Y20_zonal", "exp_cos", "exp_sin")


def h_eval(p: Tensor, kind: str) -> Tensor:
    """3D harmonic function evaluation."""
    x = p[..., 0]; y = p[..., 1]; z = p[..., 2]
    if kind == "x":
        return x
    if kind == "xy":
        return x * y
    if kind == "x2_minus_y2":
        return x * x - y * y
    if kind == "Y20_zonal":
        return 2.0 * z * z - x * x - y * y
    if kind == "exp_cos":
        return torch.exp(x) * torch.cos(y)
    if kind == "exp_sin":
        return torch.exp(x) * torch.sin(y)
    raise ValueError(f"unknown BC kind: {kind}")


def relative_l2_t(u_pred: Tensor, u_true: Tensor) -> float:
    num = torch.norm(u_pred - u_true).item()
    den = torch.norm(u_true).item() + 1e-12
    return float(num / den)


def evaluate_one(
    model: HarmonicMeasureField,
    sol_path: Path,
    grid_size: int,
    n_surface: int,
    n_interior: int,
    n_queries: int,
    device: torch.device,
    seed: int | None = None,
) -> dict:
    gen = MCBSolShapeGenerator(
        sol_npz_path=sol_path,
        grid_size=grid_size,
        n_surface=n_surface,
        n_interior_anchors=n_interior,
        device=str(device),
        seed=None if seed is None else stable_seed(sol_path.parent.parent.name, seed),
    )
    spec = gen()
    surface = spec["shape_ctx"]["surface_points"].to(device)
    surface_normals = spec["shape_ctx"]["surface_normals"].to(device)
    interior = spec["shape_ctx"]["interior_points"].squeeze(0).to(device)
    total_area = float(spec.get("total_area_model_frame", 4.0))

    if interior.shape[0] < n_queries:
        n_queries = interior.shape[0]
    p = interior[:n_queries]                                       # (B_q, 3)

    with torch.no_grad():
        sl_single = model.encode({
            "surface_points": surface,
            "surface_normals": surface_normals,
            "interior_points": spec["shape_ctx"]["interior_points"].to(device),
        })
        shape_latent = ShapeLatent(tokens=sl_single.tokens.expand(n_queries, -1, -1))

        N_s = surface.shape[1]
        surf_pts = surface.expand(n_queries, -1, -1)
        surf_normals = surface_normals.expand(n_queries, -1, -1)
        weight = torch.full(
            (n_queries, N_s), total_area / N_s,
            device=device, dtype=p.dtype,
        )

        log_K_tilde = model.log_kernel(p, surf_pts, surf_normals, shape_latent)
        log_w = torch.log(weight + 1e-30)
        log_Z = torch.logsumexp(log_K_tilde + log_w, dim=-1)             # (B_q,)
        Z = torch.exp(log_Z)
        K = torch.exp(log_K_tilde)
        K_norm = K / Z.unsqueeze(-1)

    out: dict = {
        "shape_id": sol_path.parent.parent.name,
        "n_queries": int(n_queries),
        "log_Z_mean": float(log_Z.mean().item()),
        "log_Z_max_abs": float(log_Z.abs().max().item()),
    }
    for kind in BC_KINDS:
        with torch.no_grad():
            h_at_surf = h_eval(surf_pts, kind)                    # (B_q, N_s)
            u_pred = (weight * K_norm * h_at_surf).sum(dim=-1)    # (B_q,)
            u_true = h_eval(p, kind)
        out[f"rel_l2_{kind}"] = relative_l2_t(u_pred, u_true)
    return out


def main() -> int:
    pa = argparse.ArgumentParser()
    pa.add_argument("--ckpt", required=True, type=str)
    pa.add_argument("--mcb-root", type=str,
                    default=os.environ.get("MCB_ROOT", "data/MCB_benchmark"))
    pa.add_argument("--category", type=str, default="fitting")
    pa.add_argument("--split", type=str, default="known_shape_known_prob",
                    choices=["known_shape_known_prob", "known_shape_unknown_prob",
                             "unknown_shape_known_prob", "unknown_shape_unknown_prob"])
    pa.add_argument("--n-shapes", type=int, default=10)
    pa.add_argument("--n-queries", type=int, default=256)
    pa.add_argument("--grid-size", type=int, default=32)
    pa.add_argument("--n-surface", type=int, default=2000)
    pa.add_argument("--n-interior", type=int, default=512)
    pa.add_argument("--seed", type=int, default=0,
                    help="seed of the per-shape surface samples and anchors (default 0; -1 = unseeded)")
    pa.add_argument("--out", type=str, default=None)
    args = pa.parse_args()
    seed = None if args.seed < 0 else args.seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mcb-laplace] loading {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    print(f"[mcb-laplace] step={ckpt.get('step')}  normalize={cfg['kernel']['normalize']}  "
          f"category={args.category}  split={args.split}")

    model = HarmonicMeasureField(
        encoder_cfg=cfg["encoder"],
        kernel_head_cfg=cfg["kernel_head"],
        normalize=cfg["kernel"]["normalize"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    idx = MCBSolDataset(
        mcb_root=Path(args.mcb_root),
        categories=[args.category],
        split_name=args.split,
    )
    seen: set = set()
    selected = []
    for cat, sid, path in idx._entries:
        if sid in seen:
            continue
        seen.add(sid)
        selected.append((cat, sid, path))
        if len(selected) >= args.n_shapes:
            break

    print(f"[mcb-laplace] evaluating {len(selected)} unique shapes")
    results = []
    skipped_err = 0
    t0 = time.time()
    for i, (cat, sid, path) in enumerate(selected):
        try:
            r = evaluate_one(
                model=model, sol_path=path,
                grid_size=args.grid_size, n_surface=args.n_surface,
                n_interior=args.n_interior, n_queries=args.n_queries,
                device=device, seed=seed,
            )
        except Exception as e:
            skipped_err += 1
            print(f"[mcb-laplace] {sid}: FAIL {type(e).__name__} {e!s:.120}")
            continue
        results.append(r)
        el = time.time() - t0
        line = f"[mcb-laplace] [{i+1}/{len(selected)}] shape={sid[:12]} log_Z={r['log_Z_mean']:+.3f}"
        for k in BC_KINDS:
            line += f"  {k}={r[f'rel_l2_{k}']:.3f}"
        line += f"  ({el:.1f}s)"
        print(line)

    el = time.time() - t0
    print(f"\n[mcb-laplace] DONE: {len(results)} shapes  "
          f"skipped_err={skipped_err}  elapsed={el:.1f}s")
    if not results:
        return 1

    print("\n--- log_Z statistics ---")
    lz = np.array([r["log_Z_mean"] for r in results])
    print(f"log_Z_mean across shapes: mean={lz.mean():+.4f}  median={np.median(lz):+.4f}  "
          f"|max|={np.max(np.abs(lz)):+.4f}")

    print("\n--- aggregate rel-L2 per BC kind ---")
    print(f"{'BC':<14} {'mean':>10} {'median':>10} {'95th':>10} {'min':>10} {'max':>10}")
    for kind in BC_KINDS:
        vals = np.array([r[f"rel_l2_{kind}"] for r in results])
        print(
            f"{kind:<14} "
            f"{vals.mean():>10.4f} "
            f"{np.median(vals):>10.4f} "
            f"{np.percentile(vals, 95):>10.4f} "
            f"{vals.min():>10.4f} "
            f"{vals.max():>10.4f}"
        )

    if args.out:
        agg = {k: float(np.mean([r[f"rel_l2_{k}"] for r in results])) for k in BC_KINDS}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "ckpt": args.ckpt, "category": args.category, "split": args.split, "seed": seed,
            "n_evaluated": len(results), "skipped_err": skipped_err,
            "mean_rel_l2": agg, "per_shape": results,
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
