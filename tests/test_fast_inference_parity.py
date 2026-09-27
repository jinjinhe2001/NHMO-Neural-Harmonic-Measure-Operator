"""Parity of the optimized 3D inference path against the reference path on one shape.

Checks, on the first test shape of one MCB-B category (default: nut):
  1. K_eff build: folded vs reference, fp32. Expected bitwise identical.
  2. Lift forward: query-folded vs per-query reference, fp32. Expected equal to
     fp32 rounding (different GEMM/attention kernel shapes, not bitwise).
  3. Full prediction on 4 problems: cached fast solve vs the reference
     evaluation arithmetic, same encode (isolates the algebraic refactoring).
  4. GPU SDF grid vs trimesh grid: sign disagreements and |difference|.
  5. End-to-end with the GPU SDF grid (same surface samples and anchors):
     change of the prediction and of the rel-L2 against the FEM reference.

Needs a CUDA GPU, the MCB-B data (env MCB_ROOT) and the release checkpoints
(env NHMO_CKPT_DIR, containing 3d/kernel_<cat>.pt and 3d/lift_<cat>.pt).
Run either as `python -m pytest tests/test_fast_inference_parity.py -s` or
`python tests/test_fast_inference_parity.py [--category nut] [--out parity.json]`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nhmo.core.fast_inference import (  # noqa: E402
    build_keff_folded, build_keff_reference, encode_shape, lift_forward_folded,
    lift_forward_reference, nearest_boundary_vertex, solve_cached, source_probe,
)
from nhmo.data.mcb_loader import load_mcb_eval_fields  # noqa: E402
from nhmo.eval.mcb_lift import evaluate_one_bc, load_models, split_pairs  # noqa: E402
from nhmo.geometry.sdf import trilinear_interp  # noqa: E402
from nhmo.geometry.sdf_gpu import rasterize_sdf_grid_gpu  # noqa: E402


def run_parity(category: str = "nut", seed: int = 0, n_problems: int = 4,
               grid_size: int = 32, n_surface: int = 2000, n_interior: int = 512,
               n_source_probe: int = 1024) -> dict:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    ck = Path(os.environ.get("NHMO_CKPT_DIR", "checkpoints"))
    mcb_root = Path(os.environ.get("MCB_ROOT", "data/MCB_benchmark"))
    kernel, lift, _, _ = load_models(str(ck / "3d" / f"kernel_{category}.pt"),
                                     str(ck / "3d" / f"lift_{category}.pt"), device)
    by_shape = split_pairs(mcb_root, category, "unknown_shape_unknown_prob")
    sid = next(iter(by_shape))
    probs = by_shape[sid][:n_problems]
    out: dict = {"category": category, "shape": sid, "n_problems": len(probs),
                 "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}

    with torch.no_grad():
        ref = encode_shape(kernel, probs[0], grid_size, n_surface, n_interior, device, seed=seed)
        p_model = torch.from_numpy(((ref.v_tet[ref.interior_inds] - ref.center) * ref.scale)
                                   .astype(np.float32)).to(device)
        p_wos = (p_model + 1.0) * 0.5
        sdf_at_p = trilinear_interp(ref.sdf_grid_wos.unsqueeze(0), p_wos.unsqueeze(0)).squeeze(0)
        out["n_queries"] = int(p_model.shape[0])

        # 1. K_eff
        k_ref = build_keff_reference(kernel, p_model, ref.surface, ref.surface_normals,
                                     ref.shape_latent, ref.total_area)
        k_fast = build_keff_folded(kernel, p_model, ref.surface, ref.surface_normals,
                                   ref.shape_latent, ref.total_area)
        out["keff_bitwise_equal"] = bool(torch.equal(k_ref, k_fast))
        out["keff_max_abs_diff"] = float((k_ref - k_fast).abs().max())

        # 2. lift forward
        lift_rel, lift_abs, lift_eq = [], [], []
        for p in probs:
            f = load_mcb_eval_fields(str(p))
            fp, fv = source_probe(ref, f.source_term, n_source_probe, seed)
            src_tok = lift.encode_source(fp.unsqueeze(0), fv.unsqueeze(0))
            v_ref = lift_forward_reference(lift, p_model, ref.shape_latent.tokens, src_tok, sdf_at_p)
            v_fast = lift_forward_folded(lift, p_model, ref.shape_latent.tokens, src_tok, sdf_at_p)
            lift_eq.append(bool(torch.equal(v_ref, v_fast)))
            lift_abs.append(float((v_ref - v_fast).abs().max()))
            lift_rel.append(float((v_ref - v_fast).norm() / (v_ref.norm() + 1e-12)))
        out["lift_bitwise_equal"] = lift_eq
        out["lift_max_abs_diff"] = max(lift_abs)
        out["lift_max_rel_diff"] = max(lift_rel)

        # 3. full prediction: fast cached solve vs reference evaluation (same encode)
        ref.p_model, ref.sdf_at_p, ref.keff = p_model, sdf_at_p, k_fast
        ref.nearest = nearest_boundary_vertex(ref, ref.bd_inds)
        e2e_rel, rl2_ref, rl2_fast = [], [], []
        for p in probs:
            f = load_mcb_eval_fields(str(p))
            r = evaluate_one_bc(kernel, lift, p, ref, -1, n_source_probe, 1024, device, seed)
            u_h, v = solve_cached(lift, ref, f.bd_v_inds, f.bd_v_vals, f.source_term,
                                  n_source_probe=n_source_probe, seed=seed, fast=True)
            gt = torch.from_numpy(np.asarray(f.sol).reshape(-1)[ref.interior_inds]
                                  .astype(np.float32)).to(device)
            u_fast = u_h + v
            rl2_ref.append(r["rel_l2_full"])
            rl2_fast.append(float((u_fast - gt).norm() / gt.norm()))
            # reference prediction reconstructed from the reference path pieces
            u_ref_h = k_ref @ torch.from_numpy(np.asarray(f.bd_v_vals, np.float32)).to(device)[ref.nearest]
            fp, fv = source_probe(ref, f.source_term, n_source_probe, seed)
            v_ref = lift_forward_reference(lift, p_model, ref.shape_latent.tokens,
                                           lift.encode_source(fp.unsqueeze(0), fv.unsqueeze(0)), sdf_at_p)
            u_ref = u_ref_h + v_ref
            e2e_rel.append(float((u_fast - u_ref).norm() / u_ref.norm()))
        out["e2e_pred_max_rel_diff"] = max(e2e_rel)
        out["e2e_rel_l2_reference_eval"] = rl2_ref
        out["e2e_rel_l2_fast"] = rl2_fast
        out["e2e_max_abs_rel_l2_change"] = float(np.max(np.abs(np.array(rl2_ref) - np.array(rl2_fast))))

        # 4. GPU SDF grid vs trimesh grid on the same normalized mesh
        from nhmo.data.mcb_loader import MCBSolShapeGenerator
        gen = MCBSolShapeGenerator(probs[0], grid_size=grid_size, n_surface=n_surface,
                                   n_interior_anchors=n_interior, device=str(device), seed=0)
        sdf_cpu = gen()["sdf_grid_wos"].to(device)
        sdf_gpu = rasterize_sdf_grid_gpu(gen._mesh, grid_size, device)
        d = (sdf_gpu - sdf_cpu).abs().flatten()
        out["sdf_grid_points"] = int(d.numel())
        out["sdf_sign_disagreements"] = int((torch.sign(sdf_gpu) != torch.sign(sdf_cpu)).sum())
        out["sdf_max_abs_diff"] = float(d.max())
        out["sdf_p99_abs_diff"] = float(torch.quantile(d.float(), 0.99))

        # 5. end-to-end with the GPU grid (same surface samples / anchors / latent)
        sdf_at_p_gpu = trilinear_interp(sdf_gpu.unsqueeze(0), p_wos.unsqueeze(0)).squeeze(0)
        ref.sdf_grid_wos, ref.sdf_at_p = sdf_gpu, sdf_at_p_gpu
        e2e_gpu_rel, rl2_gpu = [], []
        for p, u_ref_rl2 in zip(probs, rl2_fast):
            f = load_mcb_eval_fields(str(p))
            u_h, v = solve_cached(lift, ref, f.bd_v_inds, f.bd_v_vals, f.source_term,
                                  n_source_probe=n_source_probe, seed=seed, fast=True)
            gt = torch.from_numpy(np.asarray(f.sol).reshape(-1)[ref.interior_inds]
                                  .astype(np.float32)).to(device)
            rl2_gpu.append(float((u_h + v - gt).norm() / gt.norm()))
        out["e2e_rel_l2_gpu_sdf"] = rl2_gpu
        out["e2e_gpu_sdf_max_abs_rel_l2_change"] = float(np.max(np.abs(np.array(rl2_gpu) - np.array(rl2_fast))))
    return out


def check(out: dict) -> None:
    assert out["keff_bitwise_equal"], f"K_eff not bitwise equal (max diff {out['keff_max_abs_diff']})"
    assert out["lift_max_rel_diff"] < 1e-5, out["lift_max_rel_diff"]
    assert out["e2e_pred_max_rel_diff"] < 1e-5, out["e2e_pred_max_rel_diff"]
    assert out["e2e_max_abs_rel_l2_change"] < 1e-5, out["e2e_max_abs_rel_l2_change"]
    assert out["sdf_sign_disagreements"] <= max(2, out["sdf_grid_points"] // 10000), out["sdf_sign_disagreements"]
    assert out["sdf_p99_abs_diff"] < 1e-5, out["sdf_p99_abs_diff"]


def test_fast_inference_parity():
    import pytest
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    out = run_parity(os.environ.get("NHMO_PARITY_CATEGORY", "nut"))
    print(json.dumps(out, indent=1))
    check(out)


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--category", default="nut")
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--out", default=None)
    a = pa.parse_args()
    res = run_parity(a.category, a.seed)
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    check(res)
    print("PARITY OK")
