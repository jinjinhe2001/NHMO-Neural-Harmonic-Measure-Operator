"""Train the 3D Poisson source lift v_phi on top of a FROZEN K_theta (MCB-B).

Objective per (Omega, f, h, u) tuple from an NGF sol.npz:
    u_pred(p) = v_phi(p; Omega, f) + <h, K_theta(p, .; Omega)>
    loss = MSE(u_pred, u_true) at --n-queries random interior tet vertices

K_theta stays frozen (trained for the harmonic measure by nhmo.train.mcb_main).
Only the source pieces are trained: SourceTokenEncoder + SourceLiftHead.

Released lifts used d_model 192 (from the kernel), --n-source-slices 256
(fitting: 384), --n-cross-layers 3, --n-source-layers 2, lr 3e-4 with 200
warm-up steps and cosine decay to 1e-5, followed by warm-start rounds
(--warm-start previous checkpoint, fresh schedule); see README.

    python -m nhmo.train.poisson_lift_train \\
        --kernel-ckpt checkpoints/3d/kernel_nut.pt --category nut \\
        --n-source-slices 256 --n-cross-layers 3 --total-steps 20000 \\
        --checkpoint-every 2000 --out runs/lift_nut_round0
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from nhmo.core.fast_inference import stable_seed
from nhmo.core.kernel import HarmonicMeasureField, ShapeLatent
from nhmo.core.source_lift import PoissonLiftModule
from nhmo.data.mcb_loader import (
    MCBSolDataset, MCBSolShapeGenerator, load_mcb_eval_fields,
)
from nhmo.geometry.sdf import trilinear_interp


def _wos_to_model(p_wos: Tensor) -> Tensor:
    """[0,1]³ → [-1,1]³."""
    return p_wos * 2.0 - 1.0


def _model_to_wos(p_model: Tensor) -> Tensor:
    return (p_model + 1.0) * 0.5


class _PoissonShapeRecord:
    """Cached per-shape data: shape geometry + all (BC, u_true) pairs."""

    def __init__(
        self,
        shape_id: str,
        sol_paths: list[Path],
        gen: MCBSolShapeGenerator,
        v_tet_orig: np.ndarray,
        center: np.ndarray,
        scale: float,
        bd_inds: np.ndarray,
        device: torch.device,
        n_source_probe: int = 1024,
    ) -> None:
        self.shape_id = shape_id
        self.sol_paths = sol_paths
        self.gen = gen
        self.v_tet_orig = v_tet_orig
        self.center = center
        self.scale = scale
        self.bd_inds = bd_inds
        self.device = device
        # Pre-encode the shape (one-time cost). The kernel's encoder runs once.
        spec = gen()
        self.surface = spec["shape_ctx"]["surface_points"].to(device)
        self.surface_normals = spec["shape_ctx"]["surface_normals"].to(device)
        self.interior_pool = spec["shape_ctx"]["interior_points"].to(device)
        self.sdf_grid_wos = spec["sdf_grid_wos"].to(device)
        self.total_area = float(spec.get("total_area_model_frame", 4.0))
        # Random subsample of v_tet for source probing (saves memory)
        self.n_source_probe = min(n_source_probe, len(v_tet_orig))
        rng = np.random.RandomState(stable_seed(shape_id, 0))
        self.probe_idx = rng.choice(len(v_tet_orig), self.n_source_probe, replace=False)
        # In model frame
        v_tet_model = (v_tet_orig - center) * scale
        self.probe_pos_model = torch.from_numpy(
            v_tet_model[self.probe_idx].astype(np.float32)
        ).to(device)


class PoissonLiftDataset(Dataset):
    """Yields (shape_record, sol_npz_path) for training.

    Each __getitem__ returns one (shape, BC) pair. Workers shouldn't share
    cached records, so we keep records in the dataset object.
    """

    def __init__(
        self,
        records: list[_PoissonShapeRecord],
        virtual_length: int = 1000,
    ) -> None:
        # Flatten (shape, BC) pairs.
        self.entries: list[tuple[_PoissonShapeRecord, Path]] = []
        for r in records:
            for sp in r.sol_paths:
                self.entries.append((r, sp))
        self.virtual_length = virtual_length

    def __len__(self) -> int:
        return self.virtual_length

    def __getitem__(self, idx: int) -> tuple[_PoissonShapeRecord, Path]:
        return self.entries[idx % len(self.entries)]


def build_records(
    category: str, mcb_root: Path, split_name: str,
    n_shapes: int, gen_kwargs: dict, device: torch.device,
) -> list[_PoissonShapeRecord]:
    idx_dataset = MCBSolDataset(
        mcb_root=mcb_root, categories=[category], split_name=split_name,
    )
    # MCBSolDataset dedupes by shape (one path per unique shape). For
    # Poisson supervised training we want ALL BC configs per shape, so
    # enumerate the on-disk directory directly.
    by_shape: dict[str, list[Path]] = {}
    for cat, sid, _path in idx_dataset._entries:
        shape_dir = mcb_root / "ngf_solutions" / cat / sid
        if not shape_dir.exists():
            continue
        bc_paths: list[Path] = []
        for bc_dir in sorted(shape_dir.iterdir()):
            if not bc_dir.is_dir():
                continue
            sol_npz = bc_dir / "sol.npz"
            if sol_npz.exists():
                bc_paths.append(sol_npz)
        if bc_paths:
            by_shape[sid] = bc_paths

    records: list[_PoissonShapeRecord] = []
    skipped = 0
    for i, sid in enumerate(sorted(by_shape.keys())):
        if len(records) >= n_shapes:
            break
        paths = by_shape[sid]
        try:
            gen = MCBSolShapeGenerator(sol_npz_path=paths[0], **gen_kwargs)
            _ = gen()  # validate
        except Exception as e:
            skipped += 1
            continue
        # Load v_tet, bd_inds from the first sol.npz of this shape.
        f0 = load_mcb_eval_fields(str(paths[0]))
        v_tet = np.asarray(f0.v_tet)
        bd_inds = np.asarray(f0.bd_v_inds)
        bbox_min = v_tet.min(axis=0).astype(np.float64)
        bbox_max = v_tet.max(axis=0).astype(np.float64)
        center = 0.5 * (bbox_min + bbox_max)
        half = 0.5 * (bbox_max - bbox_min)
        scale = 0.9 / max(float(np.max(half)), 1e-12)
        rec = _PoissonShapeRecord(
            shape_id=sid, sol_paths=paths, gen=gen,
            v_tet_orig=v_tet, center=center, scale=scale,
            bd_inds=bd_inds, device=device,
        )
        records.append(rec)
    print(f"[lift-train] built {len(records)} shape records, skipped {skipped}")
    return records


def encode_shape_latent(model: HarmonicMeasureField, rec: _PoissonShapeRecord) -> Tensor:
    """Run the FROZEN encoder once per shape; return (1, M, d)."""
    with torch.no_grad():
        sl = model.encode({
            "surface_points": rec.surface,
            "surface_normals": rec.surface_normals,
            "interior_points": rec.interior_pool,
        })
    return sl.tokens                                                    # (1, M, d)


def compute_u_h_pred(
    model: HarmonicMeasureField,
    rec: _PoissonShapeRecord,
    shape_latent_tokens: Tensor,                                         # (1, M, d)
    p_model: Tensor,                                                    # (Q, 3)
    h_at_surf: Tensor,                                                  # (N_s,)
) -> Tensor:                                                              # (Q,)
    """K_θ-based boundary integral, evaluated at p with frozen kernel."""
    Q = p_model.shape[0]
    surf = rec.surface.expand(Q, -1, -1)
    surf_n = rec.surface_normals.expand(Q, -1, -1)
    sl = ShapeLatent(tokens=shape_latent_tokens.expand(Q, -1, -1))
    N_s = surf.shape[1]
    with torch.no_grad():
        log_K_tilde = model.log_kernel(p_model, surf, surf_n, sl)
        weight = torch.full((Q, N_s), rec.total_area / N_s,
                            device=p_model.device, dtype=p_model.dtype)
        log_w = torch.log(weight + 1e-30)
        log_Z = torch.logsumexp(log_K_tilde + log_w, dim=-1)
        K_norm = torch.exp(log_K_tilde) / torch.exp(log_Z).unsqueeze(-1)
        u_h = (weight * K_norm * h_at_surf.unsqueeze(0)).sum(dim=-1)
    return u_h


def sample_query_and_targets(
    rec: _PoissonShapeRecord,
    sol_path: Path,
    n_queries: int,
    device: torch.device,
) -> dict:
    """Sample n_queries interior points and gather (p, sdf_at_p, f, h, u_true).

    Also returns f_at_p in MODEL-frame units (= f_orig / scale²) so PDE-loss
    can enforce Δ_model v_φ ≈ f_model.
    """
    fields = load_mcb_eval_fields(str(sol_path))
    sol = np.asarray(fields.sol).reshape(-1)
    src = np.asarray(fields.source_term).reshape(-1)
    bd_vals = np.asarray(fields.bd_v_vals)
    v_tet = rec.v_tet_orig

    bd_mask = np.zeros(len(v_tet), dtype=bool)
    bd_mask[rec.bd_inds] = True
    interior_idx = np.where(~bd_mask)[0]
    n_q = min(n_queries, len(interior_idx))
    rng = np.random.RandomState(int(time.time() * 1000) & 0xFFFFFFFF)
    sel = rng.choice(interior_idx, n_q, replace=False)

    p_orig = v_tet[sel]
    p_model_np = (p_orig - rec.center) * rec.scale
    p_model = torch.from_numpy(p_model_np.astype(np.float32)).to(device)
    p_wos = torch.from_numpy(((p_model_np + 1.0) * 0.5).astype(np.float32)).to(device)

    sdf_at_p = trilinear_interp(rec.sdf_grid_wos.unsqueeze(0), p_wos.unsqueeze(0)).squeeze(0)

    bd_pts_model = (v_tet[rec.bd_inds] - rec.center) * rec.scale
    surf_np = rec.surface.squeeze(0).cpu().numpy()
    diffs = surf_np[:, None, :] - bd_pts_model[None, :, :]
    nearest = np.argmin((diffs * diffs).sum(axis=-1), axis=-1)
    h_at_surf = torch.from_numpy(bd_vals[nearest].astype(np.float32)).to(device)

    f_probe = torch.from_numpy(src[rec.probe_idx].astype(np.float32)).to(device)
    # f at the QUERY points themselves (for PDE consistency loss). Convert to
    # model frame units: Δ_model = (1/scale²)·Δ_orig ⇒ f_model = f_orig/scale².
    f_at_p_orig = src[sel].astype(np.float32)
    f_at_p_model = torch.from_numpy(f_at_p_orig / (rec.scale * rec.scale)).to(device)

    u_true = torch.from_numpy(sol[sel].astype(np.float32)).to(device)

    return {
        "p_model": p_model,
        "sdf_at_p": sdf_at_p,
        "h_at_surf": h_at_surf,
        "f_probe": f_probe,
        "f_probe_pos_model": rec.probe_pos_model,
        "f_at_p_model": f_at_p_model,
        "u_true": u_true,
        "p_wos": p_wos,
    }


def main() -> int:
    pa = argparse.ArgumentParser()
    pa.add_argument("--kernel-ckpt", required=True, type=str)
    pa.add_argument("--mcb-root", default=os.environ.get("MCB_ROOT", "data/MCB_benchmark"))
    pa.add_argument("--category", required=True)
    pa.add_argument("--split", default="known_shape_known_prob")
    pa.add_argument("--n-shapes", type=int, default=200)
    pa.add_argument("--n-queries", type=int, default=64)
    pa.add_argument("--total-steps", type=int, default=5000)
    pa.add_argument("--lr", type=float, default=3e-4)
    pa.add_argument("--lr-min", type=float, default=1e-5)
    pa.add_argument("--warmup-steps", type=int, default=200)
    pa.add_argument("--checkpoint-every", type=int, default=500)
    pa.add_argument("--log-every", type=int, default=50)
    pa.add_argument("--n-source-probe", type=int, default=1024)
    pa.add_argument("--n-source-slices", type=int, default=32,
                    help="Soft-attention slice count in SourceTokenEncoder")
    pa.add_argument("--n-cross-layers", type=int, default=2,
                    help="Cross-attention layers in SourceLiftHead")
    pa.add_argument("--n-source-layers", type=int, default=2,
                    help="Self-attention layers over source slice tokens")
    pa.add_argument("--n-heads", type=int, default=4)
    pa.add_argument("--grid-size", type=int, default=32)
    pa.add_argument("--n-surface", type=int, default=2000)
    pa.add_argument("--n-interior", type=int, default=512)
    pa.add_argument("--out", required=True, type=str)
    pa.add_argument("--warm-start", default=None, type=str,
                    help="Path to a previous lift checkpoint. Resumes training "
                         "from those weights with a fresh LR schedule. The "
                         "warm-start ckpt's lift_cfg overrides the CLI "
                         "architecture flags so the load works.")
    pa.add_argument("--device", default="cuda")
    args = pa.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[lift-train] loading frozen kernel from {args.kernel_ckpt}")
    ckpt = torch.load(args.kernel_ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    kernel = HarmonicMeasureField(
        encoder_cfg=cfg["encoder"], kernel_head_cfg=cfg["kernel_head"],
        normalize=cfg["kernel"]["normalize"],
    ).to(device)
    kernel.load_state_dict(ckpt["model_state_dict"])
    kernel.eval()
    for p in kernel.parameters():
        p.requires_grad_(False)
    d_model = int(cfg["kernel_head"]["d_model"])
    print(f"[lift-train] kernel d_model={d_model}; freezing.")

    print(f"[lift-train] preparing shape records: category={args.category} split={args.split}")
    gen_kwargs = dict(
        grid_size=args.grid_size, n_surface=args.n_surface,
        n_interior_anchors=args.n_interior, device=str(device),
    )
    records = build_records(
        args.category, Path(args.mcb_root), args.split,
        args.n_shapes, gen_kwargs, device,
    )
    if not records:
        raise RuntimeError("no records built")

    # Pre-encode shape_latents for each record (frozen kernel, fixed value).
    print("[lift-train] pre-encoding shape latents (frozen kernel)…")
    for rec in records:
        rec.shape_latent_tokens = encode_shape_latent(kernel, rec)      # (1, M, d)
        rec.shape_latent_tokens.requires_grad_(False)
        rec.scale_to_orig = 1.0 / rec.scale                              # |orig|/|model|

    if args.warm_start:
        print(f"[lift-train] warm-starting lift from {args.warm_start}")
        warm = torch.load(args.warm_start, map_location="cpu", weights_only=False)
        ws_cfg = warm.get("lift_cfg", None)
        if ws_cfg is None:
            raise RuntimeError(
                "warm-start ckpt has no lift_cfg — cannot reconstruct architecture. "
                "Use a checkpoint written by this trainer."
            )
        lift_cfg = ws_cfg
        print(f"[lift-train] lift_cfg from warm-start: {lift_cfg}")
        lift = PoissonLiftModule(**lift_cfg).to(device)
        lift.load_state_dict(warm["lift_state_dict"])
        print(f"[lift-train] loaded weights from warm-start step {warm.get('step', '?')}")
    else:
        lift_cfg = {
            "d_model": d_model,
            "n_source_slices": args.n_source_slices,
            "n_source_layers": args.n_source_layers,
            "n_cross_layers": args.n_cross_layers,
            "n_heads": args.n_heads,
        }
        print(f"[lift-train] lift_cfg = {lift_cfg}")
        lift = PoissonLiftModule(**lift_cfg).to(device)
    opt = torch.optim.AdamW(lift.parameters(), lr=args.lr, weight_decay=0.01)

    def cosine_lr(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * step / max(args.warmup_steps, 1)
        t = (step - args.warmup_steps) / max(args.total_steps - args.warmup_steps, 1)
        t = max(0.0, min(1.0, t))
        return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1.0 + np.cos(np.pi * t))

    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.write_text("")
    rng = np.random.RandomState(args.n_shapes + 1)

    t0 = time.time()
    total_pairs = sum(len(r.sol_paths) for r in records)
    print(f"[lift-train] {len(records)} shapes × ~{total_pairs // max(len(records),1)} BCs "
          f"= {total_pairs} (shape, BC) pairs available")

    for step in range(args.total_steps):
        lr = cosine_lr(step)
        for g in opt.param_groups:
            g["lr"] = lr

        # Pick random (shape, BC) pair
        rec = records[rng.randint(len(records))]
        sol_path = rec.sol_paths[rng.randint(len(rec.sol_paths))]
        batch = sample_query_and_targets(
            rec, sol_path, args.n_queries, device,
        )

        # Source tokens (one per (Ω, f) pair). Could cache per-pair but on-the-fly is fine.
        src_tokens = lift.encode_source(
            batch["f_probe_pos_model"].unsqueeze(0),
            batch["f_probe"].unsqueeze(0),
        )                                                                # (1, M_src, d)

        Q = batch["p_model"].shape[0]
        # Tile shape and source tokens to Q (one per query).
        sl_tokens_q = rec.shape_latent_tokens.expand(Q, -1, -1)
        src_tokens_q = src_tokens.expand(Q, -1, -1)

        v_pred = lift(
            p=batch["p_model"], shape_latent_tokens=sl_tokens_q,
            source_tokens=src_tokens_q, sdf_at_p=batch["sdf_at_p"],
        )                                                                # (Q,)

        u_f_pred = v_pred

        # Boundary integral via FROZEN kernel
        u_h_pred = compute_u_h_pred(
            kernel, rec, rec.shape_latent_tokens,
            batch["p_model"], batch["h_at_surf"],
        )                                                                # (Q,)

        u_pred = u_f_pred + u_h_pred
        u_true = batch["u_true"]

        loss_mse = F.mse_loss(u_pred, u_true)
        denom = u_true.norm() + 1e-6
        rel_l2 = (u_pred - u_true).norm() / denom

        loss = loss_mse

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lift.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0 or step == args.total_steps - 1:
            log = {
                "step": step,
                "lr": float(lr),
                "loss_total": float(loss.item()),
                "loss_mse": float(loss_mse.item()),
                "rel_l2": float(rel_l2.item()),
                "u_pred_mean": float(u_pred.mean().item()),
                "u_true_mean": float(u_true.mean().item()),
                "v_pred_abs_mean": float(v_pred.abs().mean().item()),
                "u_h_pred_abs_mean": float(u_h_pred.abs().mean().item()),
                "elapsed_s": time.time() - t0,
            }
            print(
                f"[lift] step={step} lr={lr:.2e} loss={loss.item():.4e} "
                f"mse={loss_mse:.4e} "
                f"rel_l2={rel_l2:.3f} v={log['v_pred_abs_mean']:.3f} "
                f"u_h={log['u_h_pred_abs_mean']:.3f}",
                flush=True,
            )
            with open(metrics_path, "a") as fh:
                fh.write(json.dumps(log) + "\n")

        if step > 0 and step % args.checkpoint_every == 0:
            torch.save({
                "step": step,
                "lift_state_dict": lift.state_dict(),
                "lift_cfg": lift_cfg,
                "kernel_ckpt": args.kernel_ckpt,
                "category": args.category,
                "use_local_baseline": False,
            }, out_dir / f"checkpoint_step_{step}.pt")

    torch.save({
        "step": args.total_steps,
        "lift_state_dict": lift.state_dict(),
        "lift_cfg": lift_cfg,
        "kernel_ckpt": args.kernel_ckpt,
        "category": args.category,
        "use_local_baseline": False,
    }, out_dir / f"checkpoint_step_{args.total_steps}.pt")
    print(f"[lift-train] DONE — total {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
