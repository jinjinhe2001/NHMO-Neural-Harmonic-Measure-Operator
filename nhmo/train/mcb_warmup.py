"""3D kernel warm-up on MCB-B between sphere pretraining and the per-category stage.

Starting from the sphere-pretrained kernel (nhmo.train.pretrain_analytic), run
2000 steps on the first two shapes of the MCB-B fitting training split with an
L_MV-dominant objective and soft normalization:
    lambda_NLL = 0.3, lambda_MV = 3.0, lambda_BL = off, lambda_Z = 1.0,
    linear LR ramp 1e-5 -> 3e-4, gradient clipping at 0.5.
Fixed loss weights (no step schedule). This stage prevents the NLL / boundary
peak collapse that a direct switch from spheres to MCB shapes produced.

The released checkpoint checkpoints/3d/init/kernel3d_init_warmup.pt is the output
of this stage; every per-category kernel except fitting starts from it
(fitting starts from a 44k-step fitting pre-run, see configs/mcb/fitting_pre.yaml).

    python -m nhmo.train.mcb_warmup --config configs/mcb_warmup.yaml \\
        --init runs/pretrain_sphere_10k/checkpoint_step_10000.pt
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from nhmo.core.kernel import HarmonicMeasureField, ShapeLatent
from nhmo.data.mcb_loader import MCBSolDataset, MCBSolShapeGenerator, MCBWoSDataset
from nhmo.geometry.sphere_mc import WoSHitSampler
from nhmo.losses.registry import LossRegistry
from nhmo.train.config import load_config
from nhmo.train.trainer import _build_param_groups, _move_to_device


def main() -> int:
    pa = argparse.ArgumentParser()
    pa.add_argument("--config", default="configs/mcb_warmup.yaml")
    pa.add_argument("--init", required=True, help="sphere-pretrained kernel checkpoint")
    pa.add_argument("--mcb-root", default=os.environ.get("MCB_ROOT", "data/MCB_benchmark"))
    pa.add_argument("--category", default="fitting")
    pa.add_argument("--n-shapes", type=int, default=2)
    args = pa.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda")
    torch.manual_seed(int(cfg["seed"]))
    run_dir = Path(cfg["log"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    idx_dataset = MCBSolDataset(mcb_root=Path(args.mcb_root), categories=[args.category],
                                split_name="known_shape_known_prob")
    gens = [MCBSolShapeGenerator(sol_npz_path=idx_dataset._entries[i][2],
                                 grid_size=cfg["data"]["grid_size"], n_surface=cfg["data"]["n_surface"],
                                 n_interior_anchors=cfg["data"]["n_interior_anchors"], device=str(device))
            for i in range(args.n_shapes)]
    rng = np.random.RandomState(int(cfg["seed"]))

    def round_robin_gen():
        return gens[int(rng.randint(len(gens)))]()

    wos_sampler = WoSHitSampler(epsilon=float(cfg["wos"]["epsilon"]), max_steps=int(cfg["wos"]["max_steps"]))
    total_steps = int(cfg["optim"]["total_steps"])
    dataset = MCBWoSDataset(shape_generator=round_robin_gen, wos_sampler=wos_sampler,
                            n_queries=cfg["data"]["n_queries"], n_wos_hits=cfg["data"]["n_wos_hits_per_query"],
                            n_surface=cfg["data"]["n_surface"], virtual_length=total_steps)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=lambda batch_list: batch_list[0])

    model = HarmonicMeasureField(encoder_cfg=cfg["encoder"], kernel_head_cfg=cfg["kernel_head"],
                                 normalize=cfg["kernel"]["normalize"]).to(device)
    init = torch.load(args.init, map_location=device, weights_only=False)
    model.load_state_dict(init["model_state_dict"])
    print(f"[warmup] initialized from {args.init} (step {init.get('step')})", flush=True)

    optimizer = torch.optim.AdamW(_build_param_groups(model, cfg["optim"]["weight_decay"]),
                                  lr=cfg["optim"]["lr_start"])
    loss_registry = LossRegistry(cfg["losses"])
    lr_start, lr_end = float(cfg["optim"]["lr_start"]), float(cfg["optim"]["lr_end"])
    grad_clip = float(cfg["optim"]["grad_clip_norm"])
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("")

    t0 = time.time()
    it = iter(loader)
    for step in range(total_steps):
        lr = lr_start + (step / max(total_steps - 1, 1)) * (lr_end - lr_start)
        for g in optimizer.param_groups:
            g["lr"] = lr
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = _move_to_device(batch, device)
        sl = model.encode(batch["shape_ctx"])
        batch["shape_latent"] = ShapeLatent(tokens=sl.tokens.expand(batch["p_nll"].shape[0], -1, -1))
        losses = loss_registry(batch, model)
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        rec = {k: float(v.item()) for k, v in losses.items() if hasattr(v, "item")}
        rec.update(step=step, lr=lr, t=time.time() - t0)
        with metrics_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if step % 100 == 0:
            print(f"[warmup] step={step} total={rec['total']:.4f} lr={lr:.2e}", flush=True)
        if step > 0 and step % int(cfg["log"].get("checkpoint_every", 500)) == 0:
            torch.save({"step": step, "model_state_dict": model.state_dict(), "cfg": cfg},
                       run_dir / f"checkpoint_step_{step}.pt")
    torch.save({"step": total_steps, "model_state_dict": model.state_dict(), "cfg": cfg},
               run_dir / f"checkpoint_step_{total_steps}.pt")
    print(f"[warmup] done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
