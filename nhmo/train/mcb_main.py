"""Per-category MCB-B kernel training (Stage 1 of the 3D pipeline).

Subclasses `Trainer` with `MCBTrainer`, which round-robins over the MCB-B
training shapes of one category (tet boundary only: the training path never
reads sol / source_term / bd_* fields) and supervises K_theta with Walk-on-Spheres
exit samples (L_NLL) plus L_MV, L_BL and L_Z.

    python -m nhmo.train.mcb_main --config configs/mcb/nut.yaml \\
        --warm-start checkpoints/3d/init/kernel3d_init_warmup.pt

Warm-start policy:
  --resume PATH   full checkpoint load (model + optim + RNG + step); continues
                  an interrupted run.
  --warm-start P  load model weights only, reset optimizer and step counter.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from nhmo.data.mcb_loader import MCBSolDataset, MCBSolShapeGenerator, MCBWoSDataset
from nhmo.train.config import load_config
from nhmo.train.trainer import Trainer


class MCBTrainer(Trainer):
    """Trainer subclass wiring MCB sol.npz round-robin dataset."""

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self._rebuild_mcb_data(cfg)

    def _rebuild_mcb_data(self, cfg: dict) -> None:
        ds_cfg = cfg.get("data", {})
        mcb_root = Path(
            os.environ.get(
                "MCB_ROOT",
                ds_cfg.get("mcb_root", "data/MCB_benchmark"),
            )
        )
        if not (mcb_root / "splits").exists():
            raise FileNotFoundError(
                f"MCB splits dir not found under {mcb_root}. Set MCB_ROOT or "
                "cfg.data.mcb_root."
            )

        categories = list(ds_cfg.get("categories", ["fitting", "gear", "motor", "nut", "screws_and_bolts"]))
        split_name = str(ds_cfg.get("split", "known_shape_known_prob"))
        n_shapes = int(ds_cfg.get("n_shapes", 0))  # 0 = use all unique shapes

        idx_dataset = MCBSolDataset(
            mcb_root=mcb_root,
            categories=categories,
            split_name=split_name,
        )
        entries = list(idx_dataset._entries)
        if n_shapes > 0:
            entries = entries[:n_shapes]
        if len(entries) == 0:
            raise RuntimeError(
                f"MCBSolDataset returned 0 entries for {categories=} {split_name=}"
            )

        print(
            f"[MCBTrainer] building {len(entries)} MCBSolShapeGenerators from "
            f"categories={categories} split={split_name}",
            flush=True,
        )
        import time as _t
        t_setup = _t.time()
        gens = []
        for i, (_cat, _sid, path) in enumerate(entries):
            gens.append(
                MCBSolShapeGenerator(
                    sol_npz_path=path,
                    grid_size=int(ds_cfg.get("grid_size", 64)),
                    n_surface=int(ds_cfg.get("n_surface", 2000)),
                    n_interior_anchors=int(ds_cfg.get("n_interior_anchors", 512)),
                    device=str(self.device),
                )
            )
            if (i + 1) % 25 == 0 or i == len(entries) - 1:
                elapsed = _t.time() - t_setup
                avg = elapsed / (i + 1)
                eta = avg * (len(entries) - (i + 1))
                print(
                    f"[MCBTrainer] generators built: {i+1}/{len(entries)}  "
                    f"elapsed={elapsed:.0f}s  avg={avg:.1f}s/shape  eta={eta:.0f}s",
                    flush=True,
                )

        rng = np.random.RandomState(int(cfg.get("seed", 42)))

        def round_robin_gen():
            return gens[int(rng.randint(len(gens)))]()

        self.dataset = MCBWoSDataset(
            shape_generator=round_robin_gen,
            wos_sampler=self.wos_sampler,
            n_queries=int(ds_cfg.get("n_queries", 16)),
            n_wos_hits=int(ds_cfg.get("n_wos_hits_per_query", 4)),
            n_surface=int(ds_cfg.get("n_surface", 2000)),
            virtual_length=int(cfg.get("optim", {}).get("total_steps", 100000)),
        )
        self.loader = DataLoader(
            self.dataset, batch_size=1, shuffle=False, num_workers=0,
            collate_fn=lambda batch_list: batch_list[0],
        )

    def warm_start_from(self, path: str) -> None:
        """Load model weights only; reset optimizer + step counter."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.start_step = 0
        src_step = ckpt.get("step", "?")
        print(f"[MCBTrainer] warm-started model weights from {path} (src step {src_step})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mcb_main.yaml")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override cfg.optim.total_steps")
    parser.add_argument("--resume", type=str, default=None,
                        help="Full resume (model + optim + RNG + step)")
    parser.add_argument("--warm-start", type=str, default=None,
                        help="Warm-start model weights only from a pretrain/warmup ckpt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    trainer = MCBTrainer(cfg)
    if args.warm_start is not None:
        trainer.warm_start_from(args.warm_start)
    trainer.fit(n_steps=args.steps, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
