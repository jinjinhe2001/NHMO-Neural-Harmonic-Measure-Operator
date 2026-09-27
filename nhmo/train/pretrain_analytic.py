"""Stage-0 analytic-domain pretraining (v2 plan §8).

Subclasses `Trainer` with `AnalyticPretrainTrainer`, overriding `_build_data`
to use `RandomSphereShapeGenerator` + `AnalyticBallDataset` with
`SphereHarmonicMeasure`. Provides the checkpoint that seeds Phase 7's main
MCB training.

Scope: sphere only. Half-space
and cube deferred to Phase 6.1 with specific revisit triggers documented
in TODO_POST_PHASE_6.md.
"""
from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from nhmo.data.wos_dataset import (
    AnalyticBallDataset,
    RandomSphereShapeGenerator,
    collate_wos_batches,
)
from nhmo.train.config import load_config
from nhmo.train.trainer import Trainer, _build_wos_sampler


class AnalyticPretrainTrainer(Trainer):
    """Trainer subclass wiring Phase 6's analytic pretraining dataset."""

    def __init__(self, cfg: dict) -> None:
        # Call the parent __init__ — it builds model, optim, loss registry.
        # Data pipeline is overwritten afterward with the analytic variant.
        super().__init__(cfg)
        self._rebuild_analytic_data(cfg)

    def _rebuild_analytic_data(self, cfg: dict) -> None:
        ds_cfg = cfg.get("data", {})
        curr_cfg = cfg.get("pretrain_curriculum", {})
        gen = RandomSphereShapeGenerator(
            grid_size=int(ds_cfg.get("grid_size", 64)),
            n_surface=int(ds_cfg.get("n_surface", 512)),
            n_interior_anchors=int(ds_cfg.get("n_interior_anchors", 256)),
            radius_base=float(curr_cfg.get("radius_base", 0.3)),
            scale_min=float(curr_cfg.get("scale_min", 0.5)),
            scale_max=float(curr_cfg.get("scale_max", 2.0)),
            translate_max=float(curr_cfg.get("translate_max", 0.3)),
            enable_rotation=bool(curr_cfg.get("enable_rotation", True)),
            device=str(self.device),
            seed=int(cfg.get("seed", 42)),
        )
        # Reuse the already-built WoS sampler from parent.
        self.dataset = AnalyticBallDataset(
            shape_generator=gen,
            wos_sampler=self.wos_sampler,
            n_queries=int(ds_cfg.get("n_queries", 16)),
            n_wos_hits=int(ds_cfg.get("n_wos_hits_per_query", 4)),
            n_surface=int(ds_cfg.get("n_surface", 512)),
            virtual_length=int(cfg.get("optim", {}).get("total_steps", 10000)),
        )
        self.loader = DataLoader(
            self.dataset, batch_size=1, shuffle=False, num_workers=0,
            collate_fn=collate_wos_batches,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain_analytic.yaml")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override cfg.optim.total_steps")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    trainer = AnalyticPretrainTrainer(cfg)
    trainer.fit(n_steps=args.steps, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
