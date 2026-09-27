"""Main training loop for v2 (Phase 5).

Builds the full pipeline from a YAML config:
  - Model (`TransolverEncoder` + `HarmonicMeasureField`).
  - Data (`WoSDataset` with `WoSHitSampler` / `MockWoSHitSampler`).
  - Optimizer (`AdamW` with weight-decay separation per §7.3).
  - Scheduler (`cosine_with_warmup`).
  - Loss registry (`LossRegistry`).

Strict Warp requirement: if `warp-lang` is not installed AND the config
does not explicitly enable the mock backend (which is only intended for
unit tests, not training), Trainer raises `EnvironmentError` at init.

Mock mode (`wos.backend = "mock"`) exists ONLY as a test affordance for
smoke tests that need to run without CUDA; production training always
uses `wos.backend = "warp"`.
"""
from __future__ import annotations

import argparse
import gc
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nhmo.core.encoder import TransolverEncoder
from nhmo.core.kernel import HarmonicMeasureField, ShapeLatent
from nhmo.data.wos_dataset import UnitBallShapeGenerator, WoSDataset, collate_wos_batches
from nhmo.losses.registry import LossRegistry
from nhmo.train.config import load_config
from nhmo.train.logging import Logger
from nhmo.train.schedule import cosine_with_warmup, loss_weight_schedule


def _warp_available() -> bool:
    try:
        import warp  # noqa: F401
        return True
    except ImportError:
        return False


def _build_wos_sampler(cfg: dict):
    """Return the production WoSHitSampler. Warp-backed; no fallback.

    Trainer's WoS backend is
    hardcoded to "warp". There is NO mock backend in production. Tests
    that need a CPU-runnable sampler must import MockWoSHitSampler
    directly from tests.fixtures.mock_wos — never via Trainer.

    Accepts `cfg.wos.backend` only if equal to "warp" (default) or
    absent; any other value raises ValueError.
    """
    wos_cfg = cfg.get("wos", {})
    backend = wos_cfg.get("backend", "warp")
    if backend != "warp":
        raise ValueError(
            f"Trainer only supports wos.backend='warp' (got {backend!r}). "
            "Mock backends are not permitted in production training. "
            "For unit tests, import MockWoSHitSampler from "
            "tests.fixtures.mock_wos directly — do not route through Trainer."
        )
    if not _warp_available():
        raise EnvironmentError(
            "Warp not available; Trainer requires GPU + warp-lang. "
            "Run on a CUDA machine with warp-lang installed. For CPU-only loss-math tests "
            "use MockWoSHitSampler directly (not via Trainer)."
        )
    from nhmo.geometry.sphere_mc import WoSHitSampler
    epsilon = float(wos_cfg.get("epsilon", 1e-4))
    max_steps = int(wos_cfg.get("max_steps", 256))
    return WoSHitSampler(epsilon=epsilon, max_steps=max_steps)


def _build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """AdamW param groups per §7.3: Linear.weight decays, biases + LN don't."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or "norm" in name.lower() or "embedding" in name.lower():
            no_decay.append(param)
        elif param.dim() >= 2:
            decay.append(param)
        else:
            no_decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _move_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(device, non_blocking=True) if isinstance(vv, torch.Tensor) else vv
                      for kk, vv in v.items()}
        else:
            out[k] = v
    return out


class Trainer:
    """Trainer for NHMO v2. See module docstring for strict Warp requirement."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        torch.manual_seed(int(cfg.get("seed", 42)))

        # Device. GPU if available + warp backend; CPU only for mock backend.
        backend = cfg.get("wos", {}).get("backend", "warp")
        if backend == "warp" and not torch.cuda.is_available():
            raise EnvironmentError(
                "Warp WoS backend requires CUDA. torch.cuda.is_available() is False."
            )
        self.device = torch.device("cuda" if backend == "warp" else "cpu")

        # --- Model ---
        self.model = HarmonicMeasureField(
            encoder_cfg=cfg["encoder"] if "encoder" in cfg else {k: v for k, v in cfg["encoder"].items() if k != "type"},
            kernel_head_cfg=cfg["kernel_head"],
            normalize=cfg["kernel"]["normalize"],
        ).to(self.device)

        # --- Data ---
        ds_cfg = cfg.get("data", {})
        gen = UnitBallShapeGenerator(
            grid_size=int(ds_cfg.get("grid_size", 64)),
            n_surface=int(ds_cfg.get("n_surface", 512)),
            n_interior_anchors=int(ds_cfg.get("n_interior_anchors", 512)),
            device=str(self.device),
        )
        self.wos_sampler = _build_wos_sampler(cfg)
        self.dataset = WoSDataset(
            shape_generator=gen,
            wos_sampler=self.wos_sampler,
            n_queries=int(ds_cfg.get("n_queries", 32)),
            n_wos_hits=int(ds_cfg.get("n_wos_hits_per_query", 4)),
            n_surface=int(ds_cfg.get("n_surface", 512)),
            virtual_length=int(cfg.get("optim", {}).get("total_steps", 1000)),
        )
        self.loader = DataLoader(
            self.dataset, batch_size=1, shuffle=False, num_workers=0,
            collate_fn=collate_wos_batches,
        )

        # --- Optimizer + scheduler ---
        optim_cfg = cfg.get("optim", {})
        self.total_steps = int(optim_cfg.get("total_steps", 1000))
        self.warmup_steps = int(optim_cfg.get("warmup_steps", 100))
        self.lr_max = float(optim_cfg.get("lr", 3e-4))
        self.lr_min = float(optim_cfg.get("lr_min", 1e-5))
        self.weight_decay = float(optim_cfg.get("weight_decay", 0.01))
        param_groups = _build_param_groups(self.model, self.weight_decay)
        self.optimizer = torch.optim.AdamW(param_groups, lr=self.lr_max)

        # --- Losses ---
        self.loss_registry = LossRegistry(cfg.get("losses", {}))

        # --- Logger + checkpoint dir ---
        log_cfg = cfg.get("log", {})
        self.run_name = str(log_cfg.get("run_name", "default"))
        self.logger = Logger(self.run_name, cfg)
        self.checkpoint_every = int(log_cfg.get("checkpoint_every", 500))
        self.run_dir = Path(log_cfg.get("run_dir", f"runs/{self.run_name}"))
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.start_step = 0
        self.last_loss: float | None = None

    # ------------------------------------------------------------------ step
    def _step(self, step: int, batch: dict) -> dict[str, float]:
        # LR schedule
        lr = cosine_with_warmup(
            step, self.total_steps, self.warmup_steps, self.lr_max, self.lr_min,
        )
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        # Loss-weight schedule — only override components the user did NOT
        # disable at config time. If cfg set `lambda_bl: null`, the schedule
        # never re-enables it (pretraining disables L_BL per §8.2).
        weights = loss_weight_schedule(step, self.cfg.get("losses", {}).get("schedule"))
        losses_cfg = self.cfg.get("losses", {})
        for k, v in weights.items():
            if losses_cfg.get(k, "__absent__") is None:
                continue  # user explicitly disabled this component
            if hasattr(self.loss_registry, k):
                setattr(self.loss_registry, k, v)

        # Encode shape once (batch_size=1; query dim is effective B)
        shape_latent_single = self.model.encode(batch["shape_ctx"])
        B_q = batch["p_nll"].shape[0]
        # Expand to match query batch
        shape_latent = ShapeLatent(tokens=shape_latent_single.tokens.expand(B_q, -1, -1))
        batch["shape_latent"] = shape_latent

        # Forward + loss
        losses = self.loss_registry(batch, self.model)
        total = losses["total"]

        # Backward
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {k: v.item() for k, v in losses.items()} | {"lr": lr}

    # ------------------------------------------------------------------ fit
    def fit(self, n_steps: int | None = None, resume: str | None = None) -> None:
        if resume is not None:
            self._load_checkpoint(resume)
        n_steps = n_steps or self.total_steps

        it = iter(self.loader)
        start = time.time()
        for step in range(self.start_step, n_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(self.loader)
                batch = next(it)

            batch = _move_to_device(batch, self.device)
            metrics = self._step(step, batch)
            self.last_loss = metrics["total"]

            self.logger.log(metrics, step=step)

            if step > 0 and step % self.checkpoint_every == 0:
                self._save_checkpoint(step)

        elapsed = time.time() - start
        self.logger.log({"final_elapsed_sec": elapsed}, step=n_steps)
        self._save_checkpoint(n_steps)

    # ---------------------------------------------------------- checkpoint
    def _save_checkpoint(self, step: int) -> str:
        path = self.run_dir / f"checkpoint_step_{step}.pt"
        torch.save({
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "rng_state_cpu": torch.get_rng_state(),
            "rng_state_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "cfg": self.cfg,
            "nhmo_version": "0.0.0-phase5",
        }, path)
        return str(path)

    def _load_checkpoint(self, path: str) -> None:
        # Load with map_location="cpu" for state_dict portability; restore to device after.
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # RNG state must be a CPU ByteTensor; it survives torch.load-to-cpu unchanged.
        rng_cpu = ckpt["rng_state_cpu"]
        if not isinstance(rng_cpu, torch.ByteTensor):
            rng_cpu = rng_cpu.cpu().to(torch.uint8)
        torch.set_rng_state(rng_cpu)
        rng_cuda = ckpt.get("rng_state_cuda")
        if rng_cuda is not None and torch.cuda.is_available():
            if not isinstance(rng_cuda, torch.ByteTensor):
                rng_cuda = rng_cuda.cpu().to(torch.uint8)
            torch.cuda.set_rng_state(rng_cuda)
        self.start_step = int(ckpt["step"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--steps", type=int, default=None,
                        help="Override cfg.optim.total_steps")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    trainer = Trainer(cfg)
    trainer.fit(n_steps=args.steps, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
