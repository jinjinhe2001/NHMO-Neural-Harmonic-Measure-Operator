"""Logger: local JSONL + optional W&B forwarding.

Design:
  - always write metrics to ./runs/<run_name>/metrics.jsonl (works in CI,
    no credentials, no network)
  - forward to W&B iff WANDB_API_KEY env var is set
  - log the four loss components + MV residual; in later phases also log
    (i) integral K d sigma on held-out p's every 500 steps, (ii) peak /
    median ratio of K across zeta

Phase 1: actual writer implementation so the spine is testable end-to-end
without Phase 2+ code. The extra A3 metrics are added in later phases.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class Logger:
    def __init__(
        self, run_name: str, cfg: dict | None = None,
        run_dir: str | Path | None = None,
    ) -> None:
        self.run_name = run_name
        self.cfg = cfg or {}
        if run_dir is not None:
            self.run_dir = Path(run_dir)
        else:
            # Fall back to cfg["log"]["run_dir"] if present, else default layout.
            cfg_log = (cfg or {}).get("log", {})
            self.run_dir = Path(cfg_log.get("run_dir", str(Path("runs") / run_name)))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self._wandb = None
        if os.environ.get("WANDB_API_KEY"):
            try:
                import wandb  # type: ignore
                self._wandb = wandb.init(
                    project=self.cfg.get("project", "nhmo_v2"),
                    name=run_name,
                    config=self.cfg,
                )
            except ImportError:
                # wandb not installed is not fatal; we silently fall back
                # to local JSONL.
                self._wandb = None

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        record = dict(metrics)
        if step is not None:
            record["step"] = step
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def close(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
