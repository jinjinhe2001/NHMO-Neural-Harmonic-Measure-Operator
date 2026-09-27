"""Build the benchmark subset used for the headline 2D numbers from the full generator output.

Rule (reproduces datasets/mnist_pde_2d_paramBC_lf of the paper exactly):
  - keep problems whose boundary family is poly3 or exp_mix (drop trig1 / trig2);
  - in the train split, drop shapes left without any Poisson problem (the lift
    trainers sample Poisson problems only);
  - test / test_ood keep all 50 shapes.
Result: 991 / 50 / 50 shapes and 8005 / 408 / 397 problems.

Files are hard-linked when possible (same file system), otherwise copied.

    python tools/filter_param_bc.py data/mnist_pde_2d_paramBC data/mnist_pde_2d_paramBC_lf
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

KEEP_FAMILIES = ("_poly3_", "_exp_mix_")


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> int:
    pa = argparse.ArgumentParser()
    pa.add_argument("src")
    pa.add_argument("dst")
    a = pa.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    for split in ("train", "test", "test_ood"):
        if not (src / split).exists():
            continue
        n_shapes = n_prob = n_poi = 0
        for sd in sorted(p for p in (src / split).iterdir() if p.is_dir()):
            keep = sorted(f for f in sd.glob("*.npz") if any(k in f.name for k in KEEP_FAMILIES))
            if split == "train" and not any(f.name.startswith("poisson_") for f in keep):
                continue
            out = dst / split / sd.name
            out.mkdir(parents=True, exist_ok=True)
            for extra in ("mask.npy", "boundary.npy"):
                if (sd / extra).exists():
                    link_or_copy(sd / extra, out / extra)
            for f in keep:
                link_or_copy(f, out / f.name)
            n_shapes += 1
            n_prob += len(keep)
            n_poi += sum(f.name.startswith("poisson_") for f in keep)
        print(f"{split}: {n_shapes} shapes, {n_prob} problems ({n_poi} Poisson)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
