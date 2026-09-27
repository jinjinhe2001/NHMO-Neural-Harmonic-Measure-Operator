#!/bin/bash
# 3D inference timing (per-shape build and per-problem cached solve) with the
# optimized path, on nut and motor (5 shapes x 4 problems each). Use an otherwise idle GPU.
set -e
source "$(dirname "$0")/env.sh"
for cat in nut motor; do
  python tools/bench_3d_inference.py --category $cat --reference --out $RESULTS_DIR/bench3d_$cat.json
done
