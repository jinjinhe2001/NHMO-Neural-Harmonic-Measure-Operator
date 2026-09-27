#!/bin/bash
# 3D coefficient-OOD problems (rebuttal): generate FEM references with lapy, then
# evaluate the released checkpoints with zero retraining. OOD_ROOT can instead point
# to the released problem set.
set -e
source "$(dirname "$0")/env.sh"
OOD_ROOT=${OOD_ROOT:-$PWD/data/ood3d}
if [ ! -d "$OOD_ROOT/ngf_solutions" ]; then
  python tools/gen_ood3d.py --out $OOD_ROOT --n-shapes 10 --k-per-shape 4          # Poisson OOD
  python tools/gen_ood3d.py --out ${OOD_ROOT}_lap --n-shapes 10 --k-per-shape 4 --s-f 0.0   # Laplace-only
fi
CK=$NHMO_CKPT_DIR/3d
for root in $OOD_ROOT ${OOD_ROOT}_lap; do
  for cat in nut gear motor fitting screws_and_bolts; do
    python -m nhmo.eval.mcb_lift --problems-root $root/ngf_solutions --category $cat \
        --n-shapes 10 --bcs-per-shape 4 --kernel-ckpt $CK/kernel_$cat.pt --lift-ckpt $CK/lift_$cat.pt \
        --out $RESULTS_DIR/ood3d_$(basename $root)_$cat.json
  done
done
