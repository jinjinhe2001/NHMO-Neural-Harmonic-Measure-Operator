#!/bin/bash
# Table 4 (synthetic Laplace probe of the kernel alone): analytically harmonic h,
# u_h(p) = <h, K(p, .)> vs h(p), 20 unseen shapes x 64 interior anchors per category.
set -e
source "$(dirname "$0")/env.sh"
CK=$NHMO_CKPT_DIR/3d
for cat in nut gear motor fitting screws_and_bolts; do
  python -m nhmo.eval.mcb_laplace --ckpt $CK/kernel_$cat.pt --category $cat \
      --split unknown_shape_unknown_prob --n-shapes 20 --n-queries 64 --seed 0 \
      --out $RESULTS_DIR/table4_$cat.json
done
