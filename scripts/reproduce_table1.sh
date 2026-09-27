#!/bin/bash
# Table 1 (2D MNIST benchmark), test and test_ood splits.
# About 10 minutes per (model, split) on one A100.
#   scripts/reproduce_table1.sh             NHMO (kernel + lift) and the kernel-only row
#   VARIANT=1 scripts/reproduce_table1.sh   additionally the variant with residual head
#                                           (paper Section 6 / Appendix K.1)
set -e
source "$(dirname "$0")/env.sh"
CK=$NHMO_CKPT_DIR/2d
K=$CK/kernel_2d_mask.pt
for split in test test_ood; do
  # NHMO: u = u_h + v(mask, h, f, u_h)
  python -m nhmo.eval.mnist_pde_2d --split $split --kernel-ckpt $K --lift-ckpt $CK/lift_2d_l17c_mask.pt \
      --out $RESULTS_DIR/table1_nhmo_$split.json
  # kernel only: u = u_h
  python -m nhmo.eval.mnist_pde_2d --split $split --kernel-ckpt $K \
      --out $RESULTS_DIR/table1_konly_$split.json
  if [ "${VARIANT:-0}" = "1" ]; then
    # variant with residual head: u = u_h + v(mask, sdf, f) + r(mask, h, u_h)
    python -m nhmo.eval.mnist_pde_2d --split $split --kernel-ckpt $K --lift-ckpt $CK/lift_2d_msf_mask.pt \
        --r-ckpt $CK/rhead_2d_mfr_mask.pt --out $RESULTS_DIR/table1_variant_rhead_$split.json
  fi
done
