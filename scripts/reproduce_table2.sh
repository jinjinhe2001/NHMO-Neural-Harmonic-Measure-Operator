#!/bin/bash
# Tables 2 and 3 (MCB-B Poisson, 20 unseen shapes x 16 problems per category).
# Reference path: 40-100 minutes per category on one A100 (--q-batch 1024 needs ~30 GB).
# Add FAST=1 to use the optimized inference path (a few minutes per category).
# SEEDS="0 1 2" evaluates several evaluation seeds (surface samples, anchors, source probes).
set -e
source "$(dirname "$0")/env.sh"
CK=$NHMO_CKPT_DIR/3d
SEEDS=${SEEDS:-0}
EXTRA=""; TAG=ref
if [ "${FAST:-0}" = "1" ]; then EXTRA="--fast"; TAG=fast; fi
for cat in nut gear motor fitting screws_and_bolts; do
  for s in $SEEDS; do
    python -m nhmo.eval.mcb_lift --category $cat --seed $s $EXTRA \
        --kernel-ckpt $CK/kernel_$cat.pt --lift-ckpt $CK/lift_$cat.pt \
        --out $RESULTS_DIR/table2_${cat}_${TAG}_seed$s.json
  done
done
