#!/bin/bash
# Build the axis-allocation arm on cvlab20 (plans/2026-09-05_axis-allocation.md).
#
# Two recipes, in this order:
#   1. maxstep11_u40_v2       -- REPRODUCTION of the shipped baseline. If its slim_meta
#                                does not match cvlab21's byte for byte, the materials
#                                here differ and the comparison would not be one-factor.
#   2. maxstep11_u40_qcut4_v2 -- the arm: 4 Q heads cut per layer instead of 13, the
#                                difference put into MLP channels at the same budget.
#
# --importance importance_v2_ada because that is what the shipped baseline used (its
# config.json records it); calib_100 either way, but the card differs and this project's
# scores are not bitwise identical across architectures.
set -euo pipefail
REM=/home/cvlab20/project/chan
cd $REM/alpamayo-model-compression
. $REM/cvlab20-server/env.sh
NEED=32000

pick() {   # first card with room, from the ones the user allows
  for g in 0 1 2 3 4 5 6 7; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$g")
    [ $((total - used)) -ge $NEED ] && { echo "$g"; return; }
  done
  echo ""
}
G=$(pick)
[ -z "$G" ] && { echo "REFUSING: no free card"; exit 1; }
echo "building on cuda:$G"

for cfg in maxstep11_u40_v2 maxstep11_u40_qcut4_v2; do
  out=outputs/slim_$cfg
  if [ -f "$out/slim_meta.json" ]; then echo "$cfg: exists, skipping"; continue; fi
  echo "=== $cfg"
  .venv/bin/python experiments/head_analysis/make_slim.py \
    --config "$cfg" --importance importance_v2_ada --out "$out" --no-state --gpu "$G"
done
echo "builds done"
