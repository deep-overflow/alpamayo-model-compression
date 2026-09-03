#!/bin/bash
# Stage 1 of plans/2026-09-03_calib-draw-variance.md: measure dual-objective Taylor
# importance on the 500 new calibration clips, then on block a alone for gate G0.
#
# Both runs go on the Ada cards. The plan preferred Blackwell (importance_v2, which
# defines calib_100's scores, was measured there and determinism holds only within one
# architecture), but no Blackwell card has the 40.5 GB this pass peaks at, and the
# architecture factor is handled instead by evaluating importance_v2_ada as its own arm:
# Ada vs Blackwell on the SAME clips moves 2.7% of the kept set, so calib_100 enters the
# comparison card-matched rather than uncorrected.
#
# run_retry_host.sh retries every 60 s while the cards are busy, so this can be launched
# before they free.
set -u
REPO=${ALPAMAYO_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
export ALPAMAYO_REPO=$REPO
GPUS=${GPUS:-4,5,6,7}
LOG=$REPO/logs

bash "$REPO/experiments/head_analysis/run_retry_host.sh" 120 \
  "$REPO/experiments/head_analysis/run_importance.py" \
  --calib-manifest calib_tr500 --cache train --num-clips 500 \
  --exp-id importance_tr500 --gpu "$GPUS" >>"$LOG/importance_tr500.log" 2>&1
echo "$(date -u '+%F %T') tr500 exit=$?" >>"$LOG/importance_tr500.log"

# G0: block a measured on its own must reproduce the block-a mean synthesised from the
# 500-clip run bit for bit. If it does not, the ladder cannot be derived and every rung
# needs its own run.
bash "$REPO/experiments/head_analysis/run_retry_host.sh" 120 \
  "$REPO/experiments/head_analysis/run_importance.py" \
  --calib-manifest calib_tr100_a --cache train --num-clips 100 \
  --exp-id importance_tr100_a_solo --gpu "$GPUS" >>"$LOG/importance_g0.log" 2>&1
echo "$(date -u '+%F %T') g0 exit=$?" >>"$LOG/importance_g0.log"
