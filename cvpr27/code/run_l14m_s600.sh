#!/bin/bash
# Closed loop of MoOP l0keep + layers 1-4 MLP cut to 95 / 90 / 80% of their mass on s600, one arm after another
# (cvpr27/plans/2026-10-30_l14m-closedloop-s600.md). Each arm runs on all eight cards with every scene on its
# dense-baseline card (run_dual_s600.sh defaults: scenes_smoke8.csv, scenes_d600.csv), after the cards read empty:
#   smoke (8 scenes) -> gates G1/G2 -> main (592 more) -> paired analyses
# A failed gate or merge stops the whole queue.
#
#   bash cvpr27/code/run_l14m_s600.sh > outputs/cvpr27/l14m_s600_queue.log 2>&1
set -u
REPO=/home/cvlab21/project/chan/alpamayo-model-compression
PY=/mnt/nvme1n1/ad_vla/venvs/alpasim-chan/bin/python
L0=l600_merged_slim_dual_u40_l0keep_v2
stamp() { echo "$(date -u '+%F %T') UTC $*"; }
cd "$REPO"

wait_empty() {
  local n=0
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 10' | wc -l)" = 0 ] \
        && [ -e outputs/alpasim_cocl0_s600/analysis.DONE ]; do
    [ $((n % 30)) = 0 ] && stamp "waiting for eight empty cards (and the CE-only analysis)"
    n=$((n + 1)); sleep 60
  done
}

for x in 95 90 80; do
  CFG=slim_dual_u40_l0keep_l14m${x}_v2 TAG=m$x OUT=$REPO/outputs/cvpr27/alpasim_l14m${x}_s600
  export CFG TAG OUT
  mkdir -p "$OUT"
  wait_empty
  stamp "$CFG: smoke"
  STAGE=smoke bash experiments/head_analysis/run_dual_s600.sh > "$OUT/smoke.log" 2>&1
  $PY experiments/evaluation/s600_smoke_gates.py --run ${TAG}smoke_merged_$CFG --logs ${TAG}smoke_$CFG \
    --scenes outputs/alpasim_dual_s600/scenes_smoke8.csv --out "$OUT/smoke" > "$OUT/smoke_gates.txt" 2>&1
  if ! grep -q "gates: G1 PASS  G2 PASS" "$OUT/smoke_gates.txt"; then stamp "$CFG: gates FAILED, queue stopped"; exit 1; fi
  stamp "$CFG: gates passed, main"
  STAGE=main bash experiments/head_analysis/run_dual_s600.sh > "$OUT/main.log" 2>&1
  [ -e "$OUT/main.DONE" ] || { stamp "$CFG: main did not finish, queue stopped"; exit 1; }

  ARM="--arm ${TAG}_merged_$CFG --arm-label l14m$x --no-pre-fix"
  cd experiments/evaluation
  $PY analyze_dual_s600.py $ARM --base $L0 --base-label MoOP_l0keep --out "$OUT/vs_moop_l0keep"
  $PY analyze_dual_s600.py $ARM --base $L0 --base-label MoOP_l0keep --out "$OUT/vs_moop_l0keep_same_card" \
    --exclude "$REPO/outputs/alpasim_l0keep_s600/card_mismatch.csv"
  $PY analyze_dual_s600.py $ARM --out "$OUT/vs_dense"
  $PY analyze_dual_s600.py $ARM --base d600_merged_slim_dual_u40_v2 --base-label MoOP --out "$OUT/vs_moop"
  [ "$x" != 95 ] && $PY analyze_dual_s600.py $ARM --base m95_merged_slim_dual_u40_l0keep_l14m95_v2 \
    --base-label l14m95 --out "$OUT/vs_l14m95"
  cd "$REPO"
  touch "$OUT/analysis.DONE"
  stamp "$CFG: done"
done
stamp "queue finished"
