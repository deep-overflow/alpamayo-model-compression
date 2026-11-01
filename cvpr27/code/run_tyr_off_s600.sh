#!/bin/bash
# Closed loop of the officially rebuilt Tyr-the-Pruner arm (slim_tyr_off, 2,657,452,032 removed = MoOP) on s600, GPUs
# 4-7 only (user, 2026-11-01; cvpr27/plans/2026-11-01_tyr-off-closedloop-s600.md). Scenes keep their dense-baseline card
# when it was Ada (shard = baseline GPU % 4 over "4 5 6 7"); the 300 baseline-Blackwell scenes run on Ada, so every
# comparison is repeated without the scenes the other arm ran on Blackwell.
#   smoke (4 baseline-Ada scenes, one per card) -> gates G1/G2 -> main (596 more) -> paired analyses
#
#   bash cvpr27/code/run_tyr_off_s600.sh > outputs/cvpr27/alpasim_tyr_off_s600/queue.log 2>&1
set -u
REPO=/home/cvlab21/project/chan/alpamayo-model-compression
PY=/mnt/nvme1n1/ad_vla/venvs/alpasim-chan/bin/python
L0=l600_merged_slim_dual_u40_l0keep_v2
stamp() { echo "$(date -u '+%F %T') UTC $*"; }
cd "$REPO"

CFG=slim_tyr_off TAG=tyo GPUS="4 5 6 7" OUT=$REPO/outputs/cvpr27/alpasim_tyr_off_s600
SMOKE_CSV=$OUT/scenes_smoke4.csv MAIN_CSV=$OUT/scenes_t600.csv
export CFG TAG GPUS OUT SMOKE_CSV MAIN_CSV

stamp "$CFG: smoke on GPUs $GPUS"
STAGE=smoke bash experiments/head_analysis/run_dual_s600.sh > "$OUT/smoke.log" 2>&1
$PY experiments/evaluation/s600_smoke_gates.py --run ${TAG}smoke_merged_$CFG --logs ${TAG}smoke_$CFG \
  --scenes "$SMOKE_CSV" --out "$OUT/smoke" > "$OUT/smoke_gates.txt" 2>&1
grep -q "gates: G1 PASS  G2 PASS" "$OUT/smoke_gates.txt" || { stamp "$CFG: gates FAILED"; exit 1; }
stamp "$CFG: gates passed, main"
STAGE=main bash experiments/head_analysis/run_dual_s600.sh > "$OUT/main.log" 2>&1
[ -e "$OUT/main.DONE" ] || { stamp "$CFG: main did not finish"; exit 1; }

A="--arm ${TAG}_merged_$CFG --arm-label tyr_off --no-pre-fix"
cd experiments/evaluation
$PY analyze_dual_s600.py $A --base d600_merged_slim_dual_u40_v2 --base-label MoOP --out "$OUT/vs_moop"
$PY analyze_dual_s600.py $A --base d600_merged_slim_dual_u40_v2 --base-label MoOP --out "$OUT/vs_moop_same_card" \
  --exclude "$OUT/card_mismatch_vs_dense.csv"
$PY analyze_dual_s600.py $A --base $L0 --base-label MoOP_l0keep --out "$OUT/vs_moop_l0keep"
$PY analyze_dual_s600.py $A --base $L0 --base-label MoOP_l0keep --out "$OUT/vs_moop_l0keep_same_card" \
  --exclude "$OUT/card_mismatch_vs_l0keep.csv"
$PY analyze_dual_s600.py $A --out "$OUT/vs_dense"
$PY analyze_dual_s600.py $A --out "$OUT/vs_dense_same_card" --exclude "$OUT/card_mismatch_vs_dense.csv"
if [ -e "$REPO/outputs/cvpr27/alpasim_lp_m24_s600/analysis.DONE" ]; then   # lp_m24 kept every baseline card type
  $PY analyze_dual_s600.py $A --base p24_merged_slim_lp_m24 --base-label lp_m24 --out "$OUT/vs_lp_m24"
  $PY analyze_dual_s600.py $A --base p24_merged_slim_lp_m24 --base-label lp_m24 --out "$OUT/vs_lp_m24_same_card" \
    --exclude "$OUT/card_mismatch_vs_dense.csv"
fi
cd "$REPO"
touch "$OUT/analysis.DONE"
stamp "$CFG: done"
