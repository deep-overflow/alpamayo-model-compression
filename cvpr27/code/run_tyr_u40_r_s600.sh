#!/bin/bash
# Closed loop of the paper's Tyr row (slim_tyr_u40_r, this repo's port, 2,657,452,032 removed = MoOP) on s600, all eight
# cards, started once the Wanda-sp run is over (user, 2026-11-02; cvpr27/plans/2026-11-02_tyr-u40-r-closedloop-s600.md).
# Every scene runs on its dense-baseline card (run_dual_s600.sh defaults: scenes_smoke8.csv, scenes_d600.csv).
#   wait (Wanda-sp script gone, eight cards empty) -> smoke (8) -> gates -> main (592) -> paired analyses
#
#   bash cvpr27/code/run_tyr_u40_r_s600.sh > outputs/cvpr27/tyr_u40_r_s600.log 2>&1
set -u
REPO=/home/cvlab21/project/chan/alpamayo-model-compression
PY=/mnt/nvme1n1/ad_vla/venvs/alpasim-chan/bin/python
L0=l600_merged_slim_dual_u40_l0keep_v2
stamp() { echo "$(date -u '+%F %T') UTC $*"; }
cd "$REPO"

n=0
until ! pgrep -f "^bash cvpr27/code/run_wandasp_s600.sh" > /dev/null \
      && [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 10' | wc -l)" = 0 ]; do
  [ $((n % 60)) = 0 ] && stamp "waiting for the Wanda-sp run to end and eight empty cards"
  n=$((n + 1)); sleep 60
done

CFG=slim_tyr_u40_r TAG=t40 OUT=$REPO/outputs/cvpr27/alpasim_tyr_u40_r_s600
export CFG TAG OUT
mkdir -p "$OUT"
stamp "$CFG: smoke on GPUs 0-7"
STAGE=smoke bash experiments/head_analysis/run_dual_s600.sh > "$OUT/smoke.log" 2>&1
$PY experiments/evaluation/s600_smoke_gates.py --run ${TAG}smoke_merged_$CFG --logs ${TAG}smoke_$CFG \
  --scenes outputs/alpasim_dual_s600/scenes_smoke8.csv --out "$OUT/smoke" > "$OUT/smoke_gates.txt" 2>&1
grep -q "gates: G1 PASS  G2 PASS" "$OUT/smoke_gates.txt" || { stamp "$CFG: gates FAILED"; exit 1; }
stamp "$CFG: gates passed, main"
STAGE=main bash experiments/head_analysis/run_dual_s600.sh > "$OUT/main.log" 2>&1
[ -e "$OUT/main.DONE" ] || { stamp "$CFG: main did not finish"; exit 1; }

A="--arm ${TAG}_merged_$CFG --arm-label tyr_u40_r --no-pre-fix"
cd experiments/evaluation
$PY analyze_dual_s600.py $A --base d600_merged_slim_dual_u40_v2 --base-label MoOP --out "$OUT/vs_moop"
$PY analyze_dual_s600.py $A --base $L0 --base-label MoOP_l0keep --out "$OUT/vs_moop_l0keep"
$PY analyze_dual_s600.py $A --base $L0 --base-label MoOP_l0keep --out "$OUT/vs_moop_l0keep_same_card" \
  --exclude "$REPO/outputs/alpasim_l0keep_s600/card_mismatch.csv"
$PY analyze_dual_s600.py $A --out "$OUT/vs_dense"
$PY analyze_dual_s600.py $A --base tyo_merged_slim_tyr_off --base-label tyr_off --out "$OUT/vs_tyr_off"
$PY analyze_dual_s600.py $A --base tyo_merged_slim_tyr_off --base-label tyr_off --out "$OUT/vs_tyr_off_same_card" \
  --exclude "$REPO/outputs/cvpr27/alpasim_tyr_off_s600/card_mismatch_vs_dense.csv"
$PY analyze_dual_s600.py $A --base p24_merged_slim_lp_m24 --base-label lp_m24 --out "$OUT/vs_lp_m24"
[ -e "$REPO/outputs/cvpr27/alpasim_wandasp_s600/analysis.DONE" ] && \
  $PY analyze_dual_s600.py $A --base wsp_merged_slim_wandasp_u40 --base-label wanda_sp --out "$OUT/vs_wanda_sp"
cd "$REPO"
touch "$OUT/analysis.DONE"
stamp "$CFG: done"
