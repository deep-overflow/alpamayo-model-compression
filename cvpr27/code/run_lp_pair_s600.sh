#!/bin/bash
# Closed loop of the two LLM-Pruner arms on s600 side by side, four cards each
# (cvpr27/plans/2026-10-31_lp-closedloop-s600.md):
#   lp_m24  budget-matched, 2,657,452,032 removed = MoOP   driver slim_lp_m24   GPUs 0 1 4 5
#   lp_r50  original ratio 0.5, 2,768,240,640 (24.99%)     driver lp_r50        GPUs 2 3 6 7
# Each arm gets two Blackwell and two Ada cards, and every scene runs on its dense-baseline card type
# (outputs/cvpr27/lp_s600_scenes/scenes_{smoke8,d600}_4c.csv: shard indexes the arm's GPU list B B A A).
# Per arm: smoke (8 scenes) -> gates G1/G2 -> main (592 more) -> paired analyses. A failed gate stops that arm only.
# When both are done: lp_r50 vs lp_m24.
#
#   bash cvpr27/code/run_lp_pair_s600.sh > outputs/cvpr27/lp_s600.log 2>&1      (start when the user says so)
set -u
REPO=/home/cvlab21/project/chan/alpamayo-model-compression
PY=/mnt/nvme1n1/ad_vla/venvs/alpasim-chan/bin/python
L0=l600_merged_slim_dual_u40_l0keep_v2
S=$REPO/outputs/cvpr27/lp_s600_scenes
stamp() { echo "$(date -u '+%F %T') UTC $*"; }
cd "$REPO"

for g in 0 1 2 3 4 5 6 7; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
  [ "$used" -gt 10 ] && { stamp "GPU $g holds $used MiB -- not starting"; exit 1; }
done

arm() {   # <driver folder> <tag> <label> "<gpus>"
  CFG=$1 TAG=$2 GPUS=$4 OUT=$REPO/outputs/cvpr27/alpasim_$3_s600 SMOKE_CSV=$S/scenes_smoke8_4c.csv
  MAIN_CSV=$S/scenes_d600_4c.csv
  export CFG TAG GPUS OUT SMOKE_CSV MAIN_CSV
  mkdir -p "$OUT"
  stamp "$3: smoke on GPUs $GPUS"
  STAGE=smoke bash experiments/head_analysis/run_dual_s600.sh > "$OUT/smoke.log" 2>&1
  $PY experiments/evaluation/s600_smoke_gates.py --run ${TAG}smoke_merged_$CFG --logs ${TAG}smoke_$CFG \
    --scenes "$SMOKE_CSV" --out "$OUT/smoke" > "$OUT/smoke_gates.txt" 2>&1
  grep -q "gates: G1 PASS  G2 PASS" "$OUT/smoke_gates.txt" || { stamp "$3: gates FAILED"; return 1; }
  stamp "$3: gates passed, main"
  STAGE=main bash experiments/head_analysis/run_dual_s600.sh > "$OUT/main.log" 2>&1
  [ -e "$OUT/main.DONE" ] || { stamp "$3: main did not finish"; return 1; }

  A="--arm ${TAG}_merged_$CFG --arm-label $3 --no-pre-fix"
  cd experiments/evaluation
  $PY analyze_dual_s600.py $A --base d600_merged_slim_dual_u40_v2 --base-label MoOP --out "$OUT/vs_moop"
  $PY analyze_dual_s600.py $A --base $L0 --base-label MoOP_l0keep --out "$OUT/vs_moop_l0keep"
  $PY analyze_dual_s600.py $A --base $L0 --base-label MoOP_l0keep --out "$OUT/vs_moop_l0keep_same_card" \
    --exclude "$REPO/outputs/alpasim_l0keep_s600/card_mismatch.csv"
  $PY analyze_dual_s600.py $A --out "$OUT/vs_dense"
  cd "$REPO"
  touch "$OUT/analysis.DONE"
  stamp "$3: done"
}

arm slim_lp_m24 p24 lp_m24 "0 1 4 5" > outputs/cvpr27/lp_m24_s600.log 2>&1 &
sleep 30
arm lp_r50 r50 lp_r50 "2 3 6 7" > outputs/cvpr27/lp_r50_s600.log 2>&1 &
stamp "both arms launched"
wait
cat outputs/cvpr27/lp_m24_s600.log outputs/cvpr27/lp_r50_s600.log

if [ -e outputs/cvpr27/alpasim_lp_m24_s600/analysis.DONE ] && [ -e outputs/cvpr27/alpasim_lp_r50_s600/analysis.DONE ]; then
  cd experiments/evaluation
  $PY analyze_dual_s600.py --arm r50_merged_lp_r50 --arm-label lp_r50 --base p24_merged_slim_lp_m24 \
    --base-label lp_m24 --no-pre-fix --out "$REPO/outputs/cvpr27/alpasim_lp_r50_s600/vs_lp_m24"
  cd "$REPO"
fi
stamp "pair finished"
