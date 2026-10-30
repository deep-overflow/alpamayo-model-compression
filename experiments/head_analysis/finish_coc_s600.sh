#!/bin/bash
# Paired analyses of the CE-only (layer-0 keep) closed loop on s600 (plans/2026-10-30_coc-closedloop-s600.md §3).
# The arm ran on GPUs 0-3 with FM-only's scene-to-card map, i.e. every scene on Blackwell, so each comparison is
# also repeated on the scenes the other arm ran on Blackwell too:
#   vs dense / shipped MoOP   drop their 300 Ada scenes (card_mismatch.csv)
#   vs MoOP l0keep            drop its 171 Ada scenes (card_mismatch_vs_l0keep.csv)
#   vs FM-only l0keep         same card on all 600 scenes
#
#   bash experiments/head_analysis/finish_coc_s600.sh      (after outputs/alpasim_cocl0_s600/main.DONE)
set -eu
REPO=/home/cvlab21/project/chan/alpamayo-model-compression
OUT=$REPO/outputs/alpasim_cocl0_s600
PY=/mnt/nvme1n1/ad_vla/venvs/alpasim-chan/bin/python
ARM="--arm c600_merged_slim_coc_u40_l0keep_v2 --arm-label CE-only_l0keep --no-pre-fix"
L0=l600_merged_slim_dual_u40_l0keep_v2

cd "$REPO/experiments/evaluation"
$PY analyze_dual_s600.py $ARM --base $L0 --base-label MoOP_l0keep --out "$OUT/vs_moop_l0keep"
$PY analyze_dual_s600.py $ARM --base $L0 --base-label MoOP_l0keep --out "$OUT/vs_moop_l0keep_same_card" \
  --exclude "$OUT/card_mismatch_vs_l0keep.csv"
$PY analyze_dual_s600.py $ARM --base f600_merged_slim_trajfix_u40_l0keep_v2 --base-label FM-only_l0keep \
  --out "$OUT/vs_fmonly_l0keep"
$PY analyze_dual_s600.py $ARM --out "$OUT/vs_dense"
$PY analyze_dual_s600.py $ARM --out "$OUT/vs_dense_same_card" --exclude "$OUT/card_mismatch.csv"
$PY analyze_dual_s600.py $ARM --base d600_merged_slim_dual_u40_v2 --base-label MoOP --out "$OUT/vs_moop"
$PY analyze_dual_s600.py $ARM --base d600_merged_slim_dual_u40_v2 --base-label MoOP --out "$OUT/vs_moop_same_card" \
  --exclude "$OUT/card_mismatch.csv"
touch "$OUT/analysis.DONE"
