#!/bin/bash
# Paired analyses of the FM-only (layer-0 keep + layer-35 guard) closed loop on s600
# (plans/2026-10-29_traj-closedloop-s600.md §6). The arm ran on GPUs 0-3 only, i.e. every scene on Blackwell, so
# each comparison is also repeated on the scenes the other arm ran on Blackwell too:
#   vs dense / shipped MoOP   both ran scene i on GPU i of 0-7 -> drop the 300 Ada scenes (card_mismatch.csv)
#   vs MoOP l0keep            its card per scene is in outputs/alpasim_l0keep_s600/scene_gpu_l0keep.csv -> drop its
#                             171 Ada scenes
#
#   bash experiments/head_analysis/finish_traj_s600.sh      (after outputs/alpasim_trajl0_s600/main.DONE)
set -eu
REPO=/home/cvlab21/project/chan/alpamayo-model-compression
OUT=$REPO/outputs/alpasim_trajl0_s600
PY=/mnt/nvme1n1/ad_vla/venvs/alpasim-chan/bin/python
ARM="--arm f600_merged_slim_trajfix_u40_l0keep_v2 --arm-label FM-only_l0keep --no-pre-fix"
L0=l600_merged_slim_dual_u40_l0keep_v2

"$REPO/.venv/bin/python" -c "
import pandas as pd
m = pd.read_csv('$REPO/outputs/alpasim_l0keep_s600/scene_gpu_l0keep.csv')
m[m.arch == 'Ada'][['scene_id']].to_csv('$OUT/card_mismatch_vs_l0keep.csv', index=False)
print('MoOP l0keep scenes on Ada (dropped for the same-card comparison):', int((m.arch == 'Ada').sum()))
"

cd "$REPO/experiments/evaluation"
$PY analyze_dual_s600.py $ARM --base $L0 --base-label MoOP_l0keep --out "$OUT/vs_moop_l0keep"
$PY analyze_dual_s600.py $ARM --base $L0 --base-label MoOP_l0keep --out "$OUT/vs_moop_l0keep_same_card" \
  --exclude "$OUT/card_mismatch_vs_l0keep.csv"
$PY analyze_dual_s600.py $ARM --out "$OUT/vs_dense"
$PY analyze_dual_s600.py $ARM --out "$OUT/vs_dense_same_card" --exclude "$OUT/card_mismatch.csv"
$PY analyze_dual_s600.py $ARM --base d600_merged_slim_dual_u40_v2 --base-label MoOP --out "$OUT/vs_moop"
$PY analyze_dual_s600.py $ARM --base d600_merged_slim_dual_u40_v2 --base-label MoOP --out "$OUT/vs_moop_same_card" \
  --exclude "$OUT/card_mismatch.csv"
touch "$OUT/analysis.DONE"
