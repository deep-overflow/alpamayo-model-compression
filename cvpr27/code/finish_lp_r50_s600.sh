#!/bin/bash
# lp_r50 side of run_lp_pair_split_s600.sh after its Ada shard 1 lost a host-port race to lp_m24's shard 0
# (both wizards picked 6006 one second apart, 2026-10-31 12:29 UTC; cvpr27/plans/2026-10-31_lp-closedloop-s600.md §6).
#   - Ada shard 1's 148 scenes rerun as r50c_ on GPU 7 (started by hand)
#   - this script takes over the Blackwell half from the pair script's waiter (killed): it waits for GPUs 2 3 to read
#     empty, then 5 more minutes so lp_m24's Blackwell wizards bind their ports first, and runs r50b_ there
#   - when everything is in: merge smoke + r50a sh0 + r50c + r50b -> r50_merged_lp_r50 (600), paired analyses, and
#     lp_r50 vs lp_m24 once lp_m24 is analysed
#
#   bash cvpr27/code/finish_lp_r50_s600.sh > outputs/cvpr27/alpasim_lp_r50_s600/finish.log 2>&1
set -u
REPO=/home/cvlab21/project/chan/alpamayo-model-compression
RUNS=/home/cvlab21/project/chan/alpasim-runs
PY=/mnt/nvme1n1/ad_vla/venvs/alpasim-chan/bin/python
S=$REPO/outputs/cvpr27/lp_s600_scenes
OUT=$REPO/outputs/cvpr27/alpasim_lp_r50_s600
L0=l600_merged_slim_dual_u40_l0keep_v2
stamp() { echo "$(date -u '+%F %T') UTC $*"; }
empty() {
  for g in "$@"; do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")" -gt 10 ] && return 1; done
  return 0
}
cd "$REPO"

until empty 2 3; do sleep 60; done
stamp "GPUs 2 3 empty; waiting 5 min for lp_m24's Blackwell wizards"
sleep 300
until empty 2 3; do sleep 60; done
stamp "r50b: Blackwell main on 2 3"
SUITE=public_2601_r50b PARENT_SUITE=public_2601 PREFIX=r50b_ DRIVER_OMP_THREADS=8 SCENES_CSV=$S/main_bw300.csv \
  bash experiments/head_analysis/launch_alpasim_shards.sh lp_r50 300 1 "2 3" > "$OUT/r50b.log" 2>&1
for c in $(docker ps -a --format '{{.Names}}' | grep "^r50b_"); do docker rm -f "$c" > /dev/null; done
for n in $(docker network ls --format '{{.Name}}' | grep "^r50b_"); do
  [ "$(docker network inspect -f '{{len .Containers}}' "$n")" = 0 ] && docker network rm "$n" > /dev/null
done
stamp "r50b finished"

until [ -e "$OUT/r50c.DONE" ] && ! pgrep -f "launch_alpasim_shards.sh lp_r50 296" > /dev/null; do sleep 120; done
rm -rf "${RUNS:?}/r50_merged_lp_r50"   # a partial merge the pair script may have left
"$REPO/.venv/bin/python" experiments/head_analysis/merge_alpasim_shards.py --runs-root "$RUNS" \
  --shards r50smoke_merged_lp_r50 r50a_lp_r50_sh0 r50c_lp_r50_sh0 r50b_lp_r50_sh0 r50b_lp_r50_sh1 \
  --out r50_merged_lp_r50 --expect-scenes 600 > "$OUT/merge.log" 2>&1 || { stamp "merge failed"; exit 1; }
stamp "merged 600"

A="--arm r50_merged_lp_r50 --arm-label lp_r50 --no-pre-fix"
cd experiments/evaluation
$PY analyze_dual_s600.py $A --base d600_merged_slim_dual_u40_v2 --base-label MoOP --out "$OUT/vs_moop"
$PY analyze_dual_s600.py $A --base $L0 --base-label MoOP_l0keep --out "$OUT/vs_moop_l0keep"
$PY analyze_dual_s600.py $A --base $L0 --base-label MoOP_l0keep --out "$OUT/vs_moop_l0keep_same_card" \
  --exclude "$REPO/outputs/alpasim_l0keep_s600/card_mismatch.csv"
$PY analyze_dual_s600.py $A --out "$OUT/vs_dense"
touch "$OUT/analysis.DONE"
stamp "lp_r50 analysed"

until [ -e "$REPO/outputs/cvpr27/alpasim_lp_m24_s600/analysis.DONE" ]; do sleep 120; done
$PY analyze_dual_s600.py $A --base p24_merged_slim_lp_m24 --base-label lp_m24 --out "$OUT/vs_lp_m24"
stamp "lp_r50 vs lp_m24 done"
