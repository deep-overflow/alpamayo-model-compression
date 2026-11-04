#!/bin/bash
# 76 picked scenes x 4 rollouts, split over both card groups once GPUs 4-7 freed up (user, 2026-11-05 02:5x KST;
# cvpr27/plans/2026-11-05_rollout4-76scenes.md §4). Takes over from run_rollout4_76.sh (its bash was stopped; the
# MoOP l0keep launcher it started keeps running on GPUs 0-3):
#   GPUs 4-7  tyr_off now
#   GPUs 0-3  wait for the MoOP l0keep launcher -> clean up and merge it -> lp_m24 -> dense
# The same scene list (scenes76_gpu0-3.csv) is used on both groups: its shard column indexes the GPU list, so a scene
# keeps its slot (shard i -> GPU i or GPU 4+i). After both groups: cvpr27/code/analyze_rollout4_76.py.
#
#   bash cvpr27/code/run_rollout4_76_split.sh > outputs/cvpr27/rollout4_76/queue_split.log 2>&1
set -u
REPO=/home/cvlab21/project/chan/alpamayo-model-compression
RUNS=/home/cvlab21/project/chan/alpasim-runs
OUT=$REPO/outputs/cvpr27/rollout4_76
CSV=$OUT/scenes76_gpu0-3.csv
stamp() { echo "$(date -u '+%F %T') UTC $*"; }
cd "$REPO"

finish() {   # <driver folder> <prefix tag> <label>: containers out, shards merged
  for c in $(docker ps -a --format '{{.Names}}' | grep "^$2_"); do docker rm -f "$c" > /dev/null; done
  for net in $(docker network ls --format '{{.Name}}' | grep "^$2_"); do
    [ "$(docker network inspect -f '{{len .Containers}}' "$net")" = 0 ] && docker network rm "$net" > /dev/null
  done
  "$REPO/.venv/bin/python" experiments/head_analysis/merge_alpasim_shards.py --runs-root "$RUNS" \
    --shards $(for i in 0 1 2 3; do echo "$2_$1_sh$i"; done) --out "$2_merged_$1" --expect-scenes 76 \
    > "$OUT/$2_merge.log" 2>&1 || stamp "$3: merge failed"
  stamp "$3: done ($(python3 -c "import json;print(len(json.load(open('$RUNS/$2_merged_$1/aggregate/results-summary.json'))['rollouts']))" 2>&1 | tail -1) rollouts)"
}

run_arm() {   # <driver folder> <prefix tag> <label> "<gpus>" <comma gpus>
  local n=0
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$5" | awk '$1 > 10' | wc -l)" = 0 ]; do
    [ $((n % 30)) = 0 ] && stamp "$3: waiting for GPUs $4 to be empty"
    n=$((n + 1)); sleep 60
  done
  stamp "$3: 76 scenes x 4 rollouts on GPUs $4"
  SUITE=public_2601_$2 PARENT_SUITE=public_2601 PREFIX=$2_ DRIVER_OMP_THREADS=8 SCENES_CSV=$CSV \
    bash experiments/head_analysis/launch_alpasim_shards.sh "$1" 76 4 "$4" > "$OUT/$2.log" 2>&1
  finish "$1" "$2" "$3"
}

run_arm slim_tyr_off r4ty tyr_off "4 5 6 7" 4,5,6,7 &
(
  until ! pgrep -f "launch_alpasim_shards.sh slim_dual_u40_l0keep_v2 76 4" > /dev/null; do sleep 60; done
  finish slim_dual_u40_l0keep_v2 r4l0 MoOP_l0keep
  run_arm slim_lp_m24 r4lp lp_m24 "0 1 2 3" 0,1,2,3
  run_arm baseline r4dn dense "0 1 2 3" 0,1,2,3
) &
wait

python3 cvpr27/code/analyze_rollout4_76.py > "$OUT/analysis.log" 2>&1 && touch "$OUT/analysis.DONE"
stamp "queue finished"
