#!/bin/bash
# Four rollouts per scene on the 76 scenes picked from s600 (MoOP l0keep clean, tyr_off or lp_m24 with an at-fault
# collision or offroad) for four arms, one after another on GPUs 0-3 only (user, 2026-11-05)
# (cvpr27/plans/2026-11-05_rollout4-76scenes.md). Each scene sits on the same card for every arm
# (outputs/cvpr27/rollout4_76/scenes76_gpu0-3.csv, 19 per card). After the last arm: cvpr27/code/analyze_rollout4_76.py.
#
#   bash cvpr27/code/run_rollout4_76.sh > outputs/cvpr27/rollout4_76/queue.log 2>&1
set -u
REPO=/home/cvlab21/project/chan/alpamayo-model-compression
RUNS=/home/cvlab21/project/chan/alpasim-runs
OUT=$REPO/outputs/cvpr27/rollout4_76
CSV=$OUT/scenes76_gpu0-3.csv
stamp() { echo "$(date -u '+%F %T') UTC $*"; }
cd "$REPO"

for spec in "slim_dual_u40_l0keep_v2 r4l0 MoOP_l0keep" "slim_tyr_off r4ty tyr_off" "slim_lp_m24 r4lp lp_m24" \
            "baseline r4dn dense"; do
  set -- $spec
  n=0
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1,2,3 | awk '$1 > 10' | wc -l)" = 0 ]; do
    [ $((n % 30)) = 0 ] && stamp "$3: waiting for GPUs 0-3 to be empty"
    n=$((n + 1)); sleep 60
  done
  stamp "$3: 76 scenes x 4 rollouts on GPUs 0-3"
  SUITE=public_2601_$2 PARENT_SUITE=public_2601 PREFIX=$2_ DRIVER_OMP_THREADS=8 SCENES_CSV=$CSV \
    bash experiments/head_analysis/launch_alpasim_shards.sh "$1" 76 4 "0 1 2 3" > "$OUT/$2.log" 2>&1
  for c in $(docker ps -a --format '{{.Names}}' | grep "^$2_"); do docker rm -f "$c" > /dev/null; done
  for net in $(docker network ls --format '{{.Name}}' | grep "^$2_"); do
    [ "$(docker network inspect -f '{{len .Containers}}' "$net")" = 0 ] && docker network rm "$net" > /dev/null
  done
  "$REPO/.venv/bin/python" experiments/head_analysis/merge_alpasim_shards.py --runs-root "$RUNS" \
    --shards $(for i in 0 1 2 3; do echo "$2_$1_sh$i"; done) --out "$2_merged_$1" --expect-scenes 76 \
    > "$OUT/$2_merge.log" 2>&1 || stamp "$3: merge failed"
  stamp "$3: done ($(python3 -c "import json;print(len(json.load(open('$RUNS/$2_merged_$1/aggregate/results-summary.json'))['rollouts']))" 2>&1 | tail -1) rollouts)"
done

python3 cvpr27/code/analyze_rollout4_76.py > "$OUT/analysis.log" 2>&1 && touch "$OUT/analysis.DONE"
stamp "queue finished"
