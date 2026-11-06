#!/bin/bash
# Dense arm of the 76-scene x 4-rollout run (cvpr27/plans/2026-11-05_rollout4-76scenes.md §5). The split queue
# (run_rollout4_76_split.sh) finished MoOP l0keep, tyr_off and lp_m24, then died in the 2026-11-06 10:51 KST server
# reboot while dense was still waiting for GPUs 0-3. Same launch as that queue's run_arm/finish, then the analysis.
#
#   setsid nohup bash cvpr27/code/run_rollout4_76_dense.sh > outputs/cvpr27/rollout4_76/queue_dense.log 2>&1 &
set -u
REPO=/home/cvlab21/project/chan/alpamayo-model-compression
RUNS=/home/cvlab21/project/chan/alpasim-runs
OUT=$REPO/outputs/cvpr27/rollout4_76
CSV=$OUT/scenes76_gpu0-3.csv
stamp() { echo "$(date -u '+%F %T') UTC $*"; }
cd "$REPO"

n=0
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1,2,3 | awk '$1 > 10' | wc -l)" = 0 ]; do
  [ $((n % 30)) = 0 ] && stamp "dense: waiting for GPUs 0 1 2 3 to be empty"
  n=$((n + 1)); sleep 60
done
stamp "dense: 76 scenes x 4 rollouts on GPUs 0 1 2 3"
SUITE=public_2601_r4dn PARENT_SUITE=public_2601 PREFIX=r4dn_ DRIVER_OMP_THREADS=8 SCENES_CSV=$CSV \
  bash experiments/head_analysis/launch_alpasim_shards.sh baseline 76 4 "0 1 2 3" > "$OUT/r4dn.log" 2>&1

for c in $(docker ps -a --format '{{.Names}}' | grep "^r4dn_"); do docker rm -f "$c" > /dev/null; done
for net in $(docker network ls --format '{{.Name}}' | grep "^r4dn_"); do
  [ "$(docker network inspect -f '{{len .Containers}}' "$net")" = 0 ] && docker network rm "$net" > /dev/null
done
"$REPO/.venv/bin/python" experiments/head_analysis/merge_alpasim_shards.py --runs-root "$RUNS" \
  --shards r4dn_baseline_sh0 r4dn_baseline_sh1 r4dn_baseline_sh2 r4dn_baseline_sh3 --out r4dn_merged_baseline \
  --expect-scenes 76 > "$OUT/r4dn_merge.log" 2>&1 || stamp "dense: merge failed"
stamp "dense: done ($(python3 -c "import json;print(len(json.load(open('$RUNS/r4dn_merged_baseline/aggregate/results-summary.json'))['rollouts']))" 2>&1 | tail -1) rollouts)"

/mnt/nvme1n1/ad_vla/venvs/alpasim-chan/bin/python cvpr27/code/coc_rollout4_76.py r4dn_merged_baseline > "$OUT/coc_dense.log" 2>&1
python3 cvpr27/code/analyze_rollout4_76.py > "$OUT/analysis.log" 2>&1 && touch "$OUT/analysis.DONE"
stamp "queue finished"
