#!/bin/bash
# Every camera's t0 frame excluded, open loop on test2000 (plans/2026-10-26_camera-timing-openloop.md,
# "t0 exclusion"). Card 0 is held by another job, so three shards on Blackwell 1-3 (user, 2026-11-03);
# results are paired per clip with camtiming_sync_test2000 (G0: the unperturbed path on this layout
# is bitwise identical to it, 10/10).
#
# Usage: bash experiments/evaluation/launch_t0_exclude.sh
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO" || exit 1
export ALPAMAYO_REPO=$REPO   # run_retry_host.sh would otherwise cd to the main checkout
LOG=outputs/camtiming_logs
mkdir -p "$LOG"
RETRY="bash experiments/head_analysis/run_retry_host.sh"

chain() {
  local shard=$1 gpu=$2
  for mode in black drop; do
    [ -f "outputs/camtiming_t0x_${mode}_test2000/summary_s${shard}of3.txt" ] && continue
    $RETRY 30 experiments/evaluation/run_baseline.py --set test --n-indist 2000 --save-pred \
      --t0-exclude "$mode" --exp-id "camtiming_t0x_${mode}_test2000" \
      --shard "$shard" --n-shards 3 --gpu "$gpu" >"$LOG/t0x_${mode}_s$shard.log" 2>&1
  done
}

chain 0 1 &
chain 1 2 &
chain 2 3 &
wait
echo "$(date '+%H:%M:%S') t0 exclusion runs done"
