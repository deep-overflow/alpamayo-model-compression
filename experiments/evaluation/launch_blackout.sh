#!/bin/bash
# t0 camera blackout, open loop (plans/2026-10-18_camera-blackout-openloop.md).
# One shard per Ada card (4-7): test500 shard i, then OOD-val262 shard i.
# Card 4 first runs G0: the unmodified path on the first 10 test500 clips, to be diffed
# against baseline_pred_test before that run is reused as the no-blackout reference.
#
# Usage: bash experiments/evaluation/launch_blackout.sh
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO" || exit 1
mkdir -p logs/blackout
RUN="bash experiments/head_analysis/run_retry_host.sh 30 experiments/evaluation/run_baseline.py"

chain() {
  local gpu=$1 i=$2
  if [ "$i" = 0 ]; then
    $RUN --set test --exp-id blackout_g0 --limit 10 --save-pred --gpu "$gpu" \
      >logs/blackout/g0.log 2>&1
  fi
  $RUN --set test --blackout --save-pred --exp-id blackout_test \
    --shard "$i" --n-shards 4 --gpu "$gpu" >"logs/blackout/test_s$i.log" 2>&1
  $RUN --set ood --manifest ood_val --no-tf --blackout --save-pred --exp-id blackout_oodval \
    --shard "$i" --n-shards 4 --gpu "$gpu" >"logs/blackout/oodval_s$i.log" 2>&1
}

i=0
for gpu in 4 5 6 7; do
  chain "$gpu" "$i" &
  i=$((i + 1))
done
wait
echo "blackout runs done"
