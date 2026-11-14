#!/bin/bash
# Camera timing mismatch, open loop on test2000 (plans/2026-10-26_camera-timing-openloop.md).
# Shard i of every arm runs on Blackwell card i (0-3), so arm-vs-sync pairs share a card.
# The frame-window cache builds on CPU while the cards run the unshifted (sync) arm; the
# shifted arms wait for its READY marker (all clips present, G0 byte-identical).
#
# Usage: bash experiments/evaluation/launch_camera_timing.sh
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO" || exit 1
export ALPAMAYO_REPO=$REPO   # run_retry_host.sh would otherwise cd to the main checkout
LOG=outputs/camtiming_logs
mkdir -p "$LOG"
CACHE=/mnt/nvme1n1/ad_vla/data/physicalai_av/pre_processed/test_timing
RETRY="bash experiments/head_analysis/run_retry_host.sh"

# arm name | run_baseline.py timing args, in priority order
ARMS=(
  "tm6_lag100|--timing-cams 6 --lag-ms 100"
  "tm0_lag100|--timing-cams 0 --lag-ms 100"
  "tm1_lag100|--timing-cams 1 --lag-ms 100"
  "tm2_lag100|--timing-cams 2 --lag-ms 100"
  "tmall_lag100|--lag-ms 100"
  "tmall_lag300|--lag-ms 300"
  "tmall_st200|--stride-ms 200"
  "tm6_lag300|--timing-cams 6 --lag-ms 300"
  "tm6_lag67|--timing-cams 6 --lag-ms 66.667"
  "tm6_lag200|--timing-cams 6 --lag-ms 200"
  "tmall_st33|--stride-ms 33.333"
)

$RETRY 5 experiments/evaluation/build_timing_cache.py \
  --manifest outputs/eval_sets/test_2000.parquet --cache test_timing --ref-cache test \
  --workers 24 >"$LOG/build_cache.log" 2>&1 &

chain() {
  local i=$1
  $RETRY 30 experiments/evaluation/run_baseline.py --set test --n-indist 2000 --save-pred \
    --exp-id camtiming_sync_test2000 --shard "$i" --n-shards 4 --gpu "$i" \
    >"$LOG/sync_s$i.log" 2>&1
  while [ ! -f "$CACHE/READY" ]; do
    if ! pgrep -f "build_timing_cache.py --manifest" >/dev/null; then
      echo "$(date '+%H:%M:%S') cache build ended without READY; shifted arms not run" \
        >>"$LOG/shard$i.err"
      return 1
    fi
    sleep 60
  done
  for arm in "${ARMS[@]}"; do
    local name=${arm%%|*} targs=${arm#*|}
    # shellcheck disable=SC2086
    $RETRY 30 experiments/evaluation/run_baseline.py --set test --n-indist 2000 --save-pred \
      $targs --exp-id "camtiming_${name}_test2000" --shard "$i" --n-shards 4 --gpu "$i" \
      >"$LOG/${name}_s$i.log" 2>&1
  done
}

for i in 0 1 2 3; do
  chain "$i" &
done
wait
echo "$(date '+%H:%M:%S') camera timing runs done"
