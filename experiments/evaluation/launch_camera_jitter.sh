#!/bin/bash
# Per-frame capture-time jitter, open loop on test2000 (plans/2026-10-26_camera-timing-openloop.md,
# "per-frame jitter"). Every one of the 16 slots gets dt ~ U(-50, +50) ms from the clip id; a
# t0 frame with dt > 0 would be a future frame and is blacked out. Shard i on Blackwell card i,
# paired against camtiming_sync_test2000 (same cards). Needs pre_processed/test_timing READY.
#
#   jit50_black          the design: jitter all 16, future t0 -> black
#   jit50_drop           same draws, future t0 image and its "frame 3" label left out of the prompt
# (`--jitter-future keep` and `--jitter-no-shift` split black into jitter and blackout parts;
# taken off the queue on 2026-10-26 by the user's call.) An arm whose shard summary exists is
# skipped, so a relaunch never doubles up on a shard that already finished.
#
# WAIT_EXP=<exp_id>: card i first waits for that run's shard i to write its summary, so the
# jitter arms queue behind a run already on the cards without a second model competing.
#
# Usage: [WAIT_EXP=camtiming_tm6_lag100_test2000] bash experiments/evaluation/launch_camera_jitter.sh
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO" || exit 1
export ALPAMAYO_REPO=$REPO   # run_retry_host.sh would otherwise cd to the main checkout
LOG=outputs/camtiming_logs
mkdir -p "$LOG"
[ -f /mnt/nvme1n1/ad_vla/data/physicalai_av/pre_processed/test_timing/READY ] || {
  echo "test_timing cache not READY"; exit 1; }
RETRY="bash experiments/head_analysis/run_retry_host.sh"

ARMS=(
  "jit50_black|--jitter-ms 50 --jitter-future black"
  "jit50_drop|--jitter-ms 50 --jitter-future drop"
)

chain() {
  local i=$1
  if [ -n "$WAIT_EXP" ]; then
    while [ ! -f "outputs/$WAIT_EXP/summary_s${i}of4.txt" ]; do sleep 60; done
  fi
  for arm in "${ARMS[@]}"; do
    local name=${arm%%|*} targs=${arm#*|}
    [ -f "outputs/camtiming_${name}_test2000/summary_s${i}of4.txt" ] && continue
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
echo "$(date '+%H:%M:%S') camera jitter runs done"
