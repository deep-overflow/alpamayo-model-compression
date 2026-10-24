#!/bin/bash
# Per-camera t0 blackout, open loop: one fixed camera blacked out on every clip.
# One camera per Ada card (4-7), each running test500 then OOD-val262 (762 clips, ~2 h).
#
# Usage: bash experiments/evaluation/launch_blackout_cam.sh
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO" || exit 1
mkdir -p logs/blackout_cam
RUN="bash experiments/head_analysis/run_retry_host.sh 30 experiments/evaluation/run_baseline.py"

chain() {
  local gpu=$1 cam=$2
  $RUN --set test --blackout-cam "$cam" --save-pred --exp-id "blackout_cam${cam}_test" \
    --gpu "$gpu" >"logs/blackout_cam/cam${cam}_test.log" 2>&1
  $RUN --set ood --manifest ood_val --no-tf --blackout-cam "$cam" --save-pred \
    --exp-id "blackout_cam${cam}_oodval" --gpu "$gpu" >"logs/blackout_cam/cam${cam}_oodval.log" 2>&1
}

chain 4 0 &
chain 5 1 &
chain 6 2 &
chain 7 6 &
wait
echo "per-camera blackout runs done"
