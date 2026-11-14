#!/bin/bash
# Alpamayo 2 Super under camera capture-time perturbations, open loop on test500
# (plans/2026-10-26_camera-timing-openloop.md, "Alpamayo 2"). Two replicas of two Blackwell
# cards: shard 0 on 0,1 and shard 1 on 2,3 for every arm, so arm-vs-sync pairs share cards.
# Arms in order: sync, jitter + future t0 black, jitter + future t0 dropped. An arm whose shard
# summary exists is skipped; a failed arm is retried (rows resume per clip).
#
# Usage: bash experiments/evaluation/launch_a2_timing.sh
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO" || exit 1
PY=/mnt/nvme1n1/ad_vla/venvs/alpamayo2/bin/python
LOG=outputs/camtiming_logs
mkdir -p "$LOG"
for c in test_timing test_timing_rear; do
  [ -f "/mnt/nvme1n1/ad_vla/data/physicalai_av/pre_processed/$c/READY" ] || {
    echo "$c cache not READY"; exit 1; }
done
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=$HOME/.cache/huggingface
export HF_HUB_CACHE=/mnt/nvme1n1/ad_vla/cache/hub
export HF_HUB_OFFLINE=1   # weights are pinned and cached; no Hub calls mid-run

chain() {
  local shard=$1 cards=$2
  for arm in sync black drop; do
    local name=$arm
    [ "$arm" != sync ] && name=jit50_$arm
    [ -f "outputs/a2timing_${name}_test500/summary_s${shard}of2.txt" ] && continue
    for attempt in 1 2 3; do
      echo "$(date '+%H:%M:%S') $arm shard $shard attempt $attempt" >>"$LOG/a2_${name}_s$shard.log"
      CUDA_VISIBLE_DEVICES=$cards $PY experiments/evaluation/run_a2_timing.py --arm "$arm" \
        --shard "$shard" --n-shards 2 >>"$LOG/a2_${name}_s$shard.log" 2>&1 && break
      sleep 60
    done
  done
}

chain 0 0,1 &
chain 1 2,3 &
wait
echo "$(date '+%H:%M:%S') Alpamayo 2 timing runs done"
