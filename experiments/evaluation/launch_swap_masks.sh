#!/bin/bash
# Evaluate spliced kept sets (make_swap_masks.py) on the test500 cruise subset, same path
# and seeds as every arm (run_baseline.py), so stored per-clip results of the source arms
# pair with these directly.
#
# Usage: bash experiments/evaluation/launch_swap_masks.sh <gpu> <name=model_dir> [...]
#   e.g. launch_swap_masks.sh 4 n1_L0n4=outputs/slim_swap_n1_L0n4
# MANIFEST (default test_cruise120) picks the clip subset.
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export ALPAMAYO_REPO=$REPO
cd "$REPO" || exit 1
LOGDIR=${LOGDIR-logs}
MANIFEST=${MANIFEST-test_cruise120}
mkdir -p "$LOGDIR"
gpu=$1
shift
for spec in "$@"; do
  name=${spec%%=*}
  model=${spec#*=}
  ev=swap_${name}_${MANIFEST#test_}
  if [ -f "outputs/$ev/summary_s0of1.txt" ]; then echo "skip $ev"; continue; fi
  echo "$(date '+%H:%M:%S') $name on gpu $gpu"
  bash experiments/head_analysis/run_retry_host.sh 30 experiments/evaluation/run_baseline.py \
    --set test --manifest "$MANIFEST" --model "$model" --exp-id "$ev" \
    --gpu "$gpu" --reserve-gb 26 >>"$LOGDIR/$ev.log" 2>&1
  echo "$(date '+%H:%M:%S') $name exit=$?"
done
echo "$(date '+%H:%M:%S') gpu $gpu all done"
