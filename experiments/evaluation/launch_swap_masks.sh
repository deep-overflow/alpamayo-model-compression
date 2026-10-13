#!/bin/bash
# Evaluate spliced kept sets (make_swap_masks.py and friends), same path and seeds as every
# arm (run_baseline.py), so stored per-clip results of the source arms pair with these.
#
# Usage: bash experiments/evaluation/launch_swap_masks.sh <gpu> <name=model_dir> [...]
#   e.g. launch_swap_masks.sh 4 n1_L0n4=outputs/slim_swap_n1_L0n4
# SET picks the evaluation set:
#   test    (default) official-test clips; MANIFEST (default test_cruise120) picks the
#           subset, test_500 the whole set          -> outputs/swap_<name>_<manifest sans test_>
#   indist  val500                                  -> outputs/swap_<name>_val500
#   ood     OOD-val 262, own rollout only (--no-tf) -> outputs/swap_<name>_oodval
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export ALPAMAYO_REPO=$REPO
cd "$REPO" || exit 1
LOGDIR=${LOGDIR-logs}
SET=${SET-test}
MANIFEST=${MANIFEST-test_cruise120}
mkdir -p "$LOGDIR"
gpu=$1
shift
for spec in "$@"; do
  name=${spec%%=*}
  model=${spec#*=}
  case $SET in
  test) ev=swap_${name}_${MANIFEST#test_}; setargs=(--set test --manifest "$MANIFEST") ;;
  indist) ev=swap_${name}_val500; setargs=(--set indist) ;;
  ood) ev=swap_${name}_oodval; setargs=(--set ood --manifest ood_val --no-tf) ;;
  *) echo "unknown SET $SET" >&2; exit 1 ;;
  esac
  if [ -f "outputs/$ev/summary_s0of1.txt" ]; then echo "skip $ev"; continue; fi
  echo "$(date '+%H:%M:%S') $name ($SET) on gpu $gpu"
  bash experiments/head_analysis/run_retry_host.sh 30 experiments/evaluation/run_baseline.py \
    "${setargs[@]}" --model "$model" --exp-id "$ev" \
    --gpu "$gpu" --reserve-gb 26 >>"$LOGDIR/$ev.log" 2>&1
  echo "$(date '+%H:%M:%S') $name exit=$?"
done
echo "$(date '+%H:%M:%S') gpu $gpu all done"
