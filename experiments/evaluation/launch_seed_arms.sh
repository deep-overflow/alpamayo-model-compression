#!/bin/bash
# Calibration set fixed (calib_100); only the seeds INSIDE the importance measurement move.
#
#   ref      roll 0, noise 0   the shipped estimator, re-measured (run-to-run floor
#                              against importance_v2_ada / dual_ada_u40_v2_test)
#   c1..c4   roll j, noise 0   a different sampled CoC per clip, same flow-matching noise
#   n1..n4   roll 0, noise m   the same CoC, different flow-matching noise
#
# Each arm is importance (100 clips, ~12 min) -> make_slim --no-state -> test500 (2 shards,
# ~70 min), chained on ONE Ada card so nothing waits on another arm. Everything else is the
# shipped dual_u40_v2 recipe. Arms already complete are skipped at each stage.
#
# Usage: bash experiments/evaluation/launch_seed_arms.sh <gpu> <arm> [arm ...]
#        e.g. launch_seed_arms.sh 4 ref c3 n3
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export ALPAMAYO_REPO=$REPO
cd "$REPO" || exit 1
LOGDIR=${LOGDIR-logs}
mkdir -p "$LOGDIR"
gpu=$1
shift

for arm in "$@"; do
  roll=0
  noise=0
  case $arm in
  ref) ;;
  c[0-9]*) roll=${arm#c} ;;
  n[0-9]*) noise=${arm#n} ;;
  *) echo "unknown arm $arm" >&2; continue ;;
  esac
  imp=importance_seed_$arm
  slim=outputs/slim_dual_seed_$arm
  ev=dual_seed_${arm}_test

  if ! grep -q '"n_clips": 100,' "outputs/$imp/metrics.json" 2>/dev/null; then
    echo "$(date '+%H:%M:%S') $arm importance (roll $roll, noise $noise) on gpu $gpu"
    bash experiments/head_analysis/run_retry_host.sh 30 experiments/head_analysis/run_importance.py \
      --calib-manifest calib_100 --cache calib --num-clips 100 --exp-id "$imp" \
      --roll-offset "$roll" --noise-offset "$noise" --no-perclip --gpu "$gpu" \
      >>"$LOGDIR/$imp.log" 2>&1
  fi
  grep -q '"n_clips": 100,' "outputs/$imp/metrics.json" 2>/dev/null \
    || { echo "$arm: importance incomplete, skipping"; continue; }

  if [ ! -f "$slim/slim_meta.json" ]; then
    echo "$(date '+%H:%M:%S') $arm slim"
    bash experiments/head_analysis/run_retry_host.sh 30 experiments/head_analysis/make_slim.py \
      --config dual_u40_v2 --importance "$imp" --jlens jlens_v2 --out "$slim" \
      --no-state --gpu "$gpu" >>"$LOGDIR/slim_dual_seed_$arm.log" 2>&1
  fi
  [ -f "$slim/slim_meta.json" ] || { echo "$arm: slim missing, skipping"; continue; }

  for sh in 0 1; do
    [ -f "outputs/$ev/summary_s${sh}of2.txt" ] && continue
    echo "$(date '+%H:%M:%S') $arm test500 shard $sh/2"
    bash experiments/head_analysis/run_retry_host.sh 30 experiments/evaluation/run_baseline.py \
      --set test --model "$slim" --exp-id "$ev" --shard "$sh" --n-shards 2 \
      --gpu "$gpu" --reserve-gb 26 >>"$LOGDIR/eval_${ev}_s$sh.log" 2>&1
  done
  echo "$(date '+%H:%M:%S') $arm done"
done
echo "$(date '+%H:%M:%S') gpu $gpu all done"
