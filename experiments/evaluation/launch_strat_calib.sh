#!/bin/bash
# Stage 2 of plans/2026-09-11_action-stratified-calib.md: nine dual_u40_v2 arms that differ
# only in the calibration draw rule (rd / se / su x seeds a b c).
#
#   importance <gpu>   Taylor importance on each set's 100 clips (~9 min each, 40.5 GB,
#                      so one empty Ada card). Sets already measured are skipped.
#   slim <gpu>         make_slim --no-state for every set whose importance exists.
#   eval               queue test500 for every built arm (2 shards each) through
#                      launch_arms.sh; then start `launch_arms.sh worker <gpu>` per free card.
#
# Every arm keeps the shipped recipe: dual_u40_v2, jlens_v2, expert and KV untouched. Only
# --importance changes. Runs from this checkout (ALPAMAYO_REPO), so a worktree evaluates
# with its own code.
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export ALPAMAYO_REPO=$REPO
cd "$REPO" || exit 1
mkdir -p logs

# tag  manifest  cache
SETS="rd_a calib_rd100_a calib_rd_a
rd_b calib_rd100_b calib_rd_b
rd_c calib_rd100_c calib_strat
se_a calib_se100_a calib_strat
se_b calib_se100_b calib_strat
se_c calib_se100_c calib_strat
su_a calib_su100_a calib_strat
su_b calib_su100_b calib_strat
su_c calib_su100_c calib_strat"

case ${1-} in
importance)
  gpu=$2
  echo "$SETS" | while read -r tag man cache; do
    imp=importance_${tag/_/100_}
    if [ -f "outputs/$imp/importance.npz" ]; then
      echo "skip $imp (exists)"; continue
    fi
    [ -f "outputs/eval_sets/$man.parquet" ] || { echo "no manifest $man"; continue; }
    echo "$(date '+%H:%M:%S') importance $tag on gpu $gpu"
    bash experiments/head_analysis/run_retry_host.sh "${RETRIES-480}" \
      experiments/head_analysis/run_importance.py \
      --calib-manifest "$man" --cache "$cache" --num-clips 100 \
      --exp-id "$imp" --gpu "$gpu" >>"logs/$imp.log" 2>&1
    echo "$(date '+%H:%M:%S') $imp exit=$?"
  done
  ;;

slim)
  gpu=$2
  echo "$SETS" | while read -r tag man cache; do
    imp=importance_${tag/_/100_}
    out=outputs/slim_dual_$tag
    if [ -f "$out/slim_meta.json" ]; then
      echo "skip $out (exists)"; continue
    fi
    [ -f "outputs/$imp/importance.npz" ] || { echo "no importance for $tag"; continue; }
    echo "$(date '+%H:%M:%S') slim $tag on gpu $gpu"
    bash experiments/head_analysis/run_retry_host.sh "${RETRIES-480}" \
      experiments/head_analysis/make_slim.py --config dual_u40_v2 \
      --importance "$imp" --jlens jlens_v2 --out "$out" --no-state --gpu "$gpu" \
      >>"logs/slim_dual_$tag.log" 2>&1
    echo "$(date '+%H:%M:%S') slim_dual_$tag exit=$?"
  done
  ;;

eval)
  specs=()
  while read -r tag man cache; do
    [ -f "outputs/slim_dual_$tag/slim_meta.json" ] && specs+=("dual_$tag=outputs/slim_dual_$tag")
  done <<<"$SETS"
  [ ${#specs[@]} -gt 0 ] || { echo "no built arms"; exit 1; }
  SETS=test bash experiments/evaluation/launch_arms.sh "${MODE-init}" 2 "${specs[@]}"
  echo "now: bash experiments/evaluation/launch_arms.sh worker <gpu> &   (one per free Ada card)"
  ;;

*)
  echo "usage: $0 {importance <gpu>|slim <gpu>|eval}" >&2
  exit 1
  ;;
esac
