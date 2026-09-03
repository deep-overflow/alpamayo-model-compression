#!/bin/bash
# Stage 4-5 of plans/2026-09-03_calib-draw-variance.md: build the nine dual_u40_v2
# recipes that differ only in which clips the Taylor scores came from, then queue their
# open-loop evaluation.
#
# `--no-state` is correct here and saves 17 GB per arm: dual is selection-only, and
# load_slim reconstructs the identical model from slim_meta.json by slicing the base
# weights (slim_lib.py:210-217). Only the weight-rewriting configs need slim_state.pt,
# and make_slim.py:564 refuses --no-state for those.
#
# Evaluation stays on Ada 4-7 -- two runs agree bitwise only within one architecture, and
# every arm of this comparison has to sit on the same one.
#
# Usage:
#   bash experiments/evaluation/launch_calib_arms.sh build     # recipes, one card
#   bash experiments/evaluation/launch_calib_arms.sh queue     # fill the job queue
#   bash experiments/evaluation/launch_calib_arms.sh worker 4  # one per free Ada card
set -u
REPO=${ALPAMAYO_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
export ALPAMAYO_REPO=$REPO
cd "$REPO" || exit 1
IMP=${IMP:-importance_tr500}
GPU=${GPU:-4,5,6,7}

# arm tag -> importance run supplying the dual scores
declare -A RUNS=(
  [dual_tr_a]="${IMP}_a" [dual_tr_b]="${IMP}_b" [dual_tr_c]="${IMP}_c"
  [dual_tr_d]="${IMP}_d" [dual_tr_e]="${IMP}_e"
  [dual_tr_c200]="${IMP}_c200" [dual_tr_c300]="${IMP}_c300" [dual_tr_c500]="${IMP}_c500"
  [dual_u40_v2_ada]="importance_v2_ada"
)

case ${1-} in
build)
  for tag in "${!RUNS[@]}"; do
    out=outputs/slim_$tag
    if [ -f "$out/slim_meta.json" ]; then
      echo "$tag: recipe exists, skipping"
      continue
    fi
    echo "=== $tag from ${RUNS[$tag]}"
    bash "$REPO/experiments/head_analysis/run_retry_host.sh" 60 \
      "$REPO/experiments/head_analysis/make_slim.py" \
      --config dual_u40_v2 --importance "${RUNS[$tag]}" --out "$out" --no-state \
      --gpu "${GPU%%,*}" >>"$REPO/logs/build_$tag.log" 2>&1 ||
      echo "!! $tag build failed, see logs/build_$tag.log"
  done
  ;;

queue)
  # test500 for every arm; val500 (the `indist` set, drawn from official val) only for
  # the five n=100 blocks, which is where the draw spread is read
  specs=()
  for tag in "${!RUNS[@]}"; do specs+=("$tag=outputs/slim_$tag"); done
  SETS=test bash "$REPO/experiments/evaluation/launch_arms.sh" init 2 "${specs[@]}"
  blocks=()
  for b in a b c d e; do blocks+=("dual_tr_$b=outputs/slim_dual_tr_$b"); done
  SETS=indist bash "$REPO/experiments/evaluation/launch_arms.sh" append 2 "${blocks[@]}"
  bash "$REPO/experiments/evaluation/launch_arms.sh" status
  ;;

worker)
  bash "$REPO/experiments/evaluation/launch_arms.sh" worker "$2"
  ;;

*)
  echo "usage: $0 {build|queue|worker <gpu>}" >&2
  exit 1
  ;;
esac
