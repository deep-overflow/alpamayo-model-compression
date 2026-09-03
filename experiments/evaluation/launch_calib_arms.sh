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
# IMP/TAG pick the family: importance_nt500 + dual_nt is the natural official-train
# draw, importance_tr500 + dual_tr the hard scenario-balanced recovery-cache draw the
# first pass used (see plans/2026-09-03_calib-draw-variance.md on why they differ).
IMP=${IMP:-importance_nt500}
TAG=${TAG:-dual_nt}
# make_slim takes ONE card (--gpu is an int), so pinning it to a card this study already
# holds keeps the builds off other members' jobs; reserve_gpu with no --gpu would scan
# from cuda:0 and could take 30 GB out from under a busy card.
BUILD_GPU=${BUILD_GPU:-7}

# arm tag -> importance run supplying the dual scores
declare -A RUNS=()
for b in a b c d e; do RUNS[${TAG}_$b]="${IMP}_$b"; done
for k in 200 300 500; do RUNS[${TAG}_c$k]="${IMP}_c$k"; done
# the card-matched calib_100 (G0b) belongs to whichever family is queued first; adding
# it twice would only re-evaluate the same recipe
[ -f outputs/slim_dual_u40_v2_ada/slim_meta.json ] ||
  RUNS[dual_u40_v2_ada]="importance_v2_ada"

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
      --gpu "$BUILD_GPU" >>"$REPO/logs/build_$tag.log" 2>&1 ||
      echo "!! $tag build failed, see logs/build_$tag.log"
  done
  ;;

queue | append)
  # test500 for every arm; val500 (the `indist` set, drawn from official val) only for
  # the five n=100 blocks, which is where the draw spread is read.
  # `append` adds to a live queue without rewinding the cursor -- use it when workers
  # are already running, or their claimed shards get handed out a second time.
  mode=init
  [ "$1" = append ] && mode=append
  specs=()
  for tag in "${!RUNS[@]}"; do specs+=("$tag=outputs/slim_$tag"); done
  SETS=test bash "$REPO/experiments/evaluation/launch_arms.sh" "$mode" 2 "${specs[@]}"
  blocks=()
  for b in a b c d e; do blocks+=("${TAG}_$b=outputs/slim_${TAG}_$b"); done
  SETS=indist bash "$REPO/experiments/evaluation/launch_arms.sh" append 2 "${blocks[@]}"
  bash "$REPO/experiments/evaluation/launch_arms.sh" status
  ;;

worker)
  bash "$REPO/experiments/evaluation/launch_arms.sh" worker "$2"
  ;;

*)
  echo "usage: $0 {build|queue|append|worker <gpu>}" >&2
  exit 1
  ;;
esac
