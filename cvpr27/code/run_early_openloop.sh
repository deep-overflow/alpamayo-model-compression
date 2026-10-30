#!/bin/bash
# Step 2 of cvpr27/plans/2026-10-29_early-mlp-cut.md on ONE card (user rule 2026-10-29: GPU 4 only):
# build the three maskfile checkpoints (selection only, --no-state), check them (G0), then the six
# open-loop runs one after another, with the protocol of the stored B0 / dense rows
# (run_baseline.py defaults: k=8, seed 42, max_gen 256).
# Skips a build whose slim_meta.json exists and a run whose summary_s0of1.txt exists, so a rerun resumes.
#
#   bash cvpr27/code/run_early_openloop.sh > outputs/cvpr27/early_mlp/openloop.log 2>&1
set -u
GPU=${GPU:-4}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
RETRY=$REPO/experiments/head_analysis/run_retry_host.sh
export ALPAMAYO_REPO=$REPO
cd "$REPO" || exit 1
stamp() { echo "$(date -u '+%F %T') UTC $*"; }

for spec in "b0 b0" "a10 e10" "a05 e05"; do
  set -- $spec
  out=outputs/cvpr27/slim_early_$1
  [ -f "$out/slim_meta.json" ] && continue
  stamp "build $1"
  bash "$RETRY" 3 experiments/head_analysis/make_slim.py --config "maskfile_cvpr27/early_mlp/m_$2" \
      --out "$out" --no-state --gpu "$GPU" || exit 1
done
"$REPO/.venv/bin/python" cvpr27/code/check_early_builds.py || { stamp "G0 FAIL, stopping"; exit 1; }

for arm in a10 a05; do
  for spec in "test500 --set test --manifest test_500" "val500 --set indist" "oodval --set ood --manifest ood_val"; do
    set -- $spec
    tag=$1
    shift
    d=outputs/cvpr27/early_${arm}_$tag
    [ -f "$d/summary_s0of1.txt" ] && continue
    stamp "eval $arm $tag"
    bash "$RETRY" 3 experiments/evaluation/run_baseline.py --model "outputs/cvpr27/slim_early_$arm" \
        --exp-id "cvpr27/early_${arm}_$tag" --gpu "$GPU" "$@" || exit 1
  done
done
stamp "step 2 runs done"
