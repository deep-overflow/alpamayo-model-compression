#!/bin/bash
# Build the two st2000 arms for open-loop evaluation.
#
#   dual      @ st2000 : coc + traj Taylor, both from importance_st4000_c2000
#   maxstep11 @ st2000 : ten per-step trajectory losses from importance_stepvlm_st2000
#                        plus the CoC NLL from importance_st4000_c2000
#
# Both are selection-only, so --no-state is correct: load_slim reconstructs the weights
# from slim_meta.json and the base model at evaluation time. (The alpasim driver built on
# cvlab21 carries its state file because a container has no base-weight cache.)
#
# The calibration is the first 2,000 clips of calib_st4000, which the greedy draw leaves
# distribution-matched on its own -- weighted L1 0.0015 against 0.0281 for a random 2,000
# from the same pool and 0.0306 for calib_100.
set -euo pipefail
REM=/home/cvlab20/project/chan
cd $REM/alpamayo-model-compression
. $REM/cvlab20-server/env.sh

for g in 0 1 2 3 4 5 6 7; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
  [ "$used" -le 16 ] && { GPU=$g; break; }
done
[ -z "${GPU-}" ] && { echo "REFUSING: no completely idle card"; exit 1; }
echo "building on cuda:$GPU"

build() {   # <config> <out suffix> [extra args...]
  local cfg=$1 name=$2 out=outputs/slim_$2
  shift 2                        # $2 is gone after this, hence `name` above
  if [ -f "$out/slim_meta.json" ]; then echo "$name: exists, skipping"; return; fi
  echo "=== $name  ($cfg)"
  .venv/bin/python experiments/head_analysis/make_slim.py \
    --config "$cfg" --importance importance_st4000_c2000 \
    --out "$out" --no-state --gpu "$GPU" "$@"
}

build dual_u40_v2      dual_u40_st2000
build maxstep11_u40_v2 maxstep11_u40_st2000 --stepvlm importance_stepvlm_st2000
echo "builds done"
