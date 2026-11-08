#!/bin/bash
# iclr27-rebuttal/plans/2026-11-08_comb-rules-pre-l0keep-test500.md on GPU 4 only (session rule 2026-10-29):
# wait until GPU 4 is empty, build raw-mean / raw-max without l0keep (selection only, --no-state), gate B1, then test500 for both.
# Skips a build whose slim_meta.json exists and a run whose summary_s0of1.txt exists, so a rerun resumes.
#
#   bash iclr27-rebuttal/code/run_comb_pre_l0keep.sh > outputs/iclr27-rebuttal/comb_pre_l0keep_run.log 2>&1
set -u
GPU=${GPU:-4}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
RETRY=$REPO/experiments/head_analysis/run_retry_host.sh
export ALPAMAYO_REPO=$REPO
cd "$REPO" || exit 1
stamp() { echo "$(date -d '+9 hours' '+%F %T') KST $*"; }

stamp "waiting for GPU $GPU to be empty"
free_twice=0
while [ "$free_twice" -lt 2 ]; do
  used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [ "$used" -lt 100 ]; then free_twice=$((free_twice + 1)); else free_twice=0; fi
  [ "$free_twice" -lt 2 ] && sleep 120
done
stamp "GPU $GPU empty, starting"

for cfg in rawabsavg_u40_v2 rawabsmean_u40_v2; do
  out=outputs/iclr27-rebuttal/slim_$cfg
  [ -f "$out/slim_meta.json" ] && continue
  stamp "build $cfg"
  bash "$RETRY" 3 experiments/head_analysis/make_slim.py --config "$cfg" --importance importance_v2 \
      --out "$out" --no-state --gpu "$GPU" || { stamp "build $cfg failed"; exit 1; }
done
"$REPO/.venv/bin/python" iclr27-rebuttal/code/check_comb_pre_l0keep_builds.py || { stamp "B1 FAIL, stopping"; exit 1; }

for cfg in rawabsavg_u40_v2 rawabsmean_u40_v2; do
  d=outputs/iclr27-rebuttal/${cfg}_test500
  [ -f "$d/summary_s0of1.txt" ] && continue
  stamp "eval $cfg test500"
  bash "$RETRY" 3 experiments/evaluation/run_baseline.py --model "outputs/iclr27-rebuttal/slim_$cfg" \
      --exp-id "iclr27-rebuttal/${cfg}_test500" --gpu "$GPU" --set test --manifest test_500 || { stamp "eval $cfg failed"; exit 1; }
done
stamp "all done"
