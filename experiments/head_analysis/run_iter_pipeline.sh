#!/bin/bash
# Chain one criterion's staged-recalibration tail: wait for the orchestrator's final
# masks, build the it3 recipe (--no-state: slim_meta.json only, load_slim re-applies
# the surgery), then run the test_500 open-loop eval (k=8; rows carry ade_rollout_k,
# so the minADE@6 gates reduce to the first six samples in analysis).
#
# Usage: nohup bash experiments/head_analysis/run_iter_pipeline.sh <criterion> <gpu> &
set -u
c=$1
gpu=$2
cd "$(dirname "$0")/../.." || exit 1

until [ -f "outputs/iter_${c}_u40/final_masks.npz" ]; do
  if ! pgrep -f "run_iter_prune.py --criterion ${c} " >/dev/null; then
    echo "$(date '+%H:%M:%S') [${c}] orchestrator gone without masks"; exit 1
  fi
  sleep 300
done

echo "$(date '+%H:%M:%S') [${c}] masks ready -> build recipe"
bash experiments/head_analysis/run_retry_host.sh 2000 experiments/head_analysis/make_slim.py \
  --config "${c}_u40_it3" --out "outputs/slim_${c}_u40_it3" --importance importance_v2 \
  --no-state --gpu "${gpu}" --reserve-gb 30 || exit 1

echo "$(date '+%H:%M:%S') [${c}] build done -> test_500 eval"
bash experiments/head_analysis/run_retry_host.sh 2000 experiments/evaluation/run_baseline.py \
  --set test --model "outputs/slim_${c}_u40_it3" --exp-id "iter_${c}_test" \
  --gpu "${gpu}" --reserve-gb 26 || exit 1

echo "$(date '+%H:%M:%S') [${c}] eval done"
