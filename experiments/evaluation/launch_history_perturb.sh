#!/bin/bash
# Run run_history_perturb.py for one or more arms, one after another, on one GPU.
#
# Usage: bash experiments/evaluation/launch_history_perturb.sh <gpu> <exp_id> <model> [<model> ...]
#   model: baseline | outputs/<slim dir>
# One call per card; evaluation stays on the Ada cards (4-7), and only on cards that are empty.
#
# CONDS selects the conditions (default: the runner's own, the registered main run). The addenda were
#   CONDS="clean sim5 brake5" ... <gpu> histperturb_v1b baseline outputs/slim_dual_u40_v2
#   CONDS="clean sim2 sim3"   ... <gpu> histperturb_v1c <all six arms>
# analyze_history_perturb.py --exp-id histperturb_v1 picks up the `b` and `c` directories by name.
set -o pipefail
GPU=$1
EXP=$2
shift 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ALPAMAYO_REPO=${ALPAMAYO_REPO:-"$(cd "$HERE/../.." && pwd)"}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
mkdir -p "$ALPAMAYO_REPO/outputs/$EXP"
EXTRA=()
[ -n "${CONDS:-}" ] && read -r -a C <<< "$CONDS" && EXTRA=(--conds "${C[@]}")
for m in "$@"; do
  bash "$ALPAMAYO_REPO/experiments/head_analysis/run_retry_host.sh" 3 \
      experiments/evaluation/run_history_perturb.py --model "$m" --gpu "$GPU" --exp-id "$EXP" "${EXTRA[@]}" 2>&1 \
      | tee -a "$ALPAMAYO_REPO/outputs/$EXP/log_gpu$GPU.txt" || exit 1
done
