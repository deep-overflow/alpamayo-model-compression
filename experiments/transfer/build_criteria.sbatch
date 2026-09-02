#!/bin/bash
#SBATCH --job-name=build_criteria
#SBATCH --partition=amd_a100nv_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --comment=field=efficientai;appl=pytorch
#SBATCH --output=logs/%x-%j.out
#
# Build the pruning recipes for the criterion comparison ON NEURON, so the evaluation
# that follows never has to ship a 17 GB state file either way
# (plans/2026-08-31_znorm11-neuron-eval.md).
#
#   sbatch experiments/transfer/build_criteria.sbatch                 # all three
#   sbatch experiments/transfer/build_criteria.sbatch dualfix_u40_v2  # one
#
# DEBUG PARTITION. NEURON's interactive/debug queue is short (check `sinfo -o "%P %l %D"`
# and `scontrol show partition <name>` for MaxTime before relying on it). Each build is
# one model load plus a few seconds of surgery -- 8-12 min wall clock on an A100 -- so
# three builds fit a 1 h slot comfortably, but submit them as separate jobs if the debug
# limit is 30 min. To run on the debug queue instead of the default:
#     sbatch --partition=<debug-partition> --time=00:30:00 ... build_criteria.sbatch <cfg>
#
# These configs are SELECTION-ONLY (no weight rewriting), so --no-state is correct and
# each recipe is ~3 MB rather than 17 GB; slim_lib.load_slim reconstructs the weights
# from the base model at evaluation time. Do NOT copy this flag to a tyr/dualr/dualrc
# config -- make_slim refuses those, and for good reason.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
# shellcheck disable=SC1091
source experiments/transfer/env.sh

CONFIGS=${*:-znorm11_u40_v2 dualfix_u40_v2 dual_u40_v2}

for cfg in $CONFIGS; do
  echo "=== $cfg  ($(date -Is))"
  # --importance must be the ADA reference the per-step file was normalised against:
  # mixing importance_v2 (Blackwell) with importance_stepvlm_v1 (Ada) would make the
  # criterion a two-factor change. The evaluation cards are a separate question.
  "$PY" experiments/head_analysis/make_slim.py \
      --config "$cfg" \
      --importance importance_v2_ada \
      --stepvlm importance_stepvlm_v1 \
      --no-state \
      --reserve-gb 40 \
      --out "outputs/slim_${cfg%_u40_v2}_neuron"
done
echo "done $(date -Is)"
