#!/bin/bash
#SBATCH --job-name=lingo_arms
#SBATCH --partition=amd_a100nv_8
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --comment=field=efficientai;appl=pytorch
#SBATCH --output=logs/%x-%j.out
#
# Standard-protocol LingoQA for the same arms the open-loop job measures, on the same
# NEURON cards. Submitted alongside eval_arms.sbatch rather than chained to it: LingoQA
# does not read the open-loop outputs, so a dependency would only buy a second queue
# wait. Both jobs write disjoint exp-ids, so they can also run concurrently.
#
#   ARMS="baseline_neuron dual_neuron" sbatch experiments/transfer/lingo_arms.sbatch
#
# Answering and judging run as one job step per arm, one arm per card, so the judge's
# 1.4 GB classifier is loaded once per arm rather than once per queue entry.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
# shellcheck disable=SC1091
source env.sh
PY=$ALPAMAYO_REPO/.venv/bin/python
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

declare -A MODEL=(
  [baseline_neuron]=baseline
  [dual_neuron]=outputs/slim_dual_u40_v2
  [znorm11_neuron]=outputs/slim_znorm11_u40_v2
  [dualfix_neuron]=outputs/slim_dualfix_u40_v2
  [maxstep11_neuron]=outputs/slim_maxstep11_u40_v2
)
ARMS=${ARMS-baseline_neuron dual_neuron}
NGPU=${SLURM_GPUS_ON_NODE:-${NGPU:-2}}
echo "arms: $ARMS | gpus: $NGPU"

gpu=0
pids=()
for arm in $ARMS; do
  (
    run=lingo_vqa_$arm
    CUDA_VISIBLE_DEVICES=$gpu "$PY" experiments/lingoqa/run_lingo_vqa.py \
        --arm "${MODEL[$arm]}" --exp-id "$run" --gpu 0 --reserve-gb 40 \
        > "logs/lingo_${arm}_answer.log" 2>&1
    CUDA_VISIBLE_DEVICES=$gpu "$PY" experiments/lingoqa/score_lingo_vqa.py \
        --runs "$run" --exp-id "lingo_scores_$arm" --gpu 0 \
        > "logs/lingo_${arm}_score.log" 2>&1
  ) &
  pids+=($!)
  gpu=$(((gpu + 1) % NGPU))
  if [ ${#pids[@]} -ge "$NGPU" ]; then wait "${pids[0]}"; pids=("${pids[@]:1}"); fi
done
wait
echo "done $(date -Is)"
