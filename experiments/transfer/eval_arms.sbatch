#!/bin/bash
#SBATCH --job-name=eval_arms
#SBATCH --partition=amd_a100nv_8
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --comment=field=efficientai;appl=pytorch
#SBATCH --output=logs/%x-%j.out
#
# The four-arm criterion comparison, measured entirely on NEURON cards
# (plans/2026-08-31_znorm11-neuron-eval.md).
#
#   sbatch experiments/transfer/eval_arms.sbatch
#   sbatch experiments/transfer/eval_arms.sbatch indist            # one set first
#
# WHY the reference arms are re-measured here. Paired comparison is only valid within
# one GPU architecture -- the same clip and seed gave 0.286 on Ada and 0.291 on
# Blackwell, and 3-4% of clips produce different CoC text across architectures. Every
# published arm was measured on Ada, so none of them can be paired with an A100 run:
# `baseline` and `dual` are therefore evaluated here too, and the whole table is
# internally paired and separately labelled. Do not merge these numbers into the Ada
# tables; report them as their own block.
#
# 12 jobs (4 arms x 3 sets) over 4 GPUs, two shards each: ~1,262 clips per arm at
# 8-13 s/clip is ~3.2 GPU-h per arm, ~13 GPU-h total, so ~3.5 h on four cards.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
# shellcheck disable=SC1091
source experiments/transfer/env.sh

SETS=${*:-indist test oodval}
declare -A MODEL=(
  [baseline_neuron]=baseline
  [dual_neuron]=outputs/slim_dual_neuron
  [znorm11_neuron]=outputs/slim_znorm11_neuron
  [dualfix_neuron]=outputs/slim_dualfix_neuron
)

gpu=0
pids=()
for arm in baseline_neuron dual_neuron znorm11_neuron dualfix_neuron; do
  for set_name in $SETS; do
    case $set_name in
      oodval) set_args=(--set ood --manifest ood_val) ;;
      *) set_args=(--set "$set_name") ;;
    esac
    for shard in 0 1; do
      (
        CUDA_VISIBLE_DEVICES=$gpu "$PY" experiments/evaluation/run_baseline.py \
            "${set_args[@]}" --model "${MODEL[$arm]}" \
            --exp-id "${arm}_${set_name}" --shard "$shard" --n-shards 2 \
            --gpu 0 --reserve-gb 40 \
            > "logs/eval_${arm}_${set_name}_s${shard}.log" 2>&1
      ) &
      pids+=($!)
      gpu=$(((gpu + 1) % 4))
      # four concurrent jobs, one per card
      if [ ${#pids[@]} -ge 4 ]; then wait "${pids[0]}"; pids=("${pids[@]:1}"); fi
    done
  done
done
wait
echo "done $(date -Is)"
