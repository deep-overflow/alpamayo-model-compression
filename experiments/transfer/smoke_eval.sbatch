#!/bin/bash
#SBATCH --job-name=smoke_eval
#SBATCH --partition=amd_a100nv_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=00:40:00
#SBATCH --comment=field=efficientai;appl=pytorch
#SBATCH --output=logs/%x-%j.out
#
# Batch wrapper for smoke_eval.sh. Interactive salloc requests sat PENDING(Priority) for
# 45+ minutes on this cluster while short batch jobs backfill, so the smoke test is
# submitted rather than held. Chain the real run to it:
#
#   id=$(sbatch --parsable experiments/transfer/smoke_eval.sbatch)
#   ARMS="baseline_neuron dual_neuron" sbatch --dependency=afterok:$id --gres=gpu:2 \
#       --cpus-per-task=16 --mem=120G experiments/transfer/eval_arms.sbatch
#
# afterok means a failing smoke leaves the long job unstarted, which is the point --
# the expensive mistake here is burning hours on a broken configuration.
set -u
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
/bin/bash experiments/transfer/smoke_eval.sh
