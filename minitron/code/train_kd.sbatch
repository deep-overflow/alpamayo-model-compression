#!/bin/bash
#SBATCH --job-name=mt-kd
#SBATCH --partition=amd_a100nv_8
#SBATCH --comment=field=efficientai;appl=pytorch
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Minitron KD recovery on NEURON (minitron/plans/2026-10-23_plan.md, D10-3). Submit from the
# login node, after `push_neuron.sh <USER> code weights minitron` and the usual bootstrap:
#
#   cd /scratch/$USER/project/alpamayo-model-compression
#   sbatch minitron/code/train_kd.sbatch --student outputs/minitron/slim_mtw24 \
#          --exp-id mtw24_kd --micro-batch 4
#
# Smoke first, to pick --micro-batch (the 48 GB box peaked at 40.5 GiB/rank with 1):
#   sbatch --time=01:00:00 minitron/code/train_kd.sbatch --student outputs/minitron/slim_mtw24 \
#          --exp-id mtw24_kd_smoke --micro-batch 4 --max-steps 3 --no-save
#
# Anything after the script name goes straight to train_kd.py. The global batch (64) must be
# world x micro-batch x accumulation exactly; train_kd.py asserts it.
#
# --mem: each rank stages one bf16 model on CPU at a time (teacher 22 GB, then the student
# 17 GB, cast to fp32 one FSDP unit at a time), so four ranks peak near 90 GB; 320G leaves
# room for the npz prefetch threads. NEURON's default (~481 MB per CPU) OOM-kills at load.
# --comment is mandatory on NEURON ("field=...;appl=..."; see train_recover.sbatch).
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
# env.sh (written by bootstrap_neuron.sh) carries AD_VLA_DATA / HF_HUB_CACHE / HF_HOME /
# ALPAMAYO_REPO for this box's layout.
# shellcheck disable=SC1091
source env.sh

NP=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
if [ "$NP" -gt 4 ]; then
  echo "refusing: $NP GPUs exceeds the 4-A100 allowance for this account." >&2
  exit 1
fi
# One intra-op thread per core per rank would put 4x32 busy-waiting threads on 32 cores.
export OMP_NUM_THREADS=${OMP_NUM_THREADS-$(( ${SLURM_CPUS_PER_TASK:-32} / NP ))}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF-expandable_segments:True}

GPUS=$(seq -s' ' 0 $((NP - 1)))
echo "node $(hostname)  gpus $NP  omp $OMP_NUM_THREADS  mem ${SLURM_MEM_PER_NODE:-?}M"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

srun --unbuffered "$ALPAMAYO_REPO/.venv/bin/torchrun" \
    --standalone --nproc_per_node="$NP" \
    minitron/code/train_kd.py \
    "$@" \
    --gpus $GPUS
