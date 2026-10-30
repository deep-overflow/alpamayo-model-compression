#!/bin/bash
#SBATCH --job-name=mt-kd2ef
#SBATCH --partition=amd_a100nv_8
#SBATCH --comment=field=efficientai;appl=pytorch
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# mtw24_kd2ef on NEURON (minitron/plans/2026-10-27_kd2-expert-refit.md): the expert + action
# projections of mtw24_kd2's ckpt_q100 re-fitted on its frozen VLM -- velocity KD against the
# online dense teacher on kd_r1 with R0's schedule (3 epochs, the same 64 held-out windows,
# global batch 64, checkpoint per epoch).
#
#   cd /scratch/$USER/project/alpamayo-model-compression
#   sbatch minitron/code/kd_refit_job.sbatch
set -uo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
# shellcheck disable=SC1091
source env.sh

NP=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
if [ "$NP" -gt 4 ]; then
  echo "refusing: $NP GPUs exceeds the 4-A100 allowance for this account." >&2
  exit 1
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS-$(( ${SLURM_CPUS_PER_TASK:-32} / NP ))}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF-expandable_segments:True}
GPUS=$(seq -s' ' 0 $((NP - 1)))
echo "node $(hostname)  gpus $NP  omp $OMP_NUM_THREADS  mem ${SLURM_MEM_PER_NODE:-?}M"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# shellcheck disable=SC2086
srun --unbuffered "$ALPAMAYO_REPO/.venv/bin/torchrun" --standalone --nproc_per_node="$NP" \
    minitron/code/train_kd.py --student outputs/minitron/mtw24_kd2/ckpt_q100 --exp-id mtw24_kd2ef \
    --manifest outputs/minitron/kd_sets/kd_r1.parquet --freeze-vlm --heldout 64 --epochs 3 \
    --micro-batch 8 --gpus $GPUS "$@"
