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
# Hold the 4 GPUs first, run later: submitted while the KD data is still being pushed, it
# waits for outputs/minitron/READY (written after the transfer is verified), then sizes the
# per-GPU micro-batch with 3-step smokes (largest of $MB_CANDIDATES that runs) and trains
# with it -- smoke and training in one allocation, so no second queue wait.
#
#   cd /scratch/$USER/project/alpamayo-model-compression
#   sbatch minitron/code/kd_job.sbatch                      # defaults below
#   STUDENT=outputs/minitron/slim_mtw24 EXP=mtw24_kd sbatch minitron/code/kd_job.sbatch
#   scancel <jobid>                                         # to release the GPUs
#
# Extra train_kd.py arguments can follow the script name; they reach the full run only.
set -uo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
# shellcheck disable=SC1091
source env.sh

STUDENT=${STUDENT:-outputs/minitron/slim_mtw24}
EXP=${EXP:-mtw24_kd}
MB_CANDIDATES=${MB_CANDIDATES:-8 4 2 1}
MAX_WAIT_MIN=${MAX_WAIT_MIN:-240}
READY="$ALPAMAYO_REPO/outputs/minitron/READY"

waited=0
until [ -e "$READY" ]; do
  if [ "$waited" -ge "$MAX_WAIT_MIN" ]; then
    echo "no $READY after $MAX_WAIT_MIN min -- releasing the GPUs" >&2
    exit 1
  fi
  [ $((waited % 10)) -eq 0 ] && echo "$(date '+%F %T') waiting for $READY (${waited} min)"
  sleep 60
  waited=$((waited + 1))
done
echo "$(date '+%F %T') data ready after ${waited} min"

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

run() {
  srun --unbuffered "$ALPAMAYO_REPO/.venv/bin/torchrun" --standalone --nproc_per_node="$NP" \
      minitron/code/train_kd.py "$@" --gpus $GPUS
}

MB=""
for mb in $MB_CANDIDATES; do
  echo "$(date '+%F %T') smoke: micro-batch $mb"
  if run --student "$STUDENT" --exp-id "${EXP}_smoke_mb${mb}" --micro-batch "$mb" \
         --max-steps 3 --heldout 8 --eval-every 1000 --no-save; then
    MB=$mb
    break
  fi
  echo "$(date '+%F %T') micro-batch $mb failed"
done
if [ -z "$MB" ]; then
  echo "no micro-batch in '$MB_CANDIDATES' ran" >&2
  exit 1
fi
echo "$(date '+%F %T') training with micro-batch $MB"
run --student "$STUDENT" --exp-id "$EXP" --micro-batch "$MB" "$@"
