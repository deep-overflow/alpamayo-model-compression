#!/bin/bash
#SBATCH --job-name=mt-kd2
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
# mtw24_kd2 on NEURON (minitron/plans/2026-10-26_kisti-training-plan.md): all four changes in one
# run -- online teacher on 4 x A100 80GB (FSDP over the node, no HSDP), group-mean KL, LingoQA mixed
# in (16 of every 64 samples), kd_r5 (five R1 windows per clip) for one epoch, checkpoints at
# 25/50/75/100% of the run.
#
# Hold the GPUs first, run later: submitted while the data is still being pushed, it waits for
# outputs/minitron/READY_KD2 (written after the transfer is verified), sizes the per-GPU driving
# micro-batch with 3-step smokes (largest of $MB_CANDIDATES that runs) and trains with it -- smoke
# and training in one allocation, so no second queue wait.
#
# Two days may not cover the run: train_kd stops before a step that would pass $DEADLINE (the job's
# own end time - 1 h, so a shorter --time at submission moves it too), writes
# outputs/minitron/$EXP/state and exits. A second job chained behind the first finds
# that state, skips the wait and the smoke, and resumes with the same micro-batch; it exits at once if
# the first job already wrote ckpt_q100.
#
#   cd /scratch/$USER/project/alpamayo-model-compression
#   sbatch minitron/code/kd_job.sbatch                                  # -> <jobid>
#   sbatch --dependency=afterany:<jobid> minitron/code/kd_job.sbatch    # continuation
#   scancel <jobid>                                         # to release the GPUs
#
# Extra train_kd.py arguments can follow the script name; they reach the full run only.
set -uo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
# shellcheck disable=SC1091
source env.sh

STUDENT=${STUDENT:-outputs/minitron/slim_mtw24}
EXP=${EXP:-mtw24_kd2}
MB_CANDIDATES=${MB_CANDIDATES:-6 4 3 2 1}
MB_VQA=${MB_VQA:-4}
MAX_WAIT_MIN=${MAX_WAIT_MIN:-240}
READY="$ALPAMAYO_REPO/outputs/minitron/READY_KD2"
OUT=outputs/minitron/$EXP
# 1 h before the job's end for the state save; start + 48 h if scontrol gives no EndTime
# (date -d "" is today 00:00, not an error, so convert only a non-empty EndTime)
END_T=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -oP 'EndTime=\K\S+') || END_T=""
END_S=""
[ -n "$END_T" ] && END_S=$(date -d "$END_T" +%s 2>/dev/null)
DEADLINE=${DEADLINE:-$(( ${END_S:-$(( $(date +%s) + 48 * 3600 ))} - 3600 ))}
SETS=outputs/minitron/kd_sets
DATA_ARGS=(--manifest "$SETS/kd_r5.parquet" --vqa-manifest "$SETS/vqa_lingo_train.parquet"
           --vqa-per-step 16 --micro-batch-vqa "$MB_VQA" --kl-groups --epochs 1
           --lingo-root "$AD_VLA_DATA/lingoqa/train")

if [ -e "$OUT/ckpt_q100/slim_state.pt" ]; then
  echo "$EXP already finished ($OUT/ckpt_q100) -- nothing to do"
  exit 0
fi

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
RESUME=()
if [ -e "$OUT/state/step.json" ] && [ -e "$OUT/micro_batch" ]; then
  MB=$(cat "$OUT/micro_batch")
  RESUME=(--resume)
  echo "$(date '+%F %T') resuming $(cat "$OUT/state/step.json") with micro-batch $MB"
fi
for mb in $MB_CANDIDATES; do
  [ -n "$MB" ] && break
  echo "$(date '+%F %T') smoke: micro-batch $mb (VQA $MB_VQA)"
  # no held-out files in the smoke: it sizes memory and speed only
  if run --student "$STUDENT" --exp-id "${EXP}_smoke_mb${mb}" "${DATA_ARGS[@]}" \
         --micro-batch "$mb" --max-steps 3 --heldout 8 --eval-every 1000 --no-save; then
    MB=$mb
    break
  fi
  echo "$(date '+%F %T') micro-batch $mb failed"
done
if [ -z "$MB" ]; then
  echo "no micro-batch in '$MB_CANDIDATES' ran" >&2
  exit 1
fi
mkdir -p "$OUT"
echo "$MB" > "$OUT/micro_batch"
echo "$(date '+%F %T') training with micro-batch $MB, deadline $(date -d "@$DEADLINE" '+%F %T')"
run --student "$STUDENT" --exp-id "$EXP" "${DATA_ARGS[@]}" --micro-batch "$MB" \
    --heldout-clips "$SETS/heldout_clips_r0.json" --vqa-heldout "$SETS/vqa_lingo_heldout.parquet" \
    --save-fracs 0.25 0.5 0.75 1.0 --deadline "$DEADLINE" ${RESUME[@]+"${RESUME[@]}"} "$@"
