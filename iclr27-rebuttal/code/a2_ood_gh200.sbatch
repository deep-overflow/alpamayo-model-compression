#!/bin/bash
#SBATCH --job-name=a2_ood_gh200
#SBATCH --partition=5thEx
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --time=10:00:00
#SBATCH --comment=field=efficientai;appl=pytorch
#SBATCH --output=/appsdata/e1997a06/a2_ood/logs/%x-%j.out
#
# Alpamayo 2 Super u40 -- dense / MoOP / FM-only / CE-only -- on OOD-Val262 and OOD-train1271, one A2 replica per GH200
# (iclr27-rebuttal/plans/2026-11-10_a2-u40-ood-gh200.md). Submit from glogin (launch_a2_ood_gh200.sh does it):
#   sbatch /appsdata/e1997a06/a2_ood/code/amc/iclr27-rebuttal/code/a2_ood_gh200.sbatch
# Everything lives under /appsdata/e1997a06/a2_ood because the group /scratch file quota is ~full; /scratch is only read
# (the aarch64 venv, torch 2.8.0+cu129 / transformers 4.57.1 = A2's pins). alpamayo2_super comes from $A2_DISTILL/src.
#   1. card 0 runs gate G1 on 2 OOD-Val clips (no hooks / all-ones hooks / MoOP masks: ones == none bitwise, masks change
#      the output, dense ends with EOS); a failure stops the job, a pass is remembered in queue/GATE_PASS
#   2. one worker per card claims lines of queue/tasks.txt (`<arm> <set> <shard> <n>`) by mkdir of queue/claimed/<line>,
#      as cvpr27/code/eval_a100.sbatch does; a shard whose summary exists is skipped and the runner resumes per clip,
#      so a resubmitted job picks up where a dead one stopped (claims of jobs no longer running are released at start)
set -uo pipefail
W=/appsdata/e1997a06/a2_ood
R=$W/code/amc
Q=$W/queue
mkdir -p "$W/logs" "$W/outputs" "$Q/claimed"
touch "$Q/tasks.txt" "$Q/done.log" "$Q/failed.log"
export AD_VLA_DATA=$W/data HF_HUB_CACHE=$W/hub HF_HOME=$HOME/.cache/huggingface HF_HUB_OFFLINE=1
export A2_DISTILL=$W/code/a2d CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MPLCONFIGDIR=/tmp/mpl_$USER
PY=/scratch/e1997a06/project/alpamayo-model-compression/.venv/bin/python
RUNNER=$R/iclr27-rebuttal/code/run_a2_openloop.py
PLAN=iclr27-rebuttal/plans/2026-11-10_a2-u40-ood-gh200.md
# worker k runs on the k-th card SLURM granted (cvpr27/code/eval_a100.sbatch: no device cgroup renumbering)
IFS=, read -ra DEVS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
JOB=${SLURM_JOB_ID:-local}
stamp() { echo "$(TZ=Asia/Seoul date '+%F %T') KST $*"; }

declare -A MASKS=([moop]=$W/masks/masks_moop.npz [fm_only]=$W/masks/masks_fm_only.npz [ce_only]=$W/masks/masks_ce_only.npz)
declare -A MANIFEST=([oodval]=ood_val [oodtrain]=ood_train)
declare -A CACHE=([oodval]=ood_a2 [oodtrain]=ood_train_a2)

stamp "job $JOB on $(hostname) | cards ${DEVS[*]} | queue $Q"
nvidia-smi -L

release_stale() {  # claims without an ok line whose owner job is no longer running
  local live d owner
  live=$(squeue -h -u "$USER" -t R -o %i 2>/dev/null) || return 1
  for d in "$Q"/claimed/*; do
    [ -d "$d" ] || continue
    grep -q "ok   \[${d##*/}\]" "$Q/done.log" && continue
    owner=$(cat "$d/job" 2>/dev/null || true)
    if [ -z "$owner" ] || ! grep -qx "$owner" <<< "$live"; then
      rm -f "$d/job"
      rmdir "$d" 2>/dev/null || true
    fi
  done
}
release_stale || echo "squeue unavailable: stale claims are not released"

run_task() {  # <dev> <arm> <set> <shard> <n>
  local dev=$1 arm=$2 set=$3 s=$4 n=$5 extra=()
  local exp=$W/outputs/a2_${set}_${arm}
  [ -f "$exp/summary_s${s}of${n}.txt" ] && { stamp "skip $arm $set s$s (done)"; return 0; }
  [ "$arm" != dense ] && extra=(--masks "${MASKS[$arm]}")
  CUDA_VISIBLE_DEVICES=$dev "$PY" "$RUNNER" --exp-id "$exp" --shard "$s" --n-shards "$n" --plan "$PLAN" \
      --manifest "${MANIFEST[$set]}" --set-name "${MANIFEST[$set]}" --cache-name "${CACHE[$set]}" \
      ${extra[@]+"${extra[@]}"} > "$W/logs/${arm}_${set}_s${s}of${n}_$JOB.log" 2>&1
}

claim_next() {  # prints "<line number> <task>" of the first unclaimed line, having claimed it
  local i=0 line
  while IFS= read -r line; do
    i=$((i + 1))
    case $line in '' | \#*) continue ;; esac
    if mkdir "$Q/claimed/$i" 2>/dev/null; then
      echo "$JOB" > "$Q/claimed/$i/job"
      echo "$i $line"
      return 0
    fi
  done < "$Q/tasks.txt"
  return 1
}

worker() {  # <k>
  local k=$1 dev=${DEVS[$1]} got i task
  while got=$(claim_next); do
    i=${got%% *}
    task=${got#* }
    stamp "card $k (dev $dev) start [$i] $task"
    # shellcheck disable=SC2086  # the task line is split into words on purpose
    if run_task "$dev" $task; then
      echo "$(date -Is) ok   [$i] job $JOB card $k $task" >> "$Q/done.log"
      stamp "card $k done [$i] $task: $(cat "$W/outputs/a2_$(cut -d' ' -f2 <<< "$task")_$(cut -d' ' -f1 <<< "$task")/summary_s$(cut -d' ' -f3 <<< "$task")of$(cut -d' ' -f4 <<< "$task").txt" 2>/dev/null)"
    else
      echo "$(date -Is) FAIL [$i] job $JOB card $k $task" >> "$Q/failed.log"
      stamp "card $k FAILED [$i] $task"
    fi
  done
  stamp "card $k: nothing left to claim"
}

if [ ! -e "$Q/GATE_PASS" ]; then
  stamp "gate G1 on card 0 (dev ${DEVS[0]})"
  if CUDA_VISIBLE_DEVICES=${DEVS[0]} "$PY" "$RUNNER" --exp-id "$W/outputs/a2_oodval_gate" --gate 2 --plan "$PLAN" \
      --manifest ood_val --set-name ood_val --cache-name ood_a2 --masks "${MASKS[moop]}" \
      > "$W/logs/gate_$JOB.log" 2>&1; then
    echo "$JOB $(date -Is)" > "$Q/GATE_PASS"
    stamp "gate G1 PASS: $(grep -m1 'G1 ones==none' "$W/logs/gate_$JOB.log")"
  else
    stamp "gate G1 FAIL -- see $W/logs/gate_$JOB.log; stopping"
    tail -20 "$W/logs/gate_$JOB.log"
    exit 1
  fi
fi

for k in "${!DEVS[@]}"; do worker "$k" & done
wait
stamp "job $JOB done | ok $(grep -c '^.* ok ' "$Q/done.log") | failed $(grep -c . "$Q/failed.log")"
