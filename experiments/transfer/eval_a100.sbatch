#!/bin/bash
#SBATCH --job-name=eval_a100
#SBATCH --partition=amd_a100nv_8
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=10:00:00
#SBATCH --comment=field=efficientai;appl=pytorch
#SBATCH --output=logs/%x-%j.out
#
# The A100 open-loop track (plans/2026-10-29_kisti-a100-openloop.md): hold four
# A100s and drain a task queue, so the stages after the gate start without a second queue wait.
#
#   /bin/bash experiments/transfer/a100_queue.sh gate     # fill the queue (Datamover is fine)
#   sbatch experiments/transfer/eval_a100.sbatch          # login node
#   /bin/bash experiments/transfer/a100_queue.sh refs     # later, while the job still runs
#
# One worker per card claims lines of outputs/a100/queue/tasks.txt:
#   ol <arm> <set> <shard> <n_shards> [rerun]   rollout rows -> outputs/a100/<arm>_<set>[_rerun]
#   lingo <arm>                                 LingoQA + Lingo-Judge -> outputs/a100/lingo_vqa[_scores]_<arm>
# A line is claimed by mkdir of queue/claimed/<line number> -- atomic on Lustre across nodes,
# which flock is not guaranteed to be (it depends on the mount's flock/localflock option), and
# the queue is written from the Datamover while the workers read it on the compute node.
#
# The job ends once the queue has been empty with nothing running for IDLE_MAX seconds, or as
# soon as queue/STOP exists. It holds the account's whole A100 quota while it runs, so no KISTI
# training job can start -- `a100_queue.sh stop` when done rather than waiting out IDLE_MAX.
#
# WHY every arm of an A100 table is measured here: same-card reruns are bit-identical, but A100
# vs Ada moves dense by <=0.003 m and MoOP by up to 0.018 m minADE@6, and ~5% of clips change
# CoC text (2026-10-29). Report these rows as their own block, never merged into Ada tables.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
# shellcheck disable=SC1091
source env.sh
PY=$ALPAMAYO_REPO/.venv/bin/python
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

declare -A MODEL=(
  [dense]=baseline
  [moop]=outputs/slim_dual_u40_v2
  [moop_l0keep]=outputs/slim_dual_u40_l0keep_v2
)
Q=$ALPAMAYO_REPO/outputs/a100/queue
IDLE_MAX=${IDLE_MAX:-2700}
# Worker k runs on the k-th card SLURM granted. Indexing 0..3 directly would land on cards
# outside the allocation whenever the node does not renumber them (no device cgroup).
IFS=, read -ra DEVS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

stamp() { echo "$(date '+%F %T') $*"; }

mkdir -p "$Q/claimed" "$Q/running"
touch "$Q/tasks.txt" "$Q/done.log" "$Q/failed.log"
rm -f "$Q/running/"* "$Q/STOP"
# A claim without an `ok` line is from a job that died or a task that failed: release it, so
# this job retries it (finished shards are skipped by their rows file anyway).
for d in "$Q"/claimed/*; do
  [ -d "$d" ] || continue
  grep -q "ok   \[${d##*/}\]" "$Q/done.log" || rmdir "$d"
done
touch "$Q/heartbeat"

{
  echo "job ${SLURM_JOB_ID:-?} on $(hostname) $(date -Is)"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  nvidia-smi -L
  echo "driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  "$PY" - <<'PY'
import importlib.metadata as m

import torch

print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
d = m.distribution("alpamayo1_5")
print("alpamayo1_5", d.version, d.read_text("direct_url.json"))
PY
} > "$Q/env_${SLURM_JOB_ID:-local}.txt" 2>&1
stamp "cards ${DEVS[*]} | idle exit ${IDLE_MAX}s | queue $Q"
cat "$Q/env_${SLURM_JOB_ID:-local}.txt"

run_task() {  # <device> <kind> <args...>
  local dev=$1 kind=$2
  shift 2
  case $kind in
    ol)
      local arm=$1 set_name=$2 shard=$3 n=$4 suffix=${5:-}
      local model=${MODEL[$arm]:?unknown arm $arm}
      local exp=a100/${arm}_${set_name}${suffix:+_$suffix} set_args tag
      case $set_name in
        test2000) set_args=(--set test --n-indist 2000) ;;
        oodval) set_args=(--set ood --manifest ood_val) ;;
        *) echo "unknown set $set_name"; return 2 ;;
      esac
      tag=$([ "$model" = baseline ] && echo baseline || basename "$model")
      if [ -f "outputs/$exp/${tag}_s${shard}of${n}.json" ]; then
        stamp "skip: outputs/$exp/${tag}_s${shard}of${n}.json exists"
        return 0
      fi
      CUDA_VISIBLE_DEVICES=$dev "$PY" experiments/evaluation/run_baseline.py \
          "${set_args[@]}" --model "$model" --exp-id "$exp" \
          --shard "$shard" --n-shards "$n" --no-tf --gpu 0 --reserve-gb 40 \
          > "logs/a100_${arm}_${set_name}${suffix:+_$suffix}_s${shard}of${n}.log" 2>&1
      ;;
    lingo)
      local arm=$1
      local model=${MODEL[$arm]:?unknown arm $arm}
      if [ ! -f "outputs/a100/lingo_vqa_$arm/rows.json" ]; then
        CUDA_VISIBLE_DEVICES=$dev "$PY" experiments/lingoqa/run_lingo_vqa.py \
            --arm "$model" --exp-id "a100/lingo_vqa_$arm" --gpu 0 --reserve-gb 40 \
            > "logs/a100_lingo_${arm}_answer.log" 2>&1
      fi
      if [ ! -f "outputs/a100/lingo_vqa_scores_$arm/metrics.json" ]; then
        CUDA_VISIBLE_DEVICES=$dev "$PY" experiments/lingoqa/score_lingo_vqa.py \
            --runs "a100/lingo_vqa_$arm" --exp-id "a100/lingo_vqa_scores_$arm" --gpu 0 \
            > "logs/a100_lingo_${arm}_score.log" 2>&1
      fi
      ;;
    *) echo "unknown task kind $kind"; return 2 ;;
  esac
}

claim_next() {  # prints "<line number> <task>" of the first unclaimed line, having claimed it
  local i=0 line
  while IFS= read -r line || [ -n "$line" ]; do
    i=$((i + 1))
    case $line in '' | \#*) continue ;; esac
    if mkdir "$Q/claimed/$i" 2>/dev/null; then
      echo "$i $line"
      return 0
    fi
  done < "$Q/tasks.txt"
  return 1
}

worker() {  # <k>
  local k=$1 dev=${DEVS[$1]} got i task
  while [ ! -e "$Q/STOP" ]; do
    if got=$(claim_next); then
      i=${got%% *}
      task=${got#* }
      echo "$task" > "$Q/running/$k"
      touch "$Q/heartbeat"
      stamp "card $k (dev $dev) start [$i] $task"
      # shellcheck disable=SC2086  # the task line is split into words on purpose
      if run_task "$dev" $task; then
        echo "$(date -Is) ok   [$i] card $k $task" >> "$Q/done.log"
        stamp "card $k done [$i] $task"
      else
        echo "$(date -Is) FAIL [$i] card $k $task" >> "$Q/failed.log"
        stamp "card $k FAILED [$i] $task"
      fi
      rm -f "$Q/running/$k"
      touch "$Q/heartbeat"
      continue
    fi
    if [ -z "$(ls -A "$Q/running")" ] \
        && [ $(($(date +%s) - $(stat -c %Y "$Q/heartbeat"))) -ge "$IDLE_MAX" ]; then
      stamp "card $k: queue empty and nothing running for ${IDLE_MAX}s -- exiting"
      return 0
    fi
    sleep 30
  done
  stamp "card $k: STOP"
}

for k in "${!DEVS[@]}"; do worker "$k" & done
wait
stamp "done"
