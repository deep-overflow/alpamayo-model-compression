#!/bin/bash
# Open-loop evaluation of one or more slim checkpoints over the baseline sets.
#
# This box is shared and cards free up at unpredictable times, so jobs are pulled from
# a flock-guarded queue rather than assigned to GPUs up front: start with the cards that
# are free now, add a worker later with `worker <gpu>` and it picks up the rest.
#
# `run_baseline.py` resumes from the rows file it already wrote, so a killed worker
# loses at most the clips since its last 10-clip checkpoint.
#
# Usage:
#   bash experiments/evaluation/launch_arms.sh init 4 \
#       dual_u40_v2=outputs/slim_dual_u40_v2 jtraj_u40_v2=outputs/slim_jtraj_u40_v2
#   bash experiments/evaluation/launch_arms.sh worker 5 &     # one per free GPU
#   bash experiments/evaluation/launch_arms.sh status
set -u
REPO=/home/cvlab21/project/chan/alpamayo-model-compression
cd "$REPO" || exit 1
Q=logs/arms_queue.txt
CUR=logs/arms_cursor
SETS=${SETS-"indist test ood"}
mkdir -p logs

case ${1-} in
init | append)
  [ "$1" = init ] && : >"$Q"
  NSH=$2
  shift 2
  for spec in "$@"; do
    for s in $SETS; do
      for i in $(seq 0 $((NSH - 1))); do
        echo "${spec%%=*} ${spec#*=} $s $i $NSH" >>"$Q"
      done
    done
  done
  echo 0 >"$CUR"
  echo "queued $(wc -l <"$Q") jobs"
  ;;

worker)
  gpu=$2
  while :; do
    # re-read the length every claim: `append` can grow the queue while workers run,
    # and a length captured once at start would make them quit at the old end
    n=$(wc -l <"$Q")
    # claim one line; the lock covers read-modify-write of the cursor
    idx=$(flock "$CUR" bash -c 'i=$(cat '"$CUR"'); echo $((i + 1)) > '"$CUR"'; echo $i')
    [ "$idx" -ge "$n" ] && break
    read -r tag ckpt set_name shard nsh < <(sed -n "$((idx + 1))p" "$Q")
    echo "$(date '+%H:%M:%S') gpu$gpu -> $tag $set_name shard $shard/$nsh"
    # a spec of the form `quant:<path.npz>` is a bit allocation applied to the unpruned
    # model rather than a checkpoint directory; everything else is a slim checkpoint
    if [[ "$ckpt" == quant:* ]]; then
      model_args=(--model baseline --quant "${ckpt#quant:}")
    else
      model_args=(--model "$ckpt")
    fi
    # this box is shared with other members' runs; retry for hours rather than give
    # up, or a busy stretch drains the whole queue without evaluating anything.
    # 26 GiB fits the slim model (16.8 GiB of weights) with the same headroom the
    # 30 GiB default leaves the 22.2 GiB baseline.
    bash experiments/head_analysis/run_retry_host.sh "${RETRIES-480}" \
      experiments/evaluation/run_baseline.py \
      --set "$set_name" "${model_args[@]}" --exp-id "${tag}_${set_name}" \
      --shard "$shard" --n-shards "$nsh" --gpu "$gpu" --reserve-gb "${RESERVE-26}" \
      >>"logs/eval_${tag}_${set_name}_s${shard}.log" 2>&1
  done
  echo "$(date '+%H:%M:%S') gpu$gpu done"
  ;;

status)
  echo "cursor $(cat "$CUR") / $(wc -l <"$Q") jobs claimed"
  awk '{print $1, $3}' "$Q" | sort -u | while read -r tag set_name; do
    n=$("$REPO/.venv/bin/python" - "$tag" "$set_name" <<'PY'
import json, sys
from pathlib import Path
tag, s = sys.argv[1], sys.argv[2]
d = Path("outputs") / f"{tag}_{s}"
ids = set()
for p in d.glob("*_s*of*.json"):
    ids |= {r["clip_id"] for r in json.loads(p.read_text())}
print(len(ids))
PY
    )
    printf '  %-14s %-7s %5s clips\n' "$tag" "$set_name" "$n"
  done
  ;;

*)
  echo "usage: $0 {init <n_shards> tag=ckpt ...|worker <gpu>|status}" >&2
  exit 1
  ;;
esac
