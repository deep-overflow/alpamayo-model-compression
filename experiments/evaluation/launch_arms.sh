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
# The checkout this script lives in, resolved from $0 exactly as launch_axis_arms.sh does.
# A worktree session must evaluate with ITS OWN code: the path used to be hardcoded to the
# trunk, so a checkpoint built against a patched slim_lib was loaded by the trunk's
# unpatched one (em100's zero-width MLP hit trunk's old assert). An ALPAMAYO_REPO override
# would still default to the trunk and reproduce that silently whenever it was forgotten,
# so the location is the single source. Exported for run_retry_host.sh, which reads it.
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export ALPAMAYO_REPO=$REPO
cd "$REPO" || exit 1
Q=logs/arms_queue.txt
CUR=logs/arms_cursor
SETS=${SETS-"indist test ood"}
mkdir -p logs

case ${1-} in
init | append)
  mode=$1
  [ "$mode" = init ] && : >"$Q"
  NSH=$2
  shift 2
  for spec in "$@"; do
    for s in $SETS; do
      for i in $(seq 0 $((NSH - 1))); do
        echo "${spec%%=*} ${spec#*=} $s $i $NSH" >>"$Q"
      done
    done
  done
  # only `init` rewinds the cursor: `append` on a live queue must not hand the running
  # workers' shards out a second time (two processes writing one rows file)
  [ "$mode" = init ] && echo 0 >"$CUR"
  echo "queued $(wc -l <"$Q") jobs"
  ;;

worker)
  gpu=$2
  while :; do
    # re-read the length every claim: `append` can grow the queue while workers run,
    # and a length captured once at start would make them quit at the old end
    n=$(wc -l <"$Q")
    # claim one line; the lock covers read-modify-write of the cursor. The length check
    # lives INSIDE the lock: a worker that finds the queue drained must NOT advance the
    # cursor, or every exiting worker eats one future slot and jobs appended later are
    # silently skipped (three dualr_wl shards were lost to this on 2026-08-30)
    idx=$(flock "$CUR" bash -c 'i=$(cat '"$CUR"'); n=$(wc -l < '"$Q"'); [ "$i" -lt "$n" ] && echo $((i + 1)) > '"$CUR"'; echo $i')
    [ "$idx" -ge "$n" ] && break
    read -r tag ckpt set_name shard nsh < <(sed -n "$((idx + 1))p" "$Q")
    echo "$(date '+%H:%M:%S') gpu$gpu -> $tag $set_name shard $shard/$nsh"
    # `oodval` is the 262-clip OOD *validation* draw, not the full 1,533-clip ood set;
    # it is the same --set ood run restricted by its manifest, and it keeps the exp-id
    # suffix the recovery analysis globs for (<tag>_oodval)
    case $set_name in
    oodval) set_args=(--set ood --manifest ood_val) ;;
    *) set_args=(--set "$set_name") ;;
    esac
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
      "${set_args[@]}" "${model_args[@]}" --exp-id "${tag}_${set_name}" \
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
