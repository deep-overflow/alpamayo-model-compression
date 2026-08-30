#!/bin/bash
# Worktree-local twin of launch_arms.sh for the VLM axis arms.
#
# launch_arms.sh keeps its queue at the MAIN checkout's logs/arms_queue.txt, and `init`
# truncates it. On 2026-08-30 a parallel session init'd that file while this track's 18
# jobs were still queued, dropping 15 of them, so this track carries its own queue and
# cursor inside the worktree. Same claim protocol (flock on the cursor, one line per
# claim, length re-read each round so `append` works), same run_baseline flags.
#
# Usage:
#   bash experiments/evaluation/launch_axis_arms.sh init <n_shards> tag=ckpt [tag=ckpt ...]
#   bash experiments/evaluation/launch_axis_arms.sh worker <gpu> &
#   bash experiments/evaluation/launch_axis_arms.sh status
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO" || exit 1
Q=logs/axis_queue.txt
CUR=logs/axis_cursor
SETS=${SETS-"indist test oodval"}
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
  [ "$1" = init ] && echo 0 >"$CUR"
  [ -f "$CUR" ] || echo 0 >"$CUR"
  echo "queued $(wc -l <"$Q") jobs, cursor $(cat "$CUR")"
  ;;

worker)
  gpu=$2
  while :; do
    n=$(wc -l <"$Q")
    idx=$(flock "$CUR" bash -c 'i=$(cat '"$CUR"'); echo $((i + 1)) > '"$CUR"'; echo $i')
    [ "$idx" -ge "$n" ] && break
    read -r tag ckpt set_name shard nsh < <(sed -n "$((idx + 1))p" "$Q")
    echo "$(date '+%H:%M:%S') gpu$gpu -> $tag $set_name shard $shard/$nsh"
    case $set_name in
    oodval) set_args=(--set ood --manifest ood_val) ;;
    *) set_args=(--set "$set_name") ;;
    esac
    ALPAMAYO_REPO=$REPO bash experiments/head_analysis/run_retry_host.sh "${RETRIES-480}" \
      experiments/evaluation/run_baseline.py \
      "${set_args[@]}" --model "$ckpt" --exp-id "${tag}_${set_name}" \
      --shard "$shard" --n-shards "$nsh" --gpu "$gpu" --reserve-gb "${RESERVE-26}" \
      >>"logs/eval_${tag}_${set_name}_s${shard}.log" 2>&1
  done
  echo "$(date '+%H:%M:%S') gpu$gpu done"
  ;;

status)
  echo "cursor $(cat "$CUR" 2>/dev/null) of $(wc -l <"$Q" 2>/dev/null) jobs"
  cat -n "$Q"
  ;;

*)
  echo "usage: $0 {init|append} <n_shards> tag=ckpt ... | worker <gpu> | status" >&2
  exit 1
  ;;
esac
