#!/bin/bash
# Fill, inspect or stop the queue that eval_a100.sbatch drains. Runs on NEURON (the Datamover
# is fine: it only touches files under /scratch), from the repo root or anywhere else.
#
#   /bin/bash cvpr27/code/a100_queue.sh gate     # dense test2000 x4 + shard-0 rerun,
#                                                         # dense OOD-val x2, LingoQA dense
#   /bin/bash cvpr27/code/a100_queue.sh refs     # both MoOP arms, after the gate passes
#   /bin/bash cvpr27/code/a100_queue.sh status
#   /bin/bash cvpr27/code/a100_queue.sh stop     # workers exit after their current task
#
# Lines are appended longest first, so four cards finish close together. A line already in
# the queue is not appended twice: two copies could be claimed at once and write one rows file
# from two processes.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
Q=$REPO/outputs/cvpr27/a100/queue
mkdir -p "$Q/claimed" "$Q/running"
touch "$Q/tasks.txt" "$Q/done.log" "$Q/failed.log"

add() {  # add <task line>
  if grep -qxF "$*" "$Q/tasks.txt"; then
    echo "  already queued: $*"
  else
    echo "$*" >> "$Q/tasks.txt"
    echo "  + $*"
  fi
}

case ${1:-status} in
  gate)
    # The rerun (G1) writes its own exp-id, dense_test2000_rerun, so it need not wait for the
    # first pass of shard 0; it is fifth only so that the four first-pass shards start first.
    for s in 0 1 2 3; do add ol dense test2000 "$s" 4; done
    add ol dense test2000 0 4 rerun
    for s in 0 1; do add ol dense oodval "$s" 2; done
    add lingo dense
    ;;
  refs)
    for arm in moop moop_l0keep; do
      for s in 0 1 2 3; do add ol "$arm" test2000 "$s" 4; done
    done
    for arm in moop moop_l0keep; do
      for s in 0 1; do add ol "$arm" oodval "$s" 2; done
    done
    for arm in moop moop_l0keep; do add lingo "$arm"; done
    ;;
  status)
    total=$(grep -cvE '^(#|$)' "$Q/tasks.txt" || true)
    claimed=$(find "$Q/claimed" -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "tasks $total | claimed $claimed | ok $(grep -c ' ok ' "$Q/done.log" || true)" \
         "| failed $(grep -c FAIL "$Q/failed.log" || true)"
    echo "running (per job; a dead job's leftovers stay until cleaned):"
    for f in "$Q"/running/*/*; do
      [ -f "$f" ] || continue
      j=${f%/*}
      echo "  job ${j##*/} card ${f##*/}: $(cat "$f")"
    done
    echo "last done:"
    tail -n 5 "$Q/done.log" | sed 's/^/  /'
    if [ -s "$Q/failed.log" ]; then echo "failed:"; sed 's/^/  /' "$Q/failed.log"; fi
    [ -e "$Q/STOP" ] && echo "STOP set $(date -r "$Q/STOP" '+%F %T') (stops jobs started before it)"
    for h in "$Q"/heartbeat_*; do
      [ -e "$h" ] && echo "heartbeat ${h##*/heartbeat_}: $(date -r "$h" '+%F %T')"
    done
    ;;
  stop)
    touch "$Q/STOP"
    echo "STOP set: every job started before now exits after its current tasks"
    ;;
  *)
    echo "usage: a100_queue.sh gate|refs|status|stop" >&2
    exit 2
    ;;
esac
