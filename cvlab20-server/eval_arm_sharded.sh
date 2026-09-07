#!/bin/bash
# Open-loop evaluation of one slim arm over the three frozen sets, sharded across cards.
#
# Work is not evenly split by clip count: an OOD clip costs ~12 s (it is scored twice,
# own rollout and teacher-forced gt_coc) against ~8 s in-distribution. 500+500+262 clips
# is ~3.1 GPU-hours, so four shards per set gives twelve jobs of ~1000/1000/790 s that
# pack onto four cards as 3 jobs each with almost no idle tail -- about 50 min wall,
# against 67 min for the obvious one-set-per-card split.
#
# Each shard writes to its OWN exp dir and the shards are merged at the end. run_baseline
# gives the per-clip rows and the summary a `_s<i>of<n>` suffix, but writes config.json to
# a fixed path, so shards of one set sharing a dir race on that one file.
#
# A worker is bound to its card for the whole run and waits rather than moving to another:
# the box is shared and "enough headroom" has previously put a run on someone else's card.
set -uo pipefail
ARM=${ARM:?set ARM, e.g. tyr_rd_b}
CARDS=${CARDS:-"0 1 2 3"}
NSH=${NSH:-4}
REM=/home/cvlab20/project/chan
CHAN=/mnt/dataset1/chan
cd $REM/alpamayo-model-compression || exit 1
. $REM/cvlab20-server/env.sh
mkdir -p "$CHAN/logs"

# longest first so the tail of the schedule is the short jobs
JOBS=()
for s in test indist oodval; do
  for i in $(seq 0 $((NSH - 1))); do JOBS+=("$s $i"); done
done
Q=$CHAN/logs/${ARM}_queue.txt
CUR=$CHAN/logs/${ARM}_cursor
printf '%s\n' "${JOBS[@]}" >"$Q"
echo 0 >"$CUR"
echo "$(date -u '+%F %T') $ARM: ${#JOBS[@]} jobs ($NSH shards x 3 sets) on cuda:$CARDS"

worker() {
  local gpu=$1
  while :; do
    local idx
    idx=$(flock "$CUR" bash -c 'i=$(cat '"$CUR"'); n=$(wc -l < '"$Q"');
          [ "$i" -lt "$n" ] && echo $((i+1)) > '"$CUR"'; echo $i')
    [ "$idx" -ge "${#JOBS[@]}" ] && break
    read -r set_name shard <<<"${JOBS[$idx]}"
    local used
    until used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null);
          [ -n "$used" ] && [ "$used" -le 16 ]; do
      echo "$(date -u '+%H:%M') gpu$gpu busy (${used:-?} MiB), waiting for $set_name.$shard"
      sleep 300
    done
    case $set_name in
      oodval) sargs=(--set ood --manifest ood_val) ;;
      *)      sargs=(--set "$set_name") ;;
    esac
    echo "$(date -u '+%H:%M') gpu$gpu -> $ARM $set_name shard $shard/$NSH"
    .venv/bin/python experiments/evaluation/run_baseline.py \
      "${sargs[@]}" --model "outputs/slim_$ARM" \
      --exp-id "${ARM}_${set_name}_sh${shard}" \
      --shard "$shard" --n-shards "$NSH" \
      --gpu "$gpu" --reserve-gb 26 \
      >>"$CHAN/logs/${ARM}_${set_name}_sh${shard}.log" 2>&1
    echo "$(date -u '+%H:%M') gpu$gpu $set_name.$shard exit=$?"
  done
  echo "$(date -u '+%H:%M') gpu$gpu done"
}

for g in $CARDS; do worker "$g" & sleep 5; done
wait

# merge: the row files already carry _s<i>of<n>, so they can share one dir untouched
for s in test indist oodval; do
  dst=$CHAN/outputs/${ARM}_${s}
  mkdir -p "$dst"
  n=0
  for i in $(seq 0 $((NSH - 1))); do
    src=$CHAN/outputs/${ARM}_${s}_sh${i}
    [ -d "$src" ] || { echo "MISSING shard $i of $s"; continue; }
    cp -n "$src"/*_s*of*.json "$dst"/ 2>/dev/null
    cp -n "$src"/summary_s*.txt "$dst"/ 2>/dev/null
    [ -f "$dst/config.json" ] || cp "$src/config.json" "$dst"/ 2>/dev/null
    n=$((n + 1))
  done
  got=$(ls "$dst"/*_s*of*.json 2>/dev/null | wc -l)
  echo "$s: merged $n shard dirs -> $got row files in $dst"
done
echo "$(date -u '+%F %T') $ARM evaluation done"
