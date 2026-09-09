#!/bin/bash
# One slim arm over the three sets with --save-pred, sharded 4 ways per set.
#
# eval_arm_sharded.sh's twin, differing only in --save-pred and the exp-id suffix; kept
# separate so re-running an arm for the collision proxy cannot overwrite the minADE runs
# the reports already cite. Output goes to <ARM>_pred_<set>.
set -uo pipefail
ARM=${ARM:?set ARM}
MODEL=${MODEL:-outputs/slim_$ARM}
CARDS=${CARDS:-"0 1 2 3"}
NSH=${NSH:-4}
SETS=${SETS:-"test indist oodval"}
REM=/home/cvlab20/project/chan
CHAN=/mnt/dataset1/chan
REPO_DIR=${REPO_DIR:-/mnt/dataset1/chan/repo}
cd "$REPO_DIR" || exit 1
. $REM/cvlab20-server/env.sh
export ALPAMAYO_REPO="$REPO_DIR"
mkdir -p "$CHAN/logs"

JOBS=()
for s in $SETS; do
  for i in $(seq 0 $((NSH - 1))); do JOBS+=("$s $i"); done
done
Q=$CHAN/logs/${ARM}_pred_queue.txt
CUR=$CHAN/logs/${ARM}_pred_cursor
printf '%s\n' "${JOBS[@]}" >"$Q"
echo 0 >"$CUR"
echo "$(date -u '+%F %T') ${ARM}_pred: ${#JOBS[@]} jobs on cuda:$CARDS  model=$MODEL"

worker() {
  local gpu=$1
  while :; do
    local idx
    idx=$(flock "$CUR" bash -c 'i=$(cat '"$CUR"'); n=$(wc -l < '"$Q"');
          [ "$i" -lt "$n" ] && echo $((i+1)) > '"$CUR"'; echo $i')
    [ "$idx" -ge "${#JOBS[@]}" ] && break
    read -r set_name shard <<<"${JOBS[$idx]}"
    # empty, not merely roomy -- the standing rule on this shared box
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
    echo "$(date -u '+%H:%M') gpu$gpu -> ${ARM}_pred $set_name shard $shard/$NSH"
    .venv/bin/python experiments/evaluation/run_baseline.py \
      "${sargs[@]}" --model "$MODEL" --save-pred \
      --exp-id "${ARM}_pred_${set_name}_sh${shard}" \
      --shard "$shard" --n-shards "$NSH" \
      --gpu "$gpu" --reserve-gb 26 \
      >>"$CHAN/logs/${ARM}_pred_${set_name}_sh${shard}.log" 2>&1
    # capture before anything else runs: a command substitution in the echo would clobber $?
    local rc=$?
    echo "$(date -u '+%H:%M') gpu$gpu $set_name.$shard exit=$rc"
    [ "$rc" -eq 0 ] || echo "  FAILED: $(tail -1 "$CHAN/logs/${ARM}_pred_${set_name}_sh${shard}.log")"
  done
  echo "$(date -u '+%H:%M') gpu$gpu done"
}

for g in $CARDS; do worker "$g" & sleep 5; done
wait

for s in $SETS; do
  dst=$CHAN/outputs/${ARM}_pred_${s}
  mkdir -p "$dst"
  for i in $(seq 0 $((NSH - 1))); do
    src=$CHAN/outputs/${ARM}_pred_${s}_sh${i}
    [ -d "$src" ] || { echo "MISSING shard $i of $s"; continue; }
    cp -n "$src"/*_s*of*.json "$dst"/ 2>/dev/null
    cp -n "$src"/summary_s*.txt "$dst"/ 2>/dev/null
    [ -f "$dst/config.json" ] || cp "$src/config.json" "$dst"/ 2>/dev/null
  done
  echo "$s: $(ls "$dst"/*_s*of*.json 2>/dev/null | wc -l) row files, $(du -sh "$dst" | cut -f1)"
done
echo "$(date -u '+%F %T') ${ARM}_pred done"
