#!/bin/bash
# Unpruned baseline over the three sets WITH the sampled paths kept (--save-pred).
#
# Phase 1 of plans/2026-09-08_openloop-collision-proxy.md. run_baseline computes pred_k
# and throws it away, so no existing run can be scored for collisions; the baseline goes
# first because it is what says whether the metric discriminates at all (gate G1).
#
# Same 12-shard layout as eval_arm_sharded.sh, but the model is `baseline` rather than a
# slim dir, so no checkpoint is needed on this box.
set -uo pipefail
ARM=${ARM:-baseline_pred}
CARDS=${CARDS:-"0 1 2 3"}
NSH=${NSH:-4}
REM=/home/cvlab20/project/chan
CHAN=/mnt/dataset1/chan
cd $REM/alpamayo-model-compression || exit 1
. $REM/cvlab20-server/env.sh
mkdir -p "$CHAN/logs"

JOBS=()
for s in test indist oodval; do
  for i in $(seq 0 $((NSH - 1))); do JOBS+=("$s $i"); done
done
Q=$CHAN/logs/${ARM}_queue.txt
CUR=$CHAN/logs/${ARM}_cursor
printf '%s\n' "${JOBS[@]}" >"$Q"
echo 0 >"$CUR"
echo "$(date -u '+%F %T') $ARM: ${#JOBS[@]} jobs on cuda:$CARDS"

worker() {
  local gpu=$1
  while :; do
    local idx
    idx=$(flock "$CUR" bash -c 'i=$(cat '"$CUR"'); n=$(wc -l < '"$Q"');
          [ "$i" -lt "$n" ] && echo $((i+1)) > '"$CUR"'; echo $i')
    [ "$idx" -ge "${#JOBS[@]}" ] && break
    read -r set_name shard <<<"${JOBS[$idx]}"
    # Headroom, not emptiness. A "completely idle card" rule cannot be satisfied here:
    # one member's process holds 352 MiB on ALL EIGHT cards, so `used <= 16` waits
    # forever while the cards sit unused. The repo's own runner (run_importance_st.sh)
    # already gates on free memory, which is the quantity that actually decides whether
    # a run fits and whether it would crowd anyone: 30 GB reservation + margin.
    local used total free
    until read -r used total <<<"$(nvidia-smi --query-gpu=memory.used,memory.total \
              --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr ',' ' ')"
          [ -n "$used" ] && free=$((total - used)) && [ "$free" -ge 36000 ]; do
      echo "$(date -u '+%H:%M') gpu$gpu 여유 ${free:-?} MiB < 36000, waiting for $set_name.$shard"
      sleep 300
    done
    case $set_name in
      oodval) sargs=(--set ood --manifest ood_val) ;;
      *)      sargs=(--set "$set_name") ;;
    esac
    echo "$(date -u '+%H:%M') gpu$gpu -> $ARM $set_name shard $shard/$NSH"
    .venv/bin/python experiments/evaluation/run_baseline.py \
      "${sargs[@]}" --model baseline --save-pred \
      --exp-id "${ARM}_${set_name}_sh${shard}" \
      --shard "$shard" --n-shards "$NSH" \
      --gpu "$gpu" --reserve-gb 30 \
      >>"$CHAN/logs/${ARM}_${set_name}_sh${shard}.log" 2>&1
    echo "$(date -u '+%H:%M') gpu$gpu $set_name.$shard exit=$?"
  done
  echo "$(date -u '+%H:%M') gpu$gpu done"
}

for g in $CARDS; do worker "$g" & sleep 5; done
wait

for s in test indist oodval; do
  dst=$CHAN/outputs/${ARM}_${s}
  mkdir -p "$dst"
  for i in $(seq 0 $((NSH - 1))); do
    src=$CHAN/outputs/${ARM}_${s}_sh${i}
    [ -d "$src" ] || { echo "MISSING shard $i of $s"; continue; }
    cp -n "$src"/*_s*of*.json "$dst"/ 2>/dev/null
    cp -n "$src"/summary_s*.txt "$dst"/ 2>/dev/null
    [ -f "$dst/config.json" ] || cp "$src/config.json" "$dst"/ 2>/dev/null
  done
  echo "$s: $(ls "$dst"/*_s*of*.json 2>/dev/null | wc -l) row files, $(du -sh "$dst" | cut -f1)"
done
echo "$(date -u '+%F %T') $ARM done"
