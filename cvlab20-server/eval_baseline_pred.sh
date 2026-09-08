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
# Our own checkout, not the shared one under $HOME. The account is shared with the whole
# lab and so is /home/cvlab20/project/chan/alpamayo-model-compression: another session
# rsynced its tree over that path mid-run, run_baseline.py lost --save-pred, and eight of
# twelve shards died on "unrecognized arguments" while the four that had already started
# finished fine.
REPO_DIR=${REPO_DIR:-/mnt/dataset1/chan/repo}
cd "$REPO_DIR" || exit 1
. $REM/cvlab20-server/env.sh
export ALPAMAYO_REPO="$REPO_DIR"
mkdir -p "$CHAN/logs"

# SETS lets a partial re-run skip sets that already finished -- test500's four shards
# survived the shared-checkout overwrite and re-running them would cost 18 minutes for
# rows that are already on disk.
SETS=${SETS:-"test indist oodval"}
JOBS=()
for s in $SETS; do
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
    # Empty, not merely roomy. "Free memory is enough" has put a run on a card another
    # member was using before; the standing rule on this shared box is an empty card.
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
      "${sargs[@]}" --model baseline --save-pred \
      --exp-id "${ARM}_${set_name}_sh${shard}" \
      --shard "$shard" --n-shards "$NSH" \
      --gpu "$gpu" --reserve-gb 30 \
      >>"$CHAN/logs/${ARM}_${set_name}_sh${shard}.log" 2>&1
    # capture before anything else runs: in `echo "$(date) ... exit=$?"` the command
    # substitution executes first, so $? is date's status and every job reports exit=0.
    # That is how eight failed shards were reported as successes.
    local rc=$?
    echo "$(date -u '+%H:%M') gpu$gpu $set_name.$shard exit=$rc"
    [ "$rc" -eq 0 ] || echo "  FAILED: $(tail -1 "$CHAN/logs/${ARM}_${set_name}_sh${shard}.log")"
  done
  echo "$(date -u '+%H:%M') gpu$gpu done"
}

for g in $CARDS; do worker "$g" & sleep 5; done
wait

for s in $SETS; do
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
