#!/bin/bash
# Open-loop evaluation of slim_tyr_rd_a (Tyr reconstruction calibrated on calib_rd100_a)
# over the three frozen sets, on cvlab20, using GPUs 0 and 1 only.
#
# The arm exists to ask whether the calibration draw moves Tyr the way it moves the
# Taylor arms, so its reference is `slim_tyr_u40_r` -- same config, same budget, same
# axes, calibrated on calib_100 instead. cvlab21 measured that arm at
#
#     test500  mean 0.8731  median 0.5603  CoC degen 0.008
#     val500   mean 0.7608  median 0.5305  CoC degen 0.008
#     ood_val  mean 1.0902  median 0.6276  CoC degen 0.011
#
# and the boxes are bit-identical on both the baseline and the slim path (cvlab20-usage.md
# section 4), so these numbers go in the same table.
#
# test500 runs first: it is the set the calibration work decides on, so a run cut short
# still answers the main question.
#
# One worker per card, each bound to its own card for the whole run. Binding rather than
# picking an idle card at claim time is what makes two workers safe -- a shared "find the
# first free card" scan lets both land on the same card in the seconds before the first
# one's 26 GB reservation shows up in nvidia-smi.
#
# The box is shared, so a worker whose card is busy WAITS rather than moving to another
# one: the cards are pinned by request, and "enough headroom" has previously put a run on
# a card another member was using. Only a card with nothing resident at all is taken.
set -uo pipefail
ARM=${ARM:-tyr_rd_a}
CARDS=${CARDS:-"0 1"}
REM=/home/cvlab20/project/chan
CHAN=/mnt/dataset1/chan
cd $REM/alpamayo-model-compression || exit 1
. $REM/cvlab20-server/env.sh
mkdir -p "$CHAN/logs"

JOBS=("test" "indist" "oodval")
Q=$CHAN/logs/${ARM}_queue.txt
CUR=$CHAN/logs/${ARM}_cursor
printf '%s\n' "${JOBS[@]}" >"$Q"
echo 0 >"$CUR"
echo "$(date -u '+%F %T') queued ${#JOBS[@]} jobs for $ARM on cuda:$CARDS"

free_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null
}

worker() {
  local gpu=$1
  while :; do
    local idx
    idx=$(flock "$CUR" bash -c 'i=$(cat '"$CUR"'); n=$(wc -l < '"$Q"');
          [ "$i" -lt "$n" ] && echo $((i+1)) > '"$CUR"'; echo $i')
    [ "$idx" -ge "${#JOBS[@]}" ] && break
    local set_name=${JOBS[$idx]}
    local used
    until used=$(free_mib "$gpu"); [ -n "$used" ] && [ "$used" -le 16 ]; do
      echo "$(date -u '+%H:%M') gpu$gpu busy (${used:-?} MiB), waiting for $set_name"
      sleep 300
    done
    case $set_name in
      oodval) sargs=(--set ood --manifest ood_val) ;;
      *)      sargs=(--set "$set_name") ;;
    esac
    echo "$(date -u '+%H:%M') gpu$gpu -> $ARM $set_name"
    .venv/bin/python experiments/evaluation/run_baseline.py \
      "${sargs[@]}" --model "outputs/slim_$ARM" --exp-id "${ARM}_${set_name}" \
      --gpu "$gpu" --reserve-gb 26 \
      >>"$CHAN/logs/${ARM}_${set_name}.log" 2>&1
    # capture before anything else runs: in `echo "$(date) ... exit=$?"` the command
    # substitution executes first, so $? is date's status, not the run's
    local rc=$?
    echo "$(date -u '+%H:%M') gpu$gpu $set_name exit=$rc"
  done
  echo "$(date -u '+%H:%M') gpu$gpu done"
}

for g in $CARDS; do worker "$g" & sleep 5; done
wait
echo "$(date -u '+%F %T') $ARM evaluation done"
