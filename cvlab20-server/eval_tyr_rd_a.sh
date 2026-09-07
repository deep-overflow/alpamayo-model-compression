#!/bin/bash
# Open-loop evaluation of slim_tyr_rd_a (Tyr reconstruction calibrated on calib_rd100_a)
# over the three frozen sets, on cvlab20.
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
# The box is shared and its cards free up unpredictably, so a worker that cannot find an
# idle card WAITS rather than taking a busy one -- "enough headroom" once put a run on a
# card another member was using.  Only cards with nothing resident at all are taken.
set -uo pipefail
ARM=${ARM:-tyr_rd_a}
K=${K:-1}
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
echo "$(date -u '+%F %T') queued ${#JOBS[@]} jobs for $ARM"

idle_card() {
  for g in 2 6 7 0 1 3 4 5; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null)
    [ -n "$used" ] && [ "$used" -le 16 ] && { echo "$g"; return 0; }
  done
  return 1
}

worker() {
  local wid=$1
  while :; do
    local idx
    idx=$(flock "$CUR" bash -c 'i=$(cat '"$CUR"'); n=$(wc -l < '"$Q"');
          [ "$i" -lt "$n" ] && echo $((i+1)) > '"$CUR"'; echo $i')
    [ "$idx" -ge "${#JOBS[@]}" ] && break
    local set_name=${JOBS[$idx]}
    # hold the job but wait for a card rather than crowding someone else's
    local gpu
    until gpu=$(idle_card); do
      echo "$(date -u '+%H:%M') w$wid $set_name: no idle card, waiting"
      sleep 300
    done
    case $set_name in
      oodval) sargs=(--set ood --manifest ood_val) ;;
      *)      sargs=(--set "$set_name") ;;
    esac
    echo "$(date -u '+%H:%M') w$wid gpu$gpu -> $ARM $set_name"
    .venv/bin/python experiments/evaluation/run_baseline.py \
      "${sargs[@]}" --model "outputs/slim_$ARM" --exp-id "${ARM}_${set_name}" \
      --gpu "$gpu" --reserve-gb 26 \
      >>"$CHAN/logs/${ARM}_${set_name}.log" 2>&1
    echo "$(date -u '+%H:%M') w$wid $set_name exit=$?"
  done
  echo "$(date -u '+%H:%M') w$wid done"
}

for i in $(seq 1 "$K"); do worker "$i" & sleep 5; done
wait
echo "$(date -u '+%F %T') $ARM evaluation done"
