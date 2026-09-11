#!/bin/bash
# Open-loop evaluation of the two st2000 arms over all three sets.
#
# Six jobs (2 arms x val500 / test500 / ood_val), pulled from a list by K workers so the
# cards stay busy without a job being handed out twice. test500 comes first for both
# arms: it is where the axis and criterion effects were measured, so a run cut short
# still answers the main question.
#
# The box is shared. A worker takes only a card with nothing else resident -- "enough
# headroom" once put this run on a card another member was using.
set -uo pipefail
K=${K:-2}
REM=/home/cvlab20/project/chan
CHAN=/mnt/dataset1/chan
cd $REM/alpamayo-model-compression
. $REM/cvlab20-server/env.sh
mkdir -p "$CHAN/logs"

JOBS=(
  "dual_u40_st2000 test"       "maxstep11_u40_st2000 test"
  "dual_u40_st2000 indist"     "maxstep11_u40_st2000 indist"
  "dual_u40_st2000 oodval"     "maxstep11_u40_st2000 oodval"
)
Q=$CHAN/logs/st2000_queue.txt
CUR=$CHAN/logs/st2000_cursor
printf '%s\n' "${JOBS[@]}" >"$Q"
echo 0 >"$CUR"
echo "$(date -u '+%F %T') queued ${#JOBS[@]} jobs"

worker() {
  local gpu=$1
  while :; do
    local idx
    idx=$(flock "$CUR" bash -c 'i=$(cat '"$CUR"'); n=$(wc -l < '"$Q"');
          [ "$i" -lt "$n" ] && echo $((i+1)) > '"$CUR"'; echo $i')
    [ "$idx" -ge "${#JOBS[@]}" ] && break
    read -r arm set_name <<<"${JOBS[$idx]}"
    case $set_name in
      oodval) sargs=(--set ood --manifest ood_val) ;;
      *)      sargs=(--set "$set_name") ;;
    esac
    echo "$(date -u '+%H:%M') gpu$gpu -> $arm $set_name"
    .venv/bin/python experiments/evaluation/run_baseline.py \
      "${sargs[@]}" --model "outputs/slim_$arm" --exp-id "${arm}_${set_name}" \
      --gpu "$gpu" --reserve-gb 26 \
      >>"$CHAN/logs/st2000_${arm}_${set_name}.log" 2>&1
  done
  echo "$(date -u '+%H:%M') gpu$gpu done"
}

cards=""
for g in 0 1 2 3 4 5 6 7; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
  [ "$used" -le 16 ] && cards="$cards $g"
  [ "$(echo $cards | wc -w)" -ge "$K" ] && break
done
set -- $cards
[ $# -lt "$K" ] && { echo "REFUSING: fewer than $K idle cards ($cards)"; exit 1; }
echo "workers on cuda:$cards"
for g in $cards; do worker "$g" & sleep 5; done
wait
echo "$(date -u '+%F %T') st2000 evaluation done"
