#!/bin/bash
# Cross-box control for the SLIM path.
#
# The parity check validated the unpruned baseline only; a slim checkpoint takes
# slim_lib's own attention forward (per-head K/V gather, num_key_value_groups=1), which
# that check never exercised. So re-measure here the two calib_100 arms whose cvlab21
# numbers are already known, on the first 50 test clips. Seeds come from the clip id, so
# the same 50 clips are scored on both boxes and the comparison is paired.
#
#   cvlab21 reference (500 clips): maxstep11_u40_v2 0.9351   dual_u40_v2 0.9498
set -uo pipefail
REM=/home/cvlab20/project/chan
CHAN=/mnt/dataset1/chan
cd $REM/alpamayo-model-compression
. $REM/cvlab20-server/env.sh
mkdir -p "$CHAN/logs"

run() {
  local arm=$1 gpu=$2
  .venv/bin/python experiments/evaluation/run_baseline.py \
    --set test --model "outputs/slim_$arm" --exp-id "ctl_${arm}_test50" \
    --limit 50 --gpu "$gpu" --reserve-gb 26 \
    >"$CHAN/logs/ctl_${arm}.log" 2>&1
  echo "$(date -u '+%H:%M') $arm done on gpu$gpu"
}

# only cards with nothing resident at all -- the box is shared
cards=""
for g in 2 3 4 5 6 7 0 1; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
  [ "$used" -le 16 ] && cards="$cards $g"
  [ "$(echo $cards | wc -w)" -ge 2 ] && break
done
set -- $cards
[ $# -lt 2 ] && { echo "REFUSING: fewer than 2 idle cards ($cards)"; exit 1; }
echo "control on cuda:$cards"
run maxstep11_u40_v2 "$1" &
sleep 5
run dual_u40_v2 "$2" &
wait
echo "$(date -u '+%F %T') control done"
