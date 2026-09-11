#!/bin/bash
# test500 for the axis-allocation arm. Two cards, per the standing instruction, and only
# ones that read as free -- the box is shared.
#
# The baseline (maxstep11_u40_v2) is NOT re-run: cvlab21 already has its 500 rows, and
# today's parity check showed the two boxes agree bit for bit on the baseline path.
set -euo pipefail
REM=/home/cvlab20/project/chan
CHAN=/mnt/dataset1/chan
cd $REM/alpamayo-model-compression
. $REM/cvlab20-server/env.sh
NEED=28000
K=2

cards=""
for g in 0 1 2 3 4 5 6 7; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$g")
  [ $((total - used)) -ge $NEED ] && cards="$cards $g"
  [ "$(echo $cards | wc -w)" -ge $K ] && break
done
set -- $cards
[ $# -lt $K ] && { echo "REFUSING: fewer than $K free cards ($cards)"; exit 1; }
echo "test500 on cuda:$1 and cuda:$2"

i=0
for g in $cards; do
  nohup .venv/bin/python experiments/evaluation/run_baseline.py \
    --set test --model outputs/slim_maxstep11_u40_qcut4_v2 \
    --exp-id maxstep11_qcut4_test --shard $i --n-shards $K --gpu "$g" --reserve-gb 26 \
    >>"$CHAN/logs/qcut4_s$i.log" 2>&1 &
  i=$((i + 1))
  sleep 5
done
wait
echo "qcut4 test500 done"
