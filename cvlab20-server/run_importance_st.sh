#!/bin/bash
# Importance over the 4,000 stability-curve clips on cvlab20, then merge and curve.
#
#   bash cvlab20-server/run_importance_st.sh "0 2"     # card ids, default "0 2"
#
# NOT run_retry_host.sh: that script exports HF_HOME=$HOME/.cache/huggingface, which on
# this box points at the 94%-full home partition and at a shared stored_tokens file that
# holds other people's credentials. env.sh keeps both under /mnt/dataset1 instead. No
# token is needed anyway -- every weight and config was transferred, and nothing here
# reaches the network.
#
# The box is shared. Cards are checked for free memory before use and the run refuses to
# start on one somebody else is holding.
set -uo pipefail
GPUS=${1:-"0 2"}
N=4000
CHAN=/mnt/dataset1/chan
REPO=/home/cvlab20/project/chan/alpamayo-model-compression
. /home/cvlab20/project/chan/cvlab20-server/env.sh
cd "$REPO" || exit 1
PY=$REPO/.venv/bin/python
LOG=$CHAN/logs
mkdir -p "$LOG"

read -ra CARDS <<<"$GPUS"
K=${#CARDS[@]}
NEED_MIB=42000                      # the pass peaks at 40.5 GB

echo "$(date -u '+%F %T') checking cards: ${CARDS[*]}"
for g in "${CARDS[@]}"; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$g")
  free=$((total - used))
  if [ "$free" -lt "$NEED_MIB" ]; then
    echo "REFUSING: cuda:$g has only ${free} MiB free (need $NEED_MIB) -- someone else is on it"
    exit 1
  fi
  echo "  cuda:$g  ${free} MiB free"
done

for i in "${!CARDS[@]}"; do
  g=${CARDS[$i]}
  nohup "$PY" experiments/head_analysis/run_importance.py \
    --calib-manifest calib_st4000 --cache calib_st --num-clips $N \
    --exp-id "importance_st4000_s$i" --shard "$i" --n-shards "$K" --gpu "$g" \
    >>"$LOG/importance_st_s$i.log" 2>&1 &
  echo "  shard $i -> cuda:$g (pid $!)"
  sleep 5
done
echo "$(date -u '+%F %T') $K shards launched"
wait
echo "$(date -u '+%F %T') shards finished"

shards=""
for i in $(seq 0 $((K - 1))); do shards="$shards importance_st4000_s$i"; done
"$PY" experiments/head_analysis/merge_importance.py --shards $shards \
  --out importance_st4000 --manifest calib_st4000 || { echo "MERGE FAILED"; exit 1; }
"$PY" experiments/evaluation/analyze_stability_curve.py --importance importance_st4000 ||
  { echo "ANALYSIS FAILED -- the importance run and merge are intact, only the curve"; exit 1; }
echo "$(date -u '+%F %T') stability curve complete"
