#!/bin/bash
# Fan run_racfit.py over the free cards: the 36 VLM layers split into one shard per
# GPU, all sharing a single rollout cache so the K-seed CoC generation is paid once.
# plans/2026-08-25_cot-reconstruction.md
#
# Usage:
#   # 1. build the rollout cache once (any card)
#   bash experiments/head_analysis/run_retry_host.sh 30 \
#       experiments/head_analysis/run_racfit.py --gpu 0 --exp-id racfit_v1 --rollout-only
#   # 2. fan the layers out
#   bash experiments/head_analysis/launch_racfit.sh racfit_v1 racfit_v1 "0 1 2 3"
#
# Analysis then merges the shards on the layer axis:
#   python experiments/head_analysis/analyze_racfit.py \
#       --exp-id racfit_v1_l00 racfit_v1_l09 racfit_v1_l18 racfit_v1_l27 --out racfit_v1
set -u
PREFIX=${1:?exp-id prefix}
ROLL=${2:?rollout cache exp-id}
GPUS=${3:-"0 1 2 3"}
NLAYERS=${NLAYERS:-36}
RETRIES=${RETRIES:-240}
REPO=${ALPAMAYO_REPO:-/home/cvlab21/project/chan/alpamayo-model-compression}
EXTRA=${EXTRA:-}

set -- $GPUS
N=$#
PER=$(( (NLAYERS + N - 1) / N ))
mkdir -p "$REPO/logs"

i=0
for gpu in $GPUS; do
  lo=$(( i * PER ))
  hi=$(( lo + PER )); [ "$hi" -gt "$NLAYERS" ] && hi=$NLAYERS
  i=$(( i + 1 ))
  [ "$lo" -ge "$NLAYERS" ] && continue
  tag=$(printf "%s_l%02d" "$PREFIX" "$lo")
  echo "gpu $gpu -> layers $lo..$((hi - 1))  exp-id $tag"
  ALPAMAYO_REPO=$REPO nohup bash "$REPO/experiments/head_analysis/run_retry_host.sh" \
      "$RETRIES" "$REPO/experiments/head_analysis/run_racfit.py" \
      --gpu "$gpu" --exp-id "$tag" --rollouts-from "$ROLL" \
      --layer-start "$lo" --layer-end "$hi" $EXTRA \
      > "$REPO/logs/racfit_${tag}.log" 2>&1 &
  sleep 5
done
wait
echo "all shards finished"
