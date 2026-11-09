#!/bin/bash
# One GPU pair works through its shard of the three A2 arms, dense first
# (iclr27-rebuttal/plans/2026-11-09_a2-moop-test500.md). Resumable: the runner skips clips already in its rows file,
# and an arm whose shard summary exists is skipped.
#   CARDS=4,5 SHARD=0 bash iclr27-rebuttal/code/run_a2_worker.sh > outputs/iclr27-rebuttal/a2_worker_s0.log 2>&1
#   CARDS=6,7 SHARD=1 bash iclr27-rebuttal/code/run_a2_worker.sh > outputs/iclr27-rebuttal/a2_worker_s1.log 2>&1
# Extra runner args (e.g. --gate 3 --masks ...) can be passed with ONE=1 to run a single job instead of the loop.
set -u
# env.sh first: it sets its own REPO (= alpamayo2-distill), so this script's root is ROOT, computed after it
source /home/cvlab21/project/chan/alpamayo2-distill/experiments/oodval/env.sh
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export HF_HUB_OFFLINE=1  # weights are cached at the pinned revision; the full_right token is invalid since 2026-11-09
export CUDA_VISIBLE_DEVICES=${CARDS:?set CARDS, e.g. 4,5}
PY=/mnt/nvme1n1/ad_vla/venvs/alpamayo2/bin/python
RUNNER=$ROOT/iclr27-rebuttal/code/run_a2_openloop.py
[ -f "$RUNNER" ] || { echo "runner not found: $RUNNER"; exit 1; }
cd "$ROOT" || exit 1
stamp() { echo "$(date -d '+9 hours' '+%F %T') KST $*"; }

if [ "${ONE:-0}" = "1" ]; then exec "$PY" "$RUNNER" "$@"; fi

M=outputs/iclr27-rebuttal/a2_moop
for spec in "dense -" "moop $M/masks_moop.npz" "moop_l0keep $M/masks_moop_l0keep.npz"; do
  set -- $spec
  arm=$1 masks=$2
  exp=iclr27-rebuttal/a2_test500_$arm
  if [ -f "outputs/$exp/summary_s${SHARD}of2.txt" ]; then stamp "skip $arm s$SHARD (done)"; continue; fi
  stamp "start $arm shard $SHARD on cards $CARDS"
  extra=()
  [ "$masks" != "-" ] && extra=(--masks "$masks")
  "$PY" "$RUNNER" --exp-id "$exp" --shard "$SHARD" --n-shards 2 ${extra[@]+"${extra[@]}"} || { stamp "$arm s$SHARD failed"; exit 1; }
  stamp "done $arm shard $SHARD: $(cat "outputs/$exp/summary_s${SHARD}of2.txt")"
done
stamp "worker cards $CARDS shard $SHARD finished"
