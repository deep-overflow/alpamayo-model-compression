#!/bin/bash
# One GPU pair runs its shard of the listed A2 arms in order (iclr27-rebuttal/plans/2026-11-09_a2-moop-test500.md).
# A new file next to run_a2_worker.sh so that a running worker's script is never edited (bash rereads it).
#   CARDS=4,5 SHARD=0 ARMS="fm_only ce_only" bash iclr27-rebuttal/code/run_a2_arms.sh > outputs/iclr27-rebuttal/a2_arms_s0.log 2>&1
# Arm names: dense, or any masks_<arm>.npz under outputs/iclr27-rebuttal/a2_moop/. Resumable like run_a2_worker.sh.
set -u
# env.sh first: it sets its own REPO (= alpamayo2-distill), so this script's root is ROOT, computed after it
source /home/cvlab21/project/chan/alpamayo2-distill/experiments/oodval/env.sh
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export HF_HUB_OFFLINE=1  # weights are cached at the pinned revision; the full_right token is invalid since 2026-11-09
export CUDA_VISIBLE_DEVICES=${CARDS:?set CARDS, e.g. 4,5}
: "${SHARD:?set SHARD (0 or 1)}" "${ARMS:?set ARMS, e.g. \"fm_only ce_only\"}"
PY=/mnt/nvme1n1/ad_vla/venvs/alpamayo2/bin/python
RUNNER=$ROOT/iclr27-rebuttal/code/run_a2_openloop.py
[ -f "$RUNNER" ] || { echo "runner not found: $RUNNER"; exit 1; }
cd "$ROOT" || exit 1
stamp() { echo "$(date -d '+9 hours' '+%F %T') KST $*"; }

for arm in $ARMS; do
  exp=iclr27-rebuttal/a2_test500_$arm
  if [ -f "outputs/$exp/summary_s${SHARD}of2.txt" ]; then stamp "skip $arm s$SHARD (done)"; continue; fi
  extra=()
  if [ "$arm" != "dense" ]; then
    masks=outputs/iclr27-rebuttal/a2_moop/masks_$arm.npz
    [ -f "$masks" ] || { stamp "no masks for $arm: $masks"; exit 1; }
    extra=(--masks "$masks")
  fi
  stamp "start $arm shard $SHARD on cards $CARDS"
  "$PY" "$RUNNER" --exp-id "$exp" --shard "$SHARD" --n-shards 2 ${extra[@]+"${extra[@]}"} || { stamp "$arm s$SHARD failed"; exit 1; }
  stamp "done $arm shard $SHARD: $(cat "outputs/$exp/summary_s${SHARD}of2.txt")"
done
stamp "arms [$ARMS] cards $CARDS shard $SHARD finished"
