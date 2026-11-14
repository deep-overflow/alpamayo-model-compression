#!/bin/bash
# Edge u40 worker (plans/2026-11-14_a2-edge-moop-u40.md): LingoQA, then Test500, then OOD-Val250 for one arm on one card.
# The open-loop runs wait for the Edge cache's GT gate (verify.json "passed": true). All runs are resumable.
# Usage: bash iclr27-rebuttal/code/edge_u40_worker.sh <dense|moop|fm_only|ce_only> <card>
set -u
ARM=$1
CARD=$2
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO" || exit 1
PY=/mnt/nvme1n1/ad_vla/venvs/alpamayo2-edge/bin/python
D=/mnt/nvme1n1/ad_vla/data/physicalai_av/pre_processed
LOG=outputs/iclr27-rebuttal/edge_moop/logs
mkdir -p "$LOG"
MASK=()
if [ "$ARM" != dense ]; then MASK=(--masks "outputs/iclr27-rebuttal/edge_moop/masks_${ARM}.npz"); fi
export CUDA_VISIBLE_DEVICES=$CARD CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "$(date -u +%FT%T) start $ARM on card $CARD" >> "$LOG/worker_$ARM.log"
"$PY" iclr27-rebuttal/code/run_edge_lingoqa.py --exp-id "iclr27-rebuttal/edge_lingo_$ARM" "${MASK[@]}" \
    > "$LOG/lingo_$ARM.log" 2>&1 || { echo "lingo failed" >> "$LOG/worker_$ARM.log"; exit 1; }
for S in test500 oodval; do
    V="$D/${S}_edge/verify.json"
    until [ -f "$V" ]; do sleep 60; done
    grep -q '"passed": true' "$V" || { echo "cache gate failed for $S" >> "$LOG/worker_$ARM.log"; exit 1; }
    "$PY" iclr27-rebuttal/code/run_edge_openloop.py --set "$S" --exp-id "iclr27-rebuttal/edge_${S}_$ARM" "${MASK[@]}" \
        > "$LOG/${S}_$ARM.log" 2>&1 || { echo "$S failed" >> "$LOG/worker_$ARM.log"; exit 1; }
done
echo "$(date -u +%FT%T) done $ARM" >> "$LOG/worker_$ARM.log"
