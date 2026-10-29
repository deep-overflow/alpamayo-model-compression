#!/bin/bash
# Bring the A100 track's rows (outputs/a100/) and its job logs back to this box, through the
# OTP control socket that push_neuron.sh uses. Read-only on NEURON.
#
#   bash experiments/transfer/a100_pull.sh <NEURON_USER>
#   .venv/bin/python experiments/evaluation/analyze_a100.py --stage gate
set -euo pipefail
USER_N=${1:?usage: a100_pull.sh <NEURON_USER>}
DM=${NEURON_DM-neuron-dm.ksc.re.kr}
CM=${NEURON_CM-$HOME/.ssh/cm-neuron}
SCRATCH=/scratch/$USER_N
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=$(readlink -f "$REPO/outputs")

# BatchMode: a dead socket fails at once instead of stopping at an OTP prompt
RSYNC=(rsync -a --info=progress2 --human-readable
       -e "ssh -o ControlPath=$CM -o BatchMode=yes")
mkdir -p "$OUT/a100/logs"
"${RSYNC[@]}" "$USER_N@$DM:$SCRATCH/outputs/a100/" "$OUT/a100/"
"${RSYNC[@]}" --include='a100_*' --include='eval_a100-*' --exclude='*' \
    "$USER_N@$DM:$SCRATCH/project/alpamayo-model-compression/logs/" "$OUT/a100/logs/"
echo "pulled into $OUT/a100"
