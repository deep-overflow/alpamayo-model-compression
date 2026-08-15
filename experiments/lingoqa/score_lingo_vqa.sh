#!/bin/bash
# Score a predictions.csv with LingoQA's OWN evaluate.py, not a reimplementation.
#
# Two things make this non-obvious:
#   1. evaluate.py hardcodes .to("cuda:0"). Card 0 on this box is permanently full, so
#      the card is selected with CUDA_VISIBLE_DEVICES and stays cuda:0 inside the script.
#   2. evaluate.py runs from its own directory: it does `from constants import ...`,
#      a flat import that only resolves when benchmark/ is the script's directory.
#
# It prints "Matched N predictions with references" -- N must be 500. Anything less
# means rows were dropped on the merge and the score is over a subset.
#
# Usage: bash score_lingo_vqa.sh <exp_id> [gpu]
set -u
EXP=${1:?usage: score_lingo_vqa.sh <exp_id> [gpu]}
GPU=${2:-5}
REPO=/home/cvlab21/project/chan/alpamayo-model-compression
LINGOQA=/home/cvlab21/project/chan/LingoQA
PRED="$REPO/outputs/$EXP/predictions.csv"

[ -f "$PRED" ] || { echo "no predictions.csv for $EXP"; exit 1; }

export CUDA_VISIBLE_DEVICES=$GPU
export HF_HOME=${HF_HOME-$HOME/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE-/mnt/nvme1n1/ad_vla/cache/hub}

# evaluate.py needs `click`, which this repo's venv does not carry. It is injected on
# PYTHONPATH from a side directory rather than installed, so the venv other experiments
# share is left untouched. Create it once with:
#   uv pip install --target "$PYLIBS" click
PYLIBS=${PYLIBS-/tmp/claude-1001/-home-cvlab21-project-chan-LingoQA/f6ae5ee7-52cb-444e-ad4f-d885bc149b75/scratchpad/pylibs}
[ -d "$PYLIBS/click" ] || { echo "missing click at $PYLIBS -- see header"; exit 1; }
export PYTHONPATH="$PYLIBS${PYTHONPATH:+:$PYTHONPATH}"

echo "=== $EXP ($(wc -l < "$PRED") csv lines incl. header) ==="
cd "$LINGOQA/benchmark" && "$REPO/.venv/bin/python" evaluate.py --predictions_path "$PRED"
