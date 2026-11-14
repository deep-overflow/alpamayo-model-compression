#!/bin/bash
# A2 LingoQA queue (iclr27-rebuttal/plans/2026-11-13_a2-lingoqa.md): each GPU pair in PAIRS, once both cards are empty (< 100 MiB
# twice, 2 min apart), claims arms from ARMS in order (mkdir claims) and answers the 500 LingoQA questions with run_a2_lingoqa.py;
# the first pair first runs the gate (3 questions: no hooks / all-ones / MoOP masks). When every arm has predictions.csv, one card
# scores them all with experiments/lingoqa/score_lingo_vqa.py (Lingo-Judge) into each run's scored.json.
#   PAIRS="4,5 6,7" ARMS="dense moop fm_only ce_only" bash iclr27-rebuttal/code/run_a2_lingoqa_queue.sh > outputs/iclr27-rebuttal/a2_lingo_queue.log 2>&1
# Arm names: dense, or any masks_<arm>.npz under outputs/iclr27-rebuttal/a2_moop/; runs land in outputs/iclr27-rebuttal/a2_lingo_<arm>.
# SPLIT=train (2026-11-14): the confirmatory LingoQA-train sample (run_a2_lingoqa.py --split train, its default 400 x 5 draw);
# runs land in a2_lingotrain_<arm>, the queue in a2_lingotrain_queue, and scoring uses score_a2_lingo_train.py.
set -u
source /home/cvlab21/project/chan/alpamayo2-distill/experiments/oodval/env.sh
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT" || exit 1
export HF_HUB_OFFLINE=1
PY2=/mnt/nvme1n1/ad_vla/venvs/alpamayo2/bin/python
PY15=$ROOT/.venv/bin/python
read -r -a PL <<< "${PAIRS:-4,5 6,7}"
read -r -a AL <<< "${ARMS:-dense moop fm_only ce_only}"
SPLIT=${SPLIT:-val}
if [ "$SPLIT" = train ]; then P=a2_lingotrain; SPLIT_ARGS=(--split train); else P=a2_lingo; SPLIT_ARGS=(); fi
Q=outputs/iclr27-rebuttal/${P}_queue
mkdir -p "$Q"
stamp() { echo "$(date -d '+9 hours' '+%F %T') KST $*"; }
empty() { [ "$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -lt 100 ]; }
wait_pair() {  # <a,b>: both cards empty twice in a row, 2 min apart
  local n=0
  while [ "$n" -lt 2 ]; do
    if empty "${1%,*}" && empty "${1#*,}"; then n=$((n + 1)); else n=0; fi
    [ "$n" -lt 2 ] && sleep 120
  done
}
run_arm() {  # <cards> <arm>
  local extra=()
  [ "$2" != dense ] && extra=(--masks "outputs/iclr27-rebuttal/a2_moop/masks_$2.npz")
  [ -f "outputs/iclr27-rebuttal/${P}_$2/predictions.csv" ] && return 0
  CUDA_VISIBLE_DEVICES=$1 "$PY2" iclr27-rebuttal/code/run_a2_lingoqa.py --exp-id "iclr27-rebuttal/${P}_$2" \
      ${SPLIT_ARGS[@]+"${SPLIT_ARGS[@]}"} ${extra[@]+"${extra[@]}"} > "$Q/answer_$2.log" 2>&1
}
worker() {  # <cards> <is_first>
  local cards=$1 arm
  wait_pair "$cards"
  stamp "pair $cards empty"
  if [ "$2" = 1 ] && [ ! -e "$Q/GATE_PASS" ]; then
    if CUDA_VISIBLE_DEVICES=$cards "$PY2" iclr27-rebuttal/code/run_a2_lingoqa.py --exp-id iclr27-rebuttal/a2_lingo_gate --gate 3 \
        --masks outputs/iclr27-rebuttal/a2_moop/masks_moop.npz > "$Q/gate.log" 2>&1; then
      touch "$Q/GATE_PASS"; stamp "gate PASS: $(grep -m1 'gate ones==none' "$Q/gate.log")"
    else
      touch "$Q/GATE_FAIL"; stamp "gate FAIL (see $Q/gate.log)"; return 1
    fi
  fi
  while [ ! -e "$Q/GATE_PASS" ] && [ ! -e "$Q/GATE_FAIL" ]; do sleep 30; done
  [ -e "$Q/GATE_FAIL" ] && return 1
  for arm in "${AL[@]}"; do
    mkdir "$Q/claim_$arm" 2>/dev/null || continue
    stamp "pair $cards start $arm"
    if run_arm "$cards" "$arm"; then stamp "pair $cards done $arm: $(tail -1 "$Q/answer_$arm.log" | cut -c1-120)"
    else stamp "pair $cards FAILED $arm"; fi
  done
}
pids=()
first=1
for p in "${PL[@]}"; do worker "$p" "$first" > "$Q/worker_${p/,/_}.log" 2>&1 & pids+=($!); first=0; done
wait "${pids[@]}"
runs=()
for arm in "${AL[@]}"; do
  [ -f "outputs/iclr27-rebuttal/${P}_$arm/predictions.csv" ] && runs+=("iclr27-rebuttal/${P}_$arm")
done
card=${PL[0]%,*}
wait_pair "$card,$card"
stamp "scoring ${#runs[@]} runs on card $card"
if [ "$SPLIT" = train ]; then
  CUDA_VISIBLE_DEVICES=$card "$PY15" iclr27-rebuttal/code/score_a2_lingo_train.py --runs "${runs[@]}" \
      > "$Q/score.log" 2>&1 && stamp "scored" || stamp "scoring FAILED"
else
  CUDA_VISIBLE_DEVICES=$card "$PY15" experiments/lingoqa/score_lingo_vqa.py --runs "${runs[@]}" \
      --exp-id iclr27-rebuttal/a2_lingo_scores --gpu 0 > "$Q/score.log" 2>&1 && stamp "scored" || stamp "scoring FAILED"
fi
stamp "all done"
