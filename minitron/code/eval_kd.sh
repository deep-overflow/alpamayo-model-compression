#!/bin/bash
# Evaluate a train_kd run once it finishes (plan section 5, D10-2), all on Ada 4-7:
#   1. probe val_official_238 for epoch 0 (zero-shot MT-W) and every epoch checkpoint
#   2. select_ckpt.py: lowest probe mean minADE@6 among the epoch checkpoints
#   3. the selected checkpoint on test500 / val500 / OOD-Val262 -- the protocol of the stored
#      dense / MoOP / MT-W zero-shot rows (k 8, seed 42, max_gen 256) -- and LingoQA
#      (standard protocol) for it and for the zero-shot MT-W
#   4. LingoQA judge, paired readouts -> outputs/minitron/<exp>_eval/
#
#   nohup bash minitron/code/eval_kd.sh [exp] > outputs/minitron/<exp>_eval.log 2>&1 &
set -u
EXP=${1:-mtw24_kd}
ZS=${ZS:-outputs/minitron/slim_mtw24}
ZS_ARM=${ZS_ARM:-mtw24_zs}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export ALPAMAYO_REPO=$REPO
cd "$REPO" || exit 1
RETRY=experiments/head_analysis/run_retry_host.sh
PY=.venv/bin/python
LOG=outputs/minitron/${EXP}_eval
mkdir -p "$LOG"
stamp() { echo "$(date -u '+%F %T') UTC $*"; }

stamp "waiting for outputs/minitron/$EXP/summary.txt"
until [ -f "outputs/minitron/$EXP/summary.txt" ]; do
  if ! pgrep -f "minitron/code/train_kd.py" > /dev/null; then
    sleep 60  # the summary is written just before the ranks exit
    [ -f "outputs/minitron/$EXP/summary.txt" ] && break
    stamp "train_kd is not running and wrote no summary -- stopping"
    exit 1
  fi
  sleep 120
done
while pgrep -f "minitron/code/train_kd.py" > /dev/null; do sleep 30; done
stamp "training finished"

probe() {  # <model dir> <tag> <gpu>
  bash "$RETRY" 3 experiments/evaluation/run_baseline.py --set indist --sets-id recovery_sets \
    --manifest val_official_238 --model "$1" --exp-id "minitron/${EXP}_probe_$2" --gpu "$3" \
    > "$LOG/probe_$2.log" 2>&1
}
probe "$ZS" e0 7 &
for e in 1 2 3; do probe "outputs/minitron/$EXP/ckpt_epoch$e" "e$e" $((3 + e)) & done
wait
stamp "probes done"
$PY minitron/code/select_ckpt.py --exp "$EXP" > "$LOG/select.txt" 2>&1 || { cat "$LOG/select.txt"; exit 1; }
SEL=$(cat "outputs/minitron/${EXP}_probe/selected.txt")
stamp "selected $SEL"

ol() {  # <set> <suffix> <gpu> [extra args]
  bash "$RETRY" 3 experiments/evaluation/run_baseline.py --set "$1" --model "$SEL" \
    --exp-id "minitron/${EXP}_$2" --gpu "$3" "${@:4}" > "$LOG/$2.log" 2>&1
}
ol test test 4 &
ol indist indist 5 &
ol ood oodval 6 --manifest ood_val &
(bash "$RETRY" 3 experiments/lingoqa/run_lingo_vqa.py --arm "$SEL" --exp-id "minitron/lingo_vqa_$EXP" \
   --gpu 7 > "$LOG/lingo_$EXP.log" 2>&1
 bash "$RETRY" 3 experiments/lingoqa/run_lingo_vqa.py --arm "$ZS" --exp-id "minitron/lingo_vqa_$ZS_ARM" \
   --gpu 7 > "$LOG/lingo_$ZS_ARM.log" 2>&1) &
wait
stamp "open loop + LingoQA answers done"

bash "$RETRY" 3 experiments/lingoqa/score_lingo_vqa.py \
  --runs "minitron/lingo_vqa_$EXP" "minitron/lingo_vqa_$ZS_ARM" \
  --exp-id "minitron/lingo_vqa_scores_$EXP" --gpu 4,5,6,7 > "$LOG/lingo_score.log" 2>&1
$PY experiments/lingoqa/analyze_lingo.py \
  --runs "minitron/lingo_vqa_$EXP" "minitron/lingo_vqa_$ZS_ARM" lingo_vqa_slim_dual_u40_v2 lingo_vqa_baseline \
  --baseline lingo_vqa_slim_dual_u40_v2 --exp-id "minitron/lingo_$EXP" > "$LOG/lingo_analysis.txt" 2>&1
$PY minitron/code/compare_openloop.py --arm "minitron/$EXP" --ref "minitron/$ZS_ARM" \
  > "$LOG/openloop.txt" 2>&1
stamp "all done"
touch "$LOG/DONE"
