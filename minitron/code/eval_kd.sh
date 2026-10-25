#!/bin/bash
# Evaluate a train_kd run once it finishes (plan section 5, D10-2), all on Ada 4-7:
#   1. probe val_official_238 for epoch 0 (zero-shot MT-W) and every epoch checkpoint
#   2. select_ckpt.py: lowest probe mean minADE@6 among the epoch checkpoints
#   3. one queue per card, card k running shard k/4 of: the selected checkpoint on test_2000
#      (test500 + 1,500), val500 and OOD-Val262, then the reference arms (dense, MoOP, zero-shot
#      MT-W) on the 1,500 added test clips -- skipped where their rows already exist. Protocol of
#      the stored rows: k 8, seed 42 (per-clip), max_gen 256. Card 7 first runs LingoQA
#      (standard protocol) for the selected checkpoint and the zero-shot MT-W.
#   4. LingoQA judge, paired readouts -> outputs/minitron/<exp>_eval/
#
#   nohup bash minitron/code/eval_kd.sh [exp] > outputs/minitron/<exp>_eval.log 2>&1 &
set -u
EXP=${1:-mtw24_kd}
ZS=${ZS:-outputs/minitron/slim_mtw24}
ZS_ARM=${ZS_ARM:-mtw24_zs}
MOOP=${MOOP:-outputs/slim_dual_u40_v2}
EXT=minitron/test2000x_
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

ol() {  # <card k> <model> <exp-id> <rows tag> <log name> <run_baseline args...>
  local k=$1 model=$2 exp=$3 tag=$4 name=$5
  shift 5
  if [ -f "outputs/$exp/${tag}_s${k}of4.json" ]; then
    stamp "skip $exp shard $k (rows exist)"
    return
  fi
  bash "$RETRY" 3 experiments/evaluation/run_baseline.py --model "$model" --exp-id "$exp" \
    --shard "$k" --n-shards 4 --gpu $((4 + k)) "$@" > "$LOG/${name}_s${k}.log" 2>&1
}
queue() {  # <card k>
  local k=$1 sel_tag
  sel_tag=$(basename "$SEL")
  if [ "$k" = 3 ]; then
    bash "$RETRY" 3 experiments/lingoqa/run_lingo_vqa.py --arm "$SEL" \
      --exp-id "minitron/lingo_vqa_$EXP" --gpu 7 > "$LOG/lingo_$EXP.log" 2>&1
    [ -f "outputs/minitron/lingo_vqa_$ZS_ARM/rows.json" ] \
      || bash "$RETRY" 3 experiments/lingoqa/run_lingo_vqa.py --arm "$ZS" \
           --exp-id "minitron/lingo_vqa_$ZS_ARM" --gpu 7 > "$LOG/lingo_$ZS_ARM.log" 2>&1
  fi
  ol "$k" "$SEL" "minitron/${EXP}_test" "$sel_tag" test --set test --n-indist 2000
  ol "$k" "$SEL" "minitron/${EXP}_indist" "$sel_tag" indist --set indist
  ol "$k" "$SEL" "minitron/${EXP}_oodval" "$sel_tag" oodval --set ood --manifest ood_val
  ol "$k" baseline "${EXT}dense" baseline ext_dense --set test --manifest test_2000_ext1500
  ol "$k" "$MOOP" "${EXT}moop" "$(basename "$MOOP")" ext_moop --set test --manifest test_2000_ext1500
  ol "$k" "$ZS" "${EXT}$ZS_ARM" "$(basename "$ZS")" ext_zs --set test --manifest test_2000_ext1500
}
for k in 0 1 2 3; do queue "$k" & done
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
