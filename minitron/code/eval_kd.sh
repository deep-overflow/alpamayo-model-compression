#!/bin/bash
# Evaluate a train_kd run once it finishes (plan section 5, D10-2), all on Ada 4-7:
#   1. val500 (the selection set, not reported) for every checkpoint in $CKPTS (default the three
#      epoch checkpoints; mtw24_kd2 saves ckpt_q25 .. ckpt_q100); the zero-shot MT-W rows already
#      stored are the reference
#   2. select_ckpt.py: lowest val500 mean minADE@6 among those checkpoints
#   3. one queue per card, card k running shard k/4 of: the selected checkpoint on test_2000
#      and OOD-Val262 (the reported sets), then the reference arms (dense, MoOP, zero-shot
#      MT-W) on the 1,500 test clips added to test500 -- skipped where their rows already
#      exist. Protocol of the stored rows: k 8, seed 42 (per-clip), max_gen 256. Card 7 first
#      runs LingoQA (standard protocol) for the selected checkpoint and the zero-shot MT-W.
#   4. LingoQA judge, paired readouts -> outputs/minitron/<exp>_eval/
#
#   nohup bash minitron/code/eval_kd.sh [exp] > outputs/minitron/<exp>_eval.log 2>&1 &
#   CKPTS="ckpt_q25 ckpt_q50 ckpt_q75 ckpt_q100" nohup bash minitron/code/eval_kd.sh mtw24_kd2 ...
# A run trained on NEURON is pulled back (checkpoints + summary.txt) before this starts.
set -u
EXP=${1:-mtw24_kd}
CKPTS=${CKPTS:-ckpt_epoch1 ckpt_epoch2 ckpt_epoch3}
TRIES=${TRIES:-3}  # launcher attempts per run, 60 s apart: raise it when the Ada cards are shared
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

ol() {  # <card k> <model> <exp-id> <rows tag> <log name> <run_baseline args...>
  local k=$1 model=$2 exp=$3 tag=$4 name=$5
  shift 5
  if [ -f "outputs/$exp/${tag}_s${k}of4.json" ]; then
    stamp "skip $exp shard $k (rows exist)"
    return
  fi
  bash "$RETRY" "$TRIES" experiments/evaluation/run_baseline.py --model "$model" --exp-id "$exp" \
    --shard "$k" --n-shards 4 --gpu $((4 + k)) "$@" > "$LOG/${name}_s${k}.log" 2>&1
}
select_queue() {  # <card k>
  local c s
  for c in $CKPTS; do
    s=${c#ckpt_}
    s=${s/epoch/e}  # ckpt_epoch1 -> e1 (the stored mtw24_kd runs), ckpt_q25 -> q25
    ol "$1" "outputs/minitron/$EXP/$c" "minitron/${EXP}_val_$s" "$c" "val_$s" --set indist
  done
}
for k in 0 1 2 3; do select_queue "$k" & done
wait
stamp "val500 for every checkpoint done"
# shellcheck disable=SC2086
$PY minitron/code/select_ckpt.py --exp "$EXP" --zs-arm "$ZS_ARM" --ckpts $CKPTS > "$LOG/select.txt" 2>&1 \
  || { cat "$LOG/select.txt"; exit 1; }
SEL=$(cat "outputs/minitron/${EXP}_select/selected.txt")
stamp "selected $SEL"

queue() {  # <card k>
  local k=$1 sel_tag
  sel_tag=$(basename "$SEL")
  if [ "$k" = 3 ]; then
    bash "$RETRY" "$TRIES" experiments/lingoqa/run_lingo_vqa.py --arm "$SEL" \
      --exp-id "minitron/lingo_vqa_$EXP" --gpu 7 > "$LOG/lingo_$EXP.log" 2>&1
    [ -f "outputs/minitron/lingo_vqa_$ZS_ARM/rows.json" ] \
      || bash "$RETRY" "$TRIES" experiments/lingoqa/run_lingo_vqa.py --arm "$ZS" \
           --exp-id "minitron/lingo_vqa_$ZS_ARM" --gpu 7 > "$LOG/lingo_$ZS_ARM.log" 2>&1
  fi
  ol "$k" "$SEL" "minitron/${EXP}_test" "$sel_tag" test --set test --n-indist 2000
  ol "$k" "$SEL" "minitron/${EXP}_oodval" "$sel_tag" oodval --set ood --manifest ood_val
  ol "$k" baseline "${EXT}dense" baseline ext_dense --set test --manifest test_2000_ext1500
  ol "$k" "$MOOP" "${EXT}moop" "$(basename "$MOOP")" ext_moop --set test --manifest test_2000_ext1500
  ol "$k" "$ZS" "${EXT}$ZS_ARM" "$(basename "$ZS")" ext_zs --set test --manifest test_2000_ext1500
}
for k in 0 1 2 3; do queue "$k" & done
wait
stamp "open loop + LingoQA answers done"

bash "$RETRY" "$TRIES" experiments/lingoqa/score_lingo_vqa.py \
  --runs "minitron/lingo_vqa_$EXP" "minitron/lingo_vqa_$ZS_ARM" \
  --exp-id "minitron/lingo_vqa_scores_$EXP" --gpu 4,5,6,7 > "$LOG/lingo_score.log" 2>&1
$PY experiments/lingoqa/analyze_lingo.py \
  --runs "minitron/lingo_vqa_$EXP" "minitron/lingo_vqa_$ZS_ARM" lingo_vqa_slim_dual_u40_v2 lingo_vqa_baseline \
  --baseline lingo_vqa_slim_dual_u40_v2 --exp-id "minitron/lingo_$EXP" > "$LOG/lingo_analysis.txt" 2>&1
$PY minitron/code/compare_openloop.py --arm "minitron/$EXP" --ref "minitron/$ZS_ARM" \
  --sets test2000 oodval > "$LOG/openloop.txt" 2>&1
stamp "all done"
touch "$LOG/DONE"
