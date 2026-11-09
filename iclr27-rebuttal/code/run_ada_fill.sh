#!/bin/bash
# One Ada card works through the shared queue of iclr27-rebuttal/plans/2026-11-09_ada-fill-lp-fm-ce.md:
# LLM-Pruner (paper checkpoint) open loop + LingoQA first, then FM-only / CE-only open loop with all horizons.
# Tasks are claimed with mkdir, so several workers share one list; a task whose done-file exists is skipped.
#   CARD=4 bash iclr27-rebuttal/code/run_ada_fill.sh > outputs/iclr27-rebuttal/ada_fill/logs/worker_4.log 2>&1 &
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT" || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CARD:?set CARD, e.g. 4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HOME=$HOME/.cache/huggingface
export HF_HUB_CACHE=/mnt/nvme1n1/ad_vla/cache/hub
export HF_HUB_OFFLINE=1  # weights cached at the pinned revision; the full_right token is invalid since 2026-11-09
HF_TOKEN=$(bash -c 'source /home/cvlab21/project/chan/alpasim-runs/.hf_env && printf %s "$HF_TOKEN"')
export HF_TOKEN
PY=$ROOT/.venv/bin/python
OUT=iclr27-rebuttal/ada_fill
Q=outputs/$OUT/queue
LOG=outputs/$OUT/logs
mkdir -p "$Q" "$LOG"
stamp() { echo "$(date -d '+9 hours' '+%F %T') KST $*"; }

LP=/mnt/nvme1n1/ad_vla/data/alpasim/drivers/lp_r50
declare -A MODEL=([lp_r50]=$LP [fm_only]=outputs/slim_traj_u40_v2 [ce_only]=outputs/slim_coc_u40_v2)
TASKS=()
for s in 0 1 2 3; do TASKS+=("ol lp_r50 test500 $s 4"); done
for s in 0 1; do TASKS+=("ol lp_r50 oodval262 $s 2"); done
TASKS+=("lingo lp_r50")
for arm in fm_only ce_only; do
  for s in 0 1 2 3; do TASKS+=("ol $arm test500 $s 4"); done
  for s in 0 1; do TASKS+=("ol $arm oodval262 $s 2"); done
done

run_task() {  # <kind> <args...>
  local kind=$1
  shift
  case $kind in
    ol)
      local arm=$1 set_name=$2 shard=$3 n=$4 set_args
      local exp=$OUT/${arm}_${set_name}
      case $set_name in
        test500) set_args=(--set test) ;;
        oodval262) set_args=(--set ood --manifest ood_val --no-tf) ;;
      esac
      [ -f "outputs/$exp/summary_s${shard}of${n}.txt" ] && { stamp "skip $exp s$shard (done)"; return 0; }
      "$PY" experiments/evaluation/run_baseline.py "${set_args[@]}" --model "${MODEL[$arm]}" --exp-id "$exp" \
          --shard "$shard" --n-shards "$n" --gpu 0 --reserve-gb 40 \
          > "$LOG/${arm}_${set_name}_s${shard}of${n}.log" 2>&1
      ;;
    lingo)
      local arm=$1
      if [ ! -f "outputs/$OUT/lingo_vqa_$arm/predictions.csv" ]; then
        "$PY" experiments/lingoqa/run_lingo_vqa.py --arm "${MODEL[$arm]}" --exp-id "$OUT/lingo_vqa_$arm" \
            --gpu 0 --reserve-gb 40 > "$LOG/lingo_${arm}_answer.log" 2>&1 || return 1
      fi
      if [ ! -f "outputs/$OUT/lingo_vqa_scores_$arm/metrics.json" ]; then
        "$PY" experiments/lingoqa/score_lingo_vqa.py --runs "$OUT/lingo_vqa_$arm" \
            --exp-id "$OUT/lingo_vqa_scores_$arm" --gpu 0 > "$LOG/lingo_${arm}_score.log" 2>&1
      fi
      ;;
  esac
}

stamp "worker card $CARD start ($(nvidia-smi -i "$CARD" --query-gpu=name --format=csv,noheader))"
for t in "${TASKS[@]}"; do
  key=${t// /_}
  mkdir "$Q/claim_$key" 2>/dev/null || continue
  stamp "card $CARD start: $t"
  # shellcheck disable=SC2086
  if run_task $t; then
    stamp "card $CARD done: $t"
    echo "$t" >> "$Q/done.log"
  else
    stamp "card $CARD FAILED: $t"
    echo "$t" >> "$Q/failed.log"
  fi
done
stamp "worker card $CARD finished"
