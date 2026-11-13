#!/bin/bash
# One Blackwell card (GPUs 0-3, user 11-12) works through the shared queue of
# iclr27-rebuttal/plans/2026-11-12_u50-openloop-blackwell.md: per arm, open loop on Test500 (4 shards) and OOD-Val262
# (2 shards), the dense-CoC NLL on Test500, and LingoQA (answer + Lingo-Judge). Same commands as run_ada_fill.sh;
# tasks are claimed with mkdir, a task whose done-file exists is skipped.
#   CARD=0 bash iclr27-rebuttal/code/run_u50_openloop.sh > outputs/iclr27-rebuttal/u50_openloop_bw/logs/worker_0.log 2>&1 &
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT" || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CARD:?set CARD, e.g. 0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HOME=$HOME/.cache/huggingface
export HF_HUB_CACHE=/mnt/nvme1n1/ad_vla/cache/hub
export HF_HUB_OFFLINE=1  # weights cached at the pinned revision; the full_right token is invalid since 2026-11-09
HF_TOKEN=$(bash -c 'source /home/cvlab21/project/chan/alpasim-runs/.hf_env && printf %s "$HF_TOKEN"')
export HF_TOKEN
PY=$ROOT/.venv/bin/python
OUT=iclr27-rebuttal/u50_openloop_bw
Q=outputs/$OUT/queue
LOG=outputs/$OUT/logs
mkdir -p "$Q" "$LOG"
stamp() { echo "$(date -d '+9 hours' '+%F %T') KST $*"; }

declare -A MODEL=([moop]=outputs/slim_dual_u50_v2 [tyr]=outputs/slim_tyr_u50_r [lp]=outputs/slim_lp_r60
                  [fm]=outputs/slim_traj_u50_v2 [ce]=outputs/slim_coc_u50_v2 [dense]=baseline
                  [wanda]=outputs/slim_wanda_u50_v2 [moope]=outputs/slim_dualexp_u50_em93p75)
TASKS=()
for arm in moop dense tyr lp fm ce; do
  for s in 0 1 2 3; do TASKS+=("ol $arm test500 $s 4"); done
  for s in 0 1; do TASKS+=("ol $arm oodval262 $s 2"); done
  TASKS+=("nll $arm" "lingo $arm")
done
# 11-13 (user: "Wanda and everything else"): Wanda u50 and MoOP + expert u50 on the same three evaluations, then
# OOD-train open loop (4 shards) + ground-truth-CoC NLL for every arm. The failed Tyr Test500 shard 3 re-runs with
# run_baseline's generation-cache fallback (its claim dir is removed before relaunch).
for arm in wanda moope; do
  for s in 0 1 2 3; do TASKS+=("ol $arm test500 $s 4"); done
  for s in 0 1; do TASKS+=("ol $arm oodval262 $s 2"); done
  TASKS+=("nll $arm" "lingo $arm")
done
for arm in moop moope tyr lp fm ce wanda dense; do
  for s in 0 1 2 3; do TASKS+=("ol $arm oodtrain $s 4"); done
  TASKS+=("nllgt $arm")
done

tag_of() { [ "${MODEL[$1]}" = baseline ] && echo baseline || basename "${MODEL[$1]}"; }

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
        oodtrain) set_args=(--set ood --manifest ood_train --no-tf) ;;
      esac
      [ -f "outputs/$exp/summary_s${shard}of${n}.txt" ] && { stamp "skip $exp s$shard (done)"; return 0; }
      "$PY" experiments/evaluation/run_baseline.py "${set_args[@]}" --model "${MODEL[$arm]}" --exp-id "$exp" \
          --shard "$shard" --n-shards "$n" --gpu 0 --reserve-gb 40 \
          > "$LOG/${arm}_${set_name}_s${shard}of${n}.log" 2>&1
      ;;
    nll)
      local arm=$1
      local exp=$OUT/nll_dense_$(tag_of "$arm")_test   # dir name = nll_dense_<tag>_test, as ci_openloop_lingoqa.nll_block reads it
      [ -f "outputs/$exp/summary_s0of1.txt" ] && { stamp "skip $exp (done)"; return 0; }
      "$PY" experiments/evaluation/run_nll_dense.py --set test --model "${MODEL[$arm]}" --exp-id "$exp" \
          --gpu 0 --reserve-gb 40 > "$LOG/nll_${arm}.log" 2>&1
      ;;
    nllgt)
      local arm=$1
      local exp=$OUT/nllgt_${arm}_oodtrain
      [ -f "outputs/$exp/summary_s0of1.txt" ] && { stamp "skip $exp (done)"; return 0; }
      "$PY" cvpr27/code/run_nll_gtcoc.py --manifest ood_train --model "${MODEL[$arm]}" --exp-id "$exp" \
          --gpu 0 --reserve-gb 40 > "$LOG/nllgt_${arm}.log" 2>&1
      ;;
    lingo)
      local arm=$1
      local exp=$OUT/lingo_vqa_$arm
      if [ ! -f "outputs/$exp/predictions.csv" ]; then
        "$PY" experiments/lingoqa/run_lingo_vqa.py --arm "${MODEL[$arm]}" --exp-id "$exp" \
            --gpu 0 --reserve-gb 40 > "$LOG/lingo_${arm}_answer.log" 2>&1 || return 1
      fi
      if [ ! -f "outputs/$exp/scored.json" ]; then
        "$PY" experiments/lingoqa/score_lingo_vqa.py --runs "$exp" \
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
