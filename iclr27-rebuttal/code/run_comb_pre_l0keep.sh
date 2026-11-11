#!/bin/bash
# iclr27-rebuttal/plans/2026-11-08_comb-rules-pre-l0keep-test500.md (amended 2026-11-10): combination rules under the paper's
# pre-l0keep recipe -- rank-mean (dualsum_u40_v2), raw-mean (rawabsavg_u40_v2), raw-max (rawabsmean_u40_v2) -- on test500 and
# LingoQA. rank-max (MoOP) and dense are the paper's existing Ada rows, so nothing is re-run for them.
#   1. first card of CARDS, once empty (< 100 MiB twice, 2 min apart): build raw-mean / raw-max and a fresh rank-mean
#      (selection only, --no-state), then gates B0 / B1 (check_comb_pre_l0keep_builds.py); stop if they fail
#   2. one worker per card, each starting once its card is empty: a shared task list claimed with mkdir --
#      test500 in 2 shards per arm (run_baseline.py, the ada_fill path: sharding is bit-identical), OOD-Val262 in 2 shards
#      per arm (--no-tf; added 2026-11-12 for Table 14), then LingoQA
#      (run_lingo_vqa.py answers, score_lingo_vqa.py = Lingo-Judge, as for the paper's Table 4)
# Resumable: a build with slim_meta.json, a shard with summary_s<k>of2.txt, a LingoQA answer with predictions.csv and a score
# with metrics.json are skipped; delete outputs/iclr27-rebuttal/comb_pre_l0keep/queue to re-claim after a crash.
#   CARDS="4 5 6 7" bash iclr27-rebuttal/code/run_comb_pre_l0keep.sh > outputs/iclr27-rebuttal/comb_pre_l0keep_launch.log 2>&1
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT" || exit 1
read -r -a CARD_LIST <<< "${CARDS:-4 5 6 7}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HOME=$HOME/.cache/huggingface
export HF_HUB_CACHE=/mnt/nvme1n1/ad_vla/cache/hub
export HF_HUB_OFFLINE=1  # weights cached at the pinned revision; the full_right token is invalid since 2026-11-09
HF_TOKEN=$(bash -c 'source /home/cvlab21/project/chan/alpasim-runs/.hf_env && printf %s "$HF_TOKEN"')
export HF_TOKEN
PY=$ROOT/.venv/bin/python
OUT=iclr27-rebuttal/comb_pre_l0keep
Q=outputs/$OUT/queue
LOG=outputs/$OUT/logs
mkdir -p "$Q" "$LOG"
stamp() { echo "$(date -d '+9 hours' '+%F %T') KST $*"; }
empty() { [ "$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -lt 100 ]; }
wait_empty() {  # <card>: empty twice in a row, 2 min apart
  local n=0
  while [ "$n" -lt 2 ]; do
    if empty "$1"; then n=$((n + 1)); else n=0; fi
    [ "$n" -lt 2 ] && sleep 120
  done
}

declare -A CFG=([rank_mean]=dualsum_u40_v2 [raw_mean]=rawabsavg_u40_v2 [raw_max]=rawabsmean_u40_v2)
# rank-mean is evaluated from the September build that produced dualsum_u40_v2_test, so its rerun doubles as a reproducibility
# check; the fresh build under $OUT is only for gate B0 (today's make_slim gives the same selection)
declare -A MODEL=([rank_mean]=outputs/slim_dualsum_u40_v2 [raw_mean]=outputs/$OUT/slim_rawabsavg_u40_v2
                  [raw_max]=outputs/$OUT/slim_rawabsmean_u40_v2)
TASKS=()
for arm in raw_mean raw_max rank_mean; do
  for s in 0 1; do TASKS+=("ol $arm $s"); done
done
# OOD-Val262 (added 2026-11-12 for the paper's Table 14), rollout only like the paper's OOD rows
for arm in raw_mean raw_max rank_mean; do
  for s in 0 1; do TASKS+=("ood $arm $s"); done
done
for arm in raw_mean raw_max rank_mean; do TASKS+=("lingo $arm"); done

run_task() {  # <kind> <arm> [shard]
  local kind=$1 arm=$2
  case $kind in
    ol)
      local s=$3 exp=$OUT/${arm}_test500
      [ -f "outputs/$exp/summary_s${s}of2.txt" ] && { stamp "skip $exp s$s (done)"; return 0; }
      "$PY" experiments/evaluation/run_baseline.py --set test --model "${MODEL[$arm]}" --exp-id "$exp" \
          --shard "$s" --n-shards 2 --gpu 0 --reserve-gb 40 > "$LOG/${arm}_test500_s${s}of2.log" 2>&1
      ;;
    ood)
      local s=$3 exp=$OUT/${arm}_oodval262
      [ -f "outputs/$exp/summary_s${s}of2.txt" ] && { stamp "skip $exp s$s (done)"; return 0; }
      "$PY" experiments/evaluation/run_baseline.py --set ood --manifest ood_val --no-tf --model "${MODEL[$arm]}" \
          --exp-id "$exp" --shard "$s" --n-shards 2 --gpu 0 --reserve-gb 40 > "$LOG/${arm}_oodval262_s${s}of2.log" 2>&1
      ;;
    lingo)
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

worker() {  # <card>
  local card=$1 t key
  export CUDA_VISIBLE_DEVICES=$card
  wait_empty "$card"
  stamp "worker card $card start ($(nvidia-smi -i "$card" --query-gpu=name --format=csv,noheader))"
  for t in "${TASKS[@]}"; do
    key=${t// /_}
    mkdir "$Q/claim_$key" 2>/dev/null || continue
    stamp "card $card start: $t"
    # shellcheck disable=SC2086
    if run_task $t; then
      stamp "card $card done: $t"
      echo "$t" >> "$Q/done.log"
    else
      stamp "card $card FAILED: $t"
      echo "$t" >> "$Q/failed.log"
    fi
  done
  stamp "worker card $card finished"
}

first=${CARD_LIST[0]}
stamp "builds wait for card $first to be empty"
wait_empty "$first"
for arm in raw_mean raw_max rank_mean; do
  cfg=${CFG[$arm]}
  dst=outputs/$OUT/slim_$cfg
  [ -f "$dst/slim_meta.json" ] && continue
  stamp "build $cfg on card $first"
  CUDA_VISIBLE_DEVICES=$first "$PY" experiments/head_analysis/make_slim.py --config "$cfg" --importance importance_v2 \
      --out "$dst" --no-state --gpu 0 > "$LOG/build_$cfg.log" 2>&1 || { stamp "build $cfg failed"; exit 1; }
done
"$PY" iclr27-rebuttal/code/check_comb_pre_l0keep_builds.py > "$LOG/gates_b0_b1.log" 2>&1 \
    || { stamp "B0/B1 FAIL, stopping (see $LOG/gates_b0_b1.log)"; exit 1; }
stamp "B0/B1 PASS; starting workers on cards ${CARD_LIST[*]}"

pids=()
for card in "${CARD_LIST[@]}"; do
  worker "$card" > "$LOG/worker_$card.log" 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
stamp "all workers finished; failed tasks: $(cat "$Q/failed.log" 2>/dev/null | tr '\n' ';')"
