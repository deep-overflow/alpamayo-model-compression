#!/bin/bash
# One Ada card works through the queue of iclr27-rebuttal/plans/2026-11-10_moop-disagreement-allocation-test500.md:
# build the two recipes (make_slim maskfile_, --no-state), then Test500 open loop in 4 shards per arm.
# Tasks are claimed with mkdir, so the four workers share one list; a task whose done-file exists is skipped, and an
# open-loop shard waits for its arm's build.
#   CARD=4 bash iclr27-rebuttal/code/run_disagree_alloc.sh > outputs/iclr27-rebuttal/disagree_alloc/logs/worker_4.log 2>&1 &
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
OUT=iclr27-rebuttal/disagree_alloc
Q=outputs/$OUT/queue
LOG=outputs/$OUT/logs
mkdir -p "$Q" "$LOG"
stamp() { echo "$(date -d '+9 hours' '+%F %T') KST $*"; }

ARMS=(dis050 dis100)
TASKS=()
for arm in "${ARMS[@]}"; do TASKS+=("build $arm"); done
for arm in "${ARMS[@]}"; do
  for s in 0 1 2 3; do TASKS+=("ol $arm $s 4"); done
done

run_task() {  # <kind> <args...>
  local kind=$1
  shift
  case $kind in
    build)
      local arm=$1 slim=outputs/$OUT/slim_$1
      [ -f "$slim/slim_meta.json" ] && { stamp "skip build $arm (done)"; return 0; }
      "$PY" experiments/head_analysis/make_slim.py --config "maskfile_$OUT/$arm" --importance importance_v2 \
          --no-state --out "$slim" --gpu 0 --reserve-gb 30 > "$LOG/build_${arm}.log" 2>&1
      ;;
    ol)
      local arm=$1 shard=$2 n=$3 slim=outputs/$OUT/slim_$1
      local exp=$OUT/${arm}_test500
      [ -f "outputs/$exp/summary_s${shard}of${n}.txt" ] && { stamp "skip $exp s$shard (done)"; return 0; }
      for _ in $(seq 1 60); do      # wait up to 60 min for this arm's build
        [ -f "$slim/slim_meta.json" ] && break
        grep -qx "build $arm" "$Q/failed.log" 2>/dev/null && { stamp "build $arm failed, skipping $exp s$shard"; return 1; }
        sleep 60
      done
      [ -f "$slim/slim_meta.json" ] || { stamp "no build for $arm after 60 min"; return 1; }
      "$PY" experiments/evaluation/run_baseline.py --set test --model "$slim" --exp-id "$exp" \
          --shard "$shard" --n-shards "$n" --gpu 0 --reserve-gb 40 > "$LOG/${arm}_test500_s${shard}of${n}.log" 2>&1
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
