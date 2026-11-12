#!/bin/bash
# OOD-train (1,271 clips, manifest ood_train) for the four combination rules under the paper's pre-l0keep recipe, on Ada,
# added 2026-11-13 to the Table 14 work (iclr27-rebuttal/plans/2026-11-08_comb-rules-pre-l0keep-test500.md).
# max-rank (MoOP, dual_u40_v2) runs too: its only OOD-train rows are A100 / GH200, and a table keeps one GPU type.
# One worker per card; 4 shards per arm claimed with mkdir; rollout only (--no-tf), k = 8, seed 42, max_gen 256 (defaults).
# A shard whose summary file exists is skipped. Each card is used only once it reads <= 10 MiB twice, 2 min apart.
#   CARD=4 setsid nohup bash iclr27-rebuttal/code/run_comb_oodtrain.sh > outputs/iclr27-rebuttal/comb_pre_l0keep/logs/oodtrain_worker_4.log 2>&1 < /dev/null &
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
OUT=iclr27-rebuttal/comb_pre_l0keep
Q=outputs/$OUT/queue_oodtrain
LOG=outputs/$OUT/logs
mkdir -p "$Q" "$LOG"
stamp() { echo "$(date -d '+9 hours' '+%F %T') KST $*"; }
idle() { [ "$(nvidia-smi -i "$CARD" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -le 10 ]; }

declare -A MODEL=([max_rank]=outputs/slim_dual_u40_v2 [rank_mean]=outputs/slim_dualsum_u40_v2
                  [raw_max]=outputs/$OUT/slim_rawabsmean_u40_v2 [raw_mean]=outputs/$OUT/slim_rawabsavg_u40_v2)
TASKS=()
for s in 0 1 2 3; do
  for arm in max_rank rank_mean raw_max raw_mean; do TASKS+=("$arm $s"); done
done

n=0
while [ "$n" -lt 2 ]; do
  if idle; then n=$((n + 1)); else n=0; fi
  [ "$n" -lt 2 ] && sleep 120
done
stamp "worker card $CARD start ($(nvidia-smi -i "$CARD" --query-gpu=name --format=csv,noheader))"
for t in "${TASKS[@]}"; do
  key=${t// /_}
  mkdir "$Q/claim_$key" 2>/dev/null || continue
  set -- $t
  arm=$1 s=$2 exp=$OUT/${1}_oodtrain1271
  if [ -f "outputs/$exp/summary_s${s}of4.txt" ]; then stamp "skip $exp s$s (done)"; echo "$t" >> "$Q/done.log"; continue; fi
  stamp "card $CARD start: $arm shard $s"
  if "$PY" experiments/evaluation/run_baseline.py --set ood --manifest ood_train --no-tf --model "${MODEL[$arm]}" \
      --exp-id "$exp" --shard "$s" --n-shards 4 --gpu 0 --reserve-gb 40 > "$LOG/${arm}_oodtrain1271_s${s}of4.log" 2>&1; then
    stamp "card $CARD done: $arm shard $s"
    echo "$t" >> "$Q/done.log"
  else
    stamp "card $CARD FAILED: $arm shard $s"
    echo "$t" >> "$Q/failed.log"
  fi
done
stamp "worker card $CARD finished"
