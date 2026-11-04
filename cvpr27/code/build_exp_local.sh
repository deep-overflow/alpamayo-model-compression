#!/bin/bash
# MoOP l0keep + expert MLP-only recipes (cvpr27/plans/2026-11-05_moop-l0keep-expert-mlp.md), one local card.
#   CARD=4 bash cvpr27/code/build_exp_local.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME=$HOME/.cache/huggingface HF_HUB_CACHE=/mnt/nvme1n1/ad_vla/cache/hub HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
C=${CARD:?set CARD}
L=outputs/cvpr27/exp_mlp/logs
mkdir -p "$L"
stamp() { echo "$(date -d '+9 hours' '+%F %T') KST $*"; }
for m in 75 87p5 93p75; do
  cfg=dualexp_u40_l0keep_em$m
  out=outputs/cvpr27/slim_$cfg
  if [ ! -f "$out/summary.txt" ]; then
    stamp "$cfg card $C"
    "$PY" experiments/head_analysis/make_slim.py --config "$cfg" --importance importance_v2 \
      --out "$out" --no-state --gpu "$C" > "$L/$cfg.log" 2>&1 || stamp "$cfg exited non-zero"
  fi
  stamp "$cfg: $(grep -E 'l0keep:|surgery|smoke:' "$L/$cfg.log" | tr '\n' ' ' | cut -c1-260)"
done
stamp "card $C done"
