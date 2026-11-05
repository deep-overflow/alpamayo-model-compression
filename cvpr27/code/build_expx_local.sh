#!/bin/bash
# LLM-Pruner / Tyr + expert MLP-only recipes (cvpr27/plans/2026-11-05_baselines-expert-mlp.md), one local card.
#   CARD=4 bash cvpr27/code/build_expx_local.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME=$HOME/.cache/huggingface HF_HUB_CACHE=/mnt/nvme1n1/ad_vla/cache/hub HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
C=${CARD:?set CARD}
L=outputs/cvpr27/expx/logs
mkdir -p "$L"
stamp() { echo "$(date -d '+9 hours' '+%F %T') KST $*"; }
declare -A PCT=([75]=75 [87p5]=87.5 [93p75]=93.75) TOTAL=([75]=4027023360 [87p5]=4255285248 [93p75]=4369416192)
for m in 75 87p5 93p75; do
  out=outputs/cvpr27/slim_lp_m24_em$m
  if [ ! -f "$out/summary.txt" ]; then
    stamp "lp_m24_em$m card $C"
    # lp_m24's own arguments (cvpr27/plans/2026-10-30_llm-pruner-budget-matched.md) + the expert cut
    "$PY" experiments/llm_pruner/lp_prune.py --importance cvpr27/lp_importance_v1 \
      --units qhead:15,mlp:5928 --mlp-extra-layers 24 --expert-mlp-cut "${PCT[$m]}" \
      --expect-removed "${TOTAL[$m]}" --out "$out" --no-state --gpu "$C" > "$L/lp_m24_em$m.log" 2>&1 \
      || stamp "lp_m24_em$m exited non-zero"
  fi
  stamp "lp_m24_em$m: $(grep -E '^budget=|expert MLP cut|smoke:' "$L/lp_m24_em$m.log" | tr '\n' ' ' | cut -c1-260)"
  out=outputs/cvpr27/slim_tyr_off_em$m
  if [ ! -f "$out/summary.txt" ]; then
    stamp "tyr_off_em$m card $C"
    # tyr_off's own build (tyr_off/build_tyr_off_local.sh make_slim step) + the expert cut; weights written
    "$PY" experiments/head_analysis/make_slim.py --config tyr_u40_r --tyr-supernet cvpr27/tyr_off/supernet_r4 \
      --tyr-config cvpr27/tyr_off/search_r4/final_config.json --importance importance_v2 \
      --expert-mlp-cut "${PCT[$m]}" --out "$out" --gpu "$C" > "$L/tyr_off_em$m.log" 2>&1 \
      || stamp "tyr_off_em$m exited non-zero"
  fi
  stamp "tyr_off_em$m: $(grep -E 'surgery|smoke:' "$L/tyr_off_em$m.log" | tr '\n' ' ' | cut -c1-260)"
done
stamp "card $C done"
