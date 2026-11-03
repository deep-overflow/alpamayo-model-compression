#!/bin/bash
# Local recipe builds for the ratio sweep (cvpr27/plans/2026-11-03_baselines-ratio-sweep.md). One card per call,
# steps run in order; a finished recipe (slim_meta.json / wanda.npz) is skipped on rerun.
#
#   CARD=4 bash cvpr27/code/build_ratio_local.sh wanda 35 50
#   CARD=5 bash cvpr27/code/build_ratio_local.sh wanda 45 -- lp 40 35 45 50 -- fmce 40 35 45 50
set -uo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME=$HOME/.cache/huggingface HF_HUB_CACHE=/mnt/nvme1n1/ad_vla/cache/hub HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
C=${CARD:?set CARD}
L=outputs/cvpr27/ratio_sweep/logs
mkdir -p "$L"
stamp() { echo "$(date -d '+9 hours' '+%F %T') KST $*"; }

declare -A CUTQ=([35]=11 [40]=13 [45]=14 [50]=16) CUTM=([35]=4301 [40]=4898 [45]=5530 [50]=6144)
# LLM-Pruner budget match (lp_core rule n - int(n(1-r)), heads in 128-channel groups, layers 4-33): best r's head
# count, MLP solved to MoOP u{N} exactly, the remainder +1 in the first E scope layers (u40 = lp_m24)
declare -A LPQ=([35]=13 [40]=15 [45]=17 [50]=19) LPM=([35]=5178 [40]=5928 [45]=6618 [50]=7389)
declare -A LPE=([35]=8 [40]=24 [45]=28 [50]=26)
declare -A BUDGET=([35]=2317860864 [40]=2657452032 [45]=2974777344 [50]=3321888768)

wanda() {  # <N>
  local n=$1 exp=cvpr27/wandasp_u$1 out=outputs/cvpr27/slim_wandasp_u$1
  if [ ! -f "outputs/$exp/wanda.npz" ]; then
    stamp "wanda-sp u$n collect (cut ${CUTQ[$n]} / ${CUTM[$n]}) card $C"
    "$PY" cvpr27/code/run_wanda_sp.py --exp-id "$exp" --cut-q "${CUTQ[$n]}" --cut-mlp "${CUTM[$n]}" --gpu "$C" \
      > "$L/wandasp_u${n}_collect.log" 2>&1 || { stamp "wanda-sp u$n collect FAILED"; return 1; }
  fi
  if [ ! -f "$out/slim_meta.json" ]; then
    stamp "wanda-sp u$n make_slim"
    "$PY" experiments/head_analysis/make_slim.py --config "wanda_u${n}_v2" --wanda "$exp" --importance importance_v2 \
      --out "$out" --no-state --gpu "$C" > "$L/wandasp_u${n}_build.log" 2>&1 || stamp "wanda-sp u$n make_slim exited non-zero"
  fi
  stamp "wanda-sp u$n: $(grep -E 'surgery|smoke:' "$L/wandasp_u${n}_build.log" | tr '\n' ' ' | cut -c1-220)"
}

lp() {  # <N>; u40 rebuilds lp_m24 into a check dir (gate)
  local n=$1 out=outputs/cvpr27/slim_lp_u$1
  [ "$n" = 40 ] && out=outputs/cvpr27/check_lp_m24
  if [ ! -f "$out/slim_meta.json" ]; then
    stamp "llm-pruner u$n (qhead ${LPQ[$n]}, mlp ${LPM[$n]} +1 x ${LPE[$n]}) card $C"
    "$PY" experiments/llm_pruner/lp_prune.py --importance cvpr27/lp_importance_v1 \
      --units "qhead:${LPQ[$n]},mlp:${LPM[$n]}" --mlp-extra-layers "${LPE[$n]}" --expect-removed "${BUDGET[$n]}" \
      --out "$out" --no-state --gpu "$C" > "$L/lp_u${n}.log" 2>&1 || stamp "llm-pruner u$n exited non-zero"
  fi
  stamp "llm-pruner u$n: $(grep -E '^budget=|smoke:' "$L/lp_u${n}.log" | tr '\n' ' ' | cut -c1-220)"
}

fmce() {  # <N>; u40 rebuilds both shipped controls into check dirs (gate)
  local n=$1 stem cfg out
  for stem in trajlastcoc coc; do
    cfg=${stem}_u${n}_l0keep_v2
    out=outputs/cvpr27/slim_$cfg
    [ "$n" = 40 ] && out=outputs/cvpr27/check_$cfg
    if [ ! -f "$out/slim_meta.json" ]; then
      stamp "$cfg card $C"
      "$PY" experiments/head_analysis/make_slim.py --config "$cfg" --importance importance_v2 \
        --out "$out" --no-state --gpu "$C" > "$L/$cfg.log" 2>&1 || stamp "$cfg exited non-zero"
    fi
    stamp "$cfg: $(grep -E 'l0keep:|surgery' "$L/$cfg.log" | tr '\n' ' ' | cut -c1-220)"
  done
}

kind=""
for a in "$@"; do
  case $a in
    --) kind="" ;;
    wanda | lp | fmce) kind=$a ;;
    *) "$kind" "$a" ;;
  esac
done
stamp "card $C done"
