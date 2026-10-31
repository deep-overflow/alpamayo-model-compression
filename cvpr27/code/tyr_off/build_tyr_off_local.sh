#!/bin/bash
# build_tyr_off.sbatch on this server instead of NEURON (same stages, same skip-if-done rule).
# Cards are physical ids; pick only empty ones (nvidia-smi right before launch) and keep them one
# GPU type, so every search worker scores candidates with the same kernels.
#
#   CARDS=0,1,3 bash cvpr27/code/tyr_off/build_tyr_off_local.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
export HF_HOME=$HOME/.cache/huggingface HF_HUB_CACHE=/mnt/nvme1n1/ad_vla/cache/hub HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${CARDS:?set CARDS, e.g. 0,1,3}
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
O=cvpr27/tyr_off
L=outputs/$O/logs
mkdir -p "$L"
IFS=, read -ra CS <<< "$CARDS"
TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${CS[0]}")
stamp() { echo "$(date '+%F %T') $*"; }
stamp "local build on cards $CARDS ($TYPE)"
# Search workers: every card of the same type that is empty right now (< 100 MiB), re-read before each
# search -- the supernet/teacher cards are free again by then, and the user allows more cards (2026-10-31).
# One type only, so all candidates of a generation are scored with the same kernels.
search_cards() {
  nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader,nounits |
    awk -F', ' -v t="$TYPE" '$2 == t && $3 < 100 {print $1}' | paste -sd,
}

WD_MHA=2097152
WD_MLP=6291456
for R in 1 2 3 4; do
  if [ -f "outputs/$O/search_r$R/summary.txt" ]; then
    stamp "R=$R done, skip"
  else
    prev=()
    if [ "$R" -gt 1 ]; then
      prev=(--prev-supernet "$O/supernet_r$((R - 1))" --prev-config "$O/search_r$((R - 1))/final_config.json")
    fi
    pids=()
    if [ ! -f "outputs/$O/supernet_r$R/summary.txt" ]; then
      stamp "R=$R supernet (card ${CS[0]})"
      "$PY" cvpr27/code/tyr_off/supernet.py --exp-id "$O/supernet_r$R" --iteration "$R" \
        --weights-diff-mha "$WD_MHA" --weights-diff-mlp "$WD_MLP" "${prev[@]}" --gpu 0 \
        > "$L/supernet_r$R.log" 2>&1 &
      pids+=($!)
    fi
    if [ "$R" -eq 1 ]; then
      stamp "teacher (card ${CS[1]}; skips clips already cached)"
      "$PY" cvpr27/code/tyr_off/search.py --teacher-only --teacher-id "$O/teacher" --gpu 1 \
        > "$L/teacher.log" 2>&1 &
      pids+=($!)
    fi
    for p in "${pids[@]}"; do wait "$p"; done
    SC=$(search_cards)
    [ -n "$SC" ] || { stamp "no empty $TYPE card for the search"; exit 1; }
    N=$(tr ',' '\n' <<< "$SC" | wc -l)
    stamp "R=$R search ($N workers on cards $SC)"
    CUDA_VISIBLE_DEVICES=$SC "$PY" cvpr27/code/tyr_off/search.py --supernet "$O/supernet_r$R" \
      --exp-id "$O/search_r$R" --teacher-id "$O/teacher" --workers "$N" --gpu "$(seq -s, 0 $((N - 1)))" \
      > "$L/search_r$R.log" 2>&1
    stamp "R=$R done: $(head -3 "outputs/$O/search_r$R/summary.txt" | tail -1)"
  fi
  WD_MHA=$((WD_MHA / 2))
  WD_MLP=$((WD_MLP / 2))
done

"$PY" - "$O" <<'PY'
import json, sys
from pathlib import Path
o = Path("outputs") / sys.argv[1]
meta = json.loads((o / "supernet_r4/metadata.json").read_text())
cfg = json.loads((o / "search_r4/final_config.json").read_text())
keep = {n: meta["levels"][n][str(lv)]["keep"] for n, lv in cfg.items()}
q = [keep[n] for n in meta["layer_names"] if "self_attn" in n]
m = [keep[n] for n in meta["layer_names"] if "mlp" in n]
print("final kept heads per layer", q, "sum", sum(q))
print("final kept channels per layer", m, "sum", sum(m))
removed = (36 * 32 - sum(q)) * 2 * 128 * 4096 + (36 * 12288 - sum(m)) * 3 * 4096
print("removed", removed)
assert removed == 2_657_452_032, removed
assert min(q) > 0 and min(m) > 0, "a layer lost every head or channel -- handle before make_slim"
PY

if [ ! -f "outputs/cvpr27/slim_tyr_off/slim_state.pt" ]; then
  stamp "make_slim tyr_off"
  "$PY" experiments/head_analysis/make_slim.py --config tyr_u40_r --out outputs/cvpr27/slim_tyr_off \
    --tyr-supernet "$O/supernet_r4" --tyr-config "$O/search_r4/final_config.json" --gpu 0 \
    > "$L/make_slim.log" 2>&1
fi
stamp "all done"
