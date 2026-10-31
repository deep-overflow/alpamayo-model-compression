#!/bin/bash
#SBATCH --job-name=tyr_off
#SBATCH --partition=amd_a100nv_8
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=16:00:00
#SBATCH --comment=field=efficientai;appl=pytorch
#SBATCH --output=/scratch/e1997a06/project/amc-tyr/logs/%x-%j.out
#
# Tyr-the-Pruner, official procedure (cvpr27/plans/2026-10-31_tyr-a100.md): supernet + search for
# R = 1..4 with the level spacing halved each round (example.sh), then the slim checkpoint.
# Runs from the track's own code copy /scratch/e1997a06/project/amc-tyr (outputs -> /scratch/.../outputs)
# so the kd2 tree under project/alpamayo-model-compression is left alone; the venv and env.sh are that
# tree's. A finished stage is skipped on resubmission (supernet/search summary.txt, slim_state.pt).
set -euo pipefail
cd /scratch/e1997a06/project/amc-tyr
# shellcheck disable=SC1091
source /scratch/e1997a06/project/alpamayo-model-compression/env.sh
PY=$ALPAMAYO_REPO/.venv/bin/python
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=cvpr27/tyr_off
L=logs/tyr_off
mkdir -p "$L"
stamp() { echo "$(date '+%F %T') $*"; }
stamp "job ${SLURM_JOB_ID:-?} on $(hostname), CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

WD_MHA=2097152   # 12.5% of o_proj = 4 heads; halved each round, floored at 1 head inside supernet.py
WD_MLP=6291456   # 12.5% of down_proj = 1536 channels; 768, 384, 192
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
      stamp "R=$R supernet (card 0)"
      "$PY" cvpr27/code/tyr_off/supernet.py --exp-id "$O/supernet_r$R" --iteration "$R" \
        --weights-diff-mha "$WD_MHA" --weights-diff-mlp "$WD_MLP" "${prev[@]}" --gpu 0 \
        > "$L/supernet_r$R.log" 2>&1 &
      pids+=($!)
    fi
    if [ "$R" -eq 1 ]; then
      stamp "teacher (card 1; skips clips already cached)"
      "$PY" cvpr27/code/tyr_off/search.py --teacher-only --teacher-id "$O/teacher" --gpu 1 \
        > "$L/teacher.log" 2>&1 &
      pids+=($!)
    fi
    for p in "${pids[@]}"; do wait "$p"; done
    # level files are read for every candidate: serve them from node-local disk, not Lustre
    T=/tmp/tyr_off_${SLURM_JOB_ID}/supernet_r$R
    mkdir -p "$T"
    cp -r "outputs/$O/supernet_r$R/." "$T/"
    stamp "R=$R search (4 workers), supernet staged to $T ($(du -sh "$T" | cut -f1))"
    "$PY" cvpr27/code/tyr_off/search.py --supernet "$O/supernet_r$R" --supernet-path "$T" \
      --exp-id "$O/search_r$R" --teacher-id "$O/teacher" --workers 4 --gpu 0,1,2,3 \
      > "$L/search_r$R.log" 2>&1
    rm -rf "$T"
    stamp "R=$R done: $(head -3 "outputs/$O/search_r$R/summary.txt" | tail -1)"
  fi
  WD_MHA=$((WD_MHA / 2))
  WD_MLP=$((WD_MLP / 2))
done

# a level can leave a layer with no heads or no channels (the grid stops short of 1.0 only by the
# centre's fraction); the slim surgery is not set up for that, so stop and look before building
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

if [ ! -f "outputs/$O/../slim_tyr_off/slim_state.pt" ]; then
  stamp "make_slim tyr_off"
  "$PY" experiments/head_analysis/make_slim.py --config tyr_u40_r --out outputs/cvpr27/slim_tyr_off \
    --tyr-supernet "$O/supernet_r4" --tyr-config "$O/search_r4/final_config.json" --gpu 0 \
    > "$L/make_slim.log" 2>&1
fi
stamp "all done"
