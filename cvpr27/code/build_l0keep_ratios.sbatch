#!/bin/bash
#SBATCH --job-name=build_l0k
#SBATCH --partition=amd_a100nv_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --comment=field=efficientai;appl=pytorch
#SBATCH --output=/scratch/e1997a06/project/amc-tyr/logs/%x-%j.out
#
# MoOP l0keep recipes at other ratios (cvpr27/plans/2026-11-01_l0keep-ratio-sweep.md), built on NEURON because
# every local card is busy. Runs from the track's code copy (amc-tyr, current make_slim). Gate: dual_u40_l0keep_v2
# is rebuilt and must equal the shipped outputs/slim_dual_u40_l0keep_v2 unit for unit.
set -uo pipefail
cd /scratch/e1997a06/project/amc-tyr
# shellcheck disable=SC1091
source /scratch/e1997a06/project/alpamayo-model-compression/env.sh
PY=$ALPAMAYO_REPO/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is)"
for N in 40 35 45 50; do
  out=outputs/cvpr27/slim_dual_u${N}_l0keep_v2
  [ "$N" = 40 ] && out=outputs/cvpr27/check_dual_u40_l0keep_v2
  if [ -f "$out/slim_meta.json" ]; then echo "skip $out"; continue; fi
  # the trailing smoke rollout may need a cache this node lacks; the recipe is written before it
  # --importance must be named: make_slim defaults to importance_v1, MoOP is built from importance_v2
  "$PY" experiments/head_analysis/make_slim.py --config "dual_u${N}_l0keep_v2" --importance importance_v2 \
    --out "$out" --no-state --gpu 0 \
    > "logs/build_l0k_u${N}.log" 2>&1 || echo "make_slim u${N} exited non-zero (see log)"
  grep -E "l0keep:|surgery|saved" "logs/build_l0k_u${N}.log" | sed "s/^/u${N}: /"
done
"$PY" - <<'PY'
import json
a = json.load(open("outputs/cvpr27/check_dual_u40_l0keep_v2/slim_meta.json"))
b = json.load(open("outputs/slim_dual_u40_l0keep_v2/slim_meta.json"))
same = all(a[t][i] == b[t][i] for t in ("vlm", "expert") for i in range(len(b[t])))
print("GATE u40 rebuild == shipped slim_dual_u40_l0keep_v2:", same)
for n in (35, 40, 45, 50):
    p = "outputs/cvpr27/" + ("check_dual_u40_l0keep_v2" if n == 40 else f"slim_dual_u{n}_l0keep_v2")
    m = json.load(open(p + "/slim_meta.json"))
    q = [len(layer["q"]) for layer in m["vlm"]]
    ml = [len(layer["mlp"]) for layer in m["vlm"]]
    print(f"u{n}: removed {m['params']['removed']:,} ({100 * m['params']['removed'] / m['params']['full']:.2f}% of model);"
          f" kept heads L0 {q[0]}, other layers {min(q[1:])}-{max(q[1:])}; kept MLP {min(ml)}-{max(ml)}")
PY
echo "done $(date -Is)"
