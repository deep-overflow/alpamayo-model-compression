#!/bin/bash
# Wanda u50 closed loop on Origin150 x 2 (user, 11-13 01:xx: "run Wanda u50 on Origin150, GPUs 4-7"); same harness and
# settings as the other u50 arms (iclr27-rebuttal/plans/2026-11-09_u50-closedloop-origin150.md).
#   1. build on GPU 4: wanda_u50_v2 (paper Wanda recipe wanda_u40_v2 at u50, --importance importance_v2 like the paper
#      build); G0 = wanda_u40_v2 rebuilt --no-state must equal outputs/slim_wanda_u40_v2 unit for unit; removed must be
#      3,321,888,768 (the u50 budget of every other arm)
#   2. 150 scenes x 2 rollouts round-robin over GPUs 4-7, merge into u50o_merged_slim_wanda_u50_v2, G1/G2 gates
#
#   setsid nohup bash iclr27-rebuttal/code/run_u50_wanda.sh > outputs/iclr27-rebuttal/u50_origin150/run_wanda.log 2>&1 &
set -u
REPO=/home/cvlab21/project/chan/alpamayo-model-compression
OUT=$REPO/outputs/iclr27-rebuttal/u50_origin150
RUNS=/home/cvlab21/project/chan/alpasim-runs
DRIVERS=/mnt/nvme1n1/ad_vla/data/alpasim/drivers
PY=$REPO/.venv/bin/python
APY=/mnt/nvme1n1/ad_vla/venvs/alpasim-chan/bin/python
H=$REPO/experiments/head_analysis
GPUS="4 5 6 7"
P=u50oW_
CFG=slim_wanda_u50_v2
U50=3321888768
stamp() { echo "$(date -d '+9 hours' '+%F %T') KST $*"; }
mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1"; }
empty() { for g in $GPUS; do [ "$(mib "$g")" -le 10 ] || return 1; done; }
cleanup() {  # docker compose lowercases project names, so match case-insensitively
  for c in $(docker ps -a --format '{{.Names}}' | grep -i "^$1"); do docker rm -f "$c" > /dev/null; done
  for nw in $(docker network ls --format '{{.Name}}' | grep -i "^$1"); do
    [ "$(docker network inspect -f '{{len .Containers}}' "$nw")" = 0 ] && docker network rm "$nw" > /dev/null
  done
}
cd "$REPO"
source /home/cvlab21/project/chan/alpasim-runs/.hf_env   # valid token + HF_HOME (full_right is invalid since 11-09)
export HF_HUB_CACHE=/mnt/nvme1n1/ad_vla/cache/hub CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_ENABLE_HF_TRANSFER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

until empty && sleep 30 && empty; do stamp "waiting for empty GPUs $GPUS"; sleep 60; done

# 1. build + gates
mkdir -p "$OUT/g0"
if [ ! -f outputs/$CFG/slim_state.pt ]; then
  $PY $H/make_slim.py --config wanda_u50_v2 --importance importance_v2 --out outputs/$CFG --gpu 4 \
    > "$OUT/build_wanda_u50_v2.log" 2>&1 &
fi
if [ ! -f "$OUT/g0/wanda_u40_v2/slim_meta.json" ]; then
  $PY $H/make_slim.py --config wanda_u40_v2 --importance importance_v2 --no-state \
    --out outputs/iclr27-rebuttal/u50_origin150/g0/wanda_u40_v2 --gpu 5 > "$OUT/build_g0_wanda_u40_v2.log" 2>&1 &
fi
wait
$PY - "$REPO" "$U50" > "$OUT/gates_wanda_build.txt" 2>&1 << 'EOF'
import json, sys
from pathlib import Path
R, U50 = Path(sys.argv[1]) / "outputs", int(sys.argv[2])
m = lambda d: json.loads((R / d / "slim_meta.json").read_text())
u = lambda x: [(sorted(l["q"]), sorted(l["mlp"])) for t in ("vlm", "expert") for l in x[t]]
g0 = u(m("iclr27-rebuttal/u50_origin150/g0/wanda_u40_v2")) == u(m("slim_wanda_u40_v2"))
x = m("slim_wanda_u50_v2")
b = x["params"]["removed"] == U50 and x.get("importance_from") == "importance_v2"
print(f"G0 slim_wanda_u40_v2: {'PASS' if g0 else 'FAIL'}")
print(f"B slim_wanda_u50_v2: removed {x['params']['removed']:,}, importance {x.get('importance_from')}: {'PASS' if b else 'FAIL'}")
print("gates:", "PASS" if g0 and b else "FAIL")
EOF
stamp "build: $(tail -1 "$OUT/gates_wanda_build.txt")"
grep -q "gates: PASS" "$OUT/gates_wanda_build.txt" || { stamp "build gates failed -- not launching"; exit 1; }

# 2. closed loop
$PY -c "import pandas as pd; pd.read_csv('$OUT/scenes_main.csv')[['scene_id', 'uuid']].to_csv('$OUT/scenes_all150.csv', index=False)"
[ -e "$DRIVERS/$CFG" ] || cp -al "outputs/$CFG" "$DRIVERS/$CFG"
until empty && sleep 30 && empty; do stamp "$CFG: waiting for empty GPUs $GPUS"; sleep 60; done
stamp "$CFG: 150 scenes x 2 rollouts over GPUs $GPUS"
SUITE=public_2601_u50oW PARENT_SUITE=public_2601 PREFIX=$P DRIVER_OMP_THREADS=8 SCENES_CSV=$OUT/scenes_all150.csv \
  bash "$H/launch_alpasim_shards.sh" "$CFG" 150 2 "$GPUS"
stamp "$CFG: shards finished (exit $?); Rollout FAILED lines $(cat "$RUNS"/${P}${CFG}_sh*.launch.log | grep -c 'Rollout FAILED')"
cleanup "${P}${CFG}_sh"
merged=u50o_merged_$CFG
$PY "$H/merge_alpasim_shards.py" --runs-root "$RUNS" --shards $(for i in 0 1 2 3; do echo "${P}${CFG}_sh$i"; done) \
  --out "$merged" --expect-scenes 150
rc=$?
stamp "$CFG: merged -> $RUNS/$merged (exit $rc)"
(cd experiments/evaluation && $APY s600_smoke_gates.py --run "$merged" --logs "${P}$CFG" \
   --scenes "$OUT/scenes_main.csv" --out "$OUT/gates_$CFG") > "$OUT/gates_$CFG.txt" 2>&1
stamp "$CFG: $(grep -m1 'gates:' "$OUT/gates_$CFG.txt")"
[ "$rc" = 0 ] && touch "$OUT/$CFG.DONE"
