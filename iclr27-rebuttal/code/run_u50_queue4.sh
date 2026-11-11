#!/bin/bash
# FM-only u50 and CE-only u50 closed loop on Origin150 x 2, after run_u50_queue3.sh (user, 11-12 01:xx: "when everything is
# done, also run FM-only and CE-only at u50"). Paper (ICLR) recipes at u50: traj_u50_v2 (paper FM-only = traj_u40_v2)
# and coc_u50_v2 (paper CE-only = coc_u40_v2), importance_v2, layer 0 cut too (pairs with plain MoOP u50, dual_u50_v2).
#   0. wait for queue3 to exit and for eight empty cards
#   1. builds on GPUs 0-3: traj_u50_v2, coc_u50_v2 (with state); G0 = traj_u40_v2 / coc_u40_v2 rebuilt --no-state must
#      equal the paper recipes unit for unit; removed = 3,321,888,768; importance_from = importance_v2
#   2. FM-only then CE-only (user order): 150 scenes x 2 round-robin over GPUs 0-7, merge, G1/G2 gates
#
#   setsid nohup bash iclr27-rebuttal/code/run_u50_queue4.sh > outputs/iclr27-rebuttal/u50_origin150/run_queue4.log 2>&1 &
set -u
REPO=/home/cvlab21/project/chan/alpamayo-model-compression
OUT=$REPO/outputs/iclr27-rebuttal/u50_origin150
RUNS=/home/cvlab21/project/chan/alpasim-runs
DRIVERS=/mnt/nvme1n1/ad_vla/data/alpasim/drivers
PY=$REPO/.venv/bin/python
APY=/mnt/nvme1n1/ad_vla/venvs/alpasim-chan/bin/python
H=$REPO/experiments/head_analysis
LAUNCH=$H/launch_alpasim_shards.sh
MERGE=$H/merge_alpasim_shards.py
G8="0 1 2 3 4 5 6 7"
QP=u50oQ_
U50=3321888768
stamp() { echo "$(date -d '+9 hours' '+%F %T') KST $*"; }
mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1"; }
empty() { for g in $1; do [ "$(mib "$g")" -le 10 ] || return 1; done; }
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

n=0
while pgrep -f "run_u50_queue3.sh" > /dev/null; do
  [ $((n % 60)) = 0 ] && stamp "waiting for queue3 (LLM-Pruner, MoOP u50) to finish"
  n=$((n + 1)); sleep 60
done
until empty "$G8" && sleep 30 && empty "$G8"; do stamp "waiting for eight empty cards"; sleep 60; done

# 1. builds + gates
mkdir -p "$OUT/g0"
b() {  # config gpu [extra]
  local cfg=$1 g=$2; shift 2
  $PY $H/make_slim.py --config "$cfg" --importance importance_v2 "$@" --gpu "$g" > "$OUT/build_$cfg.log" 2>&1
  echo "$cfg exit $?"
}
[ -f outputs/slim_traj_u50_v2/slim_state.pt ] || b traj_u50_v2 0 --out outputs/slim_traj_u50_v2 &
[ -f outputs/slim_coc_u50_v2/slim_state.pt ] || b coc_u50_v2 1 --out outputs/slim_coc_u50_v2 &
[ -f "$OUT/g0/traj_u40_v2/slim_meta.json" ] || b traj_u40_v2 2 --no-state --out outputs/iclr27-rebuttal/u50_origin150/g0/traj_u40_v2 &
[ -f "$OUT/g0/coc_u40_v2/slim_meta.json" ] || b coc_u40_v2 3 --no-state --out outputs/iclr27-rebuttal/u50_origin150/g0/coc_u40_v2 &
wait
$PY - "$REPO" "$U50" > "$OUT/gates_fm_ce_builds.txt" 2>&1 << 'EOF'
import json, sys
from pathlib import Path
R, U50 = Path(sys.argv[1]) / "outputs", int(sys.argv[2])
m = lambda d: json.loads((R / d / "slim_meta.json").read_text())
u = lambda x: [(sorted(l["q"]), sorted(l["mlp"])) for t in ("vlm", "expert") for l in x[t]]
ok = True
for new, paper in (("iclr27-rebuttal/u50_origin150/g0/traj_u40_v2", "slim_traj_u40_v2"),
                   ("iclr27-rebuttal/u50_origin150/g0/coc_u40_v2", "slim_coc_u40_v2")):
    same = u(m(new)) == u(m(paper)); ok &= same
    print(f"G0 {paper}: {'PASS' if same else 'FAIL'}")
for d in ("slim_traj_u50_v2", "slim_coc_u50_v2"):
    x = m(d); good = x["params"]["removed"] == U50 and x.get("importance_from") == "importance_v2"; ok &= good
    print(f"B {d}: removed {x['params']['removed']:,}, importance {x.get('importance_from')}: {'PASS' if good else 'FAIL'}")
print("gates:", "PASS" if ok else "FAIL")
EOF
stamp "builds: $(tail -1 "$OUT/gates_fm_ce_builds.txt")"
grep -q "gates: PASS" "$OUT/gates_fm_ce_builds.txt" || { stamp "build gates failed -- not launching"; exit 1; }

# 2. FM-only, then CE-only
$PY -c "import pandas as pd; pd.read_csv('$OUT/scenes_main.csv')[['scene_id', 'uuid']].to_csv('$OUT/scenes_all150.csv', index=False)"
for CFG in slim_traj_u50_v2 slim_coc_u50_v2; do
  [ -f "$OUT/$CFG.DONE" ] && { stamp "$CFG: already done"; continue; }
  [ -e "$DRIVERS/$CFG" ] || cp -al "outputs/$CFG" "$DRIVERS/$CFG"
  until empty "$G8" && sleep 30 && empty "$G8"; do stamp "$CFG: waiting for eight empty cards"; sleep 60; done
  stamp "$CFG: 150 scenes x 2 rollouts over GPUs $G8"
  SUITE=public_2601_u50oQ PARENT_SUITE=public_2601 PREFIX=$QP DRIVER_OMP_THREADS=8 SCENES_CSV=$OUT/scenes_all150.csv \
    bash "$LAUNCH" "$CFG" 150 2 "$G8"
  stamp "$CFG: shards finished (exit $?); Rollout FAILED lines $(cat "$RUNS"/${QP}${CFG}_sh*.launch.log | grep -c 'Rollout FAILED')"
  cleanup "${QP}${CFG}_sh"
  shards=$(for i in 0 1 2 3 4 5 6 7; do echo "${QP}${CFG}_sh$i"; done)
  merged=u50o_merged_$CFG
  $PY "$MERGE" --runs-root "$RUNS" --shards $shards --out "$merged" --expect-scenes 150
  rc=$?
  stamp "$CFG: merged -> $RUNS/$merged (exit $rc)"
  (cd experiments/evaluation && $APY s600_smoke_gates.py --run "$merged" --logs "${QP}$CFG" \
     --scenes "$OUT/scenes_main.csv" --out "$OUT/gates_$CFG") > "$OUT/gates_$CFG.txt" 2>&1
  stamp "$CFG: $(grep -m1 'gates:' "$OUT/gates_$CFG.txt")"
  [ "$rc" = 0 ] && touch "$OUT/$CFG.DONE"
done
stamp "queue4 done"
