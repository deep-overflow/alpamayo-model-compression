#!/bin/bash
# u50 closed loop on Hard100 x 2 (user 11-15: "u50에서 hard100도 실험을 진행해보자", MoOP first, GPUs 4-7 only).
# One arm per call. Harness and settings are those of the u50 Origin150 arms (patched hand-over harness,
# DRIVER_OMP_THREADS=8; iclr27-rebuttal/plans/2026-11-09_u50-closedloop-origin150.md); scene list and suite are those of
# the ICLR Hard100 runs (outputs/scene_difficulty/hard100_suite.csv, parent suite public_2601). Plan:
# iclr27-rebuttal/plans/2026-11-15_u50-closedloop-hard100.md
#   1. wait for GPUs $GPUS to read empty (twice, 30 s apart)
#   2. 100 scenes x 2 rollouts over $GPUS, merge into u50h_merged_<cfg>
#   3. gates: s600_smoke_gates.py (G1 = no non-gate failure, G2 = hand-over fixed); on Hard100 G1 fails by design on the
#      simulator's route sanity check (4bad2f63 / adb899bd every arm, 3b934c9a / ceb3d0d5 arm-dependent in the ICLR runs),
#      so G1h = every non-gate failure is that sanity check
#
#   setsid nohup bash iclr27-rebuttal/code/run_u50_hard100.sh slim_dual_u50_v2 \
#     > outputs/iclr27-rebuttal/u50_hard100/run_slim_dual_u50_v2.log 2>&1 &
set -u
CFG=${1:?usage: run_u50_hard100.sh <cfg>, e.g. slim_dual_u50_v2}
REPO=/home/cvlab21/project/chan/advla/alpamayo-model-compression
OUT=$REPO/outputs/iclr27-rebuttal/u50_hard100
RUNS=/home/cvlab21/project/chan/advla/alpasim-runs
DRIVERS=/mnt/nvme1n1/ad_vla/data/alpasim/drivers
PY=${TOOLS_PY:-$REPO/.venv/bin/python}   # merge / G1h / suites need only pandas; TOOLS_PY overrides (repo .venv deleted 11-17)
APY=/mnt/nvme1n1/ad_vla/venvs/alpasim-chan/bin/python
H=$REPO/experiments/head_analysis
GPUS=${GPUS-"4 5 6 7"}
P=u50h_
SCENES=$REPO/outputs/scene_difficulty/hard100_suite.csv
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
source "$RUNS/.hf_env"   # valid token + HF_HOME (full_right is invalid since 11-09)
export HF_HUB_CACHE=/mnt/nvme1n1/ad_vla/cache/hub CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_ENABLE_HF_TRANSFER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

[ -f "outputs/$CFG/slim_meta.json" ] || { stamp "$CFG: no checkpoint at outputs/$CFG"; exit 1; }
[ -e "$DRIVERS/$CFG" ] || cp -al "outputs/$CFG" "$DRIVERS/$CFG"
until empty && sleep 30 && empty; do stamp "$CFG: waiting for empty GPUs $GPUS"; sleep 60; done
stamp "$CFG: 100 Hard100 scenes x 2 rollouts over GPUs $GPUS"
SUITE=public_2601_hard100_u50 PARENT_SUITE=public_2601 PREFIX=$P DRIVER_OMP_THREADS=8 SCENES_CSV=$SCENES \
  bash "$H/launch_alpasim_shards.sh" "$CFG" 100 2 "$GPUS"
stamp "$CFG: shards finished (exit $?); Rollout FAILED lines $(cat "$RUNS"/${P}${CFG}_sh*.launch.log | grep -c 'Rollout FAILED')"
cleanup "${P}${CFG}_sh"
merged=u50h_merged_$CFG
n=$(echo $GPUS | wc -w)
$PY "$H/merge_alpasim_shards.py" --runs-root "$RUNS" --shards $(for ((i = 0; i < n; i++)); do echo "${P}${CFG}_sh$i"; done) \
  --out "$merged" --expect-scenes 100
rc=$?
stamp "$CFG: merged -> $RUNS/$merged (exit $rc)"
(cd experiments/evaluation && $APY s600_smoke_gates.py --run "$merged" --logs "${P}$CFG" \
   --scenes "$SCENES" --out "$OUT/gates_$CFG") > "$OUT/gates_$CFG.txt" 2>&1
$PY - "$RUNS/$merged" >> "$OUT/gates_$CFG.txt" 2>&1 << 'EOF'
import json, sys
from collections import Counter
from pathlib import Path
summ = json.loads((Path(sys.argv[1]) / "aggregate" / "results-summary.json").read_text())["rollouts"]
bad = [(r["clipgt_id"][7:15], str(r.get("failure_reason"))) for r in summ
       if r.get("failure_reason") not in (None, "collision_at_fault", "offroad")]
print(f"rollouts {len(summ)}, scenes {len({r['clipgt_id'] for r in summ})}")
print("non-gate failures by scene:", dict(Counter(s for s, _ in bad)))
ok = all("sanity check" in why for _, why in bad)
print(f"G1h (every non-gate failure is the route sanity check): {'PASS' if ok else 'FAIL'}")
for s, why in bad:
    if "sanity check" not in why:
        print("  other:", s, why[:200])
EOF
stamp "$CFG: $(grep -m1 'gates:' "$OUT/gates_$CFG.txt"); $(grep -m1 'G1h' "$OUT/gates_$CFG.txt")"
[ "$rc" = 0 ] && touch "$OUT/$CFG.DONE"
