#!/bin/bash
# Wanda u50 closed loop, continued on eight cards (user, 11-13 12:5x: "use GPUs 0-3 too, run on 8"). Same recipe as
# run_u50_tyr_resume.sh: the 4-card run of run_u50_wanda.sh was stopped (its launcher group killed, containers removed)
#   1. u50oW_slim_wanda_u50_v2_sh0-3 -> *_part1; scenes without both rollouts finished -> *_part1_incomplete;
#      part1 is scored by alpasim's own eval.aggregation (its rollouts already carry online metrics.parquet)
#   2. the scenes not in part1 run round-robin over GPUs 0-7 (prefix u50oW2_)
#   3. merge part1 (4 shards) + resume (8 shards) into u50o_merged_slim_wanda_u50_v2 (150 scenes), G1/G2 gates
#
#   setsid nohup bash iclr27-rebuttal/code/run_u50_wanda_resume.sh > outputs/iclr27-rebuttal/u50_origin150/run_wanda_resume.log 2>&1 &
set -u
REPO=/home/cvlab21/project/chan/alpamayo-model-compression
OUT=$REPO/outputs/iclr27-rebuttal/u50_origin150
RUNS=/home/cvlab21/project/chan/alpasim-runs
ALPASIM=/home/cvlab21/project/chan/alpasim
PY=$REPO/.venv/bin/python
APY=/mnt/nvme1n1/ad_vla/venvs/alpasim-chan/bin/python
H=$REPO/experiments/head_analysis
CFG=slim_wanda_u50_v2
RP=u50oW2_
G8="0 1 2 3 4 5 6 7"
stamp() { echo "$(date -d '+9 hours' '+%F %T') KST $*"; }
mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1"; }
empty() { for g in $G8; do [ "$(mib "$g")" -le 10 ] || return 1; done; }
cleanup() {  # docker compose lowercases project names, so match case-insensitively
  for c in $(docker ps -a --format '{{.Names}}' | grep -i "^$1"); do docker rm -f "$c" > /dev/null; done
  for nw in $(docker network ls --format '{{.Name}}' | grep -i "^$1"); do
    [ "$(docker network inspect -f '{{len .Containers}}' "$nw")" = 0 ] && docker network rm "$nw" > /dev/null
  done
}
cd "$REPO"
pgrep -f "run_u50_wanda.sh" > /dev/null && { stamp "the 4-card run is still alive -- stop it first"; exit 1; }
cleanup "u50oW_${CFG}_sh"

# 1. salvage the stopped shards
for i in 0 1 2 3; do
  d=$RUNS/u50oW_${CFG}_sh$i
  p=${d}_part1
  if [ -d "$d" ] && [ ! -e "$p" ]; then
    mv "$d" "$p" && mv "$d.launch.log" "$p.launch.log"
  fi
  [ -d "$p" ] || { stamp "no $p"; exit 1; }
  for s in "$p"/rollouts/*/; do
    n=$(ls "$s"*/_complete 2> /dev/null | wc -l)
    if [ "$n" -lt 2 ]; then mkdir -p "${p}_incomplete"; mv "$s" "${p}_incomplete/"; fi
  done
  if [ ! -f "$p/aggregate/results-summary.json" ]; then
    (cd "$ALPASIM" && $APY -m eval.aggregation.main --array_job_dir "$p" --config_path "$p/eval-config.yaml") \
      > "$OUT/wanda_part1_sh${i}_aggregation.log" 2>&1 || { stamp "aggregation of $p failed"; exit 1; }
  fi
  stamp "part1 sh$i: $(ls -d "$p"/rollouts/*/ | wc -l) scenes kept, $($PY -c "import json;print(len(json.load(open('$p/aggregate/results-summary.json'))['rollouts']))") rollouts scored"
done

# 2. the rest on eight cards
$PY - "$OUT" "$RUNS" "$CFG" << 'EOF'
import sys
from pathlib import Path
import pandas as pd
out, runs, cfg = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
main = pd.read_csv(out / "scenes_main.csv").sort_values("scene_id").reset_index(drop=True)
done = {p.name for i in range(4) for p in (runs / f"u50oW_{cfg}_sh{i}_part1" / "rollouts").iterdir()}
rest = main[~main.scene_id.isin(done)][["scene_id", "uuid"]].reset_index(drop=True)
rest["shard"] = rest.index % 8
rest.to_csv(out / "scenes_wanda_resume.csv", index=False)
print(f"part1 scenes {len(done)}, resume scenes {len(rest)}: {rest.shard.value_counts().sort_index().to_dict()}")
EOF
N=$(($(wc -l < "$OUT/scenes_wanda_resume.csv") - 1))
if [ ! -f "$OUT/wanda_resume.DONE" ]; then
  until empty && sleep 30 && empty; do stamp "waiting for eight empty cards"; sleep 60; done
  stamp "$CFG: resume $N scenes x 2 on GPUs $G8"
  SUITE=public_2601_u50oW2 PARENT_SUITE=public_2601 PREFIX=$RP DRIVER_OMP_THREADS=8 SCENES_CSV=$OUT/scenes_wanda_resume.csv \
    bash "$H/launch_alpasim_shards.sh" "$CFG" "$N" 2 "$G8"
  stamp "$CFG: resume shards finished (exit $?); Rollout FAILED lines $(cat "$RUNS"/${RP}${CFG}_sh*.launch.log | grep -c 'Rollout FAILED')"
  cleanup "${RP}${CFG}_sh"
  ok=1
  for i in 0 1 2 3 4 5 6 7; do [ -f "$RUNS/${RP}${CFG}_sh$i/aggregate/results-summary.json" ] || ok=0; done
  [ "$ok" = 1 ] && touch "$OUT/wanda_resume.DONE"
fi

# 3. one merge
shards="$(for i in 0 1 2 3; do echo "u50oW_${CFG}_sh${i}_part1"; done) $(for i in 0 1 2 3 4 5 6 7; do echo "${RP}${CFG}_sh$i"; done)"
for s in $shards; do [ -f "$RUNS/$s/aggregate/results-summary.json" ] || { stamp "$s has no summary -- not merging"; exit 1; }; done
merged=u50o_merged_$CFG
$PY "$H/merge_alpasim_shards.py" --runs-root "$RUNS" --shards $shards --out "$merged" --expect-scenes 150
rc=$?
stamp "$CFG: merged -> $RUNS/$merged (exit $rc)"
(cd experiments/evaluation && $APY s600_smoke_gates.py --run "$merged" --logs "${RP}$CFG" \
   --scenes "$OUT/scenes_main.csv" --out "$OUT/gates_$CFG") > "$OUT/gates_$CFG.txt" 2>&1
stamp "$CFG: $(grep -m1 'gates:' "$OUT/gates_$CFG.txt")"
[ "$rc" = 0 ] && touch "$OUT/$CFG.DONE"
