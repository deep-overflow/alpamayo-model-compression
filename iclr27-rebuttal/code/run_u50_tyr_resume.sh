#!/bin/bash
# Tyr u50 on GPUs 0-3 (user, 11-10 21:0x): first the rest of the Ada half that was stopped at 20:57 KST, then Tyr's own
# Blackwell half, then one 150-scene merge (iclr27-rebuttal/plans/2026-11-09_u50-closedloop-origin150.md).
#   1. u50o_slim_tyr_u50_r_sh4-7 -> *_part1; scenes without both rollouts finished move to *_part1_incomplete;
#      part1 is scored by alpasim's own eval.aggregation (its rollouts already carry online metrics.parquet)
#   2. the 51 Ada-assigned scenes not in part1 run on GPUs 0-3, shard 4+k -> GPU k (scenes_tyr_resume.csv records
#      orig_shard, so analysis can split by card type: these scenes ran on Blackwell, the other arms' on Ada)
#   3. Tyr's Blackwell half (shards 0-3 of scenes_main.csv) via run_u50_origin150.sh SHARDS="0 1 2 3"
#   4. merge sh0-3 + part1 + resume into u50o_merged_slim_tyr_u50_r (150 scenes), gates
#
#   setsid nohup bash iclr27-rebuttal/code/run_u50_tyr_resume.sh > outputs/iclr27-rebuttal/u50_origin150/run_tyr_resume.log 2>&1 &
set -u
REPO=/home/cvlab21/project/chan/alpamayo-model-compression
OUT=$REPO/outputs/iclr27-rebuttal/u50_origin150
RUNS=/home/cvlab21/project/chan/alpasim-runs
ALPASIM=/home/cvlab21/project/chan/alpasim
PY=$REPO/.venv/bin/python
APY=/mnt/nvme1n1/ad_vla/venvs/alpasim-chan/bin/python
LAUNCH=$REPO/experiments/head_analysis/launch_alpasim_shards.sh
MERGE=$REPO/experiments/head_analysis/merge_alpasim_shards.py
CFG=slim_tyr_u50_r
RP=u50oR_
stamp() { echo "$(date -d '+9 hours' '+%F %T') KST $*"; }
mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1"; }
cd "$REPO"

# 1. salvage the stopped Ada shards
for i in 4 5 6 7; do
  d=$RUNS/u50o_${CFG}_sh$i
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
      > "$OUT/tyr_part1_sh${i}_aggregation.log" 2>&1 || { stamp "aggregation of $p failed"; exit 1; }
  fi
  stamp "part1 sh$i: $(ls -d "$p"/rollouts/*/ | wc -l) scenes kept, $($PY -c "import json;print(len(json.load(open('$p/aggregate/results-summary.json'))['rollouts']))") rollouts scored"
done

# 2. the rest of the Ada half on GPUs 0-3
$PY - "$OUT" "$RUNS" "$CFG" << 'EOF'
import sys
from pathlib import Path
import pandas as pd
out, runs, cfg = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
main = pd.read_csv(out / "scenes_main.csv")
done = {p.name for i in (4, 5, 6, 7) for p in (runs / f"u50o_{cfg}_sh{i}_part1" / "rollouts").iterdir()}
rest = main[(main.shard >= 4) & ~main.scene_id.isin(done)].copy()
rest["orig_shard"] = rest.shard
rest["shard"] = rest.shard - 4
rest["card_type"] = "blackwell"
rest.to_csv(out / "scenes_tyr_resume.csv", index=False)
print(f"part1 scenes {len(done)}, resume scenes {len(rest)}: {rest.shard.value_counts().sort_index().to_dict()}")
EOF
N=$(($(wc -l < "$OUT/scenes_tyr_resume.csv") - 1))
if [ ! -f "$OUT/tyr_resume.DONE" ]; then
  empty03() { for g in 0 1 2 3; do [ "$(mib "$g")" -le 10 ] || return 1; done; }
  until empty03; do stamp "waiting for empty GPUs 0-3"; sleep 60; done
  stamp "$CFG: resume $N Ada-assigned scenes x 2 on GPUs 0-3"
  SUITE=public_2601_u50oR PARENT_SUITE=public_2601 PREFIX=$RP DRIVER_OMP_THREADS=8 SCENES_CSV=$OUT/scenes_tyr_resume.csv \
    bash "$LAUNCH" "$CFG" "$N" 2 "0 1 2 3"
  stamp "$CFG: resume shards finished (exit $?)"
  for c in $(docker ps -a --format '{{.Names}}' | grep "^${RP}${CFG}_sh"); do docker rm -f "$c" > /dev/null; done
  for nw in $(docker network ls --format '{{.Name}}' | grep "^${RP}${CFG}_sh"); do
    [ "$(docker network inspect -f '{{len .Containers}}' "$nw")" = 0 ] && docker network rm "$nw" > /dev/null
  done
  ok=1
  for i in 0 1 2 3; do [ -f "$RUNS/${RP}${CFG}_sh$i/aggregate/results-summary.json" ] || ok=0; done
  [ "$ok" = 1 ] && touch "$OUT/tyr_resume.DONE"
fi

# 3. Tyr's own Blackwell half
SHARDS="0 1 2 3" ARMS="$CFG" bash iclr27-rebuttal/code/run_u50_origin150.sh

# 4. one merge over the three parts
shards=""
for i in 0 1 2 3; do shards="$shards u50o_${CFG}_sh$i ${RP}${CFG}_sh$i"; done
for i in 4 5 6 7; do shards="$shards u50o_${CFG}_sh${i}_part1"; done
for s in $shards; do [ -f "$RUNS/$s/aggregate/results-summary.json" ] || { stamp "$s has no summary -- not merging"; exit 1; }; done
merged=u50o_merged_$CFG
mkdir "$OUT/merge_$CFG.lock" 2> /dev/null || { stamp "merge already taken"; exit 0; }
$PY "$MERGE" --runs-root "$RUNS" --shards $shards --out "$merged" --expect-scenes 150
rc=$?
stamp "$CFG: merged -> $RUNS/$merged (exit $rc)"
(cd experiments/evaluation && $APY s600_smoke_gates.py --run "$merged" --logs "u50o_$CFG" \
   --scenes "$OUT/scenes_main.csv" --out "$OUT/gates_$CFG") > "$OUT/gates_$CFG.txt" 2>&1
stamp "$CFG: $(grep -m1 'gates:' "$OUT/gates_$CFG.txt")"
[ "$rc" = 0 ] && touch "$OUT/$CFG.DONE"
