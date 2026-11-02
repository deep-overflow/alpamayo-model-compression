#!/bin/bash
# Move the tyr_off s600 closed loop from GPUs 4-7 to all eight cards once the LLM-Pruner Blackwell shards are done
# (user, 2026-11-01; cvpr27/plans/2026-11-01_tyr-off-closedloop-s600.md §5):
#   wait for the four LP Blackwell launchers to exit and GPUs 0-3 to read empty
#   -> stop run_tyr_off_s600.sh (4-card main) and remove its containers
#   -> score each stopped shard's finished rollouts with alpasim-aggregation
#   -> run the rest on GPUs 0-7, every scene on its dense-baseline card (prefix tyb_)
#   -> merge smoke + tyo sh0-3 + tyb sh0-7 into tyo_merged_slim_tyr_off (600), card lists from the cards actually used,
#      paired analyses
#
#   bash cvpr27/code/resume_tyr_off_8gpu.sh > outputs/cvpr27/alpasim_tyr_off_s600/resume.log 2>&1
set -u
REPO=/home/cvlab21/project/chan/alpamayo-model-compression
RUNS=/home/cvlab21/project/chan/alpasim-runs
PY=/mnt/nvme1n1/ad_vla/venvs/alpasim-chan/bin/python
AGG=/home/cvlab21/project/chan/alpasim/.venv/bin/alpasim-aggregation
OUT=$REPO/outputs/cvpr27/alpasim_tyr_off_s600
CFG=slim_tyr_off
L0=l600_merged_slim_dual_u40_l0keep_v2
stamp() { echo "$(date -u '+%F %T') UTC $*"; }
empty() {
  for g in "$@"; do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")" -gt 10 ] && return 1; done
  return 0
}
cleanup() {
  for c in $(docker ps -a --format '{{.Names}}' | grep "^$1"); do docker rm -f "$c" > /dev/null; done
  for n in $(docker network ls --format '{{.Name}}' | grep "^$1"); do
    [ "$(docker network inspect -f '{{len .Containers}}' "$n")" = 0 ] && docker network rm "$n" > /dev/null
  done
}
cd "$REPO"

until ! pgrep -f "launch_alpasim_shards.sh (slim_lp_m24|lp_r50) 150" > /dev/null; do sleep 120; done
stamp "LLM-Pruner Blackwell shards done"
until empty 0 1 2 3; do sleep 60; done
[ -e "$OUT/main.DONE" ] && { stamp "4-card run already finished, nothing to move"; exit 0; }

stamp "stopping the 4-card tyr_off run"
P=$(pgrep -f "^bash cvpr27/code/run_tyr_off_s600.sh" | head -1)
[ -n "$P" ] && kill -TERM -"$(ps -o pgid= -p "$P" | tr -d ' ')"
sleep 10
cleanup tyo_
stamp "left: $(docker ps -a --format '{{.Names}}' | grep -c '^tyo_') containers"

for i in 0 1 2 3; do
  d=tyo_${CFG}_sh$i
  timeout 900 "$AGG" --array_job_dir "$RUNS/$d" --config_path "$RUNS/$d/eval-config.yaml" > "$RUNS/$d.salvage.log" 2>&1
  stamp "$d salvaged (exit $?): $(python3 -c "import json;print(len(json.load(open('$RUNS/$d/aggregate/results-summary.json'))['rollouts']))" 2>&1 | tail -1) rollouts"
done

"$REPO/.venv/bin/python" - "$RUNS" "$OUT" "$CFG" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

runs, out, cfg = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
done = set()
for d in [f"tyosmoke_merged_{cfg}"] + [f"tyo_{cfg}_sh{i}" for i in range(4)]:
    done |= {r["clipgt_id"] for r in json.loads((runs / d / "aggregate/results-summary.json").read_text())["rollouts"]}
t = pd.read_csv(out / "scenes_t600.csv").drop(columns="shard")
base = pd.read_csv(out.parents[1] / "alpasim_dense_s600/scene_gpu.csv")[["scene_id", "gpu"]]
rem = t[~t.scene_id.isin(done)].merge(base, on="scene_id").rename(columns={"gpu": "shard"})   # GPU list 0..7
rem.to_csv(out / "scenes_resume_8gpu.csv", index=False)
print(f"done {len(done)}, remaining {len(rem)}, per card {rem.shard.value_counts().sort_index().to_dict()}")
PY

until empty 0 1 2 3 4 5 6 7; do sleep 60; done
n=$(($(wc -l < "$OUT/scenes_resume_8gpu.csv") - 1))
stamp "tyb: $n scenes on GPUs 0-7"
SUITE=public_2601_tyb PARENT_SUITE=public_2601 PREFIX=tyb_ DRIVER_OMP_THREADS=8 SCENES_CSV=$OUT/scenes_resume_8gpu.csv \
  bash experiments/head_analysis/launch_alpasim_shards.sh $CFG "$n" 1 "0 1 2 3 4 5 6 7" > "$OUT/tyb.log" 2>&1
cleanup tyb_
stamp "tyb finished; containers left $(docker ps -a --format '{{.Names}}' | grep -c '^tyb_')"

rm -rf "${RUNS:?}/tyo_merged_$CFG"
"$REPO/.venv/bin/python" experiments/head_analysis/merge_alpasim_shards.py --runs-root "$RUNS" \
  --shards tyosmoke_merged_$CFG $(for i in 0 1 2 3; do echo tyo_${CFG}_sh$i; done) \
           $(for i in 0 1 2 3 4 5 6 7; do echo tyb_${CFG}_sh$i; done) \
  --out tyo_merged_$CFG --expect-scenes 600 > "$OUT/merge.log" 2>&1 || { stamp "merge failed"; exit 1; }
stamp "merged 600"

"$REPO/.venv/bin/python" - "$RUNS" "$OUT" "$CFG" "$REPO" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

runs, out, cfg, repo = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
gpu = {}
smoke = pd.read_csv(out / "scenes_smoke4.csv").set_index("scene_id")["shard"]
for r in json.loads((runs / f"tyosmoke_merged_{cfg}/aggregate/results-summary.json").read_text())["rollouts"]:
    gpu[r["clipgt_id"]] = 4 + int(smoke[r["clipgt_id"]])
for prefix, cards in (("tyo", (4, 5, 6, 7)), ("tyb", range(8))):
    for i, g in enumerate(cards):
        for r in json.loads((runs / f"{prefix}_{cfg}_sh{i}/aggregate/results-summary.json").read_text())["rollouts"]:
            gpu[r["clipgt_id"]] = g
m = pd.DataFrame({"scene_id": list(gpu), "gpu": list(gpu.values())})
m["arch"] = m.gpu.map(lambda g: "Blackwell" if g < 4 else "Ada")
assert len(m) == 600, len(m)
m.to_csv(out / "scene_gpu_tyr_off.csv", index=False)
for name, f in (("dense", "outputs/alpasim_dense_s600/scene_gpu.csv"),
                ("l0keep", "outputs/alpasim_l0keep_s600/scene_gpu_l0keep.csv")):
    j = m.merge(pd.read_csv(repo / f)[["scene_id", "arch"]], on="scene_id", suffixes=("", "_base"))
    mis = j[j.arch != j.arch_base]
    mis[["scene_id"]].to_csv(out / f"card_mismatch_vs_{name}.csv", index=False)
    print(f"vs {name}: {len(mis)} scenes on a different card type")
PY

A="--arm tyo_merged_$CFG --arm-label tyr_off --no-pre-fix"
cd experiments/evaluation
for spec in "d600_merged_slim_dual_u40_v2 MoOP vs_moop dense" "$L0 MoOP_l0keep vs_moop_l0keep l0keep" \
            "s600_merged_baseline dense vs_dense dense" "p24_merged_slim_lp_m24 lp_m24 vs_lp_m24 dense"; do
  set -- $spec
  [ "$2" = lp_m24 ] && [ ! -e "$REPO/outputs/cvpr27/alpasim_lp_m24_s600/analysis.DONE" ] && continue
  $PY analyze_dual_s600.py $A --base "$1" --base-label "$2" --out "$OUT/$3"
  $PY analyze_dual_s600.py $A --base "$1" --base-label "$2" --out "$OUT/$3_same_card" --exclude "$OUT/card_mismatch_vs_$4.csv"
done
cd "$REPO"
touch "$OUT/analysis.DONE"
stamp "tyr_off: done"
