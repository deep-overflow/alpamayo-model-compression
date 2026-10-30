#!/bin/bash
# Assemble and analyse the CE-only (layer-0 keep) closed loop on s600 (plans/2026-10-30_coc-closedloop-s600.md).
#
# The run moved from four cards to eight partway (user request, 2026-10-30), so its 600 scenes live in three places:
#   c600smoke_merged_slim_coc_u40_l0keep_v2      smoke, 4 scenes, GPUs 0-3 (FM-only's scene-to-card map)
#   c600_slim_coc_u40_l0keep_v2_sh0-3            main run on GPUs 0-3, stopped; completed rollouts scored afterwards
#                                                with alpasim-aggregation
#   c600b_slim_coc_u40_l0keep_v2_sh0-7           the rest on GPUs 0-7, each scene on its dense-baseline card
# Each comparison is repeated on the scenes both arms ran on the same card type; the lists are written here from
# the cards actually used (dense / shipped MoOP: outputs/alpasim_dense_s600/scene_gpu.csv, MoOP l0keep:
# outputs/alpasim_l0keep_s600/scene_gpu_l0keep.csv, FM-only: all Blackwell).
#
#   bash experiments/head_analysis/finish_coc_s600.sh      (after outputs/alpasim_cocl0_s600/resume.DONE)
set -eu
REPO=/home/cvlab21/project/chan/alpamayo-model-compression
RUNS=/home/cvlab21/project/chan/alpasim-runs
OUT=$REPO/outputs/alpasim_cocl0_s600
CFG=slim_coc_u40_l0keep_v2
ARM=c600_merged_$CFG
PY=/mnt/nvme1n1/ad_vla/venvs/alpasim-chan/bin/python

"$REPO/.venv/bin/python" "$REPO/experiments/head_analysis/merge_alpasim_shards.py" --runs-root "$RUNS" \
  --shards c600smoke_merged_$CFG $(for i in 0 1 2 3; do echo c600_${CFG}_sh$i; done) \
           $(for i in 0 1 2 3 4 5 6 7; do echo c600b_${CFG}_sh$i; done) \
  --out "$ARM" --expect-scenes 600

"$REPO/.venv/bin/python" - "$RUNS" "$OUT" "$CFG" "$REPO" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

runs, out, cfg, repo = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
# run dir -> GPU index of each rollout's card
gpu = {}
smoke = pd.read_csv(out / "scenes_smoke4.csv").set_index("scene_id")["shard"]
for r in json.loads((runs / f"c600smoke_merged_{cfg}/aggregate/results-summary.json").read_text())["rollouts"]:
    gpu[r["clipgt_id"]] = int(smoke[r["clipgt_id"]])
for prefix, cards in (("c600", range(4)), ("c600b", range(8))):
    for i, g in enumerate(cards):
        for r in json.loads((runs / f"{prefix}_{cfg}_sh{i}/aggregate/results-summary.json").read_text())["rollouts"]:
            gpu[r["clipgt_id"]] = g
m = pd.DataFrame({"scene_id": list(gpu), "gpu": list(gpu.values())})
m["arch"] = m.gpu.map(lambda g: "Blackwell" if g < 4 else "Ada")
assert len(m) == 600, len(m)
m.to_csv(out / "scene_gpu_coc.csv", index=False)
bases = {"dense": pd.read_csv(repo / "outputs/alpasim_dense_s600/scene_gpu.csv"),
         "l0keep": pd.read_csv(repo / "outputs/alpasim_l0keep_s600/scene_gpu_l0keep.csv"),
         "fmonly": m[["scene_id"]].assign(arch="Blackwell")}
for name, b in bases.items():
    j = m.merge(b[["scene_id", "arch"]], on="scene_id", suffixes=("", "_base"))
    assert len(j) == 600, (name, len(j))
    mis = j[j.arch != j.arch_base]
    mis[["scene_id"]].to_csv(out / f"card_mismatch_vs_{name}.csv", index=False)
    print(f"vs {name}: {len(mis)} scenes on a different card type ({mis.arch.value_counts().to_dict()} here)")
PY

cd "$REPO/experiments/evaluation"
A="--arm $ARM --arm-label CE-only_l0keep --no-pre-fix"
for spec in "l600_merged_slim_dual_u40_l0keep_v2 MoOP_l0keep vs_moop_l0keep l0keep" \
            "f600_merged_slim_trajfix_u40_l0keep_v2 FM-only_l0keep vs_fmonly_l0keep fmonly" \
            "s600_merged_baseline dense vs_dense dense" \
            "d600_merged_slim_dual_u40_v2 MoOP vs_moop dense"; do
  set -- $spec
  $PY analyze_dual_s600.py $A --base "$1" --base-label "$2" --out "$OUT/$3"
  $PY analyze_dual_s600.py $A --base "$1" --base-label "$2" --out "$OUT/$3_same_card" \
    --exclude "$OUT/card_mismatch_vs_$4.csv"
done
touch "$OUT/analysis.DONE"
