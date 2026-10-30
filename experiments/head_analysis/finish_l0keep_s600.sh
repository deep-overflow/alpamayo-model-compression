#!/bin/bash
# Assemble and analyse the layer-0-keeping MoOP closed loop on s600 (plans/2026-10-29_l0keep-closedloop-s600.md).
#
# The run was split by GPU hand-backs, so its 600 scenes live in five places:
#   l600smoke_merged_slim_dual_u40_l0keep_v2           smoke, 8 scenes, one per card (pinned)
#   l600_slim_dual_u40_l0keep_v2_sh0-3                 main run, Blackwell 0-3, complete (pinned)
#   l600_slim_dual_u40_l0keep_v2_sh4-7                 main run, Ada 4-7, stopped early; completed rollouts scored
#                                                      afterwards with alpasim-aggregation
#   l600r_slim_dual_u40_l0keep_v2_sh0-1                resume on Ada 4-5, stopped; scored the same way
#   l600b_slim_dual_u40_l0keep_v2_sh0-5                the rest on GPUs 0-5 (0-3 Blackwell, 4-5 Ada)
# Every scene the dense baseline ran on Ada but this arm ran on Blackwell is listed in card_mismatch.csv, so the
# paired comparisons can be repeated without them.
#
#   bash experiments/head_analysis/finish_l0keep_s600.sh
set -eu
REPO=/home/cvlab21/project/chan/alpamayo-model-compression
RUNS=/home/cvlab21/project/chan/alpasim-runs
OUT=$REPO/outputs/alpasim_l0keep_s600
CFG=slim_dual_u40_l0keep_v2
ARM=l600_merged_$CFG
PY=/mnt/nvme1n1/ad_vla/venvs/alpasim-chan/bin/python

rm -rf "${RUNS:?}/$ARM"   # the 364-scene merge the main stage left behind
"$REPO/.venv/bin/python" "$REPO/experiments/head_analysis/merge_alpasim_shards.py" --runs-root "$RUNS" \
  --shards l600smoke_merged_$CFG $(for i in 0 1 2 3 4 5 6 7; do echo l600_${CFG}_sh$i; done) \
           l600r_${CFG}_sh0 l600r_${CFG}_sh1 $(for i in 0 1 2 3 4 5; do echo l600b_${CFG}_sh$i; done) \
  --out "$ARM" --expect-scenes 600

"$REPO/.venv/bin/python" - "$RUNS" "$OUT" "$CFG" "$REPO" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

runs, out, cfg, repo = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
# run dir -> GPU index of each rollout's card
gpu = {}
smoke = pd.read_csv(repo / "outputs/alpasim_dual_s600/scenes_smoke8.csv").set_index("scene_id")["shard"]
for r in json.loads((runs / f"l600smoke_merged_{cfg}/aggregate/results-summary.json").read_text())["rollouts"]:
    gpu[r["clipgt_id"]] = int(smoke[r["clipgt_id"]])
for prefix, cards in (("l600", range(8)), ("l600r", (4, 5)), ("l600b", range(6))):
    for i, g in enumerate(cards):
        for r in json.loads((runs / f"{prefix}_{cfg}_sh{i}/aggregate/results-summary.json").read_text())["rollouts"]:
            gpu[r["clipgt_id"]] = g
m = pd.DataFrame({"scene_id": list(gpu), "gpu": list(gpu.values())})
m["arch"] = m.gpu.map(lambda g: "Blackwell" if g < 4 else "Ada")
base = pd.read_csv(repo / "outputs/alpasim_dense_s600/scene_gpu.csv")[["scene_id", "arch"]]
m = m.merge(base, on="scene_id", suffixes=("", "_base"))
assert len(m) == 600, len(m)
m.to_csv(out / "scene_gpu_l0keep.csv", index=False)
mis = m[m.arch != m.arch_base]
mis[["scene_id"]].to_csv(out / "card_mismatch.csv", index=False)
print(f"cards: {len(m)} scenes, {len(mis)} on a different card type than the baseline "
      f"({mis.arch.value_counts().to_dict()} here vs {mis.arch_base.value_counts().to_dict()} in the baseline)")
PY

cd "$REPO/experiments/evaluation"
A="--arm $ARM --arm-label MoOP_l0keep --no-pre-fix"
$PY analyze_dual_s600.py $A --out "$OUT"
$PY analyze_dual_s600.py $A --out "$OUT/same_card" --exclude "$OUT/card_mismatch.csv"
$PY analyze_dual_s600.py $A --base d600_merged_slim_dual_u40_v2 --base-label MoOP --out "$OUT/vs_moop"
$PY analyze_dual_s600.py $A --base d600_merged_slim_dual_u40_v2 --base-label MoOP --out "$OUT/vs_moop_same_card" \
  --exclude "$OUT/card_mismatch.csv"
touch "$OUT/analysis.DONE"
