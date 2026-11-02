#!/bin/bash
# Wanda-sp (FLAP prune_wanda_sp, layer-sequential, all layers, calib_100) end to end: scores -> weighted build ->
# unit check -> s600 closed loop on all eight cards (cvpr27/plans/2026-11-02_wandasp-closedloop-s600.md).
#   1. cvpr27/code/run_wanda_sp.py on GPU 0 -> outputs/cvpr27/wandasp_v1/wanda.npz
#   2. make_slim.py --config wanda_u40_v2 --wanda cvpr27/wandasp_v1 (weights) -> outputs/cvpr27/slim_wandasp_u40
#   3. G0: built units == the units chosen during the sequential pass, 2,657,452,032 removed, expert untouched
#   4. hardlink into the drivers dir as slim_wandasp_u40
#   5. run_dual_s600.sh smoke (8 scenes) -> gates -> main (592) on GPUs 0-7, every scene on its dense-baseline card
#   6. paired analyses
#
#   bash cvpr27/code/run_wandasp_s600.sh > outputs/cvpr27/wandasp_s600.log 2>&1
set -u
REPO=/home/cvlab21/project/chan/alpamayo-model-compression
PY=/mnt/nvme1n1/ad_vla/venvs/alpasim-chan/bin/python
DRIVERS=/mnt/nvme1n1/ad_vla/data/alpasim/drivers
L0=l600_merged_slim_dual_u40_l0keep_v2
stamp() { echo "$(date -u '+%F %T') UTC $*"; }
cd "$REPO"

stamp "Wanda-sp scores (sequential, 100 clips) on GPU 0"
bash experiments/head_analysis/run_retry_host.sh 2 cvpr27/code/run_wanda_sp.py --gpu 0 --num-clips 100 \
  --exp-id cvpr27/wandasp_v1 > outputs/cvpr27/wandasp_v1.log 2>&1 || { stamp "scores failed"; exit 1; }
stamp "build with weights"
bash experiments/head_analysis/run_retry_host.sh 2 experiments/head_analysis/make_slim.py --config wanda_u40_v2 \
  --wanda cvpr27/wandasp_v1 --importance importance_v2 --out outputs/cvpr27/slim_wandasp_u40 --gpu 0 \
  > outputs/cvpr27/slim_wandasp_u40.build.log 2>&1 || { stamp "build failed"; exit 1; }
.venv/bin/python - <<'PY' || { stamp "G0 failed"; exit 1; }
import json

import numpy as np

m = json.load(open("outputs/cvpr27/slim_wandasp_u40/slim_meta.json"))
ref = json.load(open("outputs/slim_dual_u40_v2/slim_meta.json"))
z = np.load("outputs/cvpr27/wandasp_v1/wanda.npz")
same_q = all(sorted(m["vlm"][i]["q"]) == np.flatnonzero(z["keep_q"][i]).tolist() for i in range(36))
same_m = all(sorted(m["vlm"][i]["mlp"]) == np.flatnonzero(z["keep_mlp"][i]).tolist() for i in range(36))
g0 = {"units == sequential pass (Q)": same_q, "units == sequential pass (MLP)": same_m,
      "removed == 2,657,452,032": m["params"]["removed"] == 2657452032,
      "expert untouched (== MoOP's expert meta)": m["expert"] == ref["expert"],
      "layer 0 cut 13 Q heads": len(m["vlm"][0]["q"]) == 19}
print("G0:", g0)
json.dump(g0, open("outputs/cvpr27/wandasp_v1/g0_build.json", "w"), indent=2)
assert all(g0.values())
PY
[ -e "$DRIVERS/slim_wandasp_u40" ] && { stamp "driver dir exists"; exit 1; }
mkdir "$DRIVERS/slim_wandasp_u40" && ln outputs/cvpr27/slim_wandasp_u40/* "$DRIVERS/slim_wandasp_u40/"
stamp "G0 passed, driver linked"

CFG=slim_wandasp_u40 TAG=wsp OUT=$REPO/outputs/cvpr27/alpasim_wandasp_s600
export CFG TAG OUT
mkdir -p "$OUT"
stamp "$CFG: smoke on GPUs 0-7"
STAGE=smoke bash experiments/head_analysis/run_dual_s600.sh > "$OUT/smoke.log" 2>&1
$PY experiments/evaluation/s600_smoke_gates.py --run ${TAG}smoke_merged_$CFG --logs ${TAG}smoke_$CFG \
  --scenes outputs/alpasim_dual_s600/scenes_smoke8.csv --out "$OUT/smoke" > "$OUT/smoke_gates.txt" 2>&1
grep -q "gates: G1 PASS  G2 PASS" "$OUT/smoke_gates.txt" || { stamp "$CFG: gates FAILED"; exit 1; }
stamp "$CFG: gates passed, main"
STAGE=main bash experiments/head_analysis/run_dual_s600.sh > "$OUT/main.log" 2>&1
[ -e "$OUT/main.DONE" ] || { stamp "$CFG: main did not finish"; exit 1; }

A="--arm ${TAG}_merged_$CFG --arm-label wanda_sp --no-pre-fix"
cd experiments/evaluation
$PY analyze_dual_s600.py $A --base d600_merged_slim_dual_u40_v2 --base-label MoOP --out "$OUT/vs_moop"
$PY analyze_dual_s600.py $A --base $L0 --base-label MoOP_l0keep --out "$OUT/vs_moop_l0keep"
$PY analyze_dual_s600.py $A --base $L0 --base-label MoOP_l0keep --out "$OUT/vs_moop_l0keep_same_card" \
  --exclude "$REPO/outputs/alpasim_l0keep_s600/card_mismatch.csv"
$PY analyze_dual_s600.py $A --out "$OUT/vs_dense"
$PY analyze_dual_s600.py $A --base p24_merged_slim_lp_m24 --base-label lp_m24 --out "$OUT/vs_lp_m24"
$PY analyze_dual_s600.py $A --base tyo_merged_slim_tyr_off --base-label tyr_off --out "$OUT/vs_tyr_off"
$PY analyze_dual_s600.py $A --base tyo_merged_slim_tyr_off --base-label tyr_off --out "$OUT/vs_tyr_off_same_card" \
  --exclude "$REPO/outputs/cvpr27/alpasim_tyr_off_s600/card_mismatch_vs_dense.csv"
cd "$REPO"
touch "$OUT/analysis.DONE"
stamp "$CFG: done"
