#!/bin/bash
# Closed loop of the budget-matched LLM-Pruner arm lp_m24 on s600 (cvpr27/plans/2026-10-31_lp-m24-closedloop-s600.md).
# Waits for the layer 1-4 MLP queue (run_l14m_s600.sh) to finish and for all eight cards to read empty, then:
#   build lp_m24 with weights (lp_prune.py, the A100 session's options without --no-state)
#   -> check its units against the selection-only build outputs/cvpr27/slim_lp_m24 -> hardlink into the drivers dir
#   -> smoke (8 scenes) -> gates G1/G2 -> main (592 more) -> paired analyses
# Any failed check stops the script.
#
#   bash cvpr27/code/run_lp_m24_s600.sh > outputs/cvpr27/lp_m24_s600.log 2>&1
set -u
REPO=/home/cvlab21/project/chan/alpamayo-model-compression
PY=/mnt/nvme1n1/ad_vla/venvs/alpasim-chan/bin/python
DRIVERS=/mnt/nvme1n1/ad_vla/data/alpasim/drivers
L0=l600_merged_slim_dual_u40_l0keep_v2
stamp() { echo "$(date -u '+%F %T') UTC $*"; }
cd "$REPO"

n=0
until grep -q "queue finished" outputs/cvpr27/l14m_s600_queue.log \
      && [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 10' | wc -l)" = 0 ]; do
  grep -q -E "FAILED|stopped" outputs/cvpr27/l14m_s600_queue.log && { stamp "l14m queue stopped, not starting"; exit 1; }
  [ $((n % 30)) = 0 ] && stamp "waiting for the l14m queue and eight empty cards"
  n=$((n + 1)); sleep 60
done

stamp "build lp_m24 with weights on GPU 0"
bash experiments/head_analysis/run_retry_host.sh 2 experiments/llm_pruner/lp_prune.py \
  --importance cvpr27/lp_importance_v1 --arm coc_param_first --units qhead:15,mlp:5928 --mlp-extra-layers 24 \
  --expect-removed 2657452032 --out outputs/cvpr27/slim_lp_m24_state --gpu 0 \
  > outputs/cvpr27/llm_pruner_matched/build_lp_m24_state.log 2>&1 || { stamp "build failed"; exit 1; }
.venv/bin/python - <<'PY' || { stamp "unit check failed"; exit 1; }
import json
a = json.load(open("outputs/cvpr27/slim_lp_m24_state/slim_meta.json"))
b = json.load(open("outputs/cvpr27/slim_lp_m24/slim_meta.json"))
same = {k: a.get(k) == b.get(k) for k in ("vlm", "expert", "kvonly_layers", "params")}
print("units vs the selection-only build:", same)
assert all(same.values()) and a["params"]["removed"] == 2657452032
PY
[ -e "$DRIVERS/slim_lp_m24" ] && { stamp "$DRIVERS/slim_lp_m24 already exists"; exit 1; }
mkdir "$DRIVERS/slim_lp_m24" && ln outputs/cvpr27/slim_lp_m24_state/* "$DRIVERS/slim_lp_m24/"
stamp "linked $(ls "$DRIVERS/slim_lp_m24" | tr '\n' ' ')"

CFG=slim_lp_m24 TAG=p24 OUT=$REPO/outputs/cvpr27/alpasim_lp_m24_s600
export CFG TAG OUT
mkdir -p "$OUT"
stamp "$CFG: smoke"
STAGE=smoke bash experiments/head_analysis/run_dual_s600.sh > "$OUT/smoke.log" 2>&1
$PY experiments/evaluation/s600_smoke_gates.py --run ${TAG}smoke_merged_$CFG --logs ${TAG}smoke_$CFG \
  --scenes outputs/alpasim_dual_s600/scenes_smoke8.csv --out "$OUT/smoke" > "$OUT/smoke_gates.txt" 2>&1
grep -q "gates: G1 PASS  G2 PASS" "$OUT/smoke_gates.txt" || { stamp "$CFG: gates FAILED"; exit 1; }
stamp "$CFG: gates passed, main"
STAGE=main bash experiments/head_analysis/run_dual_s600.sh > "$OUT/main.log" 2>&1
[ -e "$OUT/main.DONE" ] || { stamp "$CFG: main did not finish"; exit 1; }

ARM="--arm ${TAG}_merged_$CFG --arm-label lp_m24 --no-pre-fix"
cd experiments/evaluation
$PY analyze_dual_s600.py $ARM --base d600_merged_slim_dual_u40_v2 --base-label MoOP --out "$OUT/vs_moop"
$PY analyze_dual_s600.py $ARM --base $L0 --base-label MoOP_l0keep --out "$OUT/vs_moop_l0keep"
$PY analyze_dual_s600.py $ARM --base $L0 --base-label MoOP_l0keep --out "$OUT/vs_moop_l0keep_same_card" \
  --exclude "$REPO/outputs/alpasim_l0keep_s600/card_mismatch.csv"
$PY analyze_dual_s600.py $ARM --out "$OUT/vs_dense"
cd "$REPO"
touch "$OUT/analysis.DONE"
stamp "$CFG: done"
