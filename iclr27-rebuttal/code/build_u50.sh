#!/bin/bash
# Builds for the u50 closed loop (iclr27-rebuttal/plans/2026-11-09_u50-closedloop-origin150.md, step 0) on Blackwell
# 0-3 (user, 11-10). Paper (ICLR) recipes at u50:
#   GPU 0  Tyr port supernet tyr_supernet_u50r = run_tyr_supernet.py exactly as for tyr_u40_r (damp 1e-2, OSSCAR,
#          importance_v2, calib_100, 9 levels) with the u50 cut: 16/32 Q heads and 6144/12288 MLP channels per layer
#   GPU 1  MoOP u50 = make_slim dual_u50_v2 (with state); then G0: dual_u40_v2 rebuilt --no-state
#   GPU 2  LLM-Pruner u50 = lp_prune.py coc_param_first --ratio 0.6 (the paper lp_r50 rule, layers 4-33); then G0:
#          lp_r50 rebuilt --no-state
#   then   Tyr search tyr_search_u50r on GPUs 0-2 (3 workers, 20 gens x 32 offspring, teacher tyr_teacher_u40, as u40),
#          Tyr assembly make_slim tyr_u50_r on GPU 0, gates (check_u50_builds.py) -> build.DONE
# Steps whose output exists are skipped, so a rerun resumes.
#
#   setsid nohup bash iclr27-rebuttal/code/build_u50.sh > outputs/iclr27-rebuttal/u50_origin150/build.log 2>&1 &
set -u
REPO=/home/cvlab21/project/chan/alpamayo-model-compression
OUT=$REPO/outputs/iclr27-rebuttal/u50_origin150
PY=$REPO/.venv/bin/python
H=$REPO/experiments/head_analysis
stamp() { echo "$(date -d '+9 hours' '+%F %T') KST $*"; }
cd "$REPO" && mkdir -p "$OUT/g0"
source /home/cvlab21/project/chan/alpasim-runs/.hf_env   # valid token + HF_HOME; full_right is invalid since 11-09
export HF_HUB_CACHE=/mnt/nvme1n1/ad_vla/cache/hub CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_ENABLE_HF_TRANSFER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for g in 0 1 2; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
  [ "$u" -gt 10 ] && { stamp "GPU $g holds $u MiB -- not starting"; exit 1; }
done
stamp "builds on GPUs 0-2"

( [ -f outputs/tyr_supernet_u50r/metadata.json ] || $PY $H/run_tyr_supernet.py --exp-id tyr_supernet_u50r \
    --head-cut 16 --mlp-cut 6144 --gpu 0 ) > "$OUT/build_tyr_supernet.log" 2>&1 &
p_sup=$!
( [ -f outputs/slim_dual_u50_v2/slim_state.pt ] || $PY $H/make_slim.py --config dual_u50_v2 \
    --out outputs/slim_dual_u50_v2 --gpu 1
  [ -f "$OUT/g0/dual_u40_v2/slim_meta.json" ] || $PY $H/make_slim.py --config dual_u40_v2 \
    --out outputs/iclr27-rebuttal/u50_origin150/g0/dual_u40_v2 --no-state --gpu 1 ) > "$OUT/build_moop.log" 2>&1 &
p_moop=$!
( [ -f outputs/slim_lp_r60/slim_state.pt ] || $PY experiments/llm_pruner/lp_prune.py \
    --importance cvpr27/lp_importance_v1 --arm coc_param_first --ratio 0.6 --out outputs/slim_lp_r60 --gpu 2
  [ -f "$OUT/g0/lp_r50/slim_meta.json" ] || $PY experiments/llm_pruner/lp_prune.py \
    --importance cvpr27/lp_importance_v1 --arm coc_param_first --ratio 0.5 --no-state \
    --out outputs/iclr27-rebuttal/u50_origin150/g0/lp_r50 --gpu 2 ) > "$OUT/build_lp.log" 2>&1 &
p_lp=$!

wait $p_moop; stamp "MoOP u50 + G0 rebuild: exit $?"
wait $p_lp; stamp "LLM-Pruner u50 + G0 rebuild: exit $?"
wait $p_sup; rc=$?; stamp "Tyr supernet: exit $rc"
[ "$rc" = 0 ] || exit 1

if [ ! -f outputs/tyr_search_u50r/final_config.json ]; then
  $PY $H/run_tyr_search.py --supernet tyr_supernet_u50r --teacher-id tyr_teacher_u40 --exp-id tyr_search_u50r \
    --workers 3 --gpu 0,1,2 > "$OUT/build_tyr_search.log" 2>&1 || { stamp "Tyr search failed"; exit 1; }
fi
stamp "Tyr search done: $(head -1 outputs/tyr_search_u50r/summary.txt 2>/dev/null)"
if [ ! -f outputs/slim_tyr_u50_r/slim_state.pt ]; then
  $PY $H/make_slim.py --config tyr_u50_r --tyr-supernet tyr_supernet_u50r \
    --tyr-config tyr_search_u50r/final_config.json --out outputs/slim_tyr_u50_r --gpu 0 \
    > "$OUT/build_tyr_assemble.log" 2>&1 || { stamp "Tyr assembly failed"; exit 1; }
fi
$PY iclr27-rebuttal/code/check_u50_builds.py > "$OUT/gates.txt" 2>&1
stamp "$(tail -1 "$OUT/gates.txt")"
grep -q "gates: PASS" "$OUT/gates.txt" && touch "$OUT/build.DONE"
