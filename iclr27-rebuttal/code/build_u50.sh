#!/bin/bash
# Builds for the u50 closed loop (iclr27-rebuttal/plans/2026-11-09_u50-closedloop-origin150.md, step 0) on Blackwell
# 0-3 (user, 11-10; another lab queue takes card pairs there, so the cards are env-selectable). Paper (ICLR) recipes:
#   SUP_GPU (2)  Tyr port supernet tyr_supernet_u50r = run_tyr_supernet.py exactly as for tyr_u40_r (damp 1e-2,
#                OSSCAR, importance_v2, calib_100, 9 levels) with the u50 cut: 16/32 Q heads, 6144/12288 MLP ch per layer
#   ONE_GPU (3)  in turn: MoOP u50 = make_slim dual_u50_v2 (with state); LLM-Pruner u50 = lp_prune.py coc_param_first
#                --ratio 0.6 (the paper lp_r50 rule, layers 4-33); G0 rebuilds dual_u40_v2 and lp_r50 (--no-state)
#   then         Tyr search tyr_search_u50r (20 gens x 32 offspring, teacher tyr_teacher_u40, as u40) on every empty card
#                of 0-3 -- --workers only spreads the same minibatches over cards (run_tyr_search.py), u40 used 3 --
#                then Tyr assembly make_slim tyr_u50_r, gates (check_u50_builds.py) -> build.DONE
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

SUP_GPU=${SUP_GPU:-2}
ONE_GPU=${ONE_GPU:-3}
mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1"; }
for g in $SUP_GPU $ONE_GPU; do
  [ "$(mib "$g")" -gt 10 ] && { stamp "GPU $g holds $(mib "$g") MiB -- not starting"; exit 1; }
done
stamp "builds: Tyr supernet on GPU $SUP_GPU, MoOP / LLM-Pruner / G0 on GPU $ONE_GPU"

( [ -f outputs/tyr_supernet_u50r/metadata.json ] || $PY $H/run_tyr_supernet.py --exp-id tyr_supernet_u50r \
    --head-cut 16 --mlp-cut 6144 --gpu "$SUP_GPU" ) > "$OUT/build_tyr_supernet.log" 2>&1 &
p_sup=$!
(
  [ -f outputs/slim_dual_u50_v2/slim_state.pt ] || $PY $H/make_slim.py --config dual_u50_v2 \
    --out outputs/slim_dual_u50_v2 --gpu "$ONE_GPU"
  echo "== exit $? (MoOP u50)"
  [ -f outputs/slim_lp_r60/slim_state.pt ] || $PY experiments/llm_pruner/lp_prune.py \
    --importance cvpr27/lp_importance_v1 --arm coc_param_first --ratio 0.6 --out outputs/slim_lp_r60 --gpu "$ONE_GPU"
  echo "== exit $? (LLM-Pruner u50)"
  [ -f "$OUT/g0/dual_u40_v2/slim_meta.json" ] || $PY $H/make_slim.py --config dual_u40_v2 \
    --out outputs/iclr27-rebuttal/u50_origin150/g0/dual_u40_v2 --no-state --gpu "$ONE_GPU"
  echo "== exit $? (G0 dual_u40_v2)"
  [ -f "$OUT/g0/lp_r50/slim_meta.json" ] || $PY experiments/llm_pruner/lp_prune.py \
    --importance cvpr27/lp_importance_v1 --arm coc_param_first --ratio 0.5 --no-state \
    --out outputs/iclr27-rebuttal/u50_origin150/g0/lp_r50 --gpu "$ONE_GPU"
  echo "== exit $? (G0 lp_r50)"
) > "$OUT/build_moop_lp.log" 2>&1 &
p_one=$!

wait $p_one; stamp "MoOP u50, LLM-Pruner u50, G0 rebuilds: $(grep '^== exit' "$OUT/build_moop_lp.log" | tr '\n' ' ')"
wait $p_sup; rc=$?; stamp "Tyr supernet: exit $rc"
[ "$rc" = 0 ] || exit 1

if [ ! -f outputs/tyr_search_u50r/final_config.json ]; then
  gpus=""
  for g in 0 1 2 3; do [ "$(mib "$g")" -le 10 ] && gpus="$gpus,$g"; done
  gpus=${gpus#,}
  [ -n "$gpus" ] || { stamp "no empty card on 0-3 for the Tyr search"; exit 1; }
  n=$(echo "$gpus" | tr ',' '\n' | wc -l)
  stamp "Tyr search on GPUs $gpus ($n workers)"
  $PY $H/run_tyr_search.py --supernet tyr_supernet_u50r --teacher-id tyr_teacher_u40 --exp-id tyr_search_u50r \
    --workers "$n" --gpu "$gpus" > "$OUT/build_tyr_search.log" 2>&1 || { stamp "Tyr search failed"; exit 1; }
fi
stamp "Tyr search done: $(head -1 outputs/tyr_search_u50r/summary.txt 2>/dev/null)"
if [ ! -f outputs/slim_tyr_u50_r/slim_state.pt ]; then
  $PY $H/make_slim.py --config tyr_u50_r --tyr-supernet tyr_supernet_u50r \
    --tyr-config tyr_search_u50r/final_config.json --out outputs/slim_tyr_u50_r --gpu "$SUP_GPU" \
    > "$OUT/build_tyr_assemble.log" 2>&1 || { stamp "Tyr assembly failed"; exit 1; }
fi
$PY iclr27-rebuttal/code/check_u50_builds.py > "$OUT/gates.txt" 2>&1
stamp "$(tail -1 "$OUT/gates.txt")"
grep -q "gates: PASS" "$OUT/gates.txt" && touch "$OUT/build.DONE"
