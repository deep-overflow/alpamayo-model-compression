#!/bin/bash
# Standard-protocol LingoQA for the Tyr baseline arms, end to end: answer (two cards in
# parallel) -> one Lingo-Judge scoring pass -> paired comparison against dual / baseline /
# the single-criterion arms. Every GPU stage goes through run_retry_host.sh so it queues
# for a free Ada card instead of failing. Arms are slim dirs; load_slim picks up their
# slim_state.pt, so the OSSCAR-reconstructed weights are what answers the questions
# (a --no-state recipe would silently evaluate selection-only -- see make_slim's guard).
#
# Usage: bash experiments/lingoqa/eval_lingo_tyr.sh [gpu_a] [gpu_b] [--with-sensitivity]
#   default arms : slim_tyr_uniform_u40_recon (OSSCAR selection + reconstruction, uniform)
#                  slim_tyr_u40_r             (+ global sparsity-distribution search)
#   sensitivity  : slim_tyr_sel_u40 (selection only), slim_tyr_uniform_u40_d1 / slim_tyr_u40_d1
#                  (weak reconstruction, damp 1.0) -- answered sequentially on gpu_a after
#                  the main arms, scored and analyzed together.
# Outputs: outputs/lingo_vqa_<arm>/{predictions.csv,rows.json,scored.json},
#          outputs/lingo_vqa_scores_tyr/, outputs/lingo_tyr_vs_dual/.
set -u
GPU_A=${1:-4}
GPU_B=${2:-5}
SENS=${3:-}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
RETRY=$REPO/experiments/head_analysis/run_retry_host.sh
PY=$REPO/.venv/bin/python
cd "$REPO"

MAIN_ARMS=(slim_tyr_uniform_u40_recon slim_tyr_u40_r)
SENS_ARMS=(slim_tyr_sel_u40 slim_tyr_uniform_u40_d1 slim_tyr_u40_d1)

# stage 1: answers -- the two main arms side by side
bash "$RETRY" 120 experiments/lingoqa/run_lingo_vqa.py --arm "outputs/${MAIN_ARMS[0]}" --gpu "$GPU_A" \
  > logs/lingo_${MAIN_ARMS[0]}.log 2>&1 &
bash "$RETRY" 120 experiments/lingoqa/run_lingo_vqa.py --arm "outputs/${MAIN_ARMS[1]}" --gpu "$GPU_B" \
  > logs/lingo_${MAIN_ARMS[1]}.log 2>&1 &
wait
RUNS=()
for a in "${MAIN_ARMS[@]}"; do RUNS+=("lingo_vqa_$a"); done

if [ "$SENS" = "--with-sensitivity" ]; then
  for a in "${SENS_ARMS[@]}"; do
    [ -d "outputs/$a" ] || { echo "skip $a (no slim dir)"; continue; }
    bash "$RETRY" 120 experiments/lingoqa/run_lingo_vqa.py --arm "outputs/$a" --gpu "$GPU_A" \
      > logs/lingo_$a.log 2>&1 && RUNS+=("lingo_vqa_$a")
  done
fi

for r in "${RUNS[@]}"; do
  [ -f "outputs/$r/predictions.csv" ] || { echo "missing outputs/$r/predictions.csv"; exit 1; }
done

# stage 2: one judge pass over every new run
bash "$RETRY" 60 experiments/lingoqa/score_lingo_vqa.py --runs "${RUNS[@]}" \
  --exp-id lingo_vqa_scores_tyr --gpu "$GPU_A" > logs/lingo_scores_tyr.log 2>&1 || exit 1

# stage 3: segment-clustered paired comparison, dual as the comparison arm
REF=(lingo_vqa_slim_dual_u40_v2 lingo_vqa_baseline lingo_vqa_slim_traj_u40_v2
     lingo_vqa_slim_coc_u40_v2 lingo_vqa_slim_j_u40_v2)
[ -f outputs/lingo_vqa_slim_wanda_u40_v2/scored.json ] && REF+=(lingo_vqa_slim_wanda_u40_v2)
"$PY" experiments/lingoqa/analyze_lingo.py --runs "${RUNS[@]}" "${REF[@]}" \
  --baseline lingo_vqa_slim_dual_u40_v2 --exp-id lingo_tyr_vs_dual \
  > logs/lingo_tyr_analysis.log 2>&1
tail -30 logs/lingo_tyr_analysis.log
