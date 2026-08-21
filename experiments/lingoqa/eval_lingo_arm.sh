#!/bin/bash
# Standard-protocol LingoQA for one slim arm, end to end: answer -> judge -> paired
# comparison against dual / baseline / the single-criterion arms. Every stage goes
# through run_retry_host.sh so it queues for a free Ada card instead of failing.
#
# Usage: bash experiments/lingoqa/eval_lingo_arm.sh <slim_dir_name> [gpu] [analysis_exp_id]
#   e.g. bash experiments/lingoqa/eval_lingo_arm.sh slim_wanda_u40_v2 5 lingo_wanda_vs_dual
set -u
ARM=${1:?usage: eval_lingo_arm.sh <slim_dir_name> [gpu] [analysis_exp_id]}
GPU=${2:-5}
ANALYSIS=${3:-lingo_${ARM#slim_}_vs_dual}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
RETRY=$REPO/experiments/head_analysis/run_retry_host.sh
RUN=lingo_vqa_$ARM

bash "$RETRY" 120 experiments/lingoqa/run_lingo_vqa.py --arm "outputs/$ARM" --gpu "$GPU" \
  && bash "$RETRY" 60 experiments/lingoqa/score_lingo_vqa.py --runs "$RUN" \
       --exp-id "lingo_vqa_scores_${ARM#slim_}" --gpu "$GPU" \
  && "$REPO/.venv/bin/python" experiments/lingoqa/analyze_lingo.py \
       --runs "$RUN" lingo_vqa_slim_dual_u40_v2 lingo_vqa_baseline \
              lingo_vqa_slim_traj_u40_v2 lingo_vqa_slim_coc_u40_v2 lingo_vqa_slim_j_u40_v2 \
       --baseline lingo_vqa_slim_dual_u40_v2 --exp-id "$ANALYSIS"
