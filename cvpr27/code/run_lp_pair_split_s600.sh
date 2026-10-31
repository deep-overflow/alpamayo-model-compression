#!/bin/bash
# LLM-Pruner pair on s600 while GPUs 0-3 are taken (user, 2026-10-31): same arms, cards and scene-to-card-type map as
# run_lp_pair_s600.sh, but each arm's baseline-Blackwell half starts only once its Blackwell cards read empty
# (cvpr27/plans/2026-10-31_lp-closedloop-s600.md §5).
#   lp_m24  driver slim_lp_m24   Ada 4 5 now   Blackwell 0 1 later
#   lp_r50  driver lp_r50        Ada 6 7 now   Blackwell 2 3 later
# Per arm: Ada smoke (4 scenes) -> gates G1/G2 -> Ada main (296)  |  Blackwell main (300, waits for empty cards)
#   -> merge 600 -> paired analyses. A failed gate stops that arm only. Both done -> lp_r50 vs lp_m24.
#
#   bash cvpr27/code/run_lp_pair_split_s600.sh > outputs/cvpr27/lp_s600.log 2>&1
set -u
REPO=/home/cvlab21/project/chan/alpamayo-model-compression
RUNS=/home/cvlab21/project/chan/alpasim-runs
PY=/mnt/nvme1n1/ad_vla/venvs/alpasim-chan/bin/python
LAUNCH=$REPO/experiments/head_analysis/launch_alpasim_shards.sh
MERGE=$REPO/experiments/head_analysis/merge_alpasim_shards.py
S=$REPO/outputs/cvpr27/lp_s600_scenes
L0=l600_merged_slim_dual_u40_l0keep_v2
stamp() { echo "$(date -u '+%F %T') UTC $*"; }
cd "$REPO"

empty() {   # every listed card reads empty
  for g in "$@"; do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")" -gt 10 ] && return 1; done
  return 0
}

cleanup() {   # this prefix's containers and compose networks (user rule, 2026-10-19)
  for c in $(docker ps -a --format '{{.Names}}' | grep "^$1"); do docker rm -f "$c" > /dev/null; done
  for n in $(docker network ls --format '{{.Name}}' | grep "^$1"); do
    [ "$(docker network inspect -f '{{len .Containers}}' "$n")" = 0 ] && docker network rm "$n" > /dev/null
  done
}

launch() {   # <prefix> <driver folder> <scene csv> "<gpus>"
  SUITE=public_2601_${1%_} PARENT_SUITE=public_2601 PREFIX=$1 DRIVER_OMP_THREADS=8 SCENES_CSV=$3 \
    bash "$LAUNCH" "$2" "$(($(wc -l < "$3") - 1))" 1 "$4" > "$OUT/${1%_}.log" 2>&1
  cleanup "$1"
}

arm() {   # <driver folder> <tag> <label> "<ada gpus>" "<blackwell gpus>"
  CFG=$1 TAG=$2 OUT=$REPO/outputs/cvpr27/alpasim_$3_s600
  mkdir -p "$OUT"
  empty $4 || { stamp "$3: Ada cards $4 not empty"; return 1; }
  stamp "$3: smoke on Ada $4"
  launch ${TAG}smoke_ "$CFG" "$S/smoke_ada4.csv" "$4"
  "$REPO/.venv/bin/python" "$MERGE" --runs-root "$RUNS" --shards ${TAG}smoke_${CFG}_sh0 ${TAG}smoke_${CFG}_sh1 \
    --out ${TAG}smoke_merged_$CFG --expect-scenes 4 > "$OUT/smoke_merge.log" 2>&1
  $PY experiments/evaluation/s600_smoke_gates.py --run ${TAG}smoke_merged_$CFG --logs ${TAG}smoke_$CFG \
    --scenes "$S/smoke_ada4.csv" --out "$OUT/smoke" > "$OUT/smoke_gates.txt" 2>&1
  grep -q "gates: G1 PASS  G2 PASS" "$OUT/smoke_gates.txt" || { stamp "$3: gates FAILED"; return 1; }
  stamp "$3: gates passed; Ada main on $4, Blackwell main waits for $5"
  ( launch ${TAG}a_ "$CFG" "$S/main_ada296.csv" "$4"; stamp "$3: Ada half finished" ) &
  ( until empty $5; do sleep 60; done
    stamp "$3: Blackwell main on $5"
    launch ${TAG}b_ "$CFG" "$S/main_bw300.csv" "$5"; stamp "$3: Blackwell half finished" ) &
  wait
  "$REPO/.venv/bin/python" "$MERGE" --runs-root "$RUNS" --shards ${TAG}smoke_merged_$CFG \
    ${TAG}a_${CFG}_sh0 ${TAG}a_${CFG}_sh1 ${TAG}b_${CFG}_sh0 ${TAG}b_${CFG}_sh1 \
    --out ${TAG}_merged_$CFG --expect-scenes 600 > "$OUT/merge.log" 2>&1 || { stamp "$3: merge failed"; return 1; }

  A="--arm ${TAG}_merged_$CFG --arm-label $3 --no-pre-fix"
  cd experiments/evaluation
  $PY analyze_dual_s600.py $A --base d600_merged_slim_dual_u40_v2 --base-label MoOP --out "$OUT/vs_moop"
  $PY analyze_dual_s600.py $A --base $L0 --base-label MoOP_l0keep --out "$OUT/vs_moop_l0keep"
  $PY analyze_dual_s600.py $A --base $L0 --base-label MoOP_l0keep --out "$OUT/vs_moop_l0keep_same_card" \
    --exclude "$REPO/outputs/alpasim_l0keep_s600/card_mismatch.csv"
  $PY analyze_dual_s600.py $A --out "$OUT/vs_dense"
  cd "$REPO"
  touch "$OUT/analysis.DONE"
  stamp "$3: done"
}

arm slim_lp_m24 p24 lp_m24 "4 5" "0 1" > outputs/cvpr27/lp_m24_s600.log 2>&1 &
sleep 30
arm lp_r50 r50 lp_r50 "6 7" "2 3" > outputs/cvpr27/lp_r50_s600.log 2>&1 &
stamp "both arms launched"
wait
cat outputs/cvpr27/lp_m24_s600.log outputs/cvpr27/lp_r50_s600.log

if [ -e outputs/cvpr27/alpasim_lp_m24_s600/analysis.DONE ] && [ -e outputs/cvpr27/alpasim_lp_r50_s600/analysis.DONE ]; then
  cd experiments/evaluation
  $PY analyze_dual_s600.py --arm r50_merged_lp_r50 --arm-label lp_r50 --base p24_merged_slim_lp_m24 \
    --base-label lp_m24 --no-pre-fix --out "$REPO/outputs/cvpr27/alpasim_lp_r50_s600/vs_lp_m24"
  cd "$REPO"
fi
stamp "pair finished"
