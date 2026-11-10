#!/bin/bash
# Rest of the u50 Origin150 x 2 closed loop after the Tyr resume (iclr27-rebuttal/plans/2026-11-09_u50-closedloop-origin150.md).
# User 11-10: card type (Blackwell / Ada) is no longer considered -- remaining scenes go round-robin over GPUS.
# Order (user): MoOP u50 l0keep -> Tyr u50 -> LLM-Pruner u50 -> plain MoOP u50 last.
#   0. wait for run_u50_tyr_resume.sh (Tyr: 51 + 75 scenes, merge) to exit
#   1. MoOP u50 l0keep: its 75 not-yet-run scenes (shards 0-3 of scenes_main.csv) via run_u50_origin150.sh, which then
#      merges them with the finished Ada half (150 scenes)
#   2. LLM-Pruner u50, then MoOP u50: all 150 scenes x 2 over GPUS, merged into u50o_merged_<cfg>, G1/G2 gates
#
#   GPUS="0 1 2 3" setsid nohup bash iclr27-rebuttal/code/run_u50_queue.sh > outputs/iclr27-rebuttal/u50_origin150/run_queue.log 2>&1 &
set -u
REPO=/home/cvlab21/project/chan/alpamayo-model-compression
OUT=$REPO/outputs/iclr27-rebuttal/u50_origin150
RUNS=/home/cvlab21/project/chan/alpasim-runs
DRIVERS=/mnt/nvme1n1/ad_vla/data/alpasim/drivers
PY=$REPO/.venv/bin/python
APY=/mnt/nvme1n1/ad_vla/venvs/alpasim-chan/bin/python
LAUNCH=$REPO/experiments/head_analysis/launch_alpasim_shards.sh
MERGE=$REPO/experiments/head_analysis/merge_alpasim_shards.py
GPUS=${GPUS:-"0 1 2 3"}
QP=u50oQ_
stamp() { echo "$(date -d '+9 hours' '+%F %T') KST $*"; }
mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1"; }
empty() { for g in $GPUS; do [ "$(mib "$g")" -le 10 ] || return 1; done; }
cd "$REPO"

n=0
while pgrep -f "run_u50_tyr_resume.sh" > /dev/null; do
  [ $((n % 60)) = 0 ] && stamp "waiting for the Tyr resume to finish"
  n=$((n + 1)); sleep 60
done
stamp "Tyr resume exited ($( [ -f "$OUT/slim_tyr_u50_r.DONE" ] && echo merged || echo NOT merged -- check run_tyr_resume.log))"

# 1. MoOP u50 l0keep, remaining 75 scenes
SHARDS="0 1 2 3" ARMS="slim_dual_u50_l0keep_v2" bash iclr27-rebuttal/code/run_u50_origin150.sh   # shard i on GPU i

# 2. LLM-Pruner u50, plain MoOP u50: 150 scenes round-robin over GPUS
$PY -c "import pandas as pd; pd.read_csv('$OUT/scenes_main.csv')[['scene_id', 'uuid']].to_csv('$OUT/scenes_all150.csv', index=False)"
k=$(echo $GPUS | wc -w)
for CFG in slim_lp_r60 slim_dual_u50_v2; do
  [ -f "$OUT/$CFG.DONE" ] && { stamp "$CFG: already done"; continue; }
  [ -e "$DRIVERS/$CFG" ] || cp -al "outputs/$CFG" "$DRIVERS/$CFG"
  until empty && sleep 30 && empty; do stamp "$CFG: waiting for empty GPUs $GPUS"; sleep 60; done
  stamp "$CFG: 150 scenes x 2 rollouts over GPUs $GPUS"
  SUITE=public_2601_u50oQ PARENT_SUITE=public_2601 PREFIX=$QP DRIVER_OMP_THREADS=8 SCENES_CSV=$OUT/scenes_all150.csv \
    bash "$LAUNCH" "$CFG" 150 2 "$GPUS"
  stamp "$CFG: shards finished (exit $?)"
  for c in $(docker ps -a --format '{{.Names}}' | grep "^${QP}${CFG}_sh"); do docker rm -f "$c" > /dev/null; done
  for nw in $(docker network ls --format '{{.Name}}' | grep "^${QP}${CFG}_sh"); do
    [ "$(docker network inspect -f '{{len .Containers}}' "$nw")" = 0 ] && docker network rm "$nw" > /dev/null
  done
  shards=$(for i in $(seq 0 $((k - 1))); do echo "${QP}${CFG}_sh$i"; done)
  merged=u50o_merged_$CFG
  mkdir "$OUT/merge_$CFG.lock" 2> /dev/null || { stamp "$CFG: merge already taken"; continue; }
  $PY "$MERGE" --runs-root "$RUNS" --shards $shards --out "$merged" --expect-scenes 150
  rc=$?
  stamp "$CFG: merged -> $RUNS/$merged (exit $rc)"
  (cd experiments/evaluation && $APY s600_smoke_gates.py --run "$merged" --logs "${QP}$CFG" \
     --scenes "$OUT/scenes_main.csv" --out "$OUT/gates_$CFG") > "$OUT/gates_$CFG.txt" 2>&1
  stamp "$CFG: $(grep -m1 'gates:' "$OUT/gates_$CFG.txt"); Rollout FAILED lines $(cat "$RUNS"/${QP}${CFG}_sh*.launch.log | grep -c 'Rollout FAILED')"
  [ "$rc" = 0 ] && touch "$OUT/$CFG.DONE"
done
stamp "queue done"
