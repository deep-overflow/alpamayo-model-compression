#!/bin/bash
# u50 closed loop on Origin150 x 2 rollouts (iclr27-rebuttal/plans/2026-11-09_u50-closedloop-origin150.md), one arm
# after another, each on all eight cards with the 75 / 75 scene split of scenes_main.csv (shard i -> GPU i; Blackwell
# 0-3, Ada 4-7; Ada scenes on the card the patched dense run hofix_merged_baseline used). Same harness settings as
# hofix (patched AlpaSim, pair_a15, every service of a shard on its card, DRIVER_OMP_THREADS=8, ASL images off).
# Per arm: wait for its build (outputs/<cfg>/config.json, written after make_slim's smoke) -> hard-link the driver ->
# wait for eight empty cards -> 150 x 2 -> merge -> G1/G2 (s600_smoke_gates.py on the merged run) -> remove this arm's
# containers and networks. No separate smoke stage (user, 11-10: start now); the first rollouts are checked by hand.
# SHARDS runs a subset of the same 8-way split (user, 11-10: Ada 4-7 first, the Blackwell half 0-3 later); an arm is
# merged once all eight of its shards have results, i.e. by the run that finishes the last half.
#
#   SHARDS="4 5 6 7" ARMS="slim_dual_u50_l0keep_v2 slim_dual_u50_v2" setsid nohup bash \
#     iclr27-rebuttal/code/run_u50_origin150.sh > outputs/iclr27-rebuttal/u50_origin150/run_ada.log 2>&1 &
set -u
REPO=/home/cvlab21/project/chan/alpamayo-model-compression
OUT=$REPO/outputs/iclr27-rebuttal/u50_origin150
RUNS=/home/cvlab21/project/chan/alpasim-runs
DRIVERS=/mnt/nvme1n1/ad_vla/data/alpasim/drivers
PY=$REPO/.venv/bin/python
APY=/mnt/nvme1n1/ad_vla/venvs/alpasim-chan/bin/python
LAUNCH=$REPO/experiments/head_analysis/launch_alpasim_shards.sh
MERGE=$REPO/experiments/head_analysis/merge_alpasim_shards.py
ARMS=${ARMS:-"slim_dual_u50_l0keep_v2 slim_dual_u50_v2"}
GPUS="0 1 2 3 4 5 6 7"
SHARDS=${SHARDS:-$GPUS}   # shard i always runs on GPU i
PREFIX=u50o_
stamp() { echo "$(date -d '+9 hours' '+%F %T') KST $*"; }
mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1"; }
cd "$REPO"

all_empty() { for g in $SHARDS; do [ "$(mib "$g")" -le 10 ] || return 1; done; }
shard_done() { [ -f "$RUNS/${PREFIX}$1_sh$2/aggregate/results-summary.json" ]; }

for CFG in $ARMS; do
  [ -f "$OUT/$CFG.DONE" ] && { stamp "$CFG: already done"; continue; }
  n=0
  until [ -f "outputs/$CFG/config.json" ] && [ -f "outputs/$CFG/slim_state.pt" ]; do
    [ $((n % 30)) = 0 ] && stamp "$CFG: waiting for its build"
    n=$((n + 1)); sleep 60
  done
  [ -e "$DRIVERS/$CFG" ] || cp -al "outputs/$CFG" "$DRIVERS/$CFG"
  todo=""
  for i in $SHARDS; do shard_done "$CFG" "$i" || todo="$todo $i"; done
  todo=${todo# }
  if [ -n "$todo" ]; then
  n=0
  until all_empty && sleep 30 && all_empty; do
    [ $((n % 30)) = 0 ] && stamp "$CFG: waiting for empty cards $SHARDS: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
    n=$((n + 1)); sleep 30
  done
  stamp "$CFG: shards $todo of 150 scenes x 2 rollouts (shard i on GPU i)"
  (while true; do
    echo "$(date -u +%T) load $(cut -d' ' -f1-3 /proc/loadavg)" \
         "gpu_mem $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
    sleep 60
  done) > "$OUT/${CFG}_load.log" 2>&1 &
  sampler=$!
  SUITE=public_2601_u50o PARENT_SUITE=public_2601 PREFIX=$PREFIX DRIVER_OMP_THREADS=8 SCENES_CSV=$OUT/scenes_main.csv \
    ONLY_SHARDS="$todo" bash "$LAUNCH" "$CFG" 150 2 "$GPUS"
  stamp "$CFG: shards $todo finished (exit $?)"
  kill "$sampler"
  for c in $(docker ps -a --format '{{.Names}}' | grep "^${PREFIX}${CFG}_sh"); do docker rm -f "$c" > /dev/null; done
  for nw in $(docker network ls --format '{{.Name}}' | grep "^${PREFIX}${CFG}_sh"); do
    [ "$(docker network inspect -f '{{len .Containers}}' "$nw")" = 0 ] && docker network rm "$nw" > /dev/null
  done
  fi
  missing=""
  for i in $GPUS; do shard_done "$CFG" "$i" || missing="$missing $i"; done
  if [ -n "$missing" ]; then stamp "$CFG: shards$missing not run yet -- merge waits for them"; continue; fi

  merged=${PREFIX}merged_$CFG
  $PY "$MERGE" --runs-root "$RUNS" --shards $(for i in 0 1 2 3 4 5 6 7; do echo "${PREFIX}${CFG}_sh$i"; done) \
    --out "$merged" --expect-scenes 150
  rc=$?
  stamp "$CFG: merged -> $RUNS/$merged (exit $rc)"
  (cd experiments/evaluation && $APY s600_smoke_gates.py --run "$merged" --logs "${PREFIX}$CFG" \
     --scenes "$OUT/scenes_main.csv" --out "$OUT/gates_$CFG") > "$OUT/gates_$CFG.txt" 2>&1
  stamp "$CFG: $(grep -m1 'gates:' "$OUT/gates_$CFG.txt")"
  stamp "$CFG: route sanity aborts $(grep -l -r 'fails sanity check\|does not pass sanity check' "$RUNS"/${PREFIX}${CFG}_sh*/ \
     --include='*.log' 2>/dev/null | wc -l) log files; Rollout FAILED lines $(cat "$RUNS"/${PREFIX}${CFG}_sh*.launch.log | grep -c 'Rollout FAILED')"
  [ "$rc" = 0 ] && touch "$OUT/$CFG.DONE"
done
stamp "all arms done: $ARMS"
