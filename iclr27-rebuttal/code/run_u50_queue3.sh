#!/bin/bash
# u50 Origin150 x 2 closed loop, two card groups (user, 11-11 15:3x: "run the next arm on GPUs 0-3"). Replaces the
# 8-card step of run_u50_queue2.sh. Scenes split in halves by scenes_main.csv: A = shards 0-3, B = shards 4-7.
#   group 0-3 (now):              LLM-Pruner u50 half A -> plain MoOP u50 half A
#   group 4-7 (after l0keep):     wait for the running l0keep launch (left by queue2) -> merge l0keep 150, gates
#                                 -> LLM-Pruner u50 half B -> plain MoOP u50 half B
# Plain MoOP stays last on each group (user). An arm is merged (150 scenes) and gated by whichever group finishes it last.
#
#   setsid nohup bash iclr27-rebuttal/code/run_u50_queue3.sh > outputs/iclr27-rebuttal/u50_origin150/run_queue3.log 2>&1 &
set -u
REPO=/home/cvlab21/project/chan/alpamayo-model-compression
OUT=$REPO/outputs/iclr27-rebuttal/u50_origin150
RUNS=/home/cvlab21/project/chan/alpasim-runs
DRIVERS=/mnt/nvme1n1/ad_vla/data/alpasim/drivers
PY=$REPO/.venv/bin/python
APY=/mnt/nvme1n1/ad_vla/venvs/alpasim-chan/bin/python
LAUNCH=$REPO/experiments/head_analysis/launch_alpasim_shards.sh
MERGE=$REPO/experiments/head_analysis/merge_alpasim_shards.py
stamp() { echo "$(date -d '+9 hours' '+%F %T') KST $*"; }
mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1"; }
empty() { for g in $1; do [ "$(mib "$g")" -le 10 ] || return 1; done; }
has_summary() { [ -f "$RUNS/$1/aggregate/results-summary.json" ]; }
cleanup() {
  for c in $(docker ps -a --format '{{.Names}}' | grep "^$1"); do docker rm -f "$c" > /dev/null; done
  for nw in $(docker network ls --format '{{.Name}}' | grep "^$1"); do
    [ "$(docker network inspect -f '{{len .Containers}}' "$nw")" = 0 ] && docker network rm "$nw" > /dev/null
  done
}
merge_and_gate() {  # cfg, shard run names...
  local cfg=$1; shift
  local merged=u50o_merged_$cfg
  for s in "$@"; do has_summary "$s" || { stamp "$cfg: $s not finished -- merge left to the other group"; return 0; }; done
  mkdir "$OUT/merge_$cfg.lock" 2> /dev/null || { stamp "$cfg: merge already taken"; return 0; }
  $PY "$MERGE" --runs-root "$RUNS" --shards "$@" --out "$merged" --expect-scenes 150
  local rc=$?
  stamp "$cfg: merged -> $RUNS/$merged (exit $rc)"
  (cd experiments/evaluation && $APY s600_smoke_gates.py --run "$merged" --logs "u50o_$cfg" \
     --scenes "$OUT/scenes_main.csv" --out "$OUT/gates_$cfg") > "$OUT/gates_$cfg.txt" 2>&1
  stamp "$cfg: $(grep -m1 'gates:' "$OUT/gates_$cfg.txt")"
  [ "$rc" = 0 ] && touch "$OUT/$cfg.DONE"
}
run_half() {  # cfg, half (A|B), gpus
  local cfg=$1 h=$2 gpus=$3 p=u50o$2_
  local all=1
  for i in 0 1 2 3; do has_summary "${p}${cfg}_sh$i" || all=0; done
  [ "$all" = 1 ] && { stamp "$cfg half $h: already done"; return 0; }
  [ -e "$DRIVERS/$cfg" ] || cp -al "outputs/$cfg" "$DRIVERS/$cfg"
  until empty "$gpus" && sleep 30 && empty "$gpus"; do stamp "$cfg half $h: waiting for empty GPUs $gpus"; sleep 60; done
  stamp "$cfg half $h: 75 scenes x 2 on GPUs $gpus"
  SUITE=public_2601_u50o$h PARENT_SUITE=public_2601 PREFIX=$p DRIVER_OMP_THREADS=8 SCENES_CSV=$OUT/scenes_half$h.csv \
    bash "$LAUNCH" "$cfg" 75 2 "$gpus"
  stamp "$cfg half $h: shards finished (exit $?); Rollout FAILED lines $(cat "$RUNS"/${p}${cfg}_sh*.launch.log | grep -c 'Rollout FAILED')"
  cleanup "${p}${cfg}_sh"
}
halves() {  # cfg -> the 8 shard runs of both halves
  for h in A B; do for i in 0 1 2 3; do echo "u50o${h}_$1_sh$i"; done; done
}
cd "$REPO"

$PY -c "
import pandas as pd
m = pd.read_csv('$OUT/scenes_main.csv')
a = m[m.shard < 4].copy(); a.to_csv('$OUT/scenes_halfA.csv', index=False)
b = m[m.shard >= 4].copy(); b['shard'] = b.shard - 4; b.to_csv('$OUT/scenes_halfB.csv', index=False)
print('half A', len(a), a.shard.value_counts().sort_index().to_dict(), '| half B', len(b), b.shard.value_counts().sort_index().to_dict())"

group_A() {
  for cfg in slim_lp_r60 slim_dual_u50_v2; do
    run_half "$cfg" A "0 1 2 3"
    merge_and_gate "$cfg" $(halves "$cfg")
  done
  stamp "group 0-3 done"
}
group_B() {
  local L=slim_dual_u50_l0keep_v2 n=0
  while pgrep -f "launch_alpasim_shards.sh $L " > /dev/null; do
    [ $((n % 30)) = 0 ] && stamp "$L: running launch on GPUs 4-7 -- waiting for it"
    n=$((n + 1)); sleep 60
  done
  cleanup "u50oL_${L}_sh"
  [ -f "$OUT/$L.DONE" ] || merge_and_gate "$L" $(for i in 0 1 2 3; do echo "u50oL_${L}_sh$i"; done) \
    $(for i in 4 5 6 7; do echo "u50o_${L}_sh$i"; done)
  for cfg in slim_lp_r60 slim_dual_u50_v2; do
    run_half "$cfg" B "4 5 6 7"
    merge_and_gate "$cfg" $(halves "$cfg")
  done
  stamp "group 4-7 done"
}
group_A > "$OUT/run_queue3_A.log" 2>&1 &
pa=$!
sleep 90   # the two groups write their shard-suite CSVs under different prefixes; stagger the starts anyway
group_B > "$OUT/run_queue3_B.log" 2>&1 &
pb=$!
wait $pa; wait $pb
stamp "queue3 done"
