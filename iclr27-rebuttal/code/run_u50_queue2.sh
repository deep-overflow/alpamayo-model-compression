#!/bin/bash
# u50 Origin150 x 2 closed loop, 8-card schedule (user, 11-11 10:3x: GPUs 4-7 free and usable; card type ignored since 11-10).
# Replaces run_u50_queue.sh. Tyr keeps running on GPUs 0-3 under run_u50_tyr_resume.sh.
#   1. now, GPUs 4-7: MoOP u50 l0keep's 75 not-yet-run scenes (scenes_main.csv shards 0-3; shard i -> GPU 4+i), then
#      merge with its finished half (u50o_..._sh4-7) into u50o_merged_slim_dual_u50_l0keep_v2 (150 scenes), gates
#   2. once the Tyr resume has exited and all eight cards are empty: LLM-Pruner u50, then plain MoOP u50 (user: last),
#      each 150 scenes x 2 round-robin over GPUs 0-7, merged and gated
#
#   setsid nohup bash iclr27-rebuttal/code/run_u50_queue2.sh > outputs/iclr27-rebuttal/u50_origin150/run_queue2.log 2>&1 &
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
cleanup() {  # prefix
  for c in $(docker ps -a --format '{{.Names}}' | grep "^$1"); do docker rm -f "$c" > /dev/null; done
  for nw in $(docker network ls --format '{{.Name}}' | grep "^$1"); do
    [ "$(docker network inspect -f '{{len .Containers}}' "$nw")" = 0 ] && docker network rm "$nw" > /dev/null
  done
}
merge_and_gate() {  # cfg, shard run names...
  local cfg=$1; shift
  local merged=u50o_merged_$cfg
  for s in "$@"; do [ -f "$RUNS/$s/aggregate/results-summary.json" ] || { stamp "$cfg: $s has no summary -- not merging"; return 1; }; done
  mkdir "$OUT/merge_$cfg.lock" 2> /dev/null || { stamp "$cfg: merge already taken"; return 0; }
  $PY "$MERGE" --runs-root "$RUNS" --shards "$@" --out "$merged" --expect-scenes 150
  local rc=$?
  stamp "$cfg: merged -> $RUNS/$merged (exit $rc)"
  (cd experiments/evaluation && $APY s600_smoke_gates.py --run "$merged" --logs "u50o_$cfg" \
     --scenes "$OUT/scenes_main.csv" --out "$OUT/gates_$cfg") > "$OUT/gates_$cfg.txt" 2>&1
  stamp "$cfg: $(grep -m1 'gates:' "$OUT/gates_$cfg.txt")"
  [ "$rc" = 0 ] && touch "$OUT/$cfg.DONE"
}
cd "$REPO"

# 1. MoOP u50 l0keep, remaining 75 scenes on GPUs 4-7
CFG=slim_dual_u50_l0keep_v2
LP_=u50oL_
$PY -c "
import pandas as pd
m = pd.read_csv('$OUT/scenes_main.csv')
r = m[m.shard < 4].copy(); r['orig_shard'] = r.shard; r['gpu'] = r.shard + 4
r.to_csv('$OUT/scenes_l0keep_rest.csv', index=False); print(len(r), r.shard.value_counts().sort_index().to_dict())"
if [ ! -f "$OUT/$CFG.DONE" ]; then
  ok=1; for i in 0 1 2 3; do [ -f "$RUNS/${LP_}${CFG}_sh$i/aggregate/results-summary.json" ] || ok=0; done
  if [ "$ok" = 0 ]; then
    until empty "4 5 6 7" && sleep 30 && empty "4 5 6 7"; do stamp "$CFG: waiting for empty GPUs 4-7"; sleep 60; done
    stamp "$CFG: remaining 75 scenes x 2 on GPUs 4-7"
    SUITE=public_2601_u50oL PARENT_SUITE=public_2601 PREFIX=$LP_ DRIVER_OMP_THREADS=8 SCENES_CSV=$OUT/scenes_l0keep_rest.csv \
      bash "$LAUNCH" "$CFG" 75 2 "4 5 6 7"
    stamp "$CFG: shards finished (exit $?)"
    cleanup "${LP_}${CFG}_sh"
  fi
  merge_and_gate "$CFG" $(for i in 0 1 2 3; do echo "${LP_}${CFG}_sh$i"; done) \
    $(for i in 4 5 6 7; do echo "u50o_${CFG}_sh$i"; done)
fi

# 2. LLM-Pruner u50, then plain MoOP u50, on all eight cards
n=0
while pgrep -f "run_u50_tyr_resume.sh" > /dev/null; do
  [ $((n % 60)) = 0 ] && stamp "waiting for the Tyr resume (GPUs 0-3) to finish"
  n=$((n + 1)); sleep 60
done
$PY -c "import pandas as pd; pd.read_csv('$OUT/scenes_main.csv')[['scene_id', 'uuid']].to_csv('$OUT/scenes_all150.csv', index=False)"
QP=u50oQ_
G8="0 1 2 3 4 5 6 7"
for CFG in slim_lp_r60 slim_dual_u50_v2; do
  [ -f "$OUT/$CFG.DONE" ] && { stamp "$CFG: already done"; continue; }
  [ -e "$DRIVERS/$CFG" ] || cp -al "outputs/$CFG" "$DRIVERS/$CFG"
  until empty "$G8" && sleep 30 && empty "$G8"; do stamp "$CFG: waiting for eight empty cards"; sleep 60; done
  stamp "$CFG: 150 scenes x 2 rollouts over GPUs $G8"
  SUITE=public_2601_u50oQ PARENT_SUITE=public_2601 PREFIX=$QP DRIVER_OMP_THREADS=8 SCENES_CSV=$OUT/scenes_all150.csv \
    bash "$LAUNCH" "$CFG" 150 2 "$G8"
  stamp "$CFG: shards finished (exit $?); Rollout FAILED lines $(cat "$RUNS"/${QP}${CFG}_sh*.launch.log | grep -c 'Rollout FAILED')"
  cleanup "${QP}${CFG}_sh"
  merge_and_gate "$CFG" $(for i in 0 1 2 3 4 5 6 7; do echo "${QP}${CFG}_sh$i"; done)
done
stamp "queue2 done"
