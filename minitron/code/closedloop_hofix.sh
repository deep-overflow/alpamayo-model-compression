#!/bin/bash
# Closed loop on Origin150 after the AlpaSim hand-over fix
# (minitron/plans/2026-10-26_alpasim-handover-fix.md, patch minitron/alpasim/handover_fix.patch).
#
#   STAGE=gate  first 20 Origin150 scenes x 1 rollout, dense only -> gates G1-G4 (judged by hand
#               against alpasim-runs/m2601_merged_baseline, the same scenes before the fix)
#   STAGE=main  all 150 scenes x 2 rollouts: dense, MoOP, MT-W + KD, MT-W zs, in that order
#
# Origin150 = suite public_2601, first 150 scene ids (sceneset 6f937b0c...). Drivers run on
# Ada 4-7 (one shard per card, every service of a shard on its card) with DRIVER_OMP_THREADS=8
# for every arm. Waits for the open-loop eval (outputs/minitron/mtw24_kd_eval/DONE) and for
# the four cards to be idle, merges the shards per arm, then removes this run's containers and
# compose networks (user rule, 2026-10-19).
#
#   STAGE=gate nohup bash minitron/code/closedloop_hofix.sh > outputs/minitron/closedloop_hofix_gate.log 2>&1 &
set -u
STAGE=${STAGE:-gate}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
LAUNCH=$REPO/experiments/head_analysis/launch_alpasim_shards.sh
MERGE=$REPO/experiments/head_analysis/merge_alpasim_shards.py
RUNS=/home/cvlab21/project/chan/alpasim-runs
GPUS="4 5 6 7"
stamp() { echo "$(date -u '+%F %T') UTC $*"; }

if [ "$STAGE" = gate ]; then
  PREFIX=hofix_gate_ N_SCENES=20 N_ROLLOUTS=1 CFGS="baseline"
else
  PREFIX=hofix_ N_SCENES=150 N_ROLLOUTS=2 CFGS="baseline slim_dual_u40_v2 slim_mtw24_kd slim_mtw24"
fi

stamp "waiting for the open-loop eval to finish"
until [ -f "$REPO/outputs/minitron/mtw24_kd_eval/DONE" ]; do sleep 120; done
idle=0
until [ "$idle" -ge 3 ]; do
  busy=0
  for g in $GPUS; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    [ "$used" -gt 2000 ] && busy=1
  done
  if [ "$busy" = 0 ]; then idle=$((idle + 1)); else idle=0; fi
  sleep 60
done
stamp "Ada $GPUS idle -- stage $STAGE: $CFGS, $N_SCENES scenes x $N_ROLLOUTS rollouts"

cleanup() {  # compose names containers after the log dir: <PREFIX><cfg>_sh<i>-<service>-0-1
  for c in $(docker ps -a --format '{{.Names}}' | grep "^${PREFIX}"); do docker rm -f "$c" > /dev/null; done
  for n in $(docker network ls --format '{{.Name}}' | grep "^${PREFIX}"); do
    [ "$(docker network inspect -f '{{len .Containers}}' "$n")" = 0 ] && docker network rm "$n" > /dev/null
  done
  stamp "cleanup: containers left $(docker ps -a --format '{{.Names}}' | grep -c "^${PREFIX}"), networks left $(docker network ls --format '{{.Name}}' | grep -c "^${PREFIX}")"
}

for cfg in $CFGS; do
  stamp "--- $cfg"
  SUITE=public_2601 PREFIX=$PREFIX DRIVER_OMP_THREADS=8 \
    bash "$LAUNCH" "$cfg" "$N_SCENES" "$N_ROLLOUTS" "$GPUS"
  stamp "$cfg shards finished (exit $?)"
  "$REPO/.venv/bin/python" "$MERGE" --runs-root "$RUNS" \
    --shards "${PREFIX}${cfg}_sh0" "${PREFIX}${cfg}_sh1" "${PREFIX}${cfg}_sh2" "${PREFIX}${cfg}_sh3" \
    --out "${PREFIX}merged_${cfg}" --expect-scenes "$N_SCENES"
  stamp "$cfg merged -> $RUNS/${PREFIX}merged_${cfg} (exit $?)"
  cleanup
done
stamp "stage $STAGE done"
touch "$REPO/outputs/minitron/closedloop_hofix_${STAGE}.DONE"
