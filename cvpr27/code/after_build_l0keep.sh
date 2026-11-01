#!/bin/bash
# After build_l0keep_ratios.sbatch (job $1): stop unless the u40 gate passed, then queue the three l0keep ratio arms
# and submit a second 2-card holder (the first one, already running, picks the new lines up too).
#
#   bash cvpr27/code/after_build_l0keep.sh <build job id>
set -uo pipefail
J=${1:?build job id}
R=/scratch/e1997a06/project/alpamayo-model-compression
LOG=/scratch/e1997a06/project/amc-tyr/logs/build_l0k-$J.out
F='Deprecation|Redirecting'
stamp() { echo "$(date '+%F %T') $*"; }
for _ in $(seq 1 120); do
  st=$(ssh -o BatchMode=yes -o ConnectTimeout=20 neuron-dm "sacct -j $J -X -n -o State | tr -d ' '" 2>&1 | grep -vE "$F" | head -1)
  case $st in COMPLETED|FAILED|CANCELLED*|TIMEOUT|NODE_FAIL|OUT_OF_ME*) break ;; "") stamp "ssh failure"; exit 2 ;; esac
  sleep 60
done
stamp "build job $J: $st"
out=$(ssh -o BatchMode=yes neuron-dm "cat $LOG" 2>&1 | grep -vE "$F")
echo "$out" | grep -E "^u[0-9]+:|GATE|exited|done"
grep -q "GATE u40 rebuild == shipped slim_dual_u40_l0keep_v2: True" <<< "$out" || { stamp "gate not passed -- not queueing"; exit 3; }
ssh -o BatchMode=yes neuron-dm "/bin/bash $R/cvpr27/code/a100_queue.sh arm --lingo l0keep_u35 l0keep_u45 l0keep_u50 | tail -n 22" \
  2>&1 | grep -vE "$F"
ssh -o BatchMode=yes neuron "cd $R && sbatch --job-name=eval_a100_l0k --gres=gpu:2 --cpus-per-task=16 --mem=120G \
  --time=10:00:00 cvpr27/code/eval_a100.sbatch && sleep 10 && squeue -u e1997a06 -o '%.10i %.16j %.9T %.11M %R'" \
  2>&1 | grep -vE "$F"
stamp "queued and submitted"
