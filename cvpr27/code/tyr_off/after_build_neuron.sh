#!/bin/bash
# After a Tyr build job on NEURON (build_tyr_off.sbatch): if the slim checkpoint exists, queue its A100 arm
# (test_2000, OOD-Val262, LingoQA) and submit two 2-card holders.
#
#   bash cvpr27/code/tyr_off/after_build_neuron.sh <build job id> <arm> <outputs-relative slim dir>
#   e.g. bash cvpr27/code/tyr_off/after_build_neuron.sh 957800 tyr_vt cvpr27/slim_tyr_vt
set -uo pipefail
J=${1:?build job id}
ARM=${2:?arm}
SLIM=${3:?slim dir}
R=/scratch/e1997a06/project/alpamayo-model-compression
F='Deprecation|Redirecting'
stamp() { echo "$(date '+%F %T') $*"; }
for _ in $(seq 1 1440); do
  st=$(ssh -o BatchMode=yes -o ConnectTimeout=20 neuron-dm "sacct -j $J -X -n -o State | tr -d ' '" 2>&1 | grep -vE "$F" | head -1)
  case $st in COMPLETED|FAILED|CANCELLED*|TIMEOUT|NODE_FAIL|OUT_OF_ME*) break ;; "") stamp "ssh failure"; exit 2 ;; esac
  sleep 120
done
stamp "build job $J: $st"
ok=$(ssh -o BatchMode=yes neuron-dm "ls /scratch/e1997a06/outputs/$SLIM/slim_state.pt 2>/dev/null; \
  tail -n 15 /scratch/e1997a06/project/amc-tyr/logs/*-$J.out" 2>&1 | grep -vE "$F")
echo "$ok"
grep -q "slim_state.pt" <<< "$ok" || { stamp "no slim_state.pt -- not queueing"; exit 3; }
ssh -o BatchMode=yes neuron-dm "/bin/bash $R/cvpr27/code/a100_queue.sh arm --lingo $ARM | tail -n 8" 2>&1 | grep -vE "$F"
ssh -o BatchMode=yes neuron "cd $R && for n in 1 2; do sbatch --job-name=eval_a100_$ARM --gres=gpu:2 --cpus-per-task=16 \
  --mem=120G --time=05:00:00 cvpr27/code/eval_a100.sbatch; done && sleep 10 && squeue -u e1997a06 -o '%.10i %.16j %.9T %.11M %R'" \
  2>&1 | grep -vE "$F"
stamp "queued and submitted"
