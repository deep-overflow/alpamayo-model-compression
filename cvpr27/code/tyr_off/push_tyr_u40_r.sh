#!/bin/bash
# Put the paper's Tyr row (outputs/slim_tyr_u40_r, with slim_state.pt) on the KISTI A100 track:
# wait for the OTP sockets, push + sha256 check, update the track scripts, queue, submit two 2-card holders.
#
#   bash cvpr27/code/tyr_off/push_tyr_u40_r.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."
S=outputs/slim_tyr_u40_r
R=/scratch/e1997a06/project/alpamayo-model-compression
F='Deprecation|Redirecting'
stamp() { echo "$(date '+%F %T') $*"; }
up() { ssh -O check "$1" >/dev/null 2>&1; }

for _ in $(seq 1 720); do up neuron-dm && up neuron && break; sleep 60; done
up neuron-dm && up neuron || { stamp "OTP sockets not up after 12 h"; exit 2; }
stamp "push $S (with slim_state.pt) and the track scripts"
rsync -a --partial "$S/" "neuron-dm:/scratch/e1997a06/outputs/slim_tyr_u40_r/" || exit 3
rsync -a cvpr27/code/eval_a100.sbatch cvpr27/code/a100_queue.sh "neuron-dm:$R/cvpr27/code/" || exit 3
l=$(cd "$S" && sha256sum slim_state.pt slim_meta.json | awk '{print $1}')
r=$(ssh -o BatchMode=yes neuron-dm "cd /scratch/e1997a06/outputs/slim_tyr_u40_r && sha256sum slim_state.pt slim_meta.json" \
    2>&1 | grep -vE "$F" | awk '{print $1}')
[ "$l" = "$r" ] || { stamp "SHA MISMATCH"; echo "$l"; echo "$r"; exit 4; }
stamp "sha OK"
ssh -o BatchMode=yes neuron-dm "/bin/bash $R/cvpr27/code/a100_queue.sh arm --lingo tyr_u40_r | tail -n 8" 2>&1 | grep -vE "$F"
ssh -o BatchMode=yes neuron "cd $R && for n in 1 2; do sbatch --job-name=eval_a100_tyrr --gres=gpu:2 --cpus-per-task=16 \
  --mem=120G --time=05:00:00 cvpr27/code/eval_a100.sbatch; done && sleep 10 && squeue -u e1997a06 -o '%.10i %.14j %.9T %.11M %R'" \
  2>&1 | grep -vE "$F"
stamp "submitted"
