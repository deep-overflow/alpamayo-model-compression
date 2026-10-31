#!/bin/bash
# After build_tyr_off_local.sh: norm check on the final configuration, then hand slim_tyr_off to the
# KISTI A100 open-loop track (test_2000, OOD-Val262, LingoQA). Waits for the build and for the OTP sockets.
#
#   bash cvpr27/code/tyr_off/after_build.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=cvpr27/tyr_off
S=outputs/cvpr27/slim_tyr_off
LOG=/mnt/nvme1n1/ad_vla/outputs/chan/cvpr27/tyr_off_build_local.log
R=/scratch/e1997a06/project/alpamayo-model-compression
F='Deprecation|Redirecting'
stamp() { echo "$(date '+%F %T') $*"; }
up() { ssh -O check "$1" >/dev/null 2>&1; }

until grep -q "all done" "$LOG" 2>/dev/null && [ -f "$S/slim_state.pt" ]; do
  pgrep -f build_tyr_off_local.sh > /dev/null || { stamp "build script gone without 'all done'"; tail -n 5 "$LOG"; exit 1; }
  sleep 60
done
stamp "build done: $(grep -E 'removed|surgery' outputs/$O/logs/make_slim.log | tail -n 2 | tr '\n' ' ')"

HF_HOME=$HOME/.cache/huggingface HF_HUB_CACHE=/mnt/nvme1n1/ad_vla/cache/hub HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python cvpr27/code/tyr_off/check_level_norms.py "$O/supernet_r4" "$O/search_r4/final_config.json" \
  > "outputs/$O/norms_r4.txt" 2>&1
tail -n 3 "outputs/$O/norms_r4.txt"

for _ in $(seq 1 720); do up neuron-dm && up neuron && break; sleep 60; done
up neuron-dm && up neuron || { stamp "OTP sockets not up after 12 h -- push by hand"; exit 2; }

stamp "push $S (with slim_state.pt) and the track scripts"
rsync -a --partial "$S/" "neuron-dm:/scratch/e1997a06/outputs/cvpr27/slim_tyr_off/" || exit 3
rsync -a cvpr27/code/eval_a100.sbatch cvpr27/code/a100_queue.sh "neuron-dm:$R/cvpr27/code/" || exit 3
l=$(cd "$S" && sha256sum slim_state.pt slim_meta.json | awk '{print $1}')
r=$(ssh -o BatchMode=yes neuron-dm "cd /scratch/e1997a06/outputs/cvpr27/slim_tyr_off && sha256sum slim_state.pt slim_meta.json" \
    2>&1 | grep -vE "$F" | awk '{print $1}')
[ "$l" = "$r" ] || { stamp "SHA MISMATCH"; echo "$l"; echo "$r"; exit 4; }
stamp "sha OK"
ssh -o BatchMode=yes neuron-dm "/bin/bash $R/cvpr27/code/a100_queue.sh arm --lingo tyr_off | tail -n 8" 2>&1 | grep -vE "$F"
ssh -o BatchMode=yes neuron "cd $R && for n in 1 2; do sbatch --job-name=eval_a100_tyr --gres=gpu:2 --cpus-per-task=16 \
  --mem=120G --time=05:00:00 cvpr27/code/eval_a100.sbatch; done && sleep 10 && squeue -u e1997a06 -o '%.10i %.14j %.9T %.11M %R'" \
  2>&1 | grep -vE "$F"
stamp "submitted"
