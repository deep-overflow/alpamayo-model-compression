#!/bin/bash
# After the local Tyr u45 build (cvpr27/plans/2026-11-03_baselines-ratio-sweep.md): check the budget, push the recipe
# to NEURON (slim_meta.json last), drop rounds 1-3, and make sure an A100 holder will claim its 7 queue lines.
# Stops with a message instead of retrying when a NEURON socket is down (the user reopens it with an OTP).
set -uo pipefail
cd "$(dirname "$0")/../.."
F='Deprecation|Redirecting'
stamp() { echo "$(date -d '+9 hours' '+%F %T') KST $*"; }
LOG=outputs/cvpr27/ratio_sweep/logs/tyr_u45_local.log
S=outputs/cvpr27/slim_tyr_u45
until grep -q "all done" "$LOG" 2>/dev/null; do
  if grep -qE "Traceback|AssertionError" "$LOG" 2>/dev/null; then stamp "tyr_u45 build FAILED: $(tail -n 2 "$LOG" | tr '\n' ' ')"; exit 1; fi
  sleep 120
done
stamp "tyr_u45 build finished"
removed=$(.venv/bin/python -c "import json,sys; m=json.load(open(sys.argv[1])); \
print(m['params']['removed'] + 1048576 * len(m.get('attn_placeholder_layers', [])), m.get('attn_placeholder_layers', []))" "$S/slim_meta.json")
stamp "tyr_u45 removed (+placeholders) / placeholder layers: $removed"
[ "${removed%% *}" = 2974777344 ] || { stamp "budget mismatch, not pushing"; exit 1; }
ssh -O check neuron-dm > /dev/null 2>&1 || { stamp "datamover socket down -- push needs the user to reopen it"; exit 2; }
rsync -a --exclude slim_meta.json "$S/" neuron-dm:/scratch/e1997a06/outputs/cvpr27/slim_tyr_u45/ 2>&1 | grep -vE "$F"
L=$(stat -c %s "$S/slim_state.pt")
R=$(ssh -o BatchMode=yes neuron-dm "stat -c %s /scratch/e1997a06/outputs/cvpr27/slim_tyr_u45/slim_state.pt" 2>/dev/null | grep -vE "$F")
[ "$L" = "$R" ] || { stamp "slim_state size mismatch local $L remote $R"; exit 1; }
rsync -a "$S/slim_meta.json" neuron-dm:/scratch/e1997a06/outputs/cvpr27/slim_tyr_u45/slim_meta.json 2>&1 | grep -vE "$F"
stamp "tyr_u45 pushed ($L bytes)"
rm -rf outputs/cvpr27/tyr_u45/supernet_r1 outputs/cvpr27/tyr_u45/supernet_r2 outputs/cvpr27/tyr_u45/supernet_r3
stamp "deleted tyr_u45 supernet_r1-r3"
holders=$(ssh -o BatchMode=yes neuron-dm "squeue -u e1997a06 -h -n eval_a100_ratio,eval_a100_ratio2,eval_a100_r2s -o %i" 2>/dev/null | grep -vE "$F" | wc -l)
if [ "$holders" -eq 0 ]; then
  ssh -O check neuron > /dev/null 2>&1 || { stamp "no holder queued and login socket down -- need it to submit"; exit 3; }
  ssh -o BatchMode=yes neuron "cd /scratch/e1997a06/project/alpamayo-model-compression && \
    sbatch --job-name=eval_a100_r2s --gres=gpu:2 --cpus-per-task=16 --mem=120G --time=05:00:00 \
      --export=ALL,A100_QUEUE=/scratch/e1997a06/outputs/cvpr27/a100/queue_ratio cvpr27/code/eval_a100.sbatch" 2>&1 | grep -vE "$F"
  stamp "submitted a 2-card holder for the tyr_u45 lines"
else
  stamp "$holders holder(s) already queued/running -- they will claim tyr_u45"
fi
