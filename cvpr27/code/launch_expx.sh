#!/bin/bash
# LLM-Pruner / Tyr + expert MLP on A100 (cvpr27/plans/2026-11-05_baselines-expert-mlp.md): LP recipes, eval_a100.sbatch,
# queue_expx (60 lines), two holders, then the three Tyr recipes (weights, ~13 GB each; slim_meta.json last so a holder
# only claims a complete recipe).
cd "$(dirname "$0")/../.." || exit 1
F='Deprecation|Redirecting'
stamp() { echo "$(date -d '+9 hours' '+%F %T') KST $*"; }
ssh -O check neuron-dm > /dev/null 2>&1 || { stamp "datamover socket down"; exit 2; }
[ -n "${SKIP_SUBMIT:-}" ] || ssh -O check neuron > /dev/null 2>&1 || { stamp "login socket down"; exit 3; }
push() {  # <dir>
  rsync -a --exclude slim_meta.json "./outputs/cvpr27/$1/" "neuron-dm:/scratch/e1997a06/outputs/cvpr27/$1/" 2>&1 | grep -vE "$F"
  if [ -f "./outputs/cvpr27/$1/slim_state.pt" ]; then
    L=$(stat -c %s "./outputs/cvpr27/$1/slim_state.pt")
    R=$(ssh -o BatchMode=yes neuron-dm "stat -c %s /scratch/e1997a06/outputs/cvpr27/$1/slim_state.pt" 2>/dev/null | grep -vE "$F")
    [ "$L" = "$R" ] || { stamp "$1 slim_state size mismatch local $L remote $R"; return 1; }
  fi
  rsync -a "./outputs/cvpr27/$1/slim_meta.json" "neuron-dm:/scratch/e1997a06/outputs/cvpr27/$1/slim_meta.json" 2>&1 | grep -vE "$F"
  stamp "pushed $1"
}
for m in 75 87p5 93p75; do push "slim_lp_m24_em$m"; done
rsync -a ./cvpr27/code/eval_a100.sbatch neuron-dm:/scratch/e1997a06/project/alpamayo-model-compression/cvpr27/code/ 2>&1 | grep -vE "$F"
stamp "sbatch local $(sha256sum ./cvpr27/code/eval_a100.sbatch | cut -c1-12) remote $(ssh -o BatchMode=yes neuron-dm 'sha256sum /scratch/e1997a06/project/alpamayo-model-compression/cvpr27/code/eval_a100.sbatch' 2>/dev/null | grep -vE "$F" | cut -c1-12)"
T=$(mktemp)
{
  echo "# cvpr27/plans/2026-11-05_baselines-expert-mlp.md"
  for arm in lp_m24_em93p75 lp_m24_em87p5 lp_m24_em75 tyr_off_em93p75 tyr_off_em87p5 tyr_off_em75; do
    for s in 0 1 2 3; do echo "ol $arm test2000 $s 4"; done
    for s in 0 1 2 3; do echo "ol $arm oodtrain $s 4"; done
    for s in 0 1; do echo "ol $arm oodval $s 2"; done
  done
} > "$T"
QX=/scratch/e1997a06/outputs/cvpr27/a100/queue_expx
ssh -o BatchMode=yes neuron-dm "mkdir -p $QX/claimed $QX/running; test -e $QX/tasks.txt && echo 'queue_expx already has tasks.txt'" 2>&1 | grep -vE "$F"
rsync -a --ignore-existing "$T" "neuron-dm:$QX/tasks.txt" 2>&1 | grep -vE "$F"
rm -f "$T"
stamp "queue_expx lines: $(ssh -o BatchMode=yes neuron-dm "grep -c '^ol' $QX/tasks.txt" 2>/dev/null | grep -vE "$F")"
[ -n "${SKIP_SUBMIT:-}" ] && stamp "SKIP_SUBMIT: holders not submitted" || ssh -o BatchMode=yes neuron "cd /scratch/e1997a06/project/alpamayo-model-compression && \
  sbatch --job-name=eval_a100_expx --gres=gpu:4 --cpus-per-task=32 --mem=240G --time=08:00:00 \
    --export=ALL,A100_QUEUE=$QX,IDLE_MAX=5400 cvpr27/code/eval_a100.sbatch; \
  sbatch --job-name=eval_a100_expx2 --gres=gpu:2 --cpus-per-task=16 --mem=120G --time=08:00:00 \
    --export=ALL,A100_QUEUE=$QX,IDLE_MAX=5400 cvpr27/code/eval_a100.sbatch; \
  sleep 5; squeue -u e1997a06 -o '%.10i %.16j %.9T %.11M %.11l %R'" 2>&1 | grep -vE "$F"
for m in 93p75 87p5 75; do push "slim_tyr_off_em$m"; done
stamp "all pushed"
