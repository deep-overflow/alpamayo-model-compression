#!/bin/bash
# GH200 site gates (cvpr27/plans/2026-11-05_kisti6-gh200-site.md §2.6) from cvlab21: four test_2000 shards 0/16
# (125 clips each) on one exclusive 5thEx node -- dense twice (G1), MoOP l0keep and tyr_off (G3); G2/G4 come from
# the same rows and the job log. Results land in outputs/cvpr27/gh200/ on Hangang; analyze with analyze_gh200_gate.py
# after pulling them. Needs the kisti6 login socket.
F='Deprecation|Redirecting'
R=/scratch/e1997a06/project/alpamayo-model-compression
Q=$R/outputs/cvpr27/gh200/queue_gate
ssh -O check kisti6 > /dev/null 2>&1 || { echo "kisti6 login socket down"; exit 2; }
ssh -o BatchMode=yes kisti6 "cd $R && mkdir -p $Q/claimed $Q/running && test ! -s $Q/tasks.txt && \
  printf '%s\n' '# cvpr27/plans/2026-11-05_kisti6-gh200-site.md section 2.6' \
    'ol dense test2000 0 16' 'ol dense test2000 0 16 rerun' \
    'ol moop_l0keep test2000 0 16' 'ol tyr_off test2000 0 16' > $Q/tasks.txt; cat $Q/tasks.txt; \
  sbatch -p 5thEx --exclusive --gres=gpu:4 --time=03:00:00 --job-name=eval_gh200_gate \
    --export=ALL,OUT_SITE=gh200,A100_QUEUE=$Q,IDLE_MAX=300 cvpr27/code/eval_a100.sbatch; \
  sleep 5; squeue -u e1997a06 -o '%.10i %.16j %.9T %.10M %.20S %R'" 2>&1 | grep -vE "$F"
