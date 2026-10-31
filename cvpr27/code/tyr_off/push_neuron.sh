#!/bin/bash
# Stage the tyr_off build on NEURON (needs the neuron-dm OTP socket):
#   code     this checkout -> /scratch/e1997a06/project/amc-tyr (its own copy; outputs -> /scratch/.../outputs)
#   calib    the 100 calib_100 clip caches -> datasets/physicalai_av/pre_processed/calib/samples
# Then on the login node: sbatch /scratch/e1997a06/project/amc-tyr/cvpr27/code/tyr_off/build_tyr_off.sbatch
set -euo pipefail
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
D=/scratch/e1997a06
rsync -a --delete --exclude 'outputs' --exclude '.venv' --exclude '.git' --exclude '.claude' \
  --exclude '__pycache__' --exclude 'logs' "$SRC/" "neuron-dm:$D/project/amc-tyr/"
ssh -o BatchMode=yes neuron-dm "ln -sfn $D/outputs $D/project/amc-tyr/outputs && mkdir -p $D/project/amc-tyr/logs \
  $D/datasets/physicalai_av/pre_processed/calib/samples"
rsync -a --ignore-existing /mnt/nvme1n1/ad_vla/data/physicalai_av/pre_processed/calib/samples/ \
  "neuron-dm:$D/datasets/physicalai_av/pre_processed/calib/samples/"
ssh -o BatchMode=yes neuron-dm "ls $D/datasets/physicalai_av/pre_processed/calib/samples | wc -l; \
  du -s --apparent-size -BM $D/project/amc-tyr; find $D/project/amc-tyr | wc -l"
