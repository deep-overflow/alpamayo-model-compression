#!/bin/bash
# Launcher for run_pathway.py.
#
# run_retry_host.sh documents HF_HUB_CACHE in CLAUDE.md but does not actually export
# it, and the weights live only in the shared cache on /mnt/nvme1n1 while the token
# lives in $HOME -- so both have to be set here or the run tries to re-download 22 GB.
#
# Usage: bash experiments/head_analysis/run_pathway.sh <max_attempts> [args for run_pathway.py]
set -u
MAX=${1:-20}
shift
SCRIPT=run_pathway.py
if [ "${1:-}" = "--stage2" ]; then
  SCRIPT=run_pathway2.py
  shift
fi
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV=/mnt/nvme1n1/ad_vla/venvs/alpamayo-mc/bin/python
SHARED_OUT=/home/cvlab21/project/chan/alpamayo-model-compression/outputs

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF-expandable_segments:True}
export HF_HOME=$HOME/.cache/huggingface
export HF_HUB_CACHE=/mnt/nvme1n1/ad_vla/cache/hub
export HF_HUB_ENABLE_HF_TRANSFER=0
HF_TOKEN=$("$VENV" -c "
import configparser, os
cp = configparser.ConfigParser()
cp.read(os.path.expanduser('~/.cache/huggingface/stored_tokens'))
print(cp['full_right']['hf_token'])")
export HF_TOKEN

for i in $(seq 1 "$MAX"); do
  echo "$(date '+%H:%M:%S') attempt $i/$MAX"
  if "$VENV" "$HERE/$SCRIPT" --outputs-root "$SHARED_OUT" "$@"; then
    echo "$(date '+%H:%M:%S') completed"
    exit 0
  fi
  echo "$(date '+%H:%M:%S') attempt $i failed"
  sleep 60
done
echo "gave up after $MAX attempts"
exit 1
