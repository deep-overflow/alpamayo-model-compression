#!/bin/bash
# Wait for a GPU with enough free memory, then run the given analysis script.
# Usage: bash wait_and_run.sh <needed_mib> <script.py> [args...]
set -e
NEED=${1:-26000}
shift
cd /workspace/alpamayo-model-compression
export HF_TOKEN=$(/workspace/alpamayo1.5/.venv/bin/python -c "
import configparser
cp = configparser.ConfigParser()
cp.read('/home/cvlab21/.cache/huggingface/stored_tokens')
print(cp['full_right']['hf_token'])")

# Retry rather than exec once: on a shared box another process can claim the memory
# between the check and the model load, so a single attempt is racy.
echo "waiting for a GPU with >= ${NEED} MiB free..."
ATTEMPT=0
while [ $ATTEMPT -lt 40 ]; do
  FREE_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
    awk -v need="$NEED" -F', ' '$2 >= need {print $1; exit}')
  if [ -n "$FREE_GPU" ]; then
    ATTEMPT=$((ATTEMPT + 1))
    echo "$(date '+%H:%M:%S') GPU $FREE_GPU free, launching (attempt $ATTEMPT)"
    if CUDA_VISIBLE_DEVICES=$FREE_GPU /workspace/alpamayo1.5/.venv/bin/python "$@"; then
      echo "$(date '+%H:%M:%S') completed"
      exit 0
    fi
    echo "$(date '+%H:%M:%S') attempt $ATTEMPT failed, retrying"
  fi
  sleep 60
done
echo "gave up after $ATTEMPT attempts"
exit 1
