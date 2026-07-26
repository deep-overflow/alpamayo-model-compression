#!/bin/bash
# Run an analysis script, retrying while the shared GPUs are busy.
# The script itself picks a GPU by reserving memory on it, so no CUDA_VISIBLE_DEVICES here.
# Usage: bash run_retry.sh <max_attempts> <script.py> [args...]
MAX=${1:-60}
shift
cd /workspace/alpamayo-model-compression
# Without this the CUDA runtime orders devices by speed while nvidia-smi orders by
# PCI bus, so --gpu N and `nvidia-smi -i N` refer to different cards on this box.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# reduce fragmentation so a long-CoC clip does not need a fresh large block
# (respect a caller-provided value -- CUDA-graph capture needs this OFF, pass ...="")
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF-expandable_segments:True}
export HF_TOKEN=$(/workspace/alpamayo1.5/.venv/bin/python -c "
import configparser
cp = configparser.ConfigParser()
cp.read('/home/cvlab21/.cache/huggingface/stored_tokens')
print(cp['full_right']['hf_token'])")

for i in $(seq 1 "$MAX"); do
  echo "$(date '+%H:%M:%S') attempt $i/$MAX"
  if /workspace/alpamayo1.5/.venv/bin/python "$@"; then
    echo "$(date '+%H:%M:%S') completed"
    exit 0
  fi
  echo "$(date '+%H:%M:%S') attempt $i failed"
  sleep 60
done
echo "gave up after $MAX attempts"
exit 1
