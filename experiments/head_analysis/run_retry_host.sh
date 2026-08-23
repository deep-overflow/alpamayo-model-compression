#!/bin/bash
# Host-side twin of run_retry.sh: the sibling alpamayo1.5 repo (and its venv) is
# gone, so experiments now run against this repo's own .venv, which has
# alpamayo1_5 installed from the same NVlabs commit the closed-loop eval used.
# Retries while the shared GPUs are busy; the script itself picks a card by
# reserving memory, so no CUDA_VISIBLE_DEVICES here.
#
# Usage: bash run_retry_host.sh <max_attempts> <script.py> [args...]
MAX=${1:-60}
shift
REPO=${ALPAMAYO_MC_REPO-/home/cvlab21/project/chan/alpamayo-model-compression}
cd "$REPO" || exit 1
# CUDA orders devices by speed while nvidia-smi orders by PCI bus, so without
# this --gpu N and `nvidia-smi -i N` mean different cards on this box
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF-expandable_segments:True}
# Token and blobs live apart: the shared cache on /mnt/nvme1n1 has no `full_right`
# section in its stored_tokens (only other members' tokens), while the 22 GB of
# weights moved there on 2026-08-06 to get off the 97%-full root partition.
# HF_HOME locates the token, HF_HUB_CACHE locates the blobs.
export HF_HOME=${HF_HOME_OVERRIDE-$HOME/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE-/mnt/nvme1n1/ad_vla/cache/hub}
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_TOKEN=$("$REPO/.venv/bin/python" -c "
import configparser
cp = configparser.ConfigParser()
cp.read('$HOME/.cache/huggingface/stored_tokens')
print(cp['full_right']['hf_token'])")

for i in $(seq 1 "$MAX"); do
  echo "$(date '+%H:%M:%S') attempt $i/$MAX"
  if "$REPO/.venv/bin/python" "$@"; then
    echo "$(date '+%H:%M:%S') completed"
    exit 0
  fi
  echo "$(date '+%H:%M:%S') attempt $i failed"
  sleep 60
done
echo "gave up after $MAX attempts"
exit 1
