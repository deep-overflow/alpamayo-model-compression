#!/bin/bash
# Multi-GPU twin of run_retry_host.sh: waits until EVERY requested card is idle, then
# launches the script under torchrun with one rank per card. A torchrun failure (busy
# card, transient NCCL error) falls back into the retry loop.
#
# Usage: bash run_ddp_retry.sh <max_attempts> "<gpu list>" <script.py> [args...]
#   e.g. bash experiments/recovery/run_ddp_retry.sh 60 "4 5 6 7" \
#            experiments/recovery/train_recover.py --ckpt ... --exp-id ...
MAX=${1:-60}
GPUS=$2
shift 2
REPO=${ALPAMAYO_REPO:-/home/cvlab21/project/chan/alpamayo-model-compression}
cd "$REPO" || exit 1
NP=$(wc -w <<< "$GPUS")
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF-expandable_segments:True}
export HF_HOME=${HF_HOME_OVERRIDE-$HOME/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE-/mnt/nvme1n1/ad_vla/cache/hub}
export HF_HUB_ENABLE_HF_TRANSFER=0
[ -f "$HOME/.config/wandb/recovery_key" ] && export WANDB_API_KEY=$(cat "$HOME/.config/wandb/recovery_key")
export HF_TOKEN=$("$REPO/.venv/bin/python" -c "
import configparser
cp = configparser.ConfigParser()
cp.read('$HOME/.cache/huggingface/stored_tokens')
print(cp['full_right']['hf_token'])")

for i in $(seq 1 "$MAX"); do
  busy=""
  for g in $GPUS; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    [ "$used" -gt 2000 ] && busy="$busy $g"
  done
  if [ -z "$busy" ]; then
    echo "$(date '+%H:%M:%S') attempt $i/$MAX (torchrun x$NP on:$GPUS)"
    if "$REPO/.venv/bin/torchrun" --standalone --nproc_per_node="$NP" "$@" --gpus $GPUS; then
      echo "$(date '+%H:%M:%S') completed"
      exit 0
    fi
    echo "$(date '+%H:%M:%S') attempt $i failed"
  else
    echo "$(date '+%H:%M:%S') attempt $i/$MAX waiting, busy:$busy"
  fi
  sleep 60
done
echo "gave up after $MAX attempts"
exit 1
