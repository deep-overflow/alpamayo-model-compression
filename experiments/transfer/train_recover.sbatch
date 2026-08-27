#!/bin/bash
#SBATCH --job-name=recover
#SBATCH --partition=amd_a100nv_8
#SBATCH --comment=field=efficientai;appl=pytorch
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# KI-LoRA recovery training on NEURON. Submit from the LOGIN node, never the Datamover.
#
#   cd /scratch/$USER/project/alpamayo-model-compression
#   sinfo                                     # pick a partition, fill in --partition
#   sbatch experiments/transfer/train_recover.sbatch --ckpt outputs/slim_coc_u55_v2 \
#          --exp-id recover_coc_u55 --steps 1200
#
# Anything after the script name is passed straight through to train_recover.py.
#
# Two things to get right before the first submit:
#
# 1. GPU memory. Each rank holds the full slim model plus LoRA and peaked at ~43 GB on
#    the source box's 48 GB cards, so the four cas_v100* partitions (32 GB) will OOM at
#    load. NEURON's usable partitions, from `scontrol show node`:
#
#      amd_a100nv_8   A100-80GB x8, 8 nodes   <- default here; --gres=gpu:4 reproduces
#                                                the source recipe exactly (world 4, accum 4)
#      amd_h200nv_8   H200-141GB x8, 3 nodes  <- fastest, more contended
#      amd_a100_4     A100-80GB x4, 1 node
#      amd_h100_2     H100-94GB  x2, 3 nodes  <- only 2 GPUs, so accum becomes 8
#
#    All cap at 2-00:00:00.
# 2. --mem is mandatory in practice. NEURON's default works out to ~481 MB per CPU,
#    which gave job 890909 just 15.4 GB for the whole node -- and each of the four ranks
#    stages an 11B bf16 model through CPU memory before it reaches the GPU, so it was
#    OOM-killed at 15.0 GB RSS. 240 GB is 60 GB per rank and a quarter of a 962 GB node,
#    proportionate to taking 4 of its 8 GPUs.
# 3. --comment is mandatory on NEURON and must be exactly
#    "field=<field>;appl=<program>" -- a bare value is rejected at submit time with
#    "option is required". `showappl` on glogin01-03 prints the allowed values.
#    efficientai = "Efficient & Scalable AI Systems, Quantization, Edge AI", which is
#    what this repo does; `aim` (Autonomous Driving & AI Mobility) fits the model's
#    domain if you would rather classify by application. appl=pytorch rather than
#    pytorch-ddp because train_recover.py does manual data parallelism over torchrun,
#    not the DDP wrapper (its reducer cannot trace the vlm -> detach -> expert path).
#
# Wall clock on 4x48 GB Blackwell was 6.8 s/step plus a 14.5 min probe every 150 steps,
# so 1,200 steps took ~4.6 h. --time above is deliberately generous; the trainer writes
# state_last.pt at every probe and defaults to --resume auto, so a timeout is recoverable
# by resubmitting the identical command.
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# env.sh is written by bootstrap_neuron.sh and carries AD_VLA_DATA / HF_HUB_CACHE /
# HF_HOME / ALPAMAYO_REPO for this box's layout.
# shellcheck disable=SC1091
source env.sh

# wandb: the key lives in ~/.config/wandb/recovery_key, same convention as the source
# box's run_ddp_retry.sh. api.wandb.ai answers on 443 from the login and Datamover
# nodes, but a compute node's egress is not guaranteed -- so probe it and fall back to
# offline rather than letting wandb.init stall on a dead socket. Either way
# wandb_init() swallows failures, and metrics.json is written regardless.
if [ -r "$HOME/.config/wandb/recovery_key" ]; then
  WANDB_API_KEY=$(cat "$HOME/.config/wandb/recovery_key")
  export WANDB_API_KEY
fi
if [ -z "${WANDB_MODE-}" ]; then
  if timeout 8 /bin/bash -c 'cat < /dev/null > /dev/tcp/api.wandb.ai/443' 2>/dev/null; then
    export WANDB_MODE=online
    echo "wandb: online (api.wandb.ai reachable from $(hostname))"
  else
    export WANDB_MODE=offline
    echo "wandb: offline -- no egress from $(hostname); sync later with"
    echo "       module load python; wandb sync wandb/offline-run-*  (from a login node)"
  fi
fi

NP=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
# Account cap: at most 4 A100s. --gres above asks for exactly that; this guard catches
# an edited --gres or a node handing back more than was requested.
if [ "$NP" -gt 4 ]; then
  echo "refusing: $NP GPUs exceeds the 4-A100 allowance for this account." >&2
  exit 1
fi
# Global batch is world_size x accum and the published u55 arms used 16. Holding it
# constant is what makes a NEURON run comparable to them; a different global batch is
# a different recipe, not the same one on new hardware.
if [ $((16 % NP)) -ne 0 ]; then
  echo "refusing: $NP GPUs cannot hold global batch 16 exactly (16 % $NP != 0)." >&2
  echo "Use 1, 2, 4, 8 or 16 GPUs, or pass --accum yourself and note the change." >&2
  exit 1
fi
ACCUM=$((16 / NP))

# One intra-op thread per core per rank would put 4x32 busy-waiting threads on 32
# cores; the same trap cost alpasim 2.3x wall-clock before DRIVER_OMP_THREADS existed.
export OMP_NUM_THREADS=${OMP_NUM_THREADS-$(( ${SLURM_CPUS_PER_TASK:-32} / NP ))}

GPUS=$(seq -s' ' 0 $((NP - 1)))
echo "node $(hostname)  gpus $NP  accum $ACCUM  global_batch 16  omp $OMP_NUM_THREADS  mem ${SLURM_MEM_PER_NODE:-?}M"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

srun --unbuffered "$ALPAMAYO_REPO/.venv/bin/torchrun" \
    --standalone --nproc_per_node="$NP" \
    experiments/recovery/train_recover.py \
    --accum "$ACCUM" \
    "$@" \
    --gpus $GPUS
