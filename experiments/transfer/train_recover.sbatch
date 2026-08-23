#!/bin/bash
#SBATCH --job-name=recover
#SBATCH --partition=amd_a100nv_8
#SBATCH --comment=field=efficientai;appl=pytorch
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
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
# 2. --comment is mandatory on NEURON and must be exactly
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
# HF_HOME / ALPAMAYO_MC_REPO for this box's layout.
# shellcheck disable=SC1091
source env.sh

# Compute nodes usually have no outbound network; wandb_init already degrades to a
# no-op on failure, but offline mode keeps a syncable run directory instead of nothing.
export WANDB_MODE=${WANDB_MODE-offline}

NP=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
# Global batch is world_size x accum and the published u55 arms used 16. Holding it
# constant is what makes a NEURON run comparable to them; a different global batch is
# a different recipe, not the same one on new hardware.
if [ $((16 % NP)) -ne 0 ]; then
  echo "refusing: $NP GPUs cannot hold global batch 16 exactly (16 % $NP != 0)." >&2
  echo "Use 1, 2, 4, 8 or 16 GPUs, or pass --accum yourself and note the change." >&2
  exit 1
fi
ACCUM=$((16 / NP))

GPUS=$(seq -s' ' 0 $((NP - 1)))
echo "node $(hostname)  gpus $NP  accum $ACCUM  global_batch 16"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

srun --unbuffered "$ALPAMAYO_MC_REPO/.venv/bin/torchrun" \
    --standalone --nproc_per_node="$NP" \
    experiments/recovery/train_recover.py \
    --accum "$ACCUM" \
    "$@" \
    --gpus $GPUS
