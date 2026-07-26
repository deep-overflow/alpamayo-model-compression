#!/bin/bash
# Run the stage profile on 4 idle GPUs in parallel.
#
# This box has two different GPU models, so latency is only comparable within a
# model. run_retry.sh pins CUDA_DEVICE_ORDER=PCI_BUS_ID so --gpu N matches
# `nvidia-smi -i N`; without it the two orderings differ and --gpu lands on the
# wrong card. GPUs 0-3 are RTX PRO 5000 Blackwell, 4-7 are RTX 5880 Ada. We run the
# same clip shard on one of each so the two hardware profiles are directly
# comparable, and a second shard on a same-model GPU as a noise replicate.
# FLOPs are hardware-independent, so only the first run measures them.
cd /workspace/alpamayo-model-compression
LOG=${CLAUDE_JOB_DIR:-/tmp}/tmp
mkdir -p "$LOG"

run() {  # run <gpu> <exp_id> <offset> [extra args...]
  local gpu=$1 exp=$2 off=$3; shift 3
  bash experiments/head_analysis/run_retry.sh 30 \
    experiments/head_analysis/profile_stages.py \
    --gpu "$gpu" --exp-id "$exp" --clip-offset "$off" \
    --num-clips 12 --warmup 1 "$@" > "$LOG/$exp.log" 2>&1 &
  echo "launched $exp on cuda:$gpu (pid $!)"
}

run 0 profile_blackwell_a 0
run 5 profile_ada_a       0  --no-flops
run 1 profile_blackwell_b 13 --no-flops
run 7 profile_ada_b       13 --no-flops

wait
echo "all runs finished"
for f in "$LOG"/profile_*.log; do
  echo "=== $f ==="
  tail -3 "$f"
done
