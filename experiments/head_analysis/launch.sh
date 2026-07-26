#!/bin/bash
# Launch the head analysis run using the HF token that has dataset access.
set -e
cd /workspace/alpamayo-model-compression
export HF_TOKEN=$(/workspace/alpamayo1.5/.venv/bin/python -c "
import configparser
cp = configparser.ConfigParser()
cp.read('/home/cvlab21/.cache/huggingface/stored_tokens')
print(cp['full_right']['hf_token'])")
export CUDA_VISIBLE_DEVICES=${GPU:-0}
exec /workspace/alpamayo1.5/.venv/bin/python experiments/head_analysis/run_analysis.py "$@"
