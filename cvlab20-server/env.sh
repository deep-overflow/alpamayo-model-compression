# Source this before anything on cvlab20:  . ~/project/chan/env.sh
#
# The repo reads these from the environment, so pointing them at /mnt/dataset1 is all it
# takes to run unmodified code on this box. Nothing heavy may live under $HOME here --
# the home partition is at 94%.
export ALPAMAYO_REPO=/home/cvlab20/project/chan/alpamayo-model-compression
export AD_VLA_DATA=/mnt/dataset1/chan/data
export HF_HUB_CACHE=/mnt/dataset1/chan/cache/hub
export HF_HOME=/mnt/dataset1/chan/cache
export HF_HUB_ENABLE_HF_TRANSFER=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# evaluation determinism, per CLAUDE.md -- must be set before CUDA initialises
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# The account is shared with the whole lab, so the token is NOT read from a shared
# stored_tokens file: export HF_TOKEN yourself when a gated repo is needed. The base
# weights are already in HF_HUB_CACHE, so runs that only load Alpamayo need no token.
