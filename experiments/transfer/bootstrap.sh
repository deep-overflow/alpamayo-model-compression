#!/bin/bash
# Target-side setup, run once on the new server after push.sh has landed the data.
#
# Usage: bash experiments/transfer/bootstrap.sh <ad_vla_root> [repo_dir]
#   e.g. bash experiments/transfer/bootstrap.sh /mnt/data/ad_vla ~/alpamayo-model-compression
#
# Assumes: the repo is already cloned (branch lingoqa-reasoning-probe), uv is installed,
# and push.sh has populated <ad_vla_root>/{cache/hub,data,outputs/chan}.
#
# Writes <repo>/env.sh. Source it before any run; run_retry_host.sh and run_ddp_retry.sh
# honour the same variables, so a run launched through them picks the layout up too.
set -euo pipefail

AD_VLA=${1:?usage: bootstrap.sh <ad_vla_root> [repo_dir]}
REPO=${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
AD_VLA=$(cd "$AD_VLA" && pwd)

echo "=== layout check ==="
for d in cache/hub data/physicalai_av/pre_processed outputs/chan; do
  [ -d "$AD_VLA/$d" ] || { echo "MISSING $AD_VLA/$d -- rerun push.sh core"; exit 1; }
done
df -h "$AD_VLA" | tail -1

echo
echo "=== outputs symlink ==="
# Every runner writes to REPO/outputs/<exp_id>; on the source box that is a symlink onto
# the data mount, and keeping the same shape means no script needs a path argument.
if [ -e "$REPO/outputs" ] && [ ! -L "$REPO/outputs" ]; then
  echo "REFUSING: $REPO/outputs exists and is a real directory, not a symlink."
  echo "Move it aside first -- overwriting it would orphan whatever is in there."
  exit 1
fi
ln -sfn "$AD_VLA/outputs/chan" "$REPO/outputs"
ls -la "$REPO/outputs"

echo
echo "=== venv ==="
if [ -x "$REPO/.venv/bin/python" ]; then
  echo "already present: $("$REPO/.venv/bin/python" -c 'import torch;print(torch.__version__, torch.version.cuda)')"
else
  ( cd "$REPO" && uv sync )
  # alpamayo1_5 is not on PyPI and not in pyproject; it is pinned to the commit alpasim
  # uses, so the target must install the same one or the slim surgery will not apply.
  ( cd "$REPO" && uv pip install "git+https://github.com/NVlabs/alpamayo1.5.git@f42e594" )
fi

echo
echo "=== env.sh ==="
cat > "$REPO/env.sh" <<ENVEOF
# Written by experiments/transfer/bootstrap.sh -- source before any run.
export AD_VLA_DATA=$AD_VLA/data
export HF_HUB_CACHE=$AD_VLA/cache/hub
export HF_HOME=\${HF_HOME:-\$HOME/.cache/huggingface}
export HF_HUB_ENABLE_HF_TRANSFER=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ALPAMAYO_MC_REPO=$REPO
ENVEOF
cat "$REPO/env.sh"

echo
echo "=== HF token ==="
# HF_HOME locates the token, HF_HUB_CACHE locates the blobs -- they are separate on the
# source box and stay separate here. Only the `full_right` token has gated access.
if [ -f "$HOME/.cache/huggingface/stored_tokens" ] \
   && grep -q full_right "$HOME/.cache/huggingface/stored_tokens"; then
  echo "full_right token present"
else
  echo "NO full_right token in \$HOME/.cache/huggingface/stored_tokens."
  echo "Training reads the base weights from the local cache, so this only bites if a"
  echo "run has to re-download. Copy the source box's stored_tokens if you want one."
fi

echo
echo "=== GPUs ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || echo "no nvidia-smi"

echo
echo "next: source env.sh && python experiments/transfer/preflight.py --ckpt outputs/slim_coc_u55_v2"
