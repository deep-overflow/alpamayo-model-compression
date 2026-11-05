#!/bin/bash
# Build the GH200 (aarch64) venv on KISTI Hangang (cvpr27/plans/2026-11-05_kisti6-gh200-site.md §2.4).
# Downloads only, so it runs on a glogin node (aarch64 -- dm01 is x86_64 and would build the wrong arch).
# flash-attn is compiled afterwards on a compute node: build_flash_attn_kisti6.sbatch.
#
#   /bin/bash experiments/transfer/build_venv_kisti6.sh
#
# What differs from build_venv_neuron.sh, all forced by aarch64:
#   * torch 2.8.0 has no aarch64 cu128 wheel (PyPI's aarch64 torch is CPU-only); its aarch64 CUDA build is
#     cu129, so torch 2.8.0+cu129 / torchvision 0.23.0 / triton 3.4.0 come from the cu129 index and the
#     nvidia-* 12.8 pins are dropped -- torch brings its own 12.9 set (user decision 2026-11-05).
#   * flash-attn 2.8.3 publishes x86_64 wheels only -> built from source, separately. Its build-time
#     dependencies (ninja, packaging, psutil, wheel, setuptools) are installed here, and the sdist is
#     fetched here because compute-node internet is not guaranteed.
#   * uv comes as the pinned release binary: the system python3 is 3.6, too old for `pip install uv`.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SCRATCH=${SCRATCH_ROOT-/scratch/$USER}
[ "$(uname -m)" = aarch64 ] || { echo "run on a glogin node (aarch64), not $(uname -m)"; exit 1; }
export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=${UV_CACHE_DIR-$SCRATCH/.uv-cache}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR-$SCRATCH/.uv-python}
UV_VERSION=0.11.25   # the source box's uv
TORCH_INDEX=https://download.pytorch.org/whl/cu129

cd "$REPO"
if ! command -v uv > /dev/null; then
  mkdir -p "$HOME/.local/bin"
  curl -LsSf "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-aarch64-unknown-linux-gnu.tar.gz" \
    | tar xz -C "$HOME/.local/bin" --strip-components=1
fi
echo "uv $(uv --version)"

P=$REPO/.venv/bin/python
[ -x "$P" ] || uv venv --python 3.12 "$REPO/.venv"

echo "=== 1/5  torch 2.8.0+cu129 stack (aarch64) ==="
# --extra-index-url outranks the default index in uv, and the first index that has a name serves it,
# so torch/torchvision/triton come from the cu129 index and everything else from PyPI.
uv pip install --python "$P" --extra-index-url "$TORCH_INDEX" \
  "torch==2.8.0+cu129" "torchvision==0.23.0" "triton==3.4.0"

echo "=== 2/5  remaining pins (requirements_neuron.txt minus torch, torchvision, triton, nvidia-*) ==="
REQ=$SCRATCH/.req_kisti6.txt
grep -v -E '^(torch|torchvision|triton|nvidia-[A-Za-z0-9-]+)==' "$REPO/experiments/transfer/requirements_neuron.txt" > "$REQ"
uv pip install --python "$P" -r "$REQ"

echo "=== 3/5  flash-attn build deps + sdist (compiled by build_flash_attn_kisti6.sbatch) ==="
uv pip install --python "$P" ninja packaging psutil wheel setuptools
mkdir -p "$SCRATCH/wheels"
FA=$SCRATCH/wheels/flash_attn-2.8.3.tar.gz
[ -s "$FA" ] || curl -LsSf -o "$FA" \
  "$(curl -s https://pypi.org/pypi/flash-attn/2.8.3/json | "$P" -c 'import json,sys; print([u["url"] for u in json.load(sys.stdin)["urls"] if u["packagetype"]=="sdist"][0])')"
ls -la "$FA"

echo "=== 4/5  alpamayo1_5 @ f42e594 (the commit alpasim uses) ==="
uv pip install --python "$P" --no-deps "git+https://github.com/NVlabs/alpamayo1.5.git@f42e594"

echo "=== 5/5  this repo, editable ==="
uv pip install --python "$P" --no-deps -e "$REPO"

echo "=== versions ==="
"$P" - <<'PY'
import sys

import torch
import transformers

print("python      ", sys.version.split()[0])
print("torch       ", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
try:
    import triton
    print("triton      ", triton.__version__)
except ImportError as e:
    print("triton       MISSING", e)
PY
