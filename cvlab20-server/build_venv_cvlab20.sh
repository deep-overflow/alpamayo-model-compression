#!/bin/bash
# Build the venv on cvlab20. Adapted from experiments/transfer/build_venv_neuron.sh --
# same pinned versions, same prebuilt flash-attn, different destination rules.
#
# Run it ON cvlab20:
#   bash /home/cvlab20/project/chan/alpamayo-model-compression/../build_venv_cvlab20.sh
# or from cvlab21:
#   ssh cvlab20@cvlab20 'bash -s' < cvlab20-server/build_venv_cvlab20.sh
#
# Why the venv is BUILT rather than copied: torch is pinned to +cu128 and the box's
# driver is 570.133.07 / CUDA 12.8, which matches -- but a copied venv also drags 8 GB of
# wheels compiled against another machine's libc and CUDA runtime. Building is the same
# rule the NEURON transfer follows.
#
# Everything uv writes goes to /mnt/dataset1: the venv itself, uv's cache, and uv's
# managed CPython. The home partition is at 94% (53 GB free) and torch alone unpacks to
# ~10k files -- putting any of this in $HOME is what the "no heavy files in home" rule
# exists to prevent.
set -euo pipefail

CHAN=/mnt/dataset1/chan
REPO=/home/cvlab20/project/chan/alpamayo-model-compression
VENV=$CHAN/venvs/alpamayo-mc

export UV_CACHE_DIR=$CHAN/.uv-cache
export UV_PYTHON_INSTALL_DIR=$CHAN/.uv-python
export PATH=$HOME/.local/bin:$PATH
mkdir -p "$CHAN/venvs" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"

command -v uv >/dev/null || python3 -m pip install --user --quiet uv
echo "uv $(uv --version)"

# 3.12 because pyproject pins requires-python ==3.12.*; the system python is 3.10, so
# uv fetches its own managed interpreter (into /mnt/dataset1, per the env above)
[ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV"
P=$VENV/bin/python

echo "=== 1/4  pinned deps (identical versions to cvlab21) ==="
uv pip install --python "$P" -r "$REPO/experiments/transfer/requirements_neuron.txt"

echo "=== 2/4  flash-attn 2.8.3, prebuilt wheel ==="
# config.json asks for attn_implementation=flash_attention_2, so this is required, not
# optional. Installing from sdist would compile for 30+ minutes and need a CUDA
# toolchain; cp312 / cu12torch2.8 / cxx11abiTRUE match this venv exactly.
uv pip install --python "$P" --no-deps \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

echo "=== 3/4  alpamayo1_5 @ f42e594 (the commit alpasim uses) ==="
uv pip install --python "$P" --no-deps "git+https://github.com/NVlabs/alpamayo1.5.git@f42e594"

echo "=== 4/4  this repo, editable ==="
uv pip install --python "$P" --no-deps -e "$REPO"

# the repo addresses its interpreter as <repo>/.venv/bin/python everywhere
[ -e "$REPO/.venv" ] || ln -s "$VENV" "$REPO/.venv"
[ -e "$REPO/outputs" ] || ln -s "$CHAN/outputs" "$REPO/outputs"

echo "=== versions ==="
"$P" - <<'PY'
import sys

import flash_attn
import torch
import transformers

print("python      ", sys.version.split()[0])
print("torch       ", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("flash_attn  ", flash_attn.__version__)
print("cuda avail  ", torch.cuda.is_available(), torch.cuda.device_count(), "devices")
import alpamayo1_5
print("alpamayo1_5 ", alpamayo1_5.__file__)
PY

echo
echo "venv at $VENV, symlinked as $REPO/.venv"
du -sh "$VENV" "$UV_CACHE_DIR" 2>/dev/null || true
echo "home partition:"; df -h / | tail -1
