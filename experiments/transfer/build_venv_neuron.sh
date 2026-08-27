#!/bin/bash
# Build the training venv on NEURON. Downloads only -- nothing is compiled, which is
# what keeps this inside the Datamover's remit (the site sanctions large downloads
# there and forbids compiles). Run it with /bin/bash, not bash; see MANIFEST caveat 8.
#
#   /bin/bash experiments/transfer/build_venv_neuron.sh
#
# Two site facts this works around:
#
#   * `module load python/3.12.4` is DEPRECATED and silently redirects to 3.14.2, which
#     violates pyproject's `requires-python = "==3.12.*"` and has no torch 2.8 wheels.
#     So the interpreter comes from uv's own managed CPython instead of the module system.
#   * uv.lock has flash-attn as an sdist and `[tool.uv] no-build-isolation-package`
#     set for it, so `uv sync` would compile it -- 30+ minutes and a CUDA toolchain.
#     config.json asks for attn_implementation=flash_attention_2, so it is required,
#     not optional; the prebuilt wheel below is the same 2.8.3 the source box runs.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SCRATCH=${SCRATCH_ROOT-/scratch/$USER}
export PATH=$HOME/.local/bin:$PATH
# Keep uv's cache and managed pythons off /home01: that quota is generous in bytes but
# only 100k inodes, and torch alone unpacks to ~10k files.
export UV_CACHE_DIR=${UV_CACHE_DIR-$SCRATCH/.uv-cache}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR-$SCRATCH/.uv-python}

cd "$REPO"
command -v uv >/dev/null || python3 -m pip install --user --quiet uv
echo "uv $(uv --version)"

P=$REPO/.venv/bin/python
[ -x "$P" ] || uv venv --python 3.12 "$REPO/.venv"

echo "=== 1/4  pinned deps (matches the source box exactly) ==="
uv pip install --python "$P" -r "$REPO/experiments/transfer/requirements_neuron.txt"

echo "=== 2/4  flash-attn 2.8.3, prebuilt ==="
# torch reports cxx11abi True, hence abiTRUE; cp312 matches the venv; cu12torch2.8
# matches torch 2.8.0+cu128.
uv pip install --python "$P" --no-deps \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

echo "=== 3/4  alpamayo1_5 @ f42e594 (the commit alpasim uses) ==="
uv pip install --python "$P" --no-deps "git+https://github.com/NVlabs/alpamayo1.5.git@f42e594"

echo "=== 4/4  this repo, editable ==="
uv pip install --python "$P" --no-deps -e "$REPO"

echo "=== versions ==="
"$P" - <<'PY'
import sys

import flash_attn
import pandas
import peft
import torch
import transformers

print("python      ", sys.version.split()[0])
print("torch       ", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("peft        ", peft.__version__)
print("flash_attn  ", flash_attn.__version__)
print("pandas      ", pandas.__version__)
PY

echo
echo "next: source env.sh && .venv/bin/python experiments/transfer/preflight.py --ckpt outputs/slim_coc_u55_v2"
