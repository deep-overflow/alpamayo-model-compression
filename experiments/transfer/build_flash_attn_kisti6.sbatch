#!/bin/bash
#SBATCH --job-name=build_fa
#SBATCH --partition=5thEx
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --comment=field=efficientai;appl=pytorch
#SBATCH --output=logs/%x-%j.out
#
# Compile flash-attn 2.8.3 for the Hangang venv (cvpr27/plans/2026-11-05_kisti6-gh200-site.md §2.4): no aarch64
# wheel exists, and model config.json asks for flash_attention_2, so from_pretrained needs the package even
# though evaluation switches to sdpa. Submit it after build_venv_kisti6.sh, from the repo on /scratch (glogin).
# Hangang's submit filter rejects --cpus-per-task / --mem on single-node GPU jobs (they follow the GPU count)
# unless the job is --exclusive, so this one asks for one GPU and nothing else.
# sm_90 only (GH200) and nvcc 12.9 to match torch 2.8.0+cu129. The wheel is kept in $SCRATCH/wheels.
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
# shellcheck disable=SC1091
source /etc/profile > /dev/null 2>&1   # before set -u: the site profile reads unset variables
module load cuda/12.9_26.5 gcc-native/13
set -euo pipefail
SCRATCH=/scratch/$USER
P=$SLURM_SUBMIT_DIR/.venv/bin/python
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=$SCRATCH/.uv-cache
export CUDA_HOME=${CUDA_HOME:?cuda module did not set CUDA_HOME}
# gcc-native/13 provides gcc/g++ but leaves c++/cc on the system's gcc 7.5, which torch's headers reject
export CC=gcc CXX=g++
export TORCH_CUDA_ARCH_LIST=9.0 FLASH_ATTN_CUDA_ARCHS=90 FLASH_ATTENTION_FORCE_BUILD=TRUE
# A 1-GPU job gets 195 GB; 36 parallel nvcc jobs peaked at 199.5 GB and were OOM-killed (job 1221236, 36/73 objects),
# so ~5.5 GB each -> 16 stays near 90 GB.
export MAX_JOBS=${MAX_JOBS:-16} NVCC_THREADS=2
echo "node $(hostname) $(uname -m), nvcc: $(nvcc --version | tail -1), gcc: $(gcc --version | head -1)"
echo "compute-node internet: github $(curl -s -o /dev/null -m 10 -w '%{http_code}' https://github.com || echo FAIL)," \
     "hf $(curl -s -o /dev/null -m 10 -w '%{http_code}' https://huggingface.co || echo FAIL)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
B=$SCRATCH/fa_build
rm -rf "$B" && mkdir -p "$B" && tar xzf "$SCRATCH/wheels/flash_attn-2.8.3.tar.gz" -C "$B"
cd "$B"/flash_attn-2.8.3
"$P" -m pip --version > /dev/null 2>&1 || uv pip install --python "$P" pip
time "$P" -m pip wheel --no-build-isolation --no-deps -w "$SCRATCH/wheels" .
uv pip install --python "$P" --no-deps "$SCRATCH"/wheels/flash_attn-2.8.3-*.whl
cd "$SLURM_SUBMIT_DIR"
"$P" - <<'PY'
import flash_attn
import torch
from flash_attn import flash_attn_func

q = torch.randn(1, 128, 16, 128, device="cuda", dtype=torch.bfloat16)  # (B, S, H, D)
o = flash_attn_func(q, q, q, causal=True)  # (1, 128, 16, 128)
ref = torch.nn.functional.scaled_dot_product_attention(
    q.transpose(1, 2), q.transpose(1, 2), q.transpose(1, 2), is_causal=True).transpose(1, 2)
print("flash_attn", flash_attn.__version__, "torch", torch.__version__, torch.version.cuda,
      "| max |fa - sdpa|", (o - ref).abs().max().item(), "|", torch.cuda.get_device_name(0))
PY
