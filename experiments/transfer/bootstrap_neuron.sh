#!/bin/bash
# Target-side setup on NEURON, run once from the LOGIN node after push_neuron.sh lands.
#
#   ssh <USER>@neuron.ksc.re.kr
#   cd /scratch/$USER/project/alpamayo-model-compression
#   /bin/bash experiments/transfer/bootstrap_neuron.sh
#
# Spell out /bin/bash. NEURON defines `bash` as a shell function --
# `bash () { /bin/bash --login; }` -- which drops every argument and starts an
# interactive login shell instead, so `bash script.sh` exits 0 having run nothing.
#
# The Datamover is for transfer only -- it does not build environments or submit jobs,
# which is why this runs on the login node instead.
#
# Writes env.sh. The sbatch script sources it; so should any interactive srun.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SCRATCH=${SCRATCH_ROOT-/scratch/$USER}

echo "repo     $REPO"
echo "scratch  $SCRATCH"

echo
echo "=== layout check ==="
for d in datasets/physicalai_av/pre_processed checkpoints/hub outputs; do
  [ -d "$SCRATCH/$d" ] || { echo "MISSING $SCRATCH/$d -- rerun push_neuron.sh"; exit 1; }
done
for ns in train ood eval; do
  n=$(ls "$SCRATCH/datasets/physicalai_av/pre_processed/$ns/samples" 2>/dev/null | wc -l)
  echo "  $ns namespace: $n samples"
done
df -h "$SCRATCH" | tail -1

echo
echo "=== outputs symlink ==="
# Every runner writes REPO/outputs/<exp_id>. Keeping that shape means no script needs
# a path argument, exactly as on the source box.
if [ -e "$REPO/outputs" ] && [ ! -L "$REPO/outputs" ]; then
  echo "REFUSING: $REPO/outputs exists and is a real directory, not a symlink."
  echo "Move it aside first -- overwriting it would orphan whatever is in there."
  exit 1
fi
ln -sfn "$SCRATCH/outputs" "$REPO/outputs"
ls -la "$REPO/outputs"

echo
echo "=== python environment ==="
if [ -x "$REPO/.venv/bin/python" ]; then
  echo "already present: $("$REPO/.venv/bin/python" -c 'import torch;print(torch.__version__, torch.version.cuda)')"
else
  sed "s|REPO_PLACEHOLDER|$REPO|g" <<'ENVNOTE'
No .venv yet. Build it on the LOGIN node, matching the source box: Python 3.12
(pyproject pins requires-python = "==3.12.*") and CUDA 12.8 (torch 2.8.0+cu128).
NEURON has exact modules for both, and pypi.org / files.pythonhosted.org / huggingface.co
all answer on 443 from here.

  module load python/3.12.4 cuda/12.8
  cd REPO_PLACEHOLDER
  python -m venv .venv
  .venv/bin/pip install -U pip uv
  .venv/bin/uv sync --frozen          # honours uv.lock, so versions match the source box
  .venv/bin/pip install "git+https://github.com/NVlabs/alpamayo1.5.git@f42e594"

Do NOT `module load python/3.14.2` (the default) -- it violates the 3.12 pin.

flash-attn is the long pole: pyproject asks for >=2.8.3 and PyPI ships an sdist, so pip
will compile it. That is heavy enough that it belongs in an interactive job rather than
on the login node:

  salloc -p amd_a100nv_8 --gres=gpu:1 -t 2:00:00 --comment pytorch
  # then rerun the install inside the allocation, with MAX_JOBS=8 to bound the build

alpamayo1_5 is not on PyPI and not in pyproject. The commit is pinned to the one alpasim
uses; a different one may not match the slim surgery.
ENVNOTE
fi

echo
echo "=== env.sh ==="
cat > "$REPO/env.sh" <<ENVEOF
# Written by experiments/transfer/bootstrap_neuron.sh -- source before any run.
export AD_VLA_DATA=$SCRATCH/datasets
export HF_HUB_CACHE=$SCRATCH/checkpoints/hub
export HF_HOME=\${HF_HOME:-\$HOME/.cache/huggingface}
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_OFFLINE=\${HF_HUB_OFFLINE:-1}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ALPAMAYO_MC_REPO=$REPO
ENVEOF
cat "$REPO/env.sh"

echo
echo "=== notes ==="
cat <<'NOTES'
- HF_HUB_OFFLINE=1 is set because compute nodes usually have no outbound network and
  the pinned snapshot is already local. Unset it if a download is genuinely wanted.
- /scratch is not backed up and files untouched for 15 days are purge candidates. Pull
  adapters back to the source box as they are produced; do not leave them as the only copy.
- Do not run training here. Submit it:
    sinfo
    sbatch experiments/transfer/train_recover.sbatch --ckpt outputs/slim_coc_u55_v2 \
        --exp-id recover_coc_u55 --steps 1200
NOTES

echo
if [ -x "$REPO/.venv/bin/python" ]; then
  echo "next: source env.sh && .venv/bin/python experiments/transfer/preflight.py --ckpt outputs/slim_coc_u55_v2"
else
  echo "next: build the environment above, THEN"
  echo "      source env.sh && .venv/bin/python experiments/transfer/preflight.py --ckpt outputs/slim_coc_u55_v2"
  echo "      (preflight imports pandas/torch, so it cannot run before the venv exists)"
fi
