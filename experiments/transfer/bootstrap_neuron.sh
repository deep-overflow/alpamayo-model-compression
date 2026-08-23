#!/bin/bash
# Target-side setup on NEURON, run once from the LOGIN node after push_neuron.sh lands.
#
#   ssh <USER>@neuron.ksc.re.kr
#   cd /scratch/$USER/project/alpamayo-model-compression
#   bash experiments/transfer/bootstrap_neuron.sh
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
No .venv yet. The source box ran torch 2.8.0+cu128 under uv; NEURON's CUDA stack is
site-managed, so build against what `module avail` actually offers rather than pinning
cu128 blindly. Either route works -- the runners only need `$REPO/.venv/bin/python`
and `$REPO/.venv/bin/torchrun` to exist:

  # uv, if it is installed or you can install it to $HOME
  cd REPO_PLACEHOLDER && uv sync
  uv pip install "git+https://github.com/NVlabs/alpamayo1.5.git@f42e594"

  # conda, then symlink so the launchers find it
  module load <cuda/python module>
  conda create -p REPO_PLACEHOLDER/.venv python=3.11 -y
  conda activate REPO_PLACEHOLDER/.venv
  pip install -e . && pip install "git+https://github.com/NVlabs/alpamayo1.5.git@f42e594"

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
echo "next: source env.sh && python experiments/transfer/preflight.py --ckpt outputs/slim_coc_u55_v2"
