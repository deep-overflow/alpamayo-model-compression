#!/bin/bash
# Send everything KI-LoRA recovery training needs to KISTI NEURON, and nothing else.
#
# Evaluation stays on this box (see MANIFEST.md caveat 1: paired comparisons are only
# valid within one GPU architecture, and every published arm was measured on Ada here),
# so no evaluation set travels -- only the training halves and the in-training probe.
#
#   Usage:  bash experiments/transfer/push_neuron.sh <NEURON_USER> [tier ...]
#   Tiers:  code data weights recipes   (default: all four)
#           resume   the whole run dir of $RESUME_RUNS, to continue an interrupted run
#           cosmos   the Cosmos-Reason2-8B safetensors, which `weights` skips
#           evalsets val_500 + test_500 caches and the importance files an on-NEURON
#                    criterion comparison needs. Evaluation normally stays on Ada -- send
#                    this ONLY for a self-contained comparison whose reference arms are
#                    re-measured on the same NEURON cards
#                    (plans/2026-08-31_znorm11-neuron-eval.md).
#   LIST_ONLY=1 prints what would move, with byte totals, contacting nothing.
#
# NEURON authenticates with OTP, so open ONE multiplexed connection first and every
# rsync below reuses it instead of prompting again:
#
#   ssh -fNM -S ~/.ssh/cm-neuron -o ControlPersist=8h <USER>@neuron-dm.ksc.re.kr
#   bash experiments/transfer/push_neuron.sh <USER>
#   ssh -S ~/.ssh/cm-neuron -O exit <USER>@neuron-dm.ksc.re.kr    # when done
#
# Layout written on the target, per the migration guide's recommended tree:
#   /scratch/<USER>/project/alpamayo-model-compression   code
#   /scratch/<USER>/datasets/physicalai_av/pre_processed  AD_VLA_DATA points at datasets/
#   /scratch/<USER>/checkpoints/hub                       HF_HUB_CACHE
#   /scratch/<USER>/outputs                               repo `outputs` symlinks here
#
# /scratch is not backed up and files untouched for 15 days are purge candidates. This
# script only ever writes to the target, never deletes here -- verify before cleaning up.
set -euo pipefail

USER_N=${1:?usage: push_neuron.sh <NEURON_USER> [tier ...]}
shift || true
TIERS=${*:-code data weights recipes}

DM=${NEURON_DM-neuron-dm.ksc.re.kr}
SCRATCH=/scratch/$USER_N
HOST=$USER_N@$DM
CM=${NEURON_CM-$HOME/.ssh/cm-neuron}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# The code that travels is the working tree with your uncommitted changes in it, not
# this worktree -- override with CODE_SRC if you keep the checkout somewhere else.
CODE_SRC=${CODE_SRC-/home/cvlab21/project/chan/alpamayo-model-compression}
SRC=${AD_VLA_SRC-/mnt/nvme1n1/ad_vla}
HUB=$SRC/cache/hub
PRE=$SRC/data/physicalai_av/pre_processed
OUT=$(readlink -f "$CODE_SRC/outputs")
PY=${PUSH_PY-$CODE_SRC/.venv/bin/python}
MODEL_REV=$(grep -oP 'MODEL_REV = "\K[0-9a-f]+' "$CODE_SRC/experiments/head_analysis/slim_lib.py")
RESUME_RUNS=${RESUME_RUNS-recover_coc_u55}
TRAIN_CONFIGS=${TRAIN_CONFIGS-}   # empty = every slim_* recipe (51 of them, 160 MB)

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

has() { grep -qw "$1" <<< "$TIERS"; }
step() { echo; echo "=== $* ==="; }
# shellcheck source=lists.sh
. "$HERE/lists.sh"

SSH_OPTS=()
[ -S "$CM" ] && SSH_OPTS=(-o ControlPath="$CM")
RSYNC=(rsync -aL --info=progress2 --partial --human-readable)
# The code tree keeps -a WITHOUT -L: `outputs` and `.venv` are symlinks onto the data
# mount, and dereferencing them would pull 513 GB of checkpoints through the exclude
# rules. They are excluded anyway; not following links makes that belt and braces.
CODE_RSYNC=(rsync -a --info=progress2 --partial --human-readable)
if [ ${#SSH_OPTS[@]} -gt 0 ]; then
  RSYNC+=(-e "ssh -o ControlPath=$CM")
  CODE_RSYNC+=(-e "ssh -o ControlPath=$CM")
fi
if [ -n "${DRY_RUN-}" ]; then
  RSYNC+=(--dry-run)
  CODE_RSYNC+=(--dry-run)
fi
CODE_EXCLUDES=(--exclude '.venv' --exclude 'outputs' --exclude 'wandb'
               --exclude '__pycache__' --exclude '*.pyc' --exclude 'logs/'
               --exclude '.claude/worktrees')

run() {
  if [ -n "${LIST_ONLY-}" ]; then
    echo "  would run: $*"
  else
    "$@"
  fi
}

echo "source    $SRC"
echo "code      $CODE_SRC"
echo "target    $HOST:$SCRATCH"
echo "tiers     $TIERS"
if [ -S "$CM" ]; then
  echo "control   $CM (multiplexed -- no repeat OTP)"
else
  echo "control   NONE -- every step below will ask for OTP again."
  echo "          ssh -fNM -S $CM -o ControlPersist=8h $HOST"
fi

run ssh "${SSH_OPTS[@]}" "$HOST" \
    "mkdir -p $SCRATCH/{project,datasets/physicalai_av/pre_processed,checkpoints/hub,outputs}"

if has code; then
  step "code -> $SCRATCH/project/alpamayo-model-compression"
  # Working tree rather than `git clone`: the repo is private (clone would need Git
  # auth set up on the Datamover) and the tree carries uncommitted work anyway.
  # .venv is excluded because it is 8.1 GB of CUDA 12.8 wheels that will not match
  # NEURON's stack; outputs is a symlink onto the data mount and is sent separately.
  echo "  $(du -sh --exclude=.venv --exclude=outputs --exclude=wandb \
      --exclude=logs --exclude=worktrees "$CODE_SRC" 2>/dev/null | cut -f1) (excludes applied)"
  run "${CODE_RSYNC[@]}" "${CODE_EXCLUDES[@]}" \
      "$CODE_SRC/" "$HOST:$SCRATCH/project/alpamayo-model-compression/"
  # Overlay this toolkit, which lives on its own branch and may not be in CODE_SRC yet.
  run "${CODE_RSYNC[@]}" "$HERE/" \
      "$HOST:$SCRATCH/project/alpamayo-model-compression/experiments/transfer/"
  # And the portability patches, for the same reason: CODE_SRC is checked out on the
  # research branch, so without this it ships the version of sample_cache.py that still
  # hardcodes /mnt/nvme1n1 -- which does not exist on the target, so every sample load
  # fails. Add a line here for any further file this branch patches outside transfer/.
  REPO_WT=$(cd "$HERE/../.." && pwd)
  PATCHED=(experiments/evaluation/sample_cache.py
           experiments/head_analysis/run_retry_host.sh
           experiments/recovery/run_ddp_retry.sh)
  echo "  overlaying ${#PATCHED[@]} portability patches from $REPO_WT"
  ( cd "$REPO_WT" && run "${CODE_RSYNC[@]}" --relative "${PATCHED[@]/#/./}" \
      "$HOST:$SCRATCH/project/alpamayo-model-compression/" )
fi

if has data; then
  step "training data -> $SCRATCH/datasets/physicalai_av/pre_processed"
  # train: official CE half (1,200). ood: OOD-train CE half (1,271) + ood_val probe
  # half (262) -- one namespace, so it ships whole.
  echo "  $(du -shL "$PRE/train" "$PRE/ood" | tr '\n' ' ')"
  run "${RSYNC[@]}" "$PRE/train" "$PRE/ood" \
      "$HOST:$SCRATCH/datasets/physicalai_av/pre_processed/"

  step "probe subset of the eval cache (238 of 18,868 clips)"
  # The official half of the checkpoint-selection probe. The other 18,630 clips are
  # the val_500/test_500 evaluation sets, which stay here.
  sample_list eval "$OUT/recovery_sets/val_official_238.parquet" > "$TMP/probe.txt"
  echo "  $(wc -l < "$TMP/probe.txt") files, $(total_of "$PRE" "$TMP/probe.txt")"
  run "${RSYNC[@]}" --files-from="$TMP/probe.txt" "$PRE/" \
      "$HOST:$SCRATCH/datasets/physicalai_av/pre_processed/"
fi

if has evalsets; then
  step "val_500 + test_500 caches -> $SCRATCH/datasets/physicalai_av/pre_processed"
  # ood_val's 262 clips already travel inside the `data` tier's ood namespace; these two
  # are the sets `data` deliberately leaves behind (MANIFEST.md, "what does not travel").
  # namespaces differ by set: run_baseline reads `eval` for indist and `test` for test
  : > "$TMP/evalsets.txt"
  sample_list eval "$OUT/eval_sets/indist_500.parquet" >> "$TMP/evalsets.txt"
  sample_list test "$OUT/eval_sets/test_500.parquet" >> "$TMP/evalsets.txt"
  sort -u "$TMP/evalsets.txt" -o "$TMP/evalsets.txt"
  echo "  $(wc -l < "$TMP/evalsets.txt") files, $(total_of "$PRE" "$TMP/evalsets.txt")"
  run "${RSYNC[@]}" --files-from="$TMP/evalsets.txt" "$PRE/" \
      "$HOST:$SCRATCH/datasets/physicalai_av/pre_processed/"

  step "importance files for the criterion builds -> $SCRATCH/outputs"
  # importance_v2_ada is the CoC half and the reference the per-step run was normalised
  # against; importance_stepvlm_v1 carries the ten per-step VLM gradients. They must be
  # the same architecture as each other -- not as the evaluation cards.
  echo "  $(du -shL "$OUT/importance_v2_ada" "$OUT/importance_stepvlm_v1" | tr '\n' ' ')"
  run "${RSYNC[@]}" "$OUT/importance_v2_ada" "$OUT/importance_stepvlm_v1" \
      "$OUT/importance_v2" "$OUT/slim_integrated_mag" "$HOST:$SCRATCH/outputs/"
fi

if has weights; then
  step "base weights -> $SCRATCH/checkpoints/hub  (Alpamayo rev ${MODEL_REV:0:8})"
  hub_list > "$TMP/hub.txt"
  echo "  $(wc -l < "$TMP/hub.txt") files, $(total_of "$HUB" "$TMP/hub.txt")"
  run "${RSYNC[@]}" --files-from="$TMP/hub.txt" "$HUB/" "$HOST:$SCRATCH/checkpoints/hub/"
fi

if has recipes; then
  step "manifests + pruning recipes -> $SCRATCH/outputs"
  echo "  $(du -shL "$OUT/eval_sets" "$OUT/recovery_sets" | tr '\n' ' ')"
  run "${RSYNC[@]}" "$OUT/eval_sets" "$OUT/recovery_sets" "$HOST:$SCRATCH/outputs/"
  # shellcheck disable=SC2086
  recipe_list $TRAIN_CONFIGS > "$TMP/slims.txt"
  echo "  $(grep -c 'slim_meta.json$' "$TMP/slims.txt") recipes, $(total_of "$OUT" "$TMP/slims.txt")"
  run "${RSYNC[@]}" --files-from="$TMP/slims.txt" "$OUT/" "$HOST:$SCRATCH/outputs/"
fi

if has resume; then
  for run_id in $RESUME_RUNS; do
    step "resume state for $run_id"
    echo "  $(du -shL "$OUT/$run_id" 2>/dev/null || echo '(absent)')"
    run "${RSYNC[@]}" "$OUT/$run_id/" "$HOST:$SCRATCH/outputs/$run_id/"
  done
fi

echo
if [ -n "${LIST_ONLY-}" ]; then
  echo "LIST_ONLY: nothing was transferred."
else
  cat <<NEXT

transferred. Next, on the NEURON LOGIN node (not the Datamover -- it does not build
environments or submit jobs):

  ssh $USER_N@neuron.ksc.re.kr
  cd $SCRATCH/project/alpamayo-model-compression
  /bin/bash experiments/transfer/bootstrap_neuron.sh   # NEURON shadows `bash`
  source env.sh
  python experiments/transfer/preflight.py --ckpt outputs/slim_coc_u55_v2
  sinfo                                    # pick a partition with >=40 GB/GPU
  sbatch experiments/transfer/train_recover.sbatch
NEXT
fi
