#!/bin/bash
# Push the minimum needed to continue KI-LoRA recovery training on another box.
#
# Usage: bash experiments/transfer/push.sh <user@host> <remote_ad_vla_root> [tier ...]
#   e.g. bash experiments/transfer/push.sh cvlab18 /mnt/data/ad_vla core resume
#
# <remote_ad_vla_root> is the target's equivalent of /mnt/nvme1n1/ad_vla. It does not
# have to sit at the same absolute path -- bootstrap.sh wires AD_VLA_DATA / HF_HUB_CACHE
# to wherever it lands, which is why sample_cache.AV now reads that env var.
#
# Tiers (default: core):
#   core    pinned Alpamayo snapshot + Cosmos tokenizer/configs + train/ood/probe
#           sample caches + eval_sets/recovery_sets manifests + every slim_meta.json
#   resume  the whole run dir of $RESUME_RUNS (state_last.pt + both adapters), so the
#           target continues mid-run and still holds the best-so-far checkpoint
#   openloop val_500 + test_500 sample caches, if the target also runs open-loop eval
#   cosmos  the Cosmos-Reason2-8B safetensors, which `core` deliberately skips
#   venv    the built .venv as-is; prefer bootstrap.sh's uv rebuild
#
# LIST_ONLY=1 resolves every file list and prints what would move, with byte totals,
# without contacting the target at all. Run that first. DRY_RUN=1 does contact it, but
# passes --dry-run to rsync.
#
# What deliberately does NOT travel:
#   slim_state.pt (14 GB/arm)  -- slim_meta.json reconstructs it bit-for-bit
#                                 (slim_lib.load_slim, verified tensor-by-tensor)
#   the code                   -- branch lingoqa-reasoning-probe is on origin at this
#                                 commit; the target clones it. Check `git status` for
#                                 anything uncommitted you also need.
#   raw camera chunk zips      -- training reads only the pre_processed npz caches
set -euo pipefail

if [ -n "${LIST_ONLY-}" ]; then
  HOST=${1-'(list-only)'}
  DEST=${2-'(list-only)'}
  [ $# -ge 2 ] && shift 2 || shift $#
else
  HOST=${1:?usage: push.sh <user@host> <remote_ad_vla_root> [tier ...]}
  DEST=${2:?usage: push.sh <user@host> <remote_ad_vla_root> [tier ...]}
  shift 2
fi
TIERS=${*:-core}

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SRC=${AD_VLA_SRC-/mnt/nvme1n1/ad_vla}
HUB=$SRC/cache/hub
PRE=$SRC/data/physicalai_av/pre_processed
OUT=$(readlink -f "$REPO/outputs")
PY=${PUSH_PY-$REPO/.venv/bin/python}
MODEL_REV=$(grep -oP 'MODEL_REV = "\K[0-9a-f]+' "$REPO/experiments/head_analysis/slim_lib.py")
RESUME_RUNS=${RESUME_RUNS-recover_coc_u55}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

RSYNC=(rsync -aL --info=progress2 --partial --human-readable)
[ -n "${DRY_RUN-}" ] && RSYNC+=(--dry-run)

has() { grep -qw "$1" <<< "$TIERS"; }
step() { echo; echo "=== $* ==="; }

# In LIST_ONLY mode every transfer becomes a printed command, so the file lists below
# are still resolved against the real filesystem and can be inspected before anything
# crosses the network.
run() {
  if [ -n "${LIST_ONLY-}" ]; then
    echo "  would run: $*"
  else
    "$@"
  fi
}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lists.sh
. "$HERE/lists.sh"

echo "source   $SRC"
echo "target   $HOST:$DEST"
echo "tiers    $TIERS"

if has core; then
  step "HF cache subset -> $DEST/cache/hub  (Alpamayo rev ${MODEL_REV:0:8})"
  hub_list > "$TMP/hub.txt"
  echo "  $(wc -l < "$TMP/hub.txt") files, $(total_of "$HUB" "$TMP/hub.txt")"
  run ssh "$HOST" "mkdir -p '$DEST/cache/hub'"
  run "${RSYNC[@]}" --files-from="$TMP/hub.txt" "$HUB/" "$HOST:$DEST/cache/hub/"

  step "train + ood sample caches -> $DEST/data/physicalai_av/pre_processed"
  echo "  $(du -shL "$PRE/train" "$PRE/ood" | tr '\n' ' ')"
  run ssh "$HOST" "mkdir -p '$DEST/data/physicalai_av/pre_processed'"
  run "${RSYNC[@]}" "$PRE/train" "$PRE/ood" "$HOST:$DEST/data/physicalai_av/pre_processed/"

  step "probe subset of the eval cache (238 of 18,868 clips)"
  sample_list eval "$OUT/recovery_sets/val_official_238.parquet" > "$TMP/probe.txt"
  echo "  $(wc -l < "$TMP/probe.txt") files, $(total_of "$PRE" "$TMP/probe.txt")"
  run "${RSYNC[@]}" --files-from="$TMP/probe.txt" "$PRE/" \
      "$HOST:$DEST/data/physicalai_av/pre_processed/"

  step "manifests + slim recipes -> $DEST/outputs/chan"
  echo "  $(du -shL "$OUT/eval_sets" "$OUT/recovery_sets" | tr '\n' ' ')"
  run ssh "$HOST" "mkdir -p '$DEST/outputs/chan'"
  run "${RSYNC[@]}" "$OUT/eval_sets" "$OUT/recovery_sets" "$HOST:$DEST/outputs/chan/"
  # One list, one rsync: --files-from creates the per-config directories itself, so a
  # box with 50-odd recipes does not open 50-odd ssh connections to mkdir them.
  recipe_list > "$TMP/slims.txt"
  echo "  $(grep -c 'slim_meta.json$' "$TMP/slims.txt") recipes, $(total_of "$OUT" "$TMP/slims.txt")"
  run "${RSYNC[@]}" --files-from="$TMP/slims.txt" "$OUT/" "$HOST:$DEST/outputs/chan/"
fi

if has resume; then
  for run_id in $RESUME_RUNS; do
    step "resume state for $run_id"
    echo "  $(du -shL "$OUT/$run_id" 2>/dev/null || echo '(absent)')"
    run ssh "$HOST" "mkdir -p '$DEST/outputs/chan/$run_id'"
    run "${RSYNC[@]}" "$OUT/$run_id/" \
        "$HOST:$DEST/outputs/chan/$run_id/"
  done
fi

if has openloop; then
  step "val_500 + test_500 sample caches"
  sample_list eval "$OUT/eval_sets/val_500.parquet" > "$TMP/val500.txt"
  echo "  val_500: $(wc -l < "$TMP/val500.txt") files, $(total_of "$PRE" "$TMP/val500.txt")"
  echo "  test:    $(du -shL "$PRE/test")"
  run "${RSYNC[@]}" --files-from="$TMP/val500.txt" "$PRE/" \
      "$HOST:$DEST/data/physicalai_av/pre_processed/"
  run "${RSYNC[@]}" "$PRE/test" "$HOST:$DEST/data/physicalai_av/pre_processed/"
fi

if has venv; then
  step "prebuilt .venv -- only if the target is also CUDA 12.8"
  echo "  $(du -shL "$REPO/.venv" 2>/dev/null || echo '(absent)')"
  run "${RSYNC[@]}" "$REPO/.venv/" "$HOST:$DEST/venvs/alpamayo-mc/"
fi

echo
if [ -n "${LIST_ONLY-}" ]; then
  echo "LIST_ONLY: nothing was transferred."
else
  echo "done. next: ssh $HOST, then bootstrap.sh -- see experiments/transfer/MANIFEST.md"
fi
