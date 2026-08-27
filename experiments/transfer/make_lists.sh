#!/bin/bash
# Materialise the three --files-from lists so a hand-run rsync gets the reduced set
# without sourcing anything. Writes to outputs/transfer_lists/.
#
#   bash experiments/transfer/make_lists.sh
#
# hub.txt    relative to /mnt/nvme1n1/ad_vla/cache/hub
# probe.txt  relative to /mnt/nvme1n1/ad_vla/data/physicalai_av/pre_processed
# slims.txt  relative to <repo>/outputs
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_SRC=${CODE_SRC-/home/cvlab21/project/chan/alpamayo-model-compression}
SRC=${AD_VLA_SRC-/mnt/nvme1n1/ad_vla}
HUB=$SRC/cache/hub
PRE=$SRC/data/physicalai_av/pre_processed
OUT=$(readlink -f "$CODE_SRC/outputs")
PY=${PUSH_PY-$CODE_SRC/.venv/bin/python}
MODEL_REV=$(grep -oP 'MODEL_REV = "\K[0-9a-f]+' "$CODE_SRC/experiments/head_analysis/slim_lib.py")
TIERS=${TIERS-core}
DEST_DIR=$OUT/transfer_lists

has() { grep -qw "$1" <<< "$TIERS"; }
# shellcheck source=lists.sh
. "$HERE/lists.sh"

mkdir -p "$DEST_DIR"
hub_list                                                    > "$DEST_DIR/hub.txt"
sample_list eval "$OUT/recovery_sets/val_official_238.parquet" > "$DEST_DIR/probe.txt"
recipe_list                                                 > "$DEST_DIR/slims.txt"

for f in hub probe slims; do
  case $f in
    hub)   root=$HUB ;;
    probe) root=$PRE ;;
    slims) root=$OUT ;;
  esac
  printf '%-10s %5d files  %s\n' "$f.txt" \
      "$(wc -l < "$DEST_DIR/$f.txt")" "$(total_of "$root" "$DEST_DIR/$f.txt")"
done
echo
echo "written to $DEST_DIR"
