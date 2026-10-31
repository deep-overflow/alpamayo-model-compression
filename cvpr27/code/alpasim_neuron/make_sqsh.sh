#!/bin/bash
# Convert alpasim-base to enroot squashfs for NEURON's pyxis (wizard.run_method=SLURM).
# File names follow alpasim_wizard.utils.image_to_sqsh_basename: <name>_<tag>.sqsh with '-' -> '_'.
# The renderer image needs a WORKDIR fix first -- use make_nre_sqsh.sh for it.
set -euo pipefail
OUT=/mnt/nvme1n1/ad_vla/data/alpasim/sqsh
W=/mnt/nvme1n1/ad_vla/tmp/enroot
mkdir -p "$OUT" "$W"/{cache,data,tmp,run}
export ENROOT_CACHE_PATH=$W/cache ENROOT_DATA_PATH=$W/data ENROOT_TEMP_PATH=$W/tmp ENROOT_RUNTIME_PATH=$W/run
export ENROOT_SQUASH_OPTIONS="-comp zstd -Xcompression-level 3" ENROOT_MAX_PROCESSORS=16
cd "$OUT"
for img in alpasim-base:0.111.0; do
  f=$(basename "$img" | tr ':-' '__').sqsh
  if [ -f "$f" ]; then echo "$(date '+%F %T') skip $f"; continue; fi
  echo "$(date '+%F %T') import $img -> $f"
  enroot import -o "$f.part" "dockerd://$img"
  mv "$f.part" "$f"
  echo "$(date '+%F %T') done $f $(du -h --apparent-size "$f" | cut -f1)"
done
rm -rf "$W"
sha256sum ./*.sqsh > SHA256SUMS
echo "$(date '+%F %T') all done"; cat SHA256SUMS
