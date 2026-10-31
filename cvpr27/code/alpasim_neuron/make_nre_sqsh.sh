#!/bin/bash
# nre-ga:26.04 fails in `enroot import` with "mkdir rootfs/app/run.runfiles: File exists": its WorkingDir
# /app/run.runfiles/_main goes through /app/run.runfiles, an ABSOLUTE symlink to
# /app/internal/scripts/pycena/runtime/pycena_nrm_full.runfiles, which dangles when enroot does
# `mkdir -p rootfs<WorkingDir>` from the host. Point WORKDIR at the same directory by its real path
# (metadata only, layers untouched) and convert that. The sqsh keeps the name the wizard looks up.
set -euo pipefail
SRC=nvcr.io/nvidia/nre/nre-ga:26.04
FIX=nre-ga-wdfix:26.04
REAL=/app/internal/scripts/pycena/runtime/pycena_nrm_full.runfiles/_main
OUT=/mnt/nvme1n1/ad_vla/data/alpasim/sqsh
W=/mnt/nvme1n1/ad_vla/tmp/enroot
rm -rf "$W"; mkdir -p "$W"/{cache,data,tmp,run}
export ENROOT_CACHE_PATH=$W/cache ENROOT_DATA_PATH=$W/data ENROOT_TEMP_PATH=$W/tmp ENROOT_RUNTIME_PATH=$W/run
export ENROOT_SQUASH_OPTIONS="-comp zstd -Xcompression-level 3" ENROOT_MAX_PROCESSORS=16

docker rmi nre-ga-flat:26.04 > /dev/null 2>&1 || true
printf 'FROM %s\nWORKDIR %s\n' "$SRC" "$REAL" | docker build -q -t "$FIX" - > /dev/null
docker inspect "$FIX" --format '{{len .RootFS.Layers}} layers, workdir {{.Config.WorkingDir}}, entrypoint {{json .Config.Entrypoint}}'

cd "$OUT"
echo "$(date '+%F %T') enroot import $FIX -> nre_ga_26.04.sqsh"
enroot import -o nre_ga_26.04.sqsh.part "dockerd://$FIX"
mv nre_ga_26.04.sqsh.part nre_ga_26.04.sqsh
docker rmi "$FIX" > /dev/null
rm -rf "$W"
sha256sum ./*.sqsh > SHA256SUMS
echo "$(date '+%F %T') all done"; ls -la; cat SHA256SUMS
