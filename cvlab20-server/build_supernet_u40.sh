#!/bin/bash
# Rebuild tyr_supernet_u40 on cvlab20 (H1 of plans/2026-09-08_tyr-objective-2x2.md).
#
# The cvlab21 copy was reduced to metadata.json + summary.txt at some point -- all 648
# level files are gone, and so are u40_d01 and u40_d1. Only the rd_a/rd_b supernets still
# carry weights. Rebuilding is 48 min (the shipped summary.txt records 2866 s), not the
# 6 h the plan first estimated.
#
# Flags are read off the shipped metadata.json rather than retyped, so a rebuild that
# reproduces is a rebuild of the SAME thing. Whether it actually reproduces is not assumed:
# verify_supernet_u40.sh rebuilds slim_tyr_u40_r from it and compares the kept sets against
# the shipped checkpoint, which settles in 45 s whether the existing tyr_u40_r numbers can
# still be used as tyrK's reference.
#
# 41 GB onto a mount at 99%. Checked before starting, and the plan is to delete it once
# tyrK/tyrD are built -- make_slim only ever reads one level per module.
set -uo pipefail
GPU=${GPU:-0}
REM=/home/cvlab20/project/chan
CHAN=/mnt/dataset1/chan
cd $REM/alpamayo-model-compression || exit 1
. $REM/cvlab20-server/env.sh
mkdir -p "$CHAN/logs"

free_gb=$(df -BG --output=avail /mnt/dataset1 | tail -1 | tr -dc '0-9')
[ "$free_gb" -lt 60 ] && { echo "REFUSING: /mnt/dataset1 has only ${free_gb}G free, need ~41G + headroom"; exit 1; }

until used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null);
      [ -n "$used" ] && [ "$used" -le 16 ]; do
  echo "$(date -u '+%H:%M') gpu$GPU busy (${used:-?} MiB), waiting"
  sleep 300
done

echo "$(date -u '+%F %T') building tyr_supernet_u40 on gpu$GPU (${free_gb}G free)"
.venv/bin/python experiments/head_analysis/run_tyr_supernet.py \
  --exp-id tyr_supernet_u40 \
  --num-clips 100 --calib-manifest calib_100 --cache calib \
  --head-cut 13 --head-step 1 --mlp-cut 4898 --mlp-step 256 \
  --num-levels 9 --damp 0.01 --selection osscar \
  --gpu "$GPU" --reserve-gb 40 \
  >>"$CHAN/logs/build_supernet_u40.log" 2>&1
rc=$?
echo "$(date -u '+%F %T') run_tyr_supernet exit=$rc"
n=$(find "$CHAN/outputs/tyr_supernet_u40" -name '*.pth' 2>/dev/null | wc -l)
echo "level files: $n (expect 648)"
du -sh "$CHAN/outputs/tyr_supernet_u40" 2>/dev/null
df -h /mnt/dataset1 | tail -1
