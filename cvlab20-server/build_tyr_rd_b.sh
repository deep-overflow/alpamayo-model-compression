#!/bin/bash
# Build slim_tyr_rd_b on cvlab20 -- Tyr reconstruction calibrated on calib_rd100_b, the
# second random calibration draw, so rd_a's reading has a replicate.
#
# Only 4.6 GB of the 41 GB supernet was shipped. make_slim's tyr branch does
# `torch.load(sup / n / f"{lv}.pth")` -- one level per module, chosen by
# tyr_search_rd_b/final_config.json -- so 72 of the 648 level files are ever read. The
# other 576 would be 36 GB of dead weight on a disk that is at 98%.
#
# `--no-state` is refused for tyr configs and would be wrong anyway: OSSCAR REWRITES
# o_proj/down_proj, so a recipe rebuilt from base weights evaluates selection-only.
#
# The card is pinned to 0 to stay inside the two-card footprint this work was given;
# card 1 is running rd_a's OOD-val. Waits rather than taking a card someone else holds.
set -uo pipefail
GPU=${GPU:-0}
REM=/home/cvlab20/project/chan
CHAN=/mnt/dataset1/chan
cd $REM/alpamayo-model-compression || exit 1
. $REM/cvlab20-server/env.sh
mkdir -p "$CHAN/logs"

until used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null);
      [ -n "$used" ] && [ "$used" -le 16 ]; do
  echo "$(date -u '+%H:%M') gpu$GPU busy (${used:-?} MiB), waiting"
  sleep 300
done

echo "$(date -u '+%F %T') building slim_tyr_rd_b on gpu$GPU"
.venv/bin/python experiments/head_analysis/make_slim.py \
  --config tyr_u40_r \
  --out outputs/slim_tyr_rd_b \
  --importance importance_v1 \
  --tyr-supernet tyr_supernet_rd_b \
  --tyr-config tyr_search_rd_b/final_config.json \
  --gpu "$GPU" \
  >>"$CHAN/logs/build_tyr_rd_b.log" 2>&1
rc=$?
echo "$(date -u '+%F %T') make_slim exit=$rc"

# the supernet subset is only an input to this build and is reproducible from cvlab21,
# so it goes as soon as the checkpoint is on disk -- 4.6 GB back on a 98% full mount
if [ "$rc" -eq 0 ] && [ -s outputs/slim_tyr_rd_b/slim_state.pt ]; then
  sz=$(stat -c %s outputs/slim_tyr_rd_b/slim_state.pt)
  echo "slim_state.pt ${sz} bytes -- removing the supernet subset"
  rm -rf "$CHAN/outputs/tyr_supernet_rd_b"
  df -h /mnt/dataset1 | tail -1
else
  echo "KEEPING the supernet: build did not produce a checkpoint"
fi
echo "$(date -u '+%F %T') build_tyr_rd_b done"
