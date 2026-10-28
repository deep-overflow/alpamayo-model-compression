#!/bin/bash
# MoOP (slim_dual_u40_v2) closed loop on the 600-scene s600 sample, one rollout per scene, every scene on
# the same card as in the dense baseline s600_merged_baseline (plans/2026-10-28_dual-closedloop-s600.md).
# The scene lists carry a `shard` column taken from outputs/alpasim_dense_s600/scene_gpu.csv, which
# launch_alpasim_shards.sh follows instead of its round-robin.
#
#   STAGE=smoke  one scene per card (8, none of the baseline's aborted scenes)
#                -> alpasim-runs/d600smoke_merged_slim_dual_u40_v2
#   STAGE=main   the remaining 592, skipping what the smoke run scored, merged with it into
#                alpasim-runs/d600_merged_slim_dual_u40_v2
#
# Same harness settings as the baseline (patched AlpaSim, pair_a15, one shard per card with every service
# on it, DRIVER_OMP_THREADS=8). Refuses to start unless every card reads empty, samples CPU load / GPU
# utilisation every 30 s into outputs/alpasim_dual_s600/<stage>_load.log, then merges the shards and
# removes this stage's containers and compose networks (user rule, 2026-10-19).
#
#   STAGE=smoke bash experiments/head_analysis/run_dual_s600.sh > outputs/alpasim_dual_s600/smoke.log 2>&1
set -u
STAGE=${STAGE:-smoke}
CFG=slim_dual_u40_v2
REPO=/home/cvlab21/project/chan/alpamayo-model-compression
LAUNCH=$REPO/experiments/head_analysis/launch_alpasim_shards.sh
MERGE=$REPO/experiments/head_analysis/merge_alpasim_shards.py
RUNS=/home/cvlab21/project/chan/alpasim-runs
OUT=$REPO/outputs/alpasim_dual_s600
GPUS="0 1 2 3 4 5 6 7"
stamp() { echo "$(date -u '+%F %T') UTC $*"; }

if [ "$STAGE" = smoke ]; then
  PREFIX=d600smoke_ CSV=$OUT/scenes_smoke8.csv N=8 DONE="" EXTRA_SHARDS="" EXPECT=8
else
  PREFIX=d600_ CSV=$OUT/scenes_d600.csv N=600 DONE=$RUNS/d600smoke_merged_$CFG
  EXTRA_SHARDS=d600smoke_merged_$CFG EXPECT=600
  [ -f "$OUT/smoke.DONE" ] || { stamp "smoke stage not done"; exit 1; }
fi

for g in $GPUS; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
  if [ "$used" -gt 10 ]; then stamp "GPU $g holds $used MiB -- not starting"; exit 1; fi
done
stamp "stage $STAGE: $CFG, $N scenes x 1 rollout on GPUs $GPUS"

(while true; do
  echo "$(date -u +%T) load $(cut -d' ' -f1-3 /proc/loadavg) cpu_idle $(vmstat 1 2 | tail -1 | awk '{print $15}')" \
       "gpu_util $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr '\n' ' ')"
  sleep 30
done) > "$OUT/${STAGE}_load.log" 2>&1 &
sampler=$!

SUITE=public_2601_${PREFIX%_} PARENT_SUITE=public_2601 PREFIX=$PREFIX DRIVER_OMP_THREADS=8 SCENES_CSV=$CSV \
  bash "$LAUNCH" "$CFG" "$N" 1 "$GPUS" $DONE
stamp "shards finished (exit $?)"
kill "$sampler"

merged=${PREFIX}merged_$CFG
"$REPO/.venv/bin/python" "$MERGE" --runs-root "$RUNS" \
  --shards $(for i in 0 1 2 3 4 5 6 7; do echo "${PREFIX}${CFG}_sh$i"; done) $EXTRA_SHARDS \
  --out "$merged" --expect-scenes "$EXPECT"
rc=$?
stamp "merged -> $RUNS/$merged (exit $rc)"

for c in $(docker ps -a --format '{{.Names}}' | grep "^${PREFIX}"); do docker rm -f "$c" > /dev/null; done
for n in $(docker network ls --format '{{.Name}}' | grep "^${PREFIX}"); do
  [ "$(docker network inspect -f '{{len .Containers}}' "$n")" = 0 ] && docker network rm "$n" > /dev/null
done
stamp "cleanup: containers left $(docker ps -a --format '{{.Names}}' | grep -c "^${PREFIX}")," \
      "networks left $(docker network ls --format '{{.Name}}' | grep -c "^${PREFIX}")"
[ "$rc" = 0 ] && touch "$OUT/${STAGE}.DONE"
