#!/bin/bash
# Dense Alpamayo 1.5 closed loop on the stratified 600-scene public_2601 sample, one rollout per scene,
# on all eight cards (plans/2026-10-28_dense-closedloop-full2601.md; scene list from make_strat_suite.py).
#
#   STAGE=smoke  16 sampled scenes outside Origin150, two per card -> alpasim-runs/s600smoke_merged_baseline
#   STAGE=main   the remaining 584, skipping what the smoke run scored, merged with it into
#                alpasim-runs/s600_merged_baseline
#
# Same harness settings as hofix_merged_baseline (patched AlpaSim, pair_a15, one shard per card with every
# service on that card, DRIVER_OMP_THREADS=8). Refuses to start unless every card reads empty, samples CPU
# load / GPU utilisation every 30 s into outputs/alpasim_dense_s600/<stage>_load.log, then merges the
# shards and removes this stage's containers and compose networks (user rule, 2026-10-19).
#
#   STAGE=smoke bash experiments/head_analysis/run_dense_s600.sh > outputs/alpasim_dense_s600/smoke.log 2>&1
set -u
STAGE=${STAGE:-smoke}
REPO=/home/cvlab21/project/chan/alpamayo-model-compression
LAUNCH=$REPO/experiments/head_analysis/launch_alpasim_shards.sh
MERGE=$REPO/experiments/head_analysis/merge_alpasim_shards.py
RUNS=/home/cvlab21/project/chan/alpasim-runs
OUT=$REPO/outputs/alpasim_dense_s600
GPUS="0 1 2 3 4 5 6 7"
stamp() { echo "$(date -u '+%F %T') UTC $*"; }

if [ "$STAGE" = smoke ]; then
  PREFIX=s600smoke_ CSV=$OUT/scenes_smoke16.csv N=16 DONE="" EXTRA_SHARDS="" EXPECT=16
else
  PREFIX=s600_ CSV=$OUT/scenes_s600.csv N=600 DONE=$RUNS/s600smoke_merged_baseline
  EXTRA_SHARDS=s600smoke_merged_baseline EXPECT=600
  [ -f "$OUT/smoke.DONE" ] || { stamp "smoke stage not done"; exit 1; }
fi

for g in $GPUS; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
  if [ "$used" -gt 10 ]; then stamp "GPU $g holds $used MiB -- not starting"; exit 1; fi
done
stamp "stage $STAGE: $N scenes x 1 rollout on GPUs $GPUS"

(while true; do
  echo "$(date -u +%T) load $(cut -d' ' -f1-3 /proc/loadavg) cpu_idle $(vmstat 1 2 | tail -1 | awk '{print $15}')" \
       "gpu_util $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr '\n' ' ')"
  sleep 30
done) > "$OUT/${STAGE}_load.log" 2>&1 &
sampler=$!

SUITE=public_2601_${PREFIX%_} PARENT_SUITE=public_2601 PREFIX=$PREFIX DRIVER_OMP_THREADS=8 SCENES_CSV=$CSV \
  bash "$LAUNCH" baseline "$N" 1 "$GPUS" $DONE
stamp "shards finished (exit $?)"
kill "$sampler"

merged=${PREFIX}merged_baseline
[ "$STAGE" = main ] && merged=s600_merged_baseline
"$REPO/.venv/bin/python" "$MERGE" --runs-root "$RUNS" \
  --shards $(for i in 0 1 2 3 4 5 6 7; do echo "${PREFIX}baseline_sh$i"; done) $EXTRA_SHARDS \
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
