#!/bin/bash
# Run ONE closed-loop config across several GPUs by splitting its scenes.
#
# launch_alpasim_matrix.sh gives each config a card, so wall-clock is the slowest
# config -- with an unpruned baseline in the matrix that is ~40% longer than the pruned
# arms. This runs one config at a time but shards its scenes over every free card, so
# the finish time is the total work divided by the cards, not by the configs.
#
# The wizard has no scene offset, only `limit_to_first_n` on a scene_id sort, so each
# shard gets its own generated *test suite*: a CSV of (test_suite_id, scene_id, uuid)
# rows copied verbatim from the real suite, appended to scenes.suites_csv.
#
# Do NOT shard with `scenes.scene_ids` instead. 159 of the 2601 scene_ids also exist in
# the 26.04 release, and query_by_scene_ids resolves a bare scene_id to the newer uuid --
# it silently swaps in 26.04 renders (and downloads them). Only the suite CSV carries
# the uuid, so only a suite pins the render version.
#
# Scenes already finished by an earlier run are skipped -- pass that run's log dir as
# the last argument and merge it back in afterwards with merge_alpasim_shards.py.
#
# Usage:
#   bash launch_alpasim_shards.sh <config> <n_scenes> <n_rollouts> "<gpu list>" [done_dir]
#
# Example:
#   bash launch_alpasim_shards.sh baseline 150 2 "4 5 6 7" \
#       /home/cvlab21/project/chan/alpasim-runs/m2601_150_baseline
set -u

REPO=/home/cvlab21/project/chan/alpamayo-model-compression
ALPASIM=/home/cvlab21/project/chan/alpasim
RUNS=/home/cvlab21/project/chan/alpasim-runs
SCENE_CACHE=/mnt/nvme1n1/ad_vla/data/nre-artifacts
DRIVERS=/mnt/nvme1n1/ad_vla/data/alpasim/drivers
SUITE=${SUITE-public_2601}
PREFIX=${PREFIX-m2601_}

CFG=${1:?usage: launch_alpasim_shards.sh <config> <n_scenes> <n_rollouts> "<gpus>" [done_dir]}
N_SCENES=${2:?}
N_ROLLOUTS=${3:?}
read -r -a GPUS <<<"${4:?}"
DONE_DIR=${5-}

source "$RUNS/.hf_env"

# Build one suite per shard, carrying the original suite's uuids unchanged.
SUITES_CSV=$RUNS/shard_suites_${PREFIX}${CFG}.csv
mapfile -t SHARD_SUITES < <("$REPO/.venv/bin/python" - \
    "$SUITE" "$N_SCENES" "${#GPUS[@]}" "$DONE_DIR" "$SUITES_CSV" "$CFG" <<'PY'
import sys
from pathlib import Path
import pandas as pd

suite, n, k, done_dir, out_csv, cfg = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6])
su = pd.read_csv("/home/cvlab21/project/chan/alpasim/data/scenes/sim_suites.csv")
rows = su[su.test_suite_id == suite].sort_values("scene_id").head(n)
done = set()
if done_dir:
    for p in Path(done_dir, "rollouts").glob("*/*/metrics.parquet"):
        done.add(p.parent.parent.name)
todo = rows[~rows.scene_id.isin(done)].reset_index(drop=True)
# round-robin rather than contiguous blocks: scene cost varies with route length, so
# interleaving keeps the shards equal in wall-clock, not just in count
out, names = [], []
for i in range(k):
    part = todo.iloc[i::k].copy()
    name = f"{suite}_{cfg}_sh{i}"
    part["test_suite_id"] = name
    out.append(part)
    names.append(f"{name} {len(part)}")
pd.concat(out)[["test_suite_id", "scene_id", "uuid"]].to_csv(out_csv, index=False)
print("\n".join(names))
PY
)

echo "[shards] $CFG: ${#GPUS[@]} shards over GPUs ${GPUS[*]}"
pids=()
for i in "${!GPUS[@]}"; do
  gpu=${GPUS[$i]}
  read -r suite_name n_in_shard <<<"${SHARD_SUITES[$i]}"
  [ "${n_in_shard:-0}" -eq 0 ] && continue
  # ONLY_SHARDS runs a subset while keeping the same k-way split, so a run can be
  # staged over the cards in waves without changing which scenes land in which shard.
  if [ -n "${ONLY_SHARDS-}" ] && ! grep -qw "$i" <<<"$ONLY_SHARDS"; then continue; fi
  log_dir="$RUNS/${PREFIX}${CFG}_sh${i}"
  extra=()
  [ "$CFG" != "baseline" ] && extra+=("driver.model.checkpoint_path=/mnt/drivers/${CFG}")
  # The driver burns ~11.6 cores while the model runs on the GPU: with OMP_NUM_THREADS
  # unset, torch opens one intra-op thread per core (64 here) and the OpenMP barriers
  # busy-wait, so 64 threads sit at ~18% each doing nothing. Four shards make that 256
  # threads on 64 cores and the box goes CPU-bound at 28 min/scene. Opt in per run --
  # thread count perturbs CPU reduction order, so a matrix must not mix settings.
  driver_env="'HF_HUB_OFFLINE=1'"
  if [ -n "${DRIVER_OMP_THREADS-}" ]; then
    driver_env="$driver_env,'OMP_NUM_THREADS=$DRIVER_OMP_THREADS'"
    driver_env="$driver_env,'MKL_NUM_THREADS=$DRIVER_OMP_THREADS'"
  fi
  echo "[shards]   shard $i -> GPU $gpu, $n_in_shard scenes, $log_dir"
  (
    cd "$ALPASIM" &&
    uv run alpasim_wizard \
      deploy=local topology=pair_a15 driver=alpamayo1_5 \
      defines.drivers="$DRIVERS" \
      wizard.log_dir="$log_dir" \
      wizard.run_name="a15_${PREFIX}${CFG}_sh${i}" \
      "services.renderer.gpus=[$gpu]" \
      "services.physics.gpus=[$gpu]" \
      "services.trafficsim.gpus=[$gpu]" \
      "services.driver.gpus=[$gpu]" \
      "services.driver.environments=[$driver_env]" \
      "services.runtime.environments=['ALPASIM_ASL_SKIP_IMAGES=1']" \
      scenes.scene_ids=null \
      scenes.test_suite_id="$suite_name" \
      "scenes.suites_csv=[$ALPASIM/data/scenes/sim_suites.csv,$SUITES_CSV]" \
      scenes.scene_cache="$SCENE_CACHE" \
      runtime.simulation_config.n_rollouts="$N_ROLLOUTS" \
      eval.allow_aggregation_with_failed_rollouts=true \
      eval.video.render_video=false \
      "${extra[@]}"
  ) > "$RUNS/${PREFIX}${CFG}_sh${i}.launch.log" 2>&1 &
  pids+=($!)
  [ "$i" -lt $((${#GPUS[@]} - 1)) ] && sleep 20
done

echo "[shards] waiting for ${#pids[@]} shards: ${pids[*]}"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "[shards] $CFG finished (fail=$fail)"
exit $fail
