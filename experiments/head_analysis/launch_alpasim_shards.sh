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
#
# Env:
#   SUITE=public_2601        which suite the uuids must belong to
#   PREFIX=m2601_            log-dir prefix, so a new scene set does not collide with the matrix
#   SCENES_CSV=<path>        run an explicit (scene_id, uuid) list instead of the suite's first N
#   DRIVER_OMP_THREADS=8     worth 2.3x wall-clock; must not be mixed within one comparison
#   ONLY_SHARDS="0 2"        run a subset while keeping the same k-way split
#   DRY_RUN=1                write the shard suites and print the plan, launch nothing
#
# Example, the hard-100 set (make_hard_suite.py writes the CSV):
#   SUITE=public_2601_hard100 PREFIX=h100_ DRIVER_OMP_THREADS=8 \
#   SCENES_CSV=$PWD/outputs/scene_difficulty/hard100_suite.csv \
#     bash launch_alpasim_shards.sh slim_dual_u40_v2 100 2 "4 5 6 7"
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
#
# SCENES_CSV lets a run use an explicit scene list instead of the suite's first N -- that is how
# the hard-N sets from make_hard_suite.py are run. The file must carry (scene_id, uuid) and its
# uuids must all belong to $SUITE; the check below refuses anything else, because a bare scene_id
# would let the 26.04 render slip in for the 159 ids that exist in both releases.
SCENES_CSV=${SCENES_CSV-}
# With SCENES_CSV, $SUITE is the *new* id written into the shard suites, so the uuids cannot be
# checked against it -- they belong to the release they were rendered from. PARENT_SUITE names
# that release and is what pins the render version.
PARENT_SUITE=${PARENT_SUITE-public_2601}
SUITES_CSV=$RUNS/shard_suites_${PREFIX}${CFG}.csv
mapfile -t SHARD_SUITES < <("$REPO/.venv/bin/python" - \
    "$SUITE" "$N_SCENES" "${#GPUS[@]}" "$DONE_DIR" "$SUITES_CSV" "$CFG" "$SCENES_CSV" \
    "$PARENT_SUITE" <<'PY'
import sys
from pathlib import Path
import pandas as pd

suite, n, k, done_dir, out_csv, cfg, scenes_csv, parent = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6],
    sys.argv[7], sys.argv[8])
su = pd.read_csv("/home/cvlab21/project/chan/alpasim/data/scenes/sim_suites.csv")
if scenes_csv:
    rows = pd.read_csv(scenes_csv)
    for col in ("scene_id", "uuid"):
        if col not in rows.columns:
            sys.exit(f"FATAL {scenes_csv}: '{col}' 열이 없습니다")
    rows = rows.sort_values("scene_id").head(n)
    # every uuid must come from the parent release -- a uuid from 26.04 would silently swap the
    # scene for a different rendering of the same drive
    own = set(su.loc[su.test_suite_id == parent, "uuid"])
    if not own:
        sys.exit(f"FATAL: PARENT_SUITE '{parent}' 가 sim_suites.csv 에 없습니다")
    bad = sorted(set(rows.uuid) - own)
    if bad:
        sys.exit(f"FATAL {scenes_csv}: {parent} 소속이 아닌 uuid {len(bad)}개 (예: {bad[:2]})")
    if rows.scene_id.duplicated().any():
        sys.exit(f"FATAL {scenes_csv}: scene_id 중복")
    print(f"# scene list: {scenes_csv} -> {len(rows)} scenes", file=sys.stderr)
else:
    rows = su[su.test_suite_id == suite].sort_values("scene_id").head(n)
if len(rows) < n:
    sys.exit(f"FATAL: {n}개를 요청했으나 {len(rows)}개만 있습니다")
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

# mapfile swallows the python's exit status, so a rejected scene list would otherwise fall
# through to `set -u` failing several lines later, after the plan has already been printed.
if [ "${#SHARD_SUITES[@]}" -eq 0 ]; then
  echo "[shards] 샤드 스위트 생성 실패 -- 위의 FATAL 메시지를 보세요" >&2
  exit 1
fi

echo "[shards] $CFG: ${#GPUS[@]} shards over GPUs ${GPUS[*]}"

# DRY_RUN stops here: the per-shard suites are written and the plan printed, but no container
# starts. Use it to check a new scene list before spending GPU hours on it.
if [ -n "${DRY_RUN-}" ]; then
  echo "[shards] DRY_RUN -- 컨테이너를 띄우지 않고 종료합니다"
  echo "[shards] shard suites: $SUITES_CSV"
  for i in "${!GPUS[@]}"; do
    read -r suite_name n_in_shard <<<"${SHARD_SUITES[$i]}"
    printf '[shards]   shard %d -> GPU %s, %s scenes, %s%s%s_sh%d\n' \
      "$i" "${GPUS[$i]}" "${n_in_shard:-0}" "$RUNS/" "$PREFIX" "$CFG" "$i"
  done
  [ "$CFG" != "baseline" ] && echo "[shards] driver: $DRIVERS/$CFG"
  exit 0
fi

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
