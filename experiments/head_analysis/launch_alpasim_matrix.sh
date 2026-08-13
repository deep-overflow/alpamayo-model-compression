#!/bin/bash
# Launch the closed-loop eval matrix in alpasim: one wizard run per config, in parallel.
#
# Usage:
#   bash launch_alpasim_matrix.sh <n_scenes> <n_rollouts> [config ...]
#   PREFIX=matrix150_ bash launch_alpasim_matrix.sh 150 2 baseline slim_dual_u40_v2 ...
#
# PREFIX (default `matrix_`) names the log dirs. Give a matrix with a different scene
# count its own prefix: the wizard resumes from an existing log_dir, so reusing
# `matrix_` for a 150-scene run would silently extend the stored 30-scene results.
# analyze_alpasim.py takes the same --prefix.
#
# SUITE (default `public_2604`) selects the scene set. public_2601 (913 scenes) and
# public_2604 (1606) share 159 scene_ids but ZERO artifact uuids -- 2604 re-rendered
# everything under the 26.04 release -- so the two are separate 1.7 TB / 100 GB caches
# and results from one suite cannot be compared with results from the other.
#
# Configs default to: baseline slim_cocsafe_r20 slim_cocsafe_r30 slim_integrated_mag
# Per config i: driver GPU 4+i (Ada, same arch for all drivers), renderer/physics
# GPU 2 (i<2) or 3 (i>=2). Waits for assigned GPUs to be free (<1 GB used) before
# each launch; staggers launches so the first run populates the shared scene cache.
#
# Requires /home/cvlab21/project/chan/alpasim-runs/.hf_env (HF_TOKEN + HF_HOME).

set -u

ALPASIM=/home/cvlab21/project/chan/alpasim
RUNS=/home/cvlab21/project/chan/alpasim-runs
# NuRec scenes live off-repo (49 GB); a symlink inside data/ dangles in the
# containers, so point scene_cache at the real path instead.
SCENE_CACHE=/mnt/nvme1n1/ad_vla/data/nre-artifacts
# Slim checkpoints (50 GB) moved off the root filesystem on 2026-08-06. The driver
# dirs are hardlinks of outputs/<cfg>/, so they must sit on the same filesystem as
# outputs -- defines.drivers is the docker bind-mount source, and the in-container
# path (/mnt/drivers/<cfg>) is unchanged.
DRIVERS=/mnt/nvme1n1/ad_vla/data/alpasim/drivers
N_SCENES=${1:?usage: launch_alpasim_matrix.sh <n_scenes> <n_rollouts> [config ...]}
N_ROLLOUTS=${2:?usage: launch_alpasim_matrix.sh <n_scenes> <n_rollouts> [config ...]}
shift 2
CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
  CONFIGS=(baseline slim_cocsafe_r20 slim_cocsafe_r30 slim_integrated_mag)
fi

source "$RUNS/.hf_env"

wait_gpus_free() {
  # blocks until every listed GPU index has <1 GB memory in use
  local gpus="$1"
  while true; do
    local busy=0
    for g in $gpus; do
      local used
      used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)
      if [ "$used" -ge 1024 ]; then busy=1; fi
    done
    [ "$busy" -eq 0 ] && return 0
    echo "[matrix] GPUs $gpus busy, retrying in 60s"
    sleep 60
  done
}

# fixed per-config GPU map: drivers on Ada 4-7, renderer/physics on Blackwell 2-3
driver_gpu_of() {
  case "$1" in
    baseline) echo 4 ;;
    slim_cocsafe_r20) echo 5 ;;
    slim_cocsafe_r30) echo 6 ;;
    slim_integrated_mag) echo 7 ;;
    slim_dual_uniform) echo 7 ;;
    # Ada, to match the stored matrix -- all four shipped configs ran on Ada drivers,
    # so putting this one on Blackwell would reintroduce the kernel confound the
    # original GPU map was designed to avoid.
    slim_j_traj_r20) echo 5 ;;
    # the 2026-08-09 one-factor pair: same budget, same allocation, expert and KV
    # untouched, differing only in max(rank I_traj, rank X). Ada like every other
    # driver in the stored matrix, and adjacent cards so the pair shares conditions.
    slim_dual_u40_v2) echo 5 ;;
    slim_jtraj_u40_v2) echo 6 ;;
    *) echo "unknown config: $1" >&2; exit 1 ;;
  esac
}
rend_gpu_of() {
  # REND_ON_DRIVER=1 collapses renderer/physics/trafficsim onto the driver's own card.
  # Needed when the Blackwell pair is held by someone else's multi-day job, as on
  # 2026-08-10. Safe for a comparison -- the renderer does not run the model and every
  # arm of one matrix shares the arch -- but a matrix rendered on Ada is not directly
  # comparable to one rendered on Blackwell, so give it its own PREFIX.
  if [ "${REND_ON_DRIVER-0}" = 1 ]; then
    driver_gpu_of "$1"
    return
  fi
  case "$1" in
    baseline|slim_cocsafe_r20|slim_dual_uniform) echo 2 ;;
    slim_j_traj_r20) echo 0 ;;
    *) echo 3 ;;
  esac
}

i=0
pids=()
for cfg in "${CONFIGS[@]}"; do
  driver_gpu=$(driver_gpu_of "$cfg")
  rend_gpu=$(rend_gpu_of "$cfg")
  # PREFIX separates matrices that differ in scene count: the wizard resumes from an
  # existing log_dir, so reusing matrix_ for a 150-scene run would silently mix with
  # the stored 30-scene results. analyze_alpasim.py takes the same --prefix.
  log_dir="$RUNS/${PREFIX-matrix_}${cfg}"

  extra=()
  if [ "$cfg" != "baseline" ]; then
    extra+=("driver.model.checkpoint_path=/mnt/drivers/${cfg}")
  fi

  # only gate on the dedicated driver GPU: the renderer GPU is shared by design
  wait_gpus_free "$driver_gpu"
  echo "[matrix] launching $cfg: driver GPU $driver_gpu, renderer GPU $rend_gpu -> $log_dir"
  (
    cd "$ALPASIM" &&
    uv run alpasim_wizard \
      deploy=local topology=pair_a15 driver=alpamayo1_5 \
      defines.drivers="$DRIVERS" \
      wizard.log_dir="$log_dir" \
      wizard.run_name="a15_${PREFIX-matrix_}${cfg}" \
      "services.renderer.gpus=[$rend_gpu]" \
      "services.physics.gpus=[$rend_gpu]" \
      "services.trafficsim.gpus=[$rend_gpu]" \
      "services.driver.gpus=[$driver_gpu]" \
      "services.driver.environments=['HF_HUB_OFFLINE=1']" \
      "services.runtime.environments=['ALPASIM_ASL_SKIP_IMAGES=1']" \
      scenes.test_suite_id="${SUITE-public_2604}" \
      scenes.scene_cache="$SCENE_CACHE" \
      scenes.limit_to_first_n="$N_SCENES" \
      runtime.simulation_config.n_rollouts="$N_ROLLOUTS" \
      eval.allow_aggregation_with_failed_rollouts=true \
      eval.video.render_video=false \
      "${extra[@]}"
  ) > "$RUNS/${PREFIX-matrix_}${cfg}.launch.log" 2>&1 &
  pids+=($!)
  i=$((i + 1))
  # stagger so the first run downloads scenes before the rest start
  [ "$i" -lt ${#CONFIGS[@]} ] && sleep 120
done

echo "[matrix] waiting for ${#pids[@]} runs: ${pids[*]}"
fail=0
for p in "${pids[@]}"; do
  wait "$p" || fail=1
done
echo "[matrix] all runs finished (fail=$fail)"
exit $fail
