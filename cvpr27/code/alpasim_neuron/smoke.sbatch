#!/bin/bash
#SBATCH --job-name=alpasim_smoke
#SBATCH --partition=amd_a100nv_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
# amd_a100nv_8 allows at most 8 cores per GPU (16 with 1 GPU is rejected at submit)
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --comment=field=efficientai;appl=pytorch
#SBATCH --output=/scratch/e1997a06/alpasim/runs/%x-%j.out
#
# Closed-loop smoke on NEURON (cvpr27/plans/2026-10-31_kisti-closedloop-smoke.md): dense Alpamayo 1.5, one scene,
# one rollout, every service on this job's one A100. The wizard runs here in the batch step and starts each
# service as its own `srun --overlap --container-image=<sqsh>` step on this node (AlpaSim's SLURM backend).
# Overrides mirror experiments/head_analysis/launch_alpasim_shards.sh so the run matches the local s600 runs.
set -euo pipefail
cd /scratch/e1997a06/project/alpasim
D=/scratch/e1997a06/alpasim
RUN=$D/runs/smoke_dense_${SLURM_JOB_ID}
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
mkdir -p "$D/drivers"
.venv/bin/alpasim_wizard \
  deploy=neuron topology=pair_a15 driver=alpamayo1_5 \
  wizard.log_dir="$RUN" \
  wizard.run_name="neuron_smoke_dense_${SLURM_JOB_ID}" \
  "services.renderer.gpus=[0]" \
  "services.physics.gpus=[0]" \
  "services.trafficsim.gpus=[0]" \
  "services.driver.gpus=[0]" \
  "services.driver.environments=['HF_HUB_OFFLINE=1','OMP_NUM_THREADS=8','MKL_NUM_THREADS=8']" \
  "services.runtime.environments=['ALPASIM_ASL_SKIP_IMAGES=1']" \
  scenes.scene_ids=null \
  scenes.test_suite_id=neuron_smoke1 \
  "scenes.suites_csv=[$PWD/data/scenes/sim_suites.csv,$D/neuron_smoke1.csv]" \
  scenes.scene_cache="$D/nre-artifacts" \
  runtime.simulation_config.n_rollouts=1 \
  eval.allow_aggregation_with_failed_rollouts=true \
  eval.video.render_video=false
echo "wizard exited $? at $(date -Is)"
find "$RUN/rollouts" -name metrics.parquet -o -name _complete | head
