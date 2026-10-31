# KISTI NEURON closed-loop smoke (dense, 1 scene)

User ask 2026-10-31: check whether closed loop can run on KISTI with the dense model. NEURON has no docker.

## What is being tested
Whether AlpaSim runs end to end on a NEURON A100 node through its own SLURM backend
(`wizard.run_method=SLURM`: one `srun --overlap --container-image=<sqsh>` per service, pyxis + enroot), with
the same harness, driver config and overrides as the local s600 runs. Not a measurement: one scene, one rollout.

Already verified (2026-10-31): pyxis containers start on gpu40 with the job's A100s visible (driver 580.105.08),
compute nodes have internet and a 3.5 TB local `/tmp`, `/scratch` inode quota 4M (5% used).

## Setup
- Images: `alpasim-base:0.111.0` and `nvcr.io/nvidia/nre/nre-ga:26.04` converted locally with
  `enroot import dockerd://` (zstd 3). Files `/scratch/e1997a06/alpasim/sqsh/{alpasim_base_0.111.0,nre_ga_26.04}.sqsh`
  (11 / 12 GB, sha256 checked on both sides).
  - nre-ga fails as published: its WorkingDir `/app/run.runfiles/_main` goes through an ABSOLUTE symlink
    (`/app/run.runfiles -> /app/internal/scripts/pycena/runtime/pycena_nrm_full.runfiles`), and enroot's
    `mkdir -p rootfs<WorkingDir>` follows it on the host. Fix: `FROM nre-ga:26.04` + `WORKDIR` set to the real path
    (same directory, metadata only). Flattening does not help (tried).
- AlpaSim patch (`src/wizard/alpasim_wizard/deployment/slurm.py`, SLURM path only): `--no-container-mount-home`.
  pyxis mounts the user's home by default and, with root remapping, enroot binds it AT `/root`, hiding the
  `/root/.cache/huggingface` volume (first smoke 956427: driver `LocalEntryNotFoundError`) and sending every
  `~/.cache` write into the 100k-inode NEURON home.
- `amd_a100nv_8` allows at most 8 CPU cores per GPU (16 with 1 GPU is rejected at submit).
- Code: local AlpaSim tree (with our patches: handover-braking fix, ASL image skip, slim loader) at
  `/scratch/e1997a06/project/alpasim`; wizard venv there (`uv sync --extra wizard`, cache on /scratch).
- Deploy config `deploy=neuron` (`cvpr27/code/alpasim_neuron/neuron.yaml`): filesystem `/scratch/e1997a06/alpasim`,
  HF cache `/scratch/e1997a06/checkpoints` (hub with Alpamayo-1.5-10B, offline), sqsh cache, timeout 3600 s
  (each container unpacks its image into node `/tmp` first).
- Run: `topology=pair_a15 driver=alpamayo1_5`, every service on the job's one A100, driver
  `HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8`, runtime `ALPASIM_ASL_SKIP_IMAGES=1`, `n_rollouts=1`, no video.
- Scene: `clipgt-01d503d4-449b-46fc-8d78-9085e70d3554` (uuid `4f6bf6bb…`, public_2601), first scene of the local
  s600 sh0 run, so a local dense rollout of the same scene exists for a sanity look.

## Pass criteria
1. All services open their ports (renderer, physics, driver, controller, runtime).
2. The rollout finishes: `rollouts/<scene>/<id>/metrics.parquet` and `_complete`.
3. Driver log shows Alpamayo 1.5 loading from the offline cache and CoC text in the ASL.
4. Metrics in a plausible range next to the local dense rollout (no bitwise expectation: no seed control in
   AlpaSim, and A100 is a different GPU).

## Not decided here
A100 closed-loop rows would be a new hardware block: every arm would need re-running on A100. Throughput
(account quota 4 A100s, 2 running jobs, 2-day walltime) and scene transfer (~2 GB per scene) come after the smoke.
