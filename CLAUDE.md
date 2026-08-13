# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A research fork of NVIDIA's Alpamayo 1 (Alpamayo-R1) release, used as the working repo for
**model-compression research on Alpamayo 1.5** (structured pruning of attention heads, MLP channels,
and KV groups). Two distinct codebases live here:

- `src/alpamayo_r1/` — the upstream Alpamayo 1 inference release (reference only; Apache-2.0).
- `experiments/head_analysis/` — the actual research code. It does **not** import `alpamayo_r1`; it
  imports the `alpamayo1_5` package (installed into this repo's `.venv`) and runs
  `nvidia/Alpamayo-1.5-10B`.

Research work was merged into `main` on 2026-08-06 (the `head-attention-analysis` task branch is
deleted); new tasks branch off `main`.
`experiments/`, `outputs/`, `reports/`, and `plans/` are untracked research artifacts.

## Environment & running experiments

Experiment scripts run with this repo's own `.venv`, which has `alpamayo1_5` installed from
`git+https://github.com/NVlabs/alpamayo1.5.git@f42e594` — the same commit alpasim uses. (Until
2026-07-28 they ran with a sibling repo at `/workspace/alpamayo1.5/.venv/bin/python`; that repo and
its venv are gone and its remote 404s.)

**All AD-VLA data, environments and results live under `/mnt/nvme1n1/ad_vla`** (moved 2026-08-06;
the root partition was at 97%). Everything below is reached through a symlink at its original path,
so nothing that uses `$REPO/.venv/bin/python`, `REPO / "outputs" / exp_id`, or
`--runs-root /home/cvlab21/project/chan/alpasim-runs` needs to change:

| original path | real location | size |
|---|---|---|
| `<repo>/.venv` | `ad_vla/venvs/alpamayo-mc` | 8 GB |
| `<repo>/outputs` | `ad_vla/outputs/chan` | 51 GB |
| `alpasim/.venv` | `ad_vla/venvs/alpasim-chan` | 9 GB |
| `chan/alpasim-runs` | `ad_vla/results/chan/alpasim-runs` | 9 GB |
| HF blobs | `ad_vla/cache/hub` (`HF_HUB_CACHE`) | 75 GB |

The code checkouts themselves stay on the root partition. `ad_vla/venvs/alpasim` (no `-chan`) is
**another lab member's** environment — its `alpasim_utils` resolves into `project/sangoh/`; ours is
`alpasim-chan`. Plain `uv run` was verified not to replace either symlink, so
`UV_PROJECT_ENVIRONMENT` is not required.

`/mnt/nvme1n1` is shared and sits near 98% full, so check `df` before building a new checkpoint
(each is 15–17 GB). When reading sizes there, run **one** `du` over all siblings at once: much of
that tree is hardlinked (e.g. `data/alpasim_2505` shows 1.6 TB on its own but owns only ~11 GB;
the rest is shared inodes with `data/alpasim`), and per-directory `du` calls double-count it.

Preferred launcher on this shared GPU box (retries every 60s while GPUs are busy):

```bash
bash experiments/head_analysis/run_retry_host.sh <max_attempts> \
    experiments/head_analysis/<script>.py [--gpu N] [args...]
```

It exports what a run needs: `HF_TOKEN` (read from `~/.cache/huggingface/stored_tokens`; the
`full_right` token has gated-dataset access), `HF_HOME=$HOME/.cache/huggingface`,
`HF_HUB_CACHE=/mnt/nvme1n1/ad_vla/cache/hub`, `CUDA_DEVICE_ORDER=PCI_BUS_ID` (so
`--gpu N` and `nvidia-smi -i N` refer to the same card on this box), and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Each script documents its own usage and flags in
its module docstring.

**Token and blobs live in different places** (since 2026-08-06). The 22 GB of weights moved to the
shared cache on `/mnt/nvme1n1` to get off the root partition, but that cache's `stored_tokens` holds
only other lab members' tokens — no `full_right` — so the token still has to come from `$HOME`.
`HF_HOME` locates the token, `HF_HUB_CACHE` locates the blobs; never set only one of them.
Anything resolving a cached repo path in code should read
`huggingface_hub.constants.HF_HUB_CACHE` rather than hardcode `~/.cache/huggingface/hub`.
alpasim is the exception: its wizard bind-mounts `$HF_HOME` as the container's whole cache
(`${defines.hf_cache}:/root/.cache/huggingface`), so `alpasim-runs/.hf_env` sets
`HF_HOME=/mnt/nvme1n1/ad_vla/cache` and supplies auth via its own `HF_TOKEN` export.

Stale paths to watch for — both predate the move to the host venv:

- `run_retry.sh` is the older in-container twin and still points at the missing sibling venv.
- Older scripts hardcode `REPO = Path("/workspace/alpamayo-model-compression")`, which does not
  exist on the host; new ones resolve it from `__file__` (`Path(__file__).resolve().parents[2]`).
  `build_report.py` still has the hardcode, so pass **absolute** paths to it on the host.

GPU handling inside the scripts:

- `reserve_gpu(need_gb)` in `expert_per_clip.py` scans cards, actually reserves the memory in
  PyTorch's caching allocator (held for the process lifetime), and returns the first card with room;
  `--gpu` restricts the scan so parallel runs land on distinct cards.
- Importing `expert_per_clip` also installs a `transformers` hub patch forcing
  `local_files_only=True` for `nvidia/Cosmos-Reason2-8B` (gated repo, served from local cache).
  Every runner imports it first for this reason, `slim_lib.load_slim()` included.

Model weights and the PhysicalAI-AV dataset are gated on HuggingFace; weights are ~22 GB and
inference needs ≥24 GB VRAM. The upstream smoke test (Alpamayo 1, not used by the experiments) is
`python src/alpamayo_r1/test_inference.py`.

Lint: `uvx ruff check .` (ruff is not a project dep; `pyproject.toml` only sets line-length 100, so
the effective ruleset is ruff's default and the existing scripts are not clean under it — check the
files you touched, not the whole tree). Packages are managed with `uv`. There is no unit-test
suite — verification means running the experiment scripts and checking their outputs. Long runs are
sharded by `--clip-offset` (e.g. `jsweep_s0` + `jsweep_s40`) and merged by the matching `analyze_*`.

## Model architecture (what is being pruned)

Alpamayo pairs two transformer towers (see `src/alpamayo_r1/models/alpamayo_r1.py`; 1.5 has the
same shape on a Qwen3-VL backbone):

1. **VLM**: multi-camera frames + ego-history trajectory tokens (fused via `fuse_traj_tokens`) →
   autoregressive Chain-of-Causation (CoC) reasoning text.
2. **Expert**: a second transformer (no `embed_tokens`) that serves
   as the denoiser for flow-matching diffusion over actions: `action_in_proj` embeds the noisy
   action into 64 diffusion tokens → expert forward with the VLM's KV cache as prefill (non-causal
   over the diffusion tokens, `prompt_cache.crop()` after every denoise step) → `action_out_proj`
   predicts the vector field. Actions live in a unicycle accel/curvature space and decode to a
   6.4 s / 64-waypoint / 10 Hz trajectory.

The towers are **not** the same width — only `head_dim` (128) and the 8 KV groups match, because
that is the interface the expert needs to read the VLM's cache. From `config.json`:

| tower    | hidden | Q heads | head_dim | intermediate | layers | params |
|----------|--------|---------|----------|--------------|--------|--------|
| VLM text | 4096   | 32      | 128      | 12288        | 36     | ~6.9B  |
| expert (`expert_cfg`) | 2048 | 16 | 128 | 8256 | 36 | ~2.3B |

So anything that assumes shape-compatible towers (weight tying, cross-tower delta
reparameterisation) does not apply. The expert is 20.6% of params and only 5.5% of FLOPs
(3.52 of 63.74 TFLOP) yet 22–28% of wall-clock depending on the path: it is step-bound
(10 sequential passes over 64 tokens, re-reading every weight each step), not width-bound.

Inference stages as profiled in `profile_stages.py`: ViT → VLM prefill → CoC decode → expert
denoising.

GQA coupling that all masking relies on (`mask_lib.py` docstring): KV group `h` feeds VLM Q heads
`[4h, 4h+4)` and expert Q heads `[2h, 2h+2)`. A KV group is "removed" by masking its Q heads —
zeroing cached K would just make softmax spread uniform weight.

## Experiment harness (`experiments/head_analysis/`)

Shared libraries:

- `analysis_lib.py` — builds fused prompts exactly as release inference does, identifies token
  spans (vision / traj-history / sink / text, per-camera), switches attention impl to eager for
  stat capture.
- `prune_lib.py` — Phase P1: multiplicative unit gates + dual-objective Taylor importance (CoC NLL
  vs trajectory flow-matching MSE). KV groups are scored on the cache tensors directly because
  Qwen3's k_norm makes a per-head gate on k_proj carry no gradient.
- `mask_lib.py` — Phase P2/P3: 0/1 forward-pre-hook masks on o_proj input (per Q head) and
  down_proj input (per MLP channel); masked is functionally identical to removed. Selection is
  per-layer so a width sweep does not silently become a depth sweep.
- `jlens_lib.py` — **label-free** criterion. Jacobian lens `J_l = E[dh_final/dh_l]` with rows of
  `W_U` as cotangents, so one backward per dictionary token yields that token's lens vector at
  *every* layer; unit scores then follow by matmul from second moments of activations, with no
  per-unit backward. `jfrac = j / write_norm` separates "writes a lot, none of it verbalizable"
  from the converse. Source positions are restricted to text/CoC tokens (vision tokens dominate the
  prompt and only dilute `J_l`).
- `eval_lib.py` — multi-sample minADE/minFDE, geometry-derived scenario buckets from the GT path
  (priority: decel_stop > turn > accel > cruise), paired bootstrap CIs, and `coc_degenerate()`
  whose thresholds deliberately mirror `analyze_alpasim.coc_stats` so open-loop and closed-loop
  degeneracy rates stay comparable.
- `slim_lib.py` — physical removal. Mask → real weight surgery (q_proj rows / o_proj cols, gate/up
  rows / down cols). KV heads, k/v projections, and the cache are **never** touched: the expert
  reads the VLM's per-layer cache, so 8×128 is a hard interface. Uneven Q-head pruning across KV
  groups breaks `repeat_kv`/`enable_gqa`, so the slim attention forward gathers K/V per kept Q head
  and runs `num_key_value_groups=1`; `bind_identity()` puts the unpruned model on that same path
  for honest paired latency. Checkpoints are `torch.save` state_dicts + `slim_meta.json` of kept
  indices (`save_pretrained` cannot round-trip non-uniform layer shapes); reload with `load_slim()`.

Runner tracks (each `run_*.py` pairs with an `analyze_*.py`):

| track | runner | analysis |
|---|---|---|
| P1 importance (dual-objective Taylor) | `run_importance.py` → `importance.npz` + `importance_perclip.npz` | `analyze_importance.py`, `analyze_cvar.py` (tail-risk vs mean aggregation), `analyze_agreement.py` (per-layer objective agreement → allocation) |
| P2 mask sweeps | `run_ablation.py`, `run_expert_ablation.py` | `analyze_ablation.py` |
| P3 combined configs | `run_eval.py`, `run_integrated.py`, `run_cocsafe.py`, `run_kvfix.py`, `scan_buckets.py` | `analyze_eval.py` |
| criterion × allocation grid | `run_grid.py` | `analyze_grid.py` |
| J-space (label-free) | `run_jlens.py` (build lens, gates G1/G2), `run_jspace_sweep.py` (criterion sweep at matched ratios, `--jlens` picks the lens run), `run_kviso.py` (KV-drop choice isolation; writes its own summary, no analyze pair) | `analyze_jspace.py`, `analyze_jsweep.py` |
| physical removal + latency | `make_slim.py`, `verify_slim.py`, `bench_fast*.py`, `profile_stages.py` | `analyze_slim.py`, `analyze_fastpipe.py`, `aggregate_profile.py` |
| OOD / recovery | `eval_ood.py`, `train_lora.py`, `eval_recovered.py` | `analyze_ood.py` |
| closed-loop (alpasim) | `launch_alpasim_matrix.sh` (config per GPU), `launch_alpasim_shards.sh` (one config split over GPUs) + `merge_alpasim_shards.py` | `analyze_alpasim.py`, `analyze_collisions.py`, `analyze_longitudinal.py` |

Fixed data protocol: `outputs/split.json`, created once by `make_split.py` (seed 20260721) from
`notebooks/clip_ids.parquet` — train/val/test = 900/131/150, with the 50 calibration clips a subset
of train so test stays clean. Evaluation is paired-seed: baseline and pruned configs share the same
sampling seeds so noise cancels in paired differences. Because minADE deltas are heavy-tailed (one
broken config lands at 25 m), the median and Wilcoxon are the primary readings and the mean is
reported alongside rather than trusted alone.

## Evaluation harness (`experiments/evaluation/`, from 2026-08-07)

A second, larger data protocol drawn directly from the official dataset splits, superseding
`split.json` for anything reported as a result. `outputs/eval_sets/` holds the manifests and
`EVAL_SETS.md` documents them for external reproduction:

| set | n | drawn from | role |
|---|---:|---|---|
| `indist_500` | 500 | official val | in-distribution |
| `test_500` | 500 | official test | in-distribution, held out |
| `ood` | 1,533 | `reasoning/ood_reasoning.parquet` | long-tail, carries `gt_coc` reference reasoning |
| `calib_100` | 100 | official train | **calibration only** — importance/J are measured here, never evaluated |

All four are pairwise disjoint. `make_eval_sets.py` draws them by greedy distribution matching
against the parent split (country 4.0, platform_class 2.0, time_of_day 2.0, season 1.5, month 1.0,
radar_config 1.0); any prefix of the greedy order is itself matched, so `--num-clips N` needs no
separate draw. `build_cache.py` / `build_ood_cache.py` stream each clip
(`open_file(maybe_stream=True)` gives zipfile a seekable `HfFileSystem`, so only that clip's
members cross the wire — 500 clips in 24 min instead of ~1 TB of chunks) into a per-clip npz that
`sample_cache.load_cached` reads in ~0.4 s.

Seeds come from the clip, never the loop index: `sc.clip_seed(seed, clip_id)` =
`sha256(f"{seed}:{clip_id}")[:4]`. `run_eval.py` still derives them from the loop index, which is
why a `--clip-offset` shard and a whole run disagree there; `run_baseline.py` exists partly to
avoid repeating that.

- `run_baseline.py --set {indist,test,ood} --model {baseline,<slim dir>}` — the open-loop runner
  for every arm. On OOD it evaluates twice per clip: own rollout, and the curated `gt_coc`
  teacher-forced, giving `minADE_tf` / `nll_gtcoc`.
- `launch_arms.sh` — flock-guarded job queue (`init` / `worker <gpu>` / `status`) so a shared box
  can be filled with whatever cards are free and topped up later.
- `analyze_baseline.py` (one model, `--compare` for a second), `analyze_arms.py` (three arms +
  pre-registered gates).
- `run_depth_ablation.py` / `analyze_depth_ablation.py` — per-layer kv-only ablation.

Determinism: `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA init, `use_deterministic_algorithms`,
cudnn deterministic, TF32 off. Two runs agree bitwise **within one GPU architecture** — the same
clip and seed gave 0.286 on Ada and 0.291 on Blackwell, and 3–4% of clips produce different CoC
text across architectures. Keep every arm of a comparison on one architecture and record which.
`baseline_ada_*` is the unpruned model re-measured on Ada for exactly this reason (the paired
Ada−Blackwell difference turned out to be +0.0000 / +0.0001, p=0.82, but that was not knowable
in advance).

Output convention: every run writes `outputs/<exp_id>/` containing `config.json`, `metrics.json`,
`summary.txt`, and `plots/*.png`. Large rebuildable artifacts are gitignored: `slim_state.pt`
(15–17 GB each, rebuildable from `slim_meta.json`) and `jlens_vectors.pt` (~600 MB).
`slim_cocsafe_r30` and `slim_integrated_mag` currently have **no `slim_state.pt`** — deleted
2026-08-06 to reclaim space; their `slim_meta.json` is intact, so `make_slim.py` rebuilds them
in ~1.5 h each.

## Named pruning configs

These names are the vocabulary of `outputs/`, `reports/`, and the alpasim drivers. Built by
`make_slim.py --config <name>`:

| config | criterion | allocation | axes touched |
|---|---|---|---|
| `integrated_mag` | trajectory Taylor | late-heavy graded 30/50 + layer-35 kv-only | VLM + expert + KV1 (−3.25B) |
| `cocsafe_full_r20` / `_r30` | dual `max(rank I_traj, rank I_CoC)` | uniform 20% / 30% | VLM + expert + KV1 (−2.01B at r20) |
| `dual_uniform` | dual | uniform | VLM only (bit-identical to the grid cell) |
| `j_traj_full_r20` / `_r30` | `max(rank I_traj, rank J)` — **label-free** | uniform | VLM + expert + KV1 |
| `dual_u40_v2` / `j_traj_u40_v2` | dual / `max(rank I_traj, rank J)` | uniform 0.398563 | **VLM only** (−2.66B, 24.0%) |
| `traj_u40_v2` / `coc_u40_v2` / `j_u40_v2` | `I_traj` / `I_CoC` / `J`, single criterion | uniform 0.398563 | **VLM only** (−2.66B, 24.0%) |

The five `*_u40_v2` configs are one family: `make_slim.build_masks` dispatches on the
`_u40_v2` suffix and the stem names the criterion, so all five hold budget, allocation, expert
and KV identical and differ only in the within-layer score. All five remove exactly
2,657,452,032 params (19/32 Q heads and 7390/12288 MLP channels in every one of the 36 layers).
`traj` / `coc` / `j` (2026-08-12) are the single-criterion controls that say what each half of
`max(rank I_traj, rank X)` does alone; `rank_norm` is skipped for them because it is a per-layer
monotone map and `select_mask_ratios` only argsorts within a layer. Kept-set overlap is 75.7%
(traj–coc), 62.7% (traj–j), 75.6% (coc–j) on Q heads.

`dual_u40_v2` and `j_traj_u40_v2` (2026-08-09) are the one-factor pair: same matched budget, same
uniform allocation, expert and KV untouched, and both halves of the criterion drawn from the same
100 calibration clips (`importance_v2` + `jlens_v2`). Only `X` in `max(rank I_traj, rank X)`
differs, so the comparison isolates "does the J-lens replace the CoC labels?". Their kept sets
overlap 84.8% (Q heads) / 83.3% (MLP), so there is something to measure. Built with
`--config dual_u40_v2 --importance importance_v2` (and `--jlens jlens_v2` for the J arm); calling
`dual_u40_v2` with `importance_v1` reproduces shipped `slim_dual_uniform` bit-identically, which is
how the recipe was verified. The uniform ratio is **0.3985632694**, not 0.40 — it comes from
`run_grid.allocations()` matching `slim_integrated_mag`'s realized budget, and rounding to 0.40
moves 17 MLP channels per layer.

`j_traj` is the label-free twin of `cocsafe`: identical structure, ratio, and expert/KV axes, with
only the reasoning half of the criterion swapped from CoC-NLL Taylor to the J-lens score — so the
comparison is a one-factor test of "do we still need reference CoC text?". The reference lens run
is `jlens_coc32` (32 calibration clips; split-half stability 0.98/0.95) — the original 8-clip
`jlens_coc` carries ~25% pick churn, but the shipped checkpoints stand: the `max(rank, rank)`
guardrail absorbs the width churn (`jsweep32_summary`) and the KV-1 group choice is insensitive
(`kviso_v1`, kv32−kv8 p≈0.6), so nothing was rebuilt.

The grid in `run_grid.py` crosses criterion (`traj`, `dual`) with allocation (`uniform`, `late`,
`agree`, `depthprior`) at matched budget, holding layer-35 and KV fixed, precisely because
`integrated_mag` vs `cocsafe` differ in three factors at once and cannot be attributed. `depthprior`
uses no importance and no CoC information at all — it is the CoC-free control.

## Closed-loop evaluation (alpasim)

Open-loop minADE hid the failure that closed-loop rollouts exposed, so headline configs get run in
the sibling **alpasim** repo at `/home/cvlab21/project/chan/alpasim`, with logs under
`/home/cvlab21/project/chan/alpasim-runs/matrix_<config>/`.

```bash
bash experiments/head_analysis/launch_alpasim_matrix.sh <n_scenes> <n_rollouts> [config ...]
```

- A slim checkpoint becomes a driver by hardlinking `outputs/<cfg>/` into
  `/mnt/nvme1n1/ad_vla/data/alpasim/drivers/<cfg>/` (the `DRIVERS` variable in the launcher, passed
  as `defines.drivers=` — that is the docker bind-mount source). The wizard then receives it as
  `driver.model.checkpoint_path=/mnt/drivers/<cfg>` (baseline passes no override).
  **Hardlinks cannot cross filesystems**, so the driver dir must stay on the same mount as
  `outputs/`; both now live under `/mnt/nvme1n1/ad_vla`. A driver dir left on another mount would
  silently pin a second 17 GB copy instead of sharing the inode.
- The GPU map is fixed per config in the script: drivers on Ada cards 4–7, renderer/physics/trafficsim
  on Blackwell 2–3. Keep new configs on **Ada** drivers — moving one to Blackwell reintroduces the
  kernel confound the map exists to avoid. Launches are staggered 120 s so the first run populates
  the shared scene cache, which lives off-repo at `/mnt/nvme1n1/ad_vla/data/nre-artifacts` (a
  symlink inside `data/` dangles in the containers).
- **`launch_alpasim_shards.sh` does not use that split** — it puts driver, renderer, physics and
  trafficsim of one shard all on the *same* card (`services.*.gpus=[$gpu]`). Every shipped
  150-scene result (`m2601_merged_*`) came from that path, four Ada cards, `OMP_NUM_THREADS=8`,
  so reproducing or extending those runs needs no Blackwell card at all — check the shipped
  `docker-compose.yaml`'s `device_ids` before assuming the matrix map applies.

The three `analyze_*` scripts run under **alpasim's** venv, not this repo's, because they need
`alpasim_utils` / `pandas` to read the ASL protobuf logs and per-rollout `metrics.parquet`:

```bash
cd /home/cvlab21/project/chan/alpasim && uv run python \
    /home/cvlab21/project/chan/alpamayo-model-compression/experiments/head_analysis/analyze_alpasim.py \
    --runs-root /home/cvlab21/project/chan/alpasim-runs --out <abs>/outputs/alpasim_eval
```

They aggregate per-rollout → per-scene mean → paired delta vs baseline with bootstrap CI and
Wilcoxon. `analyze_collisions.py` does per-collision forensics (does CoC degeneracy *concentrate*
in the 5 s before a crash?); `analyze_longitudinal.py` exists because at-fault collisions are too
rare to power a count, so it re-reads the same rollouts as continuous surrogates (time headway,
proximity exposure, braking response, speed at closest approach).

### Sharding one config over several GPUs (2026-08-10)

`launch_alpasim_matrix.sh` gives each config a card, so wall-clock is the slowest config and an
unpruned baseline drags the whole matrix. `launch_alpasim_shards.sh <config> <n_scenes>
<n_rollouts> "<gpus>"` runs one config across every free card instead, then
`merge_alpasim_shards.py --shards ... --out ... --expect-scenes N` stitches the runs back into one
directory (`analyze_alpasim.py` reads only `aggregate/results-summary.json` plus
`rollouts/<scene>/<rollout_id>/rollout.asl`, so merging is concatenating the `rollouts` array and
hardlinking the per-rollout dirs). Scores are copied verbatim, never recomputed; a shard without an
`aggregate/` is refused rather than salvaged. Verified by merging a stored 30-scene run and
reproducing its published numbers exactly.

Four things about this that will bite again:

- **Never shard with `scenes.scene_ids`.** 159 of `public_2601`'s scene_ids also exist in the 26.04
  release and `query_by_scene_ids` resolves a bare scene_id to the *newer* uuid — it silently swaps
  in 26.04 renders and downloads them. Only the suite CSV carries the uuid, so shards are expressed
  as generated per-shard test suites appended to `scenes.suites_csv`.
- **`OMP_NUM_THREADS` is worth 2.3×.** Unset, torch opens one intra-op thread per core (64) and the
  OpenMP barriers busy-wait, so four drivers put 256 spinning threads on 64 cores: 28.4 min/scene,
  11.6 cores burned per driver, load 81, GPUs at 0–64%. With `DRIVER_OMP_THREADS=8` the same run is
  12.5 min/scene, 1.1 cores per driver, load 16. Thread count changes CPU reduction order, so a
  matrix must not mix settings — the launcher makes it opt-in for that reason.
- **A killed run loses its scores.** `aggregate/` is written only on completion, and
  `alpasim-reeval` cannot regenerate it here: we pass `ALPASIM_ASL_SKIP_IMAGES=1`, so the ASL has no
  camera frames, the image scorer never produces `img_is_black`, and `processing.py`'s hardcoded
  `RemoveTrajectoryWithEvent(pl.col("img_is_black") > 0.0)` fails. 84 scenes were lost to this.
- **Startup is racy.** Four stacks coming up inside ~70 s made one runtime's gRPC version probe
  exceed its deadline; the run aborted with renderer exit 137, which looks like OOM but is the
  SIGKILL after the cascade. Relaunch that shard with `ONLY_SHARDS=<i>` (same k-way split).

### Scene suites

`public_2601` (913 scenes) and `public_2604` (1,606) share 159 scene_ids but **zero artifact
uuids** — 2604 re-rendered everything under the 26.04 release. Results from the two are not
comparable, they need separate `PREFIX`es, and the caches are separate
(`data/alpasim/nre-artifacts` is the 1.7 TB 2601 tree; `data/nre-artifacts` is ours). The
2026-08-11 matrix uses **2601** because 2604 would have needed ~100 GB of new downloads for 150
scenes while 2601 was fully local; its 913 usdz are hardlinked into our cache (free, same
filesystem, but `du` on that directory now double-counts).

Scene selection is `sorted(artifacts, key=scene_id)[:N]`. scene_id is `clipgt-<UUID>`, so that is an
unbiased random draw, deterministic, and **nested** — raising N extends the same sample. There is no
scenario metadata to stratify on (`sim_suites.csv` has only test_suite_id/scene_id/uuid), and only
~22% of scene ids join the PhysicalAI-AV clip index. Power: the per-scene paired delta has
σ ≈ 0.32, so N=30 resolves 0.179 (which is why the 2026-07 matrix was uniformly non-significant),
N=150 resolves 0.080, N=250 resolves 0.062. Half the scenes score exactly 1.0 for the baseline, so
report the bootstrap mean CI as primary and Wilcoxon as secondary.

Throughput on this box, `OMP_NUM_THREADS=8`, four stacks: 12.5 min/scene unpruned, 11.6 min/scene
for an 8.4B slim model — 150 scenes over 4 GPUs is ~8 h per config. Open-loop, for contrast, is
8 s/clip in-distribution and 13 s/clip on OOD (two conditions), i.e. ~2 h per config for all
2,533 clips.

## Plans and reports

Per the global workflow rule, non-trivial work starts with a plan `.md` in `plans/`
(`YYYY-MM-DD_topic.md`) that states the hypothesis, pre-registers the decision gates, and gets
approved before implementation. `run_*` docstrings then restate those gates so a run's own file
says what would falsify it.

Reports are self-contained HTML files named `YYYY-MM-DD_name.html` (never markdown), grouped by
track: `reports/archive/` holds the 29 reports of the 2026-07 pruning study, `reports/evaluation/`
the evaluation track that starts 2026-08-07. Author a `*_report_template.html` with
`PLOT::<key>::<file>.png` placeholders, then inline the plots as base64 data URIs:

```bash
python experiments/head_analysis/build_report.py <template.html> reports/<track>/<out>.html \
    key=$PWD/outputs/<exp_id>/plots [...]
```

The key must point at the `plots/` subdirectory, not the experiment directory. `build_report.py`
still hardcodes `REPO = /workspace/...`, so pass **absolute** paths on the host.

Plot styling (colors, background) lives at the top of `make_plots.py` and is duplicated in each
`analyze_*.py` header.

`reports/evaluation/` so far — each has its template under `experiments/evaluation/`:

| report | template | what it fixes |
|---|---|---|
| `2026-08-07_baseline-openloop.html` | `openloop_report_template.html` | the three open-loop sets and how they were drawn |
| `2026-08-09_criterion-oneshot.html` | `arms_report_template.html` | CoC Taylor vs J-lens, one-factor, 7,599 clips |
| `2026-08-11_baseline-anchor.html` | `baseline_anchor_report_template.html` | unpruned baseline in both modes + the paper table |
| `2026-08-11_criterion-closedloop.html` | `closedloop_report_template.html` | the same one-factor pair in alpasim, 150 scenes × 3 arms |

`reports/evaluation/2026-08-11_baseline_table.tex` is the anchor table for the paper's experimental
section: protocol and baseline in one table, so every pruned config is reported as a delta against
its last column.
