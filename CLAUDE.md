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

Preferred launcher on this shared GPU box (retries every 60s while GPUs are busy):

```bash
bash experiments/head_analysis/run_retry_host.sh <max_attempts> \
    experiments/head_analysis/<script>.py [--gpu N] [args...]
```

It exports what a run needs: `HF_TOKEN` (read from `~/.cache/huggingface/stored_tokens`; the
`full_right` token has gated-dataset access), `HF_HOME=$HOME/.cache/huggingface` (the shell default
points at a nearly-full `/mnt/nvme1n1` with no token file), `CUDA_DEVICE_ORDER=PCI_BUS_ID` (so
`--gpu N` and `nvidia-smi -i N` refer to the same card on this box), and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Each script documents its own usage and flags in
its module docstring.

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
| closed-loop (alpasim) | `launch_alpasim_matrix.sh` | `analyze_alpasim.py`, `analyze_collisions.py`, `analyze_longitudinal.py` |

Fixed data protocol: `outputs/split.json`, created once by `make_split.py` (seed 20260721) from
`notebooks/clip_ids.parquet` — train/val/test = 900/131/150, with the 50 calibration clips a subset
of train so test stays clean. Evaluation is paired-seed: baseline and pruned configs share the same
sampling seeds so noise cancels in paired differences. Because minADE deltas are heavy-tailed (one
broken config lands at 25 m), the median and Wilcoxon are the primary readings and the mean is
reported alongside rather than trusted alone.

Output convention: every run writes `outputs/<exp_id>/` containing `config.json`, `metrics.json`,
`summary.txt`, and `plots/*.png`. Large rebuildable artifacts are gitignored: `slim_state.pt`
(15–17 GB each, rebuildable from `slim_meta.json`) and `jlens_vectors.pt` (~600 MB).

## Named pruning configs

These names are the vocabulary of `outputs/`, `reports/`, and the alpasim drivers. Built by
`make_slim.py --config <name>`:

| config | criterion | allocation | axes touched |
|---|---|---|---|
| `integrated_mag` | trajectory Taylor | late-heavy graded 30/50 + layer-35 kv-only | VLM + expert + KV1 (−3.25B) |
| `cocsafe_full_r20` / `_r30` | dual `max(rank I_traj, rank I_CoC)` | uniform 20% / 30% | VLM + expert + KV1 (−2.01B at r20) |
| `dual_uniform` | dual | uniform | VLM only (bit-identical to the grid cell) |
| `j_traj_full_r20` / `_r30` | `max(rank I_traj, rank J)` — **label-free** | uniform | VLM + expert + KV1 |

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
  `alpasim/data/drivers/<cfg>/`; the wizard receives it as
  `driver.model.checkpoint_path=/mnt/drivers/<cfg>` (baseline passes no override).
- The GPU map is fixed per config in the script: drivers on Ada cards 4–7, renderer/physics/trafficsim
  on Blackwell 2–3. Keep new configs on **Ada** drivers — moving one to Blackwell reintroduces the
  kernel confound the map exists to avoid. Launches are staggered 120 s so the first run populates
  the shared scene cache, which lives off-repo at `/mnt/nvme1n1/ad_vla/data/nre-artifacts` (a
  symlink inside `data/` dangles in the containers).

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

## Plans and reports

Per the global workflow rule, non-trivial work starts with a plan `.md` in `plans/`
(`YYYY-MM-DD_topic.md`) that states the hypothesis, pre-registers the decision gates, and gets
approved before implementation. `run_*` docstrings then restate those gates so a run's own file
says what would falsify it.

Reports in `reports/` are self-contained HTML files named `YYYY-MM-DD_name.html` (never markdown).
Author a `*_report_template.html` with `PLOT::<key>::<file>.png` placeholders, then inline the
plots as base64 data URIs:

```bash
python experiments/head_analysis/build_report.py <template.html> reports/<out>.html \
    key=$PWD/outputs/<exp_id> [...]
```

Plot styling (colors, background) lives at the top of `make_plots.py` and is duplicated in each
`analyze_*.py` header.
