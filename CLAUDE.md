# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A research fork of NVIDIA's Alpamayo 1 (Alpamayo-R1) release, used as the working repo for
**model-compression research on Alpamayo 1.5** (structured pruning of attention heads, MLP channels,
and KV groups). Two distinct codebases live here:

- `src/alpamayo_r1/` — the upstream Alpamayo 1 inference release (reference only; Apache-2.0).
- `experiments/head_analysis/` — the actual research code. It does **not** import `alpamayo_r1`; it
  imports the `alpamayo1_5` package from the sibling repo at `/workspace/alpamayo1.5` and runs
  `nvidia/Alpamayo-1.5-10B`. That repo's own `CLAUDE.md` documents the 1.5 package (offline env
  vars, inference entry points).

`main` tracks upstream; research work happens on task branches (current: `head-attention-analysis`).
`experiments/`, `outputs/`, and `reports/` are untracked research artifacts.

## Environment & running experiments

There is no local venv. Experiment scripts run with the sibling repo's interpreter:
`/workspace/alpamayo1.5/.venv/bin/python`.

Preferred launcher on this shared GPU box (retries every 60s while GPUs are busy):

```bash
bash experiments/head_analysis/run_retry.sh <max_attempts> experiments/head_analysis/<script>.py [--gpu N] [args...]
```

`run_retry.sh` exports what a run needs: `HF_TOKEN` (read from
`~/.cache/huggingface/stored_tokens`, the `full_right` token has gated-dataset access),
`CUDA_DEVICE_ORDER=PCI_BUS_ID` (so `--gpu N` and `nvidia-smi -i N` refer to the same card on this
box), and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Each script documents its own usage
and flags in its module docstring.

GPU handling inside the scripts:

- `reserve_gpu(need_gb)` in `expert_per_clip.py` scans cards, actually reserves the memory in
  PyTorch's caching allocator (held for the process lifetime), and returns the first card with room;
  `--gpu` restricts the scan so parallel runs land on distinct cards.
- Importing `expert_per_clip` also installs a `transformers` hub patch forcing
  `local_files_only=True` for `nvidia/Cosmos-Reason2-8B` (gated repo, served from local cache).

Model weights and the PhysicalAI-AV dataset are gated on HuggingFace; weights are ~22 GB and
inference needs ≥24 GB VRAM. The upstream smoke test (Alpamayo 1, not used by the experiments) is
`python src/alpamayo_r1/test_inference.py`.

Lint: ruff, line-length 100. Packages are managed with `uv`. There is no unit-test suite —
verification means running the experiment scripts and checking their outputs.

## Model architecture (what is being pruned)

Alpamayo pairs two transformer towers (see `src/alpamayo_r1/models/alpamayo_r1.py`; 1.5 has the
same shape on a Qwen3-VL backbone):

1. **VLM**: multi-camera frames + ego-history trajectory tokens (fused via `fuse_traj_tokens`) →
   autoregressive Chain-of-Causation (CoC) reasoning text.
2. **Expert**: a second transformer built from the VLM text config (no `embed_tokens`) that serves
   as the denoiser for flow-matching diffusion over actions: `action_in_proj` embeds the noisy
   action into 64 diffusion tokens → expert forward with the VLM's KV cache as prefill (non-causal
   over the diffusion tokens, `prompt_cache.crop()` after every denoise step) → `action_out_proj`
   predicts the vector field. Actions live in a unicycle accel/curvature space and decode to a
   6.4 s / 64-waypoint / 10 Hz trajectory.

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
- `eval_lib.py` — multi-sample minADE/minFDE, geometry-derived scenario buckets from the GT path
  (priority: decel_stop > turn > accel > cruise), paired bootstrap CIs.

Runners (`run_*.py`) follow the phase progression: `run_analysis` (attention stats + Taylor
passes), `run_importance` (P1 dual-objective importance), `run_ablation` / `run_expert_ablation`
(P2 mask sweeps), `run_eval` (multi-sample eval of headline configs), `run_integrated` /
`run_cocsafe` / `run_kvfix` / `scan_buckets` (P3 combined configs and robustness checks).
`analyze_*.py` / `make_plots.py` / `aggregate_profile.py` post-process an experiment's outputs.

Fixed data protocol: `outputs/split.json`, created once by `make_split.py` (seed 20260721) from
`notebooks/clip_ids.parquet` — train/val/test = 900/131/150, with the 50 calibration clips a subset
of train so test stays clean. Evaluation is paired-seed: baseline and pruned configs share the same
sampling seeds so noise cancels in paired differences.

Output convention: every run writes `outputs/<exp_id>/` containing `config.json`, `metrics.json`,
`summary.txt`, and `plots/*.png`.

## Reports

Reports in `reports/` are self-contained HTML files named `YYYY-MM-DD_name.html` (never markdown).
Author a `*_report_template.html` with `PLOT::<key>::<file>.png` placeholders, then inline the
plots as base64 data URIs:

```bash
python experiments/head_analysis/build_report.py <template.html> reports/<out>.html key=outputs/<exp_id> [...]
```

Plot styling (colors, background) lives at the top of `make_plots.py`.
