# CLAUDE.md — Alpamayo 1.5 Model Compression Research

Project guidance for Claude Code. For archived experiment harness, runners, past workflows, and detailed configuration reference, see `CLAUDE-ARCHIVE.md`.

## What this repo is

Research fork of NVIDIA's Alpamayo 1 for **model-compression research on Alpamayo 1.5**:
- `src/alpamayo_r1/` — upstream inference release (reference only; Apache-2.0)
- `experiments/head_analysis/` — research code (imports `alpamayo1_5` package, runs `nvidia/Alpamayo-1.5-10B`)

Main branch: research merged 2026-08-06. New tasks branch from `main`.

## Environment & running experiments

Scripts run with repo's `.venv` → `alpamayo1_5` from `git+https://github.com/NVlabs/alpamayo1.5.git@f42e594`.

**All data/envs/results live under `/mnt/nvme1n1/ad_vla`** via symlinks (moved 2026-08-06; root was 97% full).

| path | location |
|---|---|
| `.venv` | `ad_vla/venvs/alpamayo-mc` |
| `outputs/` | `ad_vla/outputs/chan` |
| `alpasim/.venv` | `ad_vla/venvs/alpasim-chan` |
| `alpasim-runs/` | `ad_vla/results/chan/alpasim-runs` |
| HF cache | `ad_vla/cache/hub` |

**Check `df` before new checkpoints** (each 15–17 GB; mount near 98% full).

Preferred launcher (retries every 60s while GPUs busy):
```bash
bash experiments/head_analysis/run_retry_host.sh <max_attempts> \
    experiments/head_analysis/<script>.py [--gpu N] [args...]
```

Exports: `HF_TOKEN` (from `~/.cache/huggingface/stored_tokens`), `HF_HOME=$HOME/.cache/huggingface`,
`HF_HUB_CACHE=/mnt/nvme1n1/ad_vla/cache/hub`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, 
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**Token and blobs in different places since 2026-08-06**: never set only one of `HF_HOME`/`HF_HUB_CACHE`.

## Model architecture

Alpamayo pairs two transformer towers (Qwen3-VL backbone):

1. **VLM**: multi-camera frames + ego-history trajectory tokens (fused via `fuse_traj_tokens`) → autoregressive Chain-of-Causation (CoC) reasoning text
2. **Expert**: denoiser for flow-matching diffusion over actions; reads VLM's KV cache as prefill (non-causal over 64 diffusion tokens, cropped after each denoise step)

**Width differences**: Only `head_dim=128` and 8 KV groups match (expert interface requirement).

| tower    | hidden | Q heads | head_dim | intermediate | layers | params |
|----------|--------|---------|----------|--------------|--------|--------|
| VLM      | 4096   | 32      | 128      | 12288        | 36     | ~6.9B  |
| expert   | 2048   | 16      | 128      | 8256         | 36     | ~2.3B  |

**GQA coupling**: KV group `h` feeds VLM Q heads `[4h, 4h+4)` and expert Q heads `[2h, 2h+2)`.
A KV group is "removed" by masking its Q heads (not by zeroing K, which would spread softmax uniformly).

## Key libraries (incomplete; see `CLAUDE-ARCHIVE.md` for full library descriptions)

- `analysis_lib.py` — fused prompts, token spans (vision/traj/text per-camera), attention impl switching
- `prune_lib.py` — dual-objective Taylor importance (CoC NLL vs trajectory FM MSE)
- `mask_lib.py` — 0/1 masks on o_proj (Q heads) and down_proj (MLP channels)
- `eval_lib.py` — minADE/minFDE, scenario buckets from GT path, bootstrap CIs
- `slim_lib.py` — physical removal, state_dicts + `slim_meta.json` checkpoints

## Workflow rules

### Before running any experiment
- Confirm hypothesis: what are we testing?
- Verify setup matches purpose
- Check for confounding variables vs baseline

### During execution
- Verify first iterations, GPU memory, logs
- Monitor progress periodically

### Result reporting
- Save: `outputs/<exp_id>/` with `config.json`, `metrics.json`, `summary.txt`, plots
- Report: `reports/YYYY-MM-DD_name.md` (separate facts from analysis)

### Git workflow
- Branch per task, PR when done
- Run `uvx ruff check .` before PR
- Never force push or reset --hard without explicit confirmation

## Important notes

- Stale paths: `run_retry.sh` (older in-container), `/workspace/alpamayo-model-compression` (doesn't exist on host)
- Worktrees need `outputs/` and `.venv` symlinks recreated before running
- Determinism: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `use_deterministic_algorithms`, cudnn deterministic
- GPU policy: Ada 4-7 for eval, 0-3 OK for training (since 2026-08-22)
