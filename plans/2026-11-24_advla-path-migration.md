# Move AD-VLA code under `project/chan/advla/` and drop the old-path symlinks (2026-11-24)

## Goal
Keep `/home/cvlab21/project/chan` tidy: every AD-VLA checkout lives in `chan/advla/`, with no leftovers at the top level.

## What moved
alpamayo-model-compression, alpamayo2-distill, alpamayo2-edge, alpamayo-recipes, alpasim, alpasim-runs (itself a link to
`/mnt/nvme1n1/ad_vla/results/chan/alpasim-runs`), MoOP, LingoQA, LLM-Pruner, Tyr-the-Pruner, moop_pipeline.mp4,
hangang-server, kist-server, cvlab20-server. Same filesystem, so each move was a rename.

`world/`, the config and credential files, claude.sh, codex.sh and docker_exec.sh stay in `chan/`.

## Rules
- **Experiment records are never deleted or edited**, even where they spell out an old path. That covers logs, AlpaSim
  per-run configs and logs (`alpasim/outputs/`), wandb, `reports/`, `plans/`, `outputs/` and `open-loop-viz/` data.
  They record what was run.
- Rewrite `/home/cvlab21/project/chan/<name>` → `/home/cvlab21/project/chan/advla/<name>` only in live files: code
  (.py, .sh, .bash) and usage docs.
- Other users' files that reference our old paths are not touched. They will break, which the user accepted (they ran
  almost all of these experiments).

## Scope (scan of 2026-11-24)
| target | files | action |
|---|---|---|
| this repo: experiments, iclr27-rebuttal, paper-analysis, cvpr27, closed-loop-viz, minitron, CLAUDE-ARCHIVE.md | 166 | rewrite (this commit) |
| alpamayo2-distill/experiments | 3 | rewrite + commit there |
| hangang-server, kist-server docs/scripts | 3 | rewrite |
| MoOP/.venv (bin scripts, .pth, direct_url.json) | 33 | rewrite |
| /mnt venvs alpasim-chan (bin scripts, .pth, editable finders, direct_url.json), alpamayo2-edge (.pth, direct_url.json) | 136 | rewrite |
| Claude memory notes | 10 | rewrite |
| records (logs, alpasim/outputs, wandb, reports, plans, ...) | 4,682 | untouched |
| alpasim/src/utils_rs/target (Rust build cache) | 52 | untouched; regenerated on rebuild |

## Then
Remove the 14 top-level symlinks. Check that:
- no live file references an old path;
- the venvs import;
- the closed-loop tools resolve `advla/alpasim-runs`;
- the Claude project-memory keys for `advla/...` paths (symlinked dirs in `.claude/projects`) still load.
