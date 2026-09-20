# Plan: OOD-val open-loop qualitative visualization

## Purpose
Open-loop numbers on OOD-val are aggregate (minADE@6 / minFDE@6). This adds a
per-clip qualitative view so a failure can be *looked at*: what the front camera
saw, what trajectory the model rolled out versus ground truth, and what CoC text
it produced versus the curated reference.

## What is already on disk (verified 2026-09-19)
- Manifest: `outputs/eval_sets/ood_val.parquet` — 262 rows,
  columns `clip_id, t0_us, gt_coc, cluster, split, chunk`.
- Per-clip predicted trajectories: `outputs/<arm>_pred_oodval/<model>_s{i}of4.json`,
  4 shards summing to 262 rows. Key `pred_xy_k` is `(8, 64, 2)` float — 8 samples,
  64 future steps, xy. Rows also carry `gen_coc`, `gt_coc`, `bucket`, `cluster`,
  `minADE_rollout`, `minFDE_rollout`, `ade_rollout_k` / `fde_rollout_k` `(8,)`,
  plus h16/h32 horizon variants and the teacher-forced `*_tf` twins.
  Arms with a `pred_oodval` dump, each covering 262/262: `baseline`,
  `dual_u40_v2`, `tyrK`, `tyr_u40_r`.
- Frames + ground-truth future:
  `/mnt/nvme1n1/ad_vla/data/physicalai_av/pre_processed/ood/samples/<clip_id>__t0_<t0_us>.npz`.
  All 262 present. Keys: `jpeg_bytes (16,)` object at 1920x1080,
  `n_cam=4`, `n_frame=4`, `camera_indices=[0,1,2,6]`,
  `ego_history_xyz (1,1,16,3)`, `ego_future_xyz (1,1,64,3)`, `gt_coc`, `cluster`.
  JPEG order is camera-major: index `c * n_frame + f`, per `sample_cache.load_cached`.
  Camera index 1 is the front camera; frame `f = n_frame - 1` is nearest t0.

Note: the plain `*_oodval` / `*_ood` dirs (e.g. `baseline_ada_ps_oodval`) hold the
same metrics and CoC but **no** `pred_xy_k`. Only the four `*_pred_oodval` dirs
can be used for trajectory plots.

## Non-goal
The other `*_oodval` arms cannot get trajectory panels without re-running eval
with the prediction dump enabled. Out of scope here.

## Design
`experiments/evaluation/viz_openloop_ood.py`, one figure per clip:

- **Left panel** — front camera frame nearest t0, with the clip id, bucket and
  cluster, and the generated CoC against `gt_coc` underneath.
- **Right panel** — bird's-eye trajectory. Ego history (16 steps), ground-truth
  future (64 steps) as a thick line, the 8 rollout samples thin, and the
  best-of-6 sample highlighted. Forward `x` on the vertical axis, lateral `y`
  on the horizontal, equal aspect.
- Title carries minADE/minFDE, both as stored (over 8) and recomputed best-of-6,
  which is the protocol the tables use.
- `--arms` takes several arms; each becomes one row of the figure so arms can be
  compared on the same clip.

Paths are absolute-by-default constants (`OUT`, `CACHE`) with CLI overrides,
because `outputs/` is an untracked symlink and is absent inside a git worktree.

## Outputs
`outputs/<exp-id>/` (default exp-id `viz_oodval`), per repo convention:
`config.json`, `summary.txt`, `plots/<clip_id>.png`.

## Test plan
Render a single clip for `baseline` and `dual_u40_v2`, confirm the frame decodes,
the GT and predicted trajectories are on the same scale, and the printed
minADE matches the stored `minADE_rollout` for that clip.
