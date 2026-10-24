# clipgt-0f1f60d5-785b-4a47-8436-7046ad1a3d21 — qualitative vis bundle (origin150 / matrix150)

Everything is a symlink unless marked (copied/derived). Nothing outside eb_qual/ was modified.

## video/
- `matrix150_6arm_topdown.mp4` → video/matrix150 (dual, coc, traj, llm-pruner, tyr, wanda; worst rollout per arm)
- `<arm>/camera_<score>_*.mp4` — 4-camera NuRec replay of the rollout the model drove (= what the model saw:
  front_wide_120, front_tele_30, cross_left_120, cross_right_120), 1918x1202
- `<arm>/topdown_<score>_*.mp4` — whole + ±45 m zoom top-down, 1080x720
- Score in filename = scene score (mean of 2 rollouts); videos show the WORST rollout.

## runs/<arm>/
- `rollouts/` → the scene's rollout dirs (2 each): `rollout.asl` (full sim log: ego/actor poses, 6.4 s plans,
  CoC reasoning, render requests), `metrics.parquet` (per-timestep metrics), `_complete`
- `results-summary.json` → the run's aggregate (all 150 scenes)
- arms: dual=slim_dual_u40_v2, coc=slim_coc_u40_v2, traj=slim_traj_u40_v2, tyr=slim_tyr_u40_r, tyrK=slim_tyrK,
  wanda=slim_wanda_u40_v2, baseline, llm-pruner=soowon lp_r50, dual_mlp93.75=slim_dualexp_u40_em93p75
  (no video for dual_mlp93.75)

## metrics/ (derived)
- `rollout_metrics.csv` — one row per rollout: score, progress, dist traveled, offroad, collision, d2gt, ...
- `scene_metrics.json` — per arm: run dir, scene mean score, which rollout the videos show, full rollout records

## reasoning/ (derived from rollout.asl)
`<arm>__<rollout>__{WORST,best}.json`: gt_path_xy, ego_xy [t_s,x,y,yaw] per frame (200 @ 10 Hz),
and 181 driver steps with t_s, CoC `reasoning_text`, `plan_xy` (65 poses). Regenerate with `dump_scene.py`
(run under /home/cvlab21/project/chan/alpasim `uv run`).

## scene/
- `c788a915-....usdz` → NuRec scene (26.01 release, 1.78 GB; mesh, map.xodr, clipgt/*.parquet map+obstacles, ckpt)
- `recorded_frames/` (copied from usdz) — the only real recorded camera images in it: 1 JPEG per driver camera
- `extracted/` (copied from usdz) — metadata.yaml, data_info.json, map.xodr
- Raw PhysicalAI-AV footage for this clip is NOT on disk (clip_id absent from physicalai_av/clip_index.parquet).

## scene scores (mean of 2 rollouts)
dual 0.913 | coc 0.789 | wanda 0.762 | dual_mlp93.75 0.711 | tyrK 0.444 | tyr 0.437 | traj 0.424 | baseline 0.218 | llm-pruner 0.000
Static difficulty percentile 98.6 (ppt-fig/difficulty_scene_data.csv). GT path 57.8 m, ~134° heading change (right turn at signalized intersection).
