# closed-loop-viz

Top-down replay of stored alpasim rollouts. No GPU, no re-run: everything is read back out
of `rollout.asl`, which each run already keeps per rollout (~3 MB).

## The images were dropped, but they are recoverable without a re-run

The launchers pass `ALPASIM_ASL_SKIP_IMAGES=1`
(`alpasim_runtime/event_loop.py:68` — *"LogWriter that drops driver camera frames"*), so the
logs carry `batch_render_request` entries with **no matching return**: the render requests
were recorded and the images were thrown away.

An earlier version of this file concluded from that that camera footage needs a full re-run.
**That is wrong.** Each recorded item is already a complete `RGBRenderRequest` — scene_id,
ftheta intrinsics, the rolling-shutter sensor pose (start *and* end), the frame window, and
every dynamic object's pose for that frame — so the renderer can be replayed on its own and
the pixels come back. `replay_camera.py` does that; `render_replay.py` stays useful because a
top-down view shows the plan, the GT path and the whole scene at once, which no camera does.

## What is drawn, and from where

| element | source in `rollout.asl` |
|---|---|
| ego box + driven trail | `actor_poses` entry with `actor_id == "EGO"` |
| other traffic | the remaining `actor_poses` entries, oriented by quaternion |
| vehicle footprints | `rollout_metadata.actor_definitions.actor_aabb`, joined on `actor_id` |
| the 6.4 s plan held at each step | `driver_return.trajectory.poses` (65 poses) |
| GT reference path | `rollout_metadata.ego_rig_recorded_ground_truth_trajectory` |
| reasoning caption | `driver_return.debug_info.unstructured_debug_info` → `reasoning_text` |
| score, gates, distance | the run's own `aggregate/results-summary.json` |

Two frame facts the drawing depends on, both checked rather than assumed:

- **Ego, traffic, plans and GT share one frame.** `transform_ego_coords_rig_to_aabb` looks
  like a frame change but is a pure translation (`quat w=1`, x = 1.4485 m) between the rig
  origin and the box centre. Nothing is transformed here.
- **Sim timestamps are not comparable across rollouts** — two runs of the same scene differ
  by ~10¹² µs — so each rollout is replayed against its own `t0`.

The renderer asserts every arm's GT path agrees to 0.01 m before drawing. If they do not,
the panels are not the same scene and it refuses rather than producing a misleading set.

An empty caption reads *"(empty reasoning output)"*, not "not logged": the model emitted an
empty string, which is what CoC degeneracy looks like frame by frame. `wanda` degenerates on
45.78% of its closed-loop rollouts and that is what shows.

## Usage

```bash
cd /home/cvlab21/project/chan/alpasim && CUDA_VISIBLE_DEVICES="" uv run python \
  <repo>/closed-loop-viz/render_replay.py \
  --scene clipgt-2431387d-b9c5-4241-a219-497a99d1f4e5 \
  --arm "dual=slim_dual_u40_v2" --arm "tyr=slim_tyr_u40_r" \
  --out <repo>/closed-loop-viz/video/out.mp4
```

Runs under **alpasim's** venv (it needs `alpasim_utils` to read the protobuf log), CPU only —
verified with `CUDA_VISIBLE_DEVICES=""`. ~0.2 s to parse a rollout; a six-panel 200-frame
video takes ~90 s and lands at ~3 MB. `--arm` is repeatable; past three arms the panels wrap
into a grid. `--worst` (default) shows each arm's worse of the scene's two rollouts, since
the failure is usually the thing worth watching.

**`config` may be an absolute path** to a merged run directory, not just one of our run names
(which resolve as `m2601_merged_<config>`). That is how an arm living in someone else's
runs_root gets in — see LLM-Pruner below.

`inspect_scene.py` prints per-arm score and gates for one scene, which is how a scene worth
rendering gets picked. `render_all.sh` renders the whole 150-scene matrix (resumable; it
skips scenes whose mp4 already exists).

## Camera replay — `start_renderer.sh` + `replay_camera.py`

```bash
GPU=1 PORT=16007 bash closed-loop-viz/start_renderer.sh      # renderer alone, ~35 s to serve

cd /home/cvlab21/project/chan/alpasim && CUDA_VISIBLE_DEVICES="" uv run python \
  <repo>/closed-loop-viz/replay_camera.py \
  --scene clipgt-04749bb9-9b37-495b-bed0-77f0e33ac7da \
  --config slim_dual_u40_qcut4_v2 --camera camera_front_wide_120fov \
  --out <repo>/closed-loop-viz/video/camera.mp4

docker rm -f chan_nre_replay                                  # give the card back
```

**No driver, no physics, no trafficsim runs.** The renderer (NuRec) is the only model, and it
makes no decisions — it rasterises a view from a pose the log already fixed. So the footage is
of the rollout that produced the published score, and the GPU it runs on cannot change the
driving: a Blackwell card is fine while Ada stays with the closed-loop evaluations.

**All four cameras come back on every call**, so `--camera all` (the default) costs no extra
rendering — the driver requests `front_wide_120`, `front_tele_30`, `cross_left_120` and
`cross_right_120` in one batch, and keeping one tile instead of four only throws three away.
The scene also carries `rear_left_70` and `rear_right_70`; the driver never asks for them, so
they are not what the model saw and are not shown. Tiles are composited and written to disk
one frame at a time: holding 199 frames × 4 × 1900×1080 in a list is ~4.9 GB, and the encoder
does not need them at once.

Measured on GPU 1: 199 frames, four cameras, **33 s** (0.17 s/frame), 1918×1202, 5.5 MB of
mp4. One camera is 0.09 s/frame. The container takes ~35 s to serve, and the **first frame of
a cold scene costs ~65 s** while the usdz loads — that is the scene cache filling, not a
stall.

The caption at t = 0.0 s reads *"(empty reasoning output)"* and that is correct: the first
render request precedes the first `driver_return`, so the model has not spoken yet.

**The renderer only serves the scenes its `--artifact-glob` matched.** The sceneset in
`data/nre-artifacts/scenesets/<id>/` is whatever the last run materialised — it held 38 usdz
(one shard's worth), not all 913 — so check the `Available scenes:` line in
`docker logs chan_nre_replay` before choosing a scene, or point the glob at a sceneset that
has the one you want.

## LLM-Pruner (`lp_r50`) lives outside our runs root

`lp_r50` — the external LLM-Pruner baseline, built in soowon's `alpamayo1.5` tree — **does
have a full 150-scene closed-loop run**, but not under `chan/alpasim-runs`, so a search of
our prefix misses it:

```
/mnt/nvme1n1/ad_vla/outputs/soowon/alpasim-analysis/runs_root/lp_r50/
```

pointed at by `runs_root` in
`/home/cvlab21/project/soowon/vla-ad/alpamayo1.5/outputs/alpasim_lp_r50/config.json`.
Verified: 150 scenes / 300 rollouts, **the same scene set as our matrix**, same record
schema, and the GT-path check against our `dual` passes at 0.0000 m. It scores 0.810,
+0.0609 [+0.0132, +0.1080] vs baseline (p=0.0054).

Pass it as `--arm "llm-pruner=/mnt/nvme1n1/ad_vla/outputs/soowon/alpasim-analysis/runs_root/lp_r50"`.

## video/matrix150 — 150 scenes × 6 arms

All 150 scenes of the matrix, six panels each: `dual`, `coc`, `traj`, `llm-pruner`, `tyr`,
`wanda`. 475 MB total, ~3 MB and 20 s per scene, rendered in 36 min on 6 workers.

**The `llm-pruner` panel in this batch is `slim_spg_s1_uni_nr`** (`dual_param_first`, MLP
50.3%, 24.0% total, score 0.733) — one of soowon's masks run through our closed loop. It was
chosen before `lp_r50`'s 150-scene run was located; `lp_r50` is the better representative and
is what later renders should use.

The `video/` directory is gitignored — it rebuilds from `render_all.sh`.

## video/gt_closest10 — the ten scenes `dual` drove most like the GT

`pick_gt_close.py` ranks by `dist_to_gt_trajectory`, the sim's own per-rollout measure (the
`d2gt` column in `analyze_alpasim`); `render_gt10.sh` then renders them.

**Ranking on d2gt alone selects failures, and the guard against that is the point.** A
rollout that stops early never gets the chance to leave the GT curve, so it sits at d2gt ≈ 0
while having driven almost nothing. Unguarded, the top pick was:

```
1d205d02   d2gt 0.000   drove  16.2 m of 105.9 m  (15%)   score 0.000   offroad 2/2
05ecbbf5   d2gt 0.276   drove  15.1 m of 117.5 m  (13%)   score 0.000   offroad 2/2
278088cf   d2gt 0.429   drove  87.5 m of 304.4 m  (29%)   score 0.000   offroad 2/2
```

Half of an unguarded top-10 was that failure mode. A scene therefore qualifies only when the
ego actually drove the route (`dist_traveled_m / gt_dist_traveled_m ≥ 0.9`, `--min-progress-frac`)
and is ranked by d2gt only after that. 150 scenes → 146 with a GT path over 20 m → 91 that
drove it → top 10:

```
 #  scene       d2gt   score  driven/GT   GT m
 1  0dbb91dd   0.100   1.000    0.99      146.9
 2  25f6ad44   0.336   1.000    0.95      126.4
 3  13fb89b9   0.413   1.000    0.95      190.0
 4  0e02fb8c   0.450   1.000    0.99      115.7
 5  1920170b   0.629   1.000    0.99      243.0
 6  054b5901   0.908   1.000    1.00      132.3
 7  22a92557   1.033   1.000    1.01       52.3
 8  1a7b81b8   1.113   1.000    1.00      136.2
 9  04394343   1.286   1.000    0.92      299.2
10  1a5da90a   1.303   1.000    1.00      410.7
```

All ten pass with no gate hits, against a d2gt median of 4.078 over the qualifying scenes.

```
video/gt_closest10/
  camera/   <scene>.mp4   dual, all four cameras   132 MB
  topdown/  <scene>.mp4   the same ten from matrix150, six arms    30 MB
```

The top-down side is **gathered, not re-rendered**: `matrix150` already holds the six-arm
render for all 150 scenes, so these are the same files. Ten camera videos took 10 min on
GPU 1 with sceneset `6f937b0c67258133d8e372901b8f2aa8`, which is the one holding all 150
usdz — a per-shard sceneset holds 38 and would have missed most of the picks.

## video/dual_vs_h4_2431387d.mp4

`dual` vs `dual+h4` on the scene where they diverge most in the 150-scene matrix.

```
dual     score 1.000  pass      drove 64 m of 61 m
dual+h4  score 0.000  OFFROAD   drove 38 m of 61 m     (both of its rollouts, 38 m and 36 m)
baseline           split — one rollout 1.000, the other 0.000 (offroad)
```

At t ≈ 12 s the two arms are reading the same intersection differently:

- `dual` — *"Stop for the red traffic light"*, and it stops.
- `dual+h4` — *"Turn right at the intersection since the right-turn traffic light is green"*,
  and it turns right off the reference path and leaves the drivable area.

Read that against what the repo already knows about this arm: `dual+h4` has **0.0% CoC
degeneracy** on all three open-loop sets — its reasoning is perfectly fluent — and it is still
the worst-driving dual variant. This scene shows the shape of that: the sentence is
well-formed and the reading of the scene is wrong. It is an illustration of an
already-measured aggregate effect (−0.091 [−0.134, −0.050] vs `dual`, p=3e−06), **not evidence
on its own** — one rollout of one scene cannot establish a mechanism, and the `offroad` gate
over all 150 scenes does not separate the two arms (20 → 28 hits, p=0.087).
