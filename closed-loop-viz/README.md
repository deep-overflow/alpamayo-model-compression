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

Measured: 199 frames in **17 s** (0.09 s/frame) at 1920×1080, 15 MB of mp4. The renderer
container takes ~35 s to come up.

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
