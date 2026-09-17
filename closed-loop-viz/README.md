# closed-loop-viz

Top-down replay of stored alpasim rollouts. No GPU, no re-run: everything is read back
out of `rollout.asl`, which each 150-scene run already keeps per rollout (~3 MB).

## Why top-down and not camera

The launchers pass `ALPASIM_ASL_SKIP_IMAGES=1`
(`alpasim_runtime/event_loop.py:68` — *"LogWriter that drops driver camera frames"*), so the
logs carry `batch_render_request` entries with **no matching return**: the render requests
were recorded and the images were thrown away. There is no camera footage to recover.
Getting it would mean re-running the sim with images enabled — GPU, ~12 min/scene, and a
*different* rollout from the one that produced the published score.

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

The renderer asserts the two arms' GT paths agree to 0.01 m before drawing. If they do not,
the panels are not the same scene and it refuses rather than producing a misleading pair.

## Usage

```bash
cd /home/cvlab21/project/chan/alpasim && uv run python \
  <repo>/closed-loop-viz/render_replay.py \
  --scene clipgt-2431387d-b9c5-4241-a219-497a99d1f4e5 \
  --arm "dual=slim_dual_u40_v2" --arm "dual+h4=slim_dual_u40_qcut4_v2" \
  --out <repo>/closed-loop-viz/video/dual_vs_h4_2431387d.mp4
```

It runs under **alpasim's** venv (it needs `alpasim_utils` to read the protobuf log), CPU
only — verified with `CUDA_VISIBLE_DEVICES=""`. About 0.2 s to parse a rollout; a 200-frame
two-panel video takes a couple of minutes to encode and lands at ~1.6 MB.

`--worst` (default) shows each arm's worse of the scene's two rollouts, since the failure is
usually the thing worth watching. `inspect_scene.py` prints per-arm score and gates for one
scene, which is how a scene worth rendering gets picked.

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
degeneracy** on all three open-loop sets — its reasoning is perfectly fluent — and it is
still the worst-driving dual variant. This scene shows the shape of that: the sentence is
well-formed and the reading of the scene is wrong. It is an illustration of a
already-measured aggregate effect (−0.091 [−0.134, −0.050] vs `dual`, p=3e−06), **not
evidence on its own** — one rollout of one scene cannot establish a mechanism, and the
`offroad` gate over all 150 scenes does not separate the two arms (20 → 28 hits, p=0.087).
