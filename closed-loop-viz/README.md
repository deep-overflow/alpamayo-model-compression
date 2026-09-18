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

## video/matrix_hard100 — 98 scenes × 4 arms

`dual` / `tyr` / `llm-pruner` (`lp_r50`) / `baseline`, each arm shown twice: whole scene and
an ego-following ±45 m view. 8 panels, 2160×1280, 392 MB.

The arm set differs from matrix150 because **hard100 has no `coc` and no `traj` run**. What
it does have, all on the same 100 scenes:

```
slim_tyrK          0.610      slim_tyr_u40_r     0.586      baseline           0.510
slim_dual_u40_v2   0.595      lp_r50             0.575      slim_wanda_u40_v2  0.406
```

`lp_r50` is under our own prefix here (`h100_merged_lp_r50`) — only its 150-scene run sits
in soowon's runs_root. `slim_wanda_u40_v2` landed 2026-09-18; the repo's hard100 report
predates it and describes five arms.

The zoom matters more here than on the matrix: hard100 routes run 158–694 m, so the
whole-scene view alone leaves every car a speck. With `--zoom` an arm owns two panels, and
only the whole-scene half carries the CoC caption — the same sentence under both says
nothing twice.

98 and not 100: `clipgt-4bad2f63` and `clipgt-adb899bd` have no `rollout.asl` on either
rollout, so those two FAILs are the ceiling, not a defect.

## video/matrix150 — 150 scenes × 6 arms

All 150 scenes of the matrix, six panels each: `dual`, `coc`, `traj`, `llm-pruner`, `tyr`,
`wanda`. 475 MB total, ~3 MB and 20 s per scene, rendered in 36 min on 6 workers.

The `llm-pruner` panel is **`lp_r50`**, the real external baseline (0.810), read from
soowon's runs_root by absolute path. An earlier batch used `slim_spg_s1_uni_nr` (0.733) as a
stand-in because a search of our own `m2601_merged_` prefix does not find `lp_r50` at all.

That absolute path found the same prefix bug a second time: `render_all.sh`'s scene-list step
also prefixed blindly, producing `m2601_merged_/mnt/nvme1n1/...` and killing the batch before
it rendered anything. Both places now pass an absolute path through unchanged.

The `video/` directory is gitignored — it rebuilds from `render_all.sh`.

## Top-down options

`--zoom R` adds an ego-following panel per arm, R metres either side. A 694 m route squeezes
the whole-scene view until the cars are specks; the pair keeps both readings on screen.

The **GT vehicle** is drawn as well as the GT path. It is placed by timestamp, not by frame
index: the recorded drive is a `PoseAtTime` series on the same clock as `actor_poses` (first
stamps agree exactly) but with a different length and rate — 202 poses against 200 frames.

With a single arm the CoC caption belongs to the figure, not to a panel: repeating the same
sentence under every panel says nothing and forces it small. Multi-arm renders keep their
per-panel captions, because there each arm says something different.

### Colour, decided by the validator rather than by eye

The GT path and GT car were `MUTED #6B6555`, which sits at normal-vision ΔE **14.7** from the
traffic grey `#8A8F98` — under the 15 floor, i.e. hard to tell apart *even with full colour
vision*, and the two are side by side constantly. They are now `#2166ac`, at ΔE 18.5, and
clear of the green driven path and the coral plan under every CVD simulation.

One known weakness is left in deliberately: plan `#D97757` against driven `#087f5b` is protan
ΔE **6.0** — the classic red/green pair. The alternatives trade it for something worse
(`#f08c00` fixes CVD at ΔE 14.8 but drops to 2.42 contrast on this cream ground, and these
are 1.6 px lines). The pair carries secondary encoding — different line weights and direct
legend labels — so it stays, recorded here rather than silently accepted.

Re-check with `scripts/validate_palette.js "<hex,...>" --mode light` from the dataviz skill.
The traffic grey fails that script's chroma floor on purpose: it is background context, not
a series, and the floor is a rule for series colours.

## Layout: video/<arm>/<suite>/{camera,topdown}

```
video/
  dual/
    origin150/camera   150 × 4-camera replay       1918×1202   2.1 GB
    origin150/topdown  150 × whole + zoom          1080×720    321 MB
    hard100/camera      98 × 4-camera replay                   1.2 GB
    hard100/topdown     98 × whole + zoom                      207 MB
  matrix150/             150 × 6-arm top-down      1620×1280
  matrix_hard100_gt10/    10 × 5-arm top-down
```

Files are `<score>_<scene>.mp4` with the score to three decimals, so listing a directory
sorts by score. It is the **scene** score — the mean over the scene's rollouts, the number
the tables use — not the score of the worse rollout the video shows; the two differ on a
split scene.

hard100 stops at 98, not 100. `clipgt-4bad2f63` and `clipgt-adb899bd` appear in
`aggregate/results-summary.json` but **both of their rollouts have no `rollout.asl` on
disk**, so there is nothing to replay and no other rollout to fall back to. Both scored
0.000. The renderers now say so plainly instead of raising a FileNotFoundError from inside
the protobuf reader.

`render_arm_all.sh` does a suite end to end (`ARM=`, `SUITE=`, `RUN=`, `SCENESET=`, `GPU=`);
existing files are skipped so an interrupted run resumes. `scene_scores.py` produces the
`<score>\t<scene>` list it works from.

Everything under an arm folder is that arm alone — a multi-panel comparison filed under
`dual/` would claim to be something it is not, so the multi-arm renders keep their own
directories. `restructure_by_arm.sh` builds this and takes `ARM=` plus a run path, so
another arm stacks up the same way.

### Which ten scenes, and the selection trap

`pick_gt_close.py` ranks by `dist_to_gt_trajectory` — the sim's own per-rollout measure,
the `d2gt` column in `analyze_alpasim`. **Ranking on it alone selects failures.** A rollout
that stops early never gets the chance to leave the GT curve:

```
1d205d02   d2gt 0.000   drove  16.2 m of 105.9 m  (15%)   score 0.000   offroad 2/2
05ecbbf5   d2gt 0.276   drove  15.1 m of 117.5 m  (13%)   score 0.000   offroad 2/2
```

Half of an unguarded top-10 was that shape. A scene qualifies only when the ego drove the
route (`dist_traveled_m / gt_dist_traveled_m ≥ 0.9`, `--min-progress-frac`) and is ranked by
d2gt after that.

```
origin150   150 scenes -> 146 with GT over 20 m -> 91 that drove it -> top 10
            d2gt 0.100-1.303 against a median of 4.078, all score 1.000, no gate hits
hard100      96 scenes ->  96                   -> 26 that drove it -> top 10
            d2gt 0.250-2.677 against a median of 3.841; #9 (d07c2655) is 1/2 at-fault
```

Only 27% of hard100 scenes were driven to completion against 62% of the matrix — the
suite's difficulty shows up in that filter. Its GT paths also run 158–694 m against
52–411 m, which is part of why its d2gt values sit higher.

`make_sceneset.py` exists because of hard100: its shards left four 25-scene scenesets and
no single one covers a selection drawn from the whole suite, so the ten needed usdz are
hardlinked into one `chan_`-prefixed sceneset (same filesystem, no extra space).

### Three bugs this batch turned up

**Frame tearing** — see below; the one that produced visibly broken video.

**The run name was prefixed twice.** `render_replay.py` expands a bare config under
`m2601_merged_`, so passing `RUN=m2601_merged_slim_dual_u40_v2` looked for
`m2601_merged_m2601_merged_...` and all 250 top-downs failed at once. A hard100 name would
have been given the wrong suite's prefix entirely. `render_arm_all.sh` now resolves RUN to
an absolute path itself and checks it exists, rather than leaving that as a rule the caller
has to remember.

**The child ate the scene list.** The camera loop was `while read ... done < list`, and the
python it spawns inherits stdin — so it consumed part of the list and `read` came back with
half a line: an empty scene, and a score field holding the whole record. It surfaced as
files named `000_clipgt-…` and `.000_clipgt-…` (the render itself was fine, only the prefix
was truncated) plus outright failures on the empty ones. The list is now read on fd 3 with
`</dev/null` for the child. `check_drift.py` had its own version of the same class of bug:
`select` matching nothing exits 0 and writes no file, so a stale PNG from the previous video
was compared — two 165-frame clips were reported torn when they were fine.

### The tearing bug, and why the first fix was not one

The first single-arm batch produced torn video — by frame 100 the bottom of the previous
frame appeared at the top. The cause is that `6.1 * nrow + 0.6` is `6.699999999999999` at
nrow=1, so the canvas came out **540×669**: an odd height. The writer declared one size to
ffmpeg and handed it buffers of another, and every frame slid by one row.

`matrix150` was never affected — at nrow=2 the same arithmetic lands on 1620×1280, both
even — which is why a six-panel frame looked right while the one-panel one did not.

An earlier `-vf scale=trunc(ih/2)*2` had been added when ffmpeg refused an odd height. That
**hid the error without fixing it**: rescaling 669 to 668 satisfies libx264 and leaves the
frame-size mismatch in place. The renderer now sizes from even integer pixels and asserts
the real canvas before encoding, so an odd dimension fails loudly instead of producing
plausible-looking garbage. Verified across all 210 videos: 0 odd dimensions.

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
