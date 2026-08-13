# Evaluation data: provenance, sampling, and reproduction

Every subset used for evaluation, how it was drawn, and the measured quality of each
draw. Written to be quotable in the paper's experimental-setup section. All numbers
here were measured on this machine, not estimated; the command that produces each one
is given alongside it.

Dates are absolute. "Official split" always means the `split` column of
`clip_index.parquet`; "our split" means `outputs/split.json`.

---

## 1. Source dataset

`nvidia/PhysicalAI-Autonomous-Vehicles`, local copy at
`/mnt/nvme1n1/ad_vla/data/physicalai_av`.

| | clips |
|---|---:|
| official `train` | 153,625 |
| official `val` | 90,928 |
| official `test` | 61,599 |
| total | 306,152 |

Camera data ships as per-chunk zips, four cameras per chunk
(`camera_front_wide_120fov`, `camera_front_tele_30fov`, `camera_cross_left_120fov`,
`camera_cross_right_120fov`); Alpamayo-1.5 consumes all four.

**The official `test` split is used, streamed rather than downloaded.** None of the 550
locally held chunks are test (350 train, 200 val), which initially read as "test is
unavailable" — wrongly. The HuggingFace repo carries all 3,146 camera chunks including
the test range, and a test clip streams with its GT trajectory intact. Evaluation
therefore reports **both** official val and official test, 500 clips each; agreement
between them is itself a result (§5).

---

## 2. Stage 1 — chunk-level selection of val (200 chunks)

Performed before this work; recorded in
`metadata/val_representative_sampling_manifest.json`.

```
method     chunk-level greedy distribution matching
seed       42
target     full-val clip-level distributions (90,928 clips / 934 chunks)
attributes country 4.0 | platform_class 2.0 | time_of_day 2.0
           season 1.5 | month 1.0 | radar_config 1.0     (weights)
selected   200 chunks / 18,868 clips  (21.4% of val chunks, 20.8% of val clips)
```

`time_of_day` is derived as `daytime` for `6 <= hour_of_day < 18` else `nighttime`;
`season` from `month` (12/1/2 winter, 3/4/5 spring, 6/7/8 summer, else fall). Those
definitions are shared with `/mnt/nvme1n1/ad_vla/code/02_select_clips.py`.

Distribution match against full val, as a function of the number of chunks
(`metadata/val_representative_sampling_quality.csv`, L1 distance per attribute):

| chunks | clips | country | platform | time_of_day | season | month | radar |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 4,450 | 0.0939 | 0.0047 | 0.0031 | 0.0214 | 0.0358 | 0.0177 |
| 100 | 9,187 | 0.0496 | 0.0065 | 0.0023 | 0.0131 | 0.0242 | 0.0175 |
| **200** | **18,868** | **0.0315** | **0.0014** | **0.0017** | **0.0054** | **0.0084** | **0.0097** |
| 300 | 28,569 | 0.0166 | 0.0003 | 0.0001 | 0.0060 | 0.0117 | 0.0045 |

25 countries are represented at every size from 50 chunks up. 200 was the operating
point; the gain from 200 to 300 is small relative to the storage it costs.

---

## 3. Stage 2 — replacing chunk zips with a per-clip cache

The evaluation pipeline reads exactly six fields off the dataset loader
(`image_frames`, `camera_indices`, `ego_history_{xyz,rot}`, `ego_future_{xyz,rot}`);
timestamps are never used. `pre_processed/eval` stores those per clip as an npz
holding 16 JPEGs (4 cameras x 4 frames) plus the ego trajectory — 2.8 MB per clip
against ~57 MB of mp4.

Coverage was verified exhaustively before anything was deleted:

- 18,868 / 18,868 val clips present, 0 missing, 0 undersized, all at
  `t0 = 5,100,000 us` — the same t0 every in-distribution runner uses.
- `sample_cache.load_cached` was compared against a direct
  `load_physical_aiavdataset` call on the same clip:

  | field | result |
  |---|---|
  | `ego_history_xyz`, `ego_history_rot` | bitwise identical |
  | `ego_future_xyz`, `ego_future_rot` | bitwise identical |
  | `camera_indices` | bitwise identical |
  | `image_frames` | same shape; mean abs difference 1.04/255, max 48 |

  The frame difference is the JPEG round-trip. Every model in a comparison reads the
  same cache, so paired differences are unaffected.

**2026-08-06: the 200 val chunks (800 zips, 0.97 TB) were deleted** once the above
verified. Val clips now load from the cache in 0.37 s instead of ~40 s. The 350 train
chunks remain on disk; the `pre_processed/train` cache was dropped because it was
locked to t0=5.1 s.

Consequence to state in the paper: in-distribution and OOD evaluation both read
JPEG-round-tripped frames. Re-deriving anything at a different t0, frame count, or
camera set requires re-streaming from HuggingFace (which works — see §4).

---

## 4. OOD evaluation set

Source: `reasoning/ood_reasoning.parquet`, 1,740 clips with a curated
chain-of-causation text and an `event_start_timestamp` marking when the out-of-
distribution situation occurs. Nine `event_cluster` categories.

OOD samples must be loaded at the **event timestamp**, not at t0=5.1 s, so the
in-distribution cache cannot serve them. `build_ood_cache.py` builds a separate cache.

The 1,740 OOD clips are spread over 1,085 chunks (1.60 clips per chunk), 840 of which
were not on disk — downloading them would have moved 4.28 TB. Instead each clip is
streamed: `physical_ai_av`'s `open_file(maybe_stream=True)` hands `zipfile` a seekable
`HfFileSystem` object, so only that clip's members are fetched. Build cost was 1.08 h
on 16 workers for 8.48 GB of output.

Attrition, exactly:

```
1,740   rows in ood_reasoning.parquet
   -9   `events` column is NaN -> no event timestamp exists
------
1,731   clips with a usable (event_start_timestamp, coc) pair
 -195   event too early in the clip: fewer than 16 ego-history steps available
   -3   event beyond the clip's interpolation range
------
1,533   cached samples          (8.48 GB, 0 schema anomalies)
```

One further clip failed on a transient HTTP timeout and was recovered by re-running
(the builder is resumable). All 198 remaining failures are properties of the data, not
of the pipeline.

Composition of the 1,533:

| | clips |
|---|---:|
| official `val` — **primary** | **262** |
| official `train` — secondary | 1,271 |

| event cluster | clips |
|---|---:|
| WORK_ZONES_TEMP_TRAFFIC_CONTROL | 739 |
| PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY | 352 |
| SPECIAL_OR_UNCOMMON_VEHICLE_BEHAVIOR | 221 |
| CYCLISTS_AND_MICROMOBILITY_COMPLEX | 68 |
| COMPLEX_INTERSECTION_INTERACTION | 47 |
| OTHER_LONGTAIL | 36 |
| EMERGENCY_INCIDENT_SCENE | 27 |
| ANIMALS_BIRDS_ROADKILL | 22 |
| ROAD_DEBRIS_OR_SAFETY_TRACES | 21 |

The 1,271 train-split OOD clips are reported separately from the 262 val ones. We do
no training, so a train-split clip is still a zero-shot evaluation — but the split is
kept visible so a reader can restrict to val alone.

---

## 5. In-distribution evaluation set (N = 500)

**Pool.** Official val, present in the cache, minus two exclusions:

```
18,868   cached val clips
  - 84   clips appearing anywhere in ood_reasoning.parquet
------
18,784   after also removing the calibration clips
```

All 1,740 OOD clips are excluded, not merely the 290 val ones, so the in-distribution
and OOD sets are disjoint by construction. Calibration clips are excluded because the
pruning criterion is fitted on them (§6).

**Draw.** Greedy: starting from the empty set, repeatedly add the clip that most
reduces the weighted L1 distance between the selection's attribute distribution and
the **full-val** clip-level distribution — the same target and the same six weighted
attributes as Stage 1. Adding one clip moves exactly six cells, one per attribute, so
each step's delta is evaluated vectorised over the whole pool. Ties are broken by a
seeded RNG, making the draw reproducible.

Measured match (L1 against full val):

| N | weighted L1 | country | platform | time_of_day | season | month | radar | countries |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 0.0165 | 0.0377 | 0.0009 | 0.0008 | 0.0054 | 0.0177 | 0.0101 | 25 |
| **500** | **0.0056** | 0.0122 | 0.0009 | 0.0012 | 0.0026 | 0.0056 | 0.0024 | **25** |
| 1000 | 0.0030 | 0.0057 | 0.0011 | 0.0008 | 0.0016 | 0.0041 | 0.0015 | 25 |

Reference points: simple random sampling at N=500 gives weighted L1
**0.0821 ± 0.0112** (20 draws) — the greedy draw is ~15x closer. The chunk stage
itself sits at 0.0144, so at N=500 the clip-level draw is no longer the limiting
factor in the two-stage match.

**Why N = 500.** Power, computed from the paired per-clip `dminADE` spread observed in
the existing 80-clip sweep (median SD across five configs = 0.494 m; baseline minADE
0.891 m). Minimum detectable effect at alpha=0.05, power=0.80:

| N | MDE | vs baseline |
|---:|---:|---:|
| 80 | 0.155 m | 17.4% |
| 200 | 0.098 m | 11.0% |
| **500** | **0.062 m** | **6.9%** |
| 1000 | 0.044 m | 4.9% |

Against effects actually observed at this scale — `cocsafe_r30` +0.210 m,
`j_traj_r30` +0.059 m, `traj_r20` +0.022 m, `j_traj_r20` +0.003 m — N=500 resolves the
r30-scale degradations and supports an equivalence bound of roughly +-0.043 m
(half-width of the 95% CI) for the r20-scale nulls. It does **not** resolve differences
below ~0.05 m; claims in that range must be stated as equivalence within a bound, not
as "no difference". This SD is inflated by the heavy tail, and the repo's primary
readings are the median and Wilcoxon, which have more power here — so 0.062 m is a
conservative figure.

---

## 6. Calibration set

The 50 calibration clips are the samples over which parameter importance is measured:
`run_importance.py` accumulates dual-objective Taylor scores (CoC NLL and
flow-matching MSE) across them, and those scores decide which heads and channels are
removed. They must not appear in any evaluation set.

The current set is `outputs/split.json`'s `calib`, defined as `train[:50]` of *our*
split — which predates the move to official splits and is contaminated under it:

```
50 calibration clips -> official train 22 | official val 16 | official test 12
                        (and 2 are OOD clips)
```

**Redrawn 2026-08-08** as `outputs/eval_sets/calib_100.parquet`: 100 clips from official
train only (153,625 − 1,450 OOD = 152,175 pool), greedy distribution matching, seed 42,
weighted L1 0.0306 against full train versus 0.1776 for a random draw, 24 countries.
Disjoint from `val_500`, `test_500` and `ood`; cached at `pre_processed/calib`
(100/100, 0.50 GB). `run_importance.py` and `run_jlens.py` both read it via
`--calib-manifest`, defaulting to `calib_100`, and both now derive their seeds from the
clip id rather than the loop index and pin the model revision. The old 50-clip set stays
reachable with `--calib-manifest ""` so earlier runs remain reproducible.

**Size.** The only direct evidence available is a 30-clip run against a 50-clip run over
nested samples (`importance_v1_n30` vs `importance_v1`; note their `config.json` both
report 50 because it is written before the loop — `metrics.json` carries the true
counts, 30 and 50). Fraction of the keep-set that agrees between the two:

| score | r=0.20 | r=0.30 | r=0.40 |
|---|---:|---:|---:|
| `traj_vlm_q` | 97.0% | 95.7% | 95.5% |
| `traj_vlm_mlp` | 95.3% | 93.6% | 92.1% |
| `coc_vlm_q` | 97.3% | 95.2% | 94.3% |
| `coc_vlm_mlp` | 95.3% | 93.5% | 91.9% |

Going from 30 to 50 clips still moves 5-8% of the keep set at the operating ratio, so
50 is not converged. For comparison, the J-lens criterion at 32 clips reaches
split-half reliability 0.98/0.95, while 8-clip selections carry ~25% churn.

**Recommendation: 100 clips, and save the per-clip importance tensors** (~6 MB/clip,
~600 MB total, gitignored). `run_importance.py` currently keeps only per-clip
diagnostics (`coc_len`, `fm_loss`, `peak_gb`), which is why the convergence curve
cannot be computed from existing runs. With per-clip tensors saved, the curve becomes a
reportable figure and the choice of n stops being a guess.

---

## 7. Seeds and determinism

Two sources of randomness: CoC generation is stochastic sampling
(`do_sample=True, temperature=0.6, top_p=0.98`), and the flow-matching denoiser draws
initial noise.

**Defect in the existing runners.** `run_eval.py` derives both from the loop index
(`torch.manual_seed(args.seed + ci)`, `seeds = [args.seed + ci*100 + k]`). With
`--clip-offset`, the same clip receives a different seed depending on how the run was
sharded, so a sharded run and a whole run do not agree.

**Fix.** Derive from the clip id, so order, sharding and restarts are irrelevant:

```python
base = int.from_bytes(hashlib.sha256(f"{GLOBAL_SEED}:{clip_id}".encode()).digest()[:4], "big")
```

with the K trajectory seeds as `base + k`. Plus the full determinism set:
`torch.use_deterministic_algorithms(True)`, `cudnn.deterministic=True`,
`cudnn.benchmark=False`, TF32 off for matmul and cudnn,
`CUBLAS_WORKSPACE_CONFIG=:4096:8` and `PYTHONHASHSEED=0` exported before the process
starts, and seeded `numpy`/`random`.

Bitwise reproducibility holds **within one GPU architecture only** — different
architectures use different kernels and reduction orders. Runs record
`torch.cuda.get_device_name`.

**Measured, 2026-08-07.** Twenty OOD clips were evaluated twice as independent
processes on the same card (RTX 5880 Ada, `outputs/det_a` and `outputs/det_b`). Every
quantity matched bitwise:

| quantity | result |
|---|---|
| `minADE_rollout`, `minFDE_rollout` | identical |
| `minADE_tf`, `minFDE_tf` | identical |
| `nll_self`, `nll_gtcoc` | identical |
| `gen_coc` (sampled reasoning text) | identical |

The generated text matching is the stronger claim: CoC decoding is stochastic sampling,
so identical strings mean the sampling RNG is fully pinned, not merely that the
trajectory head is. No non-deterministic-kernel warnings were raised under
`warn_only=True`, so no operation silently fell back.

---

## 8. Evaluation protocol and what it does not measure

Per clip: **one** chain-of-causation rollout, then **eight** flow-matching trajectories
conditioned on that single reasoning, scored as `minADE@8` / `minFDE@8` — the minimum
error across the eight samples.

```
clip -> run_rollout()  -> 1 CoC          (sampled: temperature 0.6, top_p 0.98)
          -> denoise x8 -> 8 trajectories (different noise seeds, same CoC context)
              -> minADE = min over the 8
```

`minADE@k` is the standard trajectory-prediction metric and is used here for
comparability. Only the minimum over the eight samples is stored.

Reasoning is drawn once per clip while the trajectory head is sampled eight times, so
`minADE_rollout` is an end-to-end reading: each model reasons for itself and then drives
on that reasoning. Seeds are pinned per clip, so rerunning the same model reproduces
exactly. Comparisons across models are paired on clips and seeds.

`run_eval.py` (the earlier harness) instead has the unpruned model generate one CoC and
teacher-forces every config on it — comparable across configs, but it never exercises a
pruned model's own reasoning. The OOD set carries both readings, since it is also scored
under the curated GT CoC.

## 9. Reproducing

```bash
# OOD cache (resumable; skips samples already present)
python experiments/evaluation/build_ood_cache.py --workers 16

# in-distribution draw + both manifests + quality.csv
python experiments/evaluation/make_eval_sets.py --n-indist 500 --seed 42
```

Artifacts:

| path | contents |
|---|---|
| `pre_processed/ood/{samples,index.json,manifest.parquet,errors.json,summary.txt}` | OOD cache and its attrition record |
| `pre_processed/eval/` | in-distribution per-clip cache (pre-existing) |
| `outputs/eval_sets/indist_500.parquet` | selected clips + attributes |
| `outputs/eval_sets/ood.parquet` | OOD clips + cluster + gt_coc + split |
| `outputs/eval_sets/quality.csv` | L1/JSD per attribute, per N |
