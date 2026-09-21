# How `calib_100` was sampled

Verified 2026-09-20 against the drawing script, the records it saved, the manifest
itself, and the per-clip cache. Anything relayed from the project notes rather than
re-checked is marked as such in its own section.

## Summary

`calib_100` is 100 clips from the **official train split**, chosen by **seeded greedy
distribution matching** on six collection-metadata attributes. The draw minimises a
weighted L1 distance to the clip-level distribution of the whole train split. It used
seed 42 and a fixed window start of t0 = 5.1 s for every clip.

The sampler saw only collection metadata. It never looked at the driving itself, so the
action mix of the set is whatever fell out.

| fact | value |
|---|---|
| producer | `experiments/evaluation/make_eval_sets.py` |
| drawn on | 2026-08-08 09:33 |
| parent split | official train, 153,625 clips |
| pool after exclusions | 152,175 clips |
| size | 100 clips, 99 distinct chunks |
| seed | 42 |
| t0 | 5,100,000 us for every clip |
| weighted L1 achieved | 0.0306 |
| weighted L1 of a random draw | 0.1776 |
| manifest | `outputs/eval_sets/calib_100.parquet` (+ `.csv`) |
| cache | `/mnt/nvme1n1/ad_vla/data/physicalai_av/pre_processed/calib/samples/`, 100 npz |

## Command

```bash
python experiments/evaluation/make_eval_sets.py --split train --n-indist 100 --seed 42
```

The script names its output by split: `val` gives `indist_<N>`, `train` gives
`calib_<N>`. So the calibration set is the train-mode run of the same script that draws
the evaluation sets.

## Procedure

1. **Target distribution.** All clips with `split == "train"` in `clip_index.parquet`,
   joined to `metadata/data_collection.parquet`. The target is the proportion of clips in
   each category of each attribute, computed over all 153,625 train clips.

2. **Pool.** The same train clips, minus every clip that appears anywhere in
   `reasoning/ood_reasoning.parquet`. That file lists 1,740 OOD clips in total, of which
   1,450 sit in train, leaving 152,175.
   - The pool is **not** restricted to cached clips. Train was never fully downloaded, so
     the 100 chosen clips were streamed into the cache afterwards.
   - The older 50-clip calibration set in `outputs/split.json` is **not** excluded in
     train mode. The code comment gives the reason: this run is what defines the
     calibration set. The realised overlap with it is zero anyway.

3. **Derived attributes.** Four attributes come straight from the metadata. Two are derived:
   - `time_of_day` is `daytime` when `6 <= hour_of_day < 18`, else `nighttime`.
   - `season` is winter for months 12, 1, 2; spring for 3 to 5; summer for 6 to 8; fall
     for 9 to 11. This is the northern-hemisphere convention applied to every country.

4. **Encoding.** Every attribute value is forced to a string before categories are
   built. A missing value therefore becomes its own category, `"None"`, and is matched
   like any other.

5. **Greedy selection.** For step `k = 1 .. N`, every unchosen clip is scored by the
   weighted L1 the selected set would have if that clip were added:

   ```
   score(clip) = sum over attributes a of  w_a * sum_i | p_a[i] - c_a[i] / k |
   ```

   where `p_a` is the target proportion vector, and `c_a` is the count vector after
   adding the candidate. Adding one clip changes exactly one cell per attribute, so all
   152,175 candidates are scored in one vectorised pass. The lowest score is taken.

6. **Tie-breaking.** Ties are exact floating-point equality on the score, broken uniformly
   at random by `np.random.default_rng(42)`. Ties are the normal case, not an edge case:
   **every clip with the same six attribute values scores identically.** The matcher
   therefore chooses a *combination of cells* at each step, and the specific clip inside
   that combination is a uniform random pick.

7. **Nesting.** The greedy order was computed once out to 200, the largest size in the
   quality sweep `[25, 50, 100, 200]`, and the first 100 were kept. Any prefix of the
   order is itself a matched set, so a smaller calibration set needs no separate draw.

8. **Window.** Each chosen clip gets `t0_us = 5_100_000`. The window is fixed, not chosen
   by content.

## Matching attributes and weights

The weights are the same ones the earlier chunk-level manifest used.

| attribute | weight | source | L1 vs train at n = 100 |
|---|---|---|---|
| country | 4.0 | metadata | 0.0677 |
| platform_class | 2.0 | metadata | 0.0088 |
| time_of_day | 2.0 | derived from hour | 0.0005 |
| season | 1.5 | derived from month | 0.0139 |
| month | 1.0 | metadata | 0.0300 |
| radar_config | 1.0 | metadata | 0.0120 |

The reported weighted L1 is the weighted sum divided by the total weight, 11.5. Country
carries the most weight and still has the largest residual, because it has the most
categories for 100 clips to cover.

## Match quality

From `outputs/eval_sets/quality_train.csv`. The random reference is the mean and standard
deviation over 20 uniform draws without replacement from the same pool.

| n | greedy weighted L1 | random draw, mean ± sd | countries covered |
|---|---|---|---|
| 25 | 0.1504 | 0.3624 ± 0.0623 | 10 |
| 50 | 0.0705 | 0.2461 ± 0.0328 | 19 |
| 100 | **0.0306** | 0.1776 ± 0.0286 | 24 |
| 200 | 0.0146 | 0.1350 ± 0.0167 | 25 |

At n = 100 the matched draw is 5.8 times closer to the train distribution than a random
draw, and sits about 5 standard deviations below the random mean.

## The realised set

- **Country (24).** United States 51, Germany 15, France 3, Italy 3, then 2 each for
  Sweden, Netherlands, Spain, Portugal, Finland, Austria, Greece, Croatia, and 1 each for
  Denmark, Slovakia, Belgium, Estonia, Czechia, Slovenia, Luxembourg, Poland, Romania,
  Lithuania, Hungary, Latvia.
- **Platform.** hyperion_8.1 71, hyperion_8 29.
- **Time of day.** daytime 61, nighttime 39. Hours span 0 to 23.
- **Season.** winter 34, fall 27, summer 22, spring 17.
- **Month.** Jan 13, Feb 9, Mar 6, Apr 5, May 6, Jun 5, Jul 6, Aug 11, Sep 10, Oct 7,
  Nov 10, Dec 12.
- **Radar config.** missing 48, low 28, med 17, high 7. The missing share was matched as
  its own category, per step 4.
- **Chunks.** 99 distinct chunks for 100 clips. A chunk is a storage shard of about 100
  clips from a single split, not a recording session, and the metadata carries no session
  identifier. So this shows the clips are spread across shards, and nothing stronger.

Manifest columns: `clip_id, chunk, country, platform_class, radar_config, month,
hour_of_day, time_of_day, season, t0_us`.

## Disjointness

Checked directly on the manifests. `calib_100` shares **zero** clips with each of:

| set | overlap |
|---|---|
| `indist_500` / `val_500` | 0 |
| `test_500` | 0 |
| `ood` (1,533) and `ood_val` (262) | 0 |
| `outputs/split.json` calib (50), train, test | 0 |

Disjointness from the val and test sets follows from the official splits being disjoint.
Disjointness from OOD is enforced by the pool exclusion in step 2.

## What the sampler did not control

**Driving action was never an input.** The mix below was computed on 2026-09-20 from the
cached ground-truth futures, using `eval_lib.bucket` with its priority order
decel_stop, then turn, then accel, then cruise.

| cruise | decel_stop | accel | turn |
|---|---|---|---|
| 49 | 18 | 16 | 17 |

This follows from step 6. Within a combination of six metadata cells the clip is picked
uniformly at random, so anything not among the six attributes is left to chance:
action, speed, scene density, road type, weather beyond what season implies.

Note that this uses the four-class `eval_lib.bucket`. The stratified-calibration work
uses a five-class `bucket5` from `label_actions.py`. The two taxonomies are not
interchangeable, so do not compare this row against a `bucket5` mix.

## Caveats from later studies

**Relayed from the project notes (`CLAUDE.md`), not re-verified here.**

- **`calib_100` is described as a lucky draw.** Five further sets drawn by this same
  protocol, `dual_nt_a` to `dual_nt_e`, cost +0.029, +0.041, +0.214, +0.451 and +0.381 on
  val500 relative to the shipped arm.
- **The protocol itself was found to be the weakest rule tested.** The report
  `reports/evaluation/2026-09-12_action-stratified-calib.html` compared random draws,
  draws stratified to the evaluation set's action mix, and uniform action strata. It
  found sequential greedy six-axis matching worst, with the loss concentrated in cruise,
  and recommends the action-stratified rule as the standard draw.
- **Converged selection did not buy performance.** Calibration sets of 2,000 clips gave a
  stable kept set yet landed worse than `calib_100` in closed loop, per
  `reports/evaluation/2026-09-06_calibration-size-closedloop.html`.
- **No single clip explains the draw effect.**
  `reports/evaluation/2026-09-15_calibration-clip-influence.html` found large set-level
  effects across 50-clip halves of `calib_100` but no attributable per-clip contribution.

The practical reading: matching collection metadata makes a set that *looks*
representative, but the quantity that matters for pruning, the importance ranking, is
sensitive to content the matcher never saw.

## Records to distrust

`outputs/eval_sets/config_train.json` and `summary_train.txt` both describe the target as
**val**: `"method": "... against full-val"`, `"target": {"split": "val", ...}`, and
`weighted L1 vs full val`. These are **hardcoded strings** in the script, written the
same way for every split.

The target was train. Two independent checks agree: the code builds the target from
`ci[ci.split == args.split]` with `--split train`, and the recorded `n_clips` of 153,625
equals the official train size exactly. Val is 90,928 and test is 61,599.

The same files report `"calib_clips": 50` under exclusions. That is the size of the old
calibration set, recorded unconditionally. In train mode nothing was actually excluded,
which the pool counts confirm: `after_drop_ood` and `after_drop_calib` are both 152,175.

## Reproducing without touching live manifests

The draw is deterministic given the seed and an unchanged `clip_index.parquet`, so the
command above reproduces the same 100 clips.

**Do not rerun it into the default output directory.** Besides `calib_100.parquet`, every
run rewrites `ood.parquet`, which the OOD evaluation depends on, plus the split's config,
metrics, summary, quality CSV and plot. Redirect it instead:

```bash
python experiments/evaluation/make_eval_sets.py \
    --split train --n-indist 100 --seed 42 --exp-id eval_sets_repro
```

Then diff `outputs/eval_sets_repro/calib_100.parquet` against the live manifest.

## Where it is used

`calib_100` is the default calibration manifest for both criterion runners:

- `experiments/head_analysis/run_importance.py`, via `--calib-manifest calib_100` paired
  with `--cache calib` at the fixed calibration t0.
- `experiments/head_analysis/run_jlens.py`, via the same `--calib-manifest` default.

Per the project notes, both halves of the shipped `dual_u40_v2` criterion were measured on
these same 100 clips. The set is for calibration only and is never evaluated on.

An independent check, `experiments/evaluation/check_stratification.py`, records the same
provenance and recomputes the matcher's objective for each calibration manifest. It
expects a matched draw near 0.03 and a random one near 0.2.

## Sources

| what | where |
|---|---|
| drawing script | `experiments/evaluation/make_eval_sets.py` |
| manifest | `outputs/eval_sets/calib_100.parquet`, `calib_100.csv` |
| saved records | `outputs/eval_sets/config_train.json`, `metrics_train.json`, `summary_train.txt` |
| quality sweep | `outputs/eval_sets/quality_train.csv` |
| match plot | `outputs/eval_sets/plots/match_train.png` |
| set documentation | `outputs/eval_sets/EVAL_SETS.md` |
| clip index and metadata | `/mnt/nvme1n1/ad_vla/data/physicalai_av/clip_index.parquet`, `metadata/data_collection.parquet` |
| OOD exclusion list | `/mnt/nvme1n1/ad_vla/data/physicalai_av/reasoning/ood_reasoning.parquet` |
| per-clip cache | `/mnt/nvme1n1/ad_vla/data/physicalai_av/pre_processed/calib/samples/` |
