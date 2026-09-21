# How `test_500` was sampled

Verified 2026-09-20 against the drawing script, the records it saved, the manifest
itself, the clip index, and the per-clip cache. Anything relayed from the project notes
rather than re-checked is marked as such.

Companion to `calib100_sampling.md`. Both sets come from the same script and the same
matcher, so this file states the shared procedure briefly and spends its length on what
differs.

## Summary

`test_500` is 500 clips from the **official test split**, chosen by **seeded greedy
distribution matching** on six collection-metadata attributes, against the clip-level
distribution of the whole test split. Seed 42, fixed window start t0 = 5.1 s.

| fact | value |
|---|---|
| producer | `experiments/evaluation/make_eval_sets.py` |
| drawn on | 2026-08-07 10:56, one day before `calib_100` |
| parent split | official test, 61,599 clips |
| pool after exclusions | 61,587 clips |
| size | 500 clips, 323 distinct chunks |
| seed | 42 |
| t0 | 5,100,000 us for every clip |
| weighted L1 achieved | 0.0055 |
| weighted L1 of a random draw | 0.0842 |
| manifest | `outputs/eval_sets/test_500.parquet` (+ `.csv`) |
| cache | `/mnt/nvme1n1/ad_vla/data/physicalai_av/pre_processed/test/samples/`, 500 of 500 npz present |
| role | in-distribution, held out |

## Command

```bash
python experiments/evaluation/make_eval_sets.py --split test --n-indist 500 --seed 42
```

The output stem falls through to the generic `<split>_<N>`, giving `test_500`. The val
run is named `indist_<N>` and the train run `calib_<N>`.

The chosen clips were then streamed into the per-clip cache:

```bash
python experiments/evaluation/build_cache.py \
    --manifest outputs/eval_sets/test_500.parquet --cache test --workers 16
```

The test split's camera chunks were never downloaded, so the pool was the whole split and
only the 500 selected clips crossed the wire.

## Procedure

The matcher is identical to the calibration draw. In brief:

1. **Target.** Proportions of each category of six attributes over all 61,599 official
   test clips.
2. **Pool.** The test clips minus two exclusion lists, detailed in the next section.
3. **Attributes and weights.** country 4.0, platform_class 2.0, time_of_day 2.0,
   season 1.5, month 1.0, radar_config 1.0. Time of day is daytime for hours 6 to 17.
   Season follows the northern-hemisphere months for every country. Missing values are
   matched as their own category.
4. **Greedy step.** At step `k`, every unchosen clip is scored by the weighted L1 the
   selected set would have with that clip added. The lowest score is taken.
5. **Ties.** Broken uniformly at random by `np.random.default_rng(42)`. Clips with the
   same six attribute values score identically, so the matcher chooses a combination of
   cells and the clip within it is a uniform random pick.
6. **Nesting.** The order was computed once out to 1,000, the largest size in the
   evaluation sweep `[100, 200, 500, 1000]`, and the first 500 were kept. Any prefix is
   itself matched, and extending to 1,000 would keep these 500. The manifest rows are
   stored in greedy order, so its first N rows are a matched subset. On an unsharded run,
   `run_baseline.py --limit N` takes exactly those. The project notes call this flag
   `--num-clips`, which does not exist on that runner.
7. **Window.** Every clip gets `t0_us = 5_100_000`.

See `calib100_sampling.md` for the scoring formula.

## Exclusions, and what they actually removed

This is where the test draw differs from the calibration draw.

| exclusion | listed | actually in the test pool | removed |
|---|---|---|---|
| every clip in `reasoning/ood_reasoning.parquet` | 1,740 | 0 | 0 |
| old calibration set, `outputs/split.json["calib"]` | 50 | 12 | 12 |

- **The OOD exclusion was a no-op here.** None of the 1,740 OOD clips sit in the official
  test split. They are all in train (1,450) and val (290).
- **The old-calibration exclusion did real work.** The 50-clip calibration set from the
  2026-07 study was drawn from `notebooks/clip_ids.parquet` without regard to the official
  splits. Its clips land 22 in train, 16 in val and **12 in test**. Those 12 were removed,
  taking the pool from 61,599 to 61,587. Without this step, `test_500` could have held
  clips that an importance measurement had already seen.

`calib_100` did not exist yet when this set was drawn. The two are disjoint anyway,
because `calib_100` comes from official train and the official splits share no clips.

## Match quality

From `outputs/eval_sets/quality_test.csv`. The random reference is the mean and standard
deviation over 20 uniform draws without replacement from the same pool.

| n | greedy weighted L1 | random draw, mean ± sd | countries covered |
|---|---|---|---|
| 100 | 0.0302 | 0.1750 ± 0.0259 | 22 |
| 200 | 0.0161 | 0.1324 ± 0.0213 | 25 |
| 500 | **0.0055** | 0.0842 ± 0.0131 | 25 |
| 1000 | 0.0034 | 0.0600 ± 0.0073 | 25 |

At n = 500 the matched draw is about 15 times closer to the test distribution than a
random draw, and sits 6 standard deviations below the random mean. At n = 100 it scores
0.0302, essentially the same as `calib_100`'s 0.0306, so the matcher behaves the same on
both splits and the tighter fit here is purely the larger n.

Per-attribute L1 against the test split at n = 500:

| country | platform_class | time_of_day | season | month | radar_config |
|---|---|---|---|---|---|
| 0.0107 | 0.0010 | 0.0013 | 0.0021 | 0.0092 | 0.0031 |

## The realised set

- **Country (25).** United States 250, Germany 74, France 18, Italy 16, Sweden 12,
  Greece 11, Spain 11, Portugal 10, Finland 9, Austria 9, Croatia 9, Estonia 9,
  Lithuania 8, Czechia 7, Netherlands 7, Denmark 7, Belgium 6, Romania 6, Poland 5,
  Slovakia 5, Slovenia 4, Latvia 2, Luxembourg 2, Hungary 2, Bulgaria 1.
- **Platform.** hyperion_8.1 360, hyperion_8 140.
- **Time of day.** daytime 301, nighttime 199.
- **Season.** winter 173, fall 137, summer 104, spring 86.
- **Month.** Jan 62, Feb 44, Mar 27, Apr 25, May 34, Jun 19, Jul 30, Aug 55, Sep 43,
  Oct 37, Nov 57, Dec 67.
- **Radar config.** missing 243, low 139, med 79, high 39.
- **Validity.** All 500 are flagged `clip_is_valid`. The script does not filter on that
  flag, but every clip in the index is valid, so nothing was at risk.

The proportions track `calib_100` closely: half United States, about 15% Germany, 72%
hyperion_8.1, 60% daytime. That is expected, since train and test are splits of one
collection and both draws match their parent.

### Chunks

The 500 clips span 323 chunks. 127 chunks contribute more than one clip, and one
contributes 9.

A chunk is a **storage shard** of about 100 clips, median 99 in the test split, which has
636 of them. Each chunk belongs to a single split. It is not a recording session, and the
metadata carries no session identifier, so **session-level independence of the 500 clips
cannot be checked from what is on disk.**

Uniform placement of 500 clips over 636 shards would occupy about 346, so 323 is mild
concentration. The likely reason is that 543 of the 636 test chunks hold a single
country, and the matcher must hit small countries repeatedly. This is an inference, not a
measurement.

## Disjointness

Checked directly on the manifests.

| set | overlap with `test_500` |
|---|---|
| `calib_100` | 0 |
| `indist_500` / `val_500` | 0 |
| `ood` (1,533) and `ood_val` (262) | 0 |
| `outputs/split.json` calib (50) | 0 |
| `outputs/split.json` test, val | 0 |
| `outputs/split.json` **train** (900) | **1** |

All 500 clips carry the official `test` label.

**The one overlap.** Clip `01460b45-8c01-4dea-b98d-c9ec2c4caa5a` is in the old split's
train list. The script excludes that split's calibration subset but not the rest of its
train list. The clip is not in the calibration subset.

Who this can affect, as far as I traced it:

- **Pruning-only arms: not affected.** They never train, and their importance comes from
  `calib_100`, which is disjoint.
- **The current recovery track: not affected.** `experiments/recovery/make_train_set.py`
  draws its training pool from official train, which shares no clips with official test.
- **The 2026-07 LoRA track: possibly, by 1 clip in 500.**
  `experiments/head_analysis/train_lora.py` reads `split.json`. A model trained there and
  then scored on `test_500` would have seen this one clip. I did not check whether any
  reported number is such a model.

## What the sampler did not control

Driving action was never an input. The mix below was computed on 2026-09-20 from the
cached ground-truth futures with the four-class `eval_lib.bucket`.

| cruise | turn | accel | decel_stop |
|---|---|---|---|
| 278 (55.6%) | 82 (16.4%) | 75 (15.0%) | 65 (13.0%) |

For comparison, `calib_100` under the same rule is cruise 49, decel_stop 18, accel 16,
turn 17. The two are similar, with the calibration set somewhat lighter on cruise and
heavier on stops. Neither was steered there.

The project notes quote the `test_500` mix as 56/13/15/9/7. That is the five-class
`bucket5` from `label_actions.py`, which splits turns by direction. It is the reference
mix that the action-stratified calibration rule matches to. The two taxonomies agree on
cruise at 56% but are otherwise not interchangeable.

## Properties relayed from the project notes

**Not re-verified here.**

- **Determinism.** Seeds come from the clip, as `sha256(f"{seed}:{clip_id}")[:4]`, so a
  sharded run and a whole run agree on this set.
- **Architecture.** Results on this set are bitwise reproducible within one GPU
  architecture. Across Ada and A100 the 500-clip baseline mean agrees to within 0.004,
  while 3 to 4% of clips produce different CoC text.
- **Baseline.** The unpruned model scores minADE@6 of 0.842 on this set, against 0.824 on
  the val set and 1.000 on OOD-val.
- **Reporting rule.** Because minADE deltas are heavy-tailed, the median and Wilcoxon are
  read first and the mean is reported alongside with its own bootstrap interval.

## Records to distrust

`outputs/eval_sets/config_test.json` and `summary_test.txt` describe the target as
**val**: `"target": {"split": "val", ...}` and `weighted L1 vs full val`. These are
hardcoded strings in the script. The target was the test split: the code selects
`ci[ci.split == args.split]`, and the recorded `n_clips` of 61,599 is exactly the official
test size. Val is 90,928.

The exclusions block reports `"ood_clips_total": 1740` and `"calib_clips": 50`. Those are
the sizes of the lists, not what they removed. The pool counts give the real effect:
61,599 to 61,599 after OOD, then 61,587 after calibration.

## Reproducing without touching live manifests

The draw is deterministic given the seed and an unchanged clip index. Do not rerun into
the default directory: every run also rewrites `ood.parquet`, which the OOD evaluation
depends on. Redirect it:

```bash
python experiments/evaluation/make_eval_sets.py \
    --split test --n-indist 500 --seed 42 --exp-id eval_sets_repro
```

Then diff `outputs/eval_sets_repro/test_500.parquet` against the live manifest.

## Sources

| what | where |
|---|---|
| drawing script | `experiments/evaluation/make_eval_sets.py` |
| cache builder | `experiments/evaluation/build_cache.py` |
| manifest | `outputs/eval_sets/test_500.parquet`, `test_500.csv` |
| saved records | `outputs/eval_sets/config_test.json`, `metrics_test.json`, `summary_test.txt` |
| quality sweep | `outputs/eval_sets/quality_test.csv` |
| set documentation | `outputs/eval_sets/EVAL_SETS.md` |
| old split | `outputs/split.json` |
| clip index and metadata | `/mnt/nvme1n1/ad_vla/data/physicalai_av/clip_index.parquet`, `metadata/data_collection.parquet` |
| per-clip cache | `/mnt/nvme1n1/ad_vla/data/physicalai_av/pre_processed/test/samples/` |
| open-loop runner | `experiments/evaluation/run_baseline.py --set test` |
