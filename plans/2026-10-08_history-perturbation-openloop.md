# Plan: open-loop ego-history perturbation (2026-10-08)

Follows `plans/2026-10-07_dense-pace-cause.md`. In closed loop the simulator hands the car over
with -8 m/s² commanded, every arm loses a median 1.06 m/s in the first 0.1 s, and only the
unpruned VLM goes on to plan a further slowdown. That was read off rollout logs, where the
perturbation, the scene and the loop all move together. This experiment injects the same history
into the open-loop harness, where nothing else changes.

## Hypothesis

Continuing a deceleration seen in the ego history is a property of the dense model that needs no
simulator: shorten the last history step by the amount the simulator removes and the dense model
predicts negative acceleration, while the pruned VLMs do not, in proportion to how much of the
VLM was pruned.

## Setup

- Clips: the first 150 of `test_500`, keeping those moving at t0 (last history step >= 2 m/s):
  120 clips. Nothing here is calibration data for any arm.
- Arms: `baseline`, `slim_dual_u40_v2` (MoOP), `slim_dual_u20_v2`, `slim_dual_u30_v2` (dose),
  `slim_expert_znorm_r25` (VLM untouched, expert pruned), `slim_tyr_u40_r`.
- Inference as in `run_baseline.py`: own CoC rollout (temperature 0.6, top-p 0.98), then K = 4
  flow-matching samples; seeds from `clip_seed`, identical across conditions, so every
  comparison is paired within clip and seed. Ada cards 4–7 only, one architecture for all arms.
- Conditions on `ego_history_xyz` (16 poses, 0.1 s apart, t0 frame, last pose the origin), with
  u the direction of the last step and δ = min(1.06, 0.9·v_last) × 0.1 s:

  | condition | change | what it mimics |
  |---|---|---|
  | `clean` | none | the plan at the hand-over |
  | `drop1` | poses 0–14 moved by +δ·u (last step shorter by δ) | the history 0.1 s after the hand-over |
  | `drop2` | poses 0–13 by +2δ·u, pose 14 by +δ·u (last two steps shorter) | 0.2 s after, where the closed-loop response peaked |
  | `rise1` | poses 0–14 by −δ·u (last step longer) | mirror control: is it deceleration, or any change? |
  | `slow10` | all poses × 0.9 (10% slower throughout, nothing abrupt) | level control: a slower history without a braking event |
  | `drop2@cleanCoC` | `drop2` history, the clean rollout's CoC teacher-forced | does the response pass through the reasoning text? |

- Measured per sample: v0 the action space derives from the history (the same function for every
  arm), the predicted acceleration (denormalised, 64 steps), and the planned speed
  v0 + cumsum(a·dt). Reported as means over 0–1, 1–2, 2–3 s.

## Confounds considered

- **v0 is mechanical.** The action space integrates predicted acceleration from a speed it fits
  to the history, so a perturbed history lowers the planned speed in every arm before the model
  has said anything. The model's response is therefore read from the **predicted acceleration**,
  and planned speed is reported next to it, never instead of it.
- **CoC sampling.** A changed prompt can change the sampled text even with the same seed; that is
  part of the response and `drop2@cleanCoC` isolates it.
- **Architecture.** All arms on Ada.

## Gates (fixed before any run)

Primary quantity: Δa = mean predicted acceleration over the first 2 s under the perturbation
minus under `clean`, per clip (mean over the K samples), in m/s². CIs are clip bootstraps.

- **S0 sanity (smoke, first clip).** `drop1` changes the history token of the last step and at
  most the first token, `clean` reproduces `run_baseline`'s inputs, and v0 under each condition is
  identical across arms.
- **H1 reproduction.** For the dense model, Δa(`drop2`) < 0 with the CI excluding 0.
- **H2 contrast.** Δa(`drop2`) is more negative for dense than for MoOP: clip-paired difference
  with the CI excluding 0. Same for `drop1` as a secondary reading.
- **H3 dose.** Ordering of Δa(`drop2`) from most to least negative follows VLM pruning:
  {dense, expert-only} < MoOP 12% < MoOP 18% < MoOP 24%, with at most one adjacent inversion.
- **H4 reasoning.** With the clean CoC teacher-forced, the dense model keeps at least two thirds
  of its Δa(`drop2`). Below one third would mean the response runs through the text.
- **Exploratory, no gate:** `rise1` (does dense also extrapolate an acceleration?) and `slow10`
  (does a slower but steady history lower the plan by the full 10%, i.e. is the level anchored?).

Reading: H1 + H2 make the closed-loop finding a model property. H1 failing means the closed-loop
response needs the loop (compounding over steps) or this perturbation misses what the simulator
does; then compare the history actually sent to the driver, which the ASL holds.

## Addendum, written mid-run (about clip 50 of 120)

A health check on the first 40 clips showed no arm changing its predicted acceleration under
`drop1` or `drop2` (dense +0.02 [-0.05, +0.09] m/s² for `drop2`), so H1 looks set to fail. Before
the run finished, and before any further look, two sustained conditions were added as a separate
run (`histperturb_v1b`, dense and MoOP, same clips and seeds) to tell "one or two steps are too
weak a cue" from "the history does not move the predicted acceleration at all":

- `sim5` — the last five steps slower by 1.06, 1.29, 1.45, 1.66, 1.80 m/s: the history a
  dense-driven car has 0.5 s after the hand-over, read off the logged rollouts.
- `brake5` — the last five steps slower by 0.3, 0.6, 0.9, 1.2, 1.5 m/s: a uniform -3 m/s².

Both are scaled down on slow clips so no step loses more than 90% of its length.

Reading, fixed here:
- dense's Δa(`sim5`) < 0 with the CI excluding 0, and more negative than MoOP's → the history
  channel exists but needs a sustained pattern;
- both indistinguishable from 0 → the predicted acceleration does not follow the ego history, and
  the closed-loop response must come through what open loop cannot change: the frames, which in
  closed loop are rendered by the simulator and show the same deceleration. That can only be
  tested by replaying the logged render requests and re-running the model on them.

**Correction to the paragraph above (after the main arms finished, n = 120).** The 40-clip
impression overstated the failure and a first full-sample reading overstated the pass: with a
5,000-draw bootstrap H1's upper bound landed at −0.001 ("pass"), with 200,000 draws it is +0.001
(fail). H1 is not met; H2, H3 and H4 are (results below). The verdict moved three times — at 40
clips, at 120 clips, and when the bootstrap was converged — which is itself the finding about
`drop2`: its effect on the dense model is too close to zero to call. The addendum conditions stay
as registered, read as registered.

## Second addendum, written before `histperturb_v1c` was launched

With the four finished arms analysed, the open-loop response has the registered sign, contrast and
dose order but is small: on the plan slope (planned speed over 1–2 s minus over 0–1 s, a quantity
v0 cancels out of and the closed-loop logs also hold) the dense model moves −0.102 under `drop1`
against −0.392 in closed loop 0.1 s after the hand-over, and −0.065 under `drop2` against −0.875
at 0.2 s. Re-reading the logs for that comparison showed that **`drop2` is not the history the
simulator produces at 0.2 s**. Speed lost against the hand-over, median over moving starts:

| control steps after hand-over | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| dense | −1.06 | −1.28 | −1.38 | −1.51 | −1.61 |
| MoOP 24% | −1.06 | −1.27 | −1.32 | −1.34 | −1.34 |

The second step is slower again by 0.22 m/s in **every** arm (the arms only part from step 3), so
the history at 0.2 s still shows −2.2 m/s² in its last step, whereas `drop2` holds the speed and
shows 0. A model that continues the latest acceleration has nothing to continue under `drop2`.
(`sim5`, registered above from one typical rollout, overstates the dense car's last two steps by
about 0.15–0.2 m/s against these medians.)

Added as `histperturb_v1c` (same clips, seeds, cards):

- `sim2` — last two steps slower by 1.06 and 1.28 m/s: what **every** arm is fed 0.2 s after the
  hand-over, where the closed-loop response peaks. Also with the clean CoC teacher-forced.
- `sim3` — last three steps slower by 1.06, 1.28, 1.38 m/s: the dense car's history at 0.3 s.

Arms: dense and MoOP 24% first, MoOP 12% and 18% as cards free up. Extended, before any `sim2`
result beyond a 10-clip health check had been looked at, to the two remaining arms of the main run
on the cards they had just freed: expert-only pruning (VLM untouched — read as the same level as
dense, as in H3) and Týr 24% (a different pruning method at MoOP's ratio; no gate).

Reading, fixed here. Quantity: change in plan slope against `clean`, set against the closed-loop
change at the matching control step (dense: −0.875 at 0.2 s, −0.760 at 0.3 s; MoOP 24%: +0.052).

- **R1 reproduction at the faithful history.** dense's Δslope(`sim2`) < 0 with the CI excluding
  0, and more negative than MoOP 24%'s (paired CI excluding 0).
- **R2 how much the history alone carries.** dense's Δslope(`sim2`) as a share of −0.875:
  at least two thirds → the history is sufficient and neither the rendered frames nor the loop is
  needed to explain the closed-loop response; below one third → the history is a minor channel
  and the frames or the loop carry the rest; in between → both, reported as such.
- **R3 dose at the faithful history.** dense < MoOP 12% < MoOP 18% < MoOP 24% on Δslope(`sim2`),
  at most one adjacent inversion (closed loop: −0.875, −0.667, −0.054, +0.052).
- `sim2` − `drop2` is the effect of the second step continuing to slow; no gate.

Caveat fixed in advance: the closed-loop numbers come from Origin150 renders and the open-loop
ones from Test500 recordings, so R2 compares two scene sets; the first step is identical in the
two (1.06 m/s) but the scenes are not.

## Cost

About 25 s per clip per arm (five rollouts, seven denoising sets), so about 50 min per arm and
5 GPU-h for six arms — more than the 2 GPU-h first quoted, which assumed two arms. The two
addenda add about 1 GPU-h (`v1b`, two arms) and 2 GPU-h (`v1c`, four arms).

## Outputs

`outputs/histperturb_v1/{config.json, <arm>.json, metrics.json, summary.txt, plots/}`; runner
`experiments/evaluation/run_history_perturb.py`, analysis `analyze_history_perturb.py`. The
addenda are `outputs/histperturb_v1b/` and `outputs/histperturb_v1c/`; the analysis reads them
by name and writes everything into `histperturb_v1/`.

## Results (all runs complete 2026-10-08; 6 arms × 120 clips, Ada 4–7, 9.2 GPU-h)

Sanity first: the clean plan is **bit-identical** in the main run and in both addendum runs for
every arm (largest difference 0.0), so the three runs can be read as one.

### Registered gates

| gate | reading | verdict |
|---|---|---|
| S0 | `drop1` changes the last step's token (45) in 102 / 120 clips, plus one neighbouring or first-step token in the other 18; `drop2` tokens 42 and 45. v0 moves by −1.516 / −1.344 m/s, identically in every arm | PASS |
| H1 | dense Δa(`drop2`) = −0.065 [−0.137, +0.001] m/s², t-test p = 0.068. Per clip: median +0.012, signed-rank p = 0.80, 10% trimmed mean −0.010; without the 12 most negative clips the mean is +0.034 | **FAIL** |
| H2 | dense − MoOP 24%, clip-paired: `drop2` −0.108 [−0.166, −0.059], `drop1` −0.131 [−0.205, −0.066]; 77 of 120 clips negative, signed-rank p = 0.0001 | PASS |
| H3 | Δa(`drop2`): dense −0.065, expert-only −0.080, MoOP 12% −0.023, 18% +0.017, 24% +0.042 — no inversion (Týr 24% +0.047) | PASS |
| H4 | dense keeps 76% of Δa(`drop2`) with the clean CoC teacher-forced (89% for `drop1`) — a ratio whose numerator H1 could not tell from zero | PASS, uninformative here |

So the registered experiment says: a one- or two-step drop **after which the speed holds** does
not move the typical dense plan. It flips about 13% of them into braking (5% for MoOP 24%), and
that tail is the whole mean. The contrast between arms is real and ordered; its size is small.
`rise1` is not a mirror image of `drop1` (dense −0.064, n.s.) and `slow10` lowers every arm's
plan by the full 10% (0.905–0.909 of clean), so the level is anchored on the history and there
is no asymmetry to read.

### Addenda (registered before each run)

Read on the plan slope (planned speed over 1–2 s minus over 0–1 s, m/s per s).

| history fed in (m/s off the last steps) | dense | expert-only | MoOP 12% | MoOP 18% | MoOP 24% | Týr 24% |
|---|---:|---:|---:|---:|---:|---:|
| `drop1` −1.06 | −0.102* | −0.120* | −0.069 | +0.001 | +0.023 | +0.044 |
| `sim2` −1.06, −1.28 (every arm's history at 0.2 s) | **−0.502*** | −0.541* | −0.411* | −0.114* | +0.014 | +0.057 |
| closed loop, same step | −0.875* | −0.780* | −0.667* | −0.054 | +0.052 | +0.164* |
| `sim3` −1.06, −1.28, −1.38 (dense car at 0.3 s) | −0.491* | −0.553* | −0.409* | −0.204* | −0.091 | −0.056 |
| `sim5` −1.06 … −1.80 (dense car at 0.5 s) | −0.393* | | | | −0.086 | |
| `brake5` uniform −3 m/s² for 0.5 s | −0.282* | | | | **+0.140*** | |

- **First addendum, as registered:** dense Δa(`sim5`) = −0.447 [−0.566, −0.325], more negative
  than MoOP's by −0.316 [−0.397, −0.236] → *the history channel exists but needs a sustained
  pattern.* It is a typical-clip response (74% of clips, median −0.345), not a tail.
- **R1 PASS.** dense Δslope(`sim2`) = −0.502 [−0.663, −0.348]; dense − MoOP 24% = −0.516
  [−0.640, −0.401]. 72% of clips, 40% of plans falling by more than 0.5 against 11% for MoOP.
- **R2: 57% [38, 83] → "both".** The history alone carries more than half of the closed-loop
  change at 0.2 s and not all of it. The share grows with the step: 26% [3, 61] at 0.1 s,
  57% at 0.2 s, 65% [44, 93] at 0.3 s, 83% [51, 134] at 0.5 s (`sim5` is about 12% deeper than
  the median history there). Post hoc, restricted to closed-loop scenes that lost at least
  1.0 m/s in the first step and re-weighted to their speed mix: 32% at 0.1 s, 68% at 0.2 s.
- **R3 PASS.** No inversion; across all six arms open loop and closed loop agree at r = +0.99
  (Spearman +0.94, the two unpruned VLMs swapping places), with the closed-loop changes 1.7×
  the open-loop ones.
- With the clean CoC teacher-forced the dense model keeps 93% of Δslope(`sim2`) — the H4
  question asked where there is an effect to keep.
- `sim2` − `drop2` for dense: −0.437 [−0.580, −0.298]. The only difference between the two is
  that the last step is 0.22 m/s slower again.

### What was learned

1. **Continuing a deceleration seen in the ego history is a property of the unpruned VLM, with no
   simulator involved** — but the cue is a deceleration that is *still going on*, over at least
   two steps. One slower step, or two after which the speed holds, is not enough.
2. **Pruning the VLM removes it in the order and with the cliff seen in closed loop:** four fifths
   left at 12%, a quarter at 18%, none at 24%, and none for Týr at the same ratio; pruning only
   the expert leaves it whole. After a plain braking history MoOP 24% speeds back up (+0.116
   [+0.019, +0.207] m/s² over 0–2 s) where dense keeps braking (−0.349).
3. **Where the second slow step comes from** (found while checking `drop2` against the logs): the
   action space fits v0 to the history, and that fit reads a one-step drop as a trend — v0 moves
   by −1.52 m/s for a −1.06 m/s step in every arm. In the simulator the plan therefore starts
   below the car (MoOP 24%: 0.41 m/s below at 0.1 s, 0.33 at 0.2 s, 0 from 0.3 s) and the
   controller follows it down. Chain: release at −8 m/s² → −1.06 m/s → v0 overshoot → a second
   slower step in every arm → a history that reads as ongoing braking → the unpruned VLM
   continues it, the pruned ones do not.
4. **Not explained:** the first step. With the identical one-step history the simulator shows a
   typical response from the dense model (median −0.248, 37% of plans falling by more than 0.5)
   and recorded frames show a tail (median +0.013, 13%). The inputs differ only in the frames
   (rendered against recorded) and the scene set, and the response is speed-dependent in both
   (strongest at 6–12 m/s, reversed above 20 m/s in open loop). Separating frames from scenes
   needs the logged render requests replayed into this harness; not run.
5. **Method.** A gate whose bound sits at zero must be read off a converged bootstrap — 5,000
   draws moved this one across zero — and with the median beside the mean; and a perturbation
   meant to mimic a logged input should be checked against the log before it is registered,
   not after.
