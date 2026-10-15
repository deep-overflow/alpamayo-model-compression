# Plan: why does the dense model score below the pruned ones in closed loop? (2026-10-07)

GPU-free. Uses only the stored per-timestep `metrics.parquet` of the eight paper arms on
Origin150 and Hard100 (4,000 rollouts). Answers review item 1 (2026-10-07) and Todo P4-a / P4-e.

## Question

Table 2 has the dense model at 0.750 against 0.78–0.83 for every non-collapsed pruned arm. The
gap is all in `progress` (0.747 vs 0.828 for MoOP) and the gates are equal. Three explanations
are on the table and have never been separated:

- **H-route** — the dense model more often takes a valid route other than the recorded one;
  scoring stops 4 m from the recorded path, so that progress is never credited.
- **H-stop** — pruned models stop and hesitate less, so they cover more of the route in the
  20 s; the score has no penalty for running a stop.
- **H-gate** — the difference is in off-road / collision events after all, and `progress` only
  reflects where the score was frozen.

## Method

Reimplement alpasim's chain exactly (`processing.DEFAULT_MODIFIERS` + `main._run_aggregation`):
drop timesteps before `eval_relevant`; keep up to the first `collision_any | offroad`; keep up to
the first `dist_to_gt_trajectory >= 4.0`; `progress_clipped_rel = last(progress_rel_to_total)`;
`progress_score = min(clamp(p, 0, 1) / 0.8, 1)` (1.0 when `gt_dist_traveled_m < 5`); score is 0
on an at-fault collision or off-road, else `progress_score`.

Per rollout, with `P(t)` the progress score at time `t`:

    1 - score = [1 - P(T_end)]                 slow      route not covered even with no truncation
              + [P(T_end) - P(t_gate)]         frozen    progress frozen at a collision / off-road
              + [P(t_gate) - P(t_ship)]        cut4m     progress frozen by the 4 m rule
              + [P(t_ship) - score]            zeroed    score set to 0 by a gate

The four terms sum to the loss exactly, so `score_arm - score_dense` splits into four
scene-paired differences. A rollout cut at 4 m with `P < 1` is then followed into the unscored
remainder: no gate and at least 10 m more driven = **alternative route** (the reviewer's
"multi-answer"); a later off-road / at-fault collision = **hidden failure**; back within 4 m =
**excursion**.

Also reported: stop time (speed < 0.5 m/s), mean speed, ego/GT progress ratio (`progress_rel`);
the score with the 4 m rule switched off; gate events per km; rollout-1 vs rollout-2 within each
arm as the A/A noise floor.

## Gates (fixed before running)

- **G0 reproduction.** The recomputed score equals the shipped score on every rollout that has a
  parquet (max |diff| < 1e-6). Otherwise stop — nothing below is trustworthy.
- **G1 which term carries the gap.** A term "explains" MoOP − dense if its scene-paired 95% CI
  excludes 0 and it is at least half of the gap on Origin150, with the same sign on Hard100.
- **G2 multi-answer.** H-route is supported if the dense model's alternative-route rate exceeds
  MoOP's with a scene-paired CI excluding 0 on both sets.
- **G3 stopping.** H-stop is supported if the dense model's `slow` term or stop-time share exceeds
  MoOP's with a CI excluding 0 on both sets.

Reading: G1 names the term; G2 / G3 say which story that term belongs to. If `cut4m` carries
the gap and G2 passes, the closed-loop advantage over dense is a scoring artifact and the paper
should say so. If `slow` carries it and G3 passes, pruned models gain by stopping less — a
behaviour change the paper must report next to the score. Both can hold.

## What this cannot do

No per-timestep GT speed is in the parquet, so "stopped where the human did not" is approximated
by `progress_rel`; the stop-compliance metric (Todo P4-d) needs the GT pose track from the usdz.
CoC intent distributions need the ASL logs and are left for a second pass.

## Outputs

`outputs/closedloop_gap_v1/{config.json,metrics.json,summary.txt,rollouts.parquet,plots/}`,
script `experiments/evaluation/analyze_closedloop_gap.py`.

## Amendments after the first run (2026-10-07) — everything below is post hoc

G0–G3 were evaluated as written. Three things were added once the first numbers were in; none of
them was pre-registered, and the summary marks each section "post hoc".

1. **Loss by ending rule.** The telescoping split spreads a zeroed rollout's loss over three terms
   using whatever the car did after the event, and it files an alternative route under `slow`
   (the projection onto the recorded path stops advancing). So each rollout's whole loss is also
   given to the single rule that ended its window: `gate` / `rear` / `cut` / `full`.
2. **A route-end-aware "4 m rule off".** Switching the modifier off literally also counts events
   after the recorded route has ended, which penalises whichever arm finishes the route sooner.
   The second variant stops scoring when `progress_rel_to_total >= 0.999`.
3. **The human driver as the yardstick** — the part "What this cannot do" ruled out. The usdz
   track turned out to share the sim's clock (GT arc length rebuilt from the parquet agrees to a
   median 1 mm), so pace, metres ahead / behind and the behaviour at each place the human stood
   still are all available without a GPU.

4. **Is it the stopping?** (summary `[9]`, added the same day in answer to that question.) G3
   passes on its letter, but "stops less, therefore scores more" needs its own test: scenes are
   split by whether either model ever stood still inside its scored window, and the parts add up
   to the gap. 82% of the gap sits in scenes where neither model stood (87% where the human never
   stood either), and 80% of the pace difference is speed while moving. So H-stop holds as
   "slower", not as "stops more".

The stop classification went through three definitions, and the first two were wrong in ways only
the time series showed: passing 5 m beyond the human's point counted a car pulling up closer to
its leader as "left early"; requiring 20 m before the human moved off made a slower car that also
never stopped look as if it had waited. The one kept is behavioural — inside the band from 15 m
before to 10 m past the point, did the car stand for 1 s, dip below 2 m/s, or neither — with the
timing reported as strata rather than used to decide the class.

## Result

G0 PASS (3,958 rollouts, max |diff| 0). G1: `slow` explains (53% of the gap on Origin150, 44% on
Hard100, CI excluding 0 on both). G2 H-route: not supported (alternative-route rate 6.3% dense vs
5.7% MoOP; 10.9% vs 13.0%). G3 H-stop: supported. Numbers and the post hoc sections are in
`outputs/closedloop_gap_v1/summary.txt` and Todo P4.

## Addendum 2026-10-12 (post hoc; added on request, no number above changed)

Two tables were added to the summary, each on its own random generator so that every number
printed before stays as it was (verified: the new summary differs from the old one by added
lines only).

- **[2c] pace table.** Rear-ended rollouts, rollouts that ran all 20 s short of 80% of the route,
  what each costs when it happens, and rollouts that drove past the end of the recorded route.
  Origin150: rear-ended 9.3% dense vs 4.3% MoOP; ran short 5.7% vs 2.0%; past the route end 25.0%
  vs 37.7%. The two pace terms give MoOP +0.041 [+0.021, +0.066] of its +0.079 (53%; pooled 45%).
  They give LLM-Pruner nothing (−0.000 [−0.035, +0.031]): it is as slow as the dense model and
  earns its +0.061 through the 4 m and gate terms. "Half the gap is pace" describes MoOP, not
  pruned models in general.
- **[5b] stops the car reaches while moving.** Looking at Hard100's nine places one by one showed
  that five are stops already under way when the scored clip starts: the car is handed over
  standing, "stopped" is automatic and a drive-through impossible. The same holds for 13 of
  Origin150's 61. Those places only pad the denominator, so the drive-through rate is now also
  reported on the stops that begin at least 2 s into the clip (48 on Origin150): dense 4.9%,
  MoOP 16.9%, MoOP+E 23.5%, FM-only 26.6%, Týr 33.3%, CE-only 42.9%, LLM-Pruner 56.9%, Wanda
  95.2%. Order and significance are those of the 61-place table; this is the version to quote.
  Hard100 keeps one usable place (five start-of-clip stops, two reached by no arm, one passed by
  every arm at over 10 m/s with the window ending inside), which the dense model never reached:
  the set was selected on speed and yaw and cannot test stop compliance.
