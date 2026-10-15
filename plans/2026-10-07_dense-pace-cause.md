# Plan: why is the dense model slower while moving in closed loop? (2026-10-07)

GPU-free. Follows `plans/2026-10-07_closedloop-gap-decomposition.md`, which found that MoOP's
closed-loop advantage over the dense model is pace — the dense model covers 0.952x the human's
distance, at 0.916x the human's speed while moving — and that the deficit builds in the first
5 s after the forced-GT hand-over. This plan asks where that deficit comes from.

## What prompted the hypotheses (seen before the gates were written)

One scene (`127c4f85`, one dense and one MoOP rollout) was read step by step from its
`rollout.asl`. In it:

- during forced GT the controller's reported acceleration ramps down and saturates at
  **-8.0 m/s²**, although the car is being moved along the recording at constant speed;
- at the first free step the car's speed falls from 10.85 to 9.78 m/s in **both** rollouts;
- the dense model's next plan continues the deceleration (9.78 now, 8.8 over the next second,
  7.6 two to three seconds out, then 5.3), with a CoC that reads "Keep lane since the lane is
  clear ahead"; MoOP's next plan returns to 10.7 m/s.

Everything below is fixed before the other 3,956 rollouts are read.

## Hypotheses

- **H-artifact.** The speed lost at hand-over is produced by the harness (the controller's
  dynamic state), not by any driver model.
- **H-extrapolate.** The dense model reads that deceleration from its ego history and plans to
  continue it; the pruned models plan back to the speed the scene calls for. The dense model's
  slower pace is this response, held.
- **H-caution** (the alternative). The dense model slows for reasons it states: its CoC asks for
  a slowdown where the pruned models' does not.

## Data

`rollout.asl` of the eight paper arms on Origin150 and Hard100. Per control step from 1.7 s before
hand-over to 10 s after: the state handed to the controller (`controller_request.state`, the
car's speed), the controller's reported acceleration (`controller_return.dynamic_state`), the
driver's plan (`driver_return.trajectory`, 65 poses over 6.4 s; mean planned speed over 0–1,
1–2 and 2–3 s) and its CoC text. The human's speed comes from the usdz track, which shares the
sim clock. "Moving start" = at least 2 m/s at hand-over.

## Gates

- **A1 (artifact).** On moving starts, the relative speed change over the first free control step
  is the same in every arm: per scene, max minus min across all arms' rollouts below 0.5 points
  of speed in at least 95% of scenes. And the controller's acceleration at hand-over is at
  saturation (<= -7.9 m/s²) in a majority of moving starts.
- **B0 (control).** At the last plan issued on identical inputs (the plan at hand-over, before any
  arm has driven a step), `r = planned speed 2–3 s out / current speed - 1` differs between dense
  and MoOP by less than a quarter of the B1 difference.
- **B1 (extrapolation).** At the first plan after the drop, `r` is lower for dense than for MoOP,
  scene-paired 95% CI excluding 0, on both sets.
- **C1 (dose).** Across scenes, the larger the hand-over drop, the larger dense's shortfall 2.5 s
  later relative to MoOP's: Spearman between the drop and (dense - MoOP) speed change, CI
  excluding 0 with the predicted sign, on both sets.
- **D1 (score).** The MoOP - dense score gap is larger in the top tertile of scenes by drop size
  than in the bottom tertile, pooled CI excluding 0. Weakest link: drop size rises with speed, so
  scene type is confounded; a pass is supportive, not decisive.
- **E1 (H-caution).** Among dense plans in the first second after hand-over that ask for a slowdown
  of more than 10% within 2–3 s, the share whose CoC names a slowdown (decelerate / slow / stop /
  yield / brake / keep distance / follow). Above one half supports H-caution; below one quarter
  rejects it.

Reading: A1 + B1 + B0 identify the cause as harness artifact x history extrapolation. C1 and D1
connect it to pace and to the score. E1 says whether the reasoning text takes part.

## Amendment: an across-arm test (written after the eight-arm result, before any other arm was read)

D1 failed and was a poor instrument: the absolute drop is the same 1.06 m/s everywhere, so the
relative drop is only inverse speed. The eight arms suggested a better one — the dense model is
the only arm that extrapolates, and every pruning method removes the behaviour — and about 27
further Origin150 arms have logs (budget ladder u20 / u30 / u40, head-vs-MLP reallocations,
MLP-only, expert-only, quantised, recovered, other calibration draws, Tyr and LLM-Pruner
variants). They give a graded dose of pruning that the eight arms do not.

Index, fixed here: `X` = over moving starts, the scene-mean of (planned speed 2–3 s out / speed
now − 1) for the plan made **0.2 s** after the hand-over (the step at which the dense model's
response peaked in the eight-arm table). Pace `P` = car / human speed 2.5 s after the hand-over.

- **G-arms.** Across all Origin150 arms, Spearman(X, score) > 0 with a bootstrap CI over arms
  excluding 0. Arms whose reasoning has collapsed lose their score to gates rather than pace, so
  the same statistic is also reported without arms whose zeroed-rollout rate exceeds 15%.
- **G-dense.** The arms with an unpruned VLM (the dense model and the expert-only arm) have the
  two lowest X.
- **G-pace.** Spearman(X, P) > 0, CI excluding 0.

Not independent evidence in the strict sense — the arms share scenes and most share a criterion —
so this is a dose-response description, not a significance test of a new hypothesis.

## Result (2026-10-07; 3,958 rollouts on the eight arms, 10,500 on 35 Origin150 arms)

| gate | Origin150 | Hard100 | what it turned out to mean |
|---|---|---|---|
| A1 | FAIL | FAIL | The drop is the same in every arm on average (−10.67 to −10.69%; −7.25 to −7.47%) and equals 0.136 s × the controller's acceleration at hand-over (r = 0.998), which is at −8 m/s² in 77% of moving starts. The spread criterion was too tight: 88.7% / 93.2% of scenes are inside 0.5 points, not 95% — the first free step already executes each arm's own first plan. |
| B0 | FAIL | FAIL | There **is** a standing difference on identical inputs: dense plans 3.4 / 2.0 points lower than MoOP 2–3 s out, just over a quarter of the post-drop difference. It is MoOP that departs from the human (plans 1.012x against the human's 0.983x; dense 0.998x). |
| B1 | PASS | PASS | 0.1 s after the drop dense plans 12.9 [9.1, 16.8] / 7.8 [3.7, 13.0] points lower than MoOP; every one of the seven pruned arms plans higher than dense. |
| C1 | PASS | PASS | rho +0.23 [+0.07, +0.38] / +0.46 [+0.22, +0.66]. |
| D1 | FAIL (pooled) | | The score gap is not larger where the relative drop is larger (+0.028 [−0.057, +0.109]). |
| E1 | "supports H-caution" | same | By the letter only. The registered word list matches two thirds of **all** CoCs; with explicit slowdown words, 45% / 21% of dense's slowing plans name one against 36% / 16% of all its plans, and the CoC heads are distributed alike in dense and MoOP while dense asks for a slowdown twice as often (47% against 23%). |
| G-dense | PASS | | The two lowest X are the two arms with an unpruned VLM (dense −25.2%, expert-only −24.4%). |
| G-pace | PASS | | rho(X, P) = +0.78 [+0.60, +0.89] over 35 arms. |
| G-arms | FAIL / PASS | | +0.19 [−0.20, +0.58] over all 35; +0.56 [+0.20, +0.82] over the 29 that zero at most 15% of rollouts. |

The harness half is in the code: the forced-GT trajectory handed to the controller ends at the
hand-over (`event_loop.py:268-317`, `events/force_gt_utils.py:13-21`), the MPC clamps horizon
times past the end of its reference to the last pose (`linear_mpc.py:186-187`) and so brakes
toward it, and forced steps overwrite the vehicle model's velocity but not its acceleration
(`system.py:155-157`). The car is released with about −8 m/s² already commanded.

So: **cause identified for the pace, not for the score.** The dense model is slower because it
continues a deceleration the simulator itself injects; pruning the VLM removes that response in
proportion to how much is pruned (12% −19.7%, 18% −3.9%, 24% +1.2%); pruning the expert does
not. What the dense model would score without the artifact is not in these logs.

## What this cannot do

It observes plans, not the mechanism inside the model, and it cannot say how the dense model
would score without the artifact. Two follow-ups would: an open-loop perturbation (shift the last
ego-history pose so the history shows the same drop; about 2 GPU-h) and a closed-loop rerun of
dense and MoOP with the controller state reset at hand-over (2 x 32 GPU-h).

## Outputs

`outputs/handover_cause_v1/{config.json,metrics.json,summary.txt,steps.parquet,plots/}`, script
`experiments/evaluation/analyze_handover.py` (extraction needs alpasim's venv for `alpasim_utils`).
