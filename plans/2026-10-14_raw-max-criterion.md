# Plan: raw-value max with a rescaled trajectory score (2026-10-14)

## Question

`dual` = `max(rank_norm I_traj, rank_norm I_CoC)`. `rank_norm` throws away magnitude, and the
raw `max(I_traj, I_CoC)` was rejected because `I_traj` is ~10x `I_CoC` (FM decides 93.9% of
kept Q heads, `2026-09-09_criterion-balance.html`). The ~10x is mostly the **sum over the ten
FM steps** (`I_traj = E_c |sum_s g_cs|`; one step's `|grad|` averages 1.39x / 0.85x `I_CoC`
on Q / MLP). Proposal: take the max on raw values after putting the trajectory score on a
per-step scale, either

- `rawmean`    `max(E_c |mean_s g_cs|, I_CoC)` = `max(I_traj / 10, I_CoC)`
- `rawabsmean` `max(E_c mean_s |g_cs|, I_CoC)`

Hypothesis H (user): once the scale is fixed, raw magnitudes carry information `rank_norm`
discards, and the raw max selects better than `dual`.

## Pre-check (GPU-free, done 2026-10-14)

Kept sets computed in numpy from `importance_v2` (Blackwell, the shipped arm's file) and, for
`rawabsmean`, `importance_stepvlm_v1` + `importance_v2_ada` (Ada; the per-step file's summed
score equals `importance_v2_ada`'s `I_traj` to 3e-3 relative). Budget u40 (13 / 32 heads,
4,898 / 12,288 channels per layer). Script: `tmp/rawmax_{precheck,bands,extra}.py`, to be
committed as `experiments/evaluation/analyze_rawmax_precheck.py`.

| rule | overlap with dual Q / MLP | kept decided by `I_traj`, layers 0-21 Q / MLP | layers 22-34 Q / MLP | layers where one score decides >= 90% (of 35), Q / MLP | late band = `coc_u40_v2`'s Q / MLP |
|---|---|---|---|---|---|
| `dual` | 1 / 1 | 0.46 / 0.49 | 0.48 / 0.49 | 0 / 0 | 0.80 / 0.81 |
| raw `max` (`|sum|`) | 0.873 / 0.870 | (94% / 96% overall) | | | |
| `rawmean` | 0.895 / 0.888 | 0.76 / 0.81 | 0.11 / 0.03 | 14 / 14 | **0.976 / 0.991** |
| `rawabsmean` | 0.901 / 0.881 | 0.89 / 0.96 | 0.17 / 0.06 | 15 / **29** | 0.923 / 0.957 |
| `lnorm` = `max(T / mean_l T, C / mean_l C)` | **0.952 / 0.977** | 0.44 / 0.48 | 0.48 / 0.48 | 0 / 0 | 0.82 / 0.82 |

Why: the per-layer ratio `mean(I_traj/10) / mean(I_CoC)` is not a constant. It is 0.94-2.01
(Q) / 1.07-1.50 (MLP) in layers 0-21 and 0.17-0.59 / 0.17-0.76 from layer 22 on (the 4.6x step
of `2026-09-20_gradient-anatomy.html`). A global constant fixes the *average* (`rawmean` is
50% / 50% overall) but not each layer, so neither proposal is a raw union: both are a
**depth-switched** criterion, a trajectory-leaning trunk plus a CoC-only late band.
`rawabsmean` leans further because `mean_s |g_s|` has a different per-layer ratio to `I_CoC`
(and 47% of it is the t = 0.05 step); its within-layer ranks are almost those of `|sum|`
(Spearman median 0.99, kept 96-97%).

`lnorm` divides each score by its own per-layer mean, so it removes the scale but keeps the
raw distribution shape. It lands at 95.2% / 97.7% of `dual` with no one-sided layer: **the
magnitude information beyond rank moves 5% / 2% of the kept set.**

Layer 0 (the confound that made calibration draws collapse, `2026-10-13_layer0-exemption.html`):
`rawmean` and `lnorm` drop head 7 (single-head cost +0.089 on test500), `rawabsmean` drops 24
(+0.091). Unexempted, those arms would be read through the layer-0 lottery (17 arms
0.950-1.774 vs 0.930-0.992 exempted). **Every arm here is built with `_l0keep`.**

Also found: the priors this plan would lean on (`armmix_v1` / `armheld_v1`, the 2026-09-21
causal round: "trunk union FM +4.8% vs `coc` +17.0% vs `traj` +26.1%") used masks that
include layer 0, where `coc` drops head 30 (+0.318 alone) and `traj` drops 24. Those trunk
numbers are therefore not clean evidence for the union either; the `_l0keep` endpoints below
re-measure them.

## Arms (all `_u40_l0keep_v2`, VLM only, budget = `dual_u40_l0keep_v2`'s)

| id | config | importance | role |
|---|---|---|---|
| R | `dual` | v2 | reference, built; test500 / val500 / OOD-val = `swap_ship_L0fullM_{500,val500,oodval}` |
| A1 | `rawmean` | v2 | proposal 1 |
| A2 | `rawabsmean` | v2_ada + stepvlm_v1 | proposal 2 |
| R' | `dual` | v2_ada | A2's reference (same file pair), out `slim_dual_ada_u40_l0keep_v2` |
| C1 | `lnorm` | v2 | scale removed, shape kept: does raw magnitude matter at all? |
| C2 | `dualv3` | v2 | `dual` trunk + `coc`-only late: isolates A1's late half (plan 2026-09-25, never run) |
| E1 / E2 | `traj` / `coc` | v2 | endpoints; also the open paper-baseline re-measurement |

## Step 0 - code (CPU + one card for building)

`make_slim.py`: three stems in the `uni` branch, next to `dualsum`/`dualprod` -
`rawmean` (`op = max`, no `rank_norm`, traj half / 10), `rawabsmean` (traj half =
`stepvlm` `{q,mlp}_abs_step.mean(0)`), `lnorm` (each half / its per-layer mean, guarded
against layer 35's all-zero `I_traj`). Build all seven with `--no-state` (~30 s each).
Checks: unexempted `rawmean_u40_v2` / `lnorm_u40_v2` reproduce the pre-check kept sets
exactly; `dual_u40_v2` and `dual_u40_l0keep_v2` regressions unchanged; every arm keeps 32
layer-0 heads and removes 2,657,447,936 params.

## Step 1 - held-out mechanism (Ada, ~25 min on four cards)

`run_token_ablation.py --arm-masks` on the same 100 held-out `indist_500` clips and seeds as
`armmix_v1` (K = 8): dense + R, A1, A2, R', C1, C2, E1, E2 (masks via `slim_to_mask.py`).
Read: FM loss, CoC NLL, minADE deltas, paired per clip. Not a decision gate; registered
predictions:

- A1's late band is C2's (97.6% / 99.1%), so NLL(A1) and NLL(C2) are both below NLL(R).
- A1 / A2 trunks lean to `traj` (trunk overlap with `traj_u40_v2` 0.93-0.96 vs R's 0.89-0.90),
  so FM(A2) >= FM(A1) >= FM(R).
- E1 / E2 trunk damage is smaller than `armmix_v1`'s (+26.1% / +17.0%) once layer 0 is kept.

## Step 2 - decision, test500 open loop (Ada, ~25 min per arm on four cards)

`run_baseline.py --set test` (own rollout, minADE@6 / minFDE@6), paired on clips and seeds
with R (A2 with R'). Seven arms ~3 h. Gates, per arm vs its reference:

- primary: mean paired delta minADE@6, bootstrap 95% CI with 20,000 draws, median + Wilcoxon
  alongside.
- **H accepted** (arm better than `dual`): CI upper < 0 **and** mean <= -0.03 m **and** it
  replicates on val500 (CI upper < 0). The 0.03 margin is the floor measured after the
  exemption: arms that differ only in calibration draw or seed sit at SD 0.018, residual
  +0.015 to +0.029 over the shipped arm, so a smaller gap cannot be told from kept-set churn
  in other layers.
- worse: CI lower > 0. Equivalent: CI inside +/-0.03.
- language: CoC degeneracy rate, paired McNemar vs the reference.

Registered predictions: A1 equivalent (|delta| < 0.03) with degeneracy <= R; A2 equivalent
or worse; C1 equivalent. E1 / E2: no prediction (the only prior is layer-0-contaminated).

## Step 3 - replication

val500 for any arm that clears "better" or "worse" on test500 (~25 min each). OOD-val only if
an arm is accepted. Closed loop is out of scope: new closed-loop arms are on hold until the
handover-braking artefact is removed.

## What each outcome means

- A1/A2 better and replicated: H holds; the trunk-lean / late-CoC split is the mechanism to
  name, and C1 says whether magnitude or the depth switch did it.
- A1/A2 equivalent, C1 equivalent: rank vs raw is not a lever at this budget; `dual` stands.
- A1/A2 worse with C2 not worse: the trajectory-leaning trunk costs; the union is needed there.
- E1 ~ E2 ~ R after the exemption: the union's open-loop advantage was the layer-0 lottery,
  which the paper's "why dual" argument must then say.

## Outputs

`outputs/slim_{rawmean,rawabsmean,lnorm,dualv3,traj,coc}_u40_l0keep_v2`,
`outputs/slim_dual_ada_u40_l0keep_v2`, `outputs/rawmax_heldout_s{0..3}` + merged,
`outputs/<arm>_l0keep_500` (+ `_val500`), `outputs/rawmax_compare/{config.json,metrics.json,
summary.txt,plots/}`, report `reports/evaluation/2026-10-1x_raw-max-criterion.html`.
