# Plan: can a functional-redundancy signal see which layer-0 head sets collapse? (2026-10-14)

## Question

Which 13 of layer 0's 32 Q heads a `dual_u40_v2`-family arm drops decides whether it collapses
on cruise (`2026-10-13_calibration-variance-is-layer0.html`). The criterion is blind there:
damage vs dual score rho -0.30, and the damage is a **combination** effect (block d drops 4, 7,
10, 26 together and loses +1.34; only head 30 hurts alone, +0.318). The working fix,
`_l0keep`, exempts the layer by hand.

Grouping units by similar score does not help on its own: the cut is noise-decided in every
layer (bootstrap, `outputs/cut_ambiguity/`: Q median 7 of 32 heads with 0.1 < P(kept) < 0.9,
layer 0 at 13 but layers 15/33/18 at 10-12 and harmless; exact ties at the cut in 8/36 Q
layers). Inside such a band the choice needs a signal about **which heads substitute for
each other**.

Hypothesis H: a functional-redundancy signal computed **from calibration data only, with no
outcome fitted**, ranks the stored arms by their measured damage, as well as or better than
the outcome-informed layer-0 census does. If it does, it can choose inside the band and replace
the hand exemption at an unchanged budget.

## Data already on disk (no new evaluation for Steps 1-2)

The outcome is each arm's whole score. After the exemption the arms' other layers move the
score by SD 0.018, so the score is mostly a function of the layer-0 set.

| group | arms | distinct layer-0 sets | outcome | bar set by the census (`swap_masks/summary_census.txt`) |
|---|---:|---:|---|---|
| A: dual-family, test500 | 50 | 34 | mean minADE@6, Ada | `n_off` rho +0.553; ridge fitted on B -> A +0.549 |
| B: calib_100 half-split | 64 | 44 | delta minADE@6 on 150 val500 clips | `n_off` rho +0.204 (p = 0.11); ridge fitted on A -> B +0.435 |
| C: single-head map under the shipped mask | 19 | 19 | cruise60 minADE@6 (`metrics_l0.json`) | dual score rho -0.30 |

Groups A and B together hold 73 distinct sets (5 shared), and 19 heads vary at all (6 are
dropped by every arm). `n_off` counts heads dropped that the shipped arm keeps, so it uses
the fact that the shipped arm is good, and the ridge is fitted to outcomes. The signals below
use neither.

## Signals (all at layer 0, from calib_100)

Let `x` be the o_proj input at layer 0 (32 heads x 128), `H = sum_t x x^T` over the forward's
own token mixture (vision + text + the model's own CoC, the K = 4 rollouts of `racfit_v1`),
`W` the o_proj weight and `W_h` its 128 columns of head h. Every signal is a closed-form
function of `H` and `W`, so one capture scores all 73 sets and any future set in seconds.

- `G_hk = tr(W_h H_hk W_k^T)` (32 x 32): the Gram matrix of the heads' contributions to the
  layer output. `G_hk > 0` means heads h and k write the same direction.
- **P1, primary**: `e_mask(D) = sum_{h,k in D} G_hk / sum G`, the relative output energy the
  dropped set D removes, **cross terms included**. Selection-only arms do not refit, so this is
  the local quantity they actually change. Its cross terms are the combination effect: dropping
  heads that write the same direction removes more than the sum of the parts.
- S1 `e_refit(D)`: the output energy left after the best linear refit from the kept heads
  (Schur complement of `H_KK`, damping 1e-2 x mean diag, as in Tyr). This measures redundancy
  in the strict sense: how much of D's information the kept heads still carry.
- S2 cluster coverage: average-linkage clustering of heads on `1 - G_hk / sqrt(G_hh G_kk)`.
  Score = the largest energy-weighted fraction of any one cluster that D removes. k = 6
  clusters is reported; k = 4 and 8 are descriptive.
- S3 stream-restricted `e_mask`: the same P1 with `H` from vision only, and from own-CoC only.
- B0 control, outcome-free: `sum_{h in D}` of the shipped importance file's dual score.
  This is what the criterion itself believes.

The prediction direction for every signal is positive: a larger value means a worse arm.

## Step 1 - capture + GPU-free test (~5 min on one card, then seconds)

```bash
ALPAMAYO_REPO=$PWD bash experiments/head_analysis/run_retry_host.sh 30 experiments/head_analysis/run_racfit.py \
    --exp-id l0red_h_calib100 --layer-start 0 --layer-end 1 --folds 1 --mixes VT \
    --keeps-q 19 --keeps-m 7390 --no-dual --dump-hessian --rollouts-from racfit_v1 --gpu <free>
.venv/bin/python experiments/evaluation/analyze_l0_redundancy.py      # new, GPU-free
```

`--folds 1 --dump-hessian` saves the per-stream `H` (V / T / D, about 200 MB). The capture
path is `run_racfit.py`'s, whose G0 gate (layer-0 Tyr reproduction < 1e-3) passed on
2026-08-25. A wrong-module or wrong-stream bug is still caught here by **G0**: `racfit_g0h`
(2026-08-25: same clips, same rollouts, layer 0, `--folds 1`, K = 1) stored the layer-0
o_proj stream traces. V and T do not depend on K, so the new dump's `tr(H_V)` and `tr(H_T)`
must match them to a relative 1e-3. D is checked against its token count (K = 4 against
K = 1). *(Amended 2026-10-15 before running: the first draft's direct-hook check needed a
second forward pipeline, and the Tyr checkpoint the original G0 compared against no longer
has its weights on disk.)*

Gates, pre-registered:

- **G1 (P1)**: Spearman(`e_mask`, outcome) >= +0.45 with p < 0.01 on group A, **and**
  >= +0.20 with p < 0.05 on group B. That is the bar `n_off` reaches on A but fails on B
  (p = 0.11), using no outcome. The same is reported on distinct sets (outcomes averaged over
  arms sharing a set: 34 / 44), with arm-bootstrap 95% CIs (20,000 draws).
- **G1b**: P1 beats B0 on both groups (paired arm bootstrap of the rho difference, CI above 0).
- Secondary signals S1-S3 and group C are reported with Holm correction across S1-S3. None of
  them can pass G1 alone: if only a secondary passes, it goes to a replication on the other
  outcome group before Step 3.

Prediction, registered: P1 about 40% likely to pass. That needs the substitutable heads to
write overlapping directions; a downstream nonlinearity (a threshold on a signal several heads
supply) would leave P1 blind and is what Step 2 is for.

## Step 2 - propagated co-ablation on calibration clips (~2.5 h on four Ada)

Cost is measured, not guessed: `armmix_v1` ran 8 masks at K = 8 in ~37 s per clip, i.e.
~4.5 s per mask-clip, so 75 masks x 100 clips is ~9.4 card-hours.

If the combination effect lives downstream, no layer-0-local quantity can see it, and the
signal has to be measured after the rest of the network. `run_token_ablation.py --arm-masks
--manifest calib_100 --cache calib`: the shipped mask in layers 1-35, layer 0 replaced by each of
the 73 sets, plus dense and the shipped arm. Read FM loss and CoC NLL, teacher-forced, on all
100 calibration clips. Masks come from `make_l0_redundancy_masks.py` (new; a splice like
`make_l0_masks.py`).

- **G2**: Spearman(delta FM loss, outcome) with the same thresholds as G1 on groups A and B.

Run Step 2 whatever Step 1 shows, because the reading needs both columns:

| P1 (local) | P2 (propagated) | reading |
|---|---|---|
| pass | pass | redundancy is geometric and local; P1 is the cheap selector |
| fail | pass | the effect is downstream; selection needs a calibration co-ablation search (greedy), not grouping |
| pass | fail | P1 is right for a reason P2's teacher-forced loss misses; trust it only after Step 3 |
| fail | fail | calibration data cannot see the damage; `_l0keep` stays; record that redundancy-based grouping is not the lever |

## Step 3 - the selector replaces the exemption (only if G1 or G2 passes; ~3 h on four Ada)

Rule: in layer 0, heads outside the bootstrap band (P(kept) >= 0.9 or <= 0.1, from each arm's
own importance file) are fixed by the dual score. Inside the band, the heads to drop are chosen
to minimise the passing signal, and the number dropped stays 13. So **the budget is bit-identical
to `dual_u40_v2`**, with no MLP payback.

- P1 can be enumerated exactly: about C(13, 6) = 1,716 subsets, scored in milliseconds from `G`.
  P2 is scored greedily, one head at a time.
- Apply the rule to the eight arms the exemption was tested on: shipped, ref, c3, n1 and nt
  blocks b-e. `H` comes from each arm's own calibration clips, so nt b-e need four more 5-minute
  captures.
- Kept sets are spliced as in `make_l0_exempt.py`. Evaluate on `test_cruise120` first (the
  collapse is +1 m on cruise), then test500, on Ada.
- **G3**: across the eight arms, test500 SD <= 0.05 and no arm above 1.05. For comparison: as
  built 0.950-1.774, SD 0.337; exempted SD 0.015. Each arm must also be within +0.03 of its
  `_l0keep` twin (95% CI upper bound).

If G3 passes, the rule does what the exemption does, at the original budget and without naming a
layer. Extending it to all 36 layers is a follow-up plan. Other layers are harmless at u40, so
it matters only at higher budgets.

## What this does not settle

- Why layer 0 alone. The other layers' bands are just as wide and cost nothing.
- Closed loop. New closed-loop arms are on hold until the handover-braking artefact is removed.
- Arms built from criteria other than dual.

## Outputs

`outputs/l0red_h_calib100/sol/L00_{o,m}_H.pt`; `outputs/l0_redundancy/{config.json,
metrics.json, summary.txt, plots/}`; `outputs/l0red_prop_s{0..3}` + merged; (Step 3)
`outputs/slim_swap_<arm>_L0red`, `outputs/swap_<arm>_L0red_{cruise120,500}`; report
`reports/evaluation/2026-10-1x_l0-functional-redundancy.html`.
