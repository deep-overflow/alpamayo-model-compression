# Why do `I_traj` and `I_CoC` differ? — follow-up analysis to Figure 1

Date: 2026-09-20. Branch: `worktree-why-importance-differs`.
Script: `experiments/paper/fig_why_differs.py` → `figures/fig3_*` and
`figures/fig3_why_differs_stats.json` (every number below that is not attributed to an
older run comes from that file).

## Purpose

Figure 1 (`fig1_depth_mlp`, `fig1_depth_q_head`, `fig1_rank_agreement`) establishes
*that* the two objectives weight the VLM differently: `I_traj` peaks at layer 18–19 and
`I_CoC` at 24–27, and within-layer rank agreement falls after layer ~22. The manuscript
builds its mechanism claim (C2) on that observation but does not say *why* the profiles
part where they do. This note asks three questions a reviewer will ask:

1. Are these two independent depth profiles, or one phenomenon seen twice?
2. Is the late-layer rank disagreement real, or is it what unreliable scores look like?
3. What changes in the network at the depth where they part?

No new GPU measurement was made. Everything is recomputed from stored runs, and the one
claim that stored data cannot settle is written up as a plan
(`plans/2026-09-20_gradient-anatomy.md`) rather than asserted.

## Setup

| item | value |
|---|---|
| model | dense `nvidia/Alpamayo-1.5-10B`, VLM tower: 36 layers, 32 Q heads, 12,288 MLP channels |
| scores | gate-Taylor `E_c|dL/dg|`, exactly as in Figure 1 (`paper/2026-09-09_figure-reproducibility.md`) |
| main run | `outputs/importance_v2` (`calib_100`), aggregate and per-clip arrays |
| other runs | `importance_rd100_a/b/c`, `se100_a`, `su100_a`, `val100`, `ood`, `nt500`, `st4000`, `selftraj_v1`, `dim0`, `dim1` (robustness); `importance_vqa` (VQA-NLL and CoC-NLL on LingoQA images); `jlens_v2` (Jacobian lens) |
| causal evidence cited | `outputs/pathway_x_v1`, `outputs/pathway_e_v1` (attention knockouts, n=50), `outputs/cacheuse_v1` (expert cache-read knockouts, n=100) — all already in the draft's Sec. 7 |
| layer 35 | excluded everywhere: `I_traj` is identically zero there |
| bands | "early" = layers 0–21, "late" = 22–34, the same shading as `fig1_rank_agreement` |

## Results (facts)

### R1. The two depth profiles are one profile times one step

Take the raw layer sums (not the per-curve normalised ones) and divide:

| axis | `I_traj / I_CoC`, layers 0–22 | layers 23–34 | step | bootstrap 95% CI | R² of a single step |
|---|---:|---:|---:|---|---:|
| Q head | 14.4× | 3.1× | 4.64× | [3.85, 5.47] | 0.86 |
| MLP channel | 12.7× | 2.8× | 4.56× | [3.37, 6.07] | 0.93 |

- Inside the early plateau the MLP ratio has a coefficient of variation of 9% while
  `I_traj` itself varies 15.0× over the same layers. The two curves rise together.
- The least-squares change point is layer 23 (first layer of the lower plateau) on both
  axes; it is 23 in 100% (Q) and 97.9% (MLP) of 2,000 clip-bootstrap resamples, and the
  step is present in 100% / 99% of the 100 clips taken one at a time.
- The step alone reproduces Figure 1's peak shift. Multiplying `I_CoC` by the two-level
  ratio puts the trajectory peak at layer 18 (Q; actual 19) and 21 (MLP; actual 18, on a
  top that is flat over 18–21); dividing `I_traj` by it puts the CoC peak at 24 (Q; actual
  24) and 25 (MLP; actual 27).
- The levels 14× and 3× mean nothing — the losses are in different units. Only the
  flat–step–flat *shape* is a finding.

Panels: `fig3_ratio_step`.

### R2. The step belongs to the network, not to the calibration data or the target

Change point of the same fit in twelve other stored runs (MLP, Q):

| run | what differs from `importance_v2` | change point |
|---|---|---|
| `rd100_a/b/c`, `se100_a`, `su100_a`, `val100` | disjoint 100-clip calibration sets, three drawing rules | 23, 23 |
| `nt500`, `st4000` | 500 and 4,000 clips | 23, 23 |
| `ood` | long-tail OOD clips | 23, 23 |
| `selftraj_v1` | trajectory target sampled from the model instead of GT | 23, 23 |
| `dim0` | acceleration dimension only | 23, 23 |
| `dim1` | curvature dimension only | 24, 23 |

`selftraj_v1` matters most: `I_CoC` scores the model's own rollout while `I_traj` is
anchored on ground truth, and that asymmetry is a natural suspect. Removing it moves the
step's size (3.2× instead of 4.6× on MLP, from the earlier sweep) but not its place.

Panel: `fig3_ratio_step_draws`.

### R3. What moves at the boundary differs by axis

| | `I_traj`, layer 21 → 23 | `I_CoC`, layer 23 → 27 |
|---|---:|---:|
| MLP channel | ÷2.36 | ×2.32 |
| Q head | ÷2.90 | ×0.41 (no rise) |

On the MLP axis the step has two parts — trajectory importance falls, then CoC importance
rises. On the Q-head axis it is the trajectory fall alone; CoC head importance stays in
the 0.02–0.06 band from layer 14 to 35.

### R4. The rank disagreement is real for Q heads and partly attenuation for MLP channels

Split the 100 calibration clips into disjoint halves (50 random splits) and correlate
half-sample scores across halves. "Self" is the ceiling any agreement can reach.

| axis | `I_traj` vs itself | `I_CoC` vs itself | `I_traj` vs `I_CoC` | corrected for the ceiling |
|---|---|---|---|---|
| Q head, early → late | 0.85 → 0.82 | 0.90 → 0.89 | 0.74 → 0.33 | **0.85 → 0.37** |
| MLP, early → late | 0.69 → 0.46 | 0.80 → 0.50 | 0.69 → 0.26 | **0.92 → 0.54** |

Three disjoint 100-clip draws measured as separate GPU runs (`rd100_a/b/c`) give the same
corrected values by a different route: Q 0.85 → 0.37, MLP 0.90 → 0.53.

- Q heads: each objective reproduces its own ranking equally well in both depth bands, so
  the late-layer collapse in `fig1_rank_agreement` is a real disagreement. The falling
  ceiling accounts for 3% of the (multiplicative) fall in agreement, by either route.
- MLP channels: the ceiling itself falls in late layers — full-sample (n = 100)
  self-agreement from the disjoint draws is 0.87 → 0.68 and 0.87 → 0.63 — so the late
  value of 0.34 in Figure 1c has to be read against an attainable ~0.65, not against 1.
  The falling ceiling accounts for 34% of the fall at n = 100 (44% at n = 50); the rest is
  disagreement, 0.92 → 0.54 once corrected. Late-layer MLP agreement is about half of
  what is attainable (0.52), against 0.38 for Q heads.
- In layers 6–19 on the MLP axis, the *other* objective predicts a held-out half as well
  as the same one: CoC-on-half-A vs trajectory-on-half-B is 0.60, trajectory-on-A vs
  trajectory-on-B is 0.59. Below the boundary the two scores are interchangeable.

Panels: `fig3_noise_ceiling_q_head`, `fig3_noise_ceiling_mlp`.

### R5. A clip × objective partition says what replaces the shared component

Single-clip rank vectors, three mean correlations per layer: other clip and other
objective (= shared), same clip and other objective (adds the clip-specific part), other
clip and same objective (adds the objective-specific part).

| axis | band | shared | objective-specific | clip-specific | shared / (shared + objective) |
|---|---|---:|---:|---:|---:|
| Q head | 0–5 | 0.353 | 0.040 | 0.045 | 0.89 |
| Q head | 6–21 | 0.251 | 0.042 | 0.066 | 0.85 |
| Q head | 22–34 | 0.138 | **0.127** | 0.040 | 0.49 |
| MLP | 0–5 | 0.613 | 0.003 | 0.019 | 0.99 |
| MLP | 6–21 | 0.096 | 0.008 | 0.082 | 0.92 |
| MLP | 22–34 | 0.036 | 0.027 | **0.118** | 0.60 |

- The last column is what aggregate agreement should be once clip noise averages out. It
  matches the ceiling-corrected agreement of R4 without using it — exactly in the early
  band (0.85 and 0.92 against 0.85 and 0.92) and to within 0.06–0.12 in the late band
  (0.49 and 0.60 against 0.37 and 0.54) — which is the check that the partition is
  measuring something.
- Q heads: the shared part halves and the objective-specific part triples. The
  objective-specific share exceeds the shared one at layers 23, 28, 29, 30, 31 and 33;
  at 28 and 29 it is 0.20 and 0.17 against 0.04 and 0.04.
- MLP channels: what grows late is the *clip*-specific part. Which late-layer channels
  matter depends on the scene, for both objectives alike — this is why the ceiling falls
  in R4, and it is the layer range where a 100-clip calibration set is thinnest.
- MLP layers 1–5 are a separate regime: 60–82% of single-clip rank variance is shared by
  every clip and both objectives, and the ranking is the unit's write norm (ρ 0.88–0.97).

The unplotted remainder (55–82%) is single-clip measurement noise plus the clip ×
objective interaction; one measurement per clip cannot separate them.

Panels: `fig3_partition_q_head`, `fig3_partition_mlp`.

### R6. Three independent measurements put a regime change at the same depth

| signal | what it uses | before | after | where |
|---|---|---|---|---|
| importance ratio | both losses' gradients | 12.7× | 2.8× | change point 23 |
| FM sensitivity to the cache tensors (`traj_kv_k + traj_kv_v`) | trajectory loss, expert side | layers 8–22 mean | 3.6× lower over 25–35 | 81% of the total lies at or below layer 22 |
| excess kurtosis of the Jacobian-lens readout over random vocabulary | **no loss at all** | 6.4 (layers 14–20) | 21.0 (layers 24–34) | two-level change point 22, R² 0.91 |

The third row is the one that is not circular: it says the residual stream at text
positions becomes token-peaked — starts carrying discrete, verbalizable content — over
layers 21–24, measured without either objective.

The causal knockouts already in Sec. 7 agree, at coarser (9-layer) resolution:

- Expert cache-read knockouts (`cacheuse_v1`): reliance by cache-layer band [0–7, 8–15,
  16–23, 24–35] = 0.41 / 0.40 / 0.09 / 0.10. The expert leans on the late third of the
  cache for a tenth of what it needs.
- VLM-internal edge knockouts (`pathway_e_v1`): attention edges *into a vision position*
  stop mattering after layer 17 — every such cell is within ±3% on both channels
  (same-camera earlier frames: L18–26 +1.0% / +1.1%, L27–35 −0.1% / +0.2% on action /
  language; the one FDR-significant cell, cross-camera at L18–26, is +1.6% on language).
  The edges that matter in layers 18–35 are text positions reading vision — CoC←vision
  NLL +18.2% and +12.8%, instruction←vision +12.4% and +4.2% — and they move language
  only (action within ±1.8%).
- Expert span knockouts (`pathway_x_v1`): 73% of the expert's attention mass is on the
  2,880 vision positions and blocking them costs +2.38 m; everything that is not vision
  is 216 positions.

Panel: `fig3_boundary`.

### R7. Late-layer CoC importance is text-position activity; trajectory importance is not

The J-lens run stores each unit's write norm over text and CoC positions only. Rank
correlation with each objective's score, layers 6–21 → 22–34:

| axis | `I_CoC` | `I_traj` |
|---|---|---|
| Q head | 0.50 → **0.77** | 0.24 → 0.25 |
| MLP channel | 0.48 → 0.44 | 0.23 → 0.30 |

For Q heads, how much a late-layer head writes at text positions nearly determines its
CoC importance and says nothing about its trajectory importance. The MLP axis does not
show the contrast.

Panels: `fig3_text_write_q_head`, `fig3_text_write_mlp`.

### R8. Late layers follow the output, not the input and not the read-out head

Within-layer agreement for four pairs of scores, layers 6–21 → 22–34:

| pair | shares | Q head | MLP |
|---|---|---|---|
| CoC-NLL on `calib_100` vs CoC-NLL on LingoQA images | output, not input | 0.80 → **0.87** | 0.74 → 0.50 |
| acceleration vs curvature (two halves of the FM loss) | input and read-out | 0.79 → 0.67 | 0.55 → 0.36 |
| VQA-NLL vs CoC-NLL, same images | input images and LM head, not the text | 0.57 → 0.42 | 0.63 → 0.27 |
| trajectory vs CoC | input (same forward pass) | 0.77 → **0.33** | 0.68 → 0.34 |

- Q heads, late: a different dataset changes nothing (0.87) while a different output on
  the *same forward pass* changes almost everything (0.33).
- Two language objectives through the same LM head disagree late nearly as much as
  trajectory and CoC do (0.42 vs 0.33; on MLP 0.27 vs 0.34). So the late-layer split is
  not "action head versus language head". It is output-specific.
- On the MLP axis every pair falls late, including two halves of the same loss — R4's
  attenuation again. Read the MLP column only next to the noise ceiling.

Panels: `fig3_pairs_q_head`, `fig3_pairs_mlp`.

## Analysis (opinion)

**One switch, not two profiles.** The most useful reframing is R1: Figure 1a/1b do not
show two objectives with their own idea of depth. They show a single profile that both
losses share, multiplied by a step at layers 22–23. Figure 1's different peaks, and the
collapse in Figure 1c, are the same event. The question "why do they differ" reduces to
"what happens at layers 22–23", and R2 says the answer has to be about the network.

**What happens there.** Putting R6–R8 together, the account I would defend is:

- *Below the boundary the VLM is building the scene representation, and both outputs
  consume it.* The expert reads it directly from the cache (its cache sensitivity and its
  causal reliance both sit here). The rationale needs it too, because the CoC is grounded
  in the same vision tokens. Both gradients therefore arrive at the same computation and
  rank its units the same way — to the point of being interchangeable (R4).
- *Above it the VLM is writing the rationale.* The stream at text positions turns
  token-peaked (R6), the edges that matter are text positions pulling from vision (R6),
  and a head's CoC importance becomes how much it writes at text positions (R7). The
  expert has little use for this: it reads mostly vision-position cache entries, no edge
  into a vision position matters after layer 17, and its sensitivity to the late cache is
  3.6× lower. So trajectory importance drops, CoC importance rises (MLP), and the rankings
  decouple.
- *A unit's gate is shared across all ~3,100 positions.* If the two losses' gradients
  land on different tokens in late layers — CE on ~200 text positions, FM on ~2,900
  vision positions — then the same head is being scored on two different jobs. That would
  explain why the split is real for heads (R4), why it tracks the output rather than the
  input (R8), and why two *language* objectives with different text also disagree (R8).

The last bullet is the part stored data supports but does not prove. R7 is the only
position-resolved evidence and it covers text positions only. The plan below measures it
directly and says in advance what would refute it.

**What it is not.** Not estimation noise (Q heads, R4). Not the calibration set, its size
or its drawing rule (R2). Not the GT-versus-own-rollout asymmetry between the two scores
(R2, `selftraj_v1`). Not "the LM head versus the expert" as such (R8, VQA).

**Two consequences for the method section.**

- The union is needed exactly where R5 says the objective-specific share overtakes the
  shared one, and is nearly free elsewhere: below layer 22 either score would select the
  same units (corrected agreement 0.85–0.92). That is a sharper statement of C2 than
  "each loss illuminates its own substrate", and it predicts where single-criterion arms
  do their damage.
- R8 generalises the argument beyond this model's two losses: late-layer units are
  specific to *each output*, so a criterion has to carry a term for every output the
  compressed model must keep. This is consistent with trajectory-only selection
  collapsing LingoQA (73.2 → 37.0) and suggests VQA capability would need its own term
  rather than riding on `I_CoC`.

## Suggested manuscript changes

**1. A paragraph for Sec. 7 (or after Figure 1).** Draft, to be cut to length:

> *Why the objectives diverge.* The two depth profiles of Fig. 1 are not independent.
> Their ratio `I_traj / I_CoC` is constant over layers 0–21 (coefficient of variation 9%
> on the MLP axis, while `I_traj` itself varies 15×), constant again over layers 23–34,
> and falls 4.6× in between (Fig. 3a); a single step accounts for 93% of the ratio's
> variation over depth. The change point is layer 23 in all eleven full-loss importance
> runs available to us — disjoint calibration sets, 100 to 4,000 clips,
> out-of-distribution clips, and a trajectory target sampled from the model itself — so it
> is a property of the network rather than of the calibration data, and applying it to
> either profile places the other's peak within three layers. Three independent
> measurements place a regime change at the same depth (Fig. 3b): the FM loss's
> sensitivity to the cache it reads lies 81% at or below layer 22; the excess kurtosis of
> a Jacobian-lens readout, which involves neither loss, rises from 6.4 to 21.0 as the
> residual stream becomes token-peaked; and attention knockouts find that edges into
> vision tokens stop mattering after layer 17 (every cell within ±3%), whereas the
> text←vision edges of layers 18–35 affect the rationale only. Below the boundary the VLM
> builds the scene representation both outputs consume, and the two scores are
> interchangeable: CoC importance measured on one half of the calibration clips predicts
> trajectory importance on the other half as well as trajectory importance itself does
> (ρ = 0.60 vs. 0.59, MLP, layers 6–19). Above it the VLM writes the rationale, which the
> expert — reading mostly vision-token cache entries — barely uses. The disagreement
> there is not estimation noise: across disjoint halves of the calibration clips each
> objective reproduces its own head ranking at ρ = 0.82–0.90 in both depth bands, while
> agreement between objectives falls from 0.85 to 0.37 once that ceiling is accounted for
> (Fig. 3c).

**2. A three-panel Figure 3.** (a) `fig3_ratio_step`, (b) `fig3_boundary`,
(c) `fig3_noise_ceiling_q_head`. Draft caption:

> **Figure 3. One shared depth profile and one switch.** (a) Ratio of the two importance
> scores per layer (raw layer sums, log axis; dotted: two-level fit). (b) Three signals,
> min–max normalised: the ratio in (a), the FM loss's sensitivity to the VLM cache tensors,
> and the excess kurtosis of a Jacobian-lens readout of the residual stream; shading marks
> layers 21–24. (c) Rank agreement across disjoint halves of the calibration clips: each
> objective against itself (the attainable ceiling) and against the other objective. Layer
> indices are zero-based; layer 35 is omitted because the trajectory score is zero there.

Supplementary: `fig3_ratio_step_draws`, `fig3_noise_ceiling_mlp`, the two partition
panels, the two pairs panels, `fig3_text_write_q_head`.

**3. A correction to how Figure 1c is described.** Any sentence that reads the MLP curve's
fall to ~0.3 as objective disagreement alone should be softened: about a third of that
fall is the ceiling dropping (R4), and 0.34 is half of what is attainable there, not a
third of it. Either cite the corrected values (Q 0.85 → 0.37, MLP 0.92 → 0.54) or put
the ceiling in the supplementary and point to it. The Q-head curve needs no change — its
ceiling is flat.

**4. Layer bands.** C2 and Sec. 7 give "action: L0–17; language: L18–35", which is the
9-layer resolution of the knockout grid. The importance ratio localises the switch to
layers 22–23 at single-layer resolution, and it is compatible with the knockouts (the
L18–26 band contains it). Worth saying once so the two numbers do not look inconsistent.

## Caveats

- The ratio's *levels* are not interpretable (different loss units); only its shape is.
- The ceiling correction applies the Pearson attenuation formula to Spearman values and
  uses half-sample reliabilities. Two routes agreeing (split halves, disjoint draws) is
  the reason to trust it, not the formula.
- The lens kurtosis is measured at text and CoC positions by construction. It describes
  the text-position stream; nothing here measures the vision-position stream in late
  layers.
- "No edge into a vision position matters after layer 17" is about attention edges. MLP
  updates at vision positions in late layers were never knocked out.
- The token-support explanation (third bullet of the analysis) is inferred, not measured.
  If the plan's P2 fails, the late-layer split is directional — same tokens, different
  subspaces — and the paragraph above should drop its last-but-one sentence's "reading
  mostly vision-token cache entries" as the stated reason.
- R7's contrast exists on the Q-head axis only, and R8's MLP column is dominated by
  attenuation. The mechanism claim is best supported for heads.
- Everything is one model. Whether the switch sits at two-thirds depth in other
  VLM-plus-action-expert models is unknown.
