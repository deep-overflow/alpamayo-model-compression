# Why do `I_traj` and `I_CoC` differ? — follow-up analysis to Figure 1

Date: 2026-09-20. Branch: `worktree-why-importance-differs`.
Script: `experiments/paper/fig_why_differs.py` → `figures/fig3_*`, `figures/fig4_*` and
`figures/fig3_why_differs_stats.json` (every number below that is not attributed to an
older run comes from that file).

Two passes on one day. R1–R8 use stored runs only. R9–R13 are a new GPU measurement
(`plans/2026-09-20_gradient-anatomy.md`, approved and run the same day) that tested the
explanation R1–R8 pointed to. **It confirmed the decisive prediction and refuted three of
the others, so the Analysis section below replaces the one written before the run**; what
changed is listed at its top.

A third pass on 2026-09-21 (branch `worktree-importance-causal-validation`,
`plans/2026-09-21_importance-causal-validation.md`, approved by the user; panels
`figures/fig5_*`, numbers in `figures/fig5_causal_validation_stats.json`) asked whether the
account describes the network's *function*, and whether it makes the paper's case for the
dual criterion: R14–R19. About half of its pre-registered gates failed; the plan's Outcome
table lists them, and the Analysis section has a second part written after them.

## Purpose

Figure 1 (`fig1_depth_mlp`, `fig1_depth_q_head`, `fig1_rank_agreement`) establishes
*that* the two objectives weight the VLM differently: `I_traj` peaks at layer 18–19 and
`I_CoC` at 24–27, and within-layer rank agreement falls after layer ~22. The manuscript
builds its mechanism claim (C2) on that observation but does not say *why* the profiles
part where they do. This note asks three questions a reviewer will ask:

1. Are these two independent depth profiles, or one phenomenon seen twice?
2. Is the late-layer rank disagreement real, or is it what unreliable scores look like?
3. What changes in the network at the depth where they part?

R1–R8 answer them from stored runs. They cannot say *where* each loss's gradient lands,
because the shipped score sums over token positions before taking the absolute value, so
a fourth question needed a new measurement:

4. On which tokens, and through which cache layers, does each loss see a unit? (R9–R13)

## Setup

| item | value |
|---|---|
| model | dense `nvidia/Alpamayo-1.5-10B`, VLM tower: 36 layers, 32 Q heads, 12,288 MLP channels |
| scores | gate-Taylor `E_c|dL/dg|`, exactly as in Figure 1 (`paper/2026-09-09_figure-reproducibility.md`) |
| main run | `outputs/importance_v2` (`calib_100`), aggregate and per-clip arrays |
| other runs | `importance_rd100_a/b/c`, `se100_a`, `su100_a`, `val100`, `ood`, `nt500`, `st4000`, `selftraj_v1`, `dim0`, `dim1` (robustness); `importance_vqa` (VQA-NLL and CoC-NLL on LingoQA images); `jlens_v2` (Jacobian lens) |
| causal evidence cited | `outputs/pathway_x_v1`, `outputs/pathway_e_v1` (attention knockouts, n=50), `outputs/cacheuse_v1` (expert cache-read knockouts, n=100) — all already in the draft's Sec. 7 |
| new runs (R9–R13) | `outputs/gradanat_v1` (token-type and cache-port split of the shipped gate gradients, `calib_100`, 100 clips, one RTX 5880 Ada, 12 s/clip, peak 42.8 GB), `outputs/gradanat_verify` (3 clips, integrity), `outputs/portmap_v1` (single-cache-layer ports, 100 clips in 3 shards, 30 s/clip). Same protocol as `importance_v2`: own-rollout CoC teacher-forced, 10-step FM loss against the GT action, clip seeds `sha256("42:clip_id")`. They reproduce `importance_v2_ada`'s within-layer ranking at ρ ≥ 0.9993 in every layer |
| third pass (R14–R19) | `outputs/gradanat_{dual,traj,coc}_u40` (the anatomy of the three shipped arms, teacher-forced on the dense text; masks from `slim_to_mask.py`), `gradanat_probes_v1` (random-readout probes), `tokabl_sets_v1` → `tokabl_v1_s{0..3}` (53 unit-set configs × 100 held-out `indist_500` clips), `armheld_v1_s*` and `armmix_v1_s*` (whole, band-restricted and crossed arm masks on the same 100 clips), `vvdepth_s*` and `vision_census_v1_s*` (nested vision–vision knockouts and the attention census on the 50 `val` clips and seeds of `pathway_e_v1`), `failure_by_situation_v1` (stored open-loop records of the three arms, 2,533 clips). Four RTX 5880 Ada. Every loss readout of R15–R18 is teacher-forced on the **dense** model's rollout, so arms and sets see the same tokens, positions and noise |
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

The unplotted remainder (36–82% by band, smallest in MLP layers 0–5 where the shared part
dominates) is single-clip measurement noise plus the clip × objective interaction; one
measurement per clip cannot separate them.

Panels: `fig3_partition_q_head`, `fig3_partition_mlp`.

### R6. Three independent measurements put a regime change at the same depth

| signal | what it uses | before | after | where |
|---|---|---|---|---|
| importance ratio | both losses' gradients | 12.7× | 2.8× | change point 23 |
| FM sensitivity to the cache tensors (`traj_kv_k + traj_kv_v`) | trajectory loss, expert side | layers 8–22 mean | 3.6× lower over 25–35 | 81% of the total lies at or below layer 22 |
| excess kurtosis of the Jacobian-lens readout over random vocabulary | **no loss at all** | 6.4 (layers 14–20) | 21.0 (layers 24–34) | two-level change point 22, R² 0.91 |

(Read the second row with R12. It is the loss's sensitivity to the cache tensors
*themselves*, `|k·dk| + |v·dv|`; the part of the FM gradient that upstream units can move
arrives mostly through later cache layers. Both are measured, and they answer different
questions.)

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

### R9. Where each loss sees a unit: both scores hand over from vision tokens to text tokens

A unit's gate gradient is a sum over token positions, `dL/dg_u = Σ_p x_u(p)·dL/dx_u(p)`.
`TypedUnitGates` keeps one gate per (token type, unit), which splits that sum exactly by
the type of position `p` — vision (2,880 tokens), ego history (48), prompt text (157),
generated CoC (16.4 on average), sink (1). For a Q head, `p` is the *query* position.
Integrity: the typed gradients sum to the shipped single-gate gradient to 6.0e-06, and the
run reproduces `importance_v2_ada`'s within-layer ranking at ρ ≥ 0.9993 in every layer.

Shares below are **additive**: `Σ_u G_type,u · sign(G_full,u)`, which over token types
sums to the layer's shipped score exactly. (Shares of `E|G_type|` do not add up; they tell
the same story and are in `outputs/gradanat_v1/summary.txt`.)

| share of the layer's score earned at… | `I_traj`, L6–17 → L27–34 | `I_CoC`, L6–17 → L27–34 |
|---|---|---|
| vision tokens, MLP | 0.90 → 0.26 | 0.75 → 0.02 |
| vision tokens, Q heads | 0.72 → 0.35 | 0.63 → 0.02 |
| text-side tokens (prompt + CoC), MLP | 0.07 → 0.66 | 0.24 → 0.98 |
| text-side tokens (prompt + CoC), Q heads | 0.18 → 0.49 | 0.34 → 0.98 |

- In the trunk both scores are earned where the unit acts on **vision** tokens. Both move
  to text-side tokens over the same layers: on the MLP axis the vision share falls below
  one half for good at layer 18 (`I_CoC`) and layer 21 (`I_traj`); on Q heads at 17 and 19.
- `I_CoC` completes the move (2% vision in late layers). `I_traj` does not: 26–35% stays
  at vision tokens and 8–16% at the ego-history tokens.
- My pre-registered prediction that the FM gradient *stays* on vision tokens (> 0.65 in
  both bands) is **wrong**: its late-layer vision share of `E|G|` is 0.28 (MLP) / 0.33 (Q).
- Per-position gradient *energy* is a different quantity and should not be confused with
  this: `‖dL/dh(p)‖²` sits 97% (layers 6–17) to 99.5% (27–34) at text-side positions for
  CE, and 76% to 97% off the vision positions for FM. The trunk's importance is nevertheless earned at vision
  tokens, because there are 2,880 of them and a unit's contributions add up across them.

Panels: `fig4_token_handoff_mlp`, `fig4_token_handoff_q_head`.

### R10. At matched tokens the two losses agree — the pre-registered test of the explanation

P2 of the plan: restrict *both* scores to the unit's action at vision tokens and ask
whether late-layer Q-head rankings still disagree. ≥ 0.70 supports a token-mixture
account, ≤ 0.45 refutes it. All values are corrected for the restricted score's own
split-half ceiling (50 random disjoint halves; ceilings 0.78–0.94, so nothing here is a
noise artefact), layers 0–21 → 22–34.

| Q heads: `I_traj` vs `I_CoC` | early | late |
|---|---:|---:|
| pooled — the shipped scores | 0.85 | **0.38** |
| both at vision tokens (**the decisive value**) | 0.96 | **0.79** |
| both at prompt-text tokens | 0.86 | 0.81 |
| both at text-side tokens (prompt + CoC) | 0.72 | 0.53 |
| both at the generated CoC tokens only | 0.47 | 0.41 |
| `I_traj` at vision tokens vs `I_CoC` at text-side tokens | 0.32 | 0.24 |
| *within one loss:* `I_traj` at vision vs `I_traj` at text-side | 0.39 | 0.44 |
| *within one loss:* `I_CoC` at vision vs `I_CoC` at text-side | 0.35 | 0.48 |

- **Verdict: supported (0.786 ≥ 0.70).** The pooled value reproduces R4's 0.85 → 0.37
  from an independent run (0.85 → 0.38).
- Read as a 2 × 2: *which tokens* a head is scored on matters more than *which loss*
  scores it. Same tokens, different loss: 0.79–0.96 (vision), 0.81–0.86 (prompt). Same
  loss, different tokens: 0.35–0.48, at every depth.
- The exception is the generated CoC tokens, where the two losses rank heads differently
  at every depth (0.41–0.47) — including the trunk, where it does not show in the pooled
  score because those ~16 tokens earn 5% (`I_traj`) to 18% (`I_CoC`) of a layer's score
  there. In late layers they earn 32% and 62% (Q heads).
- MLP channels, descriptive (ceilings 0.45–0.84): pooled 0.92 → 0.54, both at vision
  tokens 0.98 → 0.72, both at text-side 0.85 → 0.59, CoC tokens only 0.78 → 0.37.

Panel: `fig4_same_token_q_head`.

### R11. The ratio step of R1 lives in the text tokens; at vision tokens there is none

R1's ratio, recomputed with both scores restricted to one token type (raw layer sums of
`E|G|`, geometric means over layers 0–21 and 23–34):

| `I_traj / I_CoC` | L0–21 | L23–34 | step | R² of the best single step |
|---|---:|---:|---:|---:|
| MLP, pooled | 12.9× | 2.8× | 4.6× | 0.93 |
| MLP, at vision tokens | 15.0× | 12.4× | **1.2×** | 0.21 |
| MLP, at text-side tokens | 6.3× | 2.1× | 3.0× | 0.82 |
| Q heads, pooled | 14.3× | 3.1× | 4.6× | 0.86 |
| Q heads, at vision tokens | 16.7× | 18.2× | **0.9×** | 0.25 |
| Q heads, at text-side tokens | 9.0× | 1.9× | 4.8× | 0.74 |

So R1's two plateaus are, to a first approximation, the vision-token ratio (~15×) and the
text-token ratio (~2×), and the step is the hand-over of R9 plus a step inside the
text-token component. What moves, layers 18–21 → 24–29 (MLP):

| | at vision tokens | at text-side tokens |
|---|---:|---:|
| `I_traj` | ×0.23 | ×0.77 |
| `I_CoC` | ×0.39 | **×2.84** |

`I_traj` falls because its vision-token component collapses; `I_CoC` rises because its
text-token component nearly triples. On Q heads the CoC text component does not rise
(×0.98) and both `I_traj` components fall (×0.12, ×0.24) — R3's axis difference, now located.

Panels: `fig4_ratio_by_token_mlp`, `fig4_ratio_by_token_q_head`.

### R12. `I_traj` rebuilt from cache ports: the step is the main port closing

The FM loss reaches a VLM unit only through the KV cache, and a unit in layer `l` can only
change cache layers `m > l` (layer `m`'s keys and values are computed from the stream
before layer `m` writes to it). Seeding the VLM backward with one cache layer at a time
gives the part of each unit's gradient that arrives through that layer. With signed shares
`S(m, l) = E_c Σ_u G_m,u · sign(G_full,u)` the decomposition is additive:
`I_traj(l) = Σ_m S(m, l)`. Measured: the rebuilt profile equals the shipped one to within
0.05% (MLP) / 0.1% (Q) at every layer, and every cell with `m ≤ l` is exactly 0.

| where a unit's `I_traj` arrives from | MLP | Q heads |
|---|---:|---:|
| units in layers 16–21: through cache layers 22–24 (three layers) | **43.1%** | **43.1%** |
| … through cache layers 25–29 / 30–35 / ≤ 21 | 22.8 / 20.2 / 13.9% | 21.2 / 19.5 / 16.2% |
| all VLM units: through cache layer 22 alone | 19.1% | 22.0% |
| … through cache layers 21–23 | 34.7% | 40.9% |
| … through cache layers ≤ 20 (all twenty-one of them) | 3.7% | 7.2% |
| … through cache layers ≥ 22 | 88.8% | 84.4% |
| trunk units (layers 6–21): through cache layers ≥ 22 | 84.0% | 81.0% |

- **PM1 (pre-registered before the port-map data): pass.** The profile over cache layers
  peaks at layer 22 on both axes; it is a port, not an artefact of where bands were cut.
  The other peaks are layers 26, 30–31 and 33.
- **PM2: pass.** The fall `I_traj(21) − I_traj(23)` splits exactly into [share through
  cache layers 22–23, which a layer-23 unit cannot reach] + [change through cache layers
  ≥ 24]. The first term is **89.5%** of the fall on the MLP axis and **72.6%** on Q heads.
- **PM3: pass.** Ports ≤ 15 are read through vision cache entries (78% MLP / 89% Q of the
  signed share); ports ≥ 22 mostly through non-vision entries (39% / 36% vision). Ports 22
  and 26 are the exceptions (65% and 82% vision on MLP), port 23 is 13% vision.
- **P3 of the original plan: refuted.** I predicted mid-layer units would receive < 20% of
  their FM mass through cache layers 25–35; it is 42–43%. Late cache layers are real ports.
  The step is a port effect, but not the one I described: the expert's ports do not end
  near layer 24 — the densest one sits at 21–23 and a unit above it cannot write into it.
- **P4: fails.** Only 35–38% of FM mass enters through vision cache entries; the ~16 CoC
  tokens alone carry 29–32% (per token 12–15× the prompt text and ~140–160× the vision
  tokens). The expert's *direct* read `|k·dk| + |v·dv|` is 73% at vision entries, matching
  its attention mass; the two are different quantities.

This is a statement about where the **first-order score** comes from, not about what the
expert needs. The knockout map (`cacheuse_v1`) puts the expert's causal reliance on cache
layers 0–15 (81%); the score's ports are 21–35. A large perturbation of a whole cache layer
and an infinitesimal one through a single unit are different questions, and the paper's
score is the second.

Panels: `fig4_itraj_ports_mlp`, `fig4_itraj_ports_q_head`, `fig4_port_profile`.

### R13. Integrity, and the fallback measurement

- Typed gradients summed over token types vs the shipped single gate: max relative error
  6.0e-06 over 3 clips × 13 backwards (threshold 1e-3).
- Seed splits (six bands, five position types, 72 single-layer ports) summed vs one
  full-seed backward: median layer 3.2e-03 to 3.8e-03, worst layer of a clip 1.3e-02 to
  1.5e-02 (median over clips), within-layer rank ρ ≥ 0.961 for the band and position
  splits. **This misses the
  pre-registered 1e-3.** It is the rounding floor of a deep bf16 backward — linear in its
  seed only up to rounding — and the same floor `analyze_stepvlm` measured for the per-step
  split (~5e-3 median). It touches R12 only; R9–R11 are read off a single full-seed
  backward, the operation the shipped score itself performs.
- Fallback (H-dir), measured in the same pass: the cosine between the two losses'
  residual-stream gradients at matched positions is ≈ 0 at every depth and token type
  (+0.019 to −0.001). The two gradients are orthogonal even where the unit rankings agree
  at 0.96, so that agreement is not the two losses "pointing the same way"; it comes from
  the units' side of the inner product.

### R14. What each shipped criterion kept, in token currency (no GPU)

Plan: `plans/2026-09-21_importance-causal-validation.md`, Stage 0; the prediction was pushed
(commit `1b4f981`) before the table was computed. The dense token-resolved scores of R9 are
crossed with the kept sets of the three shipped arms (`dual_u40_v2`, `traj_u40_v2`,
`coc_u40_v2`: 19 of 32 heads and 7,390 of 12,288 channels in every layer, expert untouched).
Entries are the share of the dense score's mass, layers 22–34, that sits on kept units; a
random kept set would hold 0.59 (heads) / 0.60 (channels).

| | | `dual` | `traj` | `coc` |
|---|---|---|---|---|
| Q heads | `I_traj` at vision + history tokens | 0.787 | 0.828 | **0.678** |
| | `I_traj` at text-side tokens | 0.835 | 0.842 | 0.777 |
| | `I_CoC` at CoC tokens | 0.763 | **0.650** | 0.827 |
| | `I_CoC` at prompt tokens | 0.852 | 0.781 | 0.865 |
| MLP channels | `I_traj` at vision + history tokens | 0.697 | 0.712 | **0.651** |
| | `I_CoC` at CoC tokens | 0.715 | **0.669** | 0.729 |

- Ordering as predicted on both axes: the CoC-only criterion keeps the least trajectory
  importance earned at vision and history tokens, the trajectory-only criterion the least
  CoC importance earned at the CoC tokens.
- "`dual` within 5 pp of the better single criterion": 1.6 pp and 1.4 pp on MLP channels,
  4.1 pp on the trajectory side of Q heads, **6.4 pp on the CoC side of Q heads — missed.**
- What the CoC-only criterion loses is late and token-specific: against `traj` it gives up
  15 pp of `I_traj` at vision/history tokens in layers 22–34 but 3 pp in layers 6–21, and
  10 pp more at vision/history tokens than at text-side tokens. The late kept sets of the
  two single criteria overlap 0.68 (Q) / 0.69 (MLP): about 6 of 19 kept heads per layer.

### R15. The pruned models: what survives, and the three arms on the same text

`run_gradient_anatomy.py --mask`, 100 `calib_100` clips per arm. Every arm is teacher-forced
on the **dense** model's rollout (same tokens, positions and noise seeds; all 100 texts
identical across runs, and the dense FM loss reproduces R9's run exactly). A mask on the
full model *is* the shipped checkpoint here: the parameters the mask removes are the
2,657,452,032 the build removed.

Structure (gate A1, judged on `dual`; all statistics among kept units):

| | dense | `dual` | `traj` | `coc` |
|---|---|---|---|---|
| ratio change point, Q / MLP | 23 / 23 | **23 / 23** | 7 / 22 | 21 / 22 |
| step size, Q / MLP | 4.5× / 4.5× | 3.3× / 4.0× | — / 2.3× | — / 2.3× |
| MLP hand-over layer, `I_CoC` / `I_traj` | 18 / 21 | **18 / 21** | 10 / 20 | 10 / 19 |
| agreement at vision tokens, late Q heads (corrected) | 0.786 | **0.804** | 0.746 | 0.888 |
| pooled agreement, late Q heads (corrected) | 0.382 | 0.257 | 0.425 | 0.553 |
| layer-profile correlation with dense, MLP (`I_traj` / `I_CoC`) | — | 0.977 / 0.990 | 0.849 / 0.702 | 0.939 / 0.804 |
| same, Q heads | — | **0.919 / 0.664** | 0.324 / −0.130 | 0.699 / 0.327 |

- Three of A1's four clauses hold for `dual`; **the fourth (profile correlation ≥ 0.98)
  fails on the Q axis.** The cause is at the bottom of the network, not at the step: the
  surviving heads of layers 0–1 carry 3.9× (`I_traj`) and 8.3× (`I_CoC`) their dense
  importance while layers 6–34 carry 1.1× and 1.9×. Without the first two layers the Q
  correlations are 0.985 and 0.925 (not a gate; reported because it says where the change is).
- The single-criterion arms lose more of the structure: on the Q axis their ratio no longer
  has a step (R² 0.34 and 0.15 against 0.79 for `dual`), the early-layer load is 14× / 66×
  (`traj`) and 11× / 25× (`coc`), and their `I_CoC` hands over to text tokens at layer 6–10
  instead of 17–18.
- Cache bands (descriptive): the share of the FM gradient entering through cache layers
  22–24 is 0.316 (dense) → 0.272 (`dual`) → 0.234 (`traj`) → 0.167 (`coc`) on Q heads, and
  the share through layers 30–35 rises 0.262 → 0.292 → 0.424 → 0.543.

Function on matched text (gate A2, passed on all four tests; paired one-sided Wilcoxon over
the 100 clips):

| | FM loss | CoC NLL |
|---|---|---|
| dense | 0.2553 | 0.1593 |
| `dual` | 0.2728 | 0.2182 |
| `traj` | 0.2846 | **0.5022** |
| `coc` | **0.2899** | 0.2714 |

- FM loss: `coc` > `dual` (p = 4.6e-06), `coc` > `traj` (p = 1.0e-03). NLL: `traj` > `dual`
  (p = 1.1e-17), `traj` > `coc` (p = 5.0e-12).
- Not pre-registered: `dual` is also below each specialist on the specialist's own channel —
  FM loss `traj` > `dual` (median paired difference +0.0038, p = 2.8e-03), NLL `coc` >
  `dual` (+0.0163, p = 7.8e-05).
- These clips are the calibration clips, so each single criterion is in-sample for its own
  loss. R17 repeats the comparison on held-out clips and splits it by depth.

Panels: `fig5_matched_fm`, `fig5_matched_nll`; run outputs `outputs/gradanat_{dual,traj,coc}_u40`,
analysis `outputs/gradanat_pruned_v1`.

### R16. Random readouts through the same two doors reproduce the step and the late split

`run_gradient_anatomy.py --probes`, dense model, 100 `calib_100` clips. Each loss is
replaced by a random linear readout with no language or driving content: **head door**
`Σ_p ⟨r_p, h_final(p)⟩` over the positions the CE loss reads; **expert door**
`Σ_s ⟨R_s, v_θ(x_s, t_s)⟩` over the ten FM steps on the same `x_s`; **bare cache door**
`⟨R, [K_m; V_m]⟩` on every cache layer and position alike, no expert involved.

| | Q heads | MLP channels |
|---|---|---|
| layer profile, expert-door probe vs `I_traj` (Pearson) | 0.980 | 0.964 |
| layer profile, head-door probe vs `I_CoC` | 0.871 | 0.936 |
| change point of expert-door / head-door | **23** (5.4×, R² 0.87) | **23** (5.5×, R² 0.91) |
| for reference, `I_traj` / `I_CoC` | 23 (4.5×, R² 0.86) | 23 (4.5×, R² 0.93) |
| `I_traj` / expert-door probe, layers 22–24 and 25–34 relative to 0–15 | 0.96, 0.82 | 0.95, 0.74 |
| `I_CoC` / head-door probe, same | 0.81, 0.63 | 0.73, 0.57 |

- **D1 passed**: what any demand on the expert's output does to the VLM has `I_traj`'s
  depth profile, and the two doors' content-free probes have the step of R1 at the same
  layer. **D2 failed on the Q axis** (0.871 against 0.90; MLP 0.936).
- **D3 passed**, and by more than asked: within a layer the expert-door probe ranks Q heads
  like `I_traj` at 0.990 (layers 6–21) and **0.992 (layers 22–34)** after ceiling
  correction; the head-door probe and `I_CoC`, 0.965 and 0.880. The two probes against each
  other: 0.786 in the trunk and **0.353 late** — R4's 0.85 → 0.37 for the real losses,
  reproduced with no objective on either side.
- The probes hand over between tokens like the scores they stand in for: share of `E|G|`
  at vision tokens, layers 6–17 → 27–34, expert door 0.750 → 0.277 (`I_traj` 0.754 → 0.276,
  MLP), head door 0.520 → 0.049 (`I_CoC` 0.616 → 0.062).
- **D4 failed** on both clauses as written. (i) Cache side, first-order read by cache
  layer, FM loss vs expert-door probe over cache layers 16–35: Pearson 0.857 against 0.9
  (top layers 22, 21, 26, 23, 31 vs 21, 35, 22, 26, 31; the unit-side port share of R12
  follows the same cache-side profile at 0.86–0.88). (ii) I predicted the bare cache door
  would show no step. It does not show a *step*, but it is not flat either: relative to the
  head door it declines as a ramp from about layer 10 (MLP: 0.70 of its trunk level over
  layers 16–21, 0.33 over 22–24, 0.16 after; a straight line fits it almost as well as a
  step, R² 0.72 vs 0.85, where for the real ratio it is 0.66 vs 0.93). What the expert adds
  to the bare door is a boost of layers 16–21 only — 1.9× (Q) and 1.3× (MLP) — which is the
  signature of its demand being concentrated on cache layers 21–23 (R12).
- This does not contradict R8. R8 changed the data, the prompt and the read positions
  together with the text; here the forward pass and the read positions are fixed and only
  the content of the readout changes. What R8 called output-specific is specific to where
  and in what context the network is read, not to what is asked for there.

Panels: `fig5_probe_ratio_mlp`, `fig5_probe_ratio_q_head`, `fig5_probe_rank_q_head`; run
`outputs/gradanat_probes_v1`.

### R17. Held-out ablation: late layers carry the language channel and not the action channel

**Token-targeted unit sets (part B).** `make_token_sets.py` → `run_token_ablation.py`, dense
model, 100 held-out `indist_500` clips (disjoint from `calib_100`, where every set was
chosen), paired with the dense model on the same clip, dense text and noise: FM loss 0.2425,
CoC NLL 0.1708, minADE@8 0.837 unmasked. Per layer 6 of 32 heads or 2,304 of 12,288 channels
(19%) are removed, in layers 22–34 ("late") or, as the control, 6–17 ("trunk"). T sets are the
units one score ranks far above the other on the trajectory side, C sets the same on the CoC
side, from token-resolved ("tok") or pooled ("pool") scores; five random sets per family.

What a band is worth, whatever the rule (mean over its five random sets, share of dense):

| | FM loss | CoC NLL |
|---|---|---|
| Q heads, layers 22–34 | **−0.0%** | +6.1% |
| Q heads, layers 6–17 | +3.9% | +3.8% |
| MLP channels, layers 22–34 | **−0.6%** | +8.3% |
| MLP channels, layers 6–17 | +0.7% | +1.0% |

Late layers, mean damage as a share of dense (median in brackets):

| set | FM loss, Q | NLL, Q | FM loss, MLP | NLL, MLP |
|---|---|---|---|---|
| T-tok | +0.2% | +1.6% (+1.6%) | −0.4% | +1.5% (−0.0%) |
| C-tok | −0.2% | **+14.4%** (+7.3%) | −0.1% | +10.3% (+3.7%) |
| T-pool | +0.1% | +13.5% (+1.7%) | −0.9% | +8.3% (+0.4%) |
| C-pool | −0.2% | **+15.0%** (+8.1%) | −0.3% | +8.4% (+3.1%) |
| random, range of five | −0.6 to +0.4% | +4.2 to +8.4% | −0.9 to −0.2% | +5.0 to +11.0% |

- **B1 failed everywhere.** No late-layer set moves the FM loss: every 95% interval contains
  zero or sits within 1% of dense, the T sets are not above the C sets (p = 0.23, 0.13 on Q;
  0.87, 0.92 on MLP) and above none of the random sets. The units `I_traj` singles out in
  layers 22–34 do not carry the action at this dose. The first-order score says the same
  thing in a different way: over the 17 late configs the summed `I_traj` of the removed
  units does not order the measured FM damage (Spearman +0.09 on Q, −0.80 on MLP, all
  damages within ±1%), while the summed `I_CoC` orders the NLL damage at +0.65 / +0.71.
- **B2 passed on Q heads, half-passed on MLP channels.** Removing the heads `I_CoC` singles
  out raises the NLL more than removing the ones `I_traj` singles out (tok p = 1.6e-06, pool
  p = 6.7e-04) and more than every random set (5/5 both): 2.4× the random sets' mean damage,
  against one quarter of it for T-tok. On MLP the cross-over holds for the
  token-resolved sets (p = 3.1e-04; pooled p = 0.038) but the C sets are not above the random
  sets (0/5): there `I_CoC` tells which channels are safe to drop, not which are critical.
- **B3, the trunk control, failed for the token-resolved Q sets and was not evaluable on
  MLP.** In layers 6–17 the T and C sets *do* part, on the action side: FM loss +2.8% (T-tok)
  and +3.5% (T-pool) against +0.6% and +1.3% (C) — paired p = 3.9e-04 and 2.0e-03 — with no
  difference on the NLL. Both stay below the random sets (+1.8 to +6.0%): a rank-difference
  rule picks specialised heads, not the important ones. MLP trunk damages are too small for a
  ratio (a mean ≤ 0).
- **B4 passed** trivially on the T side (no FM effect to compare) and on the C side (C-tok and
  C-pool are indistinguishable): token resolution explains the scores (R10) but selects no
  better than the pooled scores.
- **B5, what the dual criterion saves, passed on one of four.** The heads the trajectory-only
  criterion dropped and `dual` kept (S-traj, 3.5 per late layer): NLL +8.6% against +1.3 to
  +3.2% for three size-matched random sets (3/3), FM loss unchanged — passed. The heads the
  CoC-only criterion dropped and `dual` kept (S-coc): FM loss +0.6% [−0.0000, +0.0027], above
  one random set of three — failed; both MLP sets failed (damages within the random range).
- **`I_CoC` is calibrated across depth; `I_traj` is not** (descriptive). Measured damage per
  unit of removed score mass, on the random sets — an unbiased sample of a band's units. Q
  heads: the NLL costs 0.113 per unit of `I_CoC` in the trunk and 0.136 late; the FM loss
  costs 0.0111 per unit of `I_traj` in the trunk and −0.0000 late. (MLP: 0.0041 and 0.0099;
  0.0003 and −0.0003 — the trunk dose moves neither loss by more than 1% there.) It is not a
  matter of sign: the gate gradient's sign agrees across clips at only 0.31 / 0.28
  (FM, trunk / late) and 0.16 / 0.20 (CE), and the signed first-order prediction `−Σ E_c[G_u]`
  orders nothing (ρ = +0.19, −0.24 over the 26 Q configs). What a finite removal costs is a
  second-order matter, which `E|G|` tracks for the CE loss at every depth and for the FM
  loss only below the step.
- minADE@8 gates nothing, as planned: every late-set interval contains zero; the trunk T sets
  average +0.16 to +0.18 m with intervals that reach zero ([−0.004, +0.482]).

**The shipped arms on the same held-out clips, whole and by depth band (A2-heldout,
A2-bands).** `run_token_ablation.py --arm-masks`: the three arms' masks, and each mask applied
in layers 0–21 only ("trunk") or 22–35 only ("late") with the other band dense; same 100
clips, dense text, same seeds. Mean change against dense [95% interval]:

| mask | FM loss | CoC NLL | minADE@8 (m) |
|---|---|---|---|
| `dual`, all layers | +5.5% | +32.6% | +0.01 [−0.10, +0.11] |
| `traj`, all layers | +6.6% | **+240.6%** | +0.06 [−0.04, +0.16] |
| `coc`, all layers | **+15.8%** | +67.2% | **+0.56** [+0.37, +0.76] |
| `dual`, late only | +0.1% | +10.1% | −0.01 |
| `traj`, late only | −0.1% | **+65.5%** | −0.01 |
| `coc`, late only | +0.1% | +6.3% | +0.00 |
| `dual`, trunk only | +4.8% | +25.2% | +0.06 [−0.05, +0.15] |
| `traj`, trunk only | **+26.1%** | +51.2% | **+2.64** [+1.81, +3.54] |
| `coc`, trunk only | +17.0% | +28.3% | +0.46 [+0.29, +0.64] |

- **A2's ordering holds out of sample.** FM loss `coc` > `dual` (p = 8.7e-08) and > `traj`
  (1.9e-06); NLL `traj` > `dual` (5.1e-18) and > `coc` (7.0e-14); NLL `coc` > `dual` (3.1e-05).
  The pre-registered gate also asked for `dual` below `traj` on the FM loss, and that is
  **not** significant out of sample (median difference +0.0016, p = 0.33), so the gate as
  written failed; and the NLL gap between `traj` and `coc` is larger out of sample (+0.296)
  than in sample (+0.231), not smaller as predicted. In-sample selection is not what made
  R15's language ordering.
- **(i) passed: the late layers carry language only.** Applied in layers 22–35 alone, no
  arm's mask moves the FM loss (+0.13%, −0.07%, +0.08%) or minADE, while the trajectory-only
  mask alone raises the NLL by 65.5% against 6.3% for the CoC-only mask (p = 2.9e-17) and
  10.1% for `dual`. R17's unit sets and whole arms agree.
- **(ii) failed, in the opposite direction.** Applied in layers 0–21 alone, it is the
  *trajectory-only* mask that costs the action most: FM loss +26.1% against +17.0% (CoC-only)
  and +4.8% (`dual`); minADE +2.64 m against +0.46 m and +0.06 m. `dual`'s trunk selection
  beats both single criteria on the FM loss (p = 6.7e-13 and 3.1e-15) and is no worse on the
  NLL (+25% against +28% and +51%).
- **(iii) failed: depth bands interact, except under `dual`.** Trunk-only plus late-only
  damage over the full mask's damage: `dual` 0.90 (FM) and 1.08 (NLL); `coc` 1.08 and 0.51;
  `traj` **3.98** and 0.49. The trajectory-only arm's late-layer pruning, which by itself
  changes nothing on the action side, removes three quarters of the FM damage its trunk
  pruning causes (+26.1% → +6.6%; minADE +2.64 m → +0.06 m). The masks were checked (trunk
  ∧ late = full for every arm and axis).
- **A2-mix: the repair is about late capacity, not about which late units** (all three
  predictions failed; both runs reproduce the dense model and the `traj` trunk mask per clip
  exactly). The `traj` trunk mask with a *random* late mask of the same size: FM loss +8.8%,
  minADE +0.42 m — most of the way from +26.1% / +2.64 m to the full arm's +6.6% / +0.06 m
  (above the full mask at p = 0.016 only). With `dual`'s late mask +7.7% / +0.26 m; with
  `coc`'s, which keeps the language-side late units, the least repair, +13.5% / +0.98 m
  (I predicted none, ≥ +20%). The reverse cross does nothing for the `coc` trunk (+18.1%
  with `traj`'s late mask against +17.0% alone; a random late mask: +15.1%, minADE +0.46 →
  +0.26 m). Reading: a trunk chosen by `I_traj` alone damages what it feeds the late layers;
  the intact late layers — a language network the action does not otherwise need — turn that
  into a late cache the expert reads, and thinning them in any way damps it. The
  trajectory-only arm is a usable driver because it also prunes its late layers, at the
  price of the language (NLL +241%); `dual`'s trunk needs no such repair (+4.8%, additive).

Panels: `fig5_dissociation_q_head`, `fig5_dissociation_mlp`, `fig5_dual_saves`, `fig5_damage_by_band_fm`,
`fig5_damage_by_band_nll`; runs `outputs/tokabl_sets_v1`, `tokabl_v1_s{0..3}`, `armheld_v1_s{0..3}`,
`armmix_v1_s{0..3}`; analyses `outputs/tokabl_v1`, `armheld_v1`, `armmix_v1`.

### R18. Vision–vision interaction by depth: the user's hypothesis, tested causally

The hypothesis: the action loss draws mostly on vision tokens, so action importance should be
high where vision–vision interaction is high. `run_pathway2.py --cuts` on the 50 `val` clips
and seeds of `pathway_e_v1` (K = 8): the edge "vision token ← X" is blocked in the nested
windows [l, 36), l = 0, 3, …, 33, for X = any other vision token (**VV**), same-camera earlier
frames (E1), other cameras (E2), the rest of its own image (V3); VV also in [0, l). Damage is
the paired median change over the baseline median (minADE 0.468 m, NLL 0.125), the statistic
of the stored map. Why nested: the stored 9-layer bands showed the layers make up for each
other (E1: at most +16.5% in one band, +63.9% everywhere), so single windows have no power.

Integrity: the causal-mask-only control is at +1.8e-04 NLL, and the unblocked run,
`E1` from 0 and `E2` from 0 reproduce the stored `E0_none`, `E1@all` and `E2@all` per clip
exactly (max |difference| 0.000).

minADE change when the edge is blocked from layer l to the last [95% interval]:

| l | VV | E1 (time) | E2 (cameras) | V3 (own image) | NLL, VV |
|---|---|---|---|---|---|
| 0 | +58.9% [+30, +134] | +63.9% [+6, +82] | +17.4% [−4, +28] | +46.4% [+25, +75] | +54.5% |
| 9 | +35.1% [+14, +100] | +16.4% [+5, +35] | +7.6% | +37.9% | +31.9% |
| 12 | +51.3% | +9.0% [+1, +31] | +12.0% | +30.0% [+7, +47] | +23.2% |
| 15 | +35.3% [+9, +71] | +3.1% | +6.3% | +9.9% [+1, +27] | +15.6% |
| 18 | **+27.8% [+6, +55]** | +0.1% [−6, +7] | +4.3% [−1, +8] | +2.1% [−1, +9] | +4.1% |
| 21 | +6.3% [−2, +18] | +0.1% | +0.9% | +0.2% | +0.8% |
| 24 | +6.5% [−1, +11] | −0.2% | +0.5% | +0.2% | −0.2% |
| 27 | +1.3% [−1, +4] | −0.1% | −0.1% | +0.1% | −0.2% |

VV blocked in [0, l): +0.1% (l = 6), +6.7% (9), +24.5% (12), +59.8% (18), +52.3% (24).

- **C1 passed.** What each 3-layer window adds to the VV action damage (−0.4, +2.2, +17.8,
  +14.3, +4.9, **+20.2** (layers 15–17), +11.6, +5.5, +1.7, +1.2, +0.5, −0.9%) follows
  `I_traj` at vision tokens at Spearman **+0.85** (Q heads) and +0.83 (MLP).
- **C2: the hypothesis holds up to the importance peak, and my prediction failed.** I expected
  vision–vision interaction to be finished before layers 16–20, where `I_traj` at vision
  tokens peaks, because the stored map had E1 and E2 at +1.0% and +3.0% in layers 18–26.
  Blocked one kind at a time that is still what happens (from layer 18 on: +0.1%, +4.3%,
  +2.1%). Blocked together, VV from layer 18 on costs **+27.8%**, 47% of blocking it
  everywhere (the bar for the hypothesis was 25%, my prediction below 10%). The three kinds
  of vision–vision attention substitute for each other late in the trunk, which a
  one-edge-at-a-time map cannot see.
- **C3 failed narrowly.** VV from layer 24 on still costs +6.5% [−0.9, +11.0] (the null band
  is ±5%); from layer 27 on, +1.3%. The single edges are within 1% from layer 21 on.
- **Different interactions finish at different depths.** Time (same camera, earlier frames)
  is done by layers 9–12 (+16.4% from 9, +9.0% from 12, +0.1% from 18); the own image by
  15–18; cameras never cost more than +20%; *some* vision–vision mixing is needed through
  layers 18–23. Early interaction is fully replaceable: blocking VV in layers 0–5 costs
  +0.1%, in 0–8 +6.7%; the damage arrives when the block reaches layers 9–17.
- **C4, the causal half: it is a shared substrate.** VV's NLL profile follows its action
  profile over the windows below layer 24 at +0.86, and the VV action profile follows
  `I_CoC` *at vision tokens* (+0.87 Q, +0.83 MLP) as well as it follows `I_traj` there. What
  it does not follow is the pooled `I_CoC` (+0.20 Q, −0.31 MLP; pooled `I_traj`: +0.84,
  +0.64). So the user's hypothesis is right and it is half of the answer to "why are the
  distributions different": both scores are earned, through layer ~21, where vision tokens
  are being integrated — that is the shared profile of R1 — and the pooled profiles part
  because `I_CoC` has a second, text-token component after the hand-over that no vision
  interaction explains, while `I_traj`'s late component is the first-order shadow of R17.
- **C4, the observational half, passed — and shows why attention mass is the wrong ruler.**
  `run_vision_census.py`, same 50 clips: where the 2,880 vision queries put their attention.
  Mass on tokens of *other* images (earlier frames + other cameras) is 0.25 in layers 0–5,
  0.23 in 6–11, 0.35 in 12–17, **0.64 in 18–21, 0.60 in 22–27, 0.61 in 28–35**; the sink
  takes 0.51 and 0.37 in layers 6–11 and 12–17. So cross-image attention stays at 0.60 after
  layer 22, where the knockouts find it does nothing, and over depth it is unrelated to importance at vision
  tokens (Spearman −0.10 / +0.19 for `I_traj`, −0.15 / +0.15 for `I_CoC`, Q / MLP). Within a
  layer it does pick out the important heads, and for both scores alike: ρ = +0.64
  (`I_traj`) and +0.59 (`I_CoC`) over layers 6–17 — a gap of 0.05 against the 0.1 bar —
  falling to +0.20 / +0.29 in layers 18–21 and +0.17 / +0.03 after. The hypothesis holds for
  the causal measure of interaction, not for the observed attention mass.

Panels: `fig5_vv_nested`, `fig5_vv_window_profile`, `fig5_vision_census`; runs
`outputs/vvdepth_s{0,13,26,38}`, `vision_census_v1_s{0,13,26,38}`, analysis `outputs/vvdepth_v1`.

### R19. Where the single-criterion arms fail: driving situation, free-running CoC, end tokens

Stored open-loop records only (`analyze_failure_by_situation.py`): `coc_u40_v2`,
`traj_u40_v2`, `dual_u40_v2` against `baseline_ada`, paired per clip over `indist_500` +
`test_500` + OOD (2,533 clips, K = 8, same seeds), split by the GT manoeuvre
(`eval_lib.bucket`; recomputed labels match the stored ones on every clip). The question
was the user's: do the two single criteria fail in *different* situations — CoC-only on
turns, trajectory-only on straight driving?

ΔminADE against dense, mean over clips (share of the dense model's minADE in that situation):

| | cruise (1,385) | accelerate (413) | decelerate / stop (456) | turn (279) | all |
|---|---|---|---|---|---|
| CoC-only | +0.641 (+88%) | +0.387 (+46%) | +0.551 (+68%) | +0.888 (+83%) | +0.611 (+76%) |
| trajectory-only | +0.162 (+22%) | +0.080 (+10%) | +0.325 (+40%) | +0.286 (+27%) | +0.192 (+24%) |
| dual | +0.068 (+9%) | +0.078 (+9%) | +0.119 (+15%) | +0.147 (+14%) | +0.088 (+11%) |

- **No cross-over.** The clip-wise difference `d_coc − d_traj` is positive in every
  situation (cruise +0.479 [+0.416, +0.544], accelerate +0.307 [+0.184, +0.456],
  decelerate/stop +0.226 [+0.104, +0.376], turn +0.602 [+0.383, +0.822]). Left and right
  turns do not differ for either arm (−0.237 [−0.677, +0.184] on the difference).
- **The trajectory-only arm is weakest at decelerate/stop and at turns**, not on straight
  driving: against cruise, +0.163 [+0.058, +0.266] and +0.124 [+0.002, +0.267]. The same
  two contrasts for `dual` are +0.051 [−0.047, +0.156] and +0.079 [−0.004, +0.159], and
  `dual` recovers most exactly there (OOD: decelerate/stop +0.381 → +0.102, turn
  +0.429 → +0.169). Both contrasts are significant on OOD and pooled, not on the
  in-distribution sets alone; no multiple-comparison correction.
- **The CoC-only arm fails everywhere, and part of it is its own CoC collapsing.** 12.1% of
  its rollouts are degenerate (dense 0.6%, trajectory-only 2.7%, `dual` 2.4%); those clips
  cost +2.06 m against +0.41 m for the healthy ones and carry 41% of the arm's action
  damage. Restricted to clips with a healthy CoC the pattern stays (CoC-only +61 / +25 /
  +60 / +49%, trajectory-only +17 / +7 / +38 / +21%, `dual` +10 / +8 / +10 / +14%).
- **The CoC-only criterion is not the language-safe one.** Free-running: it is the only arm
  that often fails to stop — 62 / 65 / 118 rollouts (12.4% / 13.0% / 7.7% of `indist` /
  `test` / OOD) reach the 256-token limit without the stop token, against 2 / 5 / 23
  (trajectory-only), 4 / 4 / 13 (`dual`) and 0 (dense); every one of them is classed
  degenerate. Teacher-forced on GT text (OOD): NLL +0.192 (CoC-only), +0.143
  (trajectory-only), +0.032 (`dual`). LingoQA: 30.2 (CoC-only), 37.0 (trajectory-only),
  68.8 (`dual`), 73.2 (dense). The one language readout on which CoC-only beats
  trajectory-only is the one it was selected on — the NLL of the dense model's own text
  (R15) — and there `dual` beats it too.
- **End tokens are in `I_CoC`, and nearly silent.** The scored span is
  `seq[prompt_len : eos_pos + 1]`: the CoC text, `<|cot_end|>` and the stop token
  `<|traj_future_start|>`; all 100 calibration rollouts ended with that pair. Measured on 30
  calibration clips: text tokens carry 94.5% of the gradient energy the loss sends into the
  final hidden states, `<|cot_end|>` 5.5% (6.9% of the tokens, mean NLL 0.061), the stop
  token 0.0% (NLL 0.0000 — it is certain once `<|cot_end|>` is out). So the decision to stop
  enters `I_CoC` through one position in about fourteen, under teacher forcing, on a prefix
  that is always well-formed — which says nothing about reaching that state when the prefix
  is the pruned model's own.
- **Which clips make each score** (`calib_100`, Q heads, per-clip gate gradients): turns
  are 17% of the clips and 46.7% of `I_traj`'s mass (mean FM loss 0.53 against 0.18 for
  cruise); decelerate/stop clips are 18% and 7.6%. `I_CoC`'s mass follows the clip shares
  (44 / 17 / 19 / 21%). Ten clips carry 49% of `I_traj` and 27% of `I_CoC`. The
  trajectory-only arm's weakest situation is the one its score under-represents most; with
  four buckets this is a coincidence worth testing, not a result.

## Analysis (opinion)

**What the GPU pass changed.** Before it, this section argued that in late layers the CE
gradient lands on text tokens while the FM gradient stays on vision tokens, and that the
expert's useful cache ports end near layer 24. The decisive prediction built on that
(agreement returns at matched tokens) held, 0.79 against a 0.70 bar. The picture behind it
did not: the FM gradient also leaves the vision tokens (R9), late cache layers carry 42–43%
of a mid-layer unit's FM mass (R12), and the expert reaches units mostly through non-vision
cache entries (R12). What follows is the account the measurements support.

**One switch, and it is a hand-over between tokens.** R1's reframing stands — one shared
depth profile times one step — and R9–R11 say what the step is. Up to about layer 17 both
scores are earned where units act on vision tokens, and there the two losses are
interchangeable: constant ratio (R11), near-identical head rankings (0.96, R10), and
CoC-on-half-A predicting trajectory-on-half-B as well as trajectory itself does (R4). Over
layers 17–24 the network hands the work from the vision tokens to the text tokens, and
both scores follow it. After the hand-over they are earned mostly at text tokens, where
the ratio is ~2× instead of ~15× and where, on the MLP axis, the CoC's own component nearly
triples as the stream turns token-peaked (R6, R11).

**Why the rankings part.** Not because the two losses want different things from the same
computation: at matched tokens they largely agree even in late layers (0.79 at vision
tokens, 0.81 at prompt tokens). They part because (i) a head does different jobs at vision
tokens and at text tokens — within either loss its two rankings correlate at only
0.35–0.48 — and (ii) after the hand-over the two scores mix those jobs differently:
`I_CoC` is 98% text-side, `I_traj` keeps 26–35% at vision tokens and 8–16% at the
ego-history tokens. The one place the losses disagree on the *same* tokens is the generated
CoC tokens (0.41), which is where the CE loss is applied and where the expert reads at the
highest per-token density; on Q heads those tokens earn a third (`I_traj`) to two-thirds
(`I_CoC`) of a late layer's score and a twentieth to a fifth of a trunk layer's.

**Why `I_traj` falls exactly at layers 22–23.** Because the FM loss sees VLM units almost
only through the late cache — 84–89% of all `I_traj` arrives through cache layers ≥ 22 —
and the densest port is cache layers 21–23 (35–41%; layer 22 alone about a fifth). A unit
above that port cannot write into it, and that closing accounts for 73–90% of the fall
(R12). This is a structural fact about the two-tower interface, which is why the step sits
at layer 23 in every run of R2 whatever the data, the target or the action dimension.

**Why the trunk agrees.** R12 also explains R4's interchangeability from the trajectory
side: a trunk unit reaches the FM loss through the same door it reaches the CE loss — the
late layers — rather than through the mid-layer vision cache one might have expected the
expert to read it from (12% of a trunk unit's `I_traj`). Stated carefully: this is the
route of the *gradient*. The knockouts show the expert's trajectory depends on the early
and middle cache; the first-order score is nevertheless dominated by the late one.

**What it is not.** Not estimation noise (Q heads, R4, and again R10 with ceilings
0.79–0.94). Not the calibration set, its size or its drawing rule (R2). Not the
GT-versus-own-rollout asymmetry (R2). Not "the LM head versus the expert" as such (R8).
And, as of R13, not aligned-versus-misaligned gradients: the two losses' residual
gradients are orthogonal everywhere, including where their rankings agree.

**Consequences for the method section.**

- The union is needed where the token mixtures differ, i.e. after the hand-over, and is
  nearly free before it (corrected agreement 0.85–0.96). That is a sharper statement of C2
  than "each loss illuminates its own substrate": the substrate is shared; what differs is
  which tokens each loss scores a late unit on.
- It also says what a single criterion *cannot see*. `I_CoC` is blind to a late unit's work
  at vision and history tokens (2% of its late score; 34–51% of `I_traj`'s). `I_traj` is not
  blind to text tokens (49–66%) but ranks heads differently at the CoC tokens themselves.
  **Corrected by R17 (2026-09-21):** this paragraph went on to say that the blindness
  "matches" CoC-only losing closed-loop driving (0.660, −0.089 against the unpruned model)
  and trajectory-only losing language (LingoQA 73.2 → 37.0). Half of that does not survive
  measurement. At the shipped dose the action does not depend on late-layer units at all —
  the CoC-only arm's late mask costs +0.1% FM loss and 0.00 m — so what `I_CoC` cannot see
  up there is not why it loses driving; it loses it in the trunk (R17). The language half
  stands (the trajectory-only arm's late mask alone: NLL +65.5%), with the amendment that
  on LingoQA *both* single criteria collapse (CoC-only 30.2).
- R8 generalises beyond these two losses: late-layer units are specific to each output, so
  a criterion needs a term per output the compressed model must keep. VQA would not ride
  on `I_CoC`.
- Practical: the step's location is fixed by the architecture (cache port 21–23), not by
  data. An allocation that treats layers ≤ 21 and ≥ 23 as two regimes does not need to be
  re-derived per calibration set.

**What the causal round (R14–R19) changed.** R1–R13 described two *scores*. The second plan
asked whether that description is about the network's function, with every prediction
written down first. Roughly half of them failed, and the failures are what reorganise the
account.

*The split is between two doors, not between two tasks (R16).* A random readout through
the expert has `I_traj`'s depth profile (0.98 / 0.96), hands over between tokens like it and
ranks heads like it (0.99, early and late); two content-free probes reproduce the step at
layer 23 and the late rank split (0.79 → 0.35). What a loss can see of the VLM is fixed by
where it reads it — the final hidden states at a dozen text positions, or every cache layer
at every position through the expert — and hardly by what it asks for there. The step itself
decomposes: the VLM's wiring gives *any* cache-side demand a ramp (a deeper unit feeds fewer
cache layers), the expert's concentration on cache layers 21–23 lifts layers 16–21 by
1.3–1.9× and so turns the ramp into a step, and the FM objective adds a factor of about 0.8
after layer 22. "Each loss illuminates its own substrate" should read "each interface does".

*After the step the VLM is, functionally, a language network (R17).* Removing 19% of the
late heads or channels — by either score, by their difference, at random, or as whole arms —
does not move the FM loss (−0.0%, −0.6%; arms +0.1%) or minADE. It moves the NLL: +6 to +8%
for random sets, +14 to +15% for the heads `I_CoC` singles out, +65% for the
trajectory-only arm's late mask. So `I_traj`'s late-layer scores, 18–47% of its peak on the
MLP axis, have no functional counterpart at this dose. They are the first-order shadow of the
route the gradient takes through the late cache ports (R12), a route the knockouts had
already shown the expert does not depend on. `I_CoC` is calibrated across depth (0.113 vs
0.136 NLL per unit of score on Q heads); `I_traj` is not (0.0111 vs 0). I predicted a double
dissociation in the late layers and found a single one.

*That corrects an explanation given above.* The first round's "Consequences" said CoC-only
loses driving because `I_CoC` is blind to a late unit's work at vision and history tokens.
Stage 0 confirms the blindness in score currency (R14) — and R17 shows it does not matter:
the CoC-only arm's late mask costs the action nothing (+0.1% FM loss, +0.00 m). CoC-only
loses the action in the trunk (+17.0% FM loss, +0.46 m with the trunk mask alone, against
+15.8% and +0.56 m for the whole arm).

*The measured case for the dual criterion is in the shared trunk (R17).* Below layer 22 the
two scores agree at 0.85–0.96 and differ, arm against arm, in about two heads per layer. Yet
a trunk pruned by `I_traj` alone costs the action +26.1% FM loss and +2.64 m, by `I_CoC`
alone +17.0% and +0.46 m, by their union +4.8% and +0.06 m, on held-out clips with the late
layers left dense. Neither first-order score identifies on its own the trunk units the
action needs; keeping a unit when *either* door ranks it high does. The port route does not
explain which units those are (the rescued heads reach the FM loss through the early cache
no more than any other trunk head, 7% against 7%). The cross-over runs say where the
trajectory-only trunk goes wrong: most of its action damage disappears when the late layers
are thinned in *any* way — a random late mask takes it from +26.1% to +8.8% — so it is damage
that the intact late layers amplify into the cache the expert reads, not damage to the
action pathway itself. The likeliest reading is that `I_traj` alone lets go of trunk units
the language pathway needs, and the language network above them then writes a corrupted
late cache. The action is thus not independent of the language pathway even though it does
not *use* the late layers: the expert reads what they write. That is an argument for the
language-side score that holds for a user who only cares about driving.

*The user's hypothesis is right, and it is the shared half of the answer (R18).* Action
importance is high where vision–vision interaction is needed: the depth profile of the
action's need for it follows `I_traj` at vision tokens at +0.85, up to and under the
importance peak (blocking every vision–vision edge from layer 18 on still costs +27.8%), and
it ends with the hand-over, at layers 21–24. But it follows `I_CoC` at vision tokens just as
well (+0.87), and blocking the same edges raises the NLL with the same profile (+0.86). So
vision integration is what *both* scores are earned on below the step — it is the shared
profile of R1, now with a causal footing — and it cannot be what separates them. What
separates the pooled profiles is what lies above it: a text-token component that is real
function for `I_CoC` and a first-order shadow for `I_traj`. Observed attention mass is the
wrong ruler for any of this: cross-image attention is 0.60 of a vision query's mass in the
late layers, where removing it changes nothing.

*Single criteria fail unevenly; the union fails evenly (R15, R17, R19).* On held-out clips
and matched text the trajectory-only arm loses the language (NLL +241%) and the CoC-only arm
the action (FM loss +15.8%, minADE +67%); free-running, the CoC-only arm also loses the
language it was selected on (12% degenerate chains, 8–13% that never stop), because a
teacher-forced first-order score protects next-token likelihood on a well-formed prefix, not
the ability to reach one. Across driving situations the trajectory-only arm is weakest at
decelerate/stop and turns, the CoC-only arm everywhere, `dual` evenly (+9 to +15%).

**Consequences for the paper, second round.**

- C2 can now be stated at the level of function, in three parts: (i) a criterion that reads
  the VLM through one output interface is blind by construction, whatever its objective
  (R16); (ii) above layer 22 only the language channel depends on which units are kept, so
  letting `I_traj` decide there costs language for nothing (R17); (iii) below layer 22 the
  union of the two scores selects a far better network than either score, for the action as
  much as for the language (R17). Item 6 below is a draft paragraph.
- "Each loss illuminates its own substrate" should become "each interface" (R16), and the
  late-layer `I_traj` profile should not be read as the action's dependence on late layers
  (R17).
- The shared trunk profile can be given its cause: both scores are earned where vision
  tokens are integrated, layers ~6–21 (R18).
- A side finding for the allocation work: pruning every layer at the same ratio loads the
  surviving heads of layers 0–1 with 4–8× their dense importance under `dual`, and 11–66×
  under the single criteria (R15).

## Suggested manuscript changes

**1. A paragraph for Sec. 7 (or after Figure 1).** Draft, to be cut to length:

> *Why the objectives diverge.* The two depth profiles of Fig. 1 are not independent.
> Their ratio `I_traj / I_CoC` is constant over layers 0–21 (coefficient of variation 9%
> on the MLP axis, while `I_traj` itself varies 15×), constant again over layers 23–34,
> and falls 4.6× in between (Fig. 3a); a single step accounts for 93% of the ratio's
> variation over depth, and its location is layer 23 in all eleven full-loss importance
> runs available to us — disjoint calibration sets, 100 to 4,000 clips,
> out-of-distribution clips, and a trajectory target sampled from the model itself. To
> see what the step is, we split each unit's gate gradient, which is a sum over token
> positions, by the type of token the unit acts on; the split is exact (the parts
> reproduce the unsplit gradient to 6×10⁻⁶). Through layer 17 both scores are earned at
> the vision tokens (90% of `I_traj` and 75% of `I_CoC` on the MLP axis), and there the two
> losses are interchangeable: their ratio shows no step at any depth (15.0× before,
> 12.4× after) and they rank heads identically (ρ = 0.96 after correcting for the
> split-half ceiling). Over layers 17–24 the network hands the computation over to the
> text tokens and both scores follow (Fig. 3b): `I_CoC` entirely (98% text-side in layers
> 27–34), `I_traj` in part (26% still at vision tokens, 8% at ego-history tokens). The
> late-layer disagreement of Fig. 1c is largely this difference in token mixture: a head's
> rank at vision tokens predicts its rank at text tokens poorly under either loss
> (ρ = 0.35–0.48), whereas at matched tokens the two losses still agree in layers 22–34
> (ρ = 0.79 at vision tokens and 0.81 at prompt tokens, against 0.38 for the pooled
> scores; Fig. 3c); only at the generated CoC tokens do they rank heads differently
> (0.41). The location of the step has a structural cause. The FM loss reaches a VLM unit
> only through the KV cache, and a unit in layer *l* can only affect cache layers above
> *l*. Decomposing `I_traj` additively by the cache layer its gradient arrives through
> (Fig. 3d), 84–89% arrives through cache layers ≥ 22 and 35–41% through layers 21–23
> alone; a unit above that port can no longer write into it, which accounts for 90% (MLP)
> and 73% (Q heads) of the fall of `I_traj` between layers 21 and 23.

**2. Figure 3, four panels.** (a) `fig3_ratio_step`; (b) `fig4_token_handoff_mlp`;
(c) `fig4_same_token_q_head`; (d) `fig4_itraj_ports_mlp`. Draft caption:

> **Figure 3. One shared depth profile and one hand-over.** (a) Ratio of the two
> importance scores per layer (raw layer sums, log axis; dotted: two-level fit). (b) Share
> of each score that is earned where the unit acts on vision tokens (additive split of the
> gate gradient by token type; MLP channels). (c) Rank agreement between the two
> objectives for Q heads, corrected for the split-half ceiling: the shipped scores, and
> both scores restricted to vision tokens or to text tokens. (d) `I_traj` per layer
> rebuilt from the cache layer its gradient arrives through; the bars sum to the curve of
> Fig. 1a, and a unit in layer *l* cannot reach cache layers ≤ *l*. Layer indices are
> zero-based; layer 35 is omitted because the trajectory score is zero there.

If four panels are too many, (a) and (d) carry the structural argument and (b) and (c) the
token argument. Supplementary: `fig3_ratio_step_draws`, `fig3_boundary`, both
`fig3_noise_ceiling_*`, `fig4_ratio_by_token_*`, `fig4_port_profile`, the partition and
pairs panels.

**3. A correction to how Figure 1c is described.** Any sentence that reads the MLP curve's
fall to ~0.3 as objective disagreement alone should be softened: about a third of that
fall is the ceiling dropping (R4), and 0.34 is half of what is attainable there, not a
third of it. Either cite the corrected values (Q 0.85 → 0.37, MLP 0.92 → 0.54) or put
the ceiling in the supplementary and point to it. The Q-head curve needs no change — its
ceiling is flat.

**4. Wording of C2 and Sec. 7's last paragraph.** "Each loss's gradient illuminates only
its own substrate — the FM loss the early-layer, cache-numeric … circuitry; the CE loss the
late-layer decode path" is stronger than R9–R12 support. The FM gradient is not confined to
early layers (MLP layers 23–34 keep 18–47% of the peak layer's `I_traj`, earned mostly at
text tokens), and in the trunk the two gradients illuminate the *same* units. A version the data back:
"below layer ~22 the two losses score the same units the same way; above it each scores a
unit on a different mixture of tokens, so a unit that matters to one loss can be invisible
to the other".

**5. Layer bands.** C2 and Sec. 7 give "action: L0–17; language: L18–35", which is the
9-layer resolution of the knockout grid. The importance ratio localises the switch to
layers 22–23 and the token hand-over to layers 17–24, both at single-layer resolution and
both compatible with the knockouts. Worth saying once so the numbers do not look
inconsistent.

**6. A paragraph on why the criterion needs both losses (after the paragraph of item 1, or in
Sec. 5 where the dual criterion is introduced).** Draft, to be cut to length:

> *Why score with both losses.* The divergence of Fig. 1 is a property of the two output
> interfaces rather than of the two tasks. Replacing each loss by a random linear readout
> through the same interface — the expert's predicted field, or the final hidden states at
> the CoC positions — reproduces the two depth profiles (r = 0.96 and 0.94 on the MLP axis),
> the step at layer 23 (5.5× against 4.5×) and the late-layer rank disagreement (ρ = 0.79 →
> 0.35 between the two probes), and the expert-side probe ranks heads as `I_traj` does
> (ρ = 0.99). A criterion that reads the network through one interface is therefore blind
> by construction, and the blindness has functional consequences. We applied each shipped
> mask to one depth band at a time on 100 held-out clips, every model teacher-forced on the
> dense model's reasoning text. Above layer 22 the action does not depend on which units are
> kept (flow-matching loss within 0.13% of the dense model under all three criteria) while
> the reasoning does: the trajectory-only mask raises the CoC NLL by 65% there, the CoC-only
> mask by 6%, and removing only the heads that the trajectory-only criterion drops and the
> dual criterion keeps costs 8.6% against 1.3–3.2% for random heads. Below layer 22, where
> the two scores agree (ρ = 0.85–0.96) and the criteria differ by about two heads per layer,
> their union still selects a markedly better network than either score alone: +4.8%
> flow-matching loss and +0.06 m minADE, against +17.0% / +0.46 m (CoC-only) and +26.1% /
> +2.64 m (trajectory-only). With the full masks each single criterion loses one channel —
> trajectory-only the reasoning (NLL +241%; LingoQA 37.0), CoC-only the action
> (flow-matching loss +16%, minADE +67%) and, in free-running generation, the reasoning too
> (8–13% of its chains never terminate) — while the dual criterion loses neither (+5.5%, +33%,
> +0.01 m; LingoQA 68.8), and it is the only one whose open-loop cost is even across driving
> situations (+9 to +15% of the dense minADE, against +10 to +40% and +46 to +88%).

**7. Figure for that paragraph, three panels.** (a) `fig5_probe_ratio_mlp`;
(b) `fig5_damage_by_band_fm` next to `fig5_damage_by_band_nll` (or one of them with the other
in the supplement); (c) `fig5_failure_by_manoeuvre`. Supplementary: `fig5_probe_rank_q_head`,
`fig5_dissociation_*`, `fig5_dual_saves`, `fig5_matched_*`, the three `fig5_vv_*` panels.

**8. Two sentences of the current draft need care.** (i) Anything of the form "CoC-only
loses the driving, trajectory-only loses the language": both single criteria lose the
language on LingoQA (30.2 and 37.0), and free-running it is the CoC-only arm whose chains
degenerate (12.1% against 2.7%); name the readout. (ii) Anything that reads late-layer
`I_traj` as the action's dependence on late layers: at the shipped dose the action does not
depend on them at all (R17). Late-layer `I_traj` is where the gradient travels.

## Caveats

- The ratio's *levels* are not interpretable (different loss units); only its shape is.
- The ceiling correction applies the Pearson attenuation formula to Spearman values and
  uses half-sample reliabilities. Two routes agreeing (split halves, disjoint draws) and a
  third run reproducing it (R10) are the reason to trust it, not the formula.
- R9–R13 describe the **first-order importance score**, not the network's computation.
  Where the score's gradient travels (late cache, text-side entries) and what the expert
  causally relies on (early and middle cache, vision entries — the knockouts) are
  different, and both are true. Do not write "the expert reads the late cache".
- "Text-side" pools prompt text and generated CoC. The CE loss's first target token is
  predicted from the last *prompt* position, so part of what R9 files under prompt text is
  generation. R10's CoC-only and prompt-only rows separate them; R9 and R11 do not.
- A gate is still shared across positions within a token type; "vision" pools 2,880
  positions from four cameras and four frames.
- The seed-split decompositions (R12) miss the pre-registered 1e-3 linearity bar and sit
  at a bf16 floor of ~0.4% median, ~1.5% worst layer. The shares quoted from them are
  tens of percentage points, so the conclusions do not depend on it, but the numbers
  should not be quoted to more than two digits.
- Why cache layers 21–23, 26 and 33 are the dense ports is not explained here. It
  coincides with the depth where the lens readout turns token-peaked (R6); whether that
  is cause or coincidence would need an expert-side experiment.
- The lens kurtosis is measured at text and CoC positions by construction, and "no edge
  into a vision position matters after layer 17" is about attention edges; late-layer MLP
  updates at vision positions were never knocked out.
- The port map was run on my own judgement as a finer-resolution repeat of an approved
  measurement; its three predictions were written down after a 3-clip smoke run of the
  band-level version and before any port-map data.
- One model, one calibration set of 100 clips for R9–R13. R2 suggests depth structure does
  not depend on the calibration set, but that was shown for the pooled score only.
- R15–R18 read both losses teacher-forced on the dense model's text. That is what makes the
  arms comparable clip by clip, and it is blind to what free-running generation does to a
  pruned model (R19: 8–13% of the CoC-only arm's chains never stop). Read the two kinds of
  language readout side by side.
- R17's unit sets are one dose (19% of a band's heads or channels); the arms are the shipped
  40%. "The action does not depend on late layers" is a statement at those doses.
- R17 says the union selects a better trunk than either score; it does not say why, and the
  one explanation tried (the rescued heads reach the expert through the early cache) is
  wrong. It supports scoring with both losses, not the rank-max rule in particular.
- R18's nested windows measure need given that every later (or earlier) layer is blocked
  too. Its predictions, and those of A2-bands and A2-mix, were written after earlier results
  of the same plan had been seen; the plan says which.
- R16's probes are random *linear* readouts, one draw per clip. D4 was judged on the
  cache-side profile, not on the unit-side port profile the plan named (72 backwards per clip
  per probe were not run).
- R19 is open loop (this repo has seen open and closed loop disagree), 279 turn clips, no
  multiple-comparison correction.
