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
- It also says what a single criterion loses. `I_CoC` is blind to a late unit's work at
  vision and history tokens (2% of its late score; 34–51% of `I_traj`'s). `I_traj` is not
  blind to text tokens (49–66%) but ranks heads differently at the CoC tokens themselves.
  That matches the asymmetry the single-criterion arms showed: CoC-only is the
  single-criterion arm that loses closed-loop driving (0.660, −0.089 against the unpruned
  model), while trajectory-only loses language (LingoQA 73.2 → 37.0).
- R8 generalises beyond these two losses: late-layer units are specific to each output, so
  a criterion needs a term per output the compressed model must keep. VQA would not ride
  on `I_CoC`.
- Practical: the step's location is fixed by the architecture (cache port 21–23), not by
  data. An allocation that treats layers ≤ 21 and ≥ 23 as two regimes does not need to be
  re-derived per calibration set.

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
