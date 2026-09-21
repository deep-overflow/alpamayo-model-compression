# Gradient anatomy: on which tokens, and through which cache layers, does each loss see a unit?

Date: 2026-09-20. Branch: `worktree-why-importance-differs`.
Status: **approved and run the same day** ("진행해줘", "calib100으로 진행해줘"). Results in the
Outcome section at the end; write-up in `paper/2026-09-20_why-importance-differs.md` (R9-R13)
and `reports/evaluation/2026-09-20_gradient-anatomy.html`. Everything between here and
Outcome is the plan as approved, unedited except for the follow-up section, which is dated.

## Why

`paper/2026-09-20_why-importance-differs.md` explains the Figure 1 divergence from stored
runs: the two depth profiles are one shared profile times a single 4.6× step at layers
22–23, the step is a property of the network (change point 23 in all eleven full-loss
runs), and three independent signals put a regime change at the same depth. The proposed
reason is:

> Below the boundary both losses' gradients land on the same tokens (the vision positions
> where the scene representation is built). Above it the CE gradient lives on the ~200
> text positions where the rationale is being written, while the FM gradient still lives
> on the ~2,900 vision positions the expert reads. A unit's gate is shared across all
> positions, so late-layer units are scored on two different jobs.

Stored data supports this only indirectly (knockouts at 9-layer resolution, and a
write-norm that covers text positions only). The shipped score sums over positions before
taking `|.|`, so where the gradient lands was never recorded. This experiment records it.
If the explanation is wrong the paper should not state it, and the experiment is cheap
enough that there is no reason to state it unmeasured.

## Hypotheses

- **H-token**: the late-layer split is a difference in token support. Restricted to the
  same token type, the two objectives still agree in late layers.
- **H-dir** (the alternative): same tokens, different directions. Even on the same token
  type the two objectives rank late-layer units differently, because they read different
  subspaces of the residual stream (vocabulary-aligned vs. cache-projected).
- **H-port**: `I_traj` falls at layers 22–23 because the FM gradient enters the VLM
  through cache layers at or below ~24, and a unit can only reach cache layers above it.

H-token and H-dir are exclusive; H-port is independent of both.

## Design: three exact linear decompositions of the shipped score

The shipped quantity is `dL/dg_{l,u} = sum_p x_u(p) * dL/dx_u(p)` with one gate per unit.
All three decompositions split that sum without changing it, so each one has a built-in
integrity check: the parts must add back to the shipped signed gradient.

| | splits by | how | applies to |
|---|---|---|---|
| D1 | token type the **unit acts on** | gate of shape `(n_types, U)` indexed by each position's type, types = vision / history / prompt text / CoC / sink (`analysis_lib.compute_spans`) | both losses |
| D2 | cache **layer** the FM gradient enters through | `vlm_backward_from_cache` is linear in its seed; seed one band of cache layers at a time: 0–7, 8–15, 16–21, 22–24, 25–29, 30–35 | FM only |
| D3 | cache **position type** the FM gradient enters through | same call, seed masked to one token type's positions | FM only |

D1 is populated on every backward, so D2 and D3 come out already split by D1's token
types. Per clip: 1 CE backward + 6 band backwards + 5 position-type backwards = 12 VLM
backwards on a retained graph, against 10 in `importance_stepvlm_v1`.

Everything else is the shipped protocol unchanged: dense model, `calib_100`, `t0_us =
5_100_000`, clip seeds `sha256(f"42:{clip_id}")`, own-rollout CoC teacher-forced,
10-step midpoint FM grid with the GT action target, FP32 gates, BF16 model.

Stored per cell (loss × band-or-span × token type): `E_c|G|` for Q heads and MLP
channels, plus the per-clip signed Q-head arrays (small) for split-half ceilings. MLP
per-clip arrays are not kept (~10 GB); MLP ceilings come from two half-accumulators
filled by clip parity.

## Pre-registered predictions and gates

Shares are `E|G_part| / sum_parts E|G_part|`, reported next to the cancellation index
`sum_parts E|G_part| / E|sum_parts G_part|`, because `|.|` is taken after the sum and the
parts are therefore not additive in magnitude.

- **G0 (integrity, must pass before anything is read)**: parts sum to the single-gate
  signed gradient of the same clip and seed with relative error < 1e-3 on Q heads, and
  `E_c|sum|` reproduces `importance_v2_ada`'s Q-head ranking at ρ ≥ 0.99 in every layer.
- **P1 (token support)**: the CE gradient's text-side share (prompt text + CoC) averages
  < 0.35 over layers 6–17 and > 0.65 over layers 27–34; the FM gradient's vision share is
  > 0.65 in both bands. *If CE stays on vision positions in late layers, H-token is wrong
  at its root.*
- **P2 (same-token agreement — the decisive one)**: Q heads, layers 22–34, within-layer
  Spearman between `I_traj` and `I_CoC` both restricted to **vision** positions, corrected
  for that restricted score's own split-half ceiling.
  - ≥ 0.70 → **H-token supported** (the pooled value is 0.37; the early-band value 0.85).
  - ≤ 0.45 → **H-token refuted, H-dir stands.** Run the fallback below.
  - in between → both contribute; report the split and claim neither.
- **P3 (ports)**: for units in layers 8–21, less than 20% of FM mass arrives through cache
  layers 25–35. *If it is above 40%, late cache layers are real read ports for early
  units, and the fall of `I_traj` is not a port effect — it is late units writing content
  the expert ignores.* (For units at layer ≥ 24 the share is 100% by construction and
  carries no information.)
- **P4 (read positions)**: at every unit depth ≥ 65% of FM mass enters through
  vision-position cache entries, in line with 73% of expert attention mass and the
  +2.38 m vision knockout. Descriptive: per-token density of the CoC entries against the
  43× found by knockout.

No gate on the ratio-by-token-type curves; they are reported descriptively, because
H-token does not predict a flat ratio on any single token type (CE's vision-position
gradient must itself decline in late layers, as fewer text←vision edges remain above).

**Fallback if P2 refutes H-token**: at matched positions, the cosine between the two
losses' residual-stream gradients and the overlap of their top-k principal subspaces
across clips, by layer. That measures H-dir directly. It needs residual hooks only and
can reuse the same forward pass.

## What each outcome does to the paper

| outcome | the paragraph in `paper/2026-09-20_why-importance-differs.md` |
|---|---|
| P1 + P2 pass | keep as drafted; add one panel (CE text-side share by depth, and same-token agreement beside pooled agreement) and state the mechanism as measured |
| P1 passes, P2 in between | keep the regime-change account, describe the token split as a contributor, not the cause |
| P2 refutes | drop "reading mostly vision-token cache entries" as the reason; replace with the subspace result from the fallback |
| P3 fails | reword the `I_traj` fall: not "the expert's read ports end" but "late layers write what the expert does not use" |

R1–R5 of the note (the step, its robustness, the noise ceiling, the partition) do not
depend on any of these outcomes.

## Cost

| | |
|---|---|
| reference | `importance_stepvlm_v1`: 100 clips × 10 VLM backwards, about 20 min wall on one card (config 00:02 → npz 00:22, 2026-08-22) |
| this run | 100 clips × 12 backwards plus typed gates → estimated **30–45 min on one Ada card** |
| memory | the same retained graph as `stepvlm`; typed gates applied per token-type slice so no `(T, U)` gate tensor is materialised |
| analysis | CPU, minutes |

One Ada card (4–7), to stay on the architecture of `importance_v2_ada` and the per-step
file. `df` on `/mnt/nvme1n1` first; the output is about 250 MB.

## Follow-up: port map (added 2026-09-20, after the 3-clip smoke run, before any 100-clip result)

Status: **not separately approved.** The user approved the pass above ("진행해줘",
"calib100으로 진행해줘"). This follow-up is the same measurement at finer resolution (same
code path, same protocol, same `calib_100`, about 1 GPU-hour on idle Ada cards), run on
my own judgement because it decides how the main pass's P3 result should be read. It is
reported as an extension, and the note marks which claims rest on it.

The smoke run's layer sums (stable at 3 clips because each is a sum over 12,288 channels)
already contradict P3 as written: mid-layer units receive about half of their FM mass
through cache layers 25-35, so "the expert's read ports end around layer 24" is wrong. But
they also show the band 22-24 -- three layers -- carrying more of those units' FM gradient
than any other band, and closing exactly where `I_traj` steps down (a unit in layer l can
only write into cache layers above l). Six bands cannot tell one dense port from a smooth
profile cut at an unlucky boundary. `run_port_map.py` repeats D2 at single-cache-layer
resolution, crossed with the position group of the cache entries (vision / everything
else): 72 partial-seed backwards per clip, FM only, same protocol, `calib_100`.

It also replaces mass shares with an additive decomposition. `E|G_part|` shares do not
add up (cancellation index 1.2-1.7 at six bands, worse at 72 parts). The signed share
`sum_u G_part,u * sign(G_full,u)` does: summed over all ports it returns
`sum_u |G_full,u|`, the layer's shipped `I_traj`. So `I_traj(l) = sum_m S(m, l)` exactly
(up to the bf16 floor), with `S(m, l) = 0` for `m <= l`.

Predictions, written before the port-map data exist:

- **PM1 (a port, not a boundary)**: for units in layers 16-21 (MLP), the per-cache-layer
  signed share peaks at a cache layer in 22-24, and those three layers carry >= 30% of
  the units' `I_traj`. *If the profile over cache layers is flat or peaks elsewhere, the
  band result was an artefact of where the bands were cut.*
- **PM2 (the step is ports closing)**: write the fall `I_traj(21) - I_traj(23)` as
  [share through cache layers 22-23, which a layer-23 unit cannot reach] + [change in the
  share through cache layers >= 24]. The first term is >= 50% of the fall. *If it is
  below 25%, the step is not a port effect: late units simply write less of what the
  expert reads, through the same ports.*
- **PM3 (what is read there)**: for ports m >= 22 more than half of the signed share
  enters through non-vision cache entries; for ports m <= 15, through vision entries.

Cost: about 35 s per clip (a backward seeded at cache layer m only traverses layers below
m), three cards in round-robin shards, roughly 20 minutes.

## Files

- `experiments/head_analysis/run_port_map.py` / `analyze_port_map.py` (follow-up) →
  `outputs/portmap_v1_s{0,1,2}/`, merged into `outputs/portmap_v1/`.
- `experiments/head_analysis/prune_lib.py`: add `TypedUnitGates` as a new class. No
  existing code path changes; `UnitGates` stays as it is.
- `experiments/head_analysis/run_gradient_anatomy.py` → `outputs/gradanat_v1/`
  (`config.json`, `metrics.json`, `summary.txt`, `anatomy.npz`, `plots/*.png`).
- `experiments/head_analysis/analyze_gradient_anatomy.py`: gates G0, P1–P4, the ceilings,
  and the paper panel.

## Limits

- A gate is still shared across positions *within* a token type. "Vision" pools 2,880
  positions from four cameras and four frames; a finer split (per camera) is possible but
  not planned.
- The CoC span is about 10 tokens, so its cell is the noisiest; P1 pools it with prompt
  text for that reason.
- Type-restricted scores have lower split-half ceilings than the pooled score. Every
  agreement value is reported next to its own ceiling, never raw.
- Own-rollout teacher forcing, one model, one calibration set of 100 clips. R2 of the note
  suggests the calibration set does not matter for depth structure, but that was shown
  for the pooled score only.

## Outcome (2026-09-20)

Runs: `outputs/gradanat_v1` (100 clips, one Ada card, 12 s/clip, peak 42.8 GB, no NaN),
`outputs/gradanat_verify` (3 clips, `--verify`), `outputs/portmap_v1` (100 clips in three
shards, 30 s/clip, peak 40.9 GB). Analysis: `analyze_gradient_anatomy.py`,
`analyze_port_map.py`; each run directory has `summary.txt`, `metrics_analysis.json`, `plots/`.

One design change after the smoke test, before the 100-clip run: D1 for the FM loss is read
off a **single full-seed backward** (the shipped operation), not off the sum of the six band
backwards, so P1 and P2 are exact splits of the shipped score. That is 13 VLM backwards per
clip instead of 12. The residual-gradient fallback was measured in the same pass.

Two smaller deviations from the text above. The per-clip MLP arrays *were* kept, for the
two type-resolved scores only (signed fp32, 1.8 GB), so the MLP ceilings come from random
disjoint halves like the Q-head ones rather than from parity accumulators. And three
analyses were added that no gate rests on: the additive (signed-share) split by token type,
the same-token comparison at CoC positions only and at prompt-text positions only, and the
ratio `I_traj / I_CoC` by token type.

| gate | pre-registered | measured | verdict |
|---|---|---|---|
| G0 (a) typed grads sum to the single gate | rel err < 1e-3 | max 6.0e-06 (3 clips x 13 backwards) | **PASS** |
| G0 (b) seed splits sum to the full-seed backward | rel err < 1e-3 | median layer 3.2e-03, worst layer of a clip: median 1.3e-02, max 4.0e-02; rank rho >= 0.961 | **NOT MET** -- bf16 rounding floor, the same one `analyze_stepvlm` V0 measured (~5e-3 median) and set 2e-2 for. Affects P3/P4 only |
| G0 (c) reproduces `importance_v2_ada` | Q-head rho >= 0.99 in every layer | Q min 0.9996 / 0.9993, MLP min 1.0000 / 1.0000 (traj / CoC) | **PASS** |
| P1 CE on text-side late, on vision early | < 0.35 (L6-17), > 0.65 (L27-34) | Q 0.404 / 0.940, MLP 0.337 / 0.930 | late clause passes on both axes; early clause misses on Q by 0.05 |
| P1 FM stays on vision | > 0.65 in both bands | Q 0.605 / **0.333**, MLP 0.754 / **0.276** | **FAIL** -- the FM gradient also leaves the vision tokens |
| P2 same-token agreement (Q, vision, L22-34, corrected) | >= 0.70 supports, <= 0.45 refutes | **0.786** (ceilings 0.84 / 0.90; pooled 0.38) | **SUPPORTED** |
| P3 late cache layers are minor ports | < 0.20 (> 0.40 refutes) | Q 0.417, MLP 0.433 | **REFUTED** |
| P4 FM enters through vision cache entries | >= 0.65 at every unit depth | mean 0.35 / 0.38, min 0.15 / 0.20 | **FAIL** |
| PM1 cache 22-24 is a port, not a band artefact | peak in 22-24, >= 0.30 | peak at 22, 0.431 on both axes | **PASS** |
| PM2 the `I_traj` step is ports closing | closed-port term >= 50% of the fall | MLP 89.5%, Q 72.6% | **PASS** |
| PM3 late ports are read at non-vision entries | vision share < 0.5 for m >= 22, > 0.5 for m <= 15 | 0.39 / 0.36 vs 0.78 / 0.89 | **PASS** |

What I got wrong, stated plainly:

- I expected the FM gradient to stay on vision tokens in late layers. It does not: both
  scores move from vision tokens to text-side tokens over the same layers (additive vision
  share, MLP: `I_traj` 0.90 -> 0.26, `I_CoC` 0.75 -> 0.02). P2 passed anyway, because what it
  tests -- agreement at matched tokens -- does not depend on that picture.
- I expected the expert's useful ports to end near cache layer 24. They do not: cache layers
  25-35 carry 42-43% of a mid-layer unit's FM mass. The step is still a port effect, but a
  different one: cache layers 21-23 are the densest port in the network (35-41% of all VLM
  `I_traj`; layer 22 alone 19-22%), and a unit above them cannot write into them.
- I expected the expert to reach units through vision cache entries. Only 35-38% does; the
  ~16 generated CoC tokens alone carry 29-32%.

The 3-clip smoke numbers for position shares moved a lot at 100 clips (trunk share through
late non-vision entries: 0.80 -> 0.44), so nothing from a smoke run is quoted anywhere.
