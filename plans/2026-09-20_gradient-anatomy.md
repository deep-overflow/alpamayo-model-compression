# Gradient anatomy: on which tokens, and through which cache layers, does each loss see a unit?

Date: 2026-09-20. Branch: `worktree-why-importance-differs`.
Status: **awaiting approval — no code written, no GPU used.**

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

## Files

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
