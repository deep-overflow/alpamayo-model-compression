# Does the score anatomy predict function — and why is the shared profile shaped as it is?

Date: 2026-09-21. Branch: `worktree-importance-causal-validation`.
Status: **approved by the user on 2026-09-21 (all of A–D); in progress.**
Follows `paper/2026-09-20_why-importance-differs.md` (R1–R13) and
`plans/2026-09-20_gradient-anatomy.md`.

Revision history. *Rev. 1* (commit `1b4f981`, pushed before anything was computed) holds
the first set of predictions. *Rev. 2* (this text) records the Stage 0 outcome against the
Rev. 1 prediction, folds in a read-only survey of the evaluation harnesses, and redesigns
part C **before any C measurement**, because a stored result showed the Rev. 1 design had
no power (see C). Nothing in A, B, D was weakened; B and D were widened. *Rev. 3* adds gate
B5 to part B while A and D were running and before any B code existed; A–D are otherwise
as approved.

## Why

The note explains why the two *scores* differ: one shared depth profile times one step;
both scores hand over from vision tokens to text tokens across layers 17–24; at matched
tokens the losses agree, so late-layer rank disagreement is mostly a difference in token
mixture; and `I_traj` steps at layers 22–23 because the FM gradient's densest cache port
(layers 21–23) closes. Four things are still missing before the paper can lean on it:

| gap | question | part |
|---|---|---|
| G1 | Everything was measured on the dense model. Does the structure survive in the models we actually ship? | A |
| G2 | The account is about first-order scores. Do the units it singles out carry the function it says they carry? | B |
| G3 | The user's hypothesis: is action importance high where vision–vision interaction is high? | C |
| G4 | Why does the *shared* profile look the way it does, and is the step a property of the FM objective or of the interface? | D |

Preliminary checks on stored data (throwaway scripts, 2026-09-21, not in the repo):
importance re-measured on models pruned 13–27% by the dual criterion keeps the change point
at layer 23 on both axes (step 4.56× → 4.39× → 3.87× on MLP, profile correlation ≥ 0.992).
At the head level, attention mass on vision predicts `I_traj` at vision tokens (ρ = +0.67,
layers 6–17) but predicts `I_CoC` at vision tokens just as well (+0.61), and at the layer
level the two profiles are unrelated (ρ = +0.04). Both are reasons to measure properly,
not results.

## Stage 0 — what each shipped criterion removed, in token currency (CPU) — DONE

`outputs/gradanat_v1/anatomy.npz` holds the dense model's `E|G|` per unit split by the
token type the unit acts on. Crossed with the kept sets of `dual_u40_v2`, `traj_u40_v2`,
`coc_u40_v2` (`slim_meta.json`: 19/32 heads, 7,390/12,288 channels per layer, all built
from `importance_v2`; the expert tower is untouched in all three).

**S0 as written in Rev. 1, before computing it:** the CoC-only criterion retains the least
late-layer `I_traj` mass at vision + history tokens of the three arms; the trajectory-only
criterion retains the least `I_CoC` mass at CoC tokens; `dual` is within 5 pp of the better
single criterion on both.

Outcome (share of the dense score mass that sits on kept units; a random kept set would
retain 0.594 of heads' mass and 0.601 of channels'):

| layers 22–34 | | `dual` | `traj` | `coc` |
|---|---|---|---|---|
| Q heads | `I_traj` at vision + history tokens | 0.787 | 0.828 | **0.678** |
| | `I_traj` at text-side tokens | 0.835 | 0.842 | 0.777 |
| | `I_CoC` at CoC tokens | 0.763 | **0.650** | 0.827 |
| | `I_CoC` at prompt tokens | 0.852 | 0.781 | 0.865 |
| MLP channels | `I_traj` at vision + history tokens | 0.697 | 0.712 | **0.651** |
| | `I_traj` at text-side tokens | 0.699 | 0.710 | 0.658 |
| | `I_CoC` at CoC tokens | 0.715 | **0.669** | 0.729 |
| | `I_CoC` at prompt tokens | 0.688 | 0.646 | 0.709 |

| layers 6–21 (for contrast) | | `dual` | `traj` | `coc` |
|---|---|---|---|---|
| Q heads | `I_traj` at vision + history tokens | 0.823 | 0.836 | 0.804 |
| | `I_CoC` at CoC tokens | 0.681 | 0.643 | 0.717 |
| MLP channels | `I_traj` at vision + history tokens | 0.756 | 0.766 | 0.731 |
| | `I_CoC` at CoC tokens | 0.698 | 0.642 | 0.716 |

- Ordering: **PASS on both axes** (`coc` lowest on `I_traj`-at-vision/history, `traj` lowest
  on `I_CoC`-at-CoC-tokens).
- 5 pp bound: MLP **PASS** (1.5 pp, 1.4 pp); Q heads **PASS** on the trajectory side
  (4.1 pp) and **FAIL** on the CoC side (6.4 pp: `dual` 0.763 vs `coc` 0.827).
- Not predicted, worth carrying into B: on Q heads the CoC-only criterion loses 15 pp of
  late `I_traj`-at-vision/history mass relative to `traj` but only 3 pp in the trunk, and
  what it loses is token-specific (vision/history 0.678 vs text-side 0.777). Late kept sets
  of the two single criteria overlap 0.68 (Q) / 0.69 (MLP): about 6 of 19 kept heads per
  layer differ, which is what sets k in part B.

This is bookkeeping on the scores themselves; it cannot say the lost mass mattered. B does.
The numbers come from a throwaway script over stored files; it enters the repo with A's
analyzer once the plan is approved, so the table can be regenerated.

## A — the anatomy on the shipped pruned models (GPU ~20 min per arm)

`run_gradient_anatomy.py` gains `--mask` (npz with `q_mask`/`mlp_mask`, the key convention
of `run_importance.py --mask`; installed with `mask_lib.PruneMasks` before the gates, so a
masked unit is functionally removed and its gate gradient is exactly zero) and records the
CoC NLL next to the FM loss. A ten-line converter writes the three masks from
`slim_meta.json` (no such utility exists; `analyze_calibsize.kept_masks` is the in-memory
pattern). Verified: the `*_u40_v2` arms slice the VLM only (all 2.66 B removed parameters
are VLM parameters, expert 16/16 heads and 8,256/8,256 channels kept, no weight rewrite),
so mask ≡ shipped checkpoint.

One deviation from the shipped protocol, deliberate: every arm is teacher-forced on the
**dense** model's rollout (generated in the same process with the masks switched off), so
token positions, text and noise seeds are identical across arms and any difference is the
network. The per-clip FM loss and NLL on that fixed text are then a matched functional
readout for free. `analyze_gradient_anatomy.py` gains the same `--mask` so that every rank
statistic and split-half ceiling is taken among kept units only.

- **A1 (`dual` keeps the structure):** ratio change point at layer 23 on both axes; layer
  profile correlation with dense ≥ 0.98 for both scores; MLP hand-over layers within ±1 of
  dense (`I_CoC` 18, `I_traj` 21); ceiling-corrected agreement at vision tokens among kept
  Q heads, layers 22–34, ≥ 0.70.
- **A2 (single criteria hurt where the account says):** on the fixed dense text, the FM
  loss rises more under `coc` than under `dual` and `traj`; the NLL rises more under `traj`
  than under `dual` and `coc` (paired Wilcoxon over clips, p < 0.01 each).
- Descriptive: the port profile is *not* re-measured (72 backwards per clip); the band
  split from the same pass says whether cache 22–24 is still the main port.
- **A2-heldout (added after A2 came out, before it was run).** A2 is measured on
  `calib_100`, the clips every criterion was computed on, so each single criterion is
  in-sample for its own loss; and the stored OOD records have the CoC-only arm *worse* than
  the trajectory-only arm on the GT-CoC NLL (+0.192 vs +0.143). The same three masks are
  therefore read on the 100 held-out `indist_500` clips of part B
  (`run_token_ablation.py --arm-masks`, dense text, same seeds). Prediction: A2's ordering
  holds out of sample — FM loss highest under `coc`, NLL highest under `traj`, `dual`
  lowest on both (paired Wilcoxon, p < 0.01) — with a smaller NLL gap between `traj` and
  `coc` than in sample. *If the NLL ordering flips, A2's language half was an in-sample
  artefact and must be reported as one.*
- **A2-bands (added while B was running; not blind).** A 20-clip partial check of B, made
  to test the analyzer, showed the scale of B's effects: removing 19% of the late-layer
  heads moves the FM loss by under 1% whatever the set, while the NLL moves. If that holds,
  the question that matters for the dual criterion is *at which depth* each single
  criterion loses each channel. The same run therefore also reads each arm's mask applied
  in layers 22–35 only and in layers 0–21 only (six more configs, dense elsewhere; the two bands partition the network).
  Predictions, written with that partial look in hand: (i) late-only masks leave the FM
  loss within 1% of dense for all three arms, and the late-only `traj` mask raises the NLL
  more than the late-only `coc` mask (p < 0.01); (ii) trunk-only masks carry the action
  side — the FM loss rises more under trunk-only `coc` than under trunk-only `traj`
  (p < 0.01); (iii) for every arm, trunk-only + late-only damage adds up to within 25% of
  the full mask's damage on both losses.

## B — token-targeted ablation: does the score anatomy predict function? (GPU ~1 h on four cards)

The decisive one. No runner in the repo takes an arbitrary mask and returns paired
readouts (`run_ablation.py`, `run_eval.py`, `run_cocsafe.py`, `run_grid.py` all build
hard-coded config families), so this is one new runner, `run_token_ablation.py`, built on
`run_eval.eval_config_samples` (one masked teacher-forced forward → CoC NLL; K denoisings
on that cache → minADE) plus the 10-step GT-anchored FM loss, with clip-derived seeds
(`run_pathway2.clip_seed`, shard-invariant — `run_eval.py`'s loop-index seeds are a known
defect) and `--shard/--n-shards`.

Sets, per layer, chosen on the dense `calib_100` scores (`gradanat_v1`), on **both** axes
(k = 6 of 32 heads; k = 2,304 of 12,288 channels, the same 19%). k = 6 is the number of
heads per layer on which the two shipped single criteria actually disagree (Stage 0).

- **T-tok:** largest (rank of `I_traj` at vision + history tokens) − (rank of pooled `I_CoC`);
- **C-tok:** largest (rank of `I_CoC` at CoC tokens) − (rank of pooled `I_traj`);
- **T-pool / C-pool:** the same with pooled scores on both sides — what Figure 1 is about;
- **R:** k random units per layer, 5 seeds (the null);
- each family once in layers 22–34 and once, as the control, in layers 6–17.

That is 18 masks per axis + dense = 37 configs (53 with the B5 sets of Rev. 3), each
measured on 100 **held-out** clips
(the first 100 of `indist_500`; verified disjoint from `calib_100` and from `val`), paired
with the dense model on the same clip, dense text and noise: FM loss, NLL of the dense
rollout, minADE/minFDE at K = 8.

- **B1:** ΔFM(T) > ΔFM(C), and ΔFM(T) above every R-set;
- **B2:** ΔNLL(C) > ΔNLL(T), and ΔNLL(C) above every R-set;
  both by paired Wilcoxon at p < 0.01, for the pooled sets and for the token sets, on each
  axis → a double dissociation in the late layers.
- **B3 (control):** in layers 6–17 the T- and C-sets do *not* dissociate: the ratio
  ΔFM/ΔNLL of the two sets differs by less than the spread of the R-sets.
- **B4 (does token resolution add anything):** T-tok is at least as action-specific as
  T-pool (ΔFM no smaller, ΔNLL no larger), and symmetrically for C. If not, the token split
  explains the scores but selects no better than the pooled scores — worth knowing before
  anyone proposes a token-resolved criterion.
- **B5 (what `dual` saves — added in Rev. 3, before any B run, after the user asked how
  this analysis leads to the paper's case for the dual criterion).** Two more late-layer
  sets per axis, taken from the shipped arms rather than from a rule of mine: **S-coc** =
  units `coc_u40_v2` removed and `dual_u40_v2` kept; **S-traj** = units `traj_u40_v2`
  removed and `dual_u40_v2` kept (layers 22–34, about 4 heads and 1,400 channels per
  layer), each with three random sets of the same per-layer size. Prediction: removing
  S-coc raises the FM loss more than every size-matched random set and the NLL no more than
  their mean; removing S-traj raises the NLL more than every size-matched random set and
  the FM loss no more than their mean (paired Wilcoxon, p < 0.01). *If it holds, the units
  the dual criterion rescues from each single criterion are exactly the ones the other
  output needs — the functional case for scoring with both losses.*
- Set bookkeeping, written down before any B measurement
  (`outputs/tokabl_sets_v1/summary.txt`): a rank-difference rule picks *specialised* units,
  not top ones, so a T set holds about a random set's share of its own currency (Q heads,
  late: T-tok 18.6% of `I_traj` against 15–21% for the random sets) and well under half a
  random set's share of the other (8.0% of `I_CoC`). First order therefore expects the
  cross-over clauses of B1/B2 to hold and the "above every random set" clauses to be
  marginal. The gates stay as written.
- Also reported, not gated: across all configs, how well the first-order prediction (the
  summed `I_traj` / `I_CoC` of the removed units) orders the measured ΔFM / ΔNLL.
- minADE is measured for the dense model, every selected set and one random set per
  family (random sets otherwise skip the K denoisings); it gates nothing: at K = 8 and
  n = 100 its paired CI half-width is ~4% of baseline, coarser than the two losses.
- *Refuted if* T and C damage both channels alike in late layers. Then the token-mixture
  account describes the scores and nothing else, and the paper must say so.

## C — vision–vision interaction by depth (the user's hypothesis; GPU ~45 min on four cards)

**Why Rev. 1's design was dropped.** Rev. 1 knocked out one edge in twelve disjoint
3-layer windows. The stored 9-layer map (`outputs/pathway_e_v1`, n = 50, K = 8) already
shows that design cannot work: "vision ← same-camera earlier frames" costs +6.5% / +16.5%
/ +1.0% / −0.1% minADE band by band but **+63.9%** when blocked everywhere — the layers
compensate for each other, and a 3-layer window would sit inside the ±4.1% noise. The same
table is prior information for the predictions below and is declared as such.

**Design: nested knockouts.** `run_pathway2.py` gains `--edges` and `--cuts` (defaults
reproduce today's 42-config grid bit for bit) and one new edge. For cut points
ℓ ∈ {0, 3, …, 33}:

- *from-ℓ-on* family, block in layers [ℓ, 36): "until what depth is this interaction still
  needed?" — for four edges: **VV** = vision ← any other vision token (own image, earlier
  frames, other cameras; the diagonal stays open), and its three parts **V1** same-camera
  earlier frames (`E1`), **V2** other cameras (`E2`), **V3** own image (new);
- *up-to-ℓ* family, block in layers [0, ℓ), VV only: "how much can later layers make up?"

59 knockouts + the unblocked reference + the causal-mask-only integrity control, on the
same 50 `val` clips and seeds as `pathway_e_v1`, so that the reference, `E1@all` and
`E2@all` must reproduce the stored per-clip values (integrity gate, bf16 floor
|ΔNLL| < 0.01 as in `analyze_pathway2.py`). Readouts: minADE@8 and NLL of the dense
rollout. The depth profile of an interaction is the difference between successive cut
points of its from-ℓ-on curve.

A second, observational pass on the same clips (eager attention, one layer's probabilities
at a time, the collector pattern of `analysis_lib.py`): for **vision queries**, attention
mass per head on {sink, text, own image, same-camera earlier frames, other cameras}.

- **C1 (the hypothesis, coarse):** the VV action-damage profile over the twelve windows
  follows `I_traj` at vision tokens (Spearman ≥ 0.6 against 3-layer means, both axes).
  *Expected to pass, and weak on its own:* both are near zero above layer 24.
- **C2 (the hypothesis, sharp):** `I_traj` at vision tokens peaks at layers 16–20 (Q:
  window 18–20; MLP: 15–17). If importance there is earned by vision–vision interaction,
  blocking VV from layer 18 on must hurt: VV[18, 36) ≥ 25% of VV[0, 36) on minADE.
  **My prediction is that it does not** (< 10%; the stored `E1`/`E2` bands at 18–26 cost
  +1.0% / +3.0%): the interaction is finished before the importance peak, and the peak is
  earned by what those layers hand on, not by what they gather. Either outcome is a result;
  the first makes the user's hypothesis causal, the second locates its limit. Between 10%
  and 25% the question is reported as undecided.
- **C3:** every from-ℓ-on action curve is inside the practical-null band (±5% of baseline)
  for ℓ ≥ 24.
- **C4 (shared substrate):** the NLL damage profile of VV follows the action damage
  profile over the windows below layer 24 (Spearman ≥ 0.6), and at the head level
  cross-image attention mass predicts `I_traj` and `I_CoC` at vision tokens alike
  (band-mean within-layer ρ over layers 6–17 differing by < 0.1). Vision–vision
  interaction is then what both scores are earned on, not what separates them.

## D — is the shared profile a property of the network? (GPU ~15 min)

Replace each loss by a **random readout through the same door** and re-measure the
token-resolved importance (three extra backwards per clip on the anatomy runner's pass):

- head door: `L = Σ_p ⟨r_p, h_final(p)⟩` over the CE positions, `r ~ N(0, I)` fixed by the
  clip seed;
- expert door: `L = Σ_s ⟨R_s, v_θ(x_s, t_s)⟩` over the ten FM steps, same `x_s` as the FM
  loss, `R ~ N(0, I)`;
- bare cache door: `L = Σ_m Σ_p ⟨R_{m,p}, [K_m(p); V_m(p)]⟩`, every cache layer and
  position weighted alike, no expert at all.

None has any driving or language content. The three separate what the VLM's wiring does
to any cache-side demand (bare), what the expert's way of reading adds (expert door), and
what the FM objective adds on top (the real loss). The per-layer norm of the cache gradient
is recorded for the FM loss and the expert-door probe.

- **D1:** the expert-door probe reproduces `I_traj`'s layer profile (Pearson ≥ 0.95), and
  the log-ratio of the expert-door to the head-door probe has its change point at layer 23.
  *Then the step belongs to the interface, not to the two objectives.* If the probes show
  no step, the step is about what the FM loss wants.
- **D2:** the head-door probe reproduces `I_CoC`'s layer profile (Pearson ≥ 0.90).
- **D3:** in layers 6–21 each probe ranks Q heads like the real loss through its door
  (ceiling-corrected ρ ≥ 0.8). Together with the orthogonal gradients of R13 that would
  make trunk importance a property of the units, whatever is read out downstream.
- **D4 (where the ports come from):** the bare cache door has *no* port structure (its
  per-cache-layer share is smooth: no layer above twice its neighbours' mean), while the
  expert door reproduces the FM port profile (Pearson ≥ 0.9 over cache layers 16–35). The
  ports are then the expert's, not the VLM's and not the objective's. A stored-data clue
  in that direction, not a result: the expert's attention-output norm jumps at expert
  layer 21 (0.87 → 1.63) and dips at 25, and follows the port profile over cache layers
  16–35 with ρ = +0.65 (MLP) / +0.59 (Q) — it marks where the door opens, not why cache 22
  carries almost nine times what cache 24 does (0.191 vs 0.022 of `I_traj` on the MLP axis).

## Order, cost, outputs

| step | what | GPU |
|---|---|---|
| 0 | kept-set × token-resolved scores | none — done |
| A | 3 arms × anatomy pass, 100 `calib_100` clips | ~20 min each, three cards in parallel |
| D | probe pass, 100 `calib_100` clips | ~15 min, the fourth card, alongside A |
| B | 53 configs × 100 held-out clips (≈ 3.9 s per config per clip with the K denoisings, ≈ 0.8 s without) | ~3.5 GPU-h → ~1 h on four cards |
| C | 61 configs × 50 clips × K = 8 (3.4 s per config per clip, measured on `pathway_e_s*`) + attention census | ~3 GPU-h → ~45 min on four cards |

Ada cards only (4–7), to stay on the architecture of every score used; all eight cards
were idle at the time of writing. Smoke runs of 2–3 clips precede every launch and are
never quoted. All runs write `outputs/<exp_id>/{config.json, metrics.json, summary.txt,
plots/}`; each gets an `analyze_*.py` that prints the gates above with PASS/FAIL. Results
go into `paper/2026-09-20_why-importance-differs.md` (new R14–R17), `figures/` via
`fig_why_differs.py`, and a Korean report under `reports/evaluation/`.

New or changed code: `run_gradient_anatomy.py` (`--mask`, NLL, `--probes`),
`analyze_gradient_anatomy.py` (`--mask`, probe gates), `run_token_ablation.py` +
`analyze_token_ablation.py` (new), `run_pathway2.py` (`--edges`, `--cuts`, edge V3/VV,
attention census) + `analyze_vv_depth.py` (new), a mask converter. `UnitGates`,
`TypedUnitGates`, `PruneMasks` and every shipped runner's default behaviour stay as they are.

## What each outcome does to the paper

| outcome | consequence |
|---|---|
| A1 passes | one sentence: the two-regime structure is a property of the compressed model too |
| B1 + B2 pass | the mechanism paragraph can say the late units the scores disagree on *carry* different functions, and C2 of the contributions ("why one loss cannot see the whole model") gets its causal footing at unit level |
| B3 passes | "in the trunk the choice of loss does not matter" becomes a measured statement |
| B fails | keep the anatomy as a description of the scores; drop any functional wording |
| C2 as predicted | the trunk paragraph reads: both scores are earned where vision is integrated (layers ~6–17); the peak at 16–20 is hand-over, and the losses part only after it |
| C2 against prediction | the user's hypothesis holds to the peak; the trunk paragraph says so with a causal curve |
| D1 + D4 pass | "the step is structural" is demonstrated rather than argued, and the open question of R11 (why cache 21–23) is narrowed to the expert's weights |

## Limits

- B selects units on `calib_100` and tests on held-out clips, but with one model and one
  selection size; a dose–response over k is left out unless B is ambiguous.
- A uses the dense model's text for every arm; the shipped importance protocol uses each
  model's own rollout. The preliminary stored-run check used own rollouts and found the
  same change point, so the two are expected to agree.
- C's nested curves measure need *given* that every later (or earlier) layer is blocked
  too; successive differences are conditional, not additive, contributions. The two
  families bracket the answer rather than pin it.
- C's predictions were written after reading the stored 9-layer map; they are informed
  predictions, not blind ones.
- D's probes are random linear readouts. They say what a generic readout sees, not what
  every possible loss would.
- First-order scores throughout; B and C are the only causal parts.
