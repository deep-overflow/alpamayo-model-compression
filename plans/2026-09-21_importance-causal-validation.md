# Does the score anatomy predict function — and why is the shared profile shaped as it is?

Date: 2026-09-21. Branch: `worktree-importance-causal-validation`.
Status: **awaiting approval — no experiment code written, no GPU used.**
Follows `paper/2026-09-20_why-importance-differs.md` (R1–R13) and
`plans/2026-09-20_gradient-anatomy.md`.

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

## Stage 0 — what each shipped criterion removed, in token currency (CPU, minutes)

`outputs/gradanat_v1/anatomy.npz` holds the dense model's `E|G|` per unit split by the
token type the unit acts on. Cross it with the kept sets of `dual_u40_v2`, `traj_u40_v2`,
`coc_u40_v2` (`slim_meta.json`: 19/32 heads, 7,390/12,288 channels per layer, all built
from `importance_v2`), layers 22–34:

- retained share of `I_traj` mass earned at vision + history tokens, and at text-side tokens;
- retained share of `I_CoC` mass earned at CoC tokens, and at prompt tokens.

**S0 (written before computing it):** the CoC-only criterion retains the least late-layer
`I_traj` mass at vision + history tokens of the three arms; the trajectory-only criterion
retains the least `I_CoC` mass at CoC tokens; `dual` is within 5 pp of the better single
criterion on both. *If `coc` retains as much vision/history `I_traj` mass as `dual`, the
token-mixture account does not explain why CoC-only loses driving.*

## A — the anatomy on the shipped pruned models (GPU ~20 min per arm)

`run_gradient_anatomy.py` gains `--mask` (a `q_mask`/`mlp_mask` npz, installed with
`mask_lib.PruneMasks` before the gates, exactly as `run_importance.py --mask` does — a
masked unit is functionally removed and its gate gradient is exactly zero) and records the
CoC NLL next to the FM loss. Masks come from the three `slim_meta.json` files; the
`*_u40_v2` family changes no weights, so mask ≡ shipped checkpoint.

One deviation from the shipped protocol, deliberate: every arm is teacher-forced on the
**dense** model's rollout (generated in the same process with the masks switched off), so
token positions, text and noise seeds are identical across arms and any difference is the
network. The per-clip FM loss and NLL on that fixed text are then a matched functional
readout for free.

- **A1 (`dual` keeps the structure):** ratio change point at layer 23 on both axes; layer
  profile correlation with dense ≥ 0.98 for both scores; MLP hand-over layers within ±1 of
  dense (`I_CoC` 18, `I_traj` 21); ceiling-corrected agreement at vision tokens among kept
  Q heads, layers 22–34, ≥ 0.70.
- **A2 (single criteria hurt where the account says):** on the fixed dense text, the FM
  loss rises more under `coc` than under `dual` and `traj`; the NLL rises more under `traj`
  than under `dual` and `coc` (paired Wilcoxon over clips, p < 0.01 each).
- Descriptive: port profile is *not* re-measured (it needs 72 backwards per clip); the
  band split from the same pass says whether cache 22–24 is still the main port.

## B — token-targeted ablation: does the score anatomy predict function? (GPU ~1–2 h)

The decisive one. In layers 22–34, per layer, pick k = 6 of 32 heads three ways from the
dense token-resolved scores (`gradanat_v1`, measured on `calib_100`):

- **T-set:** largest (rank of `I_traj` at vision + history tokens) − (rank of pooled `I_CoC`);
- **C-set:** largest (rank of `I_CoC` at CoC tokens) − (rank of pooled `I_traj`);
- **R-sets:** k random heads per layer, 5 seeds (the null);
- **control:** the same three rules applied in layers 6–17, where the scores are interchangeable.

Mask each set on the dense model and measure on **held-out** clips (100 from `indist_500`;
the sets were chosen on `calib_100`), paired with the dense model on the same clip, text
and noise: FM loss (10 steps, GT-anchored), NLL of the dense rollout, and open-loop
minADE@6 for dense / T / C / one R-set. Measurement design as in `run_ablation.py`
(reference CoC generated once, every config teacher-forced on it).

- **B1:** ΔFM(T) > ΔFM(C), and ΔFM(T) above every R-set;
- **B2:** ΔNLL(C) > ΔNLL(T), and ΔNLL(C) above every R-set;
  both by paired Wilcoxon at p < 0.01 → a double dissociation.
- **B3 (control):** in layers 6–17 the T- and C-sets do *not* dissociate: the ratio
  ΔFM/ΔNLL of the two sets differs by less than the spread of the R-sets.
- *Refuted if* T and C damage both channels alike in late layers. Then the token-mixture
  account describes the scores and nothing else, and the paper must say so.

## C — vision–vision interaction by depth (the user's hypothesis; GPU ~1–2 h)

`run_pathway2.py` already knocks out attention edges inside the VLM over a layer window
and reads both channels. Its grid used 9-layer bands, which cannot line up with a
single-layer importance profile. Re-run one edge — vision ← same-camera earlier frames —
over twelve 3-layer windows, plus the unblocked reference (50 clips, K as in the original).

- **C1:** the action damage profile over the twelve windows follows `I_traj` at vision
  tokens (Spearman ≥ 0.6 against the 3-layer means), and is inside the practical-null band
  (±5% of baseline) for every window from layer 24 up.
- **C2:** the same edge damages the CoC NLL with a similar depth profile in the trunk
  (Spearman ≥ 0.6 with C1's profile over windows 0–21): vision–vision interaction is a
  shared substrate, not an action-specific one.
- *If C1 fails*, high action importance is not where temporal vision interaction is, and
  the trunk's importance has to be explained some other way (D says where to look).

## D — is the shared profile a property of the network? (GPU ~20 min)

Replace each loss by a **random readout through the same door** and re-measure the
token-resolved importance:

- head door: `L = Σ_p ⟨r_p, h_final(p)⟩` over the CE positions, `r ~ N(0, I)` fixed by the
  clip seed;
- cache door: `L = Σ_s ⟨R_s, v_θ(x_s, t_s)⟩` over the ten FM steps, same `x_s` as the FM
  loss, `R ~ N(0, I)`.

Neither has any driving or language content.

- **D1:** the cache-door probe reproduces `I_traj`'s layer profile (Pearson ≥ 0.95) and
  its change point at layer 23. *Then the step belongs to the interface, not to the FM
  objective.* If the probe has no step, the step is about what the FM loss wants.
- **D2:** the head-door probe reproduces `I_CoC`'s layer profile (Pearson ≥ 0.90).
- **D3:** in layers 6–21 each probe ranks Q heads like the real loss through its door
  (ceiling-corrected ρ ≥ 0.8). Together with the orthogonal gradients of R13 that would
  make trunk importance a property of the units, whatever is read out downstream.

## Order, cost, outputs

| step | what | GPU |
|---|---|---|
| 0 | kept-set × token-resolved scores | none |
| A | 3 arms × anatomy pass, 100 clips | ~20 min each, three cards in parallel |
| D | probe pass, 100 clips | ~20 min, one card |
| B | ~15 mask configs × 100 held-out clips, forward only, + minADE for 4 configs | 1–2 h over four cards |
| C | 13 configs × 50 clips × K samples | 1–2 h over four cards |

Ada cards only (4–7), to stay on the architecture of every score used. All runs write
`outputs/<exp_id>/{config.json, metrics.json, summary.txt, plots/}`; each gets an
`analyze_*.py` that prints the gates above. Results go into
`paper/2026-09-20_why-importance-differs.md` (new R14–R17) and a Korean report.

## What each outcome does to the paper

| outcome | consequence |
|---|---|
| A1 passes | one sentence: the two-regime structure is a property of the compressed model too |
| B1 + B2 pass | the mechanism paragraph can say the token-specific units *carry* the function, and C2 of the contributions ("why one loss cannot see the whole model") gets its causal footing at unit level |
| B fails | keep the anatomy as a description of the scores; drop any functional wording |
| C1 + C2 pass | "both scores are earned where vision tokens are integrated over time, layers ~9–21" becomes a causal statement |
| D1 passes | "the step is structural" is demonstrated rather than argued |

## Limits

- B selects units on `calib_100` and tests on held-out clips, but with one model and one
  selection size (k = 6); a dose–response over k is left out unless B is ambiguous.
- A uses the dense model's text for every arm; the shipped importance protocol uses each
  model's own rollout. The preliminary stored-run check used own rollouts and found the
  same change point, so the two are expected to agree.
- D's probes are random linear readouts. They say what a generic readout sees, not what
  every possible loss would.
- First-order scores throughout; B and C are the only causal parts.
