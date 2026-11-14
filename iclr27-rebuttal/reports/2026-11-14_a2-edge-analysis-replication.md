# Experiment: paper analyses (Appendix C.4 panels) on Alpamayo 2 Edge

2026-11-14. Plan: `iclr27-rebuttal/plans/2026-11-14_a2-edge-analysis-replication.md`. First step of "compare MoOP,
FM-only and CE-only on Alpamayo 2 Edge".

## Purpose

The six Appendix C.4 panels exist for Alpamayo 1.5 and Alpamayo 2 Super: Fig. 3 (a-c) for the VLM and Fig. 5 (a-c) for
the expert. This experiment recomputes them on Alpamayo 2 Edge.

Edge (2.65 B parameters) is built on Cosmos3-Edge: a Nemotron-based text tower, SigLIP2 vision, and a relu^2 MLP. It is
the first model in this study outside the Qwen3-VL family. The three questions:

1. Does Finding 1 hold: do I_FM and I_CE agree before a depth step and disagree after it?
2. Is late-layer I_CE carried by text positions (A1.5) or by vision (A2 Super)?
3. Expert: are Q-head rankings less stable across denoising steps than MLP rankings? Is MLP importance concentrated?

## Setup

**Model.** `nvidia/Alpamayo2-Edge` at revision `f8951d0`.

| tower | layers | hidden | Q heads | KV heads | MLP |
|---|---|---|---|---|---|
| VLM | 28 | 2048 | 16 | 8 | relu^2, 9,216 channels |
| expert | 28 | 512 | 16 | — | 3,840 channels |

**Data.** The calib_100 clips at t0 = 5.1 s, the same clips as A1.5 and A2 Super.
- The new Edge input cache `physicalai_av/pre_processed/calib_edge` was streamed from the Hub through Edge's own loader.
- Each clip has 6 cameras and 22 frames (cameras 1 and 6 at −2/−1/−0.5/−0.1/0 s, the other four at −1/−0.1/0 s), 20
  history poses and 64 GT future poses.
- 100/100 clips were built.
- The GT future and the last 16 history poses equal `calib_a2` exactly for all 100 clips (max |diff| 0.0).

**Prompt.** Edge's driving prompt with the `Navigation input is: ...` line removed (user decision). This matches the A2
Super runs, which had no navigation line. Everything else is Edge's own: system prompt, camera and time labels,
ego-history tokens, task sentence, and a closed `<think></think>`.

**Estimator.** The protocol of `run_importance_a2.py` (a2_importance_v1), ported in `code/edge_grad_lib.py` and
`code/run_importance_edge.py`:
- Rollout: one CoC per clip with release sampling (top_p 0.98, T 0.6, max 256), seed `sha256("42:clip")[:4]`.
- One teacher-forced VLM forward over prompt + CoC.
- I_CE: the CoC NLL backward.
- I_FM: a 10-step FM loss through the expert onto detached cache leaves (t_s = (s + 0.5)/10, x_t = t x1 + (1 − t) eps,
  target x1 − eps), followed by one VLM backward.
- The expert gates are read after every denoising step.
- Gate locations:
  - Q heads: input of `self_attn.o_proj`.
  - MLP channels: input of `mlp.fc2`. Edge has no `down_proj`.
- Token types follow the A2 rule. Without a navigation line, prompt text = system prompt, camera and time labels, and
  task sentence.

**Hardware.** One RTX 5880 Ada (card 7). Strict deterministic mode (`use_deterministic_algorithms(True)`, deterministic
flash backward). About 6 s per clip; 100 clips took 10.4 min, from 19:23 to 19:34 KST. Peak memory was 17.5 GB.

**Analysis.**
- `code/edge_paper_analysis.py` imports the functions the A1.5 / A2 numbers came from:
  - `a2_fig3_token_type.agreement / shares`
  - `analyze_importance_a2.ratio_block` (the step tau)
  - `a2_fig5_expert.step_matrix / cum_curve`
- `code/supp_alpamayo2_edge_figs.py` draws the panels with `supp_alpamayo2_figs.py`'s own panel functions.
- Bands are moved from A1.5 by relative depth with `analyze_importance_a2.spec(28)`: grey late band 17–27, token-share
  lo 5–13, token-share hi 21–26.
- Fig. 3(a) and the band tables use the step-aligned bands (pre-step 0..tau−1, post-step tau..26), per trap 2 of the
  A2 replication.

**Gates.** All passed.

| gate | what it checks | result |
|---|---|---|
| GA | imported agreement / shares reproduce the stored A2 Q-head series | exact (atol 1e-12) |
| GB | imported step_matrix / cum_curve reproduce the stored A2 expert numbers | exact |
| GC | `ratio_block` on a2_importance_v1 gives the tau the A2 panels use | 51 / 50, as used |
| GD | Edge run: all 100 CoCs close with `<\|traj_future_start\|>`; teacher-forced rope deltas = generation rope deltas; NLL and FM loss finite | pass |
| G0 | typed Q gates summed over types vs plain gates | max relative error 6.1e-6 |
| G0b | per-step expert gradients summed vs accumulated | 0.0 |
| G-rollout | the importance rollout equals the release generation, token for token, same seed | 3/3 smoke clips |
| G2 | one clip run twice | bitwise identical arrays |

Note on G2: without strict mode the SDPA backward is not deterministic. Arrays then differed by up to 1.7% of their
maximum between runs.

**Outputs.**
- `outputs/iclr27-rebuttal/edge_importance_v1/` (importance, anatomy, step importance, per-clip records)
- `outputs/iclr27-rebuttal/edge_paper_analysis/` (metrics.json, summary.txt)
- `outputs/iclr27-rebuttal/supp_alpamayo2_edge_figs/` (previews)
- Panels: `paper/fig/assets/supp_alpamayo2_panels/edge_*.pdf`
- Three-row composites: `paper/fig/assets/supp_alpamayo2_{vlm,expert}_3row.pdf`
- The paper's two-row PDFs are untouched.

## Results (Facts)

### CoC on the no-nav prompt (100 clips)

| | value |
|---|---|
| CoC = `<NOMINAL>` (adaptive reasoning: no reasoning text) | **48 clips** |
| reasoning CoC | 52 clips, mean 22.1 tokens |
| CoC length, all clips | mean 14.4 tokens |
| CoC NLL | nominal 0.048, reasoning 0.329, all 0.194 |
| FM loss | mean 0.296, median 0.174 |
| smoke, release path, 3 clips | minADE@6 6.4 s mean 0.61 m (sanity only, not an evaluation) |

### Depth step: two-level fit of log(Σ I_FM / Σ I_CE)

tau is the first layer of the low level.

| model | layers | tau Q / MLP | relative depth | step (FM/CE ratio falls by) | R² Q / MLP | bootstrap tau mode share |
|---|---|---|---|---|---|---|
| A1.5 | 36 | 23 / 23 | 0.66 | 4.6× / 4.6× | 0.86 / 0.93 | — |
| A2 Super | 64 | 51 / 50 | 0.81 / 0.79 | 3.1× [2.8, 3.4] / 3.3× [3.0, 3.7] | 0.72 / 0.81 | 0.58 / 1.00 |
| **A2 Edge** | 28 | **22 / 22** | **0.81** | **9.3× [7.4, 11.1] / 10.6× [8.4, 12.8]** | 0.82 / 0.86 | 0.96 / 0.99 |

Every Edge clip has the step: the early level is above the late level in 100/100 clips. The A1.5-based tau window
(16–20) does not contain Edge's tau, as it did not contain A2's.

### Fig. 3(a): I_FM vs I_CE rank agreement

Corrected Spearman ρ, with raw ρ in parentheses, step-aligned bands. No layer has a NaN corrected value. Edge Q-head
self-agreement (split-half) is 0.7–0.9 in most layers (FM 0.36 and CE 0.62 at layer 3).

**Q heads**

| model | band | layers | All | Vision | Prompt text | CoC | Ego history |
|---|---|---|---|---|---|---|---|
| A1.5 | pre-step | 0–22 | 0.84 (0.77) | 0.95 | 0.87 | 0.47 | 0.99 |
| A1.5 | post-step | 23–34 | **0.36** (0.31) | 0.77 | 0.80 | 0.41 | 0.94 |
| A2 Super | pre-step | 0–50 | 0.88 (0.84) | 0.92 | 0.93 | 0.77 | 1.00 |
| A2 Super | post-step | 51–62 | **0.37** (0.33) | 0.61 | 0.60 | 0.57 | 0.86 |
| **A2 Edge** | pre-step | 0–21 | **0.49** (0.43) | 0.96 | 0.80 | 0.48 | 0.99 |
| **A2 Edge** | post-step | 22–26 | **0.27** (0.25) | 0.81 | 0.25 | 0.16 | 0.86 |
| A2 Edge, reasoning clips (n = 52) | pre-step | 0–21 | 0.37 (0.32) | 0.96 | 0.79 | 0.57 | 0.99 |
| A2 Edge, reasoning clips (n = 52) | post-step | 22–26 | 0.23 (0.22) | 0.88 | 0.32 | 0.28 | 0.91 |

The Edge Q-head "All" series swings from layer to layer, from −0.03 to 0.92. Examples: layer 0 0.92, layer 3 −0.01,
layer 6 0.09, layers 8–9 0.83 / 0.81, layer 24 −0.03.

**MLP channels**

| model | band | layers | All | Vision | Prompt text | CoC | Ego history |
|---|---|---|---|---|---|---|---|
| A1.5 | pre-step | 0–22 | 0.91 (0.74) | 0.97 | 0.83 | 0.76 | 0.98 |
| A1.5 | post-step | 23–34 | **0.53** (0.33) | 0.70 | 0.73 | 0.37 | 0.82 |
| A2 Super | pre-step | 0–49 | 0.86 (0.64) | 0.85 | 0.90 | 0.85 | 1.01 |
| A2 Super | post-step | 50–62 | **0.43** (0.30) | 0.40 | 0.66 | 0.45 | 0.84 |
| **A2 Edge** | pre-step | 0–21 | **0.87** (0.83) | 0.98 | 0.95 | 0.88 | 1.00 |
| **A2 Edge** | post-step | 22–26 | **0.71** (0.66) | 0.94 | 0.78 | 0.75 | 0.96 |

The Edge MLP "All" series is 0.91–0.95 in layers 0–14. It falls to 0.88 at layer 15, 0.81 at 16 and 0.78 at 17, then
stays at 0.67–0.78 through layer 26.

### Fig. 3(b, c): token-type share of layer importance

Bands: lo = A1.5 6–17 / A2 11–30 / Edge 5–13; hi = A1.5 27–34 / A2 48–60 / Edge 21–26.

**Q heads**

| model | objective | band | vision | prompt | CoC | ego |
|---|---|---|---|---|---|---|
| A1.5 | FM | lo | 0.72 | 0.13 | 0.05 | 0.10 |
| A1.5 | FM | hi | 0.35 | 0.17 | 0.32 | 0.16 |
| A1.5 | CE | lo | 0.63 | 0.16 | 0.18 | 0.03 |
| A1.5 | CE | hi | 0.02 | 0.36 | 0.62 | 0.00 |
| A2 Super | FM | lo | 0.50 | 0.35 | 0.05 | 0.10 |
| A2 Super | FM | hi | 0.50 | 0.26 | 0.18 | 0.06 |
| A2 Super | CE | lo | 0.73 | 0.16 | 0.10 | 0.01 |
| A2 Super | CE | hi | 0.35 | 0.21 | 0.44 | 0.00 |
| **A2 Edge** | FM | lo | 0.73 | 0.15 | 0.07 | 0.05 |
| **A2 Edge** | FM | hi | 0.27 | 0.41 | 0.21 | 0.12 |
| **A2 Edge** | CE | lo | **0.37** | 0.12 | **0.47** | 0.04 |
| **A2 Edge** | CE | hi | **0.04** | 0.39 | **0.56** | 0.01 |

**MLP channels**

| model | objective | band | vision | prompt | CoC | ego |
|---|---|---|---|---|---|---|
| A1.5 | FM | lo | 0.90 | 0.04 | 0.03 | 0.03 |
| A1.5 | CE | lo | 0.75 | 0.09 | 0.14 | 0.01 |
| A1.5 | CE | hi | 0.02 | 0.70 | 0.28 | 0.00 |
| A2 Super | CE | hi | 0.75 | 0.15 | 0.09 | 0.00 |
| **A2 Edge** | FM | lo | 0.76 | 0.13 | 0.07 | 0.04 |
| **A2 Edge** | FM | hi | 0.37 | 0.30 | 0.25 | 0.08 |
| **A2 Edge** | CE | lo | 0.45 | 0.11 | 0.42 | 0.02 |
| **A2 Edge** | CE | hi | **0.11** | 0.27 | **0.61** | 0.01 |

On the 52 reasoning clips alone, the Edge CE share (vision / prompt / CoC) is:

| axis | lo band | hi band |
|---|---|---|
| Q heads | 0.28 / 0.07 / 0.62 | 0.02 / 0.14 / 0.84 |
| MLP channels | 0.36 / 0.08 / 0.55 | 0.05 / 0.12 / 0.81 |

### Fig. 5: expert, per-step FM importance

Columns: mean step-pair ρ, minimum step-pair ρ, C(0.9) = share of a layer's importance held by its least important 90%
of units (mean over layers), and ρ by step distance |s − s'| = 1 .. 9.

| model | axis | (S, L, U) | mean ρ | min ρ | C(0.9) | ρ by distance 1 .. 9 |
|---|---|---|---|---|---|---|
| A1.5 | Q | (10, 36, 16) | 0.668 | 0.138 | 68.5% | 0.88 0.81 0.73 0.65 0.57 0.49 0.39 0.29 0.14 |
| A1.5 | MLP | (10, 36, 8256) | 0.874 | 0.739 | 1.8% | 0.92 0.90 0.88 0.87 0.85 0.84 0.81 0.79 0.74 |
| A2 Super | Q | (10, 64, 16) | 0.673 | 0.209 | 66.0% | 0.89 0.81 0.73 0.65 0.57 0.49 0.39 0.29 0.21 |
| A2 Super | MLP | (10, 64, 6144) | 0.828 | 0.661 | 1.7% | 0.90 0.87 0.84 0.82 0.80 0.77 0.74 0.70 0.66 |
| **A2 Edge** | Q | (10, 28, 16) | **0.614** | 0.220 | **62.4%** | 0.84 0.74 0.66 0.58 0.51 0.42 0.33 0.26 0.22 |
| **A2 Edge** | MLP | (10, 28, 3840) | **0.885** | 0.762 | **12.5%** | 0.95 0.93 0.90 0.88 0.85 0.83 0.80 0.77 0.76 |

Edge expert C(0.9) per layer (min / median / max): Q 0.17 / 0.67 / 0.81, MLP 0.00 / 0.10 / 0.38.

The Edge Q step matrix has two blocks:

| step pairs | mean ρ |
|---|---|
| within steps 0–6 | 0.77 |
| within steps 7–9 | 0.74 |
| between the two blocks | 0.44 |

The MLP has no such break: ρ between steps 0–6 and 7–9 is 0.85.

### Figures

- Panels: `paper/fig/assets/supp_alpamayo2_panels/edge_{rank_correlation, share_fm, share_ce, steps_q_head, steps_mlp, mass_curve}.pdf`.
- Composites: `paper/fig/assets/supp_alpamayo2_{vlm,expert}_3row.pdf`. Rows are A1.5 / A2 Super / A2 Edge; the A1.5 and
  A2 panels are the existing files, placed unchanged.
- Edge grey band: layers 17–27. Dotted step line: at 21.5.
- One value falls below the shared y-range (−0.25): Edge Prompt text at layer 23 (−0.26).

## Analysis (Opinion)

**What replicates on Edge (outside the Qwen3-VL family).**

- **Late disagreement (the second half of Finding 1).**
  - Post-step pooled agreement is 0.27 for Q heads (A1.5 0.36, A2 0.37) and 0.71 for MLP.
  - The MLP curve has the paper's shape: a 0.91–0.95 plateau, then a drop.
  - The MLP drop is milder than in either Qwen3-VL model (0.53 / 0.43). It also starts a few layers before the fitted
    tau.
- **Late I_CE is carried by text, as in A1.5.**
  - Text share (prompt + CoC) in the hi band is 0.95 for Q and 0.88 for MLP (A1.5 0.98 / 0.98).
  - A2 Super was the exception (0.65 / 0.24). With Edge in hand, "late CE = text" looks like the common case and A2
    Super's vision-carried late MLP the outlier. That is two of three models.
- **Expert.**
  - Q-head rankings are less stable across steps than MLP rankings (0.61 vs 0.89).
  - Q importance is spread over heads (C(0.9) 62%) while MLP importance sits in a minority of channels (12.5%).
  - Both directions hold in all three models.
- **The step position.** Edge's tau sits at relative depth 0.81, the same as A2 Super. "Step at 2/3 depth" is specific
  to A1.5. The safe wording is "a sharp step in the last fifth to third of the tower". The bootstrap puts Edge's tau at
  22 in 96–99% of resamples. Edge's step is also the sharpest of the three: the FM/CE ratio falls 9–11×, against 4.6×
  for A1.5 and 3.1–3.3× for A2 Super.

**What does not replicate, and the likely reason.**

- **Early Q-head agreement is weak on Edge.**
  - Pre-step pooled ρ is 0.49, against 0.84 / 0.88, and the per-layer series is erratic.
  - It is not estimation noise: the per-layer self-agreement is 0.7–0.9 and no layer is NaN.
  - Restricting both scores to one token type restores agreement: vision 0.96, ego 0.99, prompt 0.80. So the pooled
    disagreement is the token-type composition mechanism the paper uses for late layers, appearing here early.
  - The composition differs because Edge's early-layer I_CE draws on CoC positions far more than the other models do:
    CoC share 0.47 of Q and 0.42 of MLP in the lo band, against 0.18 / 0.10 for Q. I_FM draws on vision (0.73).
  - Two Edge properties may contribute. Neither is tested here:
    - Edge's CoC is short and template-like (48% `<NOMINAL>`, reasoning 22 tokens). Its NLL may be predicted from the
      few preceding CoC tokens rather than from the scene.
    - Edge has only 16 Q heads per layer, so a layer-level rank over 16 units moves a lot when a few heads swap.
  - The reasoning-only recomputation does not remove the effect. Pooled pre-step ρ is 0.37 and early CE CoC share 0.62,
    so it is not the `<NOMINAL>` clips' tiny NLLs.
- **Within-type late agreement at text positions is low on Edge.** Post-step prompt is 0.25 and CoC 0.16 (A1.5 0.80 /
  0.41, A2 0.60 / 0.57). "CoC lowest" holds (as in A1.5, unlike A2), but prompt is nearly as low. At late text
  positions FM and CE use different heads even at the same positions.
- **Expert MLP concentration is weaker** (12.5% vs 1.7–1.8%). The median layer is 10%, and one layer reaches 38%.
  Possible reasons, untested: the relu^2 MLP (Qwen3's is gated SwiGLU) or the narrower expert (512 hidden).
- **The two-block Q step structure.** Steps 7–9 form a block distinct from steps 0–6. A1.5 and A2 decay smoothly with
  step distance. It points to the last three denoising steps using different heads.

**For the next step (MoOP vs FM-only vs CE-only on Edge).**

- On A2 Super the union's driving edge over FM-only vanished, and the A2 FM-only never broke CoC. Edge differs from A2
  Super on the paper's mechanism in two ways:
  - FM and CE already rank Q heads differently in early layers.
  - Late CE is text-carried.
- Inference, not tested: FM-only and CE-only will therefore remove more different Q heads on Edge than on A2 Super, and
  the comparison has more room to separate them.
- Before masks:
  - Decide the allocation. A2 used the make_slim per-layer ratio: u40 = 34.1%, u50 = 42.6%. On Edge the same ratio
    removes 6.4 of 16 Q heads per layer, which the paper's layer-0 lesson suggests checking first.
  - Build Edge input caches (no-nav) for the evaluation sets.
- The CE objective on Edge covers few tokens for half the clips. The CE-only arm's ranking rests mostly on the 52
  reasoning clips (reasoning NLL 0.33 vs nominal 0.05).
- Caveats:
  - The no-nav prompt is outside Edge's training distribution. The CoCs mention "follow our route" / "as guided by
    navigation" without a navigation line. Absolute Edge numbers are not comparable with a navigation-conditioned
    evaluation.
  - The post-step band has only 5 layers (22–26).
