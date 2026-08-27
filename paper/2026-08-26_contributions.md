# Paper Preparation: Experiment Summary and Contribution Statement (2026-08-26)

> Sources: all 23 reports in `reports/evaluation/` + 6 key reports in `reports/archive/` +
> `outputs/*/summary.txt` + project memory. Every number below is copied verbatim from the
> corresponding report/table.
> Companion documents: `paper/2026-08-26_draft.md` (English paper draft),
> `reports/evaluation/2026-08-19_results_tables.tex` (7 camera-ready tables),
> `reports/evaluation/2026-08-11_baseline_table.tex` (protocol + baseline anchor).

---

## 1. Full experiment map (by track)

### T1. Criterion anatomy — the core of the paper
- **Matched-budget one-factor design**: the five `*_u40_v2` arms (traj / coc / j / dual /
  jtraj) all remove exactly −2,657,452,032 params (−24.0%); per layer Q 19/32 and
  MLP 7390/12288 kept; expert and KV untouched. Uniform ratio 0.3985632694 (back-solved
  from integrated_mag's realized budget).
- Open loop (frozen protocol, minADE@6 mean, test 500): baseline 0.842 → dual **0.950** <
  jtraj 1.028 < traj 1.008 < coc 1.584 < j 2.158.
- Closed loop (150 scenes x 2, public_2601): dual **0.828 (+0.0787\*)** > jtraj 0.791
  (+0.042) > traj 0.783 (+0.033) > baseline 0.750 > coc 0.660 (−0.089\*) > j 0.536
  (−0.214\*). Criterion alone spans a **0.293** range at the same budget.
- **Combination is synergy, not a trade-off**: dual beats both of its components on every
  set (dual−traj pooled −0.0612\*; still −0.1081\* under teacher-forcing → the trajectory
  head itself is cut differently).
- **j alone**: best reasoning preservation (ref-CoC NLL *better* than baseline, −0.019\*;
  closed-loop degen 1.4%) yet 4.3x the at-fault collision rate → **CoC health is not a
  safety proxy**.
- **Open-loop minADE is not a driving proxy**: dual/jtraj/traj are worse open loop yet
  better closed loop (sign reversal). dualr imitates the GT path best (d2gt 2.97, best
  open loop) yet is worse than dual closed loop (−0.036\*).

### T2. Label-free criterion (J-lens) — a one-factor exchange
- jtraj vs dual (kept overlap 84.8/83.3%): gives up +0.028 m trajectory (p=9.7e−8) and
  buys back the reasoning channel (nll_gtcoc −0.031\*, degen 3.0→1.6% open / 2.7→0.8%
  closed). Gate C1 FAIL = "replacement" rejected; confirmed as an *exchange*.
- Label-axis clarification: no criterion uses human labels; what I_CoC needs is a
  rollout (a generation pass), and J removes even that.

### T3. Calibration — a bigger factor than the criterion
- Factor sizes (test 500 paired ΔminADE): OOD calibration +0.1905\* > the 24% pruning
  itself +0.0955 > **same-distribution redraw +0.0941\*** > mixing 100 OOD clips into 200
  +0.0838\* > 4-bit quantization +0.0816 > criterion choice +0.0707.
- LingoQA axis: swapping only the calibration images to the evaluation domain lifts
  CoC-only 30.2 → 65.4 (**+35.2pp, 97% of the gain**); swapping the objective (CoC→VQA)
  adds +1.0pp. The real utility of the combined criterion = robustness to calibration
  choice.

### T4. Pre-registered negative results (all gates rejected)
- Using score magnitudes: measured single-unit removal damage has split-half reliability
  ≈ 0; sub-additivity 1/5–1/55.
- Iterative recalibration (it3): doubles the cost of the same budget (+0.0302\* test;
  replicated on all 3 sets).
- Second-order (Fisher) score: worse on all 3 sets, significant on 2.
- Global allocation search (dualg): no gain (n.s. on all 3 sets); transplanting Týr's
  searched allocation likewise.
- Combination operator: rank-sum ≡ max (−0.001 n.s.); rank-product significantly worse
  (+0.142\*).
- Ratio knee: between 24% and 33%. u55 zero-shot: test 4.264 (5.1x), degeneracy 71%.

### T5. External baselines (same budget)
- Wanda: test 2.975, degen 87.6%, LingoQA 9.2 — collapse.
- Týr-the-Pruner: with OSSCAR reconstruction + global search, open-loop trajectory ties
  dual (n.s. on 3 sets) — but LingoQA 34.2 = at/below the constant-answer floor (37.0);
  closed loop 0.786, −0.042\* vs dual (borderline); closed-loop degen 5.9%, all empty
  output. Cost: 41 GB supernet + ~2 GPU-h vs one backward pass for dual.
- RAC diagnosis (layer reconstruction x calibration stream): a prefill-only Hessian makes
  the decode path *worse than no reconstruction at all* (+0.0197\*); adding the model's
  own CoC at its natural 1.92% token share flips the sign (88% of the effect); 88% of
  decode energy lies outside prefill's top-512 eigenspace. Deployed damp 1e-2 is not
  run-to-run reproducible (2.91% objective drift).

### T6. Quantization and composition
- CoC-only guided W4 bit allocation: significantly better than uniform W4 on all 3 sets,
  indistinguishable from baseline — yet *loses on its own metric* (nll_gtcoc +0.0595\*).
  The entire driving gain flows through the KV-cache numeric path (teacher-forced −0.0152
  = rollout −0.0151).
- Expert W8 is free (+0.001; W8-all removes 9.95 GB). Damage additivity: interaction CI
  includes 0 in both cells.
- Closed loop: dual+W8 **0.813 (+0.0635\*) at 2.35x** — significantly above the unpruned
  model at 2.35x compression. dual+W4 0.768 (+0.018 n.s.) at 4.04x.

### T7. Recovery (KI-LoRA)
- **Recovery erases the criterion**: at u55 the zero-shot gap of up to +1.78 collapses to
  ±0.05 after recovery (pre-registered H0 accepted).
- 33% cut + recovery ≥ 24% zero-shot (d5: pooled −0.053\* vs dual_u40, p=0.026).
- The lever is **data diversity** (unique scenes 1,200 → 7,750): the epoch-matched
  control (e3) fails. Mechanism is the VLM's language generalization (held-out CE −26%),
  not the expert (FM loss unchanged).
- Closed loop recover_u55: 0.823 (+0.0736\*), degeneracy 0.000.
- Step axis: the expert-Taylor defect is the |Σ_s| aggregation (znorm takes r25 from
  +0.0977 to +0.0003); it does not transfer to the VLM (znorm +0.79 regression) —
  U-shaped vs monotone step-mass profiles.

### T8. Mechanism (pathway maps)
- Stage 1 (expert←cache knockout): per-token causal damage of the generated CoC (median
  10 tokens) = **43x** prompt text, 16x vision; the sink is causally exactly 0; only the
  front-tele camera is individually significant.
- Stage 2 (VLM-internal edges): 14 language-only cells vs **0** action-only cells
  (one-directional dissociation); action channel lives in L0–17, language in L18–35;
  cross-frame (temporal integration) +64% vs cross-camera n.s.; the sink is 0 as cache
  *content* but +59% when removed *inside* the VLM.
- Unit-level dissociation: j-only preserves reasoning while destroying trajectory
  (+187%) → the two abilities live in different units.

### T9. Measured efficiency (honest accounting)
- masked ≡ removed verified (fp64-exact). Peak memory −29%, prefill −26%, but end-to-end
  latency 1.03x (predicted 1.38x) — decode/expert are kernel-dispatch-bound. Only pruning
  reduces FLOPs/memory; quantization numbers are fake-quant projections.

### T10. Evaluation methodology (a contribution in itself)
- Distribution-matched eval sets (val500 / test500 / OOD 1,533 + calib_100, pairwise
  disjoint), clip-derived paired seeds, bit-exact determinism within a GPU architecture
  (3–4% of clips change CoC text across architectures), frozen protocol (2026-08-19:
  rollout-only, minADE@6/minFDE@6 means), closed-loop power design (N=150 → δmin 0.080),
  LingoQA constant-answer floor 37.0, CoC-probe blind floor, degeneracy metrics aligned
  across open and closed loop.

---

## 2. Contribution statement (updated 2026-08-26 — dual is THE method)

**Framing**: `dual` — pruning guided jointly by the FM (flow-matching, action) loss and
the CE (cross-entropy, reasoning) loss — is the paper's method, and the central message
is that **both losses are necessary**: either alone fails on both channels.
Draft title: "One Loss Is Not Enough: Dual-Objective Structured Pruning of a Reasoning
Vision–Language–Action Model".

**The argument, in five steps** (each mapped to evidence in the draft):
1. *Disjoint substrates* (Sec. 7): reasoning and action dissociate one-directionally
   across units, layer bands (action L0–17 / language L18–35), and cache pathways →
   each loss's gradient is structurally blind to part of the network (L35 = FM-blind).
2. *Single-loss pruning therefore fails — on both channels, including its own*
   (Sec. 5, Finding 2): FM-only collapses language to the floor AND is not the best
   trajectory config; CE-only degenerates its own channel (14.2%) AND drives below the
   unpruned model.
3. *The union is synergy, not compromise* (Findings 1–2): dual beats both components
   everywhere; survives teacher-forcing → better trajectory-unit selection, not just
   better text; only −24% arm above the unpruned model closed loop (+0.079\*).
4. *The benefit lives entirely in the keep-if-either union* (Sec. 6.2): rank-sum ≡ max,
   rank-product fails; magnitudes, iteration, curvature, allocation search all add
   nothing — so the message is "use both losses", not "use our operator".
5. *Both losses are free* (Sec. 3): FM needs only logged ego-motion; CE needs only the
   model's own rollout — no human labels; one generation + one backward per calibration
   clip. Bonus: dual is robust to calibration-domain shift (2.4pp vs 35.2pp swing).

Corroborations across axes (Sec. 6): single-channel-guided quantization taxes the other
channel (even the guiding one); prefill-only reconstruction (no CE decode stream) kills
language, fixed by 1.9% CoC tokens. Scope honesty: the claim is for the zero-shot /
light-recovery regime — at the 33% knee, recovery training erases criterion differences
(stated in the draft).

- **C1 (method + necessity by ablation).** Dual-objective rank-max pruning; each
  single-loss criterion fails on both channels incl. its own; union beats both
  components; only −24% arm significantly above the *unpruned* model closed loop
  (+0.079\*). Cost: one rollout + one backward per clip; no human labels.
- **C2 (mechanism).** Why one loss cannot see the whole model: unit/layer/pathway
  dissociation (43x per-token causal density of CoC; 14 language-only vs 0 action-only
  knockout cells; L35 microcosm ~20x damage).
- **C3 (protocol + proxy refutation).** Two channels x two loops; open-loop minADE
  sign-reverses vs closed loop; best GT-path imitator loses closed loop; healthiest
  reasoning arm collides 4.3x more.
- **C4 (what does not help + what dominates).** Six pre-registered negatives;
  calibration set dominates (+35.2pp images; redraw ≈ pruning cost); Wanda/Týr below
  the constant-answer floor; reconstruction-stream diagnosis + the 1.9% CoC fix.

**Secondary results (condensed in main text / appendix)**: quantization composition
(2.35x with a closed-loop win, additivity), recovery (criterion erasure, data diversity,
33%-knee recovery), measured efficiency (dispatch-bound), the step-axis defect.

**Not in this paper**: PBD (in progress, plans/2026-08-24), denoise-step dynamic masks
(closed loop below resolution), front-tele single-camera finding (exploratory, needs a
fresh sample), the speculative-decoding axis (already published elsewhere — cite only).

---

## 3. Remaining gaps (stated as limitations in the draft; close before submission if possible)

| gap | cost | impact |
|---|---|---|
| quantization-only closed loop (w8_all / w4_all) | ~15 h/config | completes the composition decomposition (open loop only now) |
| tyr_uniform closed loop (isolates the global-search factor) | ~7.5 h | strengthens the Týr section |
| 2–4 extra in-dist calibration control draws | ~1.4 h/arm | turns +0.0941 into an interval estimate |
| dualr (reconstruction) recovering LingoQA above the floor (rac_u40) | needs approval | completes C4's "how to fix it" |
| d5 recovery closed loop | ~8 h | completes the recovery section |
| small human-eval sample (LingoQA length bias) | 50 questions, manual | defends absolute LingoQA numbers |

## 4. Draft files
- English draft: `paper/2026-08-26_draft.md` (title, abstract, full body; tables map to
  results_tables.tex)
- Tables: the 7 tables in `reports/evaluation/2026-08-19_results_tables.tex` can be
  \input as-is
