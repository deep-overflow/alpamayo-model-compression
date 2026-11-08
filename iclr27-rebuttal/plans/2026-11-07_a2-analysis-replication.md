# Plan: the paper's analyses on Alpamayo 2 Super (2026-11-07)

User ask: "가능하면 alpamayo2-super에서 분석 다시 해줘" — redo the analyses on Alpamayo 2 Super
(34B: Qwen3-VL-32B-shaped VLM, 64 layers, Q 64 / MLP 25,600; expert 64 layers, Q 16 / MLP 6,144).

## What already exists (no rerun needed)

The A2 importance run `alpamayo2-distill/outputs/a2_importance_v1` (2026-09-27) used the same estimator as
A1.5's `run_importance.py` on the **same calib_100 clips**, with integrity gates G0a–e passed
(`alpamayo2-distill/reports/2026-09-27_a2-dual-importance.md`). From it:

- Finding 1 (rank agreement by depth, token-position anatomy) — pre-registered R1–R9, same report.
- The seven `paper-analysis/figures/1_main` figures, A1.5 and A2 side by side —
  `alpamayo2-distill/figures/a2_vs_a15/` (`experiments/importance/paper_figs_a2.py`).
- Finding 2 (expert step stability, mass concentration) — printed by the same script.

Re-measuring importance on GPU would reproduce the same file, so this plan does not.

## What is new: the cvpr27 depth-structure analyses on A2 (CPU, stored scores)

`cvpr27/code/{analyze_layer_concentration,plot_depth_structure}.py` (2026-10-29/30) found that the A1.5
VLM MLP is flat except layers 1–4, which are expert-like (Gini 0.89–0.97), objective-agreeing and
calibration-stable. Not yet looked at on A2.

Script `iclr27-rebuttal/code/a2_depth_structure.py`, both models with one code path, VLM only:

| readout | definition (per layer, per objective I_traj / I_CoC) |
|---|---|
| Gini, b50, n50, n90, cut | as `analyze_layer_concentration.py`; cut = mass in the bottom round(0.3986 n) units |
| traj∩CoC top-k overlap | top 0.26% of units (A1.5 32 of 12,288 → A2 67 of 25,600) |
| Spearman ρ(I_traj, I_CoC) | within layer (Finding 1's measure) |
| split-half stability | share of one half's top-k (0.52%: 64 → 133) also top-k in the other half; 100 clips split 50/50, 20 random splits (seed 0), median |

Stability changes from the cvpr27 version (five independent calibration draws) to split-half because A2 has
one draw only; both models get the split-half version so they are read with the same ruler.

Last layer excluded from every I_traj readout and from ρ (I_traj ≡ 0).

**Band mapping (relative depth, boundaries scaled by 64/36):** A1.5 0 / 1–4 / 5 / 6–21 / 22–34 →
A2 0 / 1–7 / 8–9 / 10–38 / 39–62. Trunk/late 0–38 / 39–62 matches the 2026-09-27 report.

## Integrity checks (before reading A2)

- G1: the script reproduces cvpr27's A1.5 numbers — layer 1–4 Gini 0.89–0.97, n50 40–151, cut 0.1–1.8%,
  top-32 overlap 23–28.
  - *Amendment (first run, before reading A2):* G1 failed on n50 only (got 40–195). Every per-layer value matched
    `outputs/cvpr27/layer_concentration/summary.txt` exactly; "40–151" in the memory note is the I_traj column only
    (I_CoC layer 4 = 195). G1 now checks the per-layer n50 of both objectives against that file
    (traj 86/40/48/151, CoC 92/45/79/195), which is stricter than the range.
- G2: band-mean ρ reproduces the 2026-09-27 report — A1.5 Q 0.782 / 0.335, MLP 0.753 / 0.339;
  A2 Q 0.873 / 0.537, MLP 0.687 / 0.378.
- G3: `importance.npz` = mean of `importance_perclip.npz` (checked 2026-11-07: max rel err ≤ 2.3e-7, both models).

## Replication criteria (fixed before looking at A2's per-layer numbers)

- **Early concentration replicates** if A2 has a contiguous run of ≥ 2 layers within its first 15% of depth
  (layers 1–9) where per-layer MLP Gini ≥ 0.85 under both objectives, while the 10–38 band median is ≤ 0.5.
- **Objective agreement there replicates** if the traj∩CoC top-k overlap in those layers is ≥ 0.6 (A1.5 0.72–0.88)
  and above the 10–38 median.
- **Stability there replicates** if split-half top-k kept in those layers exceeds the 10–38 median.
- Pooled vs per-layer: report A2's pooled VLM MLP Gini next to the per-layer profile.

Outputs: `iclr27-rebuttal/outputs/a2_depth_structure/{config,metrics}.json, summary.txt, depth_structure_a15_a2.{png,pdf}`;
report `iclr27-rebuttal/reports/2026-11-07_a2-analysis-replication.md` that also gathers the Finding 1/2 A2 numbers.

## Added 2026-11-08 (user): paper Fig. 3 and Fig. 5 on A2, recomputed with our own code

The September side-by-side renders exist, but Fig. 3(a) there is **raw** ρ while the paper PDF plots
**ceiling-corrected** ρ ("Corrected ρ"), and both were produced by one script in another repo. So both figures
are recomputed here from the stored per-clip / per-step gradients:

- `iclr27-rebuttal/code/a2_fig3_token_type.py`: Fig. 3 (a) corrected same-position agreement
  (`paper_same_position.py`'s split_half/corrected; Q 50 / MLP 10 splits, seed 0), (b, c) additive token shares;
  Q heads (paper) and MLP (supplement twin).
  - G4/G5: late-band corrected means within 0.03 of A1.5 paper text (0.38 / 0.79 / 0.81 / 0.41 / 0.94) and the
    A2 09-27 report (0.58 / 0.71 / 0.73 / 0.56 / 0.92); the rng order differs from the stored series.
  - G6: raw late means within 0.006 of the September prints.
  - G7: A1.5 token shares within 0.006 of the paper text (MLP 6–17 vision 0.90 / 0.75, text 0.07 / 0.24;
    27–34 I_CoC text 0.98, I_traj 0.26 / 0.66 / 0.08; Q 27–34 I_traj 0.35 / 0.49 / 0.16).
- `iclr27-rebuttal/code/a2_fig5_expert.py`: Fig. 5 (a, b) step-pair Spearman averaged over layers, (c) C(r) of
  Eq. 4–5. G8: A1.5 equals the paper (0.668 / 0.874, C(0.9) 68.5% / 1.8%) within 0.0006.

## Not in scope (GPU; needs a separate decision)

Held-out replication on A2 (val100 / OOD100), causal band ablations, pruning A2 with MoOP. ~51 min per 100 clips on
four 48 GB cards for importance; pruning/eval far more.
