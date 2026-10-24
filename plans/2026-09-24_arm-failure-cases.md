# Plan: what do the single-criterion arms fail at? (2026-09-24)

## Question

`slim_traj_u40_v2` (trajectory-only criterion) and `slim_coc_u40_v2` (CoC-only criterion) both lose
to `dual` on every set. The manoeuvre breakdown (`outputs/failure_by_situation_v1`, 2,533 open-loop
clips) already says: CoC-only damage is broad and largest in cruise / turns with 10–15% CoC
degeneracy; trajectory-only damage concentrates in decel-stop and turns with language intact; no
manoeuvre shows a cross-over. This plan goes to the sample level: are the clips / scenes where the
trajectory arm fails the *same* ones where the CoC arm fails, and what characterizes each arm's tail?

## Sets and per-sample failure measures (all from stored records, no GPU)

| set | unit | measure of failure (paired vs the unpruned baseline) |
|---|---|---|
| open-loop test500 | clip | Δ minADE = arm − baseline; CoC degenerate flag; Δ nll_dense (arm − baseline) |
| open-loop OOD-val | clip | Δ minADE; Δ minADE_tf (GT CoC teacher-forced); degenerate flag; Δ nll_gtcoc |
| alpasim 150 scenes | scene (mean over rollouts) | Δ score = arm − baseline; offroad / at-fault gates; time out of lane; braking near obstacles |
| alpasim hard100 | scene | same |

Tail = worst decile of Δ within the set (open loop: 50 clips; closed loop: 15 / 10 scenes);
also the "exclusive" tails: in one arm's worst decile and below the median of the other.

## Analyses

1. **Overlap.** Spearman(Δ_traj, Δ_coc) over samples; Jaccard of the two worst-decile sets against
   chance (0.053 for deciles) and against a difficulty null (worst decile of the baseline's own
   minADE / score). High overlap = shared hard samples; low overlap = criterion-specific failures.
2. **Characterize each tail** (tail vs rest, enrichment ratio + Mann–Whitney / Fisher): manoeuvre
   bucket, baseline difficulty (baseline minADE, score), ego speed at t0, GT path length and yaw,
   country, time of day, OOD reasoning cluster (OOD-val), CoC content keywords (stop / yield /
   pedestrian / construction / lane change / traffic light), reference CoC length, CoC degeneracy
   and NLL of the arm (language collapse vs pure trajectory error), closed-loop lateral vs
   longitudinal surrogates and `hard_score` features (v_mean, yaw_total).
3. **Does dual cover both tails?** On the trajectory arm's tail and on the CoC arm's tail, the
   dual arm's Δ (median, share of samples worse than +0.05 m / −0.1 score). If dual is near
   baseline on both tails, the union argument holds at the sample level.
4. **Cross-set consistency.** Do the characterizations agree between test500 and OOD-val, and
   between 150 scenes and hard100 (different scene sets, so overlap can only be by covariate)?

## Pre-registered reading

- Spearman(Δ_traj, Δ_coc) ≥ 0.5 and Jaccard ≥ 2× chance on a set → failures are mostly shared
  hard samples on that set; report what makes them hard (baseline difficulty, manoeuvre).
- Otherwise → criterion-specific tails; report the covariates enriched ≥ 1.5× (with CI) in each.
- Language: if ≥ 50% of the CoC arm's worst-decile clips are CoC-degenerate, its trajectory tail
  is a language-collapse tail (consistent with the 2026-09-21 causal report); the trajectory arm's
  tail is expected to be non-degenerate.

## Outputs

`outputs/arm_failure_cases_v1/{config.json,metrics.json,summary.txt,plots/}`, a note
`paper-analysis/2026-09-24_arm_failure_cases.md` (tables + Korean summary), and the scripts
`paper-analysis/scripts/failure_cases/` (stored records only).

## Data

open loop: `outputs/{baseline_ada,coc_u40_v2,traj_u40_v2,dual_u40_v2}_{test,ood}/*.json` (rows: minADE_rollout, coc_degenerate, gen_coc, nll_self; ood adds minADE_tf, nll_gtcoc, cluster); closed loop: `alpasim-runs/{m2601,h100}_merged_{baseline,slim_*_u40_v2}/aggregate/results-summary.json`; scene features `outputs/scene_difficulty/scene_difficulty_public2601.csv`; importance `outputs/importance_v2/importance_perclip.npz`; kept sets `outputs/slim_*_u40_v2/slim_meta.json`.

## Result: see `paper-analysis/2026-09-24_arm_failure_cases.md` (2026-09-24)
