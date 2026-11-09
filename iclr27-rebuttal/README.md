# iclr27-rebuttal/

ICLR 2027 rebuttal material (from 2026-11-07, all new results go here).

Layout (same as `cvpr27/`):
- `plans/` — plans written before running
- `code/` — analysis scripts (import shared libs; do not copy them)
- `reports/` — reports (`YYYY-MM-DD_name.md`), figures in `reports/figures/`
- `outputs` — gitignored symlink to `/mnt/nvme1n1/ad_vla/outputs/chan/iclr27-rebuttal` (recreate in each worktree);
  also reachable as repo `outputs/iclr27-rebuttal/`

## Index

| date | topic | plan | code | report | outputs |
|---|---|---|---|---|---|
| 2026-11-09 | Alpamayo 2 Super + ICLR27 MoOP on test500: dense / MoOP / MoOP l0keep / FM-only / CE-only (34.1% removed). MoOP +11.8%* vs dense, = FM-only, beats CE-only (−0.074*); l0keep no effect; CoC intact | `plans/2026-11-09_a2-moop-test500.md` | `code/make_a2_moop_masks.py`, `code/run_a2_openloop.py`, `code/run_a2_worker.sh`, `code/run_a2_arms.sh`, `code/analyze_a2_test500.py` | `reports/2026-11-09_a2-moop-test500.md` | `outputs/a2_moop/`, `outputs/a2_test500_*/`, `outputs/a2_test500_analysis/` |
| 2026-11-09 | OOD-train (1,271 clips) open loop + reference-CoC NLL for the 8 ICLR27 arms on KISTI Hangang GH200 — **approved, code ready, waiting for the OTP sockets** | `plans/2026-11-09_oodtrain-iclr27-arms-gh200.md` | `code/push_oodtrain_gh200.sh`, `code/launch_oodtrain_gh200.sh`, `code/pull_oodtrain_gh200.sh`, `code/analyze_oodtrain_gh200.py`; `cvpr27/code/eval_a100.sbatch` (arms + `EXP_ROOT`) | — | `outputs/iclr27-rebuttal/gh200/` |
| 2026-11-08 | Closed loop, ICLR27 (pre-l0keep) arms on Origin150 / Hard100: MoOP, FM-only, CE-only, Wanda, Tyr, LLM-Pruner (gathered from the 11-07 CI report) | — | — | `reports/2026-11-08_closedloop-iclr27-arms.md` | `outputs/cvpr27/closedloop_ci_2026-11-07/` |
| 2026-11-08 | Combination-rule ablation, all existing results gathered (l0keep A100/Ada + pre-l0keep, layer-0 confound flagged) | — | — | `reports/2026-11-08_combination-rule-summary.md` | — |
| 2026-11-08 | Paper analyses on Alpamayo 2 Super (34B): Finding 1/2, Fig. 3 (token types), Fig. 5 (expert), VLM MLP depth structure | `plans/2026-11-07_a2-analysis-replication.md` | `code/a2_depth_structure.py`, `code/a2_fig3_token_type.py`, `code/a2_fig5_expert.py`, `code/a2_rank_band_review.py` (review: step-aligned bands) | `reports/2026-11-08_a2-analysis-replication.md` | `outputs/a2_depth_structure/`, `outputs/a2_fig3_token_type/`, `outputs/a2_fig5_expert/`, `outputs/a2_rank_band_review/` |
| 2026-11-08 | Reviewer Q3: what the earlier 50-clip calibration set (12 test clips) was used for — pilot importance only; every paper arm re-measured on Calib100; 0 overlap with Test500/Test2000, 2 clips in OOD-Val262; the uniform budget / 24% point came from the pilot grid; rebuttal draft | — | — | `reports/2026-11-08_reviewer-q3-pilot-calib50.md` | — |
| 2026-11-08 | Layer budget allocation, all existing results gathered (fixed profiles, Tyr search, scope/exemptions, head-vs-MLP split, contiguous layer removal): nothing beats uniform for dual; only the layer-0 exemption helps; dual_w4/w8 are quantized, not allocation arms | — | — | `reports/2026-11-08_layer-budget-allocation-survey.md` | — |
| 2026-11-07 | Closed-loop gap decomposition (10-07) redone on all 100 Hard100 scenes: multi-answer (alt-route) gate still not supported, pace explains the gap (moved from `cvpr27/reports/` on 11-08) | `plans/2026-10-07_closedloop-gap-decomposition.md` | `cvpr27/code/closedloop_gap_100.py` | `reports/2026-11-07_closedloop-gap-100scenes.md` | `outputs/cvpr27/closedloop_gap_100/` |
| 2026-11-07 | Closed-loop CIs on Origin150 / Hard100 (100 scenes, paper Table 2 convention), Table 2 check (2 wrong cells), aborted rollouts re-scored, score variance, MoOP vs Tyr / LLM-Pruner / FM-only (moved from `cvpr27/reports/` on 11-08) | — | `cvpr27/code/closedloop_ci.py`, `h100_reeval_aborted.py`, `h100_include_aborted.py`, `h100_gtpath_offroad.py` | `reports/2026-11-07_closedloop-ci-origin150-hard100.md` | `outputs/cvpr27/closedloop_ci_2026-11-07/` |
