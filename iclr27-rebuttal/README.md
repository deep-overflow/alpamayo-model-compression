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
| 2026-11-08 | Combination-rule ablation, all existing results gathered (l0keep A100/Ada + pre-l0keep, layer-0 confound flagged) | — | — | `reports/2026-11-08_combination-rule-summary.md` | — |
| 2026-11-08 | Paper analyses on Alpamayo 2 Super (34B): Finding 1/2, Fig. 3 (token types), Fig. 5 (expert), VLM MLP depth structure | `plans/2026-11-07_a2-analysis-replication.md` | `code/a2_depth_structure.py`, `code/a2_fig3_token_type.py`, `code/a2_fig5_expert.py`, `code/a2_rank_band_review.py` (review: step-aligned bands) | `reports/2026-11-08_a2-analysis-replication.md` | `outputs/a2_depth_structure/`, `outputs/a2_fig3_token_type/`, `outputs/a2_fig5_expert/`, `outputs/a2_rank_band_review/` |
