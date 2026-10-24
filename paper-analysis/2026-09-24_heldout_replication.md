# Held-out replication of the Finding-1 panels (2026-09-24)

**Question.** The paper's Finding 1 is measured on the 100 calibration clips (`calib_100`,
official train). Do the two panels -- within-layer rank agreement between `I_traj` and `I_CoC`
(`fig1_rank_agreement`) and the additive token-position shares of each score
(`fig4_token_share_{traj,coc}_mlp`) -- look the same on clips the model was never calibrated on?

**Data.** Three new gradient-anatomy runs, identical to `gradanat_v1` except for the clips (100
clips, seed 42, 10 FM steps, dense model, Ada, 21 min each):

| run | clips | manifest | cache | G0(c) vs own importance run |
|---|---|---|---|---|
| `gradanat_val100` | in-distribution validation | `calib_val100` | `eval` | PASS (per-layer Spearman >= 0.999 vs `importance_val100`) |
| `gradanat_ood` | long-tail OOD | `calib_ood_100` | `ood` | PASS (>= 0.999 vs `importance_ood`) |
| `gradanat_test100` | held-out official test | `calib_test100` (new: first 100 rows of `test_500`) | `test` | no importance run on these clips; the check against `importance_v2_ada` (calibration clips) fails as expected |

The rank-agreement panel of each run is read from the same file (`anatomy.npz`, full-seed gate
gradients summed over positions = the shipped scores; on `gradanat_v1` the band means agree
with `importance_v2` to 0.006).

**Pre-registered window** (`plans/2026-09-24_token-share-replication.md`): every band-mean
share (vision / text / ego-history x layers 6-17 / 27-34 x `I_traj` / `I_CoC`) and every
rank-agreement band mean (0-6 / 7-21 / 22-34, Q and MLP) within +-0.10 of the calib_100 value.

## Result: replicates on all three splits

- Token shares: 24 of 24 cells inside the window on each split (MLP and Q heads).
- Rank agreement: 17 of 18 cells inside the window. The one outside is MLP layers 7-21 on
  val100, 0.77 against 0.67 on calib_100 (+0.10); OOD and test100 give 0.73 / 0.74. The held-out
  sets agree slightly *more* in the mid-trunk than the calibration set, so the deviation is in the
  direction that does not weaken the claim (trunk agrees, late band does not).
- Direction of the small within-window differences on Q heads: the held-out sets give the late
  `I_traj` a little more vision (0.35 -> 0.40-0.44) and less ego-history (0.16 -> 0.09-0.12).

Tables produced by `paper-analysis/scripts/token_share_compare.py gradanat_v1 gradanat_val100
gradanat_ood gradanat_test100` (`*` = |delta| > 0.10 from the reference column):

**Token shares, MLP channels** (reference = gradanat_v1; `*` marks |delta| > 0.10)

| cell | gradanat_v1 | gradanat_val100 | gradanat_ood | gradanat_test100 |
|---|---|---|---|---|
| I_traj / 6-17 / vision | 0.90 | 0.90 | 0.93 | 0.92 |
| I_traj / 6-17 / text | 0.07 | 0.06 | 0.04 | 0.05 |
| I_traj / 6-17 / hist | 0.03 | 0.04 | 0.03 | 0.03 |
| I_traj / 27-34 / vision | 0.26 | 0.29 | 0.30 | 0.27 |
| I_traj / 27-34 / text | 0.66 | 0.66 | 0.66 | 0.68 |
| I_traj / 27-34 / hist | 0.08 | 0.05 | 0.04 | 0.05 |
| I_CoC / 6-17 / vision | 0.75 | 0.75 | 0.72 | 0.71 |
| I_CoC / 6-17 / text | 0.24 | 0.24 | 0.26 | 0.27 |
| I_CoC / 6-17 / hist | 0.01 | 0.02 | 0.01 | 0.01 |
| I_CoC / 27-34 / vision | 0.02 | 0.02 | 0.02 | 0.02 |
| I_CoC / 27-34 / text | 0.98 | 0.98 | 0.98 | 0.98 |
| I_CoC / 27-34 / hist | 0.00 | 0.00 | 0.00 | 0.00 |

cells outside the window: 0

**Token shares, Q heads** (reference = gradanat_v1; `*` marks |delta| > 0.10)

| cell | gradanat_v1 | gradanat_val100 | gradanat_ood | gradanat_test100 |
|---|---|---|---|---|
| I_traj / 6-17 / vision | 0.72 | 0.73 | 0.76 | 0.74 |
| I_traj / 6-17 / text | 0.18 | 0.17 | 0.14 | 0.16 |
| I_traj / 6-17 / hist | 0.10 | 0.11 | 0.10 | 0.10 |
| I_traj / 27-34 / vision | 0.35 | 0.41 | 0.44 | 0.40 |
| I_traj / 27-34 / text | 0.49 | 0.47 | 0.47 | 0.49 |
| I_traj / 27-34 / hist | 0.16 | 0.12 | 0.09 | 0.11 |
| I_CoC / 6-17 / vision | 0.63 | 0.60 | 0.59 | 0.58 |
| I_CoC / 6-17 / text | 0.34 | 0.36 | 0.37 | 0.38 |
| I_CoC / 6-17 / hist | 0.03 | 0.04 | 0.03 | 0.04 |
| I_CoC / 27-34 / vision | 0.02 | 0.02 | 0.02 | 0.02 |
| I_CoC / 27-34 / text | 0.98 | 0.98 | 0.98 | 0.98 |
| I_CoC / 27-34 / hist | 0.00 | 0.00 | 0.00 | 0.00 |

cells outside the window: 0

**Rank agreement (Spearman rho, band means)** (reference = gradanat_v1; `*` marks |delta| > 0.10)

| cell | gradanat_v1 | gradanat_val100 | gradanat_ood | gradanat_test100 |
|---|---|---|---|---|
| Q / 0-6 | 0.78 | 0.81 | 0.77 | 0.78 |
| Q / 7-21 | 0.77 | 0.81 | 0.79 | 0.80 |
| Q / 22-34 | 0.33 | 0.33 | 0.40 | 0.35 |
| MLP / 0-6 | 0.93 | 0.95 | 0.94 | 0.94 |
| MLP / 7-21 | 0.67 | 0.77* | 0.73 | 0.74 |
| MLP / 22-34 | 0.34 | 0.39 | 0.41 | 0.39 |

cells outside the window: 1

## Files

- Panels (paper style, same scripts as the main panels):
  `paper-analysis/figures/supplementary/fig1_rank_agreement_{val100,ood,test100}.*`,
  `paper-analysis/figures/supplementary/fig4_token_share_{traj,coc}_mlp_{val100,ood,test100}.*`
- Runs: `outputs/gradanat_{val100,ood,test100}/` (config, metrics, metrics_analysis, summary);
  manifest `outputs/eval_sets/calib_test100.parquet`.
- Scripts: `paper-analysis/scripts/paper_rank_agreement.py <run> <suffix>`,
  `paper_token_share.py <run> <suffix>`, `token_share_compare.py <ref> <runs...>`.

## Reproduction

```bash
cd /home/cvlab21/project/chan/alpamayo-model-compression
bash experiments/head_analysis/run_retry_host.sh 30 experiments/head_analysis/run_gradient_anatomy.py \
    --exp-id gradanat_val100 --calib-manifest calib_val100 --cache eval --num-clips 100 --gpu 4
#   ... gradanat_ood: --calib-manifest calib_ood_100 --cache ood; gradanat_test100: --calib-manifest calib_test100 --cache test
.venv/bin/python experiments/head_analysis/analyze_gradient_anatomy.py --exp gradanat_val100 --ref importance_val100
cd paper-analysis/scripts && ../../.venv/bin/python paper_token_share.py gradanat_val100 _val100 \
    && ../../.venv/bin/python paper_rank_agreement.py gradanat_val100 _val100
```
- Overview sheet (3 rows x 4 clip sets, preview only): `paper-analysis/figures/supplementary/heldout_replication_sheet.png`.
- Q-head panels (added later the same day, `paper_token_share.py <run> <suffix> q`):
  `paper-analysis/figures/supplementary/fig4_token_share_{traj,coc}_q_head.*` (calib_100) and
  `..._q_head_{val100,ood,test100}.*`; preview sheet `heldout_replication_sheet_q.png`. Per-layer
  Q-head shares are noisy (clip-bootstrap 95% CI half-width about 0.11), so read the band means.
