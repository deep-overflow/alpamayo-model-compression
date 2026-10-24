# Plan: do the Finding-1 panels replicate on held-out clips? (2026-09-24)

## Purpose

The paper's Finding 1 rests on two panels measured on the 100 calibration clips
(`calib_100`, official train): the within-layer rank agreement between `I_traj` and `I_CoC`
(`fig1_rank_agreement`) and the additive token-position shares of each score
(`fig4_token_share_{traj,coc}_mlp`). Question: do the same shapes appear on clips the model was
never calibrated on -- in-distribution validation (`calib_val100`), long-tail OOD
(`calib_ood_100`) and the held-out official test split (new `calib_test100`)?

## What already exists (no GPU needed)

Importance runs on val100 and OOD clips exist (`importance_val100`, `importance_ood`), so the
rank-agreement panel is already replicated; rendered as
`paper-analysis/figures/supplementary/fig1_rank_agreement_{val100,ood}.*`.

Band-mean Spearman rho (Q heads / MLP channels):

| clips | Q 0-6 | Q 7-21 | Q 22-34 | MLP 0-6 | MLP 7-21 | MLP 22-34 |
|---|---:|---:|---:|---:|---:|---:|
| calib_100 (paper) | 0.78 | 0.78 | 0.33 | 0.93 | 0.67 | 0.34 |
| val100 | 0.80 | 0.81 | 0.33 | 0.95 | 0.77 | 0.39 |
| OOD 100 | 0.76 | 0.79 | 0.40 | 0.94 | 0.73 | 0.41 |

Same two-step structure on every set: trunk ~0.8, late band 0.33-0.41, change at layer 22-23.

No gradient-anatomy run exists on any held-out set, and no importance run exists on test clips,
so the token-share panels (and the test-split rank panel) need three new runs.

## Runs (not launched -- start on the user's go)

One `run_gradient_anatomy.py` run per split, identical to `gradanat_v1` except for the clips:
100 clips, seed 42, 10 FM steps, max_gen 256, dense model, no `--verify` (the typed-sum
identity is a property of the code and was verified once at 6e-06). Each run's `anatomy.npz`
holds the full-seed gate gradients, which are the importance scores on those clips, so the same
run yields the rank-agreement panel as well.

| exp_id | manifest | cache split | GPU | ref for the analysis |
|---|---|---|---|---|
| `gradanat_val100` | `calib_val100` | `eval` | 4 | `importance_val100` (same clips, same seed -> G0(c) is a real check) |
| `gradanat_ood` | `calib_ood_100` | `ood` | 5 | `importance_ood` |
| `gradanat_test100` | `calib_test100` (new) | `test` | 6 | none exists; pass the default and ignore G0(c) |

`calib_test100.parquet` = the first 100 rows of `test_500.parquet` (created 2026-09-24; the
greedy draw is prefix-matched, and the 100 clips match test500 on country / time of day /
platform to within 1 pp; disjoint from calib_100, val100, ood and indist_500; all 100 present in
the `test` cache).

```bash
cd /home/cvlab21/project/chan/alpamayo-model-compression
for spec in "gradanat_val100 calib_val100 eval 4" "gradanat_ood calib_ood_100 ood 5" "gradanat_test100 calib_test100 test 6"; do
  set -- $spec
  nohup bash experiments/head_analysis/run_retry_host.sh 30 \
      experiments/head_analysis/run_gradient_anatomy.py --exp-id $1 --calib-manifest $2 --cache $3 \
      --num-clips 100 --gpu $4 > outputs/$1.launch.log 2>&1 &
done
```

Then the analysis (CPU, ~16 min each; the three can run at once):

```bash
.venv/bin/python experiments/head_analysis/analyze_gradient_anatomy.py --exp gradanat_val100 --ref importance_val100
.venv/bin/python experiments/head_analysis/analyze_gradient_anatomy.py --exp gradanat_ood --ref importance_ood
.venv/bin/python experiments/head_analysis/analyze_gradient_anatomy.py --exp gradanat_test100
```

And the panels (`paper-analysis/scripts/`, both scripts now take `<run> <suffix>`):

```bash
cd paper-analysis/scripts
for s in val100 ood test100; do
  ../../.venv/bin/python paper_token_share.py gradanat_$s _$s
  ../../.venv/bin/python paper_rank_agreement.py gradanat_$s _$s
done
```

Cost: `gradanat_v1` took 21.5 min on one Ada card (peak 42 GB, reserve 44 GB) and 2.0 GB of
disk; three runs in parallel on Ada 4/5/6 (free at 13:25 KST) -> panels about 45 min after
launch. `/mnt/nvme1n1` has 252 GB free.

## Pre-registered reading

Replicates if, on each split, every band-mean share (vision / text / ego-history x layers 6-17 /
27-34 x `I_traj` / `I_CoC`, 12 cells) lies within +-0.10 of the calib_100 value, and the
rank-agreement band means (0-6 / 7-21 / 22-34, Q and MLP) within +-0.10. Cells outside the
window are reported as such, not smoothed over. Calib_100 reference (MLP): `I_traj` 6-17 vision
0.90 / text 0.07 / hist 0.03, 27-34 vision 0.26 / text 0.66 / hist 0.08; `I_CoC` 6-17 vision
0.75 / text 0.24 / hist 0.01, 27-34 vision 0.02 / text 0.98 / hist 0.00.

## Outputs

`outputs/gradanat_{val100,ood,test100}/` (config, metrics, metrics_analysis, summary, plots);
panels in `paper-analysis/figures/supplementary/`; a comparison table appended to
`paper-analysis/README.md` (section on replication).

## Result (2026-09-24, runs launched 13:33 KST, panels 13:58 KST)

Replicates on all three splits: token shares 24/24 cells inside the window on each split (MLP
and Q), rank agreement 17/18 -- the one outside is MLP layers 7-21 on val100 (0.77 vs 0.67,
+0.10, i.e. more trunk agreement, not less). G0(c) PASS for val100 and OOD against their own
importance runs (>= 0.999). Tables, panels and file paths:
`paper-analysis/2026-09-24_heldout_replication.md`.
