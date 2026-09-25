# Plan: depth-aware dual — the union below layer 22, the CoC score alone above it (2026-09-25)

## Hypothesis

Finding 1 (paper-analysis, `sec2_rank_analysis.tex`) and the 2026-09-21 causal round say the two
scores agree in layers 0–21 and part in the late layers, and that in the late layers the
trajectory score is a first-order shadow: no late-layer set or mask moves the FM loss or minADE at
the shipped dose, while the NLL moves with every late set (late-only masks on held-out clips:
FM within ±0.1% for all three arms; NLL +65.5% under `traj`, +6.3% under `coc`, +10.1% under
`dual`). Under `max(rank I_traj, rank I_CoC)` the late `I_traj` rank therefore buys nothing for
the action and displaces language units the CoC score would keep. Hypothesis H: replacing the
union by `I_CoC` alone in layers 22–35 keeps the action channel of `dual` and improves its
language channel (NLL, CoC degeneracy).

Config `dualv3_u40_v2` (`make_slim.py`): same budget, allocation, expert and KV as
`dual_u40_v2`; layers 0–21 identical to `dual_u40_v2`, layers 22–35 identical to `coc_u40_v2`.
Its mask equals `outputs/masks_u40_v2/dual_trunk__coc_late.npz` (`combine_masks.py`); 54 of the
266 late Q heads and 20,127 of the 103,460 late channels differ from `dual`.

Known risk: bands interact under single criteria (`armmix_v1`: `traj` trunk + `coc` late cost
FM +13.5% against +7.7% with `dual` late). Whether a `coc` late band is safe under a `dual`
trunk is exactly what step 1 measures.

## Step 1 — held-out check, dense text (GPU, ~10 min on four Ada cards)

`run_token_ablation.py --arm-masks` on the 100 held-out `indist_500` clips (same clips, text and
seeds as `armheld_v1` / `armmix_v1`; K = 8), configs: `dual`, `dual_trunk`,
`dual_trunk__coc_late` (A), `dual_trunk__traj_late` (mirror), `dual_trunk__rand_late` (random
late, seed 0). Gates (`analyze_depth_mix.py`, paired per clip):

- G1 action unchanged: FM and minADE of A not worse than `dual` (one-sided Wilcoxon p ≥ 0.05)
  and the mean difference's 95% CI below +1% of the dense FM loss / +0.02 m.
- G2 language better: NLL of A below `dual` and below the random-late control (p < 0.01).
- G3 mirror: `dual_trunk__traj_late` raises the NLL above `dual` (p < 0.01).

Reading: G1 + G2 pass → build the arm (step 2). G1 fails → the `coc` late band interacts with the
`dual` trunk as it did with the `traj` trunk; report, and stop (a band-interaction result, not a
method). G2 fails → the union's late `I_traj` term costs no language at this dose; stop.

```bash
cd /home/cvlab21/project/chan/alpamayo-model-compression/.claude/worktrees/dual-v3
M=outputs/masks_u40_v2
for s in 0 1 2 3; do
  ALPAMAYO_REPO=$PWD nohup bash experiments/head_analysis/run_retry_host.sh 150 experiments/head_analysis/run_token_ablation.py \
      --exp-id depthmix_v1_s$s --shard $s --n-shards 4 --gpu $((4 + s)) --arm-masks \
      dual=$M/dual.npz dual_trunk=$M/dual_trunk.npz dual_trunk__coc_late=$M/dual_trunk__coc_late.npz \
      dual_trunk__traj_late=$M/dual_trunk__traj_late.npz dual_trunk__rand_late=$M/dual_trunk__rand_late.npz \
      > outputs/depthmix_v1_s$s.launch.log 2>&1 &
done
.venv/bin/python experiments/head_analysis/analyze_depth_mix.py --shards depthmix_v1_s0 depthmix_v1_s1 depthmix_v1_s2 depthmix_v1_s3 --out depthmix_v1
```
Cost: 6 configs, all sampled at K = 8 ≈ 30 s per clip → 25 clips per shard ≈ 13 min.

## Step 2 — the arm, open loop (GPU, ~25 min on four Ada cards)

Build `slim_dualv3_u40_v2` (`--no-state`, selection-only, ~30 s), verify its mask equals
`dual_trunk__coc_late.npz`, then `run_baseline.py --set test` in 4 shards (500 clips, own rollout,
K = 8), paired against `dual_u40_v2_test` and `baseline_ada_test` on the same clips and seeds
(Ada; keep the architecture). Gates:

- O1 non-inferior driving: minADE(dualv3) − minADE(dual) mean 95% CI upper bound < +0.02 m
  (the protocol's headline is the mean; report the median too).
- O2 language: CoC degeneracy ≤ dual's (3.0% on test500) and `nll_dense` (run_nll_dense.py,
  dense-rollout reference) below dual's 0.252 (paired p < 0.01).
- Report by manoeuvre; OOD-val (1,533 clips, minADE_tf and nll_gtcoc) if O1–O2 pass.

```bash
ALPAMAYO_REPO=$PWD bash experiments/head_analysis/run_retry_host.sh 30 experiments/head_analysis/make_slim.py \
    --config dualv3_u40_v2 --importance importance_v2 --gpu 4 --out outputs/slim_dualv3_u40_v2 --no-state
.venv/bin/python experiments/head_analysis/slim_to_mask.py --slim outputs/slim_dualv3_u40_v2 --out outputs/masks_u40_v2/dualv3.npz
for s in 0 1 2 3; do
  ALPAMAYO_REPO=$PWD nohup bash experiments/head_analysis/run_retry_host.sh 150 experiments/evaluation/run_baseline.py \
      --set test --model outputs/slim_dualv3_u40_v2 --exp-id dualv3_u40_v2_test --shard $s --n-shards 4 --gpu $((4 + s)) \
      > outputs/dualv3_u40_v2_test_s$s.launch.log 2>&1 &
done
```

## Step 3 — closed loop (later, ~8 h per config)

hard100 (`launch_alpasim_shards.sh`, Ada 4–7, `OMP_NUM_THREADS=8`) against the stored `dual`
and baseline runs; only if step 2 passes. Open loop does not see closed-loop harm (`dual+h4`),
so no claim before this.

## What each outcome means for the paper

- Pass through step 2: a method-level consequence of Finding 1 — the union is needed where the
  scores agree on the visual input, and the language score alone where the trajectory score is a
  shadow. It is a one-factor change from `dual` (same budget, same trunk).
- Fail at step 1 G1: bands interact; the late `I_traj` term, though functionless in isolation,
  co-selects with the trunk. Report as such; the union stays.
- Fail at G2: late `I_traj` costs no language at 40%; the depth split is not worth a rule.

## Outputs

`outputs/depthmix_v1_s{0..3}`, `outputs/depthmix_v1/{summary.txt,metrics.json,config.json}`,
`outputs/slim_dualv3_u40_v2`, `outputs/dualv3_u40_v2_test`; note in
`paper-analysis/2026-09-25_depth_aware_dual.md`.
