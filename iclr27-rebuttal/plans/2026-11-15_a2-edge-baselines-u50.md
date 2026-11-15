# Plan: the ICLR paper's Tyr, LLM-Pruner and Wanda on Alpamayo 2 Edge at u50

**Status: done 2026-11-15 ~14:05 KST.**
- Report: `reports/2026-11-15_a2-edge-baselines-u50.md`.
- MoOP and MoOP+E75 beat Tyr, LLM-Pruner and Wanda on all three sets.
- The expert MLP 75 % cut is free.

2026-11-15. User ask: "U50에서 iclr 논문 버전의 tyr, llm pruner, wanda도 성능 측정해줘".

User decisions:
- **Evaluation sets:** Test500 + OOD-Val250 + OOD-train1178, the same three sets as the u50 MoOP / FM-only / CE-only
  table.
- **LLM-Pruner layer scope:** the paper rule.

Reference: the code map of the A1.5 baselines (subagent report, 2026-11-15). None of the three was ported to A2 Super or
Edge before this.

## Budget

Edge u50 = MoOP u50 = 8 of 16 Q heads + 4,608 of 9,216 MLP channels in every one of the 28 layers: **645,922,816
parameters (24.4 % of 2.65 B)**. Edge costs: head 2·128·2048, channel 2·2048 (relu^2).

| method | budget rule | removed |
|---|---|---|
| Wanda | same per-layer ratio 0.5 in all 28 layers | 645,922,816 (exact) |
| LLM-Pruner | paper rule: skip the first 4 and last 2 layers → scope [4, 26), 22 layers; upstream `--pruning_ratio` = 0.5·28/22 (A1.5: 0.5·36/30 = 0.6); heads floor to whole heads → 10/16 heads + 5,865/9,216 channels per scoped layer | 643,850,240 (−2.07 M = −0.3 %; A1.5 was −6.2 M) |
| Tyr | supernet level 0 = the u50 cut (8 heads, 4,608 channels), level steps 1 head / 192 channels (A1.5 256 of 12,288, scaled); the search conserves the per-type level sum | 645,922,816 (exact) |

## Methods (paper versions, ported)

All three use calib_100 (100 clips, t0 5.1 s) from the Edge cache `calib_edge` and Edge's no-nav prompt.

**Wanda**, `code/edge_wanda.py`. Port of `run_wanda.py --tokens all` + make_slim `wanda_u*_v2`.
- Statistics: running mean over clips of Σ_tokens x² at the inputs of `o_proj` and `mlp.fc2`. Prompt prefill only.
- Scores:
  - head = √(Σ_{j∈h} scaler_o[j]·‖W_o[:,j]‖²)
  - channel = √scaler_m[c]·‖W_fc2[:,c]‖
- Selection: `select_mask_ratios` per layer.
- Gate GW: the hooked statistics equal a direct recomputation on one layer. **Passed (rel. err. 0.0).**

**LLM-Pruner**, `code/edge_lp.py`. Port of `run_lp_importance.py` objective coc + `lp_prune.py --arm coc_param_first`.
- Objective: own-rollout CoC NLL, teacher-forced, no cache.
- Gradients: weight gradients summed over the 100 clips (`lp_core.GradAccumulator`).
- Importance: `lp_core.group_importance(param_first, sum)`, element-wise first-order Taylor.
  - ATTN = {q_proj.out, o_proj.in}
  - MLP = {fc1.out, fc2.in}
  - k/v and norms frozen
- Selection: `keep_masks_for_ratio`, local per layer, in the scope.
- No recovery.
- `lp_core` is imported unchanged; only its member tables are set to Edge's module names.

**Tyr**, `code/edge_tyr.py`. Port of the paper row `tyr_u40_r` / `tyr_u50_r`: run_tyr_supernet + run_tyr_search +
make_slim assembly; `tyr_lib` imported unchanged.
- teacher: dense CoC rollout; top-1024 logits at the CoC positions; expert field at 2 noise draws × t ∈ {0.1, 0.5, 0.9}
  with x1 = GT action.
- supernet: OSSCAR at 9 levels on `o_proj` (16 head groups) and `fc2` (9,216 channel groups) from prompt-prefill Hessians.
  damp 1e-2. Median error accumulation.
- search: 20 generations × 32 offspring. Staged selection on 4 / 16 / 48 clips, elitist parent. 4 workers.
  Fitness = KL_coc/KL0 + MSE_vf/MSE0.
- assembly: weights file (`weights_tyr_u50.pt`) plus masks from the exact-zero columns.
- Tyr rewrites weights. `run_edge_openloop.py --weights` loads them before the mask hooks and asserts that the masked
  columns are exactly zero.

## Addition: MoOP + expert pruning (user, 2026-11-15)

User: "추가로 MoOP + expert pruning도 돌려줘", then "너무 많이 제거하면 안되고 75% 정도 제거해봐".
- Recipe: make_slim `dualexp_u50_em<M>` (the A1.5 rebuttal arm) with **M = 75**, not the paper's 93.75.
  - VLM half = MoOP u50 masks, verbatim. The script asserts unit-for-unit equality.
  - Expert half = 75 % of every expert layer's MLP channels: 2,880 of 3,840 cut, 960 kept, ranked by the expert FM
    importance (`edge_importance_v1/importance.npz:traj_exp_mlp`, mask_lib.select_mask).
  - Expert Q heads and KV are untouched.
- Parameters removed:
  - expert: 82,575,360 (channel = 2·512)
  - total: 728,498,176 = 27.5 % of 2.65 B
- Masks: `masks_moop_em75_u50.npz` with q_mask, mlp_mask and exp_mlp_mask. `EdgeVlmMasks` adds pre-hooks at the expert
  layers' `mlp.fc2` input.
- G1 on 3 calib clips: all-ones equals no hooks bitwise, and the masks change the output. Passed.
- Arm: `moop_em75_u50`, on the same three sets.

## Evaluation

- Same runner and protocol as the u50 table: 1 CoC rollout + 8 denoisings, minADE@6 / minFDE@6 at 1.6 / 3.2 / 6.4 s.
- Arms `edge_{test500,oodval,oodtrain}_{wanda_u50,lp_u50,tyr_u50}`. Ada cards 4–7 through `edge_queue.py`, 4 CPU threads
  per job.
- The existing dense / MoOP / FM-only / CE-only u50 rows are reused (same hardware and protocol).
- Analysis: absolute values first, then each baseline vs dense and MoOP − baseline, paired with 200,000-draw bootstrap
  CIs, all horizons.
- Not pre-registered. These are additional comparisons on the sets that already hold the u50 rows.

## Report

`reports/2026-11-15_a2-edge-baselines-u50.md`, plus a README row.
