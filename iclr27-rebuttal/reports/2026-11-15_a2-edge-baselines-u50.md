# Experiment: the ICLR paper baselines (Tyr, LLM-Pruner, Wanda) and MoOP + expert on Alpamayo 2 Edge at u50

2026-11-15. Plan: `iclr27-rebuttal/plans/2026-11-15_a2-edge-baselines-u50.md`. Follows
`reports/2026-11-15_a2-edge-moop-driving-search.md`, where u50 was the confirmed setting for MoOP vs FM-only / CE-only.

## Purpose

- Place the paper's three baselines (Tyr, LLM-Pruner, Wanda) next to MoOP on Edge at the u50 budget, ported with their
  paper recipes.
- Measure MoOP + expert pruning at a moderate expert cut (75 % of expert MLP channels, user's choice).
- Every arm is evaluated on Test500, OOD-Val250 and OOD-train1178 (user decision).

## Setup

**Shared:**
- `nvidia/Alpamayo2-Edge@f8951d0`, no-nav prompt.
- calib_100 (100 clips, t0 5.1 s) from the Edge cache for every calibration.
- Evaluation: 1 CoC rollout + 8 denoisings; minADE@6 / minFDE@6 at 1.6 / 3.2 / 6.4 s.
- Hardware: Ada RTX 5880 cards 4–7, 4 CPU threads per job.
- Bootstrap: 200,000 draws, shared across arms; the seed-1 reruns are stored.
- Dense, MoOP, FM-only and CE-only u50 rows come from the search report (same protocol and hardware).

**Arms and budgets.** Per VLM layer; Edge costs: head = 2·128·2048, channel = 2·2048, expert channel = 2·512.

| arm | what is cut | removed params | % of 2.65 B |
|---|---|---|---|
| MoOP / FM-only / CE-only u50 | 8/16 heads + 4,608/9,216 channels, all 28 layers | 645,922,816 | 24.4 % |
| **MoOP+E75** | MoOP u50 + 2,880/3,840 expert MLP channels in all 28 expert layers | 728,498,176 | 27.5 % |
| Wanda u50 | 8 heads + 4,608 channels, all layers | 645,922,816 | 24.4 % |
| LLM-Pruner u50 | 10 heads + 5,865 channels in layers 4–25 (paper rule), none in 0–3 / 26–27 | 643,850,240 | 24.3 % |
| Tyr u50 | searched per-layer allocation; heads cut per layer range 5–11, MLP levels ±2 × 192 around 4,608 | 645,922,816 | 24.4 % |

**Ports.** The paper code is imported unchanged where it is model-agnostic; only Edge specifics are changed:
- inputs: no-nav prompt, `mm_token_type_ids`
- MLP: `fc1` / `fc2`, relu^2
- the 28-layer / 16-head budget

Ports:
- **Wanda** (`code/edge_wanda.py`): structured Wanda.
  - Inputs: prompt-prefill activation statistics at the `o_proj` / `fc2` inputs.
  - Score: head = √(Σ scaler_o·‖W_o[:,j]‖²), channel = √scaler_m·‖W_fc2[:,c]‖. Same ratio in every layer.
  - Gate GW: the hooked statistics equal a direct recomputation exactly.
- **LLM-Pruner** (`code/edge_lp.py`): paper arm `coc_param_first`.
  - Objective: own-rollout CoC NLL. Weight gradients are summed over the 100 clips (`lp_core.GradAccumulator`).
  - Importance: element-wise first-order Taylor, `lp_core.group_importance(param_first, sum)`. The groups are
    {q_proj.out, o_proj.in} and {fc1.out, fc2.in}; k/v are frozen.
  - Selection: upstream ratio rule in scope [4, 26), r = 0.5·28/22. No recovery.
  - The rollouts reproduce the Edge importance run exactly (CoC 14.38 tokens, NLL 0.194).
- **Tyr** (`code/edge_tyr.py`): the paper's port (`tyr_u40_r` / `tyr_u50_r` recipe).
  - teacher: dense rollout, top-1024 logits at the CoC positions, expert field at 6 points with x1 = GT. Checked: dense
    vs teacher gives KL 0.0 and MSE 0.0.
  - supernet: OSSCAR + least-squares reconstruction of `o_proj` / `fc2` at 9 levels from prompt Hessians, damp 1e-2,
    median error accumulation. 2,127 s.
  - search: 20 generations × 32 offspring, 4 workers, fitness KL_coc/KL0 + MSE_vf/MSE0. KL0 4.379, MSE0 0.0471. Final
    fitness 0.757 (init 2.0), 1,021 s. The per-type budget is conserved (asserted).
  - assembly: reconstructed weights (`weights_tyr_u50.pt`) plus masks from the exact-zero columns.
  - `run_edge_openloop.py --weights` loads the weights and asserts that the masked columns are zero.
- **MoOP+E75** (`make_edge_moop_masks.py --pct 50 --expert-mlp 75`): make_slim `dualexp_u50_em75`.
  - VLM half = MoOP u50 masks; the script asserts they are identical.
  - Expert half = 75 % of the expert MLP channels, by the expert FM importance (`traj_exp_mlp`).
  - Expert Q heads and KV untouched.
  - New expert pre-hooks pass G1: all-ones equals no hooks bitwise, and the masks change the output.

## Results (Facts)

### minADE@6 / minFDE@6 at 6.4 s (m), mean, all three sets

| arm | Test500 ADE | Test500 FDE | OOD-Val ADE | OOD-Val FDE | OOD-train ADE | OOD-train FDE |
|---|---|---|---|---|---|---|
| Dense | 0.916 | 2.722 | 0.964 | 2.444 | 0.975 | 2.480 |
| **MoOP+E75** | **1.190** | **3.396** | **1.363** | **3.415** | **1.318** | **3.295** |
| **MoOP** | 1.196 | 3.406 | 1.374 | 3.449 | 1.322 | 3.307 |
| FM-only | 1.441 | 3.881 | 1.422 | 3.652 | 1.457 | 3.601 |
| CE-only | 1.489 | 3.942 | 1.678 | 4.094 | 1.456 | 3.457 |
| Tyr | 1.616 | 4.466 | 1.912 | 4.775 | 2.039 | 4.946 |
| LLM-Pruner | 2.236 | 6.195 | 2.391 | 6.057 | 2.343 | 5.836 |
| Wanda | 2.407 | 6.741 | 2.523 | 6.594 | 2.666 | 6.555 |

### minADE@6 at 1.6 / 3.2 s (m), mean

| arm | Test500 1.6 s | Test500 3.2 s | OOD-Val 1.6 s | OOD-Val 3.2 s | OOD-train 1.6 s | OOD-train 3.2 s |
|---|---|---|---|---|---|---|
| Dense | 0.045 | 0.199 | 0.058 | 0.242 | 0.063 | 0.255 |
| MoOP+E75 | 0.069 | 0.276 | 0.086 | 0.345 | 0.092 | 0.356 |
| MoOP | 0.069 | 0.277 | 0.087 | 0.348 | 0.092 | 0.357 |
| FM-only | 0.083 | 0.354 | 0.091 | 0.376 | 0.105 | 0.395 |
| CE-only | 0.112 | 0.404 | 0.123 | 0.479 | 0.116 | 0.425 |
| Tyr | 0.109 | 0.405 | 0.132 | 0.527 | 0.160 | 0.595 |
| LLM-Pruner | 0.155 | 0.580 | 0.170 | 0.658 | 0.188 | 0.678 |
| Wanda | 0.164 | 0.617 | 0.172 | 0.676 | 0.217 | 0.796 |

Medians, 1.6 / 3.2 s FDE and every CI are in `outputs/iclr27-rebuttal/edge_baselines_u50_analysis/`.

### Paired differences at ADE 6.4 s (mean [95 % CI])

| contrast | Test500 | OOD-Val250 | OOD-train1178 |
|---|---|---|---|
| MoOP − Tyr | −0.420 [−0.567, −0.276]\* | −0.538 [−0.776, −0.303]\* | −0.717 [−0.857, −0.580]\* |
| MoOP − LLM-Pruner | −1.041 [−1.230, −0.858]\* | −1.017 [−1.294, −0.751]\* | −1.021 [−1.161, −0.884]\* |
| MoOP − Wanda | −1.211 [−1.423, −1.006]\* | −1.149 [−1.458, −0.854]\* | −1.344 [−1.519, −1.173]\* |
| MoOP − FM-only | −0.245\* | −0.048 (n.s.) | −0.136\* |
| MoOP − CE-only | −0.293\* | −0.304\* | −0.135\* |
| **MoOP+E75 − MoOP** | **−0.006 [−0.012, −0.000]\*** | **−0.011 [−0.020, −0.002]\*** | **−0.004 [−0.012, +0.007]** |
| MoOP+E75 − Tyr / LP / Wanda | −0.43\* / −1.05\* / −1.22\* | −0.55\* / −1.03\* / −1.16\* | −0.72\* / −1.03\* / −1.35\* |

MoOP+E75 − MoOP at FDE 6.4 s: −0.010 (n.s.) / −0.034\* / −0.012 (n.s.). At ADE 1.6 / 3.2 s it is −0.000 / −0.001 to −0.003.

### CoC health (degenerate rate / mean length in tokens / CoCs that never close with the anchor)

| arm | Test500 | OOD-Val250 | OOD-train1178 |
|---|---|---|---|
| Dense | 0 % / 14.4 / 0 | 0 % / 17.5 / 0 | 0 % / 17.3 / 0 |
| MoOP+E75 | 0 % / 26.0 / 0 | 0.4 % / 27.2 / 0 | 0.1 % / 27.2 / 0 |
| MoOP | 0 % / 26.0 / 0 | 0.4 % / 27.2 / 0 | 0.1 % / 27.2 / 0 |
| FM-only | 24.6 % / 55.2 / 4 | 26.8 % / 54.8 / 2 | 27.3 % / 55.3 / 9 |
| CE-only | 0 % / 24.4 / 0 | 0 % / 24.8 / 0 | 0 % / 24.9 / 0 |
| Tyr | 65.0 % / 132 / 114 | 69.6 % / 143 / 58 | 66.3 % / 134 / 261 |
| LLM-Pruner | 62.6 % / 164 / 285 | 69.2 % / 179 / 156 | 69.9 % / 179 / 733 |
| Wanda | 85.6 % / 171 / 179 | 88.0 % / 170 / 85 | 83.2 % / 162 / 390 |

## Analysis (Opinion)

- **At u50 on Edge, both MoOP arms are the best pruned drivers on all three sets.**
  - Ranking at 6.4 s ADE: MoOP+E75 ≈ MoOP < FM-only ≈ CE-only < Tyr < LLM-Pruner < Wanda.
  - The only non-significant gap is MoOP vs FM-only on OOD-Val, already noted in the search report.
  - The baselines lose 0.4–1.3 m against MoOP, and all of them break the CoC: 63–88 % degenerate, 130–180 tokens, many
    never closing within 256 tokens. The same failure mode separates FM-only from MoOP, at a much larger scale.
  - The order Tyr < LLM-Pruner < Wanda matches the A1.5 paper table.
- **Tyr is the strongest baseline but still collapses the CoC.**
  - Its fitness (CoC top-k KL + expert MSE) improved from 2.0 to 0.76, but that did not keep the generated CoC healthy.
  - Its search moved heads between layers: 5–11 cuts per layer, fewer in layers 3–4 and more in layer 10.
  - Interpretation: matching the teacher's next-token distribution under teacher forcing does not prevent free-running
    degeneration.
- **The expert MLP cut is free on Edge, as it was on A1.5.**
  - Removing 75 % of every expert layer's MLP channels on top of MoOP u50 does not hurt. It is marginally better on
    Test500 and OOD-Val (−0.006\* / −0.011\* m) and equal on OOD-train.
  - The CoC is unchanged, since the expert does not influence the VLM rollout.
  - So MoOP+E75 removes 27.5 % of Edge's parameters at the same driving cost as MoOP's 24.4 %.
  - A deeper cut (the paper's 93.75 %) was not run here. A1.5 found that width free as well, but it is untested on Edge.
- **Caveats.**
  - Not pre-registered. OOD-train was the confirmatory set for MoOP vs FM / CE only; these baseline comparisons are
    additional.
  - Ports, not the original model:
    - Wanda and LLM-Pruner use calib_100 with Edge's no-nav prompt.
    - Tyr's level steps are scaled to Edge's width (192 channels).
    - LLM-Pruner's scope follows the paper's skip-4/2 rule, so it is 2.07 M parameters lighter.
  - Every u50 arm is far from dense (MoOP +30–43 %).
  - No-nav prompt.
- **Suggested next steps.**
  - LingoQA for the u50 arms.
  - The paper's 93.75 % expert cut.
  - A q-cut MoOP+E for a near-dense recipe (u40-q2 + expert cut).

## Execution notes

- Each Wanda and Tyr evaluation launch first failed on the same kind of bug: `.numpy()` on weights with `requires_grad`.
  Both were fixed (`torch.no_grad()` / frozen params) and rerun; no reported number comes from a failed run.
- The scheduler briefly co-located Wanda evaluation with the Tyr teacher and supernet on cards 6–7. Those jobs had not
  yet created a CUDA context, so they were invisible to the check. All co-located processes were this job's own.
