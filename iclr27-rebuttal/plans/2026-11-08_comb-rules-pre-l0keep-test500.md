# Plan: combination rules under the paper's (pre-l0keep) recipe, test500 (2026-11-08)

User ask: "l0keep 이전 방식에서 성능을 보여주는 게 좋을 것 같아. rank-mean, rank-max, raw-mean, raw-max를 test 500에서 비교해줘."

The paper's main rows use the pre-l0keep recipe (`dual_u40_v2`, layer 0 cut like every other layer, 13 of 32 Q heads), so the
ablation should use the same recipe. The l0keep ablation (2026-11-05, A100 test_2000) stays as the clean
(layer-0-controlled) companion.

## Arms (u40, uniform per layer: 13 Q heads + 4,898 MLP channels removed in every layer, layer 0 included; expert and KV untouched)

| row | config | FM score | CoC score | norm | combine | test500 run |
|---|---|---|---|---|---|---|
| rank-max (MoOP, paper) | `dual_u40_v2` | I_traj (10-step sum) | I_CoC | rank in layer | max | existing `dual_u40_v2_ps_test` (0.9498 = paper 0.950) |
| rank-mean | `dualsum_u40_v2` | same | same | rank in layer | mean | existing `dualsum_u40_v2_test` |
| raw-mean | `rawabsavg_u40_v2` | mean_s \|per-step Taylor\| (`importance_stepvlm_v1`) | I_CoC | none | mean | **new** |
| raw-max | `rawabsmean_u40_v2` | same | same | none | max | **new** |

Definitions are the 2026-11-05 paper ablation's (`cvpr27/plans/2026-11-04_combination-rule-ablation.md`), with `_l0keep` dropped.
Importance: `importance_v2` (calib_100) for every arm.

## Protocol (same as the existing rows; all Ada, model revision 7aba829, same 500 clip IDs as manifest test_500 — checked)

`run_baseline.py --set test --manifest test_500`, k=8, seed 42 per clip (`sha256(seed:clip_id)`), max_gen 256, no teacher forcing.
Readouts: minADE@6 / minFDE@6 at 1.6 / 3.2 / 6.4 s (first six of eight samples), CoC degeneracy; paired Δ vs rank-max, bootstrap
200,000; dense `baseline_ada_ps_test` for reference.

## Gates

- B1: each new build removes exactly what `dual_u40_v2` removes (params), keeps 19 Q heads and 7,390 MLP channels per layer.
- B2: smoke passes (make_slim's own check).

## Compute

GPU 4 only (session rule 2026-10-29). GPU 4 was busy with another job at plan time, so the runner waits until it is empty (< 100 MiB
twice, 2 min apart) before starting. About 1 h per test500 run, two runs in sequence.

## Caveat to carry into the report

Pre-l0keep, which layer-0 Q heads get cut decides cruise collapses; across arms that differ only in calibration draw or seed, test500
spread was SD 0.26–0.34 (`reports/evaluation/2026-10-13_calibration-variance-is-layer0.html`). Each rule picks its own layer-0 heads,
so a pre-l0keep Δ mixes the rule with its layer-0 choice. The report lists each arm's layer-0 heads and the cruise-bucket Δ, and points
to the l0keep table for the controlled comparison.
