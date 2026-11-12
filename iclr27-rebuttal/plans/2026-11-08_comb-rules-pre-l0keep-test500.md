# Plan: combination rules under the paper's (pre-l0keep) recipe, test500 (2026-11-08)

User ask: "l0keep 이전 방식에서 성능을 보여주는 게 좋을 것 같아. rank-mean, rank-max, raw-mean, raw-max를 test 500에서 비교해줘."

**Status: done 2026-11-12 23:56 KST.** Report `iclr27-rebuttal/reports/2026-11-12_table14-combination-rules.md`.

- **Run.** Ada 4–7, 22:02–23:56 KST, after the u50 closed-loop queue's last arm freed the cards (the closed-loop session was
  told). 15/15 tasks, 0 failed.
- **Checks.** B0/B1 PASS again at launch. The rank-mean Test500 rerun equals the September run on 500/500 clips.
- **Table 14** (LingoQA / Test500 / OOD-Val262):

  | rule | LingoQA | Test500 | OOD-Val262 |
  |---|---|---|---|
  | max-rank | 68.8 | 0.950 | 1.119 |
  | mean-rank | 67.0 | 0.960 | 1.160 |
  | max-raw | 59.2 | 0.974 | 1.173 |
  | mean-raw | 14.0 | 1.216 | 1.373 |

  Mean-raw collapses: cruise / turn losses and CoC degeneration 9.4%, the signature of a layer-0 head-choice collapse.

(Earlier status, kept for the record: builds done on CPU at 02:00 KST, the launch waited for the closed-loop queue.)

- **Builds.** The three recipes were built on CPU with `code/build_recipes_cpu.py` (user: "바로 평가만 하면 되도록 준비해줘").
  - Why CPU: every GPU was busy, and a `--no-state` recipe needs no GPU.
  - What it runs: make_slim's own `build_masks`, `apply_surgery` and `save_slim`, with `--importance importance_v2`. Only the
    one-clip smoke test is skipped.
  - Outputs: `outputs/iclr27-rebuttal/comb_pre_l0keep/slim_{dualsum,rawabsavg,rawabsmean}_u40_v2/`. Log in `logs/build_cpu.log`.
- **Gates.**
  - B0 PASS: the CPU rebuild of `dualsum_u40_v2` equals the September GPU build unit for unit.
  - B1 PASS: all three remove 2,657,452,032 params (= MoOP) and keep 19 Q / 7,390 MLP per layer; the expert is as in MoOP.
  - Layer-0 Q heads shared with rank-max: rank-mean 17/19, raw-mean 17/19, raw-max 18/19.
  - Log in `logs/gates_b0_b1.log`.
- **Launch** (from the main checkout, once the user has cleared a window on GPUs 4–7). The runner skips the builds because
  `slim_meta.json` exists, re-runs B0/B1, then claims 6 Test500 + 6 OOD-Val262 shards and 3 LingoQA runs on the four cards
  (about 1.5 h):
  ```
  CARDS="4 5 6 7" setsid nohup bash iclr27-rebuttal/code/run_comb_pre_l0keep.sh \
      > outputs/iclr27-rebuttal/comb_pre_l0keep_launch.log 2>&1 < /dev/null &
  ```
- **Table afterwards:** `.venv/bin/python iclr27-rebuttal/code/table14_comb_rules.py`.

## Amendment 2026-11-12 (user: fill the paper's Table 14, "Combination rules with 95% confidence intervals")

- **Scope.** Table 14 has three columns: LingoQA, Test500 minADE@6 and OOD-Val262 minADE@6. So OOD-Val262 is added to the runner:
  2 shards per arm, rollout only (`--no-tf`), the same protocol as the paper's OOD rows.
- **Table script.** `code/table14_comb_rules.py` writes the table, paste-ready LaTeX rows, all horizons and paired deltas to
  `outputs/iclr27-rebuttal/table14_comb_rules/`. Missing runs print `[x]`.
  - Every cell uses the estimator of the paper's max-rank cells: 200k clip bootstrap for open loop, 200k segment bootstrap
    for LingoQA.
  - The dry run reproduces the max-rank row exactly (68.8 [64.4, 73.0], 0.950 [0.869, 1.034], 1.119 [0.958, 1.307]).
  - It already fills mean-rank Test500 from the September run: 0.960 [0.874, 1.051], +0.010 [−0.029, +0.053] vs max-rank.
- **Why not launched.**
  - All eight cards run AlpaSim for the u50 closed-loop queue (`run_u50_queue3.sh` → `run_u50_queue4.sh`). Cards 4–7 are
    expected to stay busy until about 11-13.
  - That queue starts its next arm as soon as a card frees. This runner's own `wait_empty` could win a gap and take 40 GB
    from a card a closed-loop arm needs.
  - So it is launched only in a window the user arranges.
- **Cost.** About 1.5 h on four Ada cards: 3 builds, 6 Test500 shards (~30 min each), 6 OOD shards (~15 min each), 3 LingoQA
  runs (~13 min each).

## Amendment 2026-11-10 (user: "실험을 다시 해야 할 것 같아. l0keep 이전 버전으로 rank-mean, raw-max, raw-mean을 진행해줘",
"test500과 lingoqa 결과가 필요해")

- **Scope**
  - LingoQA is added for every rule, read the same way as the paper's Table 4.
    - `run_lingo_vqa.py`: greedy, max_gen 128, 500 questions.
    - `score_lingo_vqa.py` (Lingo-Judge) writes `scored.json`.
    - CIs resample the 100 segments (`ci_openloop_lingoqa.lingo_block`).
  - rank-max and dense reuse the paper's Ada rows: `dual_u40_v2_ps_test`, `lingo_vqa_slim_dual_u40_v2` (68.8%), `baseline_ada_ps_test`, `lingo_vqa_baseline` (73.2%).
- **rank-mean is re-run** on test500 and newly measured on LingoQA.
  - It runs from the September build `outputs/slim_dualsum_u40_v2`.
  - Its test500 rerun is compared clip by clip with `dualsum_u40_v2_test` (same card type = bit-identical expected).
  - Gate B0 checks that today's make_slim rebuilds that selection unit for unit.
- **Outputs.** New builds and runs go under `outputs/iclr27-rebuttal/comb_pre_l0keep/`:
  - builds: `slim_{rawabsavg,rawabsmean,dualsum}_u40_v2`
  - test500: `{rank_mean,raw_mean,raw_max}_test500`
  - LingoQA: `lingo_vqa_<arm>`, `lingo_vqa_scores_<arm>`
  - analysis: `outputs/iclr27-rebuttal/comb_pre_l0keep_analysis/`
- **Compute**
  - Cards come from `CARDS`, default Ada 4-7. The "GPU 4 only" rule above has no recorded source.
  - Each card is used only once it is empty: < 100 MiB twice, 2 min apart.
  - Builds and gates B0/B1 run on the first card. After that, one worker per card claims tasks with mkdir: test500 in 2 shards per arm, then LingoQA.
  - About 6 × 30 min + 3 × 13 min ≈ 1 h 10 min on four cards.
- **Launch**
  ```
  CARDS="4 5 6 7" bash iclr27-rebuttal/code/run_comb_pre_l0keep.sh > outputs/iclr27-rebuttal/comb_pre_l0keep_launch.log 2>&1
  ```
- **Analysis**
  ```
  .venv/bin/python iclr27-rebuttal/code/analyze_comb_pre_l0keep.py
  ```
  - Missing arms are skipped, so it also works mid-run.
- **Dry run (11-10, no GPU)**
  - The analysis was run with rank-mean pointed at the September run. It reproduces the paper's rows: rank-max test500 0.950 / 2.546, LingoQA 68.8%; dense 0.842 / 73.2%.
  - The September rank-mean is +0.010 [−0.029, +0.053] vs rank-max on minADE@6 6.4 s and keeps 17/19 of rank-max's layer-0 heads.

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
