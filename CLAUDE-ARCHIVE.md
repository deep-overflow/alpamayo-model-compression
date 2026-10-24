# CLAUDE-ARCHIVE.md

Archived reference for experiment harness, runner tracks, closed-loop evaluation setup, named configurations, and past workflows. This is detailed reference material from prior research phases.

## Experiment harness (`experiments/head_analysis/`)

### Shared libraries (full documentation)

- `analysis_lib.py` — builds fused prompts exactly as release inference does, identifies token spans (vision / traj-history / sink / text, per-camera), switches attention impl to eager for stat capture.
- `prune_lib.py` — Phase P1: multiplicative unit gates + dual-objective Taylor importance (CoC NLL vs trajectory flow-matching MSE). KV groups are scored on the cache tensors directly because Qwen3's k_norm makes a per-head gate on k_proj carry no gradient.
- `mask_lib.py` — Phase P2/P3: 0/1 forward-pre-hook masks on o_proj input (per Q head) and down_proj input (per MLP channel); masked is functionally identical to removed. Selection is per-layer so a width sweep does not silently become a depth sweep.
- `jlens_lib.py` — **rollout-free** criterion. Jacobian lens `J_l = E[dh_final/dh_l]` with rows of `W_U` as cotangents, so one backward per dictionary token yields that token's lens vector at *every* layer.
- `eval_lib.py` — multi-sample minADE/minFDE, geometry-derived scenario buckets from the GT path, paired bootstrap CIs, and `coc_degenerate()` whose thresholds deliberately mirror `analyze_alpasim.coc_stats`.
- `slim_lib.py` — physical removal. Mask → real weight surgery (q_proj rows / o_proj cols, gate/up rows / down cols). Checkpoints are `torch.save` state_dicts + `slim_meta.json` of kept indices; reload with `load_slim()`.
- `run_streamprop.py` / `analyze_streamprop.py` — decomposition on propagated quantities (accumulated hidden state and per-layer KV cache the expert reads).
- `run_streamerr.py` / `analyze_streamerr.py` — reconstruction error decomposed by token type.
- `tyr_lib.py` — layer-output reconstruction (ported from Týr-the-Pruner / OSSCAR). `HessianHook` accumulates `H = Σ x xᵀ` at o_proj/down_proj input.
- `run_gradient_anatomy.py` / `analyze_gradient_anatomy.py` — exact splits of the shipped gate gradient by token type / cache port.

### Runner tracks

| track | runner | analysis |
|---|---|---|
| P1 importance (dual-objective Taylor) | `run_importance.py` | `analyze_importance.py`, `analyze_cvar.py`, `analyze_agreement.py` |
| P2 mask sweeps | `run_ablation.py`, `run_expert_ablation.py` | `analyze_ablation.py` |
| P3 combined configs | `run_eval.py`, `run_integrated.py`, `run_cocsafe.py`, `run_kvfix.py` | `analyze_eval.py` |
| criterion × allocation grid | `run_grid.py` | `analyze_grid.py` |
| J-space (rollout-free) | `run_jlens.py`, `run_jspace_sweep.py`, `run_kviso.py` | `analyze_jspace.py`, `analyze_jsweep.py` |
| physical removal + latency | `make_slim.py`, `verify_slim.py`, `bench_fast*.py` | `analyze_slim.py`, `analyze_fastpipe.py` |
| OOD / recovery | `eval_ood.py`, `train_lora.py`, `eval_recovered.py` | `analyze_ood.py` |
| closed-loop (alpasim) | `launch_alpasim_matrix.sh`, `launch_alpasim_shards.sh` | `analyze_alpasim.py`, `analyze_collisions.py` |
| layer-output reconstruction × calibration stream (RAC) | `run_racfit.py`, `launch_racfit.sh` | `analyze_racfit.py` |

## Closed-loop evaluation (alpasim)

Open-loop minADE hid the failure that closed-loop rollouts exposed, so headline configs get run in the
sibling **alpasim** repo at `/home/cvlab21/project/chan/alpasim`, with logs under
`/home/cvlab21/project/chan/alpasim-runs/matrix_<config>/`.

```bash
bash experiments/head_analysis/launch_alpasim_matrix.sh <n_scenes> <n_rollouts> [config ...]
```

A slim checkpoint becomes a driver by hardlinking `outputs/<cfg>/` into
`/mnt/nvme1n1/ad_vla/data/alpasim/drivers/<cfg>/`. The wizard receives it as
`driver.model.checkpoint_path=/mnt/drivers/<cfg>` (baseline passes no override).

**Sharding one config over several GPUs**:
```bash
launch_alpasim_shards.sh <config> <n_scenes> <n_rollouts> "<gpus>"
```

Then merge with `merge_alpasim_shards.py`.

### Scene suites

`public_2601` (913 scenes) and `public_2604` (1,606) share 159 scene_ids but **zero artifact uuids**.
Results from the two are not comparable; they need separate `PREFIX`es.

**Difficulty scoring offline from usdz** (no GPU): `hard_score = z(v_mean) + z(yaw_total_deg)`.
Hard100 band (80–100th percentile) sits at ≈0.50; easy vs hard differ by factor 1.55× on effect size
against 1.12× rise in σ — half the scenes for the same power.

Throughput: 12.5 min/scene unpruned, 11.6 min/scene for 8.4B slim on four stacks, `OMP_NUM_THREADS=8`.

## Evaluation harness (`experiments/evaluation/`)

A second, larger data protocol drawn directly from official dataset splits, superseding `split.json`.
`outputs/eval_sets/` holds the manifests:

| set | n | drawn from | role |
|---|---:|---|---|
| `indist_500` | 500 | official val | in-distribution |
| `test_500` | 500 | official test | in-distribution, held out |
| `ood` | 1,533 | `reasoning/ood_reasoning.parquet` | long-tail, carries `gt_coc` reference reasoning |
| `calib_100` | 100 | official train | **calibration only** — importance/J measured here, never evaluated |

All four are pairwise disjoint. Seeds from clip, not loop index: `sc.clip_seed(seed, clip_id)`.

- `run_baseline.py --set {indist,test,ood} --model {baseline,<slim dir>}` — open-loop runner
- `launch_arms.sh` — flock-guarded job queue
- `analyze_baseline.py`, `analyze_arms.py`
- `run_depth_ablation.py` / `analyze_depth_ablation.py` — per-layer kv-only ablation
- `analyze_calibsize.py` — closed-loop arm-vs-arm matrix (all pairwise paired deltas)
- `coc_action_consistency.py` / `analyze_criterion_aug.py` — RL consistency reward
- `collect_results.py` → `make_master.py` → `push_to_sheet.py` — results spreadsheet

Determinism: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `use_deterministic_algorithms`, cudnn deterministic, TF32 off.

## LLM-Pruner port (`experiments/llm_pruner/`)

External baseline `lp_r50` built in soowon's tree; ported 2026-09-06 for second-order arm.
`lp_core.py`, `lp_objectives.py`, `run_lp_importance.py`, `lp_prune.py`.

**Different from `prune_lib.py`**: differentiates weights (sign preserved), keeps sign while summing over clips.

## Named pruning configs

| config | criterion | allocation | axes touched |
|---|---|---|---|
| `integrated_mag` | trajectory Taylor | late-heavy graded 30/50 + layer-35 kv-only | VLM + expert + KV1 (−3.25B) |
| `cocsafe_full_r20` / `_r30` | dual `max(rank I_traj, rank I_CoC)` | uniform 20% / 30% | VLM + expert + KV1 (−2.01B at r20) |
| `j_traj_full_r20` / `_r30` | `max(rank I_traj, rank J)` — **rollout-free** | uniform | VLM + expert + KV1 |
| `dual_u40_v2` / `j_traj_u40_v2` | dual / rollout-free | uniform 0.398563 | **VLM only** (−2.66B, 24.0%) |
| `traj_u40_v2` / `coc_u40_v2` / `j_u40_v2` | single criterion | uniform 0.398563 | **VLM only** |
| `expert_u25` | expert Taylor | uniform 25% | **expert only** (−0.53B, 4.8%) |
| `dualexp_u40_e25` | dual (VLM) + znorm (expert) | uniform | **VLM + expert** (−3.19B, 28.8%) |
| `dual_u40_l0keep_v2` | dual (VLM fixed Q0) | uniform | **VLM only** (−2.66B, 24.0%) |

All five `*_u40_v2` configs remove exactly 2,657,452,032 params (19/32 Q heads and 7390/12288 channels).
Kept-set overlap: 75.7% (traj–coc), 62.7% (traj–j), 75.6% (coc–j) on Q heads.

`dual_u40_v2` and `j_traj_u40_v2` (one-factor pair): same matched budget, uniform allocation, expert/KV untouched.
Kept sets overlap 84.8% (Q heads) / 83.3% (MLP).

The uniform ratio is **0.3985632694**, not 0.40 — from `run_grid.allocations()` matching `slim_integrated_mag`'s realized budget.

## Plans and reports

Non-trivial work starts with a plan `.md` in `plans/` (`YYYY-MM-DD_topic.md`) that pre-registers decision gates.

Reports are self-contained HTML files in `reports/evaluation/` (76+ entries as of 2026-09-21).
Author a `*_report_template.html` with `PLOT::<key>::<file>.png` placeholders, then inline plots as base64:

```bash
python experiments/head_analysis/build_report.py <template.html> reports/<track>/<out>.html \
    key=$PWD/outputs/<exp_id>/plots [...]
```

Output convention: every run writes `outputs/<exp_id>/` containing `config.json`, `metrics.json`,
`summary.txt`, and `plots/*.png`. Large rebuildable artifacts (slim_state.pt, jlens_vectors.pt) are gitignored.

**Baseline table**: `reports/evaluation/2026-08-11_baseline_table.tex` is the anchor table for the paper's
experimental section: protocol and baseline in one table.

## Fixed data protocol

`outputs/split.json` (seed 20260721): train/val/test = 900/131/150, with 50 calibration clips a subset of train.
Evaluation is paired-seed: baseline and pruned configs share the same sampling seeds so noise cancels.

## Architecture-invariance (2026-09-05)

`baseline` and `dual_u40_v2` were re-run on A100 (KISTI NEURON) over all three sets.
Pruning cost is architecture-invariant (deltas ≤0.011 m). What moves: per-clip values and **discrete threshold metrics**
(CoC degeneracy, LingoQA judge). Keep every arm of a comparison on one architecture; continuous aggregates measured
elsewhere have ~0.01 m error (fifth of gate threshold).
