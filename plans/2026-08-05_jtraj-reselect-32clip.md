# Re-select j_traj from jlens_coc32 and re-evaluate

## Hypothesis

The shipped `j_traj` selections were built from 8-clip J scores (`make_slim.py:74` hardcodes
`jlens_coc`), and ~24–27% of their bottom-20% picks change identity under the 32-clip estimate
(`jlens_coc32`, split-half rho 0.977/0.954). H: re-selecting with the cleaner scores preserves —
and at r30 possibly improves — the label-free criterion's parity with `cocsafe`
(original open-loop head-to-head: j_traj_r20 − cocsafe_r20 dNLL −0.000, p=0.96; r30 −0.004).

## Setup — Stage 1 (open-loop criterion sweep, mask-based)

Exact replication of the original protocol with only the jlens factor changed:

```bash
bash experiments/head_analysis/run_retry_host.sh 60 \
    experiments/head_analysis/run_jspace_sweep.py --gpu 2 \
    --exp-id jsweep32_s0 --num-clips 40 --clip-offset 0 --jlens jlens_coc32
bash experiments/head_analysis/run_retry_host.sh 60 \
    experiments/head_analysis/run_jspace_sweep.py --gpu 3 \
    --exp-id jsweep32_s40 --num-clips 40 --clip-offset 40 --jlens jlens_coc32
python experiments/head_analysis/analyze_jsweep.py \
    --shards jsweep32_s0 jsweep32_s40 --exp-id jsweep32_summary
```

- Same 80 clips (`combined_eval_clips.json` 0–79), ratios 0.2/0.3, K=8, seed 42,
  `importance_v1`. 13 configs incl. baseline.
- The non-J configs (magnitude/traj/coc/cocsafe) select identical masks to the original run —
  they are replication controls. GPUs 2–3 are Blackwell (original shards may have run on Ada);
  if the controls' paired deltas match `jsweep_summary` within CI, cross-run comparisons are
  clean despite the arch change. Within-run comparisons (the primary reading) are unaffected.
- Additional analysis beyond `analyze_jsweep`: per-clip paired comparison of new `j_traj_*`
  vs old `j_traj_*` (same clips, same seeds, across runs) — did the re-selection itself move
  the metric? Ad-hoc script in scratchpad; not repo code.

## Decision gates (pre-registered)

- **G-repro**: control configs' dNLL/dADE medians reproduce `jsweep_summary` within the paired
  bootstrap CI. Fails → cross-run readings are void; only within-run comparisons reported.
- **G-parity**: `j_traj_r20(32)` vs `cocsafe_r20` head-to-head |dNLL med| ≈ 0 with p > 0.05 →
  label-free parity survives the noise fix. This is the headline gate.
- **G-improve**: `j_traj_r30(32)` vs old `j_traj_r30` paired per-clip: dNLL mean < 0 with CI
  excluding 0 → the cleaner selection measurably helps where the old one lagged.

## Stage 2 (conditional, separate approval — closed-loop)

Only if Stage 1 passes G-parity: add `--jlens` flag to `make_slim.py` (default `jlens_coc`,
preserving provenance of existing checkpoints), build `slim_j_traj32_r20` (~16 GB, ~1 h GPU),
run the alpasim matrix (drivers need Ada 4–7, currently 100% busy) + the three closed-loop
analyses vs the existing `matrix_j_traj_full_r20` logs. Deferred: separate GPU commitment,
requires Ada availability.

## Cost

Stage 1: 2 idle Blackwell GPUs (2–3), ~6–7 h wall-clock (13 configs × 40 clips × K=8 per shard,
run in parallel). Stage 2: ~1 h build + multi-hour 6-GPU alpasim run.

## Stage 1b (amendment 2026-08-06, approved): KV-axis isolation

Stage 1 result: width selection is noise-robust (max-rank guardrail), but the KV-1 drop flips in
11/36 layers between `jlens_coc` and `jlens_coc32` (8 candidates, ranks quantized to 1/7, exact
ties in layers 1/25/35) — an axis the VLM-width-only sweep never tested.

`run_kviso.py`: hold the j_traj r20 **width** masks fixed (built from `jlens_coc32`) and vary
only the KV-group drop. Four configs, same 80 clips / seeds / K=8 protocol, VLM-side masks only
(both KV variants treated identically, so the paired kv32−kv8 difference isolates the choice):

  baseline / width32 (no KV drop) / width32_kv8 / width32_kv32

Gate: kv32−kv8 paired dNLL ≈ 0 (p > 0.05) → KV axis insensitive to the re-estimate, Stage 2
unnecessary. kv32 significantly better → rebuild `slim_j_traj32_r20`, reconsider closed-loop.
Secondary: (kv* − width32) sizes the KV-drop cost itself. Cost: 1 GPU, ~1–1.5 h.
