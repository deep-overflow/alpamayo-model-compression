# Plan: fill the missing Ada cells of the Table 1 / Table 4 CI report

2026-11-09 · follow-up to `reports/2026-11-09_openloop-lingoqa-ci.md` (user: "위에서 빠진 내용 채우기 위한 실험 진행해줘")

## Purpose

The CI report left two gaps:

1. **LLM-Pruner** has no Ada per-clip open-loop or LingoQA rows, so its Table 1 / Table 4 cells have no CI. Its
   paper checkpoint still exists: `/mnt/nvme1n1/ad_vla/data/alpasim/drivers/lp_r50`. It is the checkpoint behind the
   Ada NLL row and the 10-clip Ada OOD run `outputs/lp_r50_oodval`.
2. **FM-only / CE-only** Ada rows store 6.4 s only (run before the horizon fields existed), so their 1.6 / 3.2 s
   cells are n/a.

Hypothesis: Ada inference is deterministic on the same card type, so a re-run reproduces the stored values bit for
bit and the new rows can stand in for the paper rows.

## Runs (RTX 5880 Ada, cards 4-7, one process per card)

| arm | model | sets | runner |
|---|---|---|---|
| LLM-Pruner | `/mnt/nvme1n1/ad_vla/data/alpasim/drivers/lp_r50` | Test500 (4 shards), OOD-Val262 (2 shards), LingoQA + Lingo-Judge | `run_baseline.py`, `run_lingo_vqa.py`, `score_lingo_vqa.py` |
| FM-only | `outputs/slim_traj_u40_v2` | Test500, OOD-Val262 | `run_baseline.py` |
| CE-only | `outputs/slim_coc_u40_v2` | Test500, OOD-Val262 | `run_baseline.py` |

Protocol = the paper rows: k = 8, seed 42, max_gen 256, rollout only (`--no-tf` on OOD; the original FM/CE OOD runs
also ran the GT-CoC teacher-forced condition, which does not touch the rollout). Outputs
`outputs/iclr27-rebuttal/ada_fill/<arm>_<set>/`, LingoQA `.../lingo_vqa[_scores]_lp_r50/`. Worker
`iclr27-rebuttal/code/run_ada_fill.sh` (mkdir-claimed queue, LLM-Pruner first).

## Gates

- **G1 (FM-only / CE-only)**: on all 762 clips per arm, the new `ade_rollout_k`, `fde_rollout_k` and `gen_coc` equal
  the stored rows exactly. Pass -> the CI report reads these arms from the new runs (adds 1.6 / 3.2 s). Fail -> report
  the difference, keep the old 6.4 s rows.
- **G2 (LLM-Pruner = paper row)**: the 10 OOD-Val262 clips in `outputs/lp_r50_oodval` are bit-identical, and the 6.4 s
  means reproduce Table 1 (1.451 / 3.889, 1.662 / 4.153 to 3 decimals) and LingoQA 44.8. Pass -> LLM-Pruner joins
  the Ada tables. Fail -> report it as a separate row with the mismatch stated.

Then: `ci_openloop_lingoqa.py` reads the new rows, re-run, update the report (the A100 block stays as a hardware
cross-check).
