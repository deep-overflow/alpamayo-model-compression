# u50 closed loop on Hard100 (plan, 2026-11-15)

User, 11-15: "u50에서 hard100도 실험을 진행해보자" → "일단 MoOP부터 진행해줘" → "4-7번 GPU만 사용해줘."

## Purpose
- Origin150 (easy draw) showed MoOP u50 ahead of every u50 baseline and of the ICLR dense run
  (`reports/2026-11-12_u50-closedloop-origin150-ci.md`).
- Question: does that order hold on the hard set the paper also reports (Table 2 / 8 Hard100)?

## Setup (same as the u50 Origin150 arms unless noted)
| item | value |
|---|---|
| scenes | Hard100 = `outputs/scene_difficulty/hard100_suite.csv` (the ICLR Hard100 list), 2 rollouts each |
| harness | patched hand-over harness, `DRIVER_OMP_THREADS=8`, parent suite `public_2601` |
| cards | GPUs 4–7 (Ada) only, user rule for this run; card per scene recorded in the shard logs |
| arms | MoOP u50 `slim_dual_u50_v2` first; the other u50 arms (Tyr `slim_tyr_u50_r`, LLM-Pruner `slim_lp_r60`, FM-only `slim_traj_u50_v2`, CE-only `slim_coc_u50_v2`, Wanda `slim_wanda_u50_v2`) only when the user asks |
| dense reference | ICLR Hard100 dense `h100_merged_baseline` (pre-fix harness), as chosen for Origin150 |
| runner | `iclr27-rebuttal/code/run_u50_hard100.sh <cfg>` → `alpasim-runs/u50h_merged_<cfg>`, logs `outputs/iclr27-rebuttal/u50_hard100/` |

## Preparation done 11-15
- 27 of the 100 Hard100 usdz were missing from the launcher's scene cache (`ad_vla/data/nre-artifacts/all-usdzs`).
- They were hardlinked from `ad_vla/data/alpasim/nre-artifacts/all-usdzs`. This is the same filesystem, so it needs no extra space.
- Each file is a valid zip and matches its `scenesets/chan_hard100_all` copy in size.

## Gates
- **G1** (`s600_smoke_gates.py`): every scene scored and no non-gate failure.
  - On Hard100 this fails by design. The simulator's route sanity check aborts `4bad2f63` and `adb899bd` in every arm, and `3b934c9a` and `ceb3d0d5` in some arms (ICLR runs).
  - **G1h** replaces it: every non-gate failure must be that sanity check.
- **G2**: the hand-over fix holds.
- **Aborted rollouts:** they keep the forced score 0, as in the paper. The count per arm is reported, because it is arm-dependent.

## Confounds to state in the report
- **Dense reference:** it ran on the pre-fix harness, while the u50 arms run on the patched one (same caveat as Origin150).
- **Aborted rollouts:** they differ by arm (see above).
