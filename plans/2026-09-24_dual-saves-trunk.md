# Plan: do the trunk units that dual keeps but a single criterion drops carry the other function? (2026-09-24)

## Why

The sample-level analysis (`paper-analysis/2026-09-24_arm_failure_cases.md`, `outputs/arm_failure_cases_v1`)
found that the trajectory-only arm's damage is almost purely action-channel (+0.15 m with the GT CoC
teacher-forced, concentrated in decel_stop and hard turns) and that the dual arm removes it (+0.05)
**without keeping more decel_stop trajectory importance than the trajectory arm does** (calibration
mass removed 0.172 vs 0.172, Q trunk). What dual keeps and the trajectory arm drops are units the
CoC score ranks high. Hypothesis H: those units carry trajectory function that the first-order
trajectory score under-values (its trunk gradient is 90% at vision positions and dominated by a
few turn clips), so removing them from the dense model hurts the flow-matching loss and minADE even
with the reasoning held fixed. The mirror for the CoC arm: the units it drops and dual keeps carry
trajectory function with no language cost. The band-level version already exists (trunk-only arm
masks, `armheld_v1`: FM +26.1% trajectory-only vs +4.8% dual); this isolates the units.

## Sets (`experiments/head_analysis/make_dual_saves_sets.py` → `outputs/tokabl_sets_trunk_v1`)

Chosen on the dense model's calib_100 scores (`gradanat_v1`) and the shipped kept sets
(`slim_{traj,coc,dual}_u40_v2`), bands trunk21 = layers 0–21 (primary, the paper's trunk) and
mid = 6–21 (robustness, excludes the concentrated first layers).

| set | definition | Q heads / layer | MLP ch. / layer | band mass held (I_traj / I_CoC, trunk21, Q) |
|---|---|---:|---:|---:|
| Straj | dual kept, trajectory-only removed | 2.1 | 709 | 3.3% / 7.4% |
| MStraj | score-matched control: not in Straj, nearest pooled I_traj per unit | 2.1 | 709 | 3.3% / 4.3% |
| RStraj0–2 | random, same count per layer | 2.1 | 709 | 5.6–7.2% / 6.3–7.1% |
| Scoc | dual kept, CoC-only removed | 2.0 | 722 | 5.9% / 3.7% |
| MScoc | matched on pooled I_CoC | 2.0 | 722 | 3.1% / 3.8% |
| RScoc0–2 | random, same count | 2.0 | 722 | 5.3–6.4% / 6.2–6.4% |

Straj vs MStraj is the test: same first-order trajectory score, different CoC score. Per-axis
configs (Q alone, MLP alone) in `sets.npz`; joint both-axis masks (as in the arms) in `joint/`.

## Measurement (`run_token_ablation.py`, unchanged)

100 held-out clips (first 100 of `indist_500`, official val; the same clips as `tokabl_v1` /
`armheld_v1`), each set removed from the dense model, paired per clip with the dense model on the
same clip, text and noise: dFM (10-step GT-anchored flow-matching loss on the masked cache), dNLL
(CoC NLL of the dense rollout teacher-forced through the masked model), dminADE (K = 8 denoisings
on the masked cache; sampled for the S and M sets). Reasoning is held fixed at the dense text, so
dFM / dminADE are the action channel.

## Pre-registered gates (`experiments/head_analysis/analyze_dual_saves.py`)

- G1 (H, action): dFM(Straj) > dFM(MStraj) and > every RStraj_i, paired one-sided Wilcoxon p < 0.01.
- G2 (H, minADE): dminADE(Straj) 95% CI excludes 0 and dminADE(Straj) > dminADE(MStraj), p < 0.05.
- G3 (where): relative dFM of Straj larger on decel_stop + turn clips than on cruise + accel,
  one-sided Mann–Whitney p < 0.05.
- G4 (mirror): dFM(Scoc) > MScoc and every RScoc_i (p < 0.01) and dNLL(Scoc) not above the random
  mean (one-sided p ≥ 0.05).
- G5 (reported): dNLL(Straj) vs random — expected above random (these are I_CoC-high units).

Reading: G1 + G2 pass → the trajectory function depends on units the trajectory score cannot see,
which is why the union beats the trajectory criterion on its own objective (the paper's "dual"
argument closes causally). G1 fails while G5 passes → those units are language-only in the trunk,
and dual's action advantage has to be attributed to the trunk–late interaction instead (say so).
G4 pass → dual's advantage over the CoC criterion is trajectory units with no language cost.
Per-axis and joint results are both reported; the joint (both axes) is the arm-like condition.

## Commands (launch only on the user's go; Ada 4–7 free at 17:30 KST)

```bash
cd /home/cvlab21/project/chan/alpamayo-model-compression/.claude/worktrees/dual-saves-trunk
export ALPAMAYO_REPO=$PWD
# run 1: per-axis sets, 40 configs (16 sampled at K=8), 4 shards, ~45 min
for s in 0 1 2 3; do
  nohup bash experiments/head_analysis/run_retry_host.sh 90 experiments/head_analysis/run_token_ablation.py \
      --sets tokabl_sets_trunk_v1 --exp-id tokabl_trunk_v1_s$s --shard $s --n-shards 4 --gpu $((4 + s)) \
      > outputs/tokabl_trunk_v1_s$s.launch.log 2>&1 &
done
# run 2: joint both-axis masks, trunk21 only, 10 configs (all sampled), queued behind run 1 on the same cards, ~20 min
J=outputs/tokabl_sets_trunk_v1/joint
MASKS=$(for n in Straj MStraj RStraj0 RStraj1 RStraj2 Scoc MScoc RScoc0 RScoc1 RScoc2; do printf "trunk21_%s=%s/both_trunk21_%s.npz " $n $J $n; done)
for s in 0 1 2 3; do
  nohup bash experiments/head_analysis/run_retry_host.sh 150 experiments/head_analysis/run_token_ablation.py \
      --exp-id tokabl_trunkjoint_v1_s$s --shard $s --n-shards 4 --gpu $((4 + s)) --arm-masks $MASKS \
      > outputs/tokabl_trunkjoint_v1_s$s.launch.log 2>&1 &
done
# analysis (CPU)
.venv/bin/python experiments/head_analysis/analyze_dual_saves.py \
    --per-axis tokabl_trunk_v1_s0 tokabl_trunk_v1_s1 tokabl_trunk_v1_s2 tokabl_trunk_v1_s3 \
    --joint tokabl_trunkjoint_v1_s0 tokabl_trunkjoint_v1_s1 tokabl_trunkjoint_v1_s2 tokabl_trunkjoint_v1_s3 \
    --out dual_saves_trunk_v1
```

Cost: `tokabl_v1` ran 53 configs (25 sampled) at 140 s per clip; run 1 is 40 configs (16 sampled)
≈ 100 s per clip → 25 clips per shard ≈ 45 min; run 2 ≈ 50 s per clip ≈ 20 min; 4 Ada cards,
about 70 min end to end. Disk: a few MB per shard.

## Outputs

`outputs/tokabl_trunk_v1_s{0..3}`, `outputs/tokabl_trunkjoint_v1_s{0..3}`,
`outputs/dual_saves_trunk_v1/{summary.txt,metrics.json,config.json,plots/}`; findings appended to
`paper-analysis/2026-09-24_arm_failure_cases.md`.

## Part 1 result (per-axis, 2026-09-24): H not supported in the dense model

Removing Straj from the dense model leaves the action channel unchanged on every band and axis
(dFM +0.2–0.3%, dminADE +0.025 MLP / −0.039 Q, all CIs include 0; random sets of the same size
hurt FM more, up to +2.7% for Q heads). Only the language channel moves: MLP Straj dNLL +1.7%
against −0.1% for the matched control. G1–G4 FAIL. Joint (both-axis) run pending.

## Part 2: the same question in the PRUNED context (add-back)

A unit's function can be invisible when every redundant partner is intact (the 2026-09-21 round:
one-edge-at-a-time knockouts understate redundant edges). The decisive form of the question is
therefore: inside the trajectory-only arm, does switching the Straj units back on repair its
trajectory damage more than switching on the same number of other dropped units?
Masks: `experiments/head_analysis/make_addback_masks.py` → `outputs/tokabl_sets_trunk_v1/addback/`
(controls drawn from the arm's own removed trunk units: MStraj matched per unit on pooled
I_traj, three random draws; every add-back restores +46 Q heads and +15,592 channels).

Gates (paired per clip, d = config − dense on the same clip):
- A1: d(trajarm+Straj) < d(trajarm+MStraj) and < every d(trajarm+RStraj_i) on FM, one-sided
  Wilcoxon p < 0.01; A2: the same on minADE, p < 0.05.
- A3 (reported): NLL recovery of trajarm+Straj vs the controls (expected largest for Straj).
- A4 (mirror): d(cocarm+Scoc) < controls on FM (p < 0.01) with NLL recovery no larger than
  the controls'.
Reading: A1 + A2 pass → the union's extra trunk units carry trajectory function that only shows
once the arm's redundancy is gone (H holds in the pruned context, not in the dense one). Fail →
the trunk units dual keeps over the trajectory arm are language-only; dual's action-channel
advantage is attributed to the trunk–late interaction and reported as such.

```bash
cd /home/cvlab21/project/chan/alpamayo-model-compression/.claude/worktrees/dual-saves-trunk
A=outputs/tokabl_sets_trunk_v1/addback
for s in 0 1 2 3; do
  ALPAMAYO_REPO=$PWD nohup bash experiments/head_analysis/run_retry_host.sh 150 experiments/head_analysis/run_token_ablation.py \
      --exp-id tokabl_addback_v1_s$s --shard $s --n-shards 4 --gpu $((4 + s)) --arm-masks \
      trajarm=$A/trajarm.npz trajarm+Straj=$A/trajarm+Straj.npz trajarm+MStraj=$A/trajarm+MStraj.npz \
      trajarm+RStraj0=$A/trajarm+RStraj0.npz trajarm+RStraj1=$A/trajarm+RStraj1.npz trajarm+RStraj2=$A/trajarm+RStraj2.npz \
      cocarm=$A/cocarm.npz cocarm+Scoc=$A/cocarm+Scoc.npz cocarm+MScoc=$A/cocarm+MScoc.npz \
      cocarm+RScoc0=$A/cocarm+RScoc0.npz cocarm+RScoc1=$A/cocarm+RScoc1.npz cocarm+RScoc2=$A/cocarm+RScoc2.npz \
      > outputs/tokabl_addback_v1_s$s.launch.log 2>&1 &
done
```
Cost: 12 whole-arm configs, all sampled at K = 8 ≈ 60 s per clip → 25 clips per shard ≈ 25 min on
Ada 4–7. Analysis: `analyze_dual_saves.py --addback tokabl_addback_v1_s0 … --out dual_saves_addback_v1`
(to be added to the analyzer).
