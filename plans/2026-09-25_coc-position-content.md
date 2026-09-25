# Plan: content and causality behind the CoC-position disagreement (2026-09-25)

Follow-up of `plans/2026-09-25_coc-position-disagreement.md`. What is settled: at generated-CoC
positions the two scores rank Q heads at 0.36 raw (0.41 corrected; 0.42 / 0.47 in layers 0–21)
against 0.72–0.86 elsewhere; not noise, modality, clip mixture or the loss path (the own-token and
cross-position parts of `I_CoC` rank heads alike, ρ 0.9); the CE-favoured heads read the earlier
CoC tokens and the FM-favoured heads read the ego history (attention census, observational).
Two things are missing: whether the disagreement is a *token-weighting* effect inside the CoC
(the two losses weight different CoC tokens) or a *content* effect at the same tokens, and a
causal test of the reader story. The residual-gradient cosine is ≈ 0 at every position type
(vision +0.006, prompt +0.003, CoC −0.005), so gradient direction alone does not single out CoC
positions and is not tested again.

## A. Per-position anatomy inside the CoC (`run_coc_position_anatomy.py`, calib_100)

One typed gate per generated-CoC position (5 + n_coc types), one CE and one full-seed FM
backward per clip as in the anatomy: for every Q head its signed contribution at every CoC
position under each loss; MLP contributions summed over three sub-spans (head clause = tokens of
the first three words, rest, end token); the residual-gradient norm of each loss at every CoC
position and layer; the decoded token of every position.

Gates (`analyze_coc_position_anatomy.py`, Q heads, bands 0–21 / 22–34, raw agreement on the
100-clip means with split-half corrected values alongside):
- A0: the sum over positions reproduces the anatomy's CoC row (per-layer Spearman ≥ 0.99).
- A1 (token weighting): the share of each loss's CoC-position gradient norm on the head clause /
  rest / end token, per band. Reported; the prediction is that FM concentrates on the head
  clause and the end token more than CE does.
- A2 (same tokens): agreement of `I_traj` and `I_CoC` restricted to the same sub-span (head
  clause; rest; end token). Reading: ≥ 0.70 corrected on some sub-span → the disagreement is a
  within-CoC token mixture and the position story extends one level down; < 0.50 on every
  sub-span → the two losses want different content from the same tokens.
- A3 (single positions): agreement across heads at one and the same CoC position (per clip,
  layer, position; averaged), so that aggregation over positions and clips is removed entirely.
  If even single-position agreement stays low, content is the only remaining explanation.

## B. Double dissociation at CoC positions (`make_coc_position_masks.py` → `run_coc_position_ablation.py`, held-out 100 clips)

Per layer the k = 8 heads with the largest rank(`I_traj`@CoC) − rank(`I_CoC`@CoC) (T, FM-favoured)
and the k smallest (C, CE-favoured), plus three random sets of k per layer, each for all layers
and for layers 0–21 only. Each set's heads are switched off at the generated-CoC positions only
(typed gate = 0 at the CoC type), on the first 100 clips of `indist_500` (the held-out clips of
`tokabl_v1`), reading the CoC NLL of the dense rollout, the FM loss and minADE (K = 8), paired
with the unmasked model.

Gates (`analyze_coc_position_ablation.py`, paired one-sided Wilcoxon over clips):
- D1 (language): dNLL(C) > dNLL(T) and > every random set, p < 0.01, for the all-layer sets.
- D2 (action): dFM(T) > dFM(C) and > every random set, p < 0.01, for the layers-0–21 sets (the
  late band is known to be FM-insensitive); the all-layer sets reported alongside.
- D3 (reported): dminADE in the same directions.
- Reading: D1 and D2 pass → the heads each score favours at CoC positions carry that objective's
  function there (a crossed dissociation), which turns the reader story from correlation into
  cause. D1 passes and D2 fails → CE-favoured heads are causal for the text but the FM-favoured
  heads' CoC-position outputs are not needed for the action (the expert's dependence on the CoC
  representation runs through other heads or other positions); say so.

## Commands (launch on the user's go; Ada 4–7)

```bash
cd /home/cvlab21/project/chan/alpamayo-model-compression/.claude/worktrees/dual-saves-trunk
.venv/bin/python experiments/head_analysis/make_coc_position_masks.py --exp-id coc_posabl_sets_v1 --k 8
# A: 4 shards, ~15 s per clip (rollout + CE + expert pass + VLM backward, 44 GB) -> ~7 min
for s in 0 1 2 3; do
  ALPAMAYO_REPO=$PWD nohup bash experiments/head_analysis/run_retry_host.sh 30 experiments/head_analysis/run_coc_position_anatomy.py \
      --exp-id coc_posanat_v1_s$s --shard $s --n-shards 4 --gpu $((4 + s)) > outputs/coc_posanat_v1_s$s.launch.log 2>&1 &
done
# B: queued behind A on the same cards (11 configs, 7 sampled at K=8, ~40 s per clip -> ~17 min)
for s in 0 1 2 3; do
  ALPAMAYO_REPO=$PWD nohup bash experiments/head_analysis/run_retry_host.sh 120 experiments/head_analysis/run_coc_position_ablation.py \
      --exp-id coc_posabl_v1_s$s --shard $s --n-shards 4 --gpu $((4 + s)) > outputs/coc_posabl_v1_s$s.launch.log 2>&1 &
done
.venv/bin/python experiments/head_analysis/analyze_coc_position_anatomy.py --shards coc_posanat_v1_s0 coc_posanat_v1_s1 coc_posanat_v1_s2 coc_posanat_v1_s3 --out coc_posanat_v1
.venv/bin/python experiments/head_analysis/analyze_coc_position_ablation.py --shards coc_posabl_v1_s0 coc_posabl_v1_s1 coc_posabl_v1_s2 coc_posabl_v1_s3 --out coc_posabl_v1
```

Not in this plan: direct logit attribution (equivalent to the own-token path already measured)
and the ego-state probe (n = 100 clips is too few for 4096-d probes; if wanted, pool the calib,
val100 and test100 anatomy clips and probe the head's contribution to the cache V).
