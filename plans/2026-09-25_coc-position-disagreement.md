# Why the two scores disagree at generated-CoC positions only (2026-09-25)

Stored-data analysis (`experiments/head_analysis/analyze_coc_position_heads.py`, calib_100 anatomy
`outputs/gradanat_v1`, Q heads) of the one exception in `fig4_same_token5_q_head`: with both scores
restricted to one position type, the late-band rank agreement is 0.79 (vision), 0.81 (prompt
text), 0.94 (ego history) but 0.41 at generated-CoC positions, and 0.42 in layers 0–21.

## What is ruled out

- **Sampling noise.** The 0.41 is ceiling-corrected; split-half reliabilities of the CoC-position
  scores are 0.79 / 0.87 (Q). The exception survives the correction.
- **Modality.** Prompt text is text too and agrees at 0.81. The split is not vision vs language.
- **Clip mixture.** At CoC positions the two objectives do weight clips differently — `I_traj`'s
  CoC mass is turn-heavy (40% from 17% of clips, 10 clips carry 54%) and grows with CoC length
  (Spearman +0.28), `I_CoC`'s follows the clip mix (cruise 0.43 / decel_stop 0.19 / turn 0.24) and
  does not (−0.15); across clips the two masses correlate only +0.23. But reweighting either
  objective's clips to the other's masses moves the agreement only from 0.42 to 0.44–0.46 (0–21)
  and 0.36 to 0.37–0.38 (22–34), and restricting both to the same ten heaviest FM clips gives
  0.37 / 0.36. The disagreement is present inside single clips: per-clip within-layer agreement
  is 0.23 at CoC positions against 0.46 at vision positions (0–21; late 0.24 vs 0.30). Averaging
  over clips lifts vision to 0.89 but CoC only to 0.42.
- **Gradient direction.** The per-token cosine between the two losses' residual-stream gradients
  is ≈ 0 at every position type (vision +0.006, prompt +0.003, ego-history −0.016, CoC −0.005,
  layers 7–21), so direction does not single out CoC positions; unit agreement elsewhere comes
  from the magnitude structure (which heads write strongly at those positions).

## What the disagreement looks like

Top / bottom 6 heads per layer by rank(`I_traj`@CoC) − rank(`I_CoC`@CoC):

| group | layers | pooled rank I_traj / I_CoC | kept by traj / coc / dual | own score earned at CoC (traj / CoC) | dominant position under I_traj | under I_CoC |
|---|---|---|---|---|---|---|
| FM-favoured at CoC | 0–21 | 0.63 / 0.52 | 76% / 62% / 70% | 19% / 14% | vision 52%, CoC 20% | vision 45%, prompt 45%, CoC 7% |
| CE-favoured at CoC | 0–21 | 0.35 / 0.46 | 36% / 55% / 45% | 13% / 48% | vision 61% | CoC 58%, vision 29% |
| FM-favoured at CoC | 22–34 | 0.69 / 0.31 | 86% / 32% / 72% | 42% / 38% | CoC 44%, vision 31%, hist 19% | prompt 65%, CoC 28% |
| CE-favoured at CoC | 22–34 | 0.27 / 0.67 | 23% / 87% / 63% | 25% / 77% | vision 59%, CoC 23% | CoC 88% |

The CE-favoured heads are next-token heads: 77% of their language score (late) is earned at CoC
positions and 88% of them are CoC-dominant under `I_CoC`, while under `I_traj` they look like
ordinary vision heads. The FM-favoured heads earn their trajectory score at CoC positions (42%)
but their language score at prompt positions (65% prompt-dominant): their CoC-position output
matters to the expert, not to the next token. Per head, the two objectives' CoC-position
gradients correlate only +0.11 to +0.22 across clips. The union keeps 63–72% of both groups; each
single arm keeps 23–32% of the other objective's group.

## Interpretation (hypothesis, to be tested)

Generated-CoC positions are the only positions where the CE loss is **evaluated**. A head's
output at a CoC position therefore reaches the CE loss by two paths: down its own residual
stream to that position's logits (the own-token path, which nothing else in the model reads),
and through the position's K/V into later CoC queries. The FM loss reaches the same output only
through the K/V the expert reads. At vision, prompt and ego-history positions no loss is
evaluated, so both objectives consume the output through the same K/V entries (later CoC
queries for CE, diffusion tokens for FM) and rank the writers alike. The own-token path is what
the CE-favoured "next-token heads" carry, and it is absent from `I_traj` by construction.

## Proposed tests (GPU, on the user's go)

1. **Path split of `I_CoC` at CoC positions** (one extra CE backward in `run_gradient_anatomy.py`,
   calib_100, ~25 min): detach the K and V of CoC positions in every layer during the CE backward,
   so the typed gate gradient at CoC positions carries the own-token path only; attention path =
   full − direct (exact, linear). Gate: rank(`I_traj`@CoC) vs rank(`I_CoC`@CoC, attention path)
   ≥ 0.7 (like the other positions) while vs the own-token path stays ≤ 0.45. Pass → the exception
   is the loss-evaluation path, and the paper can say the two scores agree wherever they read a
   position through the same interface.
2. **Attention targets at CoC queries** (`run_coc_census.py`, eager-attention census as
   `run_vision_census.py` but for CoC-token queries, same calib_100 clips and rollout seeds as the
   anatomy): per head and layer, the mean attention mass of the CoC queries on sink / vision /
   ego-history / prompt / earlier-CoC / self keys. Analysis `analyze_coc_census.py` joins it with
   the same-position scores. Gates (per band 0–21 and 22–34, one-sided Mann–Whitney over heads):
   G-A FM-favoured heads put more mass on vision + ego-history keys than CE-favoured heads
   (p < 0.01); G-B CE-favoured heads put more on prompt + earlier-CoC keys (p < 0.01). Continuous
   check: within-layer Spearman of each key-group mass with `I_traj`@CoC and `I_CoC`@CoC.
   Reading: both pass → the FM-favoured heads ground the reasoning tokens in the scene for the
   expert, the CE-favoured heads continue the text; that is the mechanism behind the exception.
   One or both fail → the two groups read the same keys and differ in what they write, which the
   path split (test 1) would then have to carry alone.

   ```bash
   cd /home/cvlab21/project/chan/alpamayo-model-compression/.claude/worktrees/dual-saves-trunk
   for s in 0 1; do
     ALPAMAYO_REPO=$PWD nohup bash experiments/head_analysis/run_retry_host.sh 30 experiments/head_analysis/run_coc_census.py \
         --exp-id coc_census_v1_s$s --shard $s --n-shards 2 --gpu $((4 + s)) > outputs/coc_census_v1_s$s.launch.log 2>&1 &
   done
   .venv/bin/python experiments/head_analysis/analyze_coc_census.py --shards coc_census_v1_s0 coc_census_v1_s1 --out coc_census_v1
   ```
   Cost: rollout + one eager forward per clip ≈ 10 s → 50 clips per shard ≈ 10 min on two Ada
   cards; 3 MB of output.

## Test 2 result (2026-09-25, `outputs/coc_census_v1/summary.txt`, 100 clips, 4 shards on Ada 4–7)

Mean attention of the generated-CoC queries by key group (all heads): layers 0–6 prompt 0.59 /
vision 0.17 / earlier CoC 0.15; 7–21 sink 0.38 / vision 0.31 / prompt 0.13 / earlier CoC 0.11;
22–34 **vision 0.60** / prompt 0.15 / sink 0.14 / earlier CoC 0.07 / ego-history 0.02.

| band | group | sink | vision | ego-history | prompt | earlier CoC | self |
|---|---|---:|---:|---:|---:|---:|---:|
| 0–21 | FM-favoured (T-fav) | 0.294 | 0.259 | **0.047** | 0.280 | 0.066 | 0.053 |
| 0–21 | CE-favoured (C-fav) | 0.261 | 0.235 | 0.019 | 0.266 | **0.187** | 0.033 |
| 22–34 | FM-favoured | 0.149 | 0.602 | **0.030** | 0.154 | 0.039 | 0.025 |
| 22–34 | CE-favoured | 0.151 | 0.608 | 0.011 | 0.136 | **0.087** | 0.007 |

- Pre-registered gates: G-B (prompt + earlier CoC, CE-favoured > FM-favoured) PASS in 0–21
  (0.452 vs 0.346, p = 2e-4), FAIL at 0.01 in 22–34 (0.223 vs 0.194, p = 0.04). G-A (vision +
  ego-history) FAIL in both bands (p = 0.042 / 0.069): **vision does not separate the groups** —
  every late head sends ~60% of its CoC-query attention to the scene.
- Post hoc, single key groups (Mann–Whitney over heads): **ego-history** FM-favoured 0.047 vs
  0.019 (p = 4e-10) and 0.030 vs 0.011 (p = 4e-9); **earlier CoC** CE-favoured 0.187 vs 0.066
  (p = 3e-17) and 0.087 vs 0.039 (p = 3e-5); prompt slightly higher for FM-favoured late
  (0.154 vs 0.136, p = 9e-4); vision, sink, self no difference.
- Continuous, within layer (Spearman over heads): earlier-CoC mass vs `I_CoC`@CoC +0.45 / +0.48,
  vs `I_traj`@CoC +0.02 / +0.19; ego-history mass vs `I_traj`@CoC +0.22 / +0.38, vs `I_CoC`@CoC
  −0.04 / −0.03; vision mass vs the rank difference +0.01 / −0.01.

**Reading.** At the reasoning tokens the two objectives value heads with different attention
targets. The CE-important heads read the *preceding reasoning text* (language continuation); the
FM-important heads read the *ego-motion history* (and, late, the prompt) and fold it into the
reasoning-token representations the expert consumes from the cache. Both read the scene equally,
so "looks at vision" is not what distinguishes them; the hypothesis as phrased (vision vs CoC)
holds on the CoC side and is replaced by ego-history on the trajectory side. This is a mechanism
for the exception in `fig4_same_token5_q_head`: the disagreement at CoC positions is between
text-continuation heads and ego-state-grounding heads, which no other position type hosts side by
side. Figure: `outputs/coc_census_v1/plots/coc_census_groups.png` (copied to
`paper-analysis/figures/supplementary/fig8_coc_query_attention.png`).
