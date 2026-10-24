# Why the two scores disagree at generated-CoC positions only (2026-09-25)

Stored-data analysis (`experiments/head_analysis/analyze_coc_position_heads.py`, calib_100 anatomy
`outputs/gradanat_v1`, Q heads) of the one exception in `fig4_same_token5_q_head`: with both scores
restricted to one position type, the late-band rank agreement (raw Spearman on the 100-clip
means; ceiling-corrected in parentheses) is 0.72 (0.79) at vision, 0.75 (0.81) at prompt-text,
0.86 (0.94) at ego-history but 0.36 (0.41) at generated-CoC positions, and 0.42 (0.47) in layers
0–21. Numbers below are raw unless marked corrected.

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

1. **Path split of `I_CoC` at CoC positions** (`run_coc_path_split.py`, calib_100, same rollout
   seeds and typed gates as the anatomy): two CE backwards per clip, the second with the K and V
   of every generated-CoC position detached in every layer, so the CoC-position gate gradient
   carries the own-token path only; cross-position path = full − direct (linear, exact up to bf16
   rounding). `analyze_coc_path_split.py` gates: G0 the full backward's CoC row reproduces the
   anatomy's (per-layer Spearman ≥ 0.99); T1-a ceiling-corrected agreement of `I_traj`@CoC with
   the cross-position part ≥ 0.70 in both bands (like vision / prompt / ego-history positions);
   T1-b with the own-token part ≤ 0.45. Both pass → the exception is the loss-evaluation path and
   the paper can say the two scores agree wherever they read a position through the same
   interface. T1-a fails → the cross-position readers also differ (the expert and the later CoC
   queries want different things from the same K/V), and the census result (ego-history vs
   earlier-CoC readers) is the whole story.

   ```bash
   cd /home/cvlab21/project/chan/alpamayo-model-compression/.claude/worktrees/dual-saves-trunk
   for s in 0 1 2 3; do
     ALPAMAYO_REPO=$PWD nohup bash experiments/head_analysis/run_retry_host.sh 30 experiments/head_analysis/run_coc_path_split.py \
         --exp-id coc_pathsplit_v1_s$s --shard $s --n-shards 4 --gpu $((4 + s)) > outputs/coc_pathsplit_v1_s$s.launch.log 2>&1 &
   done
   .venv/bin/python experiments/head_analysis/analyze_coc_path_split.py --shards coc_pathsplit_v1_s0 coc_pathsplit_v1_s1 coc_pathsplit_v1_s2 coc_pathsplit_v1_s3 --out coc_pathsplit_v1
   ```
   Cost: rollout + two typed-gate CE backwards per clip ≈ 20 s → 25 clips per shard ≈ 10 min on
   Ada 4–7; ~400 MB of per-clip arrays per shard (MLP rows).
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

## Test 1 result (2026-09-25, `outputs/coc_pathsplit_v1/summary.txt`, 100 clips): the path is not the cause

- G0 PASS: the full CE backward reproduces the anatomy's CoC row exactly (per-layer Spearman
  1.000); the repeated forward gives the same NLL to 4 decimals on every clip.
- The own-token path does carry most of `I_CoC` at CoC positions: 87% of the |gradient| (Q heads;
  additive share 0.80), 0.70 in layers 0–21 and **0.93 in 22–34**; MLP 0.89 / 0.69 / 0.92.
- But the cross-position part (K/V read by later CoC queries) ranks the heads the same way as the
  own-token part — corrected agreement between the two parts 0.95 (0–21) / 0.91 (22–34) for Q,
  0.96 / 0.89 for MLP — and agrees with `I_traj`@CoC no better than the full score:

| Q heads, corrected ρ with `I_traj`@CoC | 0–21 | 22–34 |
|---|---:|---:|
| full `I_CoC`@CoC | 0.47 | 0.41 |
| own-token part | 0.45 | 0.39 |
| cross-position part | **0.50** | **0.49** |

  MLP channels: full 0.77 / 0.37, own-token 0.77 / 0.37, cross 0.78 / 0.39 (for MLP the
  CoC-position exception is late-band only). T1-a FAIL (0.50 / 0.49 < 0.70), T1-b moot.
- **Reading.** The hypothesis that the exception comes from the loss-evaluation (own-token) path
  is rejected: even the part of `I_CoC` that reaches the loss through the same kind of interface
  the expert uses (the position's K/V) disagrees with `I_traj`. The disagreement is between the
  two *readers* of the CoC-position representation — the later CoC queries and the expert's
  diffusion tokens — which is what test 2 found directly: the heads whose CoC-position writes
  matter to the text are the ones reading earlier CoC tokens, the heads whose writes matter to
  the expert are the ones reading ego-history. At vision, prompt and ego-history positions the
  two readers want the same content (the scene, the instruction, the ego state) and rank the
  writers alike; at the reasoning tokens the representation carries two separable contents —
  text continuation and action state — written by different heads. Figure:
  `outputs/coc_pathsplit_v1/plots/coc_path_split_q.png`.

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

## Test 3 result (2026-09-25, run A `outputs/coc_posanat_v1/summary.txt`, 99 calibration clips — one 124-token CoC exceeded the buffer): the exception is a token mixture inside the CoC

One typed gate per generated-CoC position (5 + n_coc types), one CE and one full-seed FM
backward per clip as in the anatomy, giving every Q head's signed contribution at every CoC
position under each loss. The sum over positions reproduces the anatomy's CoC row (per-layer
Spearman min 1.000 CE / 0.999 FM, A0 PASS). Every generated CoC ends with `<|cot_end|>`
`<|traj_future_start|>` (all 99 clips; length 8–47, median 16, so the two are 14% of the tokens).

- **Token weighting (A1).** Share of |Q-head contribution| at CoC positions, mean over layers
  and clips, 0–21 / 22–34 (token share in parentheses): `I_CoC` head clause 0.27 / 0.22 (0.23),
  rest of sentence 0.73 / 0.78 (0.63), `<|cot_end|>` 0.00 / 0.00 (0.07), `<|traj_future_start|>`
  0.00 / 0.00 (0.07); `I_traj` 0.12 / 0.11, 0.25 / 0.24, 0.32 / 0.21, 0.31 / 0.43. The CE mass at
  the boundary tokens is zero by construction (the last state predicts nothing; the one before it
  predicts a deterministic token). The FM mass concentrates there: 63% / 64% per layer, 71% / 75%
  pooled over a band's layers (median clip 0.76, 94–95% of clips above one half), 19× the
  per-token mass of a word, 15× its residual-gradient norm (layers 7–21). The plan's prediction
  was half right: FM concentrates on the end, but on the boundary tokens, not the head clause,
  which it weights less than CE does (0.12 vs 0.27).
- **Same sub-span agreement (A2; raw / corrected; 0–21 | 22–34).** All CoC 0.42 / 0.46 | 0.34 /
  0.39; head clause **0.91 / 0.98 | 0.85 / 0.95**; rest of sentence 0.74 / 0.89 | 0.84 / 0.92;
  `<|cot_end|>` 0.76 / 0.95 | 0.55 / 0.84 (CE self-reliability 0.72 / 0.56); **words only 0.78 /
  0.92 | 0.82 / 0.90**; dropping `<|traj_future_start|>` alone 0.50 / 0.55 | 0.63 / 0.68. Verdict
  by the pre-registered rule: token mixture (0.95 ≥ 0.70). Words-only agreement sits at the level
  of vision (0.72 / 0.79), prompt (0.75 / 0.81) and ego-history (0.86 / 0.94); the head clause is
  above all of them. MLP (three sub-spans only, `<|cot_end|>` inside "rest"): head clause 0.77 /
  0.96 | 0.57 / 0.78, rest 0.58 / 0.73 | 0.33 / 0.43.
- **Single positions (A3).** Spearman across heads at one clip, layer and position: 0.56 | 0.59,
  against 0.23 | 0.24 for the same clips after summing their positions; summed within one
  sub-span only: head clause 0.53 | 0.57, rest 0.44 | 0.47. Summing across the whole CoC halves
  the agreement even inside one clip.

**Reading.** `I_CoC` reads the reasoning at its words, `I_traj` where the expert reads it — the
cache entries of the two boundary tokens, the last two positions of the prefix. The shipped CoC
row sums a word-weighted and a boundary-weighted score; matched by token the two objectives rank
the writers alike, at the reasoning tokens as everywhere else. The "Interpretation" and the
"two separable contents at the same tokens" reading above are superseded. Figures:
`outputs/coc_posanat_v1/plots/same_subspan_agreement.png`,
`paper-analysis/figures/supplementary/fig9_coc_position_split.*`.

## Test 4 result (2026-09-25, run B `outputs/coc_posabl_v1/summary.txt`, 100 held-out clips = first 100 of `indist_500`, K = 8): double dissociation

Per layer the k = 8 heads with the largest rank(`I_traj`@CoC) − rank(`I_CoC`@CoC) (T), the k
smallest (C) and three random sets (R), switched off at the generated-CoC positions only, paired
with the unmasked model on the same text, noise and seeds. Dense NLL 0.1708, FM 0.2425, minADE
0.837. Mean [bootstrap 95% CI], median in parentheses.

| set | dNLL | dFM | dminADE |
|---|---|---|---|
| T, layers 0–21 | +0.0028 [+0.0002, +0.0060], +1.7% (+0.0001) | **+0.0285 [+0.0170, +0.0401], +11.7%** (+0.0187) | **+0.195 [+0.049, +0.349]** (+0.068) |
| C, layers 0–21 | **+0.0282 [+0.0184, +0.0392], +16.5%** (+0.0140) | −0.0004 [−0.0017, +0.0007] (0.0000) | −0.021 [−0.046, +0.000] |
| R ×3, layers 0–21 | +2.6 … +5.8% | −0.9 … +1.3% | +0.014 [−0.028, +0.064] |
| T, layers 0–34 | +0.4221 [+0.3490, +0.4972], **+247%** (+0.351) | +0.0284, +11.7% (+0.0183) | +0.210 [+0.052, +0.369] |
| C, layers 0–34 | +0.1003 [+0.0768, +0.1285], +58.7% (+0.049) | −0.0006, −0.3% | −0.018 [−0.041, +0.001] |
| R ×3, layers 0–34 | +10.4 … +16.7% | −1.3 … −0.1% | −0.014 [−0.101, +0.051] |

Gates (one-sided paired Wilcoxon): layers 0–21 **D1 PASS** (C > T p = 9.1e−10, C > every random
max p = 8.9e−07), **D2 PASS** (T > C p = 3.3e−08, T > every random max p = 5.1e−08), D3 T > C on
minADE p = 0.0013. All layers: D2 PASS (3.4e−08 / 1.8e−08), D3 p = 0.0017, **D1 FAIL** — not
because C fails (C > random p = 8.6e−13) but because T damages the text more than C. T's FM
damage by manoeuvre: decel_stop +37–39%, accel +12–14%, turn +9%, cruise +9%.

**Reading.** In the trunk a crossed double dissociation: the heads `I_traj` favours at CoC
positions carry the action there and nothing of the text, the heads `I_CoC` favours the text and
nothing of the action — each score is right about its own objective, and only its own, so the
union is what keeps both. In the late band the switch-off costs the action nothing (the expert
does not read those outputs) but the text +247% for the FM-favoured heads against +59% for the
CE-favoured ones: late heads `I_traj` ranks high and `I_CoC` ranks low at the reasoning tokens
are heads the text cannot do without, which a first-order score of small scalings does not flag
(a saturated contribution — large margin, small gradient — is where it under-predicts). This is
the position-restricted twin of the dual-saves-trunk result. Mechanism open (prompt-reading
heads, 0.154 vs 0.136 late in the census, or sink-like constant outputs whose removal moves the
RMSNorm scale). The CE-favoured sets never touch the action (dFM −0.2 / −0.3%, minADE −0.02
n.s.). Figures: `outputs/coc_posabl_v1/plots/dissociation.png`,
`paper-analysis/figures/supplementary/fig10_coc_position_dissociation.*`.

## Revised interpretation (supersedes the earlier ones)

The reasoning tokens are read twice and at different places: word by word by the text that
continues them, and as a summary at their boundary tokens by the action expert. `I_CoC` sees the
first reading, `I_traj` the second; summing over the CoC mixes them, which is the whole
exception. Matched by token the two scores agree everywhere. The readers are separately
necessary (trunk double dissociation), they draw on different keys (earlier CoC vs ego history;
vision alike), and in the late band the heads the FM score flags are also the ones the text
needs most. Paper sentences: `2026-09-25_coc_position_note.md` (판 3).
