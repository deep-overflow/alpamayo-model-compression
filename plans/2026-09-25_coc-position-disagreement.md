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
2. **Attention targets at CoC queries** (eager-attention census as `run_vision_census.py` but
   for CoC-token queries, ~15 min): per head, mass on vision / ego-history / prompt / earlier CoC /
   sink keys. Gate: FM-favoured heads put more mass on vision + ego-history keys than CE-favoured
   heads (grounding the reasoning for the expert), CE-favoured heads more on prompt + earlier CoC
   keys (language continuation). This is the "what do they do" half.
