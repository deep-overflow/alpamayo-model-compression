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

## Results A (2026-09-25, `outputs/coc_posanat_v1/summary.txt`, 99 clips — one skipped, CoC of 124 tokens > PAD; Ada 4–7, ~3 min per shard)

**The exception is a token mixture inside the CoC.** Every generated CoC ends with the two special
tokens `<|cot_end|>` and `<|traj_future_start|>` (all 99 clips; CoC length 8–47, median 16, so the
two are 14% of the tokens). The two losses weight the CoC positions in a complementary way, and
at the same tokens the two scores rank the heads alike.

- A0 PASS: the sum over positions reproduces the anatomy's CoC row (per-layer Spearman min 1.000
  CE, 0.999 FM).
- A1 token weighting (share of |Q-head contribution| at CoC positions, mean over layers and
  clips; 0–21 / 22–34; token share in parentheses):

  | score | head clause (0.23) | rest of sentence (0.63) | `<\|cot_end\|>` (0.07) | `<\|traj_future_start\|>` (0.07) |
  |---|---:|---:|---:|---:|
  | `I_CoC` | 0.27 / 0.22 | 0.73 / 0.78 | 0.00 / 0.00 | 0.00 / 0.00 |
  | `I_traj` | 0.12 / 0.11 | 0.25 / 0.24 | 0.32 / 0.21 | 0.31 / 0.43 |

  The CE mass at the two boundary tokens is zero by construction: the state at
  `<|traj_future_start|>` predicts nothing (the loss ends there) and the state at `<|cot_end|>`
  predicts `<|traj_future_start|>`, which is deterministic. The FM mass concentrates exactly
  there: 63% / 64% per layer, 71% / 75% pooled over a band's layers (median clip 0.76, 94–95% of
  clips above one half); per token a boundary token carries 19× the mass of a word (median) and
  15× the residual-gradient norm (layers 7–21, IQR 9.8–19.1). Residual-gradient shares tell the
  same story (CE head 0.32 / 0.26, rest 0.68 / 0.74, boundary 0.00; FM last two positions 0.32 +
  0.35 of the CoC total in layers 7–21, every earlier position ≤ 0.03). The plan's prediction
  was half right: FM does concentrate on the end — on the boundary tokens, not on the head clause,
  which it weights *less* than CE does (0.12 vs 0.27).
- A2 same sub-span agreement (Q heads; raw / corrected; 0–21 | 22–34):

  | restriction | 0–21 | 22–34 |
  |---|---:|---:|
  | all CoC positions (the shipped row) | 0.42 / 0.46 | 0.34 / 0.39 |
  | head clause | **0.91 / 0.98** | **0.85 / 0.95** |
  | rest of sentence | 0.74 / 0.89 | 0.84 / 0.92 |
  | `<\|cot_end\|>` | 0.76 / 0.95 | 0.55 / 0.84 (CE self-reliability 0.72 / 0.56) |
  | `<\|traj_future_start\|>` | undefined (CE = 0) | undefined |
  | **words only** (head + rest) | **0.78 / 0.92** | **0.82 / 0.90** |
  | drop `<\|traj_future_start\|>` only | 0.50 / 0.55 | 0.63 / 0.68 |

  Verdict by the pre-registered rule: token mixture (best sub-span 0.95 ≥ 0.70). Words-only
  agreement (0.82 raw / 0.90 corrected, late) sits at the level of vision (0.72 / 0.79), prompt
  (0.75 / 0.81) and ego-history (0.86 / 0.94) positions; the meta-action head clause is above
  all of them. Dropping only `<|traj_future_start|>` is not enough (0.63 / 0.68): both boundary
  tokens have to go, since `<|cot_end|>` alone holds 21–32% of the FM mass against ≈ 0 of CE.
  MLP channels (three sub-spans only; `<|cot_end|>` sits inside "rest" there): head clause
  0.77 / 0.96 | 0.57 / 0.78, rest 0.58 / 0.73 | 0.33 / 0.43 — the late MLP exception follows the
  same pattern but its rest-of-sentence value is contaminated by `<|cot_end|>`, which the runner
  cannot separate for MLP (it stored MLP sums per sub-span; a rerun with four spans would settle it).
- A3 single positions (one clip, one layer, one position; mean Spearman across heads): 0.56 |
  0.59, against 0.23 | 0.24 for the same clips after summing their CoC positions; summed within
  one sub-span only: head clause 0.53 | 0.57, rest 0.44 | 0.47, `<|cot_end|>` 0.58 | 0.50. So
  even inside one clip, summing across the whole CoC halves the agreement while summing within
  a sub-span keeps it near the single-position level (the single-clip noise ceiling is unknown,
  so the absolute values are lower bounds; the comparison across aggregation levels is the point).

**Reading.** `I_CoC` reads the reasoning at its words (the next-token loss lives there);
`I_traj` reads it where the expert reads it — through the cache entries of the two boundary
tokens that close the reasoning and open the action segment, the last two positions of the
prefix and the ones the diffusion tokens sit next to. The shipped CoC row sums a word-weighted
score and a boundary-weighted score, and that is the whole exception: matched by token the two
objectives rank the writers alike, at the reasoning tokens as everywhere else. The census
(test 2 of the previous plan) then describes the reader side of the same fact — the heads whose
boundary-token writes the expert consumes draw on the ego history, the heads whose word writes
the next token consumes draw on the earlier reasoning. The earlier main-text sentence "even at
the same tokens, the two scores favor different units" is **wrong** and must be replaced.
Figures: `outputs/coc_posanat_v1/plots/same_subspan_agreement.png`; paper-style
`paper-analysis/figures/supplementary/fig9_coc_position_split.{png,pdf,svg}`
(`paper-analysis/scripts/paper_coc_position_split.py`).

## Results B (2026-09-25, `outputs/coc_posabl_v1/summary.txt`, 100 held-out clips = first 100 of `indist_500`, 11 configs, Ada 4–7, 36 s per clip)

Paired change against the unmasked model (same rollout text, noise and seeds); dense NLL
0.1708, FM 0.2425, minADE 0.837. Mean [clip-bootstrap 95% CI], median in parentheses.

| set (k = 8 heads per layer, output switched off at CoC positions only) | dNLL | dFM | dminADE (K = 8) |
|---|---|---|---|
| `T_trunk` — `I_traj`-favoured, layers 0–21 | +0.0028 [+0.0002, +0.0060], +1.7% (+0.0001) | **+0.0285 [+0.0170, +0.0401], +11.7%** (+0.0187) | **+0.195 [+0.049, +0.349]**, +23% (+0.068) |
| `C_trunk` — `I_CoC`-favoured, layers 0–21 | **+0.0282 [+0.0184, +0.0392], +16.5%** (+0.0140) | −0.0004 [−0.0017, +0.0007], −0.2% (0.0000) | −0.021 [−0.046, +0.000] (−0.001) |
| `R0–2_trunk` — random | +0.0045 … +0.0100, +2.6 … +5.8% | −0.0021 … +0.0032, −0.9 … +1.3% | +0.014 [−0.028, +0.064] (`R0`) |
| `T_all` — layers 0–34 | +0.4221 [+0.3490, +0.4972], **+247%** (+0.351) | +0.0284 [+0.0170, +0.0400], +11.7% (+0.0183) | +0.210 [+0.052, +0.369] (+0.069) |
| `C_all` | +0.1003 [+0.0768, +0.1285], +58.7% (+0.049) | −0.0006 [−0.0020, +0.0006], −0.3% | −0.018 [−0.041, +0.001] |
| `R0–2_all` | +0.0177 … +0.0286, +10.4 … +16.7% | −0.0032 … −0.0003 | −0.014 [−0.101, +0.051] (`R0`) |

Gates (paired one-sided Wilcoxon): **layers 0–21: D1 PASS** (C > T p = 9.1e−10, C > every
random, max p = 8.9e−07), **D2 PASS** (T > C p = 3.3e−08, T > every random, max p = 5.1e−08),
D3 T > C on minADE p = 0.0013 (C > T p = 1). **All layers: D2 PASS** (3.4e−08 / 1.8e−08), D3
p = 0.0017, **D1 FAIL** — not because C fails (C > random p = 8.6e−13) but because `T_all`
damages the text more than `C_all`. T's FM damage by manoeuvre: decel_stop +37–39%, accel
+12–14%, turn +9%, cruise +9%.

**Reading.**
1. In the trunk it is a crossed double dissociation. The heads `I_traj` favours at CoC positions
   carry the action there and nothing of the text (FM +11.7%, minADE +0.20 m — a quarter of the
   dense error, most of it in decel_stop — against NLL +1.7%, the random level); the heads
   `I_CoC` favours carry the text and nothing of the action (NLL +16.5%, 3–6× random; FM −0.2%,
   minADE −0.02 n.s.). Each score is right about its own objective at these positions, and only
   about its own. With A, the picture is: the expert reads the reasoning as a summary at the two
   boundary tokens, written there by heads that draw on the ego history (census); the text reads
   it word by word through heads that draw on the earlier reasoning. The union is what keeps both.
2. In the late band the same switch-off changes the action by nothing (`T_all` dFM = `T_trunk`
   dFM, +11.7% both: the expert does not read those outputs, as the FM-insensitivity of layers
   22–34 already said) but takes the text from +1.7% to **+247%** — four times what the
   `I_CoC`-favoured heads do (+59%) and 15–24× random. So late heads that `I_traj` ranks high at
   CoC positions (and `I_CoC` ranks low — the sets are by rank *difference*) are heads the text
   cannot do without. A first-order score describes a small scaling; switching a head off at
   every reasoning word is not small, and a contribution that is saturated (large logit margin,
   small gradient) is where the score under-predicts. This is the position-restricted twin of the
   dual-saves-trunk result (the units dual keeps through `I_traj` were language units), and one
   more reason the union has to keep them. The mechanism is open — the census makes the late
   `I_traj`-favoured heads slightly heavier prompt readers at CoC queries (0.154 vs 0.136), and a
   sink-like constant output whose removal changes the RMSNorm scale is the other candidate; a
   per-layer or per-position switch-off would tell.
3. The `I_CoC`-favoured sets never touch the action (dFM −0.2 / −0.3%, minADE −0.02, n.s.): the
   text-continuation content at the reasoning tokens is not what the expert reads.

Figures: `outputs/coc_posabl_v1/plots/dissociation.png`; paper-style
`paper-analysis/figures/supplementary/fig10_coc_position_dissociation.{png,pdf,svg}`
(`paper-analysis/scripts/paper_coc_position_dissociation.py`).

## What changes in the paper

- The mechanism sentence. Not "even at the same tokens, the two scores favor different units"
  but: the two objectives read different tokens of the reasoning — the CE loss its words, the
  expert its two boundary tokens — and at the same token they rank the writers alike (words only
  0.78–0.82 raw / 0.90–0.92 corrected; head clause 0.85–0.91 / 0.95–0.98).
- The census stays as the identity of the readers (ego-history vs earlier-CoC), now with the
  causal half: switching the `I_traj`-favoured heads off at CoC positions costs the action and
  not the text, the `I_CoC`-favoured heads the text and not the action (layers 0–21).
- A late-band sentence: there the switch-off costs the action nothing and the text a lot, and
  the heads `I_traj` flags are the ones the text needs most (+247% vs +59%).
- Draft replacements are in `paper-analysis/2026-09-25_coc_position_note.md` (판 3).

## Results, per-query census (2026-09-26, `run_coc_census.py --per-query` → `outputs/coc_census_pos_v1/`, 99 calib clips, Ada 4–7)

Which CoC queries read the ego history. Mean attention mass of one CoC query on the ego-history
keys (0–21 | 22–34; all heads / FM-favoured / CE-favoured, the census groups of 6 per layer):
words 0.027–0.033 / 0.029–0.040 / 0.018–0.019 | 0.014–0.017 / 0.016–0.024 / 0.010–0.014;
`<|cot_end|>` 0.042 / **0.090** / 0.018 | 0.025 / 0.046 / 0.011; `<|traj_future_start|>` **0.079 /
0.193 / 0.027** | 0.065 / 0.155 / 0.014. The two boundary tokens (14.6% of tokens) carry 27% | 34%
of the CoC queries' ego-history attention for all heads and **43% | 47%** for the FM-favoured
heads; FM-favoured − CE-favoured at the special tokens +0.119 [+0.116, +0.122] | +0.088
[+0.084, +0.092], at words +0.014 | +0.008. No word category is enriched (ratio 0.8–1.0); the
words with the most ego-history reading are the speed/steering verbs (down, turn, slow, adapt,
nudge, yield, speed: 0.04–0.05 for the FM-favoured heads, 0–21). The CE-favoured heads read no
ego history anywhere and attend the sink at the boundary tokens (0.41–0.42). So the ego-history
reading sits where the FM gradient (A) and the expert's read sit — at the boundary tokens — and
through the heads whose CoC-position outputs the action needs (B). Analyzer
`analyze_coc_query_census.py`; token-category importance in `analyze_coc_token_importance.py`
(`outputs/coc_tokimp_v1/`); notes in `paper-analysis/2026-09-25_coc_token_importance.md`.

## D. Which text tokens carry I_FM (`run_text_position_anatomy.py` → `analyze_text_position_anatomy.py`, calib_100; prepared 2026-09-26, launch on the user's go)

Known from the anatomy (additive split of the shipped gate gradient, Q heads): I_FM's mass is
vision 0.62 / hist 0.10 / prompt text 0.17 / CoC 0.11 in layers 0–21 and vision 0.31 / hist 0.10 /
prompt text 0.21 / CoC 0.38 in 22–34 (I_CE: 0.50 / 0.04 / 0.24 / 0.22 and 0.05 / 0.00 / 0.42 /
0.53). The CoC part is resolved to the token (A: boundary tokens). The prompt text (157 tokens:
chat markers, the system sentence, 32 vision markers, camera text, the two history markers, the
instruction sentence, `<|cot_start|>`) and the 48 ego-history tokens are not. D gives every
non-vision position its own typed gate (vision stays one type; MLP not kept per position).

Readings, with the prediction stated before the run:
- D1 role weighting: share of I_FM by prompt role vs token share. Prediction from A: the
  boundary/marker tokens (`<|cot_start|>`, `<|traj_history_end|>`, the `<|vision_end|>`s) carry a
  disproportionate share; the instruction and system sentences carry little. I_CE alongside.
- D2 ego-history recency: I_FM share over the 48 history tokens by index; prediction: the last
  tokens (most recent motion) carry more than the first (last 8 / first 8 > 1.5).
- D3 same-role agreement: I_FM vs I_CE within each prompt role (raw / corrected). Prediction:
  ≥ 0.70 corrected on every role with ≥ 5 tokens (no second CoC-like exception hidden inside the
  pooled prompt-text 0.75 / 0.81); if a role falls below, name it.

```bash
cd /home/cvlab21/project/chan/alpamayo-model-compression/.claude/worktrees/dual-saves-trunk
for s in 0 1 2 3; do
  ALPAMAYO_REPO=$PWD nohup bash experiments/head_analysis/run_retry_host.sh 60 experiments/head_analysis/run_text_position_anatomy.py \
      --exp-id textpos_v1_s$s --shard $s --n-shards 4 --gpu $((4 + s)) > outputs/textpos_v1_s$s.launch.log 2>&1 &
done
.venv/bin/python experiments/head_analysis/analyze_text_position_anatomy.py --shards textpos_v1_s0 textpos_v1_s1 textpos_v1_s2 textpos_v1_s3 --out textpos_v1
```
Cost as A: ~3 min per shard on Ada 4–7 (44 GB reserved), ~130 MB of per-clip arrays per loss.

### Results D (2026-09-26, `outputs/textpos_v1/summary.txt`, 99 clips; shard 2 on Ada 6, the other three queued on the same card)

Non-vision normalisation, Q heads, 0–21 | 22–34 (enrichment = share / token share):
- **D1 — half right.** I_FM's largest text-side sink is the **48 ego-history tokens**: 36.6% (1.69×) |
  20.2% (0.93×); then the `<|vision_end|>`/`<|vision_start|>` markers 19.7% (1.36×) | 3.1%, the two CoC
  boundary tokens 11.1% (12×) | 28.4% (22–41×), `<|traj_history_end|>`+start 1.9% (2.2×) | 2.4% (2.7×),
  `<|cot_start|>` 1.7% (3.9×) | 4.1% (9.1×). Natural language is under its token share for I_FM:
  camera labels 0.38× | 0.36×, instruction 0.76× | 1.28×, system sentence 0.14× | 0.27×, CoC words
  0.97× | 1.61×. I_CE is the mirror: CoC words 28.8% (4.8×) | 49.6% (8.3×), `<|cot_start|>` 5.7% (12.5×)
  | 15.4% (34×), turn markers 8.6% (1.5×) | 13.1% (2.2×), ego history 0.76× | 0.11×, vision markers
  0.83× | 0.05×. FM − CE: history +0.202 [+0.177, +0.226], vision markers +0.077, CoC words −0.230,
  `<|cot_start|>` −0.039 (0–21). Top single prompt positions for FM: `-th` of "chain-of-thought"
  (0.038 | 0.065; also large for CE — a sink-like token, as is a space token inside the camera
  labels at position 72), the last image's `<|vision_end|>` (0.033), the newest history tokens
  (0.028 …), `<|cot_start|>` (0.018), `<|traj_history_end|>` (0.016); for CE the assistant-turn
  `<|im_start|>` (0.070 | 0.126) and `<|cot_start|>` (0.057 | 0.154).
- **D2 — PASS.** The history block is 16 timesteps × 3 slots. I_FM's share rises monotonically from
  step 4 to the newest step (0.036 → 0.130 | 0.033 → 0.118), the oldest step alone is second (first-
  token effect); last 8 / first 8 tokens = **1.78 | 1.74** (> 1.5), Spearman(index, share) +0.39 | +0.50.
  I_CE is U-shaped and flat (1.02 | 0.99). Slot 1 of each step carries 42–57%.
- **D3 — PASS.** Same-role agreement, split-half corrected: every role ≥ 0.93 in 0–21 (raw 0.83–0.98;
  ego history 1.00, vision markers 0.99, CoC words 0.95, `<|cot_start|>` 1.00) and ≥ 0.86 in 22–34
  except the single sink token (0.74). No second CoC-like exception hides inside the prompt.
- Residual-gradient shares (unit-free): FM 0–21 history 0.282, camera labels 0.226, vision markers
  0.160; CE camera labels 0.286, turn markers 0.229, CoC words 0.189.
Reading: outside vision, I_FM reads numbers and block boundaries (ego history with recency, the
markers that close each image, the tokens that close the reasoning); I_CE reads the language at the
positions where the next token is uncertain. At any token the two rank the heads alike. Note
`paper-analysis/2026-09-26_text_token_importance.md`; figures `fig13_text_tokens.*`,
`fig13_text_role_shares_analysis.png`, `fig13b_prompt_position_share_analysis.png`.

## Proposed C (not launched — on the user's go): make the boundary-token reading causal

A's 63–75% is a gradient share. Two cheap position-level knockouts would make it a cause:
1. **Reading side** — expert ← cache knockout (`run_pathway.py`, `build_configs`): add
   `X3b_coc_boundary` (the two boundary cache positions), `X3c_coc_words` (the CoC minus those
   two) and a control of two random CoC word positions. 50 val clips, K = 8, one Ada card,
   ~15 min. Prediction: `X3b` ≈ `X3_coc` on minADE, `X3c` ≈ control.
2. **Writing side** — `run_coc_position_ablation.py` with a position sub-mask: `T_trunk` off at
   the boundary tokens only vs at the words only. Prediction: the FM damage (+11.7%) sits in the
   boundary-only condition; the NLL damage of `C_trunk` in the words-only condition.
Both need a small code addition first (a config each), then the same launch pattern as B.
