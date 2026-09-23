# Plan: CoC NLL for seven arms — reference NLL on OOD-val, dense-reference NLL on test500

Status 2026-09-23 17:26: launched on Ada cards 4–7 on the user's go (the box had been
held by other members until then). Both tracks below are running from the commands as
written; 11 launcher processes, one running job per card, the rest waiting on memory.

## Purpose

Give the paper's arm comparison a language column that does not depend on what the arm
itself generated. Two columns, because the two available reference texts answer
different questions:

| column | set | reference text | question |
|---|---|---|---|
| `nll_gtcoc` | OOD-val 262 | curated human reference | does the arm still predict *correct* reasoning on long-tail cases |
| `nll_dense` | test500 | the unpruned model's own rollout on that clip | does the arm preserve the dense model's reasoning distribution in-distribution |

The second is proposed because `I_CoC` was fitted on the model's own rollout NLL over
`calib_100` (official train), so "preserve your own in-distribution CoC distribution" is
the test of what that criterion was designed to do. The first is kept because it is the
only correctness-anchored text, and because a metric must not be swapped out on account
of a surprising result (see "What we already know").

`nll_self` (NLL of the arm's own sample) is **not** used. On test500 it tracks
degeneracy: wanda's 11.9 is 88% token soup of ~226 tokens, and its 62 non-degenerate
clips score 0.97. The runner's docstring calls it "a reference number, not a quality
measure".

## Definitions

Both are `cross_entropy(logits, tokens)` with mean reduction over the CoC token span:
mean per-token NLL in nats of a text teacher-forced as context. One forward pass, no
sampling, so seed-independent. `run_eval.eval_config_samples` computes exactly this for
`nll_gtcoc`; `nll_dense` uses the same span rule with a different text.

## Arms

| arm | checkpoint | OOD-val `nll_gtcoc` rows | test500 `nll_dense` |
|---|---|---|---|
| baseline | `nvidia/Alpamayo-1.5-10B` | `baseline_pred_oodval` (stored) | to run; must land near its `nll_self` 0.175 |
| wanda | `outputs/slim_wanda_u40_v2` | **to run** (`wanda_u40_v2_oodval` was `--no-tf`) | to run |
| llm-pruner | `/mnt/nvme1n1/ad_vla/data/alpasim/drivers/lp_r50` | **to run** (no open-loop rows exist) | to run |
| tyr-the-pruner | `outputs/slim_tyr_u40_r` | `tyr_u40_r_pred_oodval` (stored) | to run |
| traj | `outputs/slim_traj_u40_v2` | `traj_u40_v2_ood`, `split == val` (stored) | to run |
| coc | `outputs/slim_coc_u40_v2` | `coc_u40_v2_ood`, `split == val` (stored) | to run |
| dual | `outputs/slim_dual_u40_v2` | `dual_u40_v2_pred_oodval` (stored) | to run |

`lp_r50` is soowon's LLM-Pruner build (block-wise grouped Taylor, `coc_param_first`,
ratio 0.5 over layers 4–33, 16 heads + 6144 MLP channels dropped per layer, 24.99% of
the model, same `model_revision`). Its `slim_meta.json` has the same `vlm`/`expert`
36-layer `{q, mlp}` structure as ours and `load_slim` reads only that plus the optional
`slim_state.pt`, both present. A variant `lp_r50_dual` (dual objective) sits beside it.

## What we already know (stored rows, Ada, n = 262)

| arm | mean | Δmean vs baseline [95% CI] | Wilcoxon p |
|---|---|---|---|
| baseline | 3.139 | — | — |
| dual | 3.185 | +0.046 [−0.014, +0.104] | 0.83 |
| tyr-the-pruner | 3.300 | +0.161 [+0.098, +0.226] | 2.5e−5 |
| traj | 3.348 | +0.209 [+0.131, +0.292] | 1.0e−5 |
| coc | 3.407 | +0.267 [+0.203, +0.332] | 1.0e−13 |

`coc` worse than `traj` on reference NLL is the surprise that motivated `nll_dense`. It
is not obviously a metric artefact: the `coc` arm is damaged elsewhere too (15% soup on
test500, worst single-criterion minADE). So `nll_dense` is added beside `nll_gtcoc`, not
in its place. If the `coc`/`traj` order flips between the two columns, that is itself the
finding. Note `nll_dense` structurally favours `coc` and `dual`, whose criterion
optimised that very quantity; it is a consistency check, not an independent one.

## Runs to launch later (Ada only, retry launcher, from the main checkout)

Every stored row is Ada (`RTX 5880`). The Blackwell `baseline_ood` differs from the Ada
run by up to 0.03 per clip on `nll_gtcoc`, so all new runs stay pinned to cards 4–7.
Flags match the stored rows: `k=8`, `seed 42`, `max_gen 256`.

### A. OOD-val `nll_gtcoc` for wanda and llm-pruner — existing runner, ~14 s/clip

```bash
cd /home/cvlab21/project/chan/alpamayo-model-compression
LP=/mnt/nvme1n1/ad_vla/data/alpasim/drivers/lp_r50
R=experiments/head_analysis/run_retry_host.sh; B=experiments/evaluation/run_baseline.py
for s in 0 1; do
  nohup bash $R 1440 $B --set ood --manifest ood_val --model $LP \
      --exp-id lp_r50_oodval --shard $s --n-shards 2 --gpu $((6+s)) --reserve-gb 26 \
      > logs/eval_lp_r50_oodval_s$s.log 2>&1 &
  nohup bash $R 1440 $B --set ood --manifest ood_val --model outputs/slim_wanda_u40_v2 \
      --exp-id wanda_u40_v2_tf_oodval --shard $s --n-shards 2 --gpu $((4+s)) --reserve-gb 26 \
      > logs/eval_wanda_u40_v2_tf_oodval_s$s.log 2>&1 &
done
```

About 31 min per shard once a card is free; four shards on four cards, ~1 h wall. These
were queued on 2026-09-23 14:28 and withdrawn before any started; the empty output
stubs were removed, so the commands above start clean.

### B. test500 `nll_dense` for all seven — new NLL-only runner, ~1–2 s/clip

Reusing `run_baseline.py` would re-run the rollout and 8 trajectory samples per clip
(~14 s/clip, ~12 GPU-hours for seven arms) to get one forward pass. A dedicated script
is the right tool: **`experiments/evaluation/run_nll_dense.py`** (written 2026-09-23,
not yet run on a GPU).

Design:
- Reference text: `gen_coc` from `outputs/baseline_ada_ps_test/*.json` (500 clips, Ada,
  the unpruned model's rollout at the stored seeds). Re-tokenised with
  `run_baseline.gt_coc_seq`, the same construction the OOD teacher-forced condition uses,
  so the span rule is identical.
- Per clip: `sample_cache.load_cached` → `analysis_lib.build_inputs` → one forward on
  `[prompt, reference CoC]` with `use_cache=False` → per-token `cross_entropy` over the
  CoC span, stored as its mean. No sampling.
- Same determinism block as `run_baseline.py`; `--model` accepts `baseline` or a slim dir;
  `--gpu`, `--shard`/`--n-shards`, `--limit`, resume by `clip_id`, `--ref-run` override.
- Rows: `clip_id, bucket, nll_dense, n_tok, ref_len, ref_nll_self, ref_empty,
  ref_degenerate, ref_coc`. Output convention `outputs/nll_dense_<arm>_<set>/` with
  `config.json` (model, revision, reference run, gpu) and `summary_s*.txt`.
- Gate before trusting it: **baseline on its own rollout must reproduce `nll_self`**
  (0.175 mean). The spans are identical by construction: the rollout stops on
  `<|traj_future_start|>` and `nll_self` scored `sequences[:, prompt_len:eos_pos + 1]`,
  which ends in the same `<|cot_end|>`, `<|traj_future_start|>` pair `gt_coc_seq`
  appends. The only residual is the decode→encode round trip of the text; the summary
  line prints the mean and max per-clip gap and how many clips changed span length.
- Checked offline without a GPU (2026-09-23): the tokenizer rebuilt as
  `base_model._build_tokenizer` does it (`nvidia/Cosmos-Reason2-8B` processor + 4,000
  trajectory tokens + special tokens) maps `<|cot_end|>` / `<|traj_future_start|>` to
  155678 / 155681 as the runner assumes, and re-tokenising every stored `gen_coc` gives
  exactly `gen_len` on **500/500** clips, with no leftover special markers in the text.
  The import chain also loads cleanly (`--help`).
- Cost: ~500 × 1.5 s ≈ 13 min per arm plus a 1–2 min model load; seven arms ≈ 1.7
  GPU-hours, one card.

Launch shape:

```bash
for m in baseline outputs/slim_wanda_u40_v2 $LP outputs/slim_tyr_u40_r \
         outputs/slim_traj_u40_v2 outputs/slim_coc_u40_v2 outputs/slim_dual_u40_v2; do
  bash $R 1440 experiments/evaluation/run_nll_dense.py --set test --model $m \
      --ref-run baseline_ada_ps_test --gpu 4 --reserve-gb 30
done
```

Sequential on one card is enough at ~15 min per arm; `--reserve-gb 30` covers the
22.2 GiB unpruned model, and the slim arms fit under it as well.

## Analysis

`experiments/evaluation/nll_table.py --metric {nll_gtcoc,nll_dense}` (CPU only) builds
either column with one format: per-arm mean / median / sd, paired Δ vs baseline with
bootstrap 95% CI of the mean (10,000 draws, seed 0) and Wilcoxon, share of clips worse,
per-bucket means, one plot, into `outputs/<metric>_table/`. Arms without rows are listed
as pending, so it is rebuilt as runs land. For `nll_dense` it also prints the round-trip
gate (baseline `nll_dense` minus stored `nll_self`, mean and max over clips). The two
columns are then reported side by side, with the `coc` vs `traj` order called out.

## Confounds held fixed

- Ada only; mixed architectures make the script print a warning.
- Same flags as every stored row; same `model_revision 7aba8293`.
- Mixing a 1,533-clip OOD run filtered to `val` with a 262-clip `ood_val` run is safe:
  `dual` from both sources is bit-identical on all 262 clips.
- Clips are the intersection across arms; n is reported.

## Pre-registered expectations (descriptive, no decision hangs on them)

- `nll_gtcoc`: wanda worst by a wide margin (rollouts 88% soup); `dual` indistinguishable
  from baseline (stored: p = 0.83).
- `nll_dense`: `dual` and `coc` closest to baseline, since it is their fitted objective;
  if `coc` is still behind `traj` here, the OOD result was not a reference-style artefact.
