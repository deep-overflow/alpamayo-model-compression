# LLM-Pruner (arXiv:2305.11627) on the Alpamayo 1.5 VLM text tower

Ported 2026-09-06 from **soowon's** `alpamayo1.5` research tree
(`/home/cvlab21/project/soowon/vla-ad/alpamayo1.5/experiments/llm_pruner/`), which is where
this port was written and where `lp_r50` — the external baseline in
`reports/evaluation/2026-09-03_difficulty-stratified-arms.html` and
`2026-09-05_hard100-closedloop.html` — came from. The code and its design decisions are
theirs; the reason it is here is stated below.

## Why it was copied here

The second-order arm was never run. soowon's own notes (`docs/llm_pruner.md`) say why, and
it is not a result:

> `param_second` / `param_mix`는 이번 실행에서 제외한다. 둘 다 elementwise Fisher
> `acc_grad = (1/N)Σg²`가 필요해서 host accumulator가 41GB → 82GB로 두 배가 되는데, 실제로
> 돌려보니 이 공유 서버에서 kcompactd가 100%로 붙고 동시에 돌던 평가 shard가 8s/clip →
> 90s/clip으로 무너졌으며 다른 사용자 job까지 느려졌다.

That 82 GB is for **two** objectives (`coc,traj`). LLM-Pruner's canonical objective is the
single language loss — here the CoC NLL — and `--objectives coc --second-order` needs
**41.2 GiB**, exactly the size soowon already ran without trouble. Accounting: layers 4–33
× (q,o = 2·4096² + gate,up,down = 3·12288·4096) = 5.536 B prunable params × 4 B × 2 buffers.

This matters to us because our own "second-order" arm (`dual2nd_u40_v2`) turned out not to
carry curvature at all — see below — so nobody has actually measured second-order Taylor
importance on this model.

## What is here, and what deliberately is not

| file | origin |
|---|---|
| `lp_core.py` | soowon's, formulas untouched (that fidelity is the point) |
| `lp_objectives.py` | soowon's |
| `run_lp_importance.py` | soowon's, `--first-order` added |
| `lp_prune.py` | soowon's, unchanged |
| `test_lp_formulas.py` | new here — cross-checks the formulas against upstream on CPU |

**Not copied:** soowon's vendored `analysis_lib.py`, `sample_cache.py`,
`expert_per_clip.py`, `slim_lib.py`. Those were vendored *from this repo* on 2026-08-09 and
say so in their headers; this repo's versions are newer (`expert_per_clip` has the
`AcceleratorError` fix that cost 44 wasted retries on 2026-09-01, `slim_lib` a pinned
`MODEL_REV` and `save_slim(write_state=)`, `sample_cache` `calib_samples()`), and the four
`slim_lib` entry points `lp_prune.py` uses have identical signatures. So the ported code
runs against this repo's libraries unchanged, and gets those fixes for free.

**Not copied:** `LLMPruner/` (the vendored upstream). Nothing here imports it — it appears
only in docstring citations. `trace_check.py`, which does use it, stayed behind.

Every change from the original is marked `PORT:` in the source.

## The one substantive change: `--first-order`

`param_mix` is the paper's Eq. 5, `S = W⊙g − ½·W⊙acc_grad⊙W`. Upstream computes its two
terms on the **same per-sample scale** (`_upstream_reference/hf_prune.py:128-148`): a
per-example loop accumulates `acc_grad = Σ_j g_j²/N`, then one batched backward leaves
`.grad` equal to the mean-loss gradient `≈ (1/N)Σ_j g_j`.

`GradAccumulator` here instead **sums** the signed per-clip gradients, so `g = N·g_batch`.
Feeding that to the upstream expression makes the first-order term N times too large.
`param_first` does not care — the pipeline is homogeneous of degree 1 in the salience, so a
global factor cannot reorder anything — but `param_mix` does: at N=100 the second-order
correction becomes numerically invisible and the arm silently collapses onto `param_first`.

`test_lp_formulas.py` measures that collapse. Rank correlation between `param_mix` and
`param_first` under `first_order="sum"`:

| N | Spearman |
|---:|---:|
| 1 | 0.881 |
| 10 | 0.902 |
| **100** | **0.979** |

So the default is `--first-order mean`, which is what upstream actually computes. Pass
`--first-order sum` to reproduce the original tree's stored arrays bit for bit.

## Verification that needs no GPU

```bash
.venv/bin/python experiments/llm_pruner/test_lp_formulas.py
```

It re-implements upstream's `TaylorImportance` branches independently (hand-transcribed
from `hf_llama_pruner.py:266-292`, without looking at `lp_core`) and compares on random
tensors: all four `--taylor` variants × both axes, and all six group reductions, agree to
**0.00e+00**. It also asserts `param_first`'s ranking is invariant to `--first-order`.

## Planned use

1. **Reproduction gate.** Re-run `coc_param_first` here and check it selects what soowon's
   `lp_r50` selected. Nothing downstream is believable until the port reproduces the arm it
   was ported from. ~20.6 GiB host RAM, no second-order buffer.
2. **The actual experiment.** `--objectives coc --taylor param_mix --second-order`, 41.2 GiB.
3. Compare against this repo's `dual2nd_u40_v2`, which is *not* the same thing: ours replaces
   the first-order score `E|g|` with `E[g²]`, and since `E[g²] = (E|g|)² + Var(|g|)` and
   `(E|g|)²` ranks identically to `E|g|`, the only thing that moved our selection was the
   across-clip **variance** (69–81% of the score, CV 1.6–2.6). It measured tail-weighting,
   not curvature, and cost +0.027 minADE (p=0.00063) on test500.

Run it when the cards are quiet, and keep `OMP_NUM_THREADS=8` — soowon measured other users'
jobs slowing 11× without it.
