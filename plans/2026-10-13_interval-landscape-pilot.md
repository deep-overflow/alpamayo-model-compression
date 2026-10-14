# Interval-deletion landscape, per objective (pilot)

Source of the method: "Does the VGGT Family Need All Its Layers?" (arXiv 2609.36842). That paper
deletes every contiguous interval `[i, j]` of layers and reads the damage as an `(i, j)` map, then
tests whether the damage of two separated intervals adds up. This pilot tries both analyses on the
Alpamayo 1.5 VLM, with one change: the map is drawn **per objective** (trajectory vs CoC), because
our causal round (2026-09-21) says layers 22-34 are a language channel.

## What is measured

- "Delete layer l" = kv-only bypass: all 32 Q heads and all 12,288 MLP channels masked, k/v
  projections left running so the expert still gets a cache entry (same definition as
  `run_depth_ablation.py` and `slim_lib.kvonly_layer_forward`).
- Runner: `run_token_ablation.py --arm-masks`, unchanged. Per clip the dense model rolls out one
  CoC; every mask is teacher-forced on it. Readouts, paired with dense on the same clip:
  `dNLL` (CoC NLL of the dense text), `dFM` (GT-anchored 10-step FM loss), `dminADE` (K=8).
- Clips: first 44 of `indist_500` (= val_500; held out from every calibration set), 4 shards.
- GPUs 0-3 (Blackwell), each shard under 2 h. All configs on one architecture; absolute values are
  not comparable to Ada-reported arms and are not reported as such.

## Configs (108 + dense)

| group | n | what |
|---|---:|---|
| single | 36 | `[l, l]` for every layer -- the diagonal, same protocol as the rest |
| grid | 56 | boundaries 2, 5, ..., 35, 36 (stride 3, layers 0-1 excluded), length <= 18 |
| pair | 16 | A in {`[8,10]`, `[8,13]`} x B in {`[17,19]`, `[20,22]`, `[23,25]`, `[26,28]`, `[29,31]`, `[32,34]`, `[23,28]`, `[29,34]`} |

Layers 0-1 are left out of the grid because their single-layer bypass is already known to be
fatal (stored 150-clip run: +1.05 m / +27.3 m median); they stay in the diagonal.

## Pre-registered gates

- **G0 (sanity)**. (a) Layer 35 bypass gives `dFM == 0` and `dminADE == 0` on every clip (its
  output never reaches the cache the expert reads). (b) Of the five layers the stored 150-clip
  single-layer run flags (0, 1, 6, 15, 16), at least four are among the pilot's top five by median
  `dminADE`. If G0 fails the masks or the merge are wrong and nothing else is read.
- **H1 (objective dependence)**. For the 12-layer late interval `[23, 34]`: median relative
  `dFM` below 5% **and** median `dNLL` above +0.10 nat. Either failing falsifies "late layers are
  a language channel" at whole-layer dose.
- **H2 (shape)**. Prediction: the trajectory map does **not** show the paper's dominant early
  region. Read as: the longest interval with median `dminADE < 0.05 m` that starts at or after
  layer 17 is longer than the longest one that ends at or before layer 16.
- **H3 (additivity)**. `D(B|A) = D(A u B) - D(A)` against `D(B)` over the 8 B's, per A and per
  metric: MAE, Pearson, Spearman. The paper's bar for "approximately additive" is rank
  correlation >= 0.97; prediction: the short A meets it on `dNLL`, the long A does not on
  `dminADE`.

## What this pilot cannot say

n = 44 with a heavy-tailed minADE: medians are the reading, means are reported alongside. The grid
is stride 3, so region boundaries are known to +-3 layers. NLL is teacher-forced on the dense text
and says nothing about free-generation collapse. No closed loop.

## Outcome

Run 2026-10-13, 44 clips (28 cruise / 6 turn / 6 decel_stop / 4 accel), 108 configs, Blackwell,
84 min per shard. `outputs/ivl_pilot_v1/` (metrics.json, summary.txt, plots/).

- **G0 PASS.** Layer 35: `dFM` and `dminADE` exactly 0 on all 44 clips. Top five single layers by
  median `dminADE` are 1, 0, 15, 16, 6 -- 5/5 flagged. Spearman against the stored 150-clip
  single-layer run +0.758.
- **H1 FAIL as registered.** `[23, 34]`: relative `dFM` +7.3% (bar 5%), `dNLL` +7.33 nat,
  `dminADE` +0.092. The interval was three layers too long: `[26, 34]` gives `dFM` -0.2%,
  `dminADE` -0.004 (mean -0.027, CI [-0.142, +0.089]) and `dNLL` +4.78. The trajectory-free band
  at whole-layer dose starts at 26, not 23.
- **H2 PASS.** Longest interval with median `dminADE < 0.05 m`: 3 layers ending at or before 16
  (`[8, 10]`), 10 layers starting at or after 17 (`[26, 35]`). The dominant region is late, the
  reverse of the paper's.
- **H3.** Short A `[8, 10]`: Spearman +0.93 / +0.91 / +0.93 (dADE / rFM / dNLL), Pearson >= 0.99 --
  below the paper's 0.97 rank bar, so the prediction for `dNLL` failed. Long A `[8, 13]` on
  `dminADE`: Spearman +0.19, as predicted. With 8 B's of which 6 are indistinguishable from zero,
  the rank correlation is mostly ordering noise; MAE against scale is the usable number.

Not registered, seen in the data:

- The two maps are near mirror images. `[8, 19]` costs +2.84 m and +0.14 nat; `[26, 34]` costs
  -0.004 m and +4.78 nat.
- Adjacent deletions are strongly superadditive on the trajectory: `[8, 16]` +1.41 m against
  +0.51 for its three 3-layer blocks and +0.27 for its nine single layers.
- "Late bypass repairs a damaged trunk" is **not** supported here: after A = `[8, 13]`, the six
  late B's move minADE by -0.027 pooled (Wilcoxon p = 0.83).
