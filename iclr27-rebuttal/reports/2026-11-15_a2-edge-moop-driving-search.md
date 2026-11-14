# Experiment: a setting where MoOP drives better than FM-only and CE-only on Alpamayo 2 Edge (explored, then confirmed on OOD-train)

2026-11-15. Plan with pre-registration: `iclr27-rebuttal/plans/2026-11-15_a2-edge-moop-driving-search.md`.
- Protocol and selection rule committed in bb220b3 (03:35 KST), before any search result.
- Picked setting and mask hashes committed in 93b5bfd (06:55 KST), before the confirmatory run.

Round 1 of the search (u40-q2/q4, `plans/2026-11-15_a2-edge-qcut-u40.md`) is reported here as well.

## Purpose

User goal: find an Edge setting where MoOP's driving is better than FM-only's and CE-only's. Time permitting, also
evaluate OOD-train.

Searching many settings for a MoOP win invites a chance win. So the protocol was:
1. Explore on Test500 + OOD-Val250.
2. Pick one setting with a rule fixed in advance.
3. Test that one setting on **OOD-train1178**, which is fresh data for Edge.

## Setup

**Common to every arm:**
- `nvidia/Alpamayo2-Edge@f8951d0`.
- calib_100 importance (`edge_importance_v1`), no-nav prompt.
- Masks by make_slim's dual / traj / coc rules (`make_edge_moop_masks.py`).
- Open-loop protocol: 1 CoC rollout + 8 denoisings; minADE@6 / minFDE@6.
- Hardware: Ada RTX 5880 cards 4–7, empty cards only, 4 CPU threads per job.
- Bootstrap: 200,000 draws, shared across arms; the seed-1 reruns agree.

**Explored settings.** Per VLM layer, in all 28 layers. The expert, KV heads, vision tower and embeddings are untouched.

| setting | Q heads cut (of 16) | MLP channels cut (of 9,216) | removed params | % of 2.65 B |
|---|---|---|---|---|
| u40 | 6 | 3,673 | 509.3 M | 19.2 % |
| u40-l0keep | 6 in layers 1–27, 0 in layer 0 | 3,673 + 27–28 | 509.3 M | 19.2 % |
| u40-q4 / u40-q2 | 4 / 2 | 3,929 / 4,185 | 509.3 M | 19.2 % |
| u50 | 8 | 4,608 | 645.9 M | 24.4 % |
| u50-q4 / u50-q2 | 4 / 2 | 5,120 / 5,376 | 645.9 M | 24.4 % |
| u60 | 10 | 5,530 | 781.0 M | 29.5 % |
| u60-q2 | 2 | 6,554 | 781.0 M | 29.5 % |
| u70-q2 | 2 | 7,603 | 901.3 M | 34.0 % |

- qN = make_slim's `_qcut`: the same per-layer parameter budget with N heads cut. One Edge Q head = 128 relu^2 channels.
- l0keep pays layer 0's heads back in MLP channels.
- Mask gates:
  - The parameterised l0keep and q-cut rules reproduce `make_a2_moop_masks` on A1.5 unit for unit.
  - Parameter parity holds within each budget.
  - The q-cut sets nest with the base allocation.

**Pre-registered selection rule.** Pooled Test500 + OOD-Val250 (750 clips), minADE@6 at 6.4 s.
- gap = max(MoOP − FM-only, MoOP − CE-only); the smallest gap wins.
- Ties within 0.005 go to lower sparsity, then fewer head cuts.

**Pre-registered confirmatory test.**
- OOD-train: 1,271 clips, of which the 1,178 with t0 >= 2 s are used, by the same rule as OOD-Val250.
- The Edge cache's GT equals `ood_train_a2` for 1,178/1,178 clips.
- Primary: minADE@6 at 6.4 s, MoOP − FM-only and MoOP − CE-only, Holm over the 2.
- SUCCESS = both negative with Holm p < 0.05.

## Results (Facts)

### Exploration: pooled Test500 + OOD-Val250 (750 clips), minADE@6 at 6.4 s in m

| setting | Dense | MoOP | FM-only | CE-only | MoOP − FM-only [95 % CI] | MoOP − CE-only [95 % CI] | gap |
|---|---|---|---|---|---|---|---|
| **u50** | 0.932 | **1.255** | 1.434 | 1.552 | **−0.179 [−0.270, −0.087]\*** | **−0.297 [−0.395, −0.199]\*** | **−0.179** |
| u70-q2 | 0.932 | 1.099 | 1.213 | 1.469 | −0.114 [−0.182, −0.048]\* | −0.370 [−0.463, −0.281]\* | −0.114 |
| u50-q2 | 0.932 | 0.971 | 1.012 | 0.983 | −0.042 [−0.096, +0.013] | −0.012 [−0.044, +0.021] | −0.012 |
| u60-q2 | 0.932 | 0.996 | 0.997 | 1.049 | −0.000 [−0.047, +0.049] | −0.053 [−0.094, −0.010]\* | −0.000 |
| u40-q2 | 0.932 | 0.940 | **0.927** | 0.983 | +0.013 [−0.031, +0.059] | −0.042 [−0.077, −0.011]\* | +0.013 |
| u50-q4 | 0.932 | 1.047 | 1.015 | 1.006 | +0.032 [−0.031, +0.096] | +0.041 [−0.005, +0.089] | +0.041 |
| u40-q4 | 0.932 | 0.997 | 0.944 | 1.010 | +0.054 [+0.006, +0.106]\* | −0.012 [−0.056, +0.033] | +0.054 |
| u40 | 0.932 | 1.121 | 1.056 | 1.055 | +0.065 [−0.004, +0.135] | +0.066 [+0.006, +0.129]\* | +0.066 |
| u60 | 0.932 | 2.162 | 1.554 | 3.135 | +0.608 [+0.472, +0.746]\* | −0.973 [−1.180, −0.774]\* | +0.608 |

u40-l0keep, Test500 only (not eligible): MoOP − FM-only +0.072, MoOP − CE-only +0.015.

**Pick by the rule: u50.**

Per set, MoOP − FM-only / MoOP − CE-only:

| setting | Test500 | OOD-Val250 |
|---|---|---|
| u50 | −0.245\* / −0.293\* | −0.048 (n.s.) / −0.304\* |
| u70-q2 | −0.154\* / −0.412\* | −0.034 / −0.286\* |
| u50-q2 | +0.002 / +0.016 | −0.129\* / −0.067\* |
| u40-q2 | +0.013 / −0.040\* | +0.013 / −0.047 |

### CoC health in exploration (Test500)

| setting | MoOP degenerate / length | FM-only degenerate / length (no-anchor) | CE-only degenerate / length (no-anchor) |
|---|---|---|---|
| u40 | 0.0 % / 23 | 3.6 % / 34 | 0.0 % / 25 |
| u40-q2 | 0.0 % / 23 | 0.0 % / 17 (`<NOMINAL>` 38 %) | 0.0 % / 24 |
| **u50** | **0.0 % / 26** | **24.6 % / 55 (4)** | 0.0 % / 24 |
| u50-q2 | 0.0 % / 24 | 0.0 % / 20 | 0.2 % / 24 |
| u60 | 21.4 % / 62 (12) | 6.4 % / 35 | 42.4 % / 86 (85) |
| u70-q2 | 38.8 % / 42 | 1.8 % / 17 | 26.2 % / 81 (111) |

OOD-Val250 shows the same pattern; e.g. u50 FM-only 26.8 %, MoOP 0.4 %.

### LingoQA, round 1 (500 val questions, segment CIs)

| setting | Dense | MoOP | FM-only | CE-only |
|---|---|---|---|---|
| u40 | 73.6 | 71.8 | 63.0 | 72.4 |
| u40-q2 | — | 69.0 | **77.4** | 69.0 |
| u40-q4 | — | 69.0 | 70.2 | 70.0 |

- u40-q2: MoOP − FM-only −8.4 pp\*, MoOP − CE-only 0.0.
- u40-q4: MoOP − FM-only −1.2, MoOP − CE-only −1.0 (n.s.).
- LingoQA was not run at u50 or above.

### Confirmation on OOD-train1178 (pre-registered, setting u50)

| arm | ADE 1.6 s | ADE 3.2 s | ADE 6.4 s (median) | FDE 6.4 s | CoC degenerate | CoC length |
|---|---|---|---|---|---|---|
| Dense | 0.063 | 0.255 | **0.975** (0.555) | 2.480 | 0.0 % (`<NOMINAL>` 30 %) | 17.3 |
| MoOP-u50 | 0.092 | 0.357 | **1.322** (0.760) | 3.307 | 0.1 % | 27.2 |
| FM-only-u50 | 0.105 | 0.395 | **1.457** (0.952) | 3.601 | **27.3 %** | 55.3 |
| CE-only-u50 | 0.116 | 0.425 | **1.456** (0.947) | 3.457 | 0.0 % | 24.9 |

**Primary (minADE@6 at 6.4 s, paired, Holm over 2):**

| contrast | Δ [95 % CI] | p | p_holm | p_holm, seed 1 |
|---|---|---|---|---|
| MoOP − FM-only | **−0.136 [−0.213, −0.056]** | 0.00097 | **0.00097** | 0.00107 |
| MoOP − CE-only | **−0.135 [−0.207, −0.062]** | 0.00037 | **0.00074** | 0.0007 |

**VERDICT: SUCCESS.** MoOP drives significantly better than both halves.

Secondary (mean [95 % CI]):

| contrast | ADE 1.6 s | ADE 3.2 s | FDE 6.4 s |
|---|---|---|---|
| MoOP − FM-only | −0.013\* | −0.038\* | −0.294 [−0.523, −0.057]\* |
| MoOP − CE-only | −0.024\* | −0.068\* | −0.150 [−0.360, +0.063] (n.s.) |
| FM-only − CE-only | −0.011\* | −0.030\* | — |

FM-only − CE-only at ADE 6.4 s is +0.001 (n.s.).

Each arm vs dense at ADE 6.4 s:

| arm | Δ vs dense | relative |
|---|---|---|
| MoOP | +0.347\* | +36 % |
| FM-only | +0.483\* | +50 % |
| CE-only | +0.481\* | +49 % |

minADE@6 at 6.4 s by bucket:

| bucket (n) | Dense | MoOP | FM-only | CE-only |
|---|---|---|---|---|
| cruise (639) | 0.853 | **1.158** | 1.296 | 1.369 |
| decel_stop (246) | 1.090 | **1.286** | 1.513 | 1.303 |
| accel (207) | 0.941 | 1.337 | 1.448 | **1.273** |
| turn (86) | 1.633 | 2.605 | **2.522** | 2.988 |

## Analysis (Opinion)

- **The goal is met, at u50.**
  - At u50 (8 of 16 Q heads and half of the MLP channels cut per layer, 24.4 % of Edge), MoOP drives better than
    FM-only and CE-only.
  - This holds in exploration (pooled −0.18 / −0.30 m). It also holds in the pre-registered test on fresh OOD-train
    data (−0.136 / −0.135 m, both Holm p < 0.001).
  - It is the paper's A1.5 claim (the union beats both halves) reproduced on a non-Qwen3-VL model, now in driving and
    not only in language.
- **Mechanism: the two halves fail on opposite axes, as on A1.5.**
  - At u50, FM-only's CoC breaks: 25–27 % degenerate CoCs, more than twice as long, some never closing.
  - CE-only keeps a clean CoC but drives worst.
  - MoOP keeps both a clean CoC (≤ 0.4 %) and the better driving.
  - At u40, FM-only's CoC was still healthy (3.6 %), so there was nothing for the CE half to protect. MoOP then only paid
    for the heads the CE criterion asked to keep. That is why u40 showed no MoOP win.
- **The regime is narrow, and that belongs in the claim.**
  - At u60 plain, MoOP's own CoC breaks (21–23 % degenerate) and MoOP drives far worse than FM-only (+0.61 m).
  - So MoOP's driving win is not a monotone "more pruning → bigger MoOP edge". It is a window where FM-only has broken
    and MoOP has not.
  - u70-q2 shows a second winning configuration (−0.11 / −0.37 m in exploration). MoOP's CoC is degenerate there too
    (39 %), so the win at that setting does not come from CoC health.
- **Cutting fewer heads (q-cut) helps every arm more than it separates them.**
  - At u40, q2 brings FM-only to dense level (0.927 vs 0.932) and MoOP close behind (0.940). No setting with 2 head cuts
    reproduces the u50 win: −0.042 at u50-q2 (n.s.), ~0 at u60-q2.
  - So the head budget is the main lever for absolute driving quality on Edge.
  - The criterion matters most when enough heads are cut for one half to break.
  - For deployment, u40-q2 / u50-q2 are better absolute choices than u50 (+36 % on OOD-train), even though MoOP does not
    beat FM-only there. On the pooled exploration clips, MoOP is 0.9 % / 4.2 % worse than dense.
  - Language at q2 went the other way at u40: FM-only 77.4 % vs MoOP 69.0 %.
- **Caveats.**
  - Exploration covered 10 settings (9 eligible). The confirmation on fresh data addresses selection, not the
    narrowness of the window.
  - OOD-train1178 drops the 93 clips with t0 < 2 s (Edge's input profile).
  - No-nav prompt.
  - Edge only.
  - Every arm at u50 is far from dense (+36–50 %).
  - The FDE gap vs CE-only is not significant.
  - Bucket view: MoOP's gain comes from cruise and decel_stop, while CE-only is better in accel and FM-only in turn.
- **Suggested next steps.**
  - LingoQA at u50 for the three arms: the A1.5 pattern predicts FM-only collapses there too.
  - A u45 / u55 bracket to map the window's edges.
  - If a recipe must be shown that both wins for MoOP and stays close to dense, test intermediate head budgets at u50
    (q6).
