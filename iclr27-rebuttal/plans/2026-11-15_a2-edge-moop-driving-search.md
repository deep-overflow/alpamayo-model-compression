# Plan: find an Edge setting where MoOP drives better than FM-only and CE-only (explore, then confirm on OOD-train)

2026-11-15, written at ~03:35 KST. User goal (`/goal`): "MoOP가 주행 능력이 FM only나 CE only보다 잘 나오는 세팅 찾아줘", plus "시간 남으면
ood train도 추가로 평가해줘".

This file was written **before any result of the search was seen**. The u40 q2/q4 runs were in flight; only the plain u40
and u40-l0keep results were known.

## Why a protocol

Trying many settings and keeping the one where MoOP wins is a multiple-comparisons problem: some setting will win by
chance. So the search is split.

- **Exploration** on Test500 (500 clips) and OOD-Val250. These sets were already used for the u40 results.
- **Confirmation** of one pre-registered setting on **OOD-train**, which is fresh for Edge. This is the user's "ood train"
  request.
  - It has 1,271 clips; the 1,178 with t0 >= 2 s are used, by the same rule as OOD-Val250.
  - Manifest: `outputs/iclr27-rebuttal/edge_moop/ood_train_1178.parquet`.
- Every explored setting is reported, wins and losses alike.

## Exploration grid

All arms use the calib_100 importance, the no-nav prompt and the same open-loop protocol. Each setting is run for MoOP,
FM-only and CE-only.

| round | settings | why |
|---|---|---|
| known | u40 plain, u40-l0keep | baseline: MoOP no better than either half |
| 1 (running) | u40-q2, u40-q4 | fewer Q heads, more MLP channels, same budget (removed MoOP's loss on A2 u50) |
| 2 | u50 plain, u50-q4, u50-q2 | higher sparsity. On A1.5 MoOP's edge appears where FM-only breaks the CoC and CE-only breaks driving; Edge's FM-only already shows 3.6 % degenerate CoCs at u40 |
| 3 (if needed) | chosen from rounds 1–2, e.g. u55 / u60 plain or a further q-cut | follow the direction that moves MoOP − min(FM, CE) downwards |

Round 2/3 picks depend on round-1 results. That is allowed, because exploration is not the claim.

## Selection rule for the confirmatory setting (fixed now)

- Metric: minADE@6 at 6.4 s, paired per clip, pooled over the 750 exploration clips (Test500 + OOD-Val250).
- For each setting, compute **gap = max(MoOP − FM-only, MoOP − CE-only)**. Negative means MoOP beats both halves.
- Pick the setting with the **smallest gap**.
- Ties within 0.005 m go to the setting with lower sparsity, then fewer Q-head cuts.
- If no setting has a negative gap, the pick is still the smallest gap. The report then says plainly that the search
  found no winning setting.

## Confirmatory test on OOD-train1178 (pre-registered; the setting is named in an addendum before the run)

- Arms: dense + MoOP / FM-only / CE-only of the picked setting. Same protocol, Ada cards only (one GPU type).
- **Primary:**
  - Metric: minADE@6 at 6.4 s.
  - Contrasts: MoOP − FM-only and MoOP − CE-only, paired per clip.
  - Statistics: 200,000-draw bootstrap with 95 % two-sided CIs, Holm-adjusted over the 2 contrasts.
  - **Success = both Holm-adjusted p < 0.05 with negative means** (MoOP better than both).
  - Partial success = one of the two.
- **Secondary** (reported, not decisive):
  - minFDE@6 at 6.4 s
  - minADE@6 at 1.6 / 3.2 s
  - bucket means
  - CoC readouts
  - each arm vs dense
- If the cards are taken by others, the run waits. Nothing runs on shared cards.

## Execution

- Ada 4–7, empty cards only, `edge_queue.py`, 4 CPU threads per job.
- Masks: `make_edge_moop_masks.py` (`--pct`, `--q-cut`).
- The OOD-train Edge cache is built in the background (`pre_processed/oodtrain_edge`) with a GT gate against
  `ood_train_a2`.

## Addendum A: pre-registration of the confirmatory setting

Written 2026-11-15 ~06:55 KST, after exploration ended and **before any OOD-train result of the three pruned arms existed**.
Dense was already queued on OOD-train because it is needed whatever the pick; its numbers were not looked at before this
addendum.

**Exploration result.** Rule applied as written; `code/edge_search_summary.py`, `outputs/iclr27-rebuttal/edge_driving_search/`.
Pooled Test500 + OOD-Val250, minADE@6 at 6.4 s.

| setting | MoOP − FM-only | MoOP − CE-only | gap |
|---|---|---|---|
| **u50** | **−0.179 [−0.270, −0.087]** | **−0.297 [−0.395, −0.199]** | **−0.179** |
| u70-q2 | −0.114 | −0.370 | −0.114 |
| u50-q2 | −0.042 | −0.012 | −0.012 |
| u60-q2 | −0.000 | −0.053 | −0.000 |
| u40-q2 | +0.013 | −0.042 | +0.013 |
| u50-q4 | +0.032 | +0.041 | +0.041 |
| u40-q4 | +0.054 | −0.012 | +0.054 |
| u40 | +0.065 | +0.066 | +0.066 |
| u60 | +0.608 | −0.973 | +0.608 |

u40-l0keep is not eligible (Test500 only).

**Picked setting: u50**, the plain literal 0.50 ratio per layer:
- 8 of 16 Q heads and 4,608 of 9,216 MLP channels removed per layer, in every one of the 28 VLM layers
- 645,922,816 parameters = 24.4 % of 2.65 B (45.8 % of the text decoder)
- expert, KV heads, vision tower and embeddings untouched
- calib_100 importance (`edge_importance_v1`), no-nav prompt

Masks, fixed by sha256:

| arm | file | sha256 |
|---|---|---|
| MoOP | `masks_moop_u50.npz` | fe64a6c4758cfbf9898ab29fa9e34fe731e3328520558dbf250fb58c4e4cc096 |
| FM-only | `masks_fm_only_u50.npz` | 4b7307536adf956f3956957d78f24c40974eb47655ac820c263b2e1d9d534892 |
| CE-only | `masks_ce_only_u50.npz` | b198fb4414e7dcdb6be245d24ec98cdd7461c538f75e41e79dd62f77ed54ca9d |

**Confirmatory run.**
- `edge_oodtrain_{moop,fm_only,ce_only}_u50` + `edge_oodtrain_dense` on OOD-train1178 (`ood_train_1178.parquet`).
- Same runner and protocol; Ada cards 4–7.

**Primary**, unchanged from above:
- Metric: minADE@6 at 6.4 s, paired.
- Contrasts: MoOP − FM-only and MoOP − CE-only. Bootstrap 200,000, Holm over the 2.
- SUCCESS = both negative with Holm p < 0.05. PARTIAL = one.
- Analysis: `code/analyze_edge_confirm.py --setting-suffix _u50 --name u50`.

**Known caveats, stated before the result:**
- On OOD-Val250 alone, u50's MoOP − FM-only was −0.048 [−0.186, +0.096], not significant. The pooled win leans on
  Test500.
- At u60, MoOP collapses (2.16 m vs FM-only 1.55 m), so MoOP's advantage is a narrow sparsity regime rather than a
  general one.

## Reports

- Exploration rounds: `reports/2026-11-15_a2-edge-qcut-u40.md` (round 1), then one search report
  `reports/2026-11-15_a2-edge-moop-driving-search.md` with every setting.
- Confirmation: added to the search report, with the pre-registration addendum quoted verbatim.
