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

## Reports

- Exploration rounds: `reports/2026-11-15_a2-edge-qcut-u40.md` (round 1), then one search report
  `reports/2026-11-15_a2-edge-moop-driving-search.md` with every setting.
- Confirmation: added to the search report, with the pre-registration addendum quoted verbatim.
