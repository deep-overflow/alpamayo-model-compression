# Plan: paper analyses (Appendix C.4 panels) on Alpamayo 2 Edge

**Status: done 2026-11-14.**
- Report: `reports/2026-11-14_a2-edge-analysis-replication.md`.
- Every gate passed.
- The importance run went on card 7 instead of cards 6+7. Card 6 was taken by co-tenants mid-launch, so the run waited
  for an empty card via `code/wait_gpu_and_run.py`.
- One card was enough: 10.4 min, bitwise deterministic.

2026-11-14. First step of "compare MoOP, FM-only and CE-only on Alpamayo 2 Edge". This plan covers only the six
analysis panels in `paper/fig/assets/supp_alpamayo2_panels/`. Masks and evaluation come in a later plan, but they
reuse the importance this run produces.

## Purpose

The six panels exist for Alpamayo 1.5 and Alpamayo 2 Super:

| figure | panel | what it shows |
|---|---|---|
| Fig. 3 | (a) `rank_correlation` | per VLM layer, corrected Spearman rho between I_FM and I_CE over Q heads, all positions pooled and per token type |
| Fig. 3 | (b) `share_fm` | share of each layer's I_FM earned at vision / prompt-text / CoC / ego-history positions |
| Fig. 3 | (c) `share_ce` | the same for I_CE |
| Fig. 5 | (a) `steps_q_head` | expert Q heads: rank agreement between denoising steps s and s' (10 x 10) |
| Fig. 5 | (b) `steps_mlp` | the same for expert MLP channels |
| Fig. 5 | (c) `mass_curve` | C(r): share of importance held by the least important fraction r of units, Q heads vs MLP channels |

Questions:
1. Does Finding 1 hold on Edge: I_FM and I_CE agree in early layers and disagree after a depth step? Where is the step?
2. Is the late-layer I_CE carried by text (A1.5) or still by vision (A2 Super)?
3. Expert: are Q-head rankings less stable across steps than MLP rankings, and is MLP importance flat (small C(0.9))?

Why Edge matters: A1.5 and A2 Super are both Qwen3-VL. Edge is built on Cosmos3-Edge (Nemotron-based text tower,
SigLIP2 vision). It is the first check outside the Qwen3-VL family, so it answers the reviewer's "only one model
family?" more strongly than A2 Super did.

## Edge vs A2 Super (facts from the checkpoint config, revision f8951d0)

| | A2 Edge | A2 Super |
|---|---|---|
| total params | 2.65 B | 35.8 B |
| VLM layers / hidden | 28 / 2048 | 64 / 5120 |
| VLM Q heads / KV heads / head_dim | 16 / 8 / 128 | 64 / 8 / 128 |
| VLM MLP | `fc1 -> relu^2 -> fc2`, 9,216 channels | gated SwiGLU (`down_proj`), 25,600 |
| expert layers / hidden | 28 / 512 | 64 / ... |
| expert Q heads / MLP channels | 16 / 3,840 | 16 / 6,144 |
| cameras / frames | 6 cams, 22 frames (front + tele 5 each at -2/-1/-0.5/-0.1/0 s, others 3 at -1/-0.1/0 s) | 6 cams x 4 frames |
| ego history | 20 poses (-1.9..0 s) | 16 poses |
| navigation line in prompt | **always** (`Navigation input is: ...`) | optional; our A2 runs used none |
| action space | unicycle accel/curvature, course heading | unicycle accel/curvature |

## Steps

### 0. Environment and checkpoint
- `uv sync --locked --extra dataset` in `/home/cvlab21/project/chan/alpamayo2-edge` (Python 3.12, torch 2.8.0,
  transformers 5.17.0). The venv goes to `/mnt/nvme1n1/ad_vla/venvs/alpamayo2-edge`, linked as `.venv` (storage rule).
- Download `nvidia/Alpamayo2-Edge@f8951d0` (5.3 GB) into `HF_HUB_CACHE=/mnt/nvme1n1/ad_vla/cache/hub`.
  `df` shows 202 GB free.
- Token: the Seongchan token in `alpasim-runs/.hf_env`, with `HF_HOME` and `HF_HUB_CACHE` both set.

### 1. Edge calibration cache
- Clips: `calib_100.parquet`, the same 100 clips at t0 = 5.1 s as A1.5 and A2 Super.
- Inputs: Edge's own `load_physical_aiavdataset`, which streams from the Hub (the raw camera mirror was pruned 10-17).
  It gives 22 frames and 20 history poses.
- GT: the 64 future poses come from the same egomotion, needed for the FM loss.
- Storage: one npz per clip in `physicalai_av/pre_processed/calib_edge/`. Layout follows `cache_a2.py`: JPEG q95,
  actual capture timestamps, `nav_text`.
- Navigation text: none (decision Q1). The cache stores no `nav_text`; the prompt builder drops the line.

### 2. Smoke (3 clips, release inference path)
- CoC is readable and ends with `<|traj_future_start|>`.
- minADE against GT is sane.
- Report how many CoCs close with the anchor, and minADE, under the no-nav prompt.

### 3. Estimator port: `iclr27-rebuttal/code/edge_grad_lib.py` + `run_importance_edge.py`
This is the same protocol as `alpamayo2-distill/experiments/importance/run_importance_a2.py`:
1. no-grad CoC rollout: release sampling (top_p 0.98, T 0.6, max 256), seed `sha256("42:clip")[:4]`
2. teacher-forced VLM forward with grad -> KV cache
3. CoC NLL backward -> I_CE by token type (TypedUnitGates)
4. 10-step FM through the expert onto detached cache leaves: `t_s = (s+0.5)/10`, `x_t = t x1 + (1-t) eps`,
   target `x1 - eps` (Edge's own training convention, `diffusion/flow_matching.py`). Then one VLM backward
   -> I_FM by token type.
5. expert gates read and zeroed after every step -> per-step expert importance

Changes for Edge, mechanical only:
- Q gates: input of `self_attn.o_proj` (16 x 128).
- MLP gates: input of `mlp.fc2` (relu^2 channels: VLM 9,216, expert 3,840). Edge has no `down_proj`.
- One card, no offload. 2.65 B in bf16 plus about 3.4k tokens of saved activations fits in 48 GB.
- Rollout: Edge's logits processors (history and future trajectory vocab masked, text EOS blocked) and its
  `StopAfterEOS`. CoC span = all generated tokens including the closing anchor, as on A2.
- GT action: Edge's course action space `traj_to_action(history 20, future 64)`.
- Expert positions: Edge's `build_expert_pos_ids_and_attn_mask`. HF5 rope deltas are relative to unpadded text;
  batch 1 has no padding.
- Token types: same rule as A2 (`vision` = image pad, `hist` = trajectory tokens, `sink` = position 0,
  `coc` = generated, `prompt_text` = the rest). The navigation line therefore counts as prompt text.

Gates before the full run:
- G0: the typed gates summed over types equal the plain gates (Q heads), max relative error < 1e-3.
- G0b: the per-step expert gradients summed over steps equal the accumulated pass.
- G1: on every clip the tail is correct, the NLL is finite and the FM loss is finite.
- G2: one clip run twice gives the same arrays (determinism flags as in CLAUDE.md).

### 4. Full run
- 100 clips on Ada cards 6 and 7, sharded 2 ways. Both showed 2 MiB at planning time. I check `nvidia-smi` right
  before launch and launch with `--exp-id`.
- Finalize into the same file layout as A2: `importance.npz`, `anatomy_perclip_{q,mlp}.npz`,
  `step_importance.npz`, `metrics.json`.
- Output directory: `outputs/iclr27-rebuttal/edge_importance_v1/`.

### 5. Analysis
- Add an `edge` entry to `a2_fig3_token_type.py` and `a2_fig5_expert.py` (n = 28, ticks, kv_group 2), or write thin
  `edge_*` wrappers that import their functions. Pick whichever leaves the A1.5 and A2 paths byte-identical.
- Grey band: the same rule as A2, the A1.5 late band moved by relative depth (22/36 x 28 = 17.1 -> layers 17-27).
- Step tau: fitted with the same step fit.
- Gate: the A1.5 and A2 rows reproduce their stored 2026-11-08 metrics exactly.
- Also report the step-aligned post-step Spearman (trap 2 of the A2 replication). Bands are not compared by relative
  depth alone.

### 6. Panels
- Write `edge_{rank_correlation, share_fm, share_ce, steps_q_head, steps_mlp, mass_curve}.pdf` into
  `supp_alpamayo2_panels/` with the same code paths as `supp_alpamayo2_figs.py`.
- Build a three-row composite (A1.5 / A2 Super / A2 Edge) as **new** files: `supp_alpamayo2_{vlm,expert}_3row.pdf`.
- The current two-row PDFs the paper includes stay untouched until you decide to swap them.

### 7. Report
- Write `iclr27-rebuttal/reports/2026-11-1X_a2-edge-analysis-replication.md` with facts and analysis separated.
- Add a row to the README index.

## Cost
- Cache: streaming 100 clips x 6 cameras, an estimated 30-60 min.
- Importance: an estimated 10-20 s per clip, so about 15-30 min on two cards.
- Analysis: CPU only.

## Not in this plan
- MoOP / FM-only / CE-only masks on Edge. This needs a decision on how much to remove per layer, made after these
  panels.
- Open-loop evaluation of those masks.
- LingoQA.
- Closed loop.

## Decisions
- **Q1. Navigation text: DROP THE LINE (user, 2026-11-14).**
  - Edge's `build_conversation` puts `Navigation input is: <nav>` and a newline between the ego-history block and the
    task sentence. Both items are removed. Everything else (system prompt, camera and time labels, history block,
    `DRIVING_TASK`, closed `<think></think>`) is Edge's own.
  - This matches the A2 Super runs, which had no navigation line.
  - Consequence: the Edge prompt is outside its training distribution. The smoke step reports how often the CoC
    still closes with `<|traj_future_start|>`, and the minADE, so we know how far off it is.
  - The same no-nav prompt will be used for the later MoOP / FM-only / CE-only evaluation.
  - Rejected alternatives:
    - nav derived from the GT future (it leaks the future direction)
    - one fixed phrase (it is wrong on turning clips)
- **Panels.** The Edge panels and the three-row composite are new files. The paper's two-row PDFs stay as they are.
