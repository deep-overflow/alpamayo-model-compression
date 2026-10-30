# mtw24_kd2 expert re-fit on a frozen VLM (mtw24_kd2ef)

Approved 2026-10-27 ("진행해줘") after the 2x2 swap diagnosis.

## Why

`mtw24_kd2` (all four KISTI changes in one run) came back with val500 minADE@6 2.48 (q25) and about
2.0 (q50), against R0 0.867 and zero-shot 1.308, with fluent CoC (degeneracy 0.2%). Training
velocity MSE rose from the first steps. The 2x2 swap on R0's 64 held-out windows (velocity relative
MSE vs the stored dense teacher) located the fault in the VLM cache, not the expert:

| VLM | expert | relMSE |
|---|---|---|
| zero-shot | zero-shot | 0.0207 |
| R0 e3 | R0 e3 | 0.0087 |
| kd2 q100 | kd2 q100 | 0.0565 |
| kd2 q100 | zero-shot | 0.1276 |
| kd2 q100 | R0 e3 | 0.0899 |
| zero-shot | kd2 q100 | 0.0275 |
| R0 e3 | kd2 q100 | 0.0173 |

The online teacher was checked exact against the stored targets (FSDP, micro-batch 6; relMSE
1e-5 to 1e-4), so the targets were right.

## Question

With the kd2 VLM frozen, can its expert be re-fitted to the cache it actually produces, bringing
driving back to R0 level while keeping kd2's VLM (and so its LingoQA behaviour) unchanged?

## Setup (one change against R0's recipe: what trains)

- Start: `outputs/minitron/mtw24_kd2/ckpt_q100` (VLM and expert). Not chosen by val500 minADE,
  which the broken interface contaminates; q100 has the lowest driving-token KL (0.165 vs 0.222).
- Frozen: the whole VLM (vision tower, language model, lm_head). It runs once per micro-batch in
  eval mode under no_grad for the cache; no KL term.
- Trained: expert, action_in_proj, action_out_proj. Loss: velocity MSE against the dense teacher at
  the GT interpolant (D1), same `kd_draw` slots 0-2 = epochs 1-3 as R0's stored targets. The teacher
  runs online on NEURON (checked equal to the stored targets 2026-10-27: relMSE 1e-5 to 1e-4).
- Everything else as R0 (`mtw24_kd`): kd_r1, 64 held-out windows (same seed-0 draw), 3 epochs,
  global batch 64, AdamW (0.9, 0.95), wd 0.1, clip 1.0, LR 1e-4 -> 1e-5 cosine, warmup 40.
- Hardware: NEURON A100 80GB x 4 (user: training runs on KISTI; Blackwell 0-3 are needed for another
  experiment), `minitron/code/kd_refit_job.sbatch`. kd_r1 windows and ckpt_q100 are already there,
  so only code moves. A cvlab21 smoke (micro-batch 4, precomputed targets) passed first: VLM
  gradient norm 0, start velocity MSE 0.12 = kd2's final held-out value, 21.6 GiB/card.
- Code: `train_kd.py --freeze-vlm`.

## Evaluation

`CKPTS="ckpt_epoch1 ckpt_epoch2 ckpt_epoch3" eval_kd.sh mtw24_kd2ef` on Ada 4-7 once `mtw24_kd2`'s
evaluation has released them: val500 selection, test_2000, OOD-Val262, LingoQA (the VLM is kd2 q100's,
so LingoQA reflects q100), CoC degeneracy. Compared with R0, kd2, dense, MoOP zs, MT-W zs.

## Reading

- Driving back near R0: the kd2 VLM keeps its driving information and the failure was the
  interface only; the frozen-VLM stage is the fix for VLM-side KD changes.
- Not back: the VLM lost driving information too; next would be a cache-preservation term during
  VLM training or reweighting the group KL / VQA share (user's call).

## Status

- 2026-10-27: NEURON job 949586 submitted, then held after the vision-tower diagnosis (the refit would fit
  the expert to kd2's damaged vision tower). 2026-10-30: cancelled at the user's request; it never ran.
