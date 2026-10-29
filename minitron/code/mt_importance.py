"""Minitron activation importance on the calibration windows (plan D9, formula in section 2).

The formula follows NVIDIA ModelOpt `importance_hooks/base_hooks.py` (L2NormHook): per
window and channel c, a_c = mean over positions of |x_c| in float32; the score is
sqrt(sum over windows of a_c^2) -- Minitron's (batch = L2, seq = mean).

Per calibration window, one teacher-forced forward of the full distillation sequence
[prompt + teacher CoC + cot_end + traj_future_start + 128 GT trajectory tokens] (D9-2),
no gradient, dense teacher.

  MLP channels     x = input of down_proj, per layer         -> mlp (36, 12288)
  hidden channels  x = output of every text-tower RMSNorm    -> hidden_by_norm (73, 4096)
                   (input + post-attention per layer, final);   hidden = sum over the 73
  heads            not scored -- the width arm keeps them

Diagnostics only (D9-3): the same statistic restricted to each token type's positions --
vision, ego history, other prompt text, CoC, trajectory -- from the same forward.

Kept sets for the 24.0% width arm (D7/D8): hidden top 3,328 by `hidden`, MLP top 9,216 per
layer by `mlp`, ascending ids, in kept.json.

Outputs outputs/minitron/<exp-id>/: importance.npz, kept.json, config.json, metrics.json,
summary.txt. Resumable: running sums are checkpointed every 64 windows.

Usage:
  ALPAMAYO_REPO=$PWD bash experiments/head_analysis/run_retry_host.sh 3 \\
      minitron/code/mt_importance.py --gpu 6
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
sys.path.insert(0, str(REPO / "experiments" / "evaluation"))
sys.path.insert(0, str(REPO / "experiments" / "recovery"))

import analysis_lib as lib
import recover_lib as rl
import sample_cache as sc
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from expert_per_clip import reserve_gpu
from run_baseline import MODEL_REV, set_determinism

TYPES = ("all", "vision", "hist", "prompt_text", "coc", "traj")
H_KEEP, I_KEEP = 3328, 9216


class L2MeanAbs:
    """sum over windows of (mean over positions |x|)^2, per position type. Row 0 = all."""

    def __init__(self, n_ch, device):
        self.sq = torch.zeros(len(TYPES), n_ch, dtype=torch.float64, device=device)

    def add(self, x, masks):
        a = x[0].float().abs()  # (T, C)
        self.sq[0] += a.mean(0).double() ** 2
        for i, m in enumerate(masks, start=1):
            if m.any():
                self.sq[i] += a[m].mean(0).double() ** 2


def type_masks(model, input_ids, coc_end, total, device):
    """Bool (T,) masks for vision / hist / prompt_text / coc / traj."""
    sp = lib.compute_spans(model, input_ids)
    p = input_ids.shape[1]
    pad = torch.zeros(total - p, dtype=torch.bool)
    out = [torch.cat([sp["vision"], pad]), torch.cat([sp["hist"], pad]),
           torch.cat([sp["text"] | sp["sink"], pad])]
    coc = torch.zeros(total, dtype=torch.bool)
    coc[p:coc_end] = True
    traj = torch.zeros(total, dtype=torch.bool)
    traj[coc_end:] = True
    return [m.to(device) for m in out + [coc, traj]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="mt_importance")
    ap.add_argument("--calib", default="outputs/minitron/kd_sets/calib_1024.parquet")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--reserve-gb", type=float, default=36.0)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N windows only")
    ap.add_argument("--allow-missing", action="store_true",
                    help="debug: skip calibration windows that are not cached yet")
    args = ap.parse_args()
    out_dir = REPO / "outputs" / "minitron" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    calib = pd.read_parquet(REPO / args.calib)
    cached = [sc.path_for("train", c, int(t)).exists() for c, t in zip(calib.clip_id, calib.t0_us)]
    if not all(cached):
        missing = len(cached) - sum(cached)
        if not args.allow_missing:
            raise SystemExit(f"{missing} calibration windows are not cached yet")
        calib = calib[cached]
        print(f"skipping {missing} uncached windows (--allow-missing)", flush=True)
    if args.limit:
        calib = calib.iloc[: args.limit]

    set_determinism(warn_only=True)
    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    gpu_name = torch.cuda.get_device_name(device)
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    tok = model.tokenizer
    lm = model.vlm.model.language_model
    tc = model.vlm.config.text_config

    mlp = [L2MeanAbs(tc.intermediate_size, "cuda") for _ in lm.layers]
    norms = [n for layer in lm.layers
             for n in (layer.input_layernorm, layer.post_attention_layernorm)] + [lm.norm]
    hid = [L2MeanAbs(tc.hidden_size, "cuda") for _ in norms]
    cur = {"masks": None}
    for li, layer in enumerate(lm.layers):
        layer.mlp.down_proj.register_forward_pre_hook(
            lambda mod, a, li=li: mlp[li].add(a[0], cur["masks"]))
    for k, norm in enumerate(norms):
        norm.register_forward_hook(lambda mod, a, o, k=k: hid[k].add(o, cur["masks"]))

    ckpt = out_dir / "partial.pt"
    done = set()
    if ckpt.exists():
        st = torch.load(ckpt, map_location="cuda")
        for s, sq in zip(mlp + hid, st["sq"]):
            s.sq.copy_(sq)
        done = set(st["done"])

    (out_dir / "config.json").write_text(json.dumps({
        **vars(args), "n_windows": len(calib), "types": TYPES, "model": "nvidia/Alpamayo-1.5-10B",
        "revision": MODEL_REV, "gpu": gpu_name, "H_keep": H_KEEP, "I_keep": I_KEEP,
        "formula": "sqrt(sum_windows (mean_positions |x|)^2); ModelOpt L2NormHook",
        "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"]}, indent=2))
    print(f"{args.exp_id} | {len(calib)} windows ({len(done)} done) | {gpu_name}", flush=True)

    t0 = time.time()
    n_new = 0
    for r in calib.itertuples():
        key = f"{r.clip_id}:{int(r.t0_us)}"
        if key in done:
            continue
        data = sc.load_cached(sc.path_for("train", r.clip_id, int(r.t0_us)))
        inputs = lib.build_inputs(model, processor, data, "cuda")
        coc = torch.tensor([rl.coc_ids(tok, r.coc)], device="cuda")  # (1, L_coc)
        traj = rl.traj_token_ids(model, data, "cuda")  # (1, 128)
        seq = torch.cat([inputs["input_ids"], coc, traj], dim=1)  # (1, T)
        cur["masks"] = type_masks(model, inputs["input_ids"],
                                  inputs["input_ids"].shape[1] + coc.shape[1], seq.shape[1], "cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model.vlm.model(input_ids=seq, attention_mask=torch.ones_like(seq),
                            pixel_values=inputs["tokenized_data"]["pixel_values"],
                            image_grid_thw=inputs["tokenized_data"]["image_grid_thw"],
                            use_cache=False)
        done.add(key)
        n_new += 1
        if len(done) % 64 == 0:
            torch.save({"sq": [s.sq for s in mlp + hid], "done": sorted(done)}, ckpt)
            print(f"[{len(done)}/{len(calib)}] {(time.time() - t0) / n_new:.2f} s/window",
                  flush=True)
    torch.save({"sq": [s.sq for s in mlp + hid], "done": sorted(done)}, ckpt)

    mlp_t = torch.stack([s.sq for s in mlp]).sqrt().cpu().numpy()  # (36, 6, 12288)
    hid_t = torch.stack([s.sq for s in hid]).sqrt().cpu().numpy()  # (73, 6, 4096)
    mlp_s = mlp_t[:, 0]  # (36, 12288)
    hidden = hid_t[:, 0].sum(0)  # (4096,)
    kept_h = np.sort(np.argsort(-hidden)[:H_KEEP])
    kept_mlp = [np.sort(np.argsort(-mlp_s[li])[:I_KEEP]).tolist() for li in range(len(lm.layers))]
    np.savez(out_dir / "importance.npz", mlp=mlp_s, hidden=hidden, hidden_by_norm=hid_t[:, 0],
             mlp_by_type=mlp_t.transpose(1, 0, 2), hidden_by_norm_by_type=hid_t.transpose(1, 0, 2),
             types=np.array(TYPES))
    (out_dir / "kept.json").write_text(json.dumps(
        {"vlm_hidden": kept_h.tolist(), "vlm_mlp": kept_mlp, "n_windows": len(done)}))

    # how much the vision-dominated pooled score agrees with each position type's own ranking
    def topk_overlap(a, b, k):
        return len(set(np.argsort(-a)[:k]) & set(np.argsort(-b)[:k])) / k

    agree = {t: {"hidden_top3328": topk_overlap(hidden, hid_t[:, i].sum(0), H_KEEP),
                 "mlp_top9216_mean": float(np.mean([topk_overlap(mlp_s[li], mlp_t[li, i], I_KEEP)
                                                    for li in range(len(lm.layers))]))}
             for i, t in enumerate(TYPES) if i > 0}
    hshare = float(hidden[kept_h].sum() / hidden.sum())
    metrics = {"n_windows": len(done), "hidden_score_share_kept": hshare,
               "pooled_vs_type_topk_overlap": agree}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    lines = [f"Minitron importance over {len(done)} windows ({gpu_name})",
             f"hidden: kept 3,328 of 4,096 carry {hshare:.1%} of the summed norm score",
             "pooled-vs-type top-k overlap (hidden / MLP mean over layers):"]
    lines += [f"  {t:12s} {v['hidden_top3328']:.3f} / {v['mlp_top9216_mean']:.3f}"
              for t, v in agree.items()]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
