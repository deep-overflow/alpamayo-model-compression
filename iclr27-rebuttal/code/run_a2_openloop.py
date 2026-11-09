"""Open-loop test500 evaluation of Alpamayo 2 Super, dense or with VLM unit masks (MoOP).

Plan: iclr27-rebuttal/plans/2026-11-09_a2-moop-test500.md. Reuses alpamayo2-distill's run_oodval.py (the A2 runner matched to
A1.5's run_baseline.py): one CoC rollout (temperature 0.6, top_p 0.98), K=8 seeded denoisings on its KV cache, seed =
sha256(seed:clip_id)[:4] + k, max_gen 256, model split by layer over the visible cards. Only the clip set (manifest +
cache), the output location and the masks differ.

Masks: 0/1 per Q head at the o_proj input and per MLP channel at the down_proj input of every VLM text layer (forward
pre-hooks; zeroing there = removing the unit). Expert, KV groups and the vision tower are untouched.

--gate runs the first N clips three times in one process (no hooks, all-ones hooks, the given masks) and checks that
all-ones is bitwise equal to no hooks and that the masks change the output (G1 of the plan).

Usage (cards from the shell, model from the local cache):
  CUDA_VISIBLE_DEVICES=4,5 HF_HUB_OFFLINE=1 /mnt/nvme1n1/ad_vla/venvs/alpamayo2/bin/python \
      iclr27-rebuttal/code/run_a2_openloop.py --exp-id iclr27-rebuttal/a2_test500_dense --shard 0 --n-shards 2
  ... --masks outputs/iclr27-rebuttal/a2_moop/masks_moop.npz --exp-id iclr27-rebuttal/a2_test500_moop
  ... --gate 3 --masks outputs/iclr27-rebuttal/a2_moop/masks_moop.npz --exp-id iclr27-rebuttal/a2_test500_gate
"""

import os
import sys
from pathlib import Path

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    raise SystemExit("set CUDA_VISIBLE_DEVICES to this replica's cards, e.g. 4,5")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
# both paths set here so the import order below does not matter (run_oodval would add the second itself)
sys.path.insert(0, "/home/cvlab21/project/chan/alpamayo2-distill/experiments/oodval")
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))  # eval_lib (numpy only)
import cache_a2 as ca
import eval_lib as el
import run_oodval as ro
from alpamayo2_super import helper
from alpamayo2_super.input_profiles import select_task_input
from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

MANIFEST = Path("/mnt/nvme1n1/ad_vla/outputs/chan/eval_sets/test_500.parquet")


class VlmMasks:
    """Forward pre-hooks that zero masked Q heads (o_proj input) and MLP channels (down_proj input)."""

    def __init__(self, model, q_mask, mlp_mask):
        layers = model.vlm.model.language_model.layers
        assert q_mask.shape[0] == mlp_mask.shape[0] == len(layers), (q_mask.shape, mlp_mask.shape, len(layers))
        self.handles = []
        for i, layer in enumerate(layers):
            attn = layer.self_attn
            n_heads, head_dim = q_mask.shape[1], attn.head_dim
            dev = attn.o_proj.weight.device
            qm = torch.tensor(q_mask[i], device=dev, dtype=torch.bfloat16).view(1, 1, n_heads, 1)
            mm = torch.tensor(mlp_mask[i], device=dev, dtype=torch.bfloat16).view(1, 1, -1)

            def q_hook(module, args, qm=qm, n_heads=n_heads, head_dim=head_dim):
                x = args[0]  # (B, T, H*D)
                b, t, _ = x.shape
                return ((x.view(b, t, n_heads, head_dim) * qm.to(x.dtype)).view(b, t, -1),)

            def m_hook(module, args, mm=mm):
                return (args[0] * mm.to(args[0].dtype),)  # (B, T, I)

            self.handles.append(attn.o_proj.register_forward_pre_hook(q_hook))
            self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(m_hook))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def eval_clip(model, processor, m, args, diffusion_kwargs):
    data = select_task_input(ca.load_cached(ca.path_for(m["clip_id"], m["t0_us"], args.cache)),
                             "trajectory")  # image_frames (6, 4, 3, H, W)
    gt_xy = data["ego_future_xyz"][0, 0, :, :2].numpy()  # (64, 2)
    model_inputs = ro.build_inputs(model, processor, data)
    base = ca.clip_seed(args.seed, m["clip_id"])
    traj_seeds = [base + k for k in range(args.k)]
    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_k, gen_coc, gen_len, has_eos, t_gen, t_den, prompt_len = ro.rollout_then_k_traj(
            model, model_inputs, base, traj_seeds, args.max_gen, diffusion_kwargs)
    if not np.isfinite(pred_k).all():
        raise RuntimeError(f"non-finite trajectory on {m['clip_id']}")
    ade_k, fde_k = el.ade_fde(pred_k, gt_xy)  # (K,), (K,)
    rec = {"clip_id": m["clip_id"], "bucket": el.bucket(gt_xy), "seed": base,
           "minADE_rollout": float(ade_k.min()), "minFDE_rollout": float(fde_k.min()),
           "ade_rollout_k": [round(float(x), 6) for x in ade_k],
           "fde_rollout_k": [round(float(x), 6) for x in fde_k],
           "gen_coc": gen_coc, "gen_len": gen_len, "has_eos": has_eos,
           **{f"coc_{k}": v for k, v in el.coc_degenerate(gen_coc).items()}}
    for h in (16, 32):
        a_h, f_h = el.ade_fde(pred_k[:, :h], gt_xy[:h])
        rec.update({f"ade_rollout_k_h{h}": [round(float(x), 6) for x in a_h],
                    f"fde_rollout_k_h{h}": [round(float(x), 6) for x in f_h]})
    rec.update({"pred_xy_k": np.round(pred_k.astype(float), 3).tolist(), "prompt_len": prompt_len,
                "sec_gen": round(t_gen, 2), "sec_denoise": round(t_den, 2), "sec_total": round(time.time() - t0, 2)})
    return rec, pred_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True, help="outputs/<exp-id> in this repo")
    ap.add_argument("--masks", default=None, help="npz with q_mask (64, 64), mlp_mask (64, 25600); none = dense")
    ap.add_argument("--cache-name", default="test500_a2")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--first-share", type=float, default=0.47)
    ap.add_argument("--tag", default="alpamayo2")
    ap.add_argument("--gate", type=int, default=0, help="G1: run the first N clips as none / ones / masks and compare")
    ap.add_argument("--plan", default="iclr27-rebuttal/plans/2026-11-09_a2-moop-test500.md")
    args = ap.parse_args()
    args.cache = ca.AV / "pre_processed" / args.cache_name
    args.gpus = os.environ["CUDA_VISIBLE_DEVICES"]
    n_gpus = torch.cuda.device_count()
    assert n_gpus == len(args.gpus.split(",")) >= 2, (n_gpus, args.gpus)

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    man = [{"clip_id": r.clip_id, "t0_us": int(r.t0_us)} for r in pd.read_parquet(MANIFEST).itertuples()]
    man = man[args.shard::args.n_shards]  # strided, the same rule as run_baseline.py
    if args.gate:
        man = man[: args.gate]
    elif args.limit:
        man = man[: args.limit]

    ro.set_determinism()
    share = args.first_share if n_gpus == 2 else 1.0 / n_gpus - 0.03
    device_map, per_gpu = ro.layer_device_map(n_gpus, share)
    gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
    print(f"{args.exp_id} | shard {args.shard}/{args.n_shards} | {len(man)} clips | cards {args.gpus} "
          f"({gpu_names[0]}) | layers per card {per_gpu} | masks {args.masks}", flush=True)
    t_load = time.time()
    model = Alpamayo2Super.from_pretrained(ro.MODEL_ID, revision=ro.MODEL_REV, dtype=torch.bfloat16,
                                           device_map=device_map)
    model.eval()
    processor = helper.get_processor(model.tokenizer, model.config)
    print(f"model loaded in {time.time() - t_load:.0f}s | mem GB "
          f"{[round(torch.cuda.memory_allocated(i) / 2**30, 1) for i in range(n_gpus)]}", flush=True)
    diffusion_kwargs = {}
    masks = None
    if args.masks:
        z = np.load(args.masks)
        masks = (z["q_mask"], z["mlp_mask"])

    if args.gate:
        res = {}
        for mode in ("none", "ones", "masks"):
            hooks = None
            if mode == "ones":
                hooks = VlmMasks(model, np.ones_like(masks[0]), np.ones_like(masks[1]))
            elif mode == "masks":
                hooks = VlmMasks(model, *masks)
            res[mode] = [eval_clip(model, processor, m, args, diffusion_kwargs) for m in man]
            if hooks:
                hooks.remove()
            print(f"gate {mode}: " + " | ".join(f"{r['clip_id'][:8]} ade@8 {r['minADE_rollout']:.4f} "
                                                f"{r['gen_coc'][:50]!r}" for r, _ in res[mode]), flush=True)
        ident = all(np.array_equal(a[1], b[1]) and a[0]["gen_coc"] == b[0]["gen_coc"]
                    for a, b in zip(res["none"], res["ones"], strict=True))
        changed = any(not np.array_equal(a[1], b[1]) or a[0]["gen_coc"] != b[0]["gen_coc"]
                      for a, b in zip(res["none"], res["masks"], strict=True))
        finite_eos = all(r["has_eos"] for r, _ in res["none"])
        g = {"G1_ones_bitwise_equal_none": bool(ident), "G1_masks_change_output": bool(changed),
             "G3_dense_has_eos": bool(finite_eos), "clips": [m["clip_id"] for m in man],
             "records": {k: [r for r, _ in v] for k, v in res.items()}}
        (out_dir / "gate.json").write_text(json.dumps(g, indent=2))
        print(f"G1 ones==none bitwise: {ident} | masks change output: {changed} | dense EOS: {finite_eos}", flush=True)
        sys.exit(0 if ident and changed and finite_eos else 1)

    hooks = VlmMasks(model, *masks) if masks is not None else None
    rows_path = out_dir / f"{args.tag}_s{args.shard}of{args.n_shards}.json"
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    done = {r["clip_id"] for r in rows}
    (out_dir / f"config_s{args.shard}of{args.n_shards}.json").write_text(json.dumps({
        "model": ro.MODEL_ID, "model_revision": ro.MODEL_REV, "set": "test", "manifest": str(MANIFEST),
        "cache": str(args.cache), "n_clips": len(man), "k": args.k, "seed": args.seed,
        "seed_rule": "sha256(f'{seed}:{clip_id}')[:4], +k per sample",
        "protocol": "1 CoC rollout, k denoisings on its KV cache (alpamayo2-distill run_oodval.py, matched to run_baseline.py)",
        "sampling": {"do_sample": True, "temperature": 0.6, "top_p": 0.98}, "max_gen": args.max_gen,
        "input_profile": "trajectory: cameras [0,1,2,3,5,6] x 4 frames", "masks": args.masks,
        "mask_placement": "forward pre-hooks: o_proj input per Q head, down_proj input per MLP channel (VLM text layers)",
        "shard": [args.shard, args.n_shards], "gpus": args.gpus, "gpu_names": gpu_names, "layers_per_gpu": per_gpu,
        "plan": args.plan}, indent=2))
    t_start = time.time()
    for i, m in enumerate(man):
        if m["clip_id"] in done:
            continue
        rec, _ = eval_clip(model, processor, m, args, diffusion_kwargs)
        rows.append(rec)
        if len(rows) % 5 == 0 or i + 1 == len(man):
            rows_path.write_text(json.dumps(rows, indent=2))
        print(f"[{i + 1}/{len(man)}] {m['clip_id'][:8]} {rec['bucket']:10s} ade@8={rec['minADE_rollout']:.3f} "
              f"gen={rec['gen_len']}tok tot={rec['sec_total']:.1f}s | {rec['gen_coc'][:80]}", flush=True)
    rows_path.write_text(json.dumps(rows, indent=2))
    a = np.array([min(r["ade_rollout_k"][:6]) for r in rows])
    line = (f"{args.tag} {args.exp_id} test500 shard {args.shard}/{args.n_shards}: {len(rows)} clips, "
            f"minADE@6 mean {a.mean():.4f} median {np.median(a):.4f}, "
            f"CoC degen {np.mean([r['coc_degenerate'] for r in rows]):.3f}")
    (out_dir / f"summary_s{args.shard}of{args.n_shards}.txt").write_text(line + "\n")
    print(f"\n{line}\n{(time.time() - t_start) / 60:.1f} min -> {rows_path}", flush=True)
    if hooks:
        hooks.remove()


if __name__ == "__main__":
    main()
