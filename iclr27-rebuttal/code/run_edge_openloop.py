"""Open-loop evaluation of Alpamayo 2 Edge, dense or with VLM unit masks (plans/2026-11-14_a2-edge-moop-u40.md). Edge venv.

The A2 protocol (run_a2_openloop.py / run_oodval.rollout_then_k_traj) on Edge: one CoC rollout (T 0.6, top_p 0.98, max 256,
seed sha256(42:clip)[:4]) through Edge's generation path, then K = 8 denoisings (10 Euler steps, seed base + k) on that
rollout's KV cache. Prompt: Edge's driving prompt without the navigation line. Inputs: the Edge caches built by
build_edge_calib_cache.py. Masks: edge_eval_lib.EdgeVlmMasks (o_proj input per Q head, mlp.fc2 input per MLP channel).
Records carry the same fields as run_a2_openloop.py (ade/fde per sample at 6.4 s and at 1.6 / 3.2 s), plus `nominal`.

--gate N: the first N clips with no hooks, all-ones hooks and the masks; all-ones must be bitwise equal to no hooks and the
masks must change the output.

Usage:
  CUDA_VISIBLE_DEVICES=0 /mnt/nvme1n1/ad_vla/venvs/alpamayo2-edge/bin/python iclr27-rebuttal/code/run_edge_openloop.py \
      --set test500 --exp-id iclr27-rebuttal/edge_test500_dense [--masks outputs/iclr27-rebuttal/edge_moop/masks_moop.npz] \
      [--shard 0 --n-shards 2] [--gate 3]
"""

import os
import sys

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    raise SystemExit("set CUDA_VISIBLE_DEVICES to one card")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import edge_common as ec

ec.setup_hf_env(offline=True)

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
import edge_eval_lib as eel
import eval_lib as el

SETS = {"test500": (REPO / "outputs" / "eval_sets" / "test_500.parquet", ec.AV / "pre_processed" / "test500_edge"),
        "oodval": (REPO / "outputs" / "iclr27-rebuttal" / "edge_moop" / "ood_val_250.parquet",
                   ec.AV / "pre_processed" / "oodval_edge"),
        "calib": (ec.CALIB_MANIFEST, ec.CALIB_CACHE)}


def to_dev(x, dev):
    if torch.is_tensor(x):
        return x.to(dev)
    if isinstance(x, dict):
        return {k: to_dev(v, dev) for k, v in x.items()}
    return x


def eval_clip(model, processor, m, cache_root, args):
    data = ec.load_cached(ec.path_for(m["clip_id"], m["t0_us"], cache_root))
    gt_xy = data["ego_future_xyz"][0, :, :2].numpy()  # (64, 2)
    inputs, _ = ec.prepare_inputs_no_nav(data, model.config, model.tokenizer, processor)
    inputs = to_dev(inputs, "cuda")
    base = ec.clip_seed(args.seed, m["clip_id"])
    traj_seeds = [base + k for k in range(args.k)]
    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_k, cot, gen_len, has_eos, t_gen, t_den, prompt_len = eel.rollout_then_k_traj(
            model, inputs, base, traj_seeds, args.max_gen)
    if not np.isfinite(pred_k).all():
        raise RuntimeError(f"non-finite trajectory on {m['clip_id']}")
    ade_k, fde_k = el.ade_fde(pred_k, gt_xy)  # (K,), (K,)
    rec = {"clip_id": m["clip_id"], "bucket": el.bucket(gt_xy), "seed": base,
           "minADE_rollout": float(ade_k.min()), "minFDE_rollout": float(fde_k.min()),
           "ade_rollout_k": [round(float(x), 6) for x in ade_k], "fde_rollout_k": [round(float(x), 6) for x in fde_k],
           "gen_coc": cot, "gen_len": gen_len, "has_eos": has_eos, "nominal": cot.strip() == "<NOMINAL>",
           **{f"coc_{k}": v for k, v in el.coc_degenerate(cot).items()}}
    for h in (16, 32):
        a_h, f_h = el.ade_fde(pred_k[:, :h], gt_xy[:h])
        rec.update({f"ade_rollout_k_h{h}": [round(float(x), 6) for x in a_h],
                    f"fde_rollout_k_h{h}": [round(float(x), 6) for x in f_h]})
    rec.update({"pred_xy_k": np.round(pred_k.astype(float), 3).tolist(), "prompt_len": prompt_len,
                "sec_gen": round(t_gen, 2), "sec_denoise": round(t_den, 2), "sec_total": round(time.time() - t0, 2)})
    return rec, pred_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--set", required=True, choices=list(SETS))
    ap.add_argument("--masks", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gate", type=int, default=0)
    ap.add_argument("--plan", default="iclr27-rebuttal/plans/2026-11-14_a2-edge-moop-u40.md")
    args = ap.parse_args()
    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, cache_root = SETS[args.set]
    man = [{"clip_id": r.clip_id, "t0_us": int(r.t0_us)} for r in pd.read_parquet(manifest).itertuples()]
    man = man[args.shard :: args.n_shards]
    if args.limit:
        man = man[: args.limit]

    eel.set_determinism()
    from alpamayo2_edge import helper

    model, _ = ec.load_model("cuda")
    model.eval()
    processor = helper.get_processor(model.tokenizer, model.config)
    gpu = torch.cuda.get_device_name(0)
    masks = None
    if args.masks:
        z = np.load(args.masks)
        masks = (z["q_mask"], z["mlp_mask"])

    if args.gate:
        assert masks is not None, "--gate needs --masks"
        res = {}
        for mode in ("none", "ones", "masks"):
            hooks = None
            if mode == "ones":
                hooks = eel.EdgeVlmMasks(model, np.ones_like(masks[0]), np.ones_like(masks[1]))
            elif mode == "masks":
                hooks = eel.EdgeVlmMasks(model, *masks)
            res[mode] = [eval_clip(model, processor, m, cache_root, args) for m in man[: args.gate]]
            if hooks:
                hooks.remove()
            print(f"gate {mode}: " + " | ".join(f"{r['clip_id'][:8]} ade@8 {r['minADE_rollout']:.4f} "
                                                f"{r['gen_coc'][:40]!r}" for r, _ in res[mode]), flush=True)
        ident = all(np.array_equal(a[1], b[1]) and a[0]["gen_coc"] == b[0]["gen_coc"]
                    for a, b in zip(res["none"], res["ones"]))
        changed = any(not np.array_equal(a[1], b[1]) for a, b in zip(res["none"], res["masks"]))
        (out_dir / "gate.json").write_text(json.dumps({"ones_equal_none": ident, "masks_change_output": changed,
                                                       "records": {k: [r for r, _ in v] for k, v in res.items()}},
                                                      indent=2))
        print(f"gate ones==none: {ident} | masks change: {changed}", flush=True)
        sys.exit(0 if ident and changed else 1)

    hooks = eel.EdgeVlmMasks(model, *masks) if masks is not None else None
    rows_path = out_dir / f"edge_s{args.shard}of{args.n_shards}.json"
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    done = {r["clip_id"] for r in rows}
    (out_dir / f"config_s{args.shard}of{args.n_shards}.json").write_text(json.dumps({
        "model": ec.MODEL_ID, "model_revision": ec.MODEL_REV, "set": args.set, "manifest": str(manifest),
        "cache": str(cache_root), "n_clips": len(man), "k": args.k, "seed": args.seed,
        "seed_rule": "sha256(f'{seed}:{clip_id}')[:4], +k per sample", "max_gen": args.max_gen,
        "sampling": {"do_sample": True, "temperature": 0.6, "top_p": 0.98}, "diffusion": {"inference_step": 10},
        "prompt": "Edge driving prompt without the navigation line", "masks": args.masks,
        "mask_placement": "forward pre-hooks: o_proj input per Q head, mlp.fc2 input per MLP channel",
        "gpu": os.environ["CUDA_VISIBLE_DEVICES"], "gpu_name": gpu, "plan": args.plan,
        "deterministic": {"use_deterministic_algorithms": True, "warn_only": True, "tf32": False}}, indent=2))
    t_start = time.time()
    for i, m in enumerate(man):
        if m["clip_id"] in done:
            continue
        rec, _ = eval_clip(model, processor, m, cache_root, args)
        rows.append(rec)
        if len(rows) % 20 == 0:
            rows_path.write_text(json.dumps(rows))
        print(f"[{i + 1}/{len(man)}] {m['clip_id'][:8]} {rec['bucket']:10s} ade@8={rec['minADE_rollout']:.3f} "
              f"ade@6={min(rec['ade_rollout_k'][:6]):.3f} len={rec['gen_len']} eos={rec['has_eos']} "
              f"{rec['sec_total']:.1f}s | {rec['gen_coc'][:70]!r}", flush=True)
    rows_path.write_text(json.dumps(rows))
    if hooks:
        hooks.remove()
    a = np.array([min(r["ade_rollout_k"][:6]) for r in rows])
    line = (f"{args.exp_id} shard {args.shard}/{args.n_shards}: {len(rows)} clips, minADE@6 mean {a.mean():.4f} median "
            f"{np.median(a):.4f}, nominal {np.mean([r['nominal'] for r in rows]):.2f}, no-anchor "
            f"{sum(not r['has_eos'] for r in rows)}, {(time.time() - t_start) / 60:.1f} min on {gpu}")
    (out_dir / f"summary_s{args.shard}of{args.n_shards}.txt").write_text(line + "\n")
    print(line, flush=True)


if __name__ == "__main__":
    main()
