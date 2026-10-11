"""CoC-level vs action-level multimodality of the unpruned model on a calibration set.

Every open-loop number in this repo (and `run_importance`) samples ONE CoC per clip and
then denoises K trajectories under it, so the spread of those K paths is the action
distribution *given one reasoning sentence*. This runner crosses the two sources:

  n_coc CoC rollouts per clip  x  k denoising seeds per CoC

and stores every path, so the spread can be split into within-CoC (same sentence,
different FM noise) and between-CoC (same FM noise, different sentence).

Design choices that the analysis relies on:
  - the denoising seeds are `base + k`, the SAME k seeds under every CoC (common random
    numbers), so a between-CoC difference at fixed k is caused by the CoC alone;
  - CoC 0 is sampled under `seed_all(base)`, exactly as `run_baseline.py` does, so
    row[coc 0] must reproduce a `run_baseline.py --k K` run bitwise on one architecture
    (G0, checked by `analyze_coc_multimode.py --g0`);
  - `nll` is the mean per-token CE of the sampled CoC under the model at temperature 1
    (teacher-forced, same quantity as `nll_self`): how unlikely the model finds what it
    just wrote. `nll_sum = nll * n_tok` is the sequence-level surprisal.

Usage:
  bash experiments/head_analysis/run_retry_host.sh 5 experiments/evaluation/run_coc_multimode.py \
      --manifest calib_nt100_a --cache calib_nt --shard 0 --n-shards 2 --gpu 4
"""

import os

# must precede any CUDA context creation for deterministic cuBLAS reductions
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
sys.path.insert(0, str(Path(__file__).parent))

import analysis_lib as lib
import eval_lib as el
import sample_cache as sc
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from expert_per_clip import reserve_gpu
from run_baseline import COT_END, MODEL_REV, TFS, seed_all, set_determinism
from run_eval import eval_config_samples
from sample_cache import clip_seed

# stride between CoC rollout seeds; far larger than k so no rollout seed equals a
# denoising seed of the same clip
COC_STRIDE = 100003


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="manifest stem under outputs/eval_sets")
    ap.add_argument("--cache", required=True, help="pre_processed cache name")
    ap.add_argument("--exp-id", default=None)
    ap.add_argument("--n-coc", type=int, default=6, help="CoC rollouts per clip")
    ap.add_argument("--k", type=int, default=6, help="trajectory samples per CoC")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    exp_id = args.exp_id or f"cocmm_{args.manifest}"
    out_dir = REPO / "outputs" / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"rows_s{args.shard}of{args.n_shards}.json"

    df = pd.read_parquet(REPO / "outputs" / "eval_sets" / f"{args.manifest}.parquet")
    man = [{"clip_id": r.clip_id, "t0_us": int(r.t0_us)} for r in df.itertuples()]
    man = man[args.shard::args.n_shards]
    if args.limit:
        man = man[: args.limit]

    set_determinism(warn_only=True)
    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    gpu_name = torch.cuda.get_device_name(device)
    print(f"{exp_id} shard {args.shard}/{args.n_shards} | {len(man)} clips | "
          f"{args.n_coc} CoC x {args.k} traj | {gpu_name}", flush=True)

    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    tok = model.tokenizer
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")

    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    done = {r["clip_id"] for r in rows}
    (out_dir / f"config_s{args.shard}of{args.n_shards}.json").write_text(json.dumps({
        "model": "baseline", "manifest": args.manifest, "cache": args.cache,
        "n_clips": len(man), "n_coc": args.n_coc, "k": args.k, "seed": args.seed,
        "coc_seed_rule": f"seed_all(clip_seed + j * {COC_STRIDE})",
        "traj_seed_rule": "clip_seed + k, shared across CoCs",
        "temperature": 0.6, "top_p": 0.98, "max_gen": args.max_gen,
        "shard": [args.shard, args.n_shards], "gpu": gpu_name, "model_revision": MODEL_REV,
    }, indent=2))

    t_start = time.time()
    for i, m in enumerate(man):
        if m["clip_id"] in done:
            continue
        t0 = time.time()
        data = sc.load_cached(sc.path_for(args.cache, m["clip_id"], m["t0_us"]))
        inputs = lib.build_inputs(model, processor, data, "cuda")
        prompt_len = inputs["input_ids"].shape[1]
        gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()  # (64, 2)

        base = clip_seed(args.seed, m["clip_id"])
        traj_seeds = [base + k for k in range(args.k)]
        rec = {"clip_id": m["clip_id"], "bucket": el.bucket(gt_xy), "seed": base,
               "gt_xy": np.round(gt_xy.astype(float), 3).tolist(), "cocs": []}

        for j in range(args.n_coc):
            seed_all(base + j * COC_STRIDE)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
            coc_end = roll["eos_pos"] + 1
            seq_gen = roll["sequences"][:, :coc_end].clone()
            gen_ids = roll["sequences"][0, prompt_len:coc_end].tolist()
            gen_coc = tok.decode([t for t in gen_ids if t not in (COT_END, TFS)],
                                 skip_special_tokens=True)
            del roll
            ade_k, fde_k, nll, pred_k = eval_config_samples(
                model, inputs, seq_gen, prompt_len, coc_end, gt_xy, traj_seeds)  # pred_k (K, 64, 2)
            rec["cocs"].append({
                "text": gen_coc, "n_tok": len(gen_ids), "nll": nll, "nll_sum": nll * len(gen_ids),
                "degenerate": el.coc_degenerate(gen_coc)["degenerate"],
                "ade_k": [round(float(x), 6) for x in ade_k],
                "fde_k": [round(float(x), 6) for x in fde_k],
                "pred_xy_k": np.round(np.asarray(pred_k, dtype=float), 3).tolist(),
            })

        rows.append(rec)
        if len(rows) % 5 == 0 or i + 1 == len(man):
            rows_path.write_text(json.dumps(rows))
        a = np.array([c["ade_k"] for c in rec["cocs"]])  # (n_coc, K)
        n_txt = len({c["text"] for c in rec["cocs"]})
        print(f"[{i + 1}/{len(man)}] {m['clip_id'][:8]} {rec['bucket']:10s} "
              f"minADE36={a.min():.3f} distinct_coc={n_txt} "
              f"nll={np.mean([c['nll'] for c in rec['cocs']]):.3f} ({time.time() - t0:.0f}s)",
              flush=True)

    rows_path.write_text(json.dumps(rows))
    print(f"\n{len(rows)} clips, {(time.time() - t_start) / 60:.1f} min -> {rows_path}", flush=True)


if __name__ == "__main__":
    main()
