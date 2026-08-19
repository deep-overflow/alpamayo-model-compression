"""Open-loop evaluation of one model on the in-distribution or the OOD set.

Reads clips from the per-clip cache (`sample_cache`), so no camera chunk zip is
touched and a clip loads in ~0.4 s instead of ~40 s.

What is measured differs between the two sets, because only OOD carries a reference
reasoning text:

  in-distribution  the model rolls out its own CoC, and the trajectory is scored under
                   that context. `nll` is then the model's NLL of its own sample -- a
                   reference number, not a quality measure.
  OOD              two conditions. (1) the curated GT CoC is teacher-forced as context,
                   giving minADE_tf and nll_gtcoc = "can the model predict the correct
                   reasoning". (2) the model's own rollout, giving minADE_rollout, the
                   end-to-end number. The generated text is stored either way, for
                   qualitative comparison and for a later judge over (gt_coc, gen_coc).

Seeds are derived from the clip id, never from the loop index, so sharding, ordering
and restarts cannot change a clip's result. `run_eval.py` derives them from the loop
index, which means a `--clip-offset` shard and a whole run disagree; that is the defect
this script exists to avoid repeating.

Determinism flags are set here, but reproducibility is claimed only after measuring it:
run the same shard twice and diff (see --limit). Bitwise agreement holds within one GPU
architecture; different architectures use different kernels.

Usage:
  python experiments/evaluation/run_baseline.py --set indist --gpu 4
  python experiments/evaluation/run_baseline.py --set ood --shard 0 --n-shards 4 --gpu 4
"""

import os

# must precede any CUDA context creation for deterministic cuBLAS reductions
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import random
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
from run_eval import eval_config_samples
from sample_cache import clip_seed

COT_END, TFS = 155678, 155681
# pinned so a repo update cannot silently change the weights under a run; this commit's
# blobs are the 2026-06-23 ones every result in this track was produced with
MODEL_REV = "7aba8293c09993f2e125c6819df05d7fa3e873ea"


def set_determinism(warn_only):
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gt_coc_seq(tok, prompt_ids, gt_text, device):
    """[prompt] + tokenize(reasoning) + cot_end + traj_future_start -> (1, L)."""
    rt = tok(gt_text, add_special_tokens=False)["input_ids"]
    coc = torch.tensor([rt + [COT_END, TFS]], device=device)
    return torch.cat([prompt_ids, coc], dim=1)


def load_manifest(which, sets_dir, n_indist, manifest=None, cache=None):
    """`indist` is the official-val draw, `test` the official-test draw, `ood` the OOD cache.

    `manifest`/`cache` override the pair, which is how the JPEG-quality probe re-evaluates
    the same val clips out of a differently encoded cache.
    """
    if which in ("indist", "test"):
        stem = manifest or (f"indist_{n_indist}" if which == "indist" else f"test_{n_indist}")
        cache = cache or ("eval" if which == "indist" else "test")
        df = pd.read_parquet(sets_dir / f"{stem}.parquet")
        return [{"clip_id": r.clip_id, "t0_us": int(r.t0_us), "cache": cache}
                for r in df.itertuples()]
    # `manifest` also reaches the OOD branch: the set carries a train/val split, and an
    # arm calibrated on OOD-train may only be judged on OOD-val. Defaults to the whole
    # 1,533 so every existing run is unchanged.
    df = pd.read_parquet(sets_dir / f"{manifest or 'ood'}.parquet")
    return [{"clip_id": r.clip_id, "t0_us": int(r.t0_us), "cache": cache or "ood",
             "gt_coc": r.gt_coc, "cluster": r.cluster, "split": r.split}
            for r in df.itertuples()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="which", required=True, choices=["indist", "test", "ood"])
    ap.add_argument("--model", default="baseline", help="'baseline' or a slim ckpt dir")
    ap.add_argument("--exp-id", default=None)
    ap.add_argument("--sets-id", default="eval_sets")
    ap.add_argument("--n-indist", type=int, default=500)
    ap.add_argument("--k", type=int, default=8, help="trajectory samples per clip")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="first N clips of the shard")
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--manifest", default=None, help="override the manifest stem")
    ap.add_argument("--cache", default=None, help="override the pre_processed cache name")
    ap.add_argument("--quant", default=None,
                    help="path to a per-row bit spec .npz (alloc_lib); VLM pool only")
    ap.add_argument("--quant-group", type=int, default=64)
    ap.add_argument("--quant-parts", nargs="+",
                    default=["vlm_text", "lm_head", "vit"],
                    help="pool parts the spec is allowed to cover. The default excludes "
                         "the expert because a CoC-guided allocation cannot score it; a "
                         "uniform budget can, so pass '... expert' for those specs.")
    ap.add_argument("--strict-deterministic", action="store_true",
                    help="error instead of warn when an op has no deterministic kernel")
    ap.add_argument("--no-tf", action="store_true",
                    help="ood only: skip the GT-CoC teacher-forced condition, roughly "
                         "halving per-clip cost when only rollout metrics are needed")
    args = ap.parse_args()

    tag = "baseline" if args.model == "baseline" else Path(args.model).name
    if args.quant:
        # the rows file has to name the intervention, not just the checkpoint it sits on
        qs = Path(args.quant).stem
        tag = qs if tag == "baseline" else f"{tag}_{qs}"
    exp_id = args.exp_id or f"baseline_{args.which}"
    out_dir = REPO / "outputs" / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"{tag}_s{args.shard}of{args.n_shards}.json"

    man = load_manifest(args.which, REPO / "outputs" / args.sets_id, args.n_indist,
                        args.manifest, args.cache)
    man = man[args.shard::args.n_shards]          # strided, so any shard count covers all
    if args.limit:
        man = man[: args.limit]

    set_determinism(warn_only=not args.strict_deterministic)
    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    gpu_name = torch.cuda.get_device_name(device)
    print(f"{exp_id} {tag} | set={args.which} shard {args.shard}/{args.n_shards} "
          f"| {len(man)} clips | {gpu_name}", flush=True)

    if args.model == "baseline":
        model = Alpamayo1_5.from_pretrained(
            "nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, dtype=torch.bfloat16).to("cuda")
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    else:
        import slim_lib as sl
        model = sl.load_slim(REPO / args.model, device="cuda")

    quant_meta = None
    if args.quant:
        import quant_lib as ql
        z = np.load(args.quant, allow_pickle=True)
        names = [str(n) for n in z["_names"]]
        spec = {n: torch.from_numpy(z[n]) for n in names}
        parts = tuple(args.quant_parts)
        pool = ql.pool_shapes(model, parts=parts)
        unknown = [n for n in names if n not in pool]
        if unknown:
            raise KeyError(f"spec has {len(unknown)} names outside pool {parts}: {unknown[:3]}")
        # a spec is a per-row bit vector, so it only fits the shapes it was written for;
        # pruning changes out_features and apply_quant checks names but not lengths
        bad = [(n, len(spec[n]), pool[n][0]) for n in names if len(spec[n]) != pool[n][0]]
        if bad:
            raise ValueError(
                f"spec row counts do not match this model ({len(bad)} tensors, e.g. {bad[:2]}). "
                "Regenerate it against the model it will be applied to "
                "(make_uniform_spec.py --model <ckpt>).")
        eff, tot_bits, tot_par = ql.effective_bits({n: pool[n] for n in names}, spec,
                                                   args.quant_group)
        report = ql.apply_quant(model, spec, group=args.quant_group, parts=parts)
        hist = {b: sum(int((spec[n] == b).sum()) for n in names) for b in ql.BITS}
        # embed_tokens is never quantizable, and the expert only when asked for: a
        # CoC-guided spec that silently reached it would invalidate that comparison
        touched = set(report)
        assert not any("embed_tokens" in n for n in touched)
        if "expert" not in parts:
            assert not any(n.startswith("expert") for n in touched)
        quant_meta = {
            "spec": str(args.quant), "group": args.quant_group, "parts": list(parts),
            "pool_tensors": len(names), "pool_params": tot_par,
            "effective_bits": eff, "projected_pool_gb": tot_bits / 8 / 1e9,
            "pool_bf16_gb": tot_par * 2 / 1e9, "row_bit_hist": hist,
            "worst_rel_mse": max(r["rel_mse"] for r in report.values()),
            "mean_rel_mse": float(np.mean([r["rel_mse"] for r in report.values()])),
        }
        print(f"quant {Path(args.quant).stem}: {len(names)} tensors, "
              f"{eff:.3f} eff bits, pool {tot_par * 2 / 1e9:.2f} -> "
              f"{tot_bits / 8 / 1e9:.2f} GB (projected), rows {hist}", flush=True)

    processor = helper.get_processor(model.tokenizer)
    tok = model.tokenizer
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")

    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    done = {r["clip_id"] for r in rows}
    (out_dir / "config.json").write_text(json.dumps({
        "model": args.model, "set": args.which, "n_clips": len(man), "k": args.k,
        "seed": args.seed, "seed_rule": "sha256(f'{seed}:{clip_id}')[:4], +k per sample",
        "max_gen": args.max_gen, "shard": [args.shard, args.n_shards],
        "sets_id": args.sets_id, "gpu": gpu_name, "model_revision": MODEL_REV,
        "manifest": args.manifest, "cache": args.cache, "quant": quant_meta,
        "deterministic": {"use_deterministic_algorithms": True,
                          "warn_only": not args.strict_deterministic,
                          "cudnn_deterministic": True, "tf32": False,
                          "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"]},
    }, indent=2))

    t_start = time.time()
    for i, m in enumerate(man):
        if m["clip_id"] in done:
            continue
        t0 = time.time()
        data = sc.load_cached(sc.path_for(m["cache"], m["clip_id"], m["t0_us"]))
        inputs = lib.build_inputs(model, processor, data, "cuda")
        prompt_len = inputs["input_ids"].shape[1]
        gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()  # (64, 2)

        base = clip_seed(args.seed, m["clip_id"])
        traj_seeds = [base + k for k in range(args.k)]

        rec = {"clip_id": m["clip_id"], "bucket": el.bucket(gt_xy), "seed": base}

        # own-rollout condition: the model writes its own reasoning, then drives on it
        seed_all(base)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
        coc_end = roll["eos_pos"] + 1
        seq_gen = roll["sequences"][:, :coc_end].clone()
        gen_ids = roll["sequences"][0, prompt_len:coc_end].tolist()
        gen_coc = tok.decode([t for t in gen_ids if t not in (COT_END, TFS)],
                             skip_special_tokens=True)
        del roll
        # keep the per-sample arrays, not just their min: seeds are base+k, so minADE@K'
        # for any K' <= k is a prefix of these and needs no second run
        ade_k, fde_k, nll_self, pred_k = eval_config_samples(
            model, inputs, seq_gen, prompt_len, coc_end, gt_xy, traj_seeds)
        rec.update({"minADE_rollout": float(ade_k.min()), "minFDE_rollout": float(fde_k.min()),
                    "ade_rollout_k": [round(float(x), 6) for x in ade_k],
                    "fde_rollout_k": [round(float(x), 6) for x in fde_k],
                    "nll_self": nll_self, "gen_coc": gen_coc, "gen_len": len(gen_ids),
                    **{f"coc_{k}": v for k, v in el.coc_degenerate(gen_coc).items()}})
        # shorter horizons (1.6 s / 3.2 s at 10 Hz) from the same denoised paths --
        # the full-horizon scalars cannot be re-reduced, so store these per sample too
        for h in (16, 32):
            a_h, f_h = el.ade_fde(pred_k[:, :h], gt_xy[:h])
            rec.update({f"ade_rollout_k_h{h}": [round(float(x), 6) for x in a_h],
                        f"fde_rollout_k_h{h}": [round(float(x), 6) for x in f_h]})

        if args.which == "ood":
            rec.update({"gt_coc": m["gt_coc"], "cluster": m["cluster"], "split": m["split"]})
            if not args.no_tf:
                # GT-CoC condition: correct reasoning as context, and its NLL
                seq_gt = gt_coc_seq(tok, inputs["input_ids"], m["gt_coc"], "cuda")
                ade_tf_k, fde_tf_k, nll_gt, pred_tf_k = eval_config_samples(
                    model, inputs, seq_gt, prompt_len, seq_gt.shape[1], gt_xy, traj_seeds)
                rec.update({"minADE_tf": float(ade_tf_k.min()),
                            "minFDE_tf": float(fde_tf_k.min()),
                            "ade_tf_k": [round(float(x), 6) for x in ade_tf_k],
                            "fde_tf_k": [round(float(x), 6) for x in fde_tf_k],
                            "nll_gtcoc": nll_gt})
                for h in (16, 32):
                    a_h, f_h = el.ade_fde(pred_tf_k[:, :h], gt_xy[:h])
                    rec.update({f"ade_tf_k_h{h}": [round(float(x), 6) for x in a_h],
                                f"fde_tf_k_h{h}": [round(float(x), 6) for x in f_h]})

        rows.append(rec)
        if len(rows) % 10 == 0 or i + 1 == len(man):
            rows_path.write_text(json.dumps(rows, indent=2))
        extra = f" ade_tf={rec['minADE_tf']:.3f}" if "minADE_tf" in rec else ""
        print(f"[{i + 1}/{len(man)}] {m['clip_id'][:8]} {rec['bucket']:10s} "
              f"ade_ro={rec['minADE_rollout']:.3f}{extra} ({time.time() - t0:.0f}s)", flush=True)

    rows_path.write_text(json.dumps(rows, indent=2))
    a = np.array([r["minADE_rollout"] for r in rows])
    line = (f"{tag} {args.which} shard {args.shard}/{args.n_shards}: {len(rows)} clips, "
            f"minADE(rollout) mean {a.mean():.4f} median {np.median(a):.4f}")
    if args.which == "ood" and rows and "minADE_tf" in rows[0]:
        t = np.array([r["minADE_tf"] for r in rows])
        g = np.array([r["nll_gtcoc"] for r in rows])
        line += f", minADE(GT CoC) mean {t.mean():.4f}, NLL(GT CoC) mean {g.mean():.4f}"
    line += f", CoC degen {np.mean([r['coc_degenerate'] for r in rows]):.3f}"
    (out_dir / f"summary_s{args.shard}of{args.n_shards}.txt").write_text(line + "\n")
    print(f"\n{line}\n{(time.time() - t_start) / 60:.1f} min -> {rows_path}", flush=True)


if __name__ == "__main__":
    main()
