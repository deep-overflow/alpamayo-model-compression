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
import hashlib
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


def blackout_slot(seed, clip_id, n_cam):
    """Camera slot whose t0 frame is blacked out, from the clip id (not the sampling seed)."""
    h = hashlib.sha256(f"blackout:{seed}:{clip_id}".encode()).digest()
    return int.from_bytes(h[:4], "big") % n_cam


def gt_coc_seq(tok, prompt_ids, gt_text, device):
    """[prompt] + tokenize(reasoning) + cot_end + traj_future_start -> (1, L)."""
    rt = tok(gt_text, add_special_tokens=False)["input_ids"]
    coc = torch.tensor([rt + [COT_END, TFS]], device=device)
    return torch.cat([prompt_ids, coc], dim=1)


@torch.no_grad()
def eval_from_generation_cache(model, inputs, roll, gt_xy, seeds):
    """K denoisings on the rollout's own KV cache (prompt + generated CoC), as the official Alpamayo / AlpaSim
    inference does, instead of eval_config_samples' teacher-forced re-prefill. Returns (ade_k, fde_k, pred_k)."""
    cache = roll["past_key_values"]
    prefill = cache.get_seq_length()
    dev = inputs["input_ids"].device
    offset = torch.tensor([prefill], device=dev)
    prefix_mask = torch.ones(1, prefill, device=dev, dtype=torch.long)
    preds = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for s in seeds:
            action = lib.denoise_with_cache(model, cache, roll["rope_deltas"], offset, prefix_mask,
                                            seed=s)  # cropped back to prefill each call
            pred_xyz, _ = model.action_space.action_to_traj(
                action.float(), inputs["ego_history_xyz"][:, -1].float(),
                inputs["ego_history_rot"][:, -1].float(),
            )
            preds.append(pred_xyz[0, :, :2].cpu().numpy())
    pred_k = np.stack(preds)  # (K, T, 2)
    ade, fde = el.ade_fde(pred_k, gt_xy)
    return ade, fde, pred_k


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
    ap.add_argument("--save-pred", action="store_true",
                    help="keep the k sampled paths themselves, not just their ADE/FDE. "
                         "Needed by anything scoring the trajectory against something "
                         "other than the GT path -- the open-loop collision proxy reads "
                         "them. 8x64x2 at mm precision is ~7 KB/clip, ~3.5 MB per set.")
    ap.add_argument("--blackout", action="store_true",
                    help="replace the t0 frame of one camera per clip with a black image; "
                         "the camera is drawn from the clip id, so sampling seeds are unchanged")
    ap.add_argument("--blackout-seed", type=int, default=0)
    ap.add_argument("--blackout-cam", type=int, default=None, choices=[0, 1, 2, 6],
                    help="black out the t0 frame of this camera index on every clip "
                         "(0 cross-left, 1 front-wide, 2 cross-right, 6 front-tele)")
    ap.add_argument("--lag-ms", type=float, default=0.0,
                    help="feed the --timing-cams cameras' frames from this far in the past: "
                         "they show t0-lag-0.3 .. t0-lag while the ego history still ends at t0")
    ap.add_argument("--stride-ms", type=float, default=100.0,
                    help="spacing of the --timing-cams cameras' 4 frames (100 = as trained)")
    ap.add_argument("--timing-cams", type=int, nargs="+", default=None, choices=[0, 1, 2, 6],
                    help="cameras --lag-ms/--stride-ms apply to; default all four")
    ap.add_argument("--timing-cache", default="test_timing",
                    help="build_timing_cache.py cache the shifted frames are picked from")
    ap.add_argument("--jitter-ms", type=float, default=0.0,
                    help="independent capture-time jitter per frame, dt ~ U(-J, +J) ms on all "
                         "16 slots, drawn from the clip id (sample_cache.jitter_requests)")
    ap.add_argument("--jitter-seed", type=int, default=0)
    ap.add_argument("--jitter-future", default="black", choices=["black", "keep", "drop"],
                    help="t0 frame whose dt > 0 (a future frame): black it out, keep the "
                         "unjittered t0 frame (separates the jitter from the blackout), or "
                         "drop the image and its 'frame 3' label from the prompt")
    ap.add_argument("--jitter-no-shift", action="store_true",
                    help="same draws, frames left on their nominal slots: only the t0 "
                         "frames flagged as future are handled (blackout part alone)")
    ap.add_argument("--t0-exclude", default=None, choices=["black", "drop"],
                    help="exclude every camera's t0 frame: black it out, or drop the image and "
                         "its 'frame 3' label from the prompt (t0-0.3 .. t0-0.1 unchanged)")
    args = ap.parse_args()
    if args.blackout and args.blackout_cam is not None:
        ap.error("--blackout and --blackout-cam are exclusive")
    blackout = args.blackout or args.blackout_cam is not None
    timing = args.lag_ms != 0 or args.stride_ms != 100
    timing_cams = sorted(args.timing_cams or [0, 1, 2, 6])
    jitter = args.jitter_ms > 0
    if jitter and timing:
        ap.error("--jitter-ms and --lag-ms/--stride-ms are exclusive")
    if jitter and args.jitter_ms > 50:
        ap.error("--jitter-ms above 50 lets adjacent slots (100 ms apart) swap order")
    if args.t0_exclude and (jitter or timing or blackout):
        ap.error("--t0-exclude is a standalone intervention")

    tag = "baseline" if args.model == "baseline" else Path(args.model).name
    if args.quant:
        # the rows file has to name the intervention, not just the checkpoint it sits on
        qs = Path(args.quant).stem
        tag = qs if tag == "baseline" else f"{tag}_{qs}"
    if args.blackout:
        tag = f"{tag}_bk1"
    if args.blackout_cam is not None:
        tag = f"{tag}_bkcam{args.blackout_cam}"
    if timing:
        cams = "all" if timing_cams == [0, 1, 2, 6] else "".join(map(str, timing_cams))
        tag = f"{tag}_tm{cams}_lag{args.lag_ms:g}_st{args.stride_ms:g}"
    if jitter:
        tag = (f"{tag}_jit{args.jitter_ms:g}_{args.jitter_future}"
               f"{'_noshift' if args.jitter_no_shift else ''}"
               f"{f'_js{args.jitter_seed}' if args.jitter_seed else ''}")
    if args.t0_exclude:
        tag = f"{tag}_t0x_{args.t0_exclude}"
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
        "blackout": ({"frame": "t0", "n_cams": 1, "camera": args.blackout_cam,
                      "rule": "fixed camera index"} if args.blackout_cam is not None else
                     {"frame": "t0", "n_cams": 1, "seed": args.blackout_seed,
                      "rule": "sha256(f'blackout:{seed}:{clip_id}')[:4] % n_cam"}
                     if args.blackout else None),
        "timing": ({"cams": timing_cams, "lag_ms": args.lag_ms, "stride_ms": args.stride_ms,
                    "cache": args.timing_cache,
                    "rule": "closest frame at or before t0 - lag - (3-i)*stride, i=0..3; "
                            "other cameras t0 - (3-i)*100 ms"} if timing else None),
        "jitter": ({"jitter_ms": args.jitter_ms, "seed": args.jitter_seed,
                    "future": args.jitter_future, "shift": not args.jitter_no_shift,
                    "cache": args.timing_cache,
                    "rule": "dt ~ U(-J, J) per slot from sha256(f'jitter:{seed}:{clip_id}'); "
                            "closest frame at or before t0 - (3-i)*100 ms + dt; t0 slot with "
                            "dt > 0 -> `future`"} if jitter else None),
        "t0_exclude": args.t0_exclude,
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
        if timing:
            req = []
            for ci in data["camera_indices"].tolist():
                lag, stride = (args.lag_ms, args.stride_ms) if ci in timing_cams else (0, 100)
                req.append(m["t0_us"] - np.round((lag + (3 - np.arange(4)) * stride) * 1000))
            # image_frames: (n_cam, 4, 3, H, W) uint8, picked: (n_cam, 4) us
            data["image_frames"], picked = sc.pick_timed_frames(
                sc.path_for(args.timing_cache, m["clip_id"], m["t0_us"]),
                data["camera_indices"].tolist(), np.stack(req).astype(np.int64))
        if jitter:
            req, dt, future = sc.jitter_requests(
                m["clip_id"], m["t0_us"], len(data["camera_indices"]), args.jitter_ms,
                args.jitter_seed, shift=not args.jitter_no_shift)  # (n_cam, 4) us, ms; (n_cam,)
            data["image_frames"], picked = sc.pick_timed_frames(
                sc.path_for(args.timing_cache, m["clip_id"], m["t0_us"]),
                data["camera_indices"].tolist(), req)
            if args.jitter_future == "black":
                data["image_frames"][torch.from_numpy(future), -1] = 0  # (n_future, 3, H, W)
        drop = ([c * 4 + 3 for c in np.flatnonzero(future)]
                if jitter and args.jitter_future == "drop" else [])
        if args.t0_exclude == "black":
            data["image_frames"][:, -1] = 0                         # (n_cam, 3, H, W)
        elif args.t0_exclude == "drop":
            drop = [c * 4 + 3 for c in range(len(data["camera_indices"]))]
        if blackout:
            # image_frames: (n_cam, n_frame, 3, H, W) uint8, frame -1 = t0
            if args.blackout_cam is None:
                slot = blackout_slot(args.blackout_seed, m["clip_id"], data["image_frames"].shape[0])
            else:
                slot = int((data["camera_indices"] == args.blackout_cam).nonzero()[0, 0])
            data["image_frames"][slot, -1] = 0                      # (3, H, W)
        inputs = lib.build_inputs(model, processor, data, "cuda", drop_images=drop)
        prompt_len = inputs["input_ids"].shape[1]
        gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()  # (64, 2)

        base = clip_seed(args.seed, m["clip_id"])
        traj_seeds = [base + k for k in range(args.k)]

        rec = {"clip_id": m["clip_id"], "bucket": el.bucket(gt_xy), "seed": base}
        if blackout:
            rec["blackout_cam"] = int(data["camera_indices"][slot])
        if timing or jitter:
            rec["frame_ts_ms"] = np.round((picked - m["t0_us"]) / 1e3, 1).tolist()
        if jitter:
            rec["jitter_dt_ms"] = np.round(dt, 2).tolist()
            rec["future_t0_cams"] = [int(c) for c, f in zip(data["camera_indices"], future) if f]
            rec["dup_frames"] = int((np.diff(picked, axis=1) == 0).sum())
            rec["n_images"] = 16 - len(drop)
        if args.t0_exclude:
            rec["n_images"] = 16 - len(drop)

        # own-rollout condition: the model writes its own reasoning, then drives on it
        seed_all(base)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
        coc_end = roll["eos_pos"] + 1
        seq_gen = roll["sequences"][:, :coc_end].clone()
        gen_ids = roll["sequences"][0, prompt_len:coc_end].tolist()
        gen_coc = tok.decode([t for t in gen_ids if t not in (COT_END, TFS)],
                             skip_special_tokens=True)
        # keep the per-sample arrays, not just their min: seeds are base+k, so minADE@K'
        # for any K' <= k is a prefix of these and needs no second run
        try:
            ade_k, fde_k, nll_self, pred_k = eval_config_samples(
                model, inputs, seq_gen, prompt_len, coc_end, gt_xy, traj_seeds)
        except ValueError as e:
            # A degenerate CoC can contain a vision placeholder token (<|image_pad|>); the teacher-forced
            # re-prefill then holds one more image token than image features and Qwen3-VL raises. Only for
            # that clip, drive on the generation's own KV cache -- the official Alpamayo / AlpaSim inference
            # path -- where the CoC NLL is undefined (iclr27-rebuttal/plans/2026-11-12_u50-openloop-blackwell.md).
            if "Image features and image tokens do not match" not in str(e):
                raise
            ade_k, fde_k, pred_k = eval_from_generation_cache(model, inputs, roll, gt_xy, traj_seeds)
            nll_self = None
            rec["tf_fallback"] = "generation_cache"
            print(f"  {m['clip_id'][:8]}: vision token in the CoC -> generation-cache fallback", flush=True)
        del roll
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
        if args.save_pred:
            # the paths in the same t0 rig frame as gt_xy, mm precision -- enough for a
            # footprint overlap test and a third of the bytes of the default repr
            rec["pred_xy_k"] = np.round(np.asarray(pred_k, dtype=float), 3).tolist()

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
