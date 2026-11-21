"""Alpamayo 2 Super under camera capture-time perturbations, open loop on test500.

Plan: plans/2026-10-26_camera-timing-openloop.md ("Alpamayo 2"). The estimator, seeds, layer
placement and generation are alpamayo2-distill's `experiments/oodval/run_oodval.py`, imported
rather than copied: 1 CoC rollout -> K=8 denoisings on its KV cache, seed = clip_seed + k, the
64 layer pairs split over two cards. What changes is where the 24 frames (6 cameras x 4) come
from:

  frames   build_timing_cache.py windows -- test_timing (cameras 0, 1, 2, 6) and
           test_timing_rear (3, 5) -- picked with the release loader's rule (closest frame at or
           before the requested time); ego from the 4-camera test cache, which equals the
           Alpamayo 2 loader's (`build_cache_a2.py --verify`)
  sync     t0 - (3-i)*100 ms, i = 0..3
  jitter   dt ~ U(-50, 50) ms per slot; cameras 0, 1, 2, 6 reuse Alpamayo 1.5's draws
           (`sample_cache.jitter_requests`), 3 and 5 come from a separate "rear" stream.
           A t0 slot with dt > 0 is a future frame: blacked out (--arm black) or left out of
           the prompt together with its "frame 3" label (--arm drop).

Usage (one replica = two cards; the launcher sets the HF env):
  CUDA_VISIBLE_DEVICES=0,1 python experiments/evaluation/run_a2_timing.py --arm sync --shard 0 --n-shards 2
"""

import os

# the cards come from the shell: the imports below initialise CUDA (see run_oodval.py)
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    raise SystemExit("set CUDA_VISIBLE_DEVICES to this replica's cards, e.g. 0,1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, "/home/cvlab21/project/chan/alpamayo2-distill/experiments/oodval")
import run_oodval as ro
from alpamayo2_super import helper

# after run_oodval, which puts the main checkout's head_analysis on the path: this worktree's
# modules go first (sample_cache: pick_timed_frames, jitter_requests)
sys.path.insert(0, str(HERE.parent / "head_analysis"))
sys.path.insert(0, str(HERE))
import eval_lib as el
import sample_cache as sc

CAMS = [0, 1, 2, 3, 5, 6]   # Alpamayo 2 trajectory profile, ascending as the prompt requires
FRONT4 = [0, 1, 2, 6]       # the cameras Alpamayo 1.5 reads; same jitter draws as its runs
REAR = [3, 5]
EGO = ("ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot")


def clip_inputs(clip_id, t0_us, arm, jitter_ms, seed):
    """Returns data (image_frames (6, 4, 3, H, W) uint8 + ego), picked (6, 4) us,
    dt (6, 4) ms, future (6,) bool."""
    base = sc.load_cached(sc.path_for("test", clip_id, t0_us))
    nominal = t0_us - (3 - np.arange(4)) * 100_000                      # (4,) us
    req = {ci: nominal.astype(np.int64) for ci in CAMS}
    dt = {ci: np.zeros(4) for ci in CAMS}
    fut = {ci: False for ci in CAMS}
    if arm != "sync":
        for cams, stream in ((FRONT4, ""), (REAR, "rear")):
            r, d, f = sc.jitter_requests(clip_id, t0_us, len(cams), jitter_ms, seed,
                                         stream=stream)                 # (n, 4), (n, 4), (n,)
            for k, ci in enumerate(cams):
                req[ci], dt[ci], fut[ci] = r[k], d[k], bool(f[k])
    frames, picked = {}, {}
    for cache, cams in (("test_timing", FRONT4), ("test_timing_rear", REAR)):
        fr, pk = sc.pick_timed_frames(sc.path_for(cache, clip_id, t0_us), cams,
                                      np.stack([req[ci] for ci in cams]))  # (n, 4, 3, H, W)
        for k, ci in enumerate(cams):
            frames[ci], picked[ci] = fr[k], pk[k]
    image_frames = torch.stack([frames[ci] for ci in CAMS])             # (6, 4, 3, H, W)
    future = np.array([fut[ci] for ci in CAMS])                         # (6,)
    if arm == "black":
        image_frames[torch.from_numpy(future), -1] = 0                  # (n_future, 3, H, W)
    data = {"image_frames": image_frames, "camera_indices": torch.tensor(CAMS),
            **{k: base[k] for k in EGO}}
    return data, np.stack([picked[ci] for ci in CAMS]), np.stack([dt[ci] for ci in CAMS]), future


def build_inputs(model, processor, data, drop=()):
    """`run_oodval.build_inputs`, except that the `drop` images (flat camera-major indices) and
    the "frame i" text right before each are left out of the messages and the image list."""
    messages = helper.create_messages(data, model.config)
    images = data["image_frames"].flatten(0, 1).float() / 255.0          # (24, 3, H, W)
    if len(drop):
        slots = [(m, p) for m, msg in enumerate(messages) if isinstance(msg["content"], list)
                 for p, it in enumerate(msg["content"]) if it.get("type") == "image"]  # (24,)
        cut = {}
        for j in drop:
            m, p = slots[j]
            assert messages[m]["content"][p - 1]["text"].startswith("frame "), \
                messages[m]["content"][p - 1]
            cut.setdefault(m, set()).update({p - 1, p})
        for m, ps in cut.items():
            messages[m]["content"] = [it for p, it in enumerate(messages[m]["content"])
                                      if p not in ps]
        images = images[[j for j in range(images.shape[0]) if j not in set(drop)]]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         add_vision_id=False, continue_final_message=False)
    tokenized = dict(processor(text=text, images=images, videos=None, padding=False,
                               return_tensors="pt", do_rescale=False))
    return helper.to_device({"tokenized_data": tokenized,
                             "ego_history_xyz": data["ego_history_xyz"],
                             "ego_history_rot": data["ego_history_rot"]}, "cuda:0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["sync", "black", "drop"])
    ap.add_argument("--manifest", default="test_500")
    ap.add_argument("--jitter-ms", type=float, default=50.0)
    ap.add_argument("--jitter-seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--first-share", type=float, default=0.47)
    args = ap.parse_args()

    name = "sync" if args.arm == "sync" else f"jit{args.jitter_ms:g}_{args.arm}"
    set_name = args.manifest.replace("_", "")
    exp_id = f"a2timing_{name}_{set_name}"
    tag = f"alpamayo2_{name}"
    gpus = os.environ["CUDA_VISIBLE_DEVICES"]
    n_gpus = torch.cuda.device_count()
    assert n_gpus == len(gpus.split(",")) >= 2, (n_gpus, gpus)
    out_dir = REPO / "outputs" / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"{tag}_s{args.shard}of{args.n_shards}.json"

    man = pd.read_parquet(REPO / "outputs" / "eval_sets" / f"{args.manifest}.parquet")
    man = [{"clip_id": r.clip_id, "t0_us": int(r.t0_us)} for r in man.itertuples()]
    man = man[args.shard::args.n_shards]          # strided, the same rule as run_baseline.py
    if args.limit:
        man = man[: args.limit]

    ro.set_determinism()
    share = args.first_share if n_gpus == 2 else 1.0 / n_gpus - 0.03
    device_map, per_gpu = ro.layer_device_map(n_gpus, share)
    gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
    print(f"{exp_id} {tag} | shard {args.shard}/{args.n_shards} | {len(man)} clips | "
          f"cards {gpus} ({gpu_names[0]}) | layers per card {per_gpu}", flush=True)
    t_load = time.time()
    model = ro.Alpamayo2Super.from_pretrained(ro.MODEL_ID, revision=ro.MODEL_REV,
                                              dtype=torch.bfloat16, device_map=device_map)
    model.eval()
    processor = helper.get_processor(model.tokenizer, model.config)
    print(f"model loaded in {time.time() - t_load:.0f}s", flush=True)

    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    done = {r["clip_id"] for r in rows}
    (out_dir / "config.json").write_text(json.dumps({
        "model": ro.MODEL_ID, "model_revision": ro.MODEL_REV, "arm": args.arm,
        "manifest": args.manifest, "n_clips_shard": len(man), "k": args.k, "seed": args.seed,
        "seed_rule": "sha256(f'{seed}:{clip_id}')[:4], +k per sample",
        "protocol": "alpamayo2-distill run_oodval.rollout_then_k_traj (matched to run_baseline.py)",
        "input_profile": "trajectory: cameras [0,1,2,3,5,6] x 4 frames",
        "frames": "test_timing (0,1,2,6) + test_timing_rear (3,5); ego from the test cache",
        "jitter": None if args.arm == "sync" else {
            "jitter_ms": args.jitter_ms, "seed": args.jitter_seed, "future": args.arm,
            "rule": "dt ~ U(-J, J) per slot; cameras 0,1,2,6 from sha256(f'jitter:{seed}:{clip_id}') "
                    "(= Alpamayo 1.5 runs), 3,5 from ...':rear'; t0 slot with dt > 0 -> future"},
        "shard": [args.shard, args.n_shards], "gpus": gpus, "gpu_names": gpu_names,
        "layers_per_gpu": per_gpu,
    }, indent=2))

    checked = False
    t_start = time.time()
    for i, m in enumerate(man):
        if m["clip_id"] in done:
            continue
        t0 = time.time()
        data, picked, dt, future = clip_inputs(m["clip_id"], m["t0_us"], args.arm,
                                               args.jitter_ms, args.jitter_seed)
        drop = [c * 4 + 3 for c in np.flatnonzero(future)] if args.arm == "drop" else []
        model_inputs = build_inputs(model, processor, data, drop)
        if not drop and not checked:
            # this builder must reproduce run_oodval's exactly when nothing is dropped
            ref = ro.build_inputs(model, processor, data)["tokenized_data"]
            for k in ("input_ids", "pixel_values", "image_grid_thw"):
                assert torch.equal(ref[k], model_inputs["tokenized_data"][k]), k
            checked = True
        gt_xy = data["ego_future_xyz"][0, 0, :, :2].numpy()               # (64, 2)

        base = sc.clip_seed(args.seed, m["clip_id"])
        traj_seeds = [base + k for k in range(args.k)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_k, gen_coc, gen_len, has_eos, t_gen, t_den, prompt_len = ro.rollout_then_k_traj(
                model, model_inputs, base, traj_seeds, args.max_gen, {})
        if not np.isfinite(pred_k).all():
            raise RuntimeError(f"non-finite trajectory on {m['clip_id']}")

        ade_k, fde_k = el.ade_fde(pred_k, gt_xy)                          # (K,), (K,)
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
        rec["pred_xy_k"] = np.round(pred_k.astype(float), 3).tolist()      # (K, 64, 2)
        rec.update({"camera_indices": CAMS,
                    "frame_ts_ms": np.round((picked - m["t0_us"]) / 1e3, 1).tolist(),
                    "jitter_dt_ms": np.round(dt, 2).tolist(),
                    "future_t0_cams": [ci for ci, f in zip(CAMS, future) if f],
                    "dup_frames": int((np.diff(picked, axis=1) == 0).sum()),
                    "n_images": 24 - len(drop), "prompt_len": prompt_len,
                    "sec_gen": round(t_gen, 2), "sec_denoise": round(t_den, 2),
                    "sec_total": round(time.time() - t0, 2)})
        rows.append(rec)
        if len(rows) % 5 == 0 or i + 1 == len(man):
            rows_path.write_text(json.dumps(rows, indent=2))
        print(f"[{i + 1}/{len(man)}] {m['clip_id'][:8]} {rec['bucket']:10s} "
              f"ade@8={rec['minADE_rollout']:.3f} imgs={rec['n_images']} gen={gen_len}tok "
              f"tot={time.time() - t0:.1f}s | {gen_coc[:80]}", flush=True)

    rows_path.write_text(json.dumps(rows, indent=2))
    a = np.array([r["minADE_rollout"] for r in rows])
    line = (f"{tag} {args.manifest} shard {args.shard}/{args.n_shards}: {len(rows)} clips, "
            f"minADE@{args.k}(rollout) mean {a.mean():.4f} median {np.median(a):.4f}, "
            f"CoC degen {np.mean([r['coc_degenerate'] for r in rows]):.3f}")
    (out_dir / f"summary_s{args.shard}of{args.n_shards}.txt").write_text(line + "\n")
    print(f"\n{line}\n{(time.time() - t_start) / 60:.1f} min -> {rows_path}", flush=True)


if __name__ == "__main__":
    main()
