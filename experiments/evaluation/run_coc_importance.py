"""Does the CoC a clip happened to sample change that clip's Taylor importance?

`run_importance.py` scores each calibration clip under ONE sampled CoC. On clips where a
CoC resample moves the trajectory (`select_coc_clips.py`), this re-measures the shipped
dual-objective importance under each of the 6 CoCs `run_coc_multimode.py` sampled, with
everything else held:

  C0..C5   CoC j  = rollout under seed  clip_seed + j * 100003,  FM noise seed = clip_seed
  N1..N3   CoC 0,                                                FM noise seed = clip_seed + m

C0 is the shipped estimator for the clip (same rollout seed, same noise seed). The C
conditions change the CoC only; the N conditions change the flow-matching noise only and
are the floor a CoC effect on I_traj has to clear. I_CoC does not use the FM noise, so
its N conditions measure the run-to-run floor of the backward pass.

`process_clip` is run_importance's own, called with the rollout seed split from the noise
seed, so the quantity is the shipped one and not a reimplementation.

G0 (checked by analyze_coc_importance.py): the CoC token ids regenerated here under C_j
decode to the text `run_coc_multimode.py` stored for CoC j of the same clip.

Usage:
  bash experiments/head_analysis/run_retry_host.sh 5 experiments/evaluation/run_coc_importance.py \
      --shard 0 --n-shards 4 --gpu 4
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
sys.path.insert(0, str(Path(__file__).parent))

import analysis_lib as lib
import sample_cache as sc
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from expert_per_clip import reserve_gpu
from run_baseline import COT_END, MODEL_REV, TFS
from run_coc_multimode import COC_STRIDE
from run_importance import process_clip

KEYS = ["coc_vlm_q", "coc_vlm_mlp", "traj_vlm_q", "traj_vlm_mlp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="outputs/coc_importance/clips.json")
    ap.add_argument("--exp-id", default="coc_importance")
    ap.add_argument("--n-coc", type=int, default=6)
    ap.add_argument("--n-noise", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--reserve-gb", type=float, default=40.0)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"s{args.shard}of{args.n_shards}"
    clips = json.loads((REPO / args.clips).read_text())[args.shard::args.n_shards]
    if args.limit:
        clips = clips[: args.limit]

    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    gpu_name = torch.cuda.get_device_name(device)
    conds = ([("C", j) for j in range(args.n_coc)]
             + [("N", m) for m in range(1, args.n_noise + 1)])
    print(f"{args.exp_id} {tag} | {len(clips)} clips x {len(conds)} conditions | {gpu_name}",
          flush=True)

    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    tok = model.tokenizer
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")
    model.vlm.enable_input_require_grads()

    tc, ec = model.vlm.config.text_config, model.expert.config
    shapes = {
        "vlm_q": (tc.num_hidden_layers, tc.num_attention_heads),
        "vlm_mlp": (tc.num_hidden_layers, tc.intermediate_size),
        "kv_k": (tc.num_hidden_layers, tc.num_key_value_heads),
        "kv_v": (tc.num_hidden_layers, tc.num_key_value_heads),
        "exp_q": (ec.num_hidden_layers, ec.num_attention_heads),
        "exp_mlp": (ec.num_hidden_layers, ec.intermediate_size),
    }
    # the shipped fm/gt estimator, every optional axis at its default
    pa = argparse.Namespace(max_gen=256, traj_mode="fm", traj_target="gt", fm_steps=10,
                            k_draws=1, traj_dims=None, traj_weight=None, n_best=6,
                            self_coupling="tied")
    (out_dir / f"config_{tag}.json").write_text(json.dumps({
        "clips": args.clips, "n_clips": len(clips), "conditions": [f"{a}{b}" for a, b in conds],
        "coc_seed_rule": f"clip_seed + j * {COC_STRIDE}", "noise_seed_rule": "clip_seed + m",
        "seed": args.seed, "estimator": "run_importance.process_clip, fm/gt, 10 steps",
        "gpu": gpu_name, "model_revision": MODEL_REV, "keys": KEYS,
    }, indent=2))

    arrs = {k: [] for k in KEYS}          # each entry (n_cond, L, U) float32
    records = []
    for ci, c in enumerate(clips):
        t0 = time.time()
        data = sc.load_cached(sc.path_for(c["cache"], c["clip_id"], c["t0_us"]))
        base = sc.clip_seed(args.seed, c["clip_id"])
        per = {k: [] for k in KEYS}
        conds_rec = []
        for kind, i in conds:
            roll_seed = base + i * COC_STRIDE if kind == "C" else base
            fm_seed = base if kind == "C" else base + i
            acc = {obj: {k: np.zeros(s) for k, s in shapes.items()} for obj in ("coc", "traj")}
            torch.cuda.reset_peak_memory_stats()
            rec = process_clip(model, processor, data, pa, fm_seed, acc, roll_seed=roll_seed)
            for k in KEYS:
                obj, name = k.split("_", 1)
                per[k].append(acc[obj][name].astype(np.float32))
            ids = rec.pop("coc_ids")
            conds_rec.append({"cond": f"{kind}{i}", **rec,
                              "text": tok.decode([t for t in ids if t not in (COT_END, TFS)],
                                                 skip_special_tokens=True)})
        for k in KEYS:
            arrs[k].append(np.stack(per[k]))
        records.append({"clip_id": c["clip_id"], "set": c["set"], "group": c["group"],
                        "conds": conds_rec})
        print(f"[{ci + 1}/{len(clips)}] {c['clip_id'][:8]} {c['group']:7s} "
              f"distinct_coc={len({r['text'] for r in conds_rec[:args.n_coc]})} "
              f"peak={max(r['peak_gb'] for r in conds_rec):.1f}GB ({time.time() - t0:.0f}s)",
              flush=True)
        if (ci + 1) % 5 == 0 or ci + 1 == len(clips):
            np.savez(out_dir / f"imp_{tag}.npz", **{k: np.stack(v) for k, v in arrs.items()})
            (out_dir / f"records_{tag}.json").write_text(json.dumps(records))
    print(f"saved -> {out_dir} ({tag})", flush=True)


if __name__ == "__main__":
    main()
