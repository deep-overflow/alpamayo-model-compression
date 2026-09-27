"""Per-position anatomy over every non-vision token: prompt text, ego-history tokens, the sink and
the generated CoC (extends run_coc_position_anatomy.py, which kept only the CoC positions).

Every non-vision position gets its own typed gate (vision stays one type), so one CE backward and
one full-seed FM backward per clip give, for every Q head, its signed contribution at every text /
history / sink / CoC position under each loss (the sum over positions reproduces the shipped
score). Also kept: the residual-gradient norm of each loss at every such position and layer, the
decoded token and kind (sink / hist / prompt text / CoC) of every position. MLP channels are not
kept per position (272 x 12288 per layer per clip is too large); analyze_coc_position_anatomy has
their CoC sub-span sums.

Same clips, rollout seeds and expert pass as gradanat_v1 / coc_posanat_v1.

Usage:
  ALPAMAYO_REPO=$PWD bash experiments/head_analysis/run_retry_host.sh 3 \
      experiments/head_analysis/run_text_position_anatomy.py --gpu 4 --exp-id textpos_v1_s0 --shard 0 --n-shards 4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

import analysis_lib as lib
import prune_lib as pl
import sample_cache as sc
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from expert_per_clip import reserve_gpu

REPO = Path(__file__).resolve().parents[2]
MODEL_REV = "7aba8293c09993f2e125c6819df05d7fa3e873ea"
KINDS = ("sink", "hist", "prompt_text", "coc")
PAD_COC = 64  # CoC positions beyond this are skipped (as coc_posanat_v1)
PAD_ALL = 280  # non-vision prompt positions (206 here) + CoC (<= 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", type=str, required=True)
    ap.add_argument("--manifest", default="calib_100")
    ap.add_argument("--cache", default="calib")
    ap.add_argument("--num-clips", type=int, default=100)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--fm-steps", type=int, default=10)
    ap.add_argument("--reserve-gb", type=float, default=44.0)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = sc.calib_samples(REPO, args.manifest)[: args.num_clips][args.shard :: args.n_shards]
    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    print(f"using {device}", flush=True)
    model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")
    model.vlm.enable_input_require_grads()
    tc = model.vlm.config.text_config
    layers = model.vlm.model.language_model.layers
    n_vlm, H, I = len(layers), tc.num_attention_heads, tc.intermediate_size

    (out_dir / "config.json").write_text(json.dumps({
        "model": "nvidia/Alpamayo-1.5-10B", "model_revision": MODEL_REV,
        "plan": "plans/2026-09-25_coc-position-content.md (text positions)", "manifest": args.manifest, "cache": args.cache,
        "num_clips": len(clips), "shard": [args.shard, args.n_shards], "seed": args.seed, "fm_steps": args.fm_steps,
        "kinds": KINDS, "pad_all": PAD_ALL, "pad_coc": PAD_COC, "mlp": False,
        "gpu": torch.cuda.get_device_name(device)}, indent=2))

    per = {k: [] for k in ("q_ce_pos", "q_fm_pos", "res_ce_norm", "res_fm_norm")}
    rows, done = [], []
    for ci, (clip_id, clip_t0) in enumerate(clips):
        t0 = time.time()
        data = sc.load_cached(sc.path_for(args.cache, clip_id, clip_t0))
        inputs = lib.build_inputs(model, processor, data, "cuda")
        prompt_len = inputs["input_ids"].shape[1]
        x1 = lib.gt_actions(model, data, "cuda").to(torch.float32)
        seed = sc.clip_seed(args.seed, clip_id)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
        coc_start, coc_end = prompt_len, roll["eos_pos"] + 1
        seq_tf = roll["sequences"][:, :coc_end]
        del roll
        n_coc = coc_end - coc_start
        if n_coc > PAD_COC:
            print(f"[{ci + 1}/{len(clips)}] {clip_id} skipped: coc={n_coc} > PAD", flush=True)
            continue
        sp = lib.compute_spans(model, seq_tf)
        T = seq_tf.shape[1]
        kind = np.full(T, -1, np.int64)
        kind[sp["sink"].numpy()] = 0
        kind[sp["hist"].numpy()] = 1
        kind[sp["text"].numpy() & (np.arange(T) < prompt_len)] = 2
        kind[prompt_len:] = 3
        nonvis = np.where(kind >= 0)[0]  # every non-vision position, in sequence order
        n_pos = len(nonvis)
        assert n_pos <= PAD_ALL, n_pos
        type_idx = torch.zeros(T, dtype=torch.long)  # vision = type 0
        type_idx[torch.as_tensor(nonvis)] = torch.arange(1, n_pos + 1)
        type_idx = type_idx.to("cuda")
        n_types = 1 + n_pos
        gates = pl.TypedUnitGates(layers, H, tc.head_dim, I, n_types, "cuda", mlp=False)
        gates.set_types(type_idx)

        res = {"ce": np.full((n_vlm, PAD_ALL), np.nan, np.float32), "fm": np.full((n_vlm, PAD_ALL), np.nan, np.float32)}
        which = ["ce"]
        hs = [None] * n_vlm
        handles = [layer.register_forward_hook(lambda m, a, out, i=i, hs=hs: hs.__setitem__(i, out[0] if isinstance(out, tuple) else out))
                   for i, layer in enumerate(layers)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden, cache, rope_deltas = pl.vlm_forward_with_grad(model, seq_tf, inputs["tokenized_data"], use_cache=True)
        for h in handles:
            h.remove()
        nonvis_t = torch.as_tensor(nonvis, device="cuda")

        def keep_norm(l, res=res, which=which, nonvis_t=nonvis_t, n_pos=n_pos):
            def hook(grad):
                res[which[0]][l, :n_pos] = grad[0].index_select(0, nonvis_t).float().norm(dim=-1).cpu().numpy()
            return hook
        for l, h in enumerate(hs):
            h.register_hook(keep_norm(l))
        cache_t = [lib.cache_layer_kv(cache, i) for i in range(n_vlm)]
        prefill = cache.get_seq_length()

        def read(gates=gates, n_pos=n_pos):
            q = gates.q_signed().astype(np.float32)  # (n_types, L, H)
            gates.zero_grads()
            q_pos = np.full((PAD_ALL, n_vlm, H), np.nan, np.float32)
            q_pos[:n_pos] = q[1:]
            return q_pos, q[0]

        nll = pl.coc_nll(model, hidden, seq_tf, coc_start, coc_end)
        nll.backward(retain_graph=True)
        q_ce, q_ce_vis = read()
        which[0] = "fm"
        fm_loss, grads, leaves = pl.expert_fm_grads(model, cache, rope_deltas, x1, args.fm_steps, seed, prefill)
        pl.vlm_backward_from_cache(cache_t, grads, retain=False)
        q_fm, q_fm_vis = read()
        gates.remove()
        del hidden, cache, cache_t, grads, leaves

        per["q_ce_pos"].append(q_ce)
        per["q_fm_pos"].append(q_fm)
        per["res_ce_norm"].append(res["ce"])
        per["res_fm_norm"].append(res["fm"])
        ids = seq_tf[0, nonvis].tolist()
        rows.append({"clip_id": clip_id, "n_pos": int(n_pos), "n_coc": int(n_coc), "prompt_len": int(prompt_len), "T": int(T),
                     "nll": float(nll), "fm_loss": float(fm_loss),
                     "kind": kind[nonvis].tolist(), "ids": ids, "tokens": [model.tokenizer.decode([t]) for t in ids],
                     "q_vision_ce": q_ce_vis.tolist(), "q_vision_fm": q_fm_vis.tolist()})
        done.append(clip_id)
        k = kind[nonvis]
        fm_abs = np.abs(q_fm[:n_pos]).sum((1, 2))
        tot = fm_abs.sum() + np.abs(q_fm_vis).sum()
        print(f"[{ci + 1}/{len(clips)}] {clip_id} pos={n_pos} coc={n_coc} nll={float(nll):.4f} fm={float(fm_loss):.4f} | FM |Q| share: "
              f"vision {np.abs(q_fm_vis).sum() / tot:.2f} sink {fm_abs[k == 0].sum() / tot:.2f} hist {fm_abs[k == 1].sum() / tot:.2f} "
              f"prompt {fm_abs[k == 2].sum() / tot:.2f} coc {fm_abs[k == 3].sum() / tot:.2f} ({time.time() - t0:.0f}s)", flush=True)
        if (ci + 1) % 5 == 0 or ci + 1 == len(clips):
            np.savez(out_dir / "textpos_perclip.npz", **{k_: np.stack(v) for k_, v in per.items()})
            (out_dir / "metrics.json").write_text(json.dumps({"n_clips": len(done), "clip_ids": done, "per_clip": rows}, indent=1))
    print("saved ->", out_dir, flush=True)


if __name__ == "__main__":
    main()
