"""Per-position anatomy inside the generated CoC (plans/2026-09-25_coc-position-content.md, A).

The shipped scores at CoC positions are sums over the CoC tokens; this run keeps the terms. Every
generated-CoC position gets its own typed gate (TypedUnitGates with 5 + n_coc types), so one CE
backward and one full-seed FM backward (the shipped expert pass, as run_gradient_anatomy) give,
for every Q head, its signed contribution at every CoC position under each loss. It also records
the residual-gradient norm of each loss at every CoC position and layer (the token weighting
inside the CoC), the MLP channels' contributions summed over three CoC sub-spans (head clause =
the tokens of the first three words, the rest, the end token), and the decoded token of every CoC
position so any other sub-span can be formed offline.

Same clips, rollout seeds and expert pass as gradanat_v1, so the sum over CoC positions must
reproduce that run's CoC row (G0).

Usage:
  ALPAMAYO_REPO=$PWD bash experiments/head_analysis/run_retry_host.sh 3 \
      experiments/head_analysis/run_coc_position_anatomy.py --gpu 4 --exp-id coc_posanat_v1_s0 --shard 0 --n-shards 4
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
TYPES = ("vision", "hist", "prompt_text", "coc", "sink")
N_BASE = len(TYPES)
PAD = 64  # CoC positions are padded to this many (max_gen is 256 but generations are 10-50 tokens)


def token_types(model, seq_tf, prompt_len):
    sp = lib.compute_spans(model, seq_tf)
    idx = torch.full((seq_tf.shape[1],), TYPES.index("prompt_text"), dtype=torch.long)
    idx[sp["vision"]] = TYPES.index("vision")
    idx[sp["hist"]] = TYPES.index("hist")
    idx[sp["sink"]] = TYPES.index("sink")
    idx[prompt_len:] = N_BASE + torch.arange(seq_tf.shape[1] - prompt_len)  # one type per CoC position
    return idx


def sub_spans(tokenizer, ids):
    """(n_coc,) int: 0 = head clause (tokens of the first three words), 1 = rest, 2 = end token."""
    pieces = [tokenizer.decode([int(t)]) for t in ids]
    words, lab = 0, np.ones(len(ids), np.int64)
    for i, s in enumerate(pieces):
        if i == 0 or s.startswith((" ", "\n")):
            words += 1
        if words <= 3:
            lab[i] = 0
        else:
            break
    lab[-1] = 2
    return lab, pieces


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
        "plan": "plans/2026-09-25_coc-position-content.md (A)", "manifest": args.manifest, "cache": args.cache,
        "num_clips": len(clips), "shard": [args.shard, args.n_shards], "seed": args.seed, "fm_steps": args.fm_steps,
        "sub_spans": ["head clause = tokens of the first three words", "rest", "end token"], "pad": PAD,
        "gpu": torch.cuda.get_device_name(device)}, indent=2))

    per = {k: [] for k in ("q_ce_pos", "q_fm_pos", "mlp_ce_sub", "mlp_fm_sub", "res_ce_norm", "res_fm_norm")}
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
        if n_coc > PAD:
            print(f"[{ci + 1}/{len(clips)}] {clip_id} skipped: coc={n_coc} > PAD", flush=True)
            continue
        type_idx = token_types(model, seq_tf, prompt_len).to("cuda")
        n_types = N_BASE + n_coc
        sub, pieces = sub_spans(model.tokenizer, seq_tf[0, coc_start:coc_end].tolist())
        gates = pl.TypedUnitGates(layers, H, tc.head_dim, I, n_types, "cuda", mlp=True)
        gates.set_types(type_idx)

        res = {"ce": np.full((n_vlm, PAD), np.nan, np.float32), "fm": np.full((n_vlm, PAD), np.nan, np.float32)}
        which = ["ce"]
        hs = [None] * n_vlm
        handles = [layer.register_forward_hook(lambda m, a, out, i=i, hs=hs: hs.__setitem__(i, out[0] if isinstance(out, tuple) else out))
                   for i, layer in enumerate(layers)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden, cache, rope_deltas = pl.vlm_forward_with_grad(model, seq_tf, inputs["tokenized_data"], use_cache=True)
        for h in handles:
            h.remove()

        def keep_norm(l, res=res, which=which, n_coc=n_coc, coc_start=coc_start, coc_end=coc_end):
            def hook(grad):
                res[which[0]][l, :n_coc] = grad[0, coc_start:coc_end].float().norm(dim=-1).cpu().numpy()
            return hook
        for l, h in enumerate(hs):
            h.register_hook(keep_norm(l))
        cache_t = [lib.cache_layer_kv(cache, i) for i in range(n_vlm)]
        prefill = cache.get_seq_length()

        def read(gates=gates, n_coc=n_coc, sub=sub):
            q = gates.q_signed().astype(np.float32)  # (n_types, L, H)
            mlp = gates.mlp_signed().astype(np.float32)  # (n_types, L, I)
            gates.zero_grads()
            q_pos = np.full((PAD, n_vlm, H), np.nan, np.float32)
            q_pos[:n_coc] = q[N_BASE:]
            m_sub = np.zeros((3, n_vlm, I), np.float32)
            for s in range(3):
                m_sub[s] = mlp[N_BASE:][sub == s].sum(0)
            return q_pos, m_sub, q[N_BASE:].sum(0), mlp[N_BASE:].sum(0)

        # CE
        nll = pl.coc_nll(model, hidden, seq_tf, coc_start, coc_end)
        nll.backward(retain_graph=True)
        q_ce, m_ce, q_ce_tot, _ = read()
        # FM: the shipped expert pass, one VLM backward seeded with the whole cache gradient
        which[0] = "fm"
        fm_loss, grads, leaves = pl.expert_fm_grads(model, cache, rope_deltas, x1, args.fm_steps, seed, prefill)
        pl.vlm_backward_from_cache(cache_t, grads, retain=False)
        q_fm, m_fm, q_fm_tot, _ = read()
        gates.remove()
        del hidden, cache, cache_t, grads, leaves

        per["q_ce_pos"].append(q_ce)
        per["q_fm_pos"].append(q_fm)
        per["mlp_ce_sub"].append(m_ce.astype(np.float16))
        per["mlp_fm_sub"].append(m_fm.astype(np.float16))
        per["res_ce_norm"].append(res["ce"])
        per["res_fm_norm"].append(res["fm"])
        rows.append({"clip_id": clip_id, "n_coc": int(n_coc), "head_len": int((sub == 0).sum()), "nll": float(nll), "fm_loss": float(fm_loss),
                     "tokens": pieces, "sub": sub.tolist(),
                     "q_coc_total_ce": q_ce_tot.tolist(), "q_coc_total_fm": q_fm_tot.tolist()})
        done.append(clip_id)
        hs_share = float(np.abs(q_fm[:n_coc][sub == 0]).sum() / np.abs(q_fm[:n_coc]).sum())
        print(f"[{ci + 1}/{len(clips)}] {clip_id} coc={n_coc} head={int((sub == 0).sum())} nll={float(nll):.4f} fm={float(fm_loss):.4f} | "
              f"head-clause share of |Q contributions|: FM {hs_share:.2f} CE {float(np.abs(q_ce[:n_coc][sub == 0]).sum() / np.abs(q_ce[:n_coc]).sum()):.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if (ci + 1) % 5 == 0 or ci + 1 == len(clips):
            np.savez(out_dir / "posanat_perclip.npz", **{k: np.stack(v) for k, v in per.items()})
            (out_dir / "metrics.json").write_text(json.dumps({"n_clips": len(done), "clip_ids": done, "per_clip": rows}, indent=1))
    print("saved ->", out_dir, flush=True)


if __name__ == "__main__":
    main()
