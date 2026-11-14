"""Dual-objective Taylor importance on Alpamayo 2 Edge, with the token-type split and per-step expert gates.

Plan: iclr27-rebuttal/plans/2026-11-14_a2-edge-analysis-replication.md. Same per-clip protocol and file layout as
alpamayo2-distill/experiments/importance/run_importance_a2.py (A2 Super, a2_importance_v1):

  1. no-grad CoC rollout (release sampling top_p 0.98 / T 0.6 / max 256, seed = sha256(f"42:{clip_id}")[:4])
  2. one teacher-forced VLM forward with grad on prompt + generated CoC -> KV cache
  3. CoC NLL backward (retain_graph)                          -> I_CE by token type
  4. 10-step FM loss through the expert onto detached cache leaves, then ONE VLM backward seeded with the
     accumulated cache gradient                               -> I_FM by token type
  5. expert gates read per denoising step                     -> I_FM(expert) per step

Prompt: Edge's driving prompt without the navigation line (plan decision Q1). Inputs: calib_edge cache.
One card (2.65 B model). Run in the Edge venv.

Gates recorded per clip: --verify (G0: typed sum vs plain Q gates; G0b: per-step expert sum vs accumulated),
rope_deltas of the teacher-forced forward == rope_deltas of the generation prefill (always), tail_ok, finite losses.

Usage:
  CUDA_VISIBLE_DEVICES=6 /mnt/nvme1n1/ad_vla/venvs/alpamayo2-edge/bin/python \
      iclr27-rebuttal/code/run_importance_edge.py --shard 0 --n-shards 2
  /mnt/nvme1n1/ad_vla/venvs/alpamayo2-edge/bin/python iclr27-rebuttal/code/run_importance_edge.py --finalize
"""

import os
import sys

FINALIZE = "--finalize" in sys.argv
if not FINALIZE and not os.environ.get("CUDA_VISIBLE_DEVICES"):
    raise SystemExit("set CUDA_VISIBLE_DEVICES to one empty card, e.g. 6")
if FINALIZE:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import json
import random
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import edge_common as ec

ec.setup_hf_env(offline=True)

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
NT = 5


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def process_clip(model, processor, data, args, seed, dims):
    import edge_grad_lib as gl

    n_vlm, H, D, I, _, EH, ED, EI = dims
    dev = torch.device("cuda")
    t_all = time.time()
    model_inputs, _ = ec.prepare_inputs_no_nav(data, model.config, model.tokenizer, processor)
    model_inputs = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in model_inputs.items()}
    model_inputs["tokenized_data"] = {k: (v.to(dev) if torch.is_tensor(v) else v)
                                      for k, v in model_inputs["tokenized_data"].items()}

    seed_all(seed)
    t = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        seq, prompt_len, has_eos, cot, rd_gen = gl.rollout_sequence(model, model_inputs, args.max_gen)
    seq = seq.clone()  # out of inference mode
    sec_roll = time.time() - t
    coc_start, coc_end = prompt_len, seq.shape[1]
    anchor = model.config.traj_ids["future_start"]
    gen_ids = seq[0, prompt_len:].tolist()
    tail_ok = has_eos and gen_ids[-1:] == [anchor]
    type_idx = gl.token_types(model, seq, prompt_len)  # (T,)
    n_tok = np.bincount(type_idx.numpy(), minlength=NT)
    x1 = gl.gt_actions(model, data, dev)  # (1, 64, 2) fp32
    td = gl.fused_tokenized(model, model_inputs)
    assert torch.equal(td["input_ids"], seq[:, :prompt_len]), "rollout prompt != fused prompt"

    vlm_layers = model.vlm.model.language_model.layers
    gates = gl.TypedUnitGates(vlm_layers, H, D, I, NT)
    gates.set_types(type_idx)
    ref = gl.UnitGates(vlm_layers, H, D, I, mlp=False) if args.verify else None
    exp_gates = gl.UnitGates(model.expert.expert.layers, EH, ED, EI)
    exp_ref = gl.UnitGates(model.expert.expert.layers, EH, ED, EI) if args.verify else None

    t = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden, cache, rd_tf = gl.vlm_forward_with_grad(model, seq, td)
    sec_fwd = time.time() - t
    rope_ok = bool(torch.equal(rd_tf.cpu(), rd_gen.cpu()))
    cache_t = [gl.cache_layer_kv(cache, i) for i in range(n_vlm)]
    if not all(k.requires_grad and v.requires_grad for k, v in cache_t):
        raise RuntimeError("cache k/v carry no graph; enable_input_require_grads() missing")
    for k, v in cache_t:
        k.retain_grad()
        v.retain_grad()
    prefill = cache.get_seq_length()
    assert prefill == seq.shape[1]

    verify = []

    def read():
        q, mlp = gates.q_signed(), gates.mlp_signed()  # (5, L, H), (5, L, I)
        if ref is not None:
            s, p = q.sum(0), ref.q_signed()
            den = np.linalg.norm(p, axis=1)
            ok = den > 0
            verify.append(float((np.linalg.norm(s - p, axis=1)[ok] / den[ok]).max()))
            ref.zero_grads()
        gates.zero_grads()
        return q, mlp

    # ---- objective 1: CoC NLL ----
    t = time.time()
    nll = gl.coc_nll(model, hidden, seq, coc_start, coc_end)
    nll.backward(retain_graph=True)
    ce_q, ce_mlp = read()
    coc_kv = gl.kv_group_scores(cache_t)
    for k, v in cache_t:
        k.grad = None
        v.grad = None
    nll_val = nll.item()
    del hidden, nll
    sec_ce = time.time() - t

    # ---- objective 2: trajectory flow matching ----
    t = time.time()
    fm_loss, grads, leaves, steps = gl.expert_fm_grads(model, cache, rd_tf, x1, args.fm_steps, seed, prefill,
                                                      step_gates=exp_gates)
    exp_q, exp_mlp = steps["q"].sum(0), steps["mlp"].sum(0)  # (L, EH), (L, EI)
    g0b = None
    if exp_ref is not None:
        g0b = {}
        for name, rec_s, acc_s in (("q", exp_q, exp_ref.q_signed()), ("mlp", exp_mlp, exp_ref.mlp_signed())):
            den = np.linalg.norm(acc_s, axis=1)
            g0b[name] = float((np.linalg.norm(np.abs(rec_s) - np.abs(acc_s), axis=1) / den).max())
    direct = gl.cache_read(leaves, grads, type_idx, NT)  # (L, 5)
    sec_exp = time.time() - t
    t = time.time()
    ts, gs = [], []
    for (k, v), (gk, gv) in zip(cache_t, grads):
        for t_, g_ in ((k, gk), (v, gv)):
            if g_ is not None:
                ts.append(t_)
                gs.append(g_.to(t_.dtype))
    torch.autograd.backward(ts, gs)
    fm_q, fm_mlp = read()
    traj_kv = gl.kv_group_scores(cache_t)
    sec_fm = time.time() - t

    peak = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    for g in (gates, exp_gates, ref, exp_ref):
        if g is not None:
            g.remove()
    del cache, cache_t, leaves, grads, ts, gs
    torch.cuda.empty_cache()

    arrays = {
        "ce_q_type": ce_q.astype(np.float32), "fm_q_type": fm_q.astype(np.float32),
        "ce_mlp_type": ce_mlp.astype(np.float32), "fm_mlp_type": fm_mlp.astype(np.float32),
        "traj_exp_q_signed": exp_q.astype(np.float32), "traj_exp_mlp_signed": exp_mlp.astype(np.float32),
        "coc_kv_k": coc_kv[0], "coc_kv_v": coc_kv[1], "traj_kv_k": traj_kv[0], "traj_kv_v": traj_kv[1],
        "direct": direct, "n_tok": n_tok,
        "exp_q_step": steps["q"].astype(np.float32),  # (S, L, EH) signed
        "exp_mlp_step": steps["mlp"].astype(np.float32),  # (S, L, EI) signed
    }
    rec = {"coc_len": int(coc_end - coc_start), "prompt_len": int(prompt_len), "has_eos": has_eos,
           "tail_ok": tail_ok, "rope_deltas_tf_eq_gen": rope_ok, "cot": cot, "nll": nll_val, "fm_loss": fm_loss,
           "n_tok": {k: int(n) for k, n in zip(gl.TYPES, n_tok)}, "peak_gb": peak,
           "sec": {"rollout": round(sec_roll, 1), "forward": round(sec_fwd, 1), "ce_backward": round(sec_ce, 1),
                   "fm_expert": round(sec_exp, 1), "fm_backward": round(sec_fm, 1),
                   "total": round(time.time() - t_all, 1)}}
    if args.verify:
        rec["typed_vs_plain_relerr_q"] = verify  # [CE, FM], max over layers
        rec["g0b_steps_vs_accumulated_relerr"] = g0b
    return arrays, rec


def finalize(out_dir, allow_partial):
    """Stack per-clip files into the layout of a2_importance_v1 (importance_v2 / gradanat_v1 / stepimp_*)."""
    clips = pd.read_parquet(ec.CALIB_MANIFEST)["clip_id"].tolist()
    have = [c for c in clips if (out_dir / "perclip" / f"{c}.npz").exists()]
    if len(have) != len(clips) and not allow_partial:
        raise SystemExit(f"{len(have)}/{len(clips)} clips done; pass --allow-partial to finalize anyway")
    recs, stack = [], {}
    for c in have:
        z = np.load(out_dir / "perclip" / f"{c}.npz")
        for k in z.files:
            stack.setdefault(k, []).append(z[k])
        r = json.loads((out_dir / "perclip" / f"{c}.json").read_text())
        r["clip_id"] = c
        recs.append(r)
    s = {k: np.stack(v) for k, v in stack.items()}  # (N, ...)
    pooled = {"coc_vlm_q": np.abs(s["ce_q_type"].sum(1)), "traj_vlm_q": np.abs(s["fm_q_type"].sum(1)),
              "coc_vlm_mlp": np.abs(s["ce_mlp_type"].sum(1)), "traj_vlm_mlp": np.abs(s["fm_mlp_type"].sum(1)),
              "traj_exp_q": np.abs(s["traj_exp_q_signed"]), "traj_exp_mlp": np.abs(s["traj_exp_mlp_signed"])}
    pooled["coc_exp_q"] = np.zeros_like(pooled["traj_exp_q"])  # the CoC NLL has no path into the expert
    pooled["coc_exp_mlp"] = np.zeros_like(pooled["traj_exp_mlp"])
    for k in ("coc_kv_k", "coc_kv_v", "traj_kv_k", "traj_kv_v"):
        pooled[k] = s[k]
    np.savez(out_dir / "importance.npz", **{k: v.mean(0) for k, v in pooled.items()})
    np.savez(out_dir / "importance_perclip.npz", **{k: v.astype(np.float32) for k, v in pooled.items()})
    np.savez(out_dir / "anatomy_perclip_q.npz", ce=s["ce_q_type"], fm_full=s["fm_q_type"])
    np.savez(out_dir / "anatomy_perclip_mlp.npz", ce_type=s["ce_mlp_type"], fm_type=s["fm_mlp_type"])
    np.savez(out_dir / "anatomy_direct.npz", direct=s["direct"], n_tok=s["n_tok"])
    st = {}
    for ax in ("q", "mlp"):
        g = s[f"exp_{ax}_step"]  # (N, S, L, U) signed
        st[f"{ax}_abs_step"] = np.abs(g).mean(0)
        st[f"{ax}_signed_step"] = g.mean(0)
        st[f"{ax}_shipped"] = np.abs(g.sum(1)).mean(0)
    np.savez(out_dir / "step_importance.npz", **st)
    np.savez(out_dir / "step_importance_perclip.npz",
             **{f"{ax}_mass": np.abs(s[f"exp_{ax}_step"]).sum((2, 3)) for ax in ("q", "mlp")})
    gates = {"n_clips": len(recs), "all_tail_ok": all(r["tail_ok"] for r in recs),
             "all_rope_deltas_tf_eq_gen": all(r["rope_deltas_tf_eq_gen"] for r in recs),
             "all_finite": bool(all(np.isfinite([r["nll"], r["fm_loss"]]).all() for r in recs)),
             "n_tail_fail": sum(not r["tail_ok"] for r in recs)}
    (out_dir / "metrics.json").write_text(json.dumps({**gates, "per_clip": recs}, indent=2))
    nll, fm = np.array([r["nll"] for r in recs]), np.array([r["fm_loss"] for r in recs])
    (out_dir / "summary.txt").write_text(
        f"Alpamayo 2 Edge dual-objective importance (calib_100, no-nav prompt)\nclips: {len(recs)}\n"
        f"gates: {json.dumps(gates)}\nCoC NLL mean {nll.mean():.4f} (median {np.median(nll):.4f})\n"
        f"FM loss mean {fm.mean():.4f} (median {np.median(fm):.4f})\n"
        f"CoC length mean {np.mean([r['coc_len'] for r in recs]):.1f} tokens\n")
    print(f"finalized {len(recs)} clips -> {out_dir}\n{json.dumps(gates)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="iclr27-rebuttal/edge_importance_v1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--fm-steps", type=int, default=10)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--clip", default=None)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="recompute clips already on disk (determinism check)")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    if args.finalize:
        finalize(out_dir, args.allow_partial)
        return
    (out_dir / "perclip").mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    from alpamayo2_edge import helper

    man = pd.read_parquet(ec.CALIB_MANIFEST)
    man = [(r.clip_id, int(r.t0_us)) for r in man.itertuples()]
    if args.clip:  # comma-separated clip-id prefixes
        man = [m for m in man if any(m[0].startswith(p) for p in args.clip.split(","))]
    man = man[args.shard :: args.n_shards]
    if args.limit:
        man = man[: args.limit]

    t = time.time()
    model, info = ec.load_model("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.vlm.enable_input_require_grads()  # frozen weights: without this the cache tensors carry no graph
    processor = helper.get_processor(model.tokenizer, model.config)
    tc, ecfg = model.config.vlm_config.text_config, model.expert.expert.config
    dims = (tc.num_hidden_layers, tc.num_attention_heads, tc.head_dim, tc.intermediate_size,
            ecfg.num_hidden_layers, ecfg.num_attention_heads, ecfg.head_dim, ecfg.intermediate_size)
    gpu = torch.cuda.get_device_name(0)
    print(f"{args.exp_id} | shard {args.shard}/{args.n_shards} | {len(man)} clips | card "
          f"{os.environ['CUDA_VISIBLE_DEVICES']} ({gpu}) | loaded in {time.time() - t:.0f}s | VLM L{dims[0]} Q{dims[1]} "
          f"I{dims[3]} | expert L{dims[4]} Q{dims[5]} I{dims[7]} | attn {model.config._attn_implementation}", flush=True)
    (out_dir / f"config_s{args.shard}of{args.n_shards}.json").write_text(json.dumps({
        "model": ec.MODEL_ID, "model_revision": ec.MODEL_REV,
        "plan": "iclr27-rebuttal/plans/2026-11-14_a2-edge-analysis-replication.md",
        "protocol": "run_importance_a2.py (a2_importance_v1) ported to Edge: TypedUnitGates on o_proj / mlp.fc2, "
                    "per-step expert gates",
        "objectives": {"coc": "own-rollout CoC NLL", "traj": "flow-matching MSE vs GT action (course-heading space)"},
        "prompt": "Edge driving prompt without the navigation line (plan decision Q1)",
        "fm_estimator": "t_s=(s+0.5)/S, fresh CPU-generator noise per step seeded with the clip seed, "
                        "x_t=t*x1+(1-t)*eps, target x1-eps, x_t fp32 under bf16 autocast",
        "types": ["vision", "hist", "prompt_text", "coc", "sink"], "manifest": str(ec.CALIB_MANIFEST),
        "cache": str(ec.CALIB_CACHE), "n_clips_shard": len(man), "seed": args.seed,
        "seed_rule": "sha256(f'{seed}:{clip_id}')[:4]", "fm_steps": args.fm_steps, "max_gen": args.max_gen,
        "verify": args.verify, "gpu": os.environ["CUDA_VISIBLE_DEVICES"], "gpu_name": gpu,
        "shapes": {"vlm_q": [dims[0], dims[1]], "vlm_mlp": [dims[0], dims[3]], "exp_q": [dims[4], dims[5]],
                   "exp_mlp": [dims[4], dims[7]]},
        "torch": torch.__version__, "loading_info": {k: len(v) if hasattr(v, "__len__") else str(v)
                                                     for k, v in (info or {}).items()},
    }, indent=2))

    t_start = time.time()
    for i, (clip_id, t0_us) in enumerate(man):
        dst = out_dir / "perclip" / f"{clip_id}.npz"
        if dst.exists() and not args.overwrite:
            continue
        torch.cuda.reset_peak_memory_stats()
        data = ec.load_cached(ec.path_for(clip_id, t0_us))
        arrays, rec = process_clip(model, processor, data, args, ec.clip_seed(args.seed, clip_id), dims)
        tmp = dst.with_suffix(".tmp.npz")
        np.savez(tmp, **arrays)
        (out_dir / "perclip" / f"{clip_id}.json").write_text(json.dumps(rec, indent=2))
        tmp.rename(dst)
        v = (f" verify {rec['typed_vs_plain_relerr_q']} g0b {rec['g0b_steps_vs_accumulated_relerr']}"
             if args.verify else "")
        print(f"[{i + 1}/{len(man)}] {clip_id[:8]} coc={rec['coc_len']} nll={rec['nll']:.3f} fm={rec['fm_loss']:.4f} "
              f"tail_ok={rec['tail_ok']} rope_ok={rec['rope_deltas_tf_eq_gen']} peakGB={rec['peak_gb']} "
              f"sec={rec['sec']}{v} | {rec['cot'][:80]}", flush=True)
    print(f"done {len(man)} clips in {(time.time() - t_start) / 60:.1f} min -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
