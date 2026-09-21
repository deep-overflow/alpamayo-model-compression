"""Gradient anatomy: on which tokens, and through which cache layers, does each loss see a unit?

plans/2026-09-20_gradient-anatomy.md. The shipped score is |dL/dg_u| with
dL/dg_u = sum_p x_u(p) dL/dx_u(p): one gate per unit, shared by every position, abs taken
after the sum. So it never records WHERE the gradient lands. This pass splits that same
sum three ways, each split exact (the parts add back to the shipped signed gradient):

  D1  by the token type the UNIT acts on   -- TypedUnitGates, one gate per (type, unit);
      populated on every backward below, for both losses
  D2  by the cache LAYER BAND the FM gradient enters the VLM through
      -- vlm_backward_from_cache is linear in its seed, so one backward per band
  D3  by the cache POSITION TYPE the FM gradient enters through -- same, seed masked

Per clip: 1 CE backward, 1 full-seed FM backward (the shipped operation; D1 is read off it),
6 band backwards and 5 position-type backwards, all on one retained graph. Everything else is the shipped run_importance protocol unchanged (own-rollout CoC
teacher-forced, expert_fm_grads with the GT action target, clip seeds, FP32 gates).

Also recorded, because they cost nothing on the same pass:
  - the FM loss's DIRECT read of the cache, |k dL/dk| + |v dL/dv| on the expert-side leaves,
    by (cache layer, position type)
  - the residual-stream gradient of each loss after every decoder layer: energy by token
    type, and the cosine between the two losses' gradients at the same positions. This is
    the plan's fallback measurement (H-dir), taken here so it needs no second run.

--verify installs the shipped UnitGates next to Q-head-only typed gates and checks, on
every backward, that the typed grads summed over types equal the single-gate grad (G0).
It is a separate mode because a second MLP gate would save one more (T, 12288) tensor per
layer, which does not fit next to a 42 GB pass on a 48 GB card.

Pre-registered gates (G0, P1-P4) are evaluated by analyze_gradient_anatomy.py.

Two later modes, plans/2026-09-21_importance-causal-validation.md:

--mask (part A) measures the same anatomy on a pruned model. The rollout runs before any
hook is installed, so every arm is teacher-forced on the DENSE model's text with the same
positions and noise; the typed gates then start at the 0/1 keep mask (TypedUnitGates.set_mask),
which removes the masked units for the forward and both backwards. The per-clip CoC NLL and
FM loss on that fixed text are the arm's matched functional readout.

--probes (part D) keeps the CE and the full-seed FM backward and replaces the band and
position splits by three random linear readouts with no language or driving content:
  head    sum over the CE positions of <r_p, h_final(p)>
  expert  the FM sweep with its loss replaced by <R_s, v_theta(x_s, t_s)> (same x_s)
  cache   <R, [K_m; V_m]> on every cache layer and position alike, no expert involved
analyze_pruned_anatomy.py and analyze_probes.py evaluate the gates of those two parts.

Usage:
  ALPAMAYO_REPO=$PWD bash experiments/head_analysis/run_retry_host.sh 3 \
      experiments/head_analysis/run_gradient_anatomy.py --gpu 4 --exp-id gradanat_v1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

import analysis_lib as lib  # noqa: E402
import prune_lib as pl  # noqa: E402
import sample_cache as sc  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  also installs the gated-repo hub patch

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MODEL_REV = "7aba8293c09993f2e125c6819df05d7fa3e873ea"
TYPES = ("vision", "hist", "prompt_text", "coc", "sink")
BANDS = ((0, 7), (8, 15), (16, 21), (22, 24), (25, 29), (30, 35))


def token_types(model, seq_tf, prompt_len):
    """(T,) long on cuda: index into TYPES for every position of the teacher-forced sequence."""
    sp = lib.compute_spans(model, seq_tf)
    idx = torch.full((seq_tf.shape[1],), TYPES.index("prompt_text"), dtype=torch.long)
    idx[sp["vision"]] = TYPES.index("vision")
    idx[sp["hist"]] = TYPES.index("hist")
    idx[sp["sink"]] = TYPES.index("sink")
    idx[prompt_len:] = TYPES.index("coc")  # everything the model generated
    return idx.to("cuda")


def by_type(values, type_idx):
    """values (T,) -> (n_types,) sums over the positions of each type."""
    return torch.zeros(len(TYPES), device=values.device,
                       dtype=torch.float64).index_add_(0, type_idx, values.double())


def rel_err(a, b):
    """max over layers of ||a_l - b_l|| / ||b_l||. a, b: (L, U)."""
    num, den = np.linalg.norm(a - b, axis=1), np.linalg.norm(b, axis=1)
    return float(np.max(num[den > 0] / den[den > 0])) if (den > 0).any() else 0.0


def process_clip(model, processor, data, args, seed, mask=None):
    inputs = lib.build_inputs(model, processor, data, "cuda")
    prompt_len = inputs["input_ids"].shape[1]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
    coc_start, coc_end = prompt_len, roll["eos_pos"] + 1
    seq_tf = roll["sequences"][:, :coc_end]  # (1, T) prompt + generated CoC
    del roll

    x1 = lib.gt_actions(model, data, "cuda").to(torch.float32)  # (1, 64, 2)
    type_idx = token_types(model, seq_tf, prompt_len)  # (T,)
    n_tok = np.bincount(type_idx.cpu().numpy(), minlength=len(TYPES))

    tc = model.vlm.config.text_config
    layers = model.vlm.model.language_model.layers
    n_vlm = len(layers)
    ref = None
    if args.verify:
        ref = pl.UnitGates(layers, tc.num_attention_heads, tc.head_dim, tc.intermediate_size,
                           "cuda", torch.float32)
    gates = pl.TypedUnitGates(layers, tc.num_attention_heads, tc.head_dim,
                              tc.intermediate_size, len(TYPES), "cuda", mlp=not args.verify)
    gates.set_types(type_idx)
    if mask is not None:
        # the rollout above ran with no hook installed: the text is the dense model's
        gates.set_mask(q=mask[0], mlp=mask[1])

    # residual stream after every decoder layer, captured so its .grad can be read
    hs = [None] * n_vlm
    handles = [layer.register_forward_hook(
        lambda m, a, out, i=i: hs.__setitem__(i, out[0] if isinstance(out, tuple) else out))
        for i, layer in enumerate(layers)]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden, cache, rope_deltas = pl.vlm_forward_with_grad(
            model, seq_tf, inputs["tokenized_data"], use_cache=True
        )
    for h in handles:
        h.remove()
    # tensor hooks rather than retain_grad: they can be switched off, so the five
    # position-type backwards do not pile another 0.9 GB of .grad onto a 44 GB pass
    res_on = [True]
    res_buf = [None] * n_vlm  # (T, 4096) bf16 each, summed over the backwards since reset

    def keep_grad(l):
        def hook(grad):
            if res_on[0]:
                if res_buf[l] is None:
                    res_buf[l] = grad[0].clone()
                else:
                    res_buf[l] += grad[0]
        return hook
    for l, h in enumerate(hs):
        h.register_hook(keep_grad(l))
    cache_t = [lib.cache_layer_kv(cache, i) for i in range(n_vlm)]
    if not all(k.requires_grad and v.requires_grad for k, v in cache_t):
        raise RuntimeError("cache k/v carry no graph; enable_input_require_grads() missing")
    prefill = cache.get_seq_length()

    verify = []

    def read():
        """Signed typed grads of the backward that just ran, then reset. (n_types, L, U)."""
        q = gates.q_signed()
        mlp = None if args.verify else gates.mlp_signed()
        if mask is not None:
            # a zeroed gate's grad is the gain of switching the unit back on, not an importance
            q, mlp = q * mask[0][None], mlp * mask[1][None]  # (5, L, H), (5, L, I)
        if args.verify:
            verify.append(rel_err(q.sum(0), ref.q_signed()))
            ref.zero_grads()
        gates.zero_grads()
        return q, mlp

    def residual(ce_cpu=None):
        """Per-layer stats of the residual gradient collected in res_buf, then reset."""
        energy = np.zeros((n_vlm, len(TYPES)))
        dot = np.zeros((n_vlm, len(TYPES)))
        cos = np.zeros((n_vlm, len(TYPES)))
        kept = []
        with torch.no_grad():
            for l in range(n_vlm):
                if res_buf[l] is None:
                    # no path from this layer's output to the loss: the FM loss cannot see
                    # h_35, which feeds no cache layer (why I_traj is exactly zero there)
                    if ce_cpu is None:
                        raise RuntimeError(f"CE left no gradient on the output of layer {l}")
                    continue
                g = res_buf[l].float()  # (T, 4096)
                energy[l] = by_type(g.pow(2).sum(-1), type_idx).cpu().numpy()
                if ce_cpu is None:
                    kept.append(res_buf[l].to("cpu"))
                else:
                    gc = ce_cpu[l].to("cuda").float()  # (T, 4096)
                    d = (g * gc).sum(-1)  # (T,)
                    dot[l] = by_type(d, type_idx).cpu().numpy()
                    c = d / (g.norm(dim=-1) * gc.norm(dim=-1)).clamp_min(1e-30)
                    cos[l] = (by_type(c, type_idx).cpu().numpy() / np.maximum(n_tok, 1))
                res_buf[l] = None
        return energy, dot, cos, kept

    # ---- CE: one backward ----
    nll = pl.coc_nll(model, hidden, seq_tf, coc_start, coc_end)
    nll.backward(retain_graph=True)
    ce_q, ce_mlp = read()
    res_ce, _, _, ce_cpu = residual()
    nll_val = float(nll)
    probe = {}
    if args.probes:
        res_on[0] = False
        gen = torch.Generator(device="cuda").manual_seed(seed + 7919)
        h = hidden[0, coc_start - 1 : coc_end - 1].float()  # (Tc, 4096) the states the CE reads
        r = torch.randn(h.shape, generator=gen, device=h.device)  # (Tc, 4096)
        (r * h).sum(-1).mean().backward(retain_graph=True)
        probe["head"] = read()
        res_on[0] = True
        del h, r
    del hidden, nll

    # ---- FM: the shipped expert pass, then the seed split two ways ----
    fm_loss, grads, leaves = pl.expert_fm_grads(
        model, cache, rope_deltas, x1, args.fm_steps, seed, prefill
    )

    def cache_read(lv, gr):
        """|k dL/dk| + |v dL/dv| on the expert-side leaves, and the energy of the cache
        gradient itself, by (cache layer, position type). (L, 5) each."""
        d, e = np.zeros((n_vlm, len(TYPES))), np.zeros((n_vlm, len(TYPES)))
        with torch.no_grad():
            for m, ((k, v), (gk, gv)) in enumerate(zip(lv, gr)):
                if gk is None or gv is None:
                    raise RuntimeError(f"expert left no gradient on cache layer {m}")
                a = (k * gk).abs().sum((0, 1, 3)) + (v * gv).abs().sum((0, 1, 3))  # (T,)
                d[m] = by_type(a, type_idx).cpu().numpy()
                en = gk.float().pow(2).sum((0, 1, 3)) + gv.float().pow(2).sum((0, 1, 3))  # (T,)
                e[m] = by_type(en, type_idx).cpu().numpy()
        return d, e

    direct, cg_fm = cache_read(leaves, grads)

    # the shipped operation itself: ONE backward seeded with the whole cache gradient. D1 and
    # the residual gradient are read off this backward, so the token-type split (P1, P2) is an
    # exact split of the shipped score. D2/D3 below re-run it with partial seeds; the VLM
    # backward is a deep bf16 graph, so those are linear in the seed only up to rounding,
    # and every clip records how far their sums land from this one
    pl.vlm_backward_from_cache(cache_t, grads, retain=True)
    full_q, full_mlp = read()  # (5, L, U) by the token type the unit acts on
    res_fm, res_dot, res_cos, _ = residual(ce_cpu)
    del ce_cpu
    res_on[0] = False

    rec = {"coc_len": int(coc_end - coc_start), "prompt_len": int(prompt_len),
           "n_tok": {k: int(n) for k, n in zip(TYPES, n_tok)}, "fm_loss": float(fm_loss),
           "nll": nll_val}
    out = {"ce_q": ce_q, "full_q": full_q, "direct": direct,
           "res_ce": res_ce, "res_fm": res_fm, "res_dot": res_dot, "res_cos": res_cos}
    if not args.verify:
        out.update({"ce_mlp": ce_mlp, "full_mlp": full_mlp})

    if args.probes:
        _, pgrads, pleaves = pl.expert_fm_grads(
            model, cache, rope_deltas, x1, args.fm_steps, seed, prefill, readout=seed + 104729
        )
        direct_pr, cg_pr = cache_read(pleaves, pgrads)
        pl.vlm_backward_from_cache(cache_t, pgrads, retain=True)
        probe["expert"] = read()
        del pgrads, pleaves
        gen = torch.Generator(device="cuda").manual_seed(seed + 15485863)
        seeds = [(torch.randn(k.shape, generator=gen, device=k.device, dtype=k.dtype),
                  torch.randn(v.shape, generator=gen, device=v.device, dtype=v.dtype))
                 for k, v in cache_t]  # (1, KV, T, D) each
        pl.vlm_backward_from_cache(cache_t, seeds, retain=False)
        probe["cache"] = read()
        del seeds
        for name, (q, mlp) in probe.items():
            out[f"pr_{name}_q"], out[f"pr_{name}_mlp"] = q, mlp  # (5, L, H), (5, L, I)
        out.update({"direct_probe": direct_pr, "cg_fm": cg_fm, "cg_probe": cg_pr})
        rec["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        gates.remove()
        hs.clear()
        del cache, cache_t, leaves, grads, inputs
        return out, rec

    band_q, band_mlp = [], []
    for lo, hi in BANDS:
        seeds = [g if lo <= m <= hi else (None, None) for m, g in enumerate(grads)]
        pl.vlm_backward_from_cache(cache_t, seeds, retain=True)
        q, mlp = read()
        band_q.append(q)
        band_mlp.append(mlp)

    pos_q, pos_mlp = [], []
    for t in range(len(TYPES)):
        keep = (type_idx == t).view(1, 1, -1, 1)
        seeds = [(gk * keep, gv * keep) for gk, gv in grads]  # (1, KV, T, D) each
        pl.vlm_backward_from_cache(cache_t, seeds, retain=t < len(TYPES) - 1)
        q, mlp = read()
        pos_q.append(q)
        pos_mlp.append(mlp)
        del seeds

    band_q, pos_q = np.stack(band_q), np.stack(pos_q)  # (6, 5, L, H), (5, 5, L, H)
    peak = torch.cuda.max_memory_allocated() / 1024**3
    gates.remove()
    if ref is not None:
        ref.remove()
    out.update({"band_q": band_q, "pos_q": pos_q})
    if not args.verify:
        out.update({"band_mlp": np.stack(band_mlp),  # (6, 5, L, I)
                    "pos_mlp": np.stack(pos_mlp)})  # (5, 5, L, I)
    # how far each seed split's total lands from the single full-seed backward (Q heads)
    tot, bsum, psum = full_q.sum(0), band_q.sum((0, 1)), pos_q.sum((0, 1))  # (L, H)

    def per_layer(a):
        return (np.linalg.norm(a - tot, axis=1)
                / np.maximum(np.linalg.norm(tot, axis=1), 1e-300))[:35]  # layer 35 is 0/0

    def rank_min(a):
        return float(min(spearmanr(np.abs(a[l]), np.abs(tot[l]))[0] for l in range(35)))

    rec.update({"peak_gb": round(peak, 2),
                "bands_vs_full_relerr_by_layer": per_layer(bsum).tolist(),
                "pos_vs_full_relerr_by_layer": per_layer(psum).tolist(),
                "bands_vs_full_rank_rho_min": rank_min(bsum),
                "pos_vs_full_rank_rho_min": rank_min(psum)})
    if args.verify:
        rec["typed_vs_single_relerr"] = verify  # [CE, full seed, 6 bands, 5 position types]
    hs.clear()
    del cache, cache_t, leaves, grads, inputs
    return out, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-clips", type=int, default=100)
    ap.add_argument("--exp-id", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--calib-manifest", default="calib_100")
    ap.add_argument("--cache", default="calib")
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--fm-steps", type=int, default=10)
    ap.add_argument("--reserve-gb", type=float, default=44.0)
    ap.add_argument("--gpu", type=str, default=None)
    ap.add_argument("--verify", action="store_true",
                    help="G0: shipped UnitGates next to Q-head-only typed gates; checks the "
                         "typed grads sum to the single-gate grad on every backward. No MLP "
                         "arrays are produced in this mode")
    ap.add_argument("--mask", type=str, default=None,
                    help="npz with q_mask/mlp_mask (L,H)/(L,I) 0-1 keep masks (the key convention "
                         "of run_importance --mask): the anatomy of that pruned model, teacher-"
                         "forced on the dense model's rollout")
    ap.add_argument("--probes", action="store_true",
                    help="random-readout probes through the head, the expert and the bare cache "
                         "in place of the band and position splits")
    args = ap.parse_args()
    if args.verify and (args.mask or args.probes):
        ap.error("--verify checks the shipped dense pass; it does not combine with --mask/--probes")
    mask = None
    if args.mask:
        z = np.load(args.mask)
        mask = (z["q_mask"].astype(np.float32), z["mlp_mask"].astype(np.float32))  # (L, H), (L, I)
        print(f"mask {args.mask}: q keep {mask[0].mean():.4f}, mlp keep {mask[1].mean():.4f}",
              flush=True)

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    calib = sc.calib_samples(REPO, args.calib_manifest)[: args.num_clips]

    devices = None if args.gpu is None else [int(x) for x in args.gpu.split(",")]
    device = reserve_gpu(args.reserve_gb, devices=devices)
    print(f"using {device}", flush=True)

    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")
    # the gates sit downstream of k/v, so without this the early layers' cache tensors
    # carry no autograd graph at all
    model.vlm.enable_input_require_grads()

    tc = model.vlm.config.text_config
    L, H, I = tc.num_hidden_layers, tc.num_attention_heads, tc.intermediate_size
    nT, nB = len(TYPES), len(BANDS)
    axes = ("q",) if args.verify else ("q", "mlp")
    U = {"q": H, "mlp": I}
    # sums over clips of |.|; every *_full / *tot / *_type entry takes the abs AFTER summing
    # the signed parts, the way the shipped single gate does
    acc = {}
    probes = ("head", "expert", "cache") if args.probes else ()
    for a in axes:
        acc.update({
            f"ce_{a}": np.zeros((nT, L, U[a])), f"ce_full_{a}": np.zeros((L, U[a])),
            f"ce_text_{a}": np.zeros((L, U[a])),
            f"fm_type_{a}": np.zeros((nT, L, U[a])), f"fm_text_{a}": np.zeros((L, U[a])),
            f"fm_full_{a}": np.zeros((L, U[a])),
        })
        if not args.probes:
            acc.update({
                f"fm_band_{a}": np.zeros((nB, nT, L, U[a])), f"fm_pos_{a}": np.zeros((nT, nT, L, U[a])),
                f"fm_bandtot_{a}": np.zeros((nB, L, U[a])), f"fm_postot_{a}": np.zeros((nT, L, U[a])),
            })
        for p in probes:
            acc.update({f"pr_{p}_{a}": np.zeros((nT, L, U[a])), f"pr_{p}_full_{a}": np.zeros((L, U[a]))})
    extra_res = ("direct_probe", "cg_fm", "cg_probe") if args.probes else ()
    acc.update({k: np.zeros((L, nT)) for k in ("direct", "res_ce", "res_fm") + extra_res})
    pt, co = TYPES.index("prompt_text"), TYPES.index("coc")
    per_q = {"ce": [], "fm_full": []}
    per_q.update({f"pr_{p}": [] for p in probes} if args.probes else {"fm_band": [], "fm_pos": []})
    per_mlp = {"ce_type": [], "fm_type": []}
    per_res = {k: [] for k in ("res_ce", "res_fm", "res_dot", "res_cos", "direct") + extra_res}

    (out_dir / "config.json").write_text(json.dumps({
        "model": "nvidia/Alpamayo-1.5-10B", "model_revision": MODEL_REV,
        "purpose": "token-type and cache-port decomposition of the shipped gate gradients",
        "plan": ("plans/2026-09-21_importance-causal-validation.md" if args.mask or args.probes
                 else "plans/2026-09-20_gradient-anatomy.md"),
        "objectives": {"coc": "own-rollout CoC NLL", "traj": "flow-matching MSE vs GT action"},
        "mask": args.mask, "probes": list(probes),
        "text": "the dense model's rollout (masks are applied after it)" if args.mask else "own rollout",
        "types": TYPES, "bands": BANDS, "verify": args.verify,
        "num_clips": len(calib), "clip_ids": [c for c, _ in calib], "seed": args.seed,
        "seed_rule": "sha256(f'{seed}:{clip_id}')[:4]",
        "calib_manifest": args.calib_manifest, "cache": args.cache,
        "fm_steps": args.fm_steps, "max_gen": args.max_gen,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
    }, indent=2))

    records = []
    for ci, (clip_id, clip_t0) in enumerate(calib):
        t0 = time.time()
        data = sc.load_cached(sc.path_for(args.cache, clip_id, clip_t0))
        torch.cuda.reset_peak_memory_stats()
        g, rec = process_clip(model, processor, data, args, sc.clip_seed(args.seed, clip_id), mask)
        for a in axes:
            ce = g[f"ce_{a}"]
            fm_type = g[f"full_{a}"]  # (nT, L, U) the full-seed backward, by unit-side type
            acc[f"ce_{a}"] += np.abs(ce)
            acc[f"ce_full_{a}"] += np.abs(ce.sum(0))
            acc[f"ce_text_{a}"] += np.abs(ce[pt] + ce[co])
            acc[f"fm_type_{a}"] += np.abs(fm_type)
            acc[f"fm_text_{a}"] += np.abs(fm_type[pt] + fm_type[co])
            acc[f"fm_full_{a}"] += np.abs(fm_type.sum(0))
            if not args.probes:
                band, pos = g[f"band_{a}"], g[f"pos_{a}"]
                acc[f"fm_band_{a}"] += np.abs(band)
                acc[f"fm_pos_{a}"] += np.abs(pos)
                acc[f"fm_bandtot_{a}"] += np.abs(band.sum(1))
                acc[f"fm_postot_{a}"] += np.abs(pos.sum(1))
            for p in probes:
                acc[f"pr_{p}_{a}"] += np.abs(g[f"pr_{p}_{a}"])
                acc[f"pr_{p}_full_{a}"] += np.abs(g[f"pr_{p}_{a}"].sum(0))
        for k in ("direct", "res_ce", "res_fm") + extra_res:
            acc[k] += g[k]
        per_q["ce"].append(g["ce_q"].astype(np.float32))
        per_q["fm_full"].append(g["full_q"].astype(np.float32))
        for p in probes:
            per_q[f"pr_{p}"].append(g[f"pr_{p}_q"].astype(np.float32))
        if not args.probes:
            per_q["fm_band"].append(g["band_q"].astype(np.float32))
            per_q["fm_pos"].append(g["pos_q"].astype(np.float32))
        if not args.verify:
            # signed fp32, (5, 36, 12288) each: 17.7 MB per clip for the pair. The band and
            # position splits of the MLP axis are kept as accumulated means only
            per_mlp["ce_type"].append(g["ce_mlp"].astype(np.float32))
            per_mlp["fm_type"].append(g["full_mlp"].astype(np.float32))
        for k, v in per_res.items():
            v.append(g[k])
        rec["clip_id"] = clip_id
        records.append(rec)
        del g
        extra = (f" typed-vs-single {max(rec['typed_vs_single_relerr']):.1e}"
                 if args.verify else "")
        split = "" if args.probes else (
            f"bands-vs-full {max(rec['bands_vs_full_relerr_by_layer']):.1e} (median layer "
            f"{np.median(rec['bands_vs_full_relerr_by_layer']):.1e}, rho min "
            f"{rec['bands_vs_full_rank_rho_min']:.4f}) pos-vs-full "
            f"{max(rec['pos_vs_full_relerr_by_layer']):.1e} (rho min "
            f"{rec['pos_vs_full_rank_rho_min']:.4f}){extra} ")
        print(f"[{ci + 1}/{len(calib)}] {clip_id} coc={rec['coc_len']} fm={rec['fm_loss']:.4f} "
              f"nll={rec['nll']:.4f} {split}"
              f"peak={rec['peak_gb']:.1f}GB ({time.time() - t0:.0f}s)", flush=True)
        if (ci + 1) % 10 == 0 or ci + 1 == len(calib):
            save(out_dir, acc, per_q, per_mlp, per_res, records, ci + 1)
    print("saved ->", out_dir, flush=True)


def save(out_dir, acc, per_q, per_mlp, per_res, records, n):
    np.savez(out_dir / "anatomy.npz", **{k: v / max(n, 1) for k, v in acc.items()})
    np.savez(out_dir / "anatomy_perclip_q.npz",
             **{k: np.stack(v) for k, v in {**per_q, **per_res}.items()})
    if per_mlp["ce_type"]:
        np.savez(out_dir / "anatomy_perclip_mlp.npz", **{k: np.stack(v) for k, v in per_mlp.items()})
    (out_dir / "metrics.json").write_text(json.dumps({"n_clips": n, "per_clip": records}, indent=2))


if __name__ == "__main__":
    main()
