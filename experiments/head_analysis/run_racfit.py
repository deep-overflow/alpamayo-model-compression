"""Per-layer output-preservation limits under CoT reconstruction (RAC).

plans/2026-08-25_cot-reconstruction.md. Two questions in one pass:

  (a) how far can each VLM layer be pruned on the o_proj / down_proj input axes
      before its OUTPUT stops being reconstructable?  Nobody has measured this
      here -- tyr_lib.prune_levels computes the OSSCAR objective and throws it
      away (tyr_lib.py:61), so no error-vs-sparsity curve exists on disk.
  (b) does the answer depend on WHICH tokens the reconstruction is fitted to?
      arXiv:2509.12464 (RAC) says standard pruning reconstructs on prompt
      activations while reasoning is decode-dominated, and that the model's own
      on-policy CoT activations belong in the calibration set.  Alpamayo's Tyr
      baseline is exactly the input-only case: run_tyr_supernet.py:142 records
      "hessian_tokens": "full fused prompt prefill, no labels".

Three token streams are accumulated separately, because H = sum_t x_t x_t^T is
additive and any mixture can then be formed without another forward:

  V  vision + traj-history + sink positions   ~2929 tok/clip  (94.9%)
  T  prompt text positions                    ~ 157 tok/clip  ( 5.1%)
  D  the model's OWN rolled-out CoC           ~  15 tok/clip  ( 0.5%)   x K seeds

That 0.5% is the whole design problem: the paper's plain concatenation
H = H_P + H_D moves Alpamayo's Hessian by half a percent, i.e. a no-op.  So the
mixture is token-count normalised, H(w) = sum_s w_s H_s / N_s, and the sweep
multiplies D's natural share by {0, 1, 10, 100, 1000, inf}:

  VT      D dropped entirely -- EXACTLY the shipped Tyr Hessian (prefill only)
  nat     D at its natural 0.5% share -- the paper's recipe taken literally
  d10..   D up-weighted 10x / 100x / 1000x, the adaptation this run is testing
  Donly   the text-only corner, known-bad on conditioning (tiny_eigs 11825/12288,
          plans/2026-08-20_tyr-baseline.md:110-112); kept as a diagnostic

so `nat` - `VT` measures what naive concatenation buys in Alpamayo (predicted:
nothing) and `d10` - `VT` measures what the normalised adaptation buys.

Errors are held-out: clips are split 50/50 by hash into folds A/B, H is fitted on
one fold and the relative output error read on the other, per eval stream.  Tyr
had fit == eval and so could not see this.  Only the Hessians are needed for
that -- rel_err = sqrt(tr(D H D^T) / tr(W H W^T)) -- so no activations are stored.

Pre-registered gates (restated from the plan):
  G0  `VT`, damp 1e-2, layer 0, --folds 1 must reproduce slim_tyr_uniform_u40_recon's
      layer-0 reconstruction to a relative Frobenius difference < 1e-3.  (The plan
      wrote `nat` here; `VT` is the mixture that is literally Tyr's prefill-only H,
      and layer 0 is before any error accumulation, so it is the exact reference.)
  G1  premise: Spearman(diag H_V, diag H_D) < 0.95 AND top-512 eigenspace energy
      overlap < 0.9.  If both streams are the same thing, RAC has no room here.
  G2  main: err_D(nat) - err_D(d10) at the u40 level, paired bootstrap over the 36
      layers, 95% CI excluding 0 and a median improvement > 2%.
  G3  kept-set overlap between mixtures below the calibration noise floor
      (Q 0.860, MLP 0.782).
  G4  cost: err_V(d10) - err_V(nat) with CI upper bound < +2%.
  G5  the per-layer r*(eps=0.10) profile has std > 0.05 across layers.
G2 names d10 by pre-registration; the whole D-multiplier sweep is reported
alongside and locating its minimum is exploratory, not a gate.

Usage:
  # 1. rollout cache once (K seeds per clip), then reuse it in every shard
  bash experiments/head_analysis/run_retry_host.sh 30 \
      experiments/head_analysis/run_racfit.py --gpu 0 --exp-id racfit_v1 --rollout-only
  # 2. shard the 36 layers over the free cards
  bash experiments/head_analysis/run_retry_host.sh 240 \
      experiments/head_analysis/run_racfit.py --gpu 0 --exp-id racfit_v1_l00 \
      --rollouts-from racfit_v1 --layer-start 0 --layer-end 9
  # smoke (~5 min)
  bash experiments/head_analysis/run_retry_host.sh 5 \
      experiments/head_analysis/run_racfit.py --gpu 0 --exp-id racfit_smoke \
      --num-clips 2 --k-seeds 1 --layer-end 2 --keeps-q 32 19 10 \
      --keeps-m 12288 7390 2048
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
import sample_cache as sc
import tyr_lib as tyr
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from expert_per_clip import reserve_gpu  # also installs the gated-repo hub patch
from slim_lib import MODEL_REV

REPO = Path(__file__).resolve().parents[2]

STREAMS = ("V", "T", "D")
FOLDS = ("A", "B")
FOLD_SEED = 20260825          # fold membership is a clip-id hash, never the loop index
# D's natural share (~0.5%) multiplied by this; None = D only.
MIX_MULT = {"VT": 0.0, "nat": 1.0, "d10": 10.0, "d100": 100.0, "d1000": 1000.0,
            "Donly": None}
DEFAULT_KEEPS_Q = [32, 29, 26, 23, 19, 16, 13, 10]
DEFAULT_KEEPS_M = [12288, 11264, 10240, 9216, 7390, 6144, 4096, 2048]
U40_KEEP_Q, U40_KEEP_M = 19, 7390       # the u40_v2 budget: cut 13/32 heads, 4898/12288 ch


def key(fold, stream):
    return f"{fold}/{stream}"


def mix_weights(name, n_fold):
    """Per-stream weights on the token-normalised Hessians for one mixture."""
    if MIX_MULT[name] is None:
        return {"V": 0.0, "T": 0.0, "D": 1.0}
    total = sum(n_fold.values())
    if total == 0:
        return {s: 0.0 for s in STREAMS}
    frac = {s: n_fold[s] / total for s in STREAMS}
    return {"V": frac["V"], "T": frac["T"], "D": MIX_MULT[name] * frac["D"]}


def spearman(a, b):
    """Rank correlation without a scipy dependency (no ties expected on diag H)."""
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def preload_inputs(model, processor, calib, cache_name):
    """Fused inputs once per clip, kept on CPU (pixel tensors are large).

    Same shape as run_tyr_supernet.preload_inputs, plus the token spans so the
    per-stream masks do not have to be recomputed on every chunk pass.
    """
    store = []
    for clip_id, t0 in calib:
        data = sc.load_cached(sc.path_for(cache_name, clip_id, t0))
        inp = lib.build_inputs(model, processor, data, "cuda")
        spans = lib.compute_spans(model, inp["input_ids"])
        store.append({
            "clip_id": clip_id,
            "input_ids": inp["input_ids"].cpu(),                       # (1, T_prompt)
            "pixel_values": inp["tokenized_data"]["pixel_values"].cpu(),
            "image_grid_thw": inp["tokenized_data"]["image_grid_thw"].cpu(),
            "vis": (spans["vision"] | spans["hist"] | spans["sink"]),   # (T_prompt,)
            "txt": spans["text"],                                       # (T_prompt,)
        })
    return store


def build_rollouts(model, store, k_seeds, seed, max_gen, out_path):
    """K on-policy CoC rollouts per clip; only the token ids are kept.

    Seed j uses clip_seed(seed + 1000*j, clip_id), so j=0 reproduces the seed the
    rest of the harness uses (run_wanda, tyr_teacher) and can be cross-checked.
    """
    rec = {"seed": seed, "k_seeds": k_seeds, "max_gen": max_gen, "clips": {}}
    for ci, item in enumerate(store):
        t0 = time.time()
        inputs = {
            "input_ids": item["input_ids"].cuda(),
            "tokenized_data": {
                "attention_mask": torch.ones_like(item["input_ids"]).cuda(),
                "pixel_values": item["pixel_values"].cuda(),
                "image_grid_thw": item["image_grid_thw"].cuda(),
            },
        }
        prompt_len = item["input_ids"].shape[1]
        cocs = []
        for j in range(k_seeds):
            s = sc.clip_seed(seed + 1000 * j, item["clip_id"])
            torch.manual_seed(s)
            torch.cuda.manual_seed_all(s)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                roll = lib.run_rollout(model, inputs, max_generation_length=max_gen)
            eos = roll["eos_pos"]
            cocs.append(roll["sequences"][0, prompt_len:eos].tolist())
            del roll
        rec["clips"][item["clip_id"]] = {"prompt_len": prompt_len, "coc": cocs}
        print(f"[roll {ci + 1}/{len(store)}] {item['clip_id']} prompt={prompt_len} "
              f"coc={[len(c) for c in cocs]} ({time.time() - t0:.0f}s)", flush=True)
    out_path.write_text(json.dumps(rec))
    return rec


def forward_pass(model, item, coc_ids, masks_for):
    """One teacher-forced forward over [prompt; CoC] with the stream masks installed."""
    ids = item["input_ids"]
    if len(coc_ids):
        ids = torch.cat([ids, torch.tensor(coc_ids, dtype=ids.dtype)[None, :]], dim=1)
    ids = ids.cuda()                                          # (1, T_prompt + T_coc)
    masks_for(ids.shape[1])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.vlm.model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            pixel_values=item["pixel_values"].cuda(),
            image_grid_thw=item["image_grid_thw"].cuda(),
            use_cache=False,
        )


def solve_module(mod, hook, n_groups, group_size, update_iter, keeps, mixes,
                 fold_pairs, damp, dual_keep=None, dump=None):
    """Sweep mixtures x folds x keep levels for one module.

    Returns dict of arrays indexed [mix, fold_pair, level] plus the kept sets, and
    the `dual`-selection control (reconstruction only, u40 level) when dual_keep
    is given as a kept-group index array. `dump` = (path, mix_name, u40_keep)
    writes that one solution to disk for the G0 comparison.
    """
    W = mod.weight.data.float()                                # (out, in)
    denom = {}
    for f in FOLDS:
        for s in STREAMS:
            k = key(f, s)
            denom[k] = tyr.dense_energy(W, hook.H[k]) if hook.n[k] > 0 else None

    n_mix, n_fp, n_lv, n_st = len(mixes), len(fold_pairs), len(keeps), len(STREAMS)
    err_r = np.full((n_mix, n_fp, n_lv, n_st), np.nan, dtype=np.float32)
    err_m = np.full((n_mix, n_fp, n_lv, n_st), np.nan, dtype=np.float32)
    kept_mask = np.zeros((n_mix, n_fp, n_lv, n_groups), dtype=bool)
    err_dual = np.full((n_mix, n_fp, n_st), np.nan, dtype=np.float32)

    for mi, mix in enumerate(mixes):
        for pi, (fit_f, ev_f) in enumerate(fold_pairs):
            n_fit = {s: hook.n[key(fit_f, s)] for s in STREAMS}
            H_fit = tyr.mix_hessians({s: hook.H[key(fit_f, s)] for s in STREAMS},
                                     n_fit, mix_weights(mix, n_fit))
            if H_fit is None:
                continue
            sols = tyr.prune_levels(mod, H_fit, n_groups, keeps, update_iter, damp=damp)
            for li, keep in enumerate(keeps):
                W_hat = sols[keep]
                kept = tyr.kept_groups(W_hat, n_groups)
                kept_mask[mi, pi, li, kept.cpu().numpy()] = True
                W_msk = tyr.mask_only(W, kept, group_size=group_size)
                if dump is not None and mix == dump[1] and keep == dump[2] and pi == 0:
                    torch.save(W_hat.to(torch.bfloat16).cpu(), dump[0])
                for si, s in enumerate(STREAMS):
                    k = key(ev_f, s)
                    if denom[k] is None:
                        continue
                    err_r[mi, pi, li, si] = tyr.recon_error(W, W_hat, hook.H[k], denom[k])
                    err_m[mi, pi, li, si] = tyr.recon_error(W, W_msk, hook.H[k], denom[k])
                del W_hat
            del sols
            if dual_keep is not None:
                cols = [int(g) * group_size + d for g in dual_keep
                        for d in range(group_size)]
                dsol = tyr.reconstruct_levels(mod, H_fit, {"dual": cols}, damp=damp)["dual"]
                for si, s in enumerate(STREAMS):
                    k = key(ev_f, s)
                    if denom[k] is not None:
                        err_dual[mi, pi, si] = tyr.recon_error(W, dsol, hook.H[k], denom[k])
                del dsol
            del H_fit
            torch.cuda.empty_cache()
    return {"err_recon": err_r, "err_mask": err_m, "kept": kept_mask, "err_dual": err_dual}


def stream_diagnostics(hook, overlap_k):
    """G1 inputs plus conditioning, on the fold-A Hessians."""
    hv, hd = hook.H[key("A", "V")], hook.H[key("A", "D")]
    out = {}
    if hook.n[key("A", "V")] > 0 and hook.n[key("A", "D")] > 0:
        dv = torch.diagonal(hv).cpu().numpy()
        dd = torch.diagonal(hd).cpu().numpy()
        out["spearman_VD"] = spearman(dv, dd)
        out["overlap_VD"] = tyr.energy_overlap(hv, hd, k=min(overlap_k, hv.shape[0] // 2))
    for s in STREAMS:
        k = key("A", s)
        if hook.n[k] > 0:
            cs = tyr.cond_stats(hook.H[k])
            out[f"pr_rank_{s}"] = cs["pr_rank"]
            out[f"trace_{s}"] = cs["trace"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="racfit_v1")
    ap.add_argument("--num-clips", type=int, default=100)
    ap.add_argument("--calib-manifest", default="calib_100")
    ap.add_argument("--cache", default="calib")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-seeds", type=int, default=4,
                    help="on-policy CoC rollouts per clip feeding the D stream")
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--rollout-only", action="store_true",
                    help="build rollouts.json and exit (shards then share it)")
    ap.add_argument("--rollouts-from", default=None,
                    help="reuse outputs/<id>/rollouts.json instead of generating")
    ap.add_argument("--layer-start", type=int, default=0)
    ap.add_argument("--layer-end", type=int, default=None)
    ap.add_argument("--layer-chunk", type=int, default=3,
                    help="layers hooked per calibration pass; 3 keeps peak VRAM ~38 GB")
    ap.add_argument("--mixes", nargs="+", default=list(MIX_MULT))
    ap.add_argument("--keeps-q", type=int, nargs="+", default=DEFAULT_KEEPS_Q)
    ap.add_argument("--keeps-m", type=int, nargs="+", default=DEFAULT_KEEPS_M)
    ap.add_argument("--damp", type=float, default=1e-2,
                    help="Hessian damping x mean diag; 1e-2 is Tyr's final verdict")
    ap.add_argument("--folds", type=int, default=2, choices=(1, 2),
                    help="2 = held-out (A fit/B eval and back); 1 = fit==eval, for G0")
    ap.add_argument("--overlap-k", type=int, default=512)
    ap.add_argument("--no-dual", action="store_true", help="skip the dual-selection control")
    ap.add_argument("--dump-u40-mix", default=None,
                    help="save this mixture's u40-level solutions to <exp>/sol/ (gate G0)")
    ap.add_argument("--importance", default="importance_v2")
    ap.add_argument("--reserve-gb", type=float, default=44.0)
    ap.add_argument("--gpu", type=str, default=None)
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
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

    tc = model.vlm.config.text_config
    layers = model.vlm.model.language_model.layers
    n_layers, n_heads = tc.num_hidden_layers, tc.num_attention_heads
    head_dim, inter = tc.head_dim, tc.intermediate_size
    lo = args.layer_start
    hi = min(args.layer_end if args.layer_end is not None else n_layers, n_layers)

    print(f"preloading {len(calib)} clip inputs...", flush=True)
    store = preload_inputs(model, processor, calib, args.cache)

    roll_path = out_dir / "rollouts.json"
    if args.rollouts_from:
        src = REPO / "outputs" / args.rollouts_from / "rollouts.json"
        rolls = json.loads(src.read_text())
        print(f"rollouts from {src} ({len(rolls['clips'])} clips)", flush=True)
    elif roll_path.exists():
        rolls = json.loads(roll_path.read_text())
        print(f"rollouts resumed from {roll_path} ({len(rolls['clips'])} clips)", flush=True)
    else:
        rolls = build_rollouts(model, store, args.k_seeds, args.seed, args.max_gen,
                               roll_path)
    if args.rollout_only:
        print("rollout-only: done ->", roll_path, flush=True)
        return

    k_seeds = min(args.k_seeds, rolls["k_seeds"])
    fold_of = {it["clip_id"]: FOLDS[sc.clip_seed(FOLD_SEED, it["clip_id"]) % 2]
               for it in store}
    fold_pairs = [("A", "B"), ("B", "A")] if args.folds == 2 else [("A", "A")]
    if args.folds == 1:                       # G0 mode: every clip in one fold
        fold_of = {c: "A" for c in fold_of}
    keys = [key(f, s) for f in FOLDS for s in STREAMS]

    dual_q = dual_m = None
    if not args.no_dual:
        imp = dict(np.load(REPO / "outputs" / args.importance / "importance.npz"))
        sq, sm = tyr.dual_scores(imp)
        dual_q = [np.setdiff1d(np.arange(n_heads), tyr.cut_lowest(sq[i], n_heads - U40_KEEP_Q))
                  for i in range(n_layers)]
        dual_m = [np.setdiff1d(np.arange(inter), tyr.cut_lowest(sm[i], inter - U40_KEEP_M))
                  for i in range(n_layers)]

    (out_dir / "config.json").write_text(json.dumps({
        "purpose": "per-layer output-preservation limit vs calibration token stream (RAC)",
        "plan": "plans/2026-08-25_cot-reconstruction.md",
        "model": "nvidia/Alpamayo-1.5-10B", "model_revision": MODEL_REV,
        "num_clips": len(calib), "clip_ids": [c for c, _ in calib],
        "calib_manifest": args.calib_manifest, "cache": args.cache,
        "seed": args.seed, "seed_rule": "sha256(f'{seed}:{clip_id}')[:4]",
        "fold_seed": FOLD_SEED, "folds": args.folds,
        "k_seeds": k_seeds, "max_gen": args.max_gen,
        "rollouts_from": args.rollouts_from,
        "streams": {"V": "vision + traj-history + sink", "T": "prompt text",
                    "D": "the model's own rolled-out CoC (K seeds)"},
        "mixes": args.mixes, "mix_mult": MIX_MULT,
        "keeps_q": args.keeps_q, "keeps_m": args.keeps_m,
        "u40_keep": {"q": U40_KEEP_Q, "mlp": U40_KEEP_M},
        "damp": args.damp, "layer_range": [lo, hi], "layer_chunk": args.layer_chunk,
        "dual_control": (not args.no_dual), "importance": args.importance,
        "gpu": torch.cuda.get_device_name(device),
    }, indent=2))

    res = {t: {} for t in ("o", "m")}
    diag = {t: {} for t in ("o", "m")}
    tok_counts = {}
    t_all = time.time()
    for c0 in range(lo, hi, args.layer_chunk):
        c1 = min(c0 + args.layer_chunk, hi)
        t0 = time.time()
        hooks = {}
        for li in range(c0, c1):
            hooks[(li, "o")] = tyr.StreamHessianHook(layers[li].self_attn.o_proj, keys)
            hooks[(li, "m")] = tyr.StreamHessianHook(layers[li].mlp.down_proj, keys)

        for item in store:
            f = fold_of[item["clip_id"]]
            r = rolls["clips"][item["clip_id"]]
            for j in range(k_seeds):
                coc = r["coc"][j]
                p_len = r["prompt_len"]

                def masks_for(total, f=f, p_len=p_len, item=item, j=j, hooks=hooks):
                    d = torch.zeros(total, dtype=torch.bool)
                    d[p_len:] = True
                    m = {key(f, "D"): d.cuda()}
                    if j == 0:                     # V/T do not depend on the CoC seed
                        v = torch.zeros(total, dtype=torch.bool)
                        t = torch.zeros(total, dtype=torch.bool)
                        v[:p_len] = item["vis"]
                        t[:p_len] = item["txt"]
                        m[key(f, "V")] = v.cuda()
                        m[key(f, "T")] = t.cuda()
                    for h in hooks.values():
                        h.masks = m

                forward_pass(model, item, coc, masks_for)
        for h in hooks.values():
            h.remove()
        t_fwd = time.time() - t0

        for li in range(c0, c1):
            for tag, mod, n_groups, gs, upd, keeps, dsel in (
                ("o", layers[li].self_attn.o_proj, n_heads, head_dim, 1, args.keeps_q,
                 None if dual_q is None else dual_q[li]),
                ("m", layers[li].mlp.down_proj, inter, 1, 16, args.keeps_m,
                 None if dual_m is None else dual_m[li]),
            ):
                hook = hooks[(li, tag)]
                if tag == "o":
                    tok_counts[li] = dict(hook.n)
                diag[tag][li] = stream_diagnostics(hook, args.overlap_k)
                dump = None
                if args.dump_u40_mix:
                    (out_dir / "sol").mkdir(exist_ok=True)
                    dump = (out_dir / "sol" / f"L{li:02d}_{tag}_{args.dump_u40_mix}.pt",
                            args.dump_u40_mix,
                            U40_KEEP_Q if tag == "o" else U40_KEEP_M)
                res[tag][li] = solve_module(mod, hook, n_groups, gs, upd, keeps,
                                            args.mixes, fold_pairs, args.damp, dsel,
                                            dump)
                hook.free()
                del hooks[(li, tag)]
                torch.cuda.empty_cache()
        print(f"[layers {c0}-{c1 - 1}] fwd {t_fwd:.0f}s total {time.time() - t0:.0f}s",
              flush=True)
        save(out_dir, args, res, diag, tok_counts, lo, hi, fold_pairs, t_all)

    save(out_dir, args, res, diag, tok_counts, lo, hi, fold_pairs, t_all)
    print("saved ->", out_dir, flush=True)


def save(out_dir, args, res, diag, tok_counts, lo, hi, fold_pairs, t_all):
    done = sorted(res["o"])
    if not done:
        return
    npz = {"layers": np.array(done, dtype=np.int32),
           "keeps_q": np.array(args.keeps_q, dtype=np.int32),
           "keeps_m": np.array(args.keeps_m, dtype=np.int32),
           "mixes": np.array(args.mixes),
           "streams": np.array(STREAMS),
           "fold_pairs": np.array([f"{a}->{b}" for a, b in fold_pairs])}
    for tag in ("o", "m"):
        for field in ("err_recon", "err_mask", "kept", "err_dual"):
            npz[f"{field}_{tag}"] = np.stack([res[tag][li][field] for li in done])
        for k in sorted({k for li in done for k in diag[tag][li]}):
            npz[f"diag_{k}_{tag}"] = np.array(
                [diag[tag][li].get(k, np.nan) for li in done], dtype=np.float32)
    np.savez_compressed(out_dir / "racfit.npz", **npz)
    (out_dir / "metrics.json").write_text(json.dumps({
        "layers_done": done, "layer_range": [lo, hi],
        "tokens_per_key": {str(li): tok_counts[li] for li in sorted(tok_counts)},
        "elapsed_s": round(time.time() - t_all, 1),
    }, indent=2))
    n_tok = tok_counts[done[0]]
    (out_dir / "summary.txt").write_text(
        f"racfit: layers {lo}..{hi - 1} ({len(done)} done), mixes {args.mixes}, "
        f"damp {args.damp}, folds {args.folds}\n"
        f"keeps_q {args.keeps_q}\nkeeps_m {args.keeps_m}\n"
        f"stream tokens (layer {done[0]}): "
        + ", ".join(f"{k}={v}" for k, v in sorted(n_tok.items())) + "\n"
        f"elapsed {time.time() - t_all:.0f}s\n")


if __name__ == "__main__":
    main()
