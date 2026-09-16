"""Latency + memory of pruned arms against the unpruned model, release path and fast path.

plans/2026-09-15_dualexp-latency-memory.md. Reads `profile_stages.py` runs (release inference
path: per-stage CUDA-event medians, rollout/denoise peak memory, KV cache) and
`bench_fastpipeline.py` runs (stock vs graphed fast path on the same clips), and writes the
stage table, the memory table against the bf16 weight arithmetic, the fast-path table and the
attribution ladder (dual - baseline = VLM cut + gather-path cost; em<X> - dual = expert-MLP cut).

Decode is compared per token and end-to-end is normalised to the baseline's median CoC length,
as `analyze_slim.py` does: a pruned model's degenerate long rollouts would otherwise inflate its
decode stage. Paired per-clip ratios (same clip, same seed across arms) give the bootstrap CIs
the gates read; arms are matched on clip_id, so an arm evaluated on a different shard set is
compared only on the clips it shares.

Usage:
  python experiments/head_analysis/analyze_profile_arms.py \
      --arms base=profile_base_ada2_s0,profile_base_ada2_s13 dual=... em75=... em87p5=... em93p75=... \
      --fast base=fastpipe_base_ada2 dual=fastpipe_dual_ada2 ... [--out outputs/profile_dualexp]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C2, C3, C4, C5 = "#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})
BOOT = 10000
BYTES_PER_PARAM = 2  # bf16


def load_profile(names):
    """per-clip records (warmup dropped) + config of the first run."""
    recs, cfg = [], None
    for n in names:
        d = REPO / "outputs" / n
        if not (d / "metrics.json").exists():
            continue
        m = json.loads((d / "metrics.json").read_text())
        recs += [dict(r, run=n) for r in m["per_clip"] if not r.get("warmup")]
        cfg = cfg or json.loads((d / "config.json").read_text())
    return recs, cfg


def stage_medians(recs):
    med = lambda k: float(np.median([r[k] for r in recs]))
    per_tok = float(np.median([r["decode_ms"] / r["decode_steps"] for r in recs]))
    ovh_tok = float(np.median([r["rollout_other_ms"] / r["decode_steps"] for r in recs]))
    return {"n": len(recs), "vit": med("vit_ms"), "prefill": med("prefill_ms"),
            "decode_per_tok": per_tok, "decode_steps": med("decode_steps"),
            "expert": med("expert_ms"), "ovh_per_tok": ovh_tok,
            "denoise_other": med("denoise_other_ms"), "total_wall": med("total_wall_ms"),
            "rollout_peak_gb": med("rollout_peak_gb"), "denoise_peak_gb": med("denoise_peak_gb"),
            "kv_cache_mb": med("kv_cache_mb"), "coc_len": med("coc_len")}


def norm_total(s, ref_steps):
    return (s["vit"] + s["prefill"] + (s["decode_per_tok"] + s["ovh_per_tok"]) * ref_steps
            + s["expert"] + s["denoise_other"])


def paired_ratio(recs_a, recs_b, key, per_step=False):
    """median over shared clips of a/b for one stage, bootstrap CI over clips."""
    a = {r["clip_id"]: r for r in recs_a}
    b = {r["clip_id"]: r for r in recs_b}
    ids = sorted(set(a) & set(b))
    if len(ids) < 3:
        return None
    va = np.array([a[i][key] / (a[i]["decode_steps"] if per_step else 1) for i in ids])
    vb = np.array([b[i][key] / (b[i]["decode_steps"] if per_step else 1) for i in ids])
    ratio = va / vb
    rng = np.random.default_rng(0)
    boot = [np.median(ratio[rng.integers(0, len(ratio), len(ratio))]) for _ in range(BOOT)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n": len(ids), "median": float(np.median(ratio)), "lo": float(lo), "hi": float(hi)}


def load_fast(name):
    d = REPO / "outputs" / name
    if not (d / "metrics.json").exists():
        return None
    m = json.loads((d / "metrics.json").read_text())
    live = [r for r in m["per_clip"] if not r["warmup"] and not r.get("denoise_capture")]
    med = lambda k: float(np.median([r[k] for r in live]))
    stock_tok = float(np.median([r["stock_decode_ms"] / r["stock_steps"] for r in live]))
    fast_tok = float(np.median([r["fast_decode_ms"] / max(r["fast_steps"] - 1, 1) for r in live]))
    ref = m["ref_steps"]
    return {"n": len(live), "ref_steps": ref, "gpu": m.get("gpu"),
            "stock_prefill": med("stock_prefill_ms"), "fast_prefill": med("fast_prefill_ms"),
            "stock_tok": stock_tok, "fast_tok": fast_tok,
            "stock_denoise": med("stock_denoise_ms"), "fast_denoise": med("fast_denoise_ms"),
            "stock_norm": med("stock_prefill_ms") + stock_tok * ref + med("stock_denoise_ms"),
            "fast_norm": med("fast_prefill_ms") + fast_tok * ref + med("fast_denoise_ms"),
            "peak_gb": m.get("peak_gb"), "decode_capture_ms": m.get("decode_capture_ms")}


def parse_map(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = v.split(",")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="name=profile_run[,profile_run]")
    ap.add_argument("--fast", nargs="*", default=[], help="name=fastpipe_run")
    ap.add_argument("--baseline", default="base")
    ap.add_argument("--vlm-only", default="dual", help="arm that isolates the VLM cut")
    ap.add_argument("--out", default="outputs/profile_dualexp")
    args = ap.parse_args()
    out = REPO / args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)

    arms = parse_map(args.arms)
    fast = {k: v[0] for k, v in parse_map(args.fast).items()}
    P, CFG, S = {}, {}, {}
    for name, runs in arms.items():
        recs, cfg = load_profile(runs)
        if recs:
            P[name], CFG[name], S[name] = recs, cfg, stage_medians(recs)
    base = args.baseline
    assert base in S, f"baseline arm {base} has no profile runs"
    ref_steps = S[base]["decode_steps"]
    for s in S.values():
        s["norm_total"] = norm_total(s, ref_steps)
        s["speedup"] = S[base]["norm_total"] / s["norm_total"]

    # ---- memory against the weight arithmetic --------------------------------------------
    mem = {}
    p_base = CFG[base]["param_counts"]["total"]
    for name in S:
        p = CFG[name]["param_counts"]["total"]
        w_gb = p * BYTES_PER_PARAM / 1e9
        pred_drop = (p_base - p) * BYTES_PER_PARAM / 1e9
        meas_drop = S[base]["rollout_peak_gb"] - S[name]["rollout_peak_gb"]
        mem[name] = {"params": p, "params_expert": CFG[name]["param_counts"]["expert"],
                     "weights_gb": w_gb, "rollout_peak_gb": S[name]["rollout_peak_gb"],
                     "denoise_peak_gb": S[name]["denoise_peak_gb"],
                     "kv_cache_mb": S[name]["kv_cache_mb"],
                     "pred_drop_gb": pred_drop, "meas_drop_gb": meas_drop,
                     # rollout_peak_gb is GiB (2**30) while the arithmetic is GB (1e9)
                     "meas_drop_gb_si": meas_drop * 2**30 / 1e9}

    # ---- paired stage ratios (gates G2/G3) ------------------------------------------------
    ratios = {}
    pairs = [(n, base) for n in S if n != base]
    if args.vlm_only in S:
        pairs += [(n, args.vlm_only) for n in S if n not in (base, args.vlm_only)]
    for a, b in pairs:
        ratios[f"{a}/{b}"] = {
            "prefill": paired_ratio(P[a], P[b], "prefill_ms"),
            "decode_per_tok": paired_ratio(P[a], P[b], "decode_ms", per_step=True),
            "expert": paired_ratio(P[a], P[b], "expert_ms"),
            "vit": paired_ratio(P[a], P[b], "vit_ms"),
        }

    F = {k: load_fast(v) for k, v in fast.items()}
    F = {k: v for k, v in F.items() if v}

    # ---- gates ---------------------------------------------------------------------------
    gates = {"G0": {"arms": {n: {"n_clips": S[n]["n"], "gpu": CFG[n]["gpu"],
                                 "params_match_meta": None} for n in S},
                    "fast": {k: v["n"] for k, v in F.items()}}}
    for n in S:
        ck = arms_ckpt = None
        meta = REPO / "outputs" / f"slim_{ {'dual': 'dual_u40_v2'}.get(n, 'dualexp_u40_' + n) }" / "config.json"
        if n != base and meta.exists():
            ck = json.loads(meta.read_text())["params"]["slim"]
            arms_ckpt = ck == CFG[n]["param_counts"]["total"]
        gates["G0"]["arms"][n]["params_match_meta"] = arms_ckpt
    gates["G1"] = {n: {"pred_drop_gb": mem[n]["pred_drop_gb"],
                       "meas_drop_gb_si": mem[n]["meas_drop_gb_si"],
                       "within_1gb": abs(mem[n]["meas_drop_gb_si"] - mem[n]["pred_drop_gb"]) <= 1.0}
                   for n in S if n != base}
    kv = [S[n]["kv_cache_mb"] for n in S]
    gates["G1"]["kv_cache_same"] = bool(max(kv) / max(min(kv), 1e-9) < 1.01)
    lad = [n for n in ("em75", "em87p5", "em93p75") if n in S]
    if args.vlm_only in S and lad:
        ex = {n: ratios[f"{n}/{args.vlm_only}"]["expert"] for n in lad}
        top = ex[lad[-1]]["median"] if ex[lad[-1]] else None
        vals = [v["median"] for v in ex.values() if v]
        spread = (max(vals) - min(vals)) / max(vals) if vals else None
        gates["G2"] = {"expert_ratio_vs_vlm_only": ex, "ladder_spread": spread,
                       "verdict": ("step-bound" if top is not None and top > 0.85 and spread is not None
                                   and spread < 0.10 else
                                   "bandwidth-bound" if top is not None and top < 0.60 else "mixed")}
    if args.vlm_only in S:
        r = ratios[f"{args.vlm_only}/{base}"]
        gates["G3"] = {"prefill": r["prefill"], "decode_per_tok": r["decode_per_tok"],
                       "prefill_in_range": bool(r["prefill"] and 0.70 <= r["prefill"]["median"] <= 0.85),
                       "decode_in_range": bool(r["decode_per_tok"] and 1.00 <= r["decode_per_tok"]["median"] <= 1.15),
                       "e2e_release": {n: S[n]["speedup"] for n in S}}
    if base in F:
        gates["G4"] = {n: {"fast_e2e_ratio_vs_base": F[base]["fast_norm"] / F[n]["fast_norm"],
                           "stock_to_fast_gain": F[n]["stock_norm"] / F[n]["fast_norm"]}
                       for n in F}

    # ---- write ---------------------------------------------------------------------------
    metrics = {"ref_steps": ref_steps, "stages": S, "memory": mem, "ratios": ratios,
               "fast": F, "gates": gates}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "config.json").write_text(json.dumps({"arms": arms, "fast": fast,
                                                 "baseline": base, "vlm_only": args.vlm_only,
                                                 "boot": BOOT}, indent=2))
    L = [(f"== release path ({CFG[base]['gpu']}; decode + loop overhead normalised to "
          f"{ref_steps:.0f} CoC tokens = baseline median) =="),
         (f"{'arm':8s} {'n':>3s} {'ViT':>7s} {'prefill':>8s} {'dec/tok':>8s} {'expert':>8s} "
          f"{'ovh/tok':>8s} {'norm e2e':>9s} {'speedup':>8s} {'roll pk':>8s} {'den pk':>7s} {'KV MB':>7s}")]
    for n, s in S.items():
        L.append(f"{n:8s} {s['n']:3d} {s['vit']:7.1f} {s['prefill']:8.1f} {s['decode_per_tok']:8.2f} "
                 f"{s['expert']:8.1f} {s['ovh_per_tok']:8.2f} {s['norm_total']:9.1f} {s['speedup']:7.3f}x "
                 f"{s['rollout_peak_gb']:8.2f} {s['denoise_peak_gb']:7.2f} {s['kv_cache_mb']:7.1f}")
    L += ["", "== memory vs bf16 weight arithmetic (GB = 1e9 bytes; peaks converted from GiB) =="]
    L.append(f"{'arm':8s} {'params':>14s} {'expert':>12s} {'weights':>8s} {'pred drop':>10s} {'meas drop':>10s}")
    for n, m in mem.items():
        L.append(f"{n:8s} {m['params']:14,d} {m['params_expert']:12,d} {m['weights_gb']:8.2f} "
                 f"{m['pred_drop_gb']:10.2f} {m['meas_drop_gb_si']:10.2f}")
    L += ["", "== paired stage ratios (median over shared clips, 95% bootstrap CI) =="]
    for k, r in ratios.items():
        cells = []
        for st in ("prefill", "decode_per_tok", "expert", "vit"):
            v = r[st]
            cells.append(f"{st} {v['median']:.3f} [{v['lo']:.3f},{v['hi']:.3f}]" if v else f"{st} -")
        L.append(f"  {k:14s} " + "  ".join(cells))
    if F:
        L += ["", (f"== fast path (bench_fastpipeline; stock vs graphed, e2e normalised to "
                   f"{next(iter(F.values()))['ref_steps']} tokens) ==")]
        L.append(f"{'arm':8s} {'n':>3s} {'prefill s/f':>14s} {'dec/tok s/f':>14s} {'denoise s/f':>14s} "
                 f"{'e2e stock':>9s} {'e2e fast':>9s} {'gain':>6s} {'peak GB':>8s}")
        for n, f in F.items():
            pk = f"{f['peak_gb']:8.2f}" if f.get("peak_gb") is not None else f"{'-':>8s}"
            L.append(f"{n:8s} {f['n']:3d} {f['stock_prefill']:6.1f}/{f['fast_prefill']:6.1f} "
                     f"{f['stock_tok']:6.2f}/{f['fast_tok']:6.2f} {f['stock_denoise']:6.1f}/{f['fast_denoise']:6.1f} "
                     f"{f['stock_norm']:9.1f} {f['fast_norm']:9.1f} {f['stock_norm'] / f['fast_norm']:5.2f}x {pk}")
        if base in F:
            L.append("fast-vs-fast e2e speedup vs baseline: " + "  ".join(
                f"{n} {F[base]['fast_norm'] / f['fast_norm']:.3f}x" for n, f in F.items()))
    L += ["", "gates: " + json.dumps(gates, default=str)]
    text = "\n".join(L) + "\n"
    (out / "summary.txt").write_text(text)
    print(text)

    # ---- plots ---------------------------------------------------------------------------
    names = list(S)
    keys = [("vit", "ViT"), ("prefill", "VLM prefill"), ("_dec", "CoC decode (norm)"),
            ("expert", "expert denoise"), ("_ovh", "loop overhead (norm)")]
    vals = {n: {"vit": S[n]["vit"], "prefill": S[n]["prefill"],
                "_dec": S[n]["decode_per_tok"] * ref_steps, "expert": S[n]["expert"],
                "_ovh": S[n]["ovh_per_tok"] * ref_steps + S[n]["denoise_other"]} for n in names}
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bottom = np.zeros(len(names))
    for (k, lab), col in zip(keys, [C4, C1, C2, C3, MUTED]):
        v = np.array([vals[n][k] for n in names])
        ax.bar(names, v, bottom=bottom, color=col, label=lab, width=0.6)
        bottom += v
    for i, n in enumerate(names):
        ax.text(i, bottom[i] + 15, f"{S[n]['speedup']:.2f}x", ha="center", fontsize=9)
    ax.set_ylabel("ms per inference (normalised)")
    ax.set_title(f"release path, {CFG[base]['gpu']}")
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "plots" / "stages_release.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    x = np.arange(len(names))
    ax.bar(x - 0.2, [mem[n]["weights_gb"] for n in names], 0.4, color=MUTED, label="bf16 weights (GB)")
    ax.bar(x + 0.2, [mem[n]["rollout_peak_gb"] * 2**30 / 1e9 for n in names], 0.4, color=C1,
           label="rollout peak allocated (GB)")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("GB")
    ax.set_title("memory: weights vs measured peak")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "plots" / "memory.png", dpi=150)
    plt.close(fig)

    if F:
        fig, ax = plt.subplots(figsize=(7.5, 3.4))
        fn = list(F)
        x = np.arange(len(fn))
        ax.bar(x - 0.2, [F[n]["stock_norm"] for n in fn], 0.4, color=MUTED, label="stock")
        ax.bar(x + 0.2, [F[n]["fast_norm"] for n in fn], 0.4, color=C2, label="graphed fast path")
        ax.set_xticks(x)
        ax.set_xticklabels(fn)
        ax.set_ylabel("ms (e2e, normalised)")
        ax.set_title("fast path bench")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "plots" / "fastpath.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
