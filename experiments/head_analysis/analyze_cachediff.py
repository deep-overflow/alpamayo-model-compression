"""Gates and maps for run_cachediff.py.

A0  instrument integrity -- layer-0 cache identical, A00/A10 on the published
    baseline / dual within the masked-vs-slim floor
A1  the causal question -- I = (A11 - A10) - (A01 - A00), paired
B   where the cache moved: depth curves, span table, per-KV-group map
B2  attribution -- does the expert cut sit on the groups that moved most
C   correctability -- optimal linear map from the accumulated second moments,
    fitted on one clip parity and scored on the other

Usage:
  .venv/bin/python experiments/head_analysis/analyze_cachediff.py \
      --shards cachediff_v1 --out cachediff_v1
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402

import eval_lib as el  # noqa: E402

BG, INK, MUTED, GRID = "#FAF9F5", "#29261B", "#6B6555", "#E8E6DC"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.size": 9, "axes.grid": True, "axes.axisbelow": True,
})
REPO = Path(__file__).resolve().parents[2]
CELLS = ["A00_denseC_denseE", "A01_denseC_prunE", "A10_prunC_denseE", "A11_prunC_prunE"]
# masked-vs-slim floor, from outputs/slim_verify/: the bf16 reduction-width drift between
# a masked model and the physically slimmed one, which A0(ii) must not be judged tighter than
SLIM_FLOOR = 0.025


def median_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    boots = [np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(n)]
    return float(np.median(d)), *np.percentile(boots, [2.5, 97.5])


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    mean, mlo, mhi = el.paired_bootstrap_ci(d)
    med, lo, hi = median_ci(d)
    try:
        p = float(wilcoxon(d).pvalue)
    except ValueError:
        p = float("nan")
    return {"d": d, "med": med, "lo": lo, "hi": hi, "mean": float(mean),
            "mlo": float(mlo), "mhi": float(mhi), "p": p,
            "sig": bool(lo > 0 or hi < 0)}


def merge(shards):
    met, arrs, n = None, None, 0
    for s in shards:
        d = json.loads((REPO / "outputs" / s / "metrics.json").read_text())
        z = dict(np.load(REPO / "outputs" / s / "cachediff.npz"))
        if met is None:
            met = d
            # div_* were saved as per-clip means, mom_* as raw sums
            arrs = {k: (v * d["n_clips"] if k.startswith("div_") else v)
                    for k, v in z.items()}
        else:
            for c in CELLS:
                met["cells"][c]["ade"] += d["cells"][c]["ade"]
                met["cells"][c]["fde"] += d["cells"][c]["fde"]
            met["cells"]["nll"]["dense"] += d["cells"]["nll"]["dense"]
            met["cells"]["nll"]["pruned"] += d["cells"]["nll"]["pruned"]
            met["cells"]["layer0_max_dk"] += d["cells"]["layer0_max_dk"]
            met["clip_ids"] += d["clip_ids"]
            met["buckets"] += d["buckets"]
            for k, v in z.items():
                arrs[k] = arrs[k] + (v * d["n_clips"] if k.startswith("div_") else v)
        n += d["n_clips"]
    for k in list(arrs):
        if k.startswith("div_"):
            arrs[k] = arrs[k] / n
    return met, arrs, n


def correctability(A, B, C, lam, min_rel=1e-4):
    """Best linear map per (layer, group), scored from second moments only.

    ||K_D - K_P M||^2 = tr(C) - 2 tr(M^T B) + tr(M^T A M);  ||K_D - K_P||^2 = tr(C-2B+A);
    ||K_D||^2 = tr(C). M is fitted on one clip parity and scored on the other, so the
    ratio cannot be an overfit -- and because M = I is always available, a cross-fold
    ratio above 1 means the fitted map does not generalise.

    Layer 0 is identical by construction (the masks sit downstream of its k/v), so its
    raw difference is exactly zero and the ratio is undefined there; `valid` marks the
    (layer, group) cells whose shift is large enough relative to the signal to divide by.
    """
    L, G, D, _ = A.shape[1:]
    raw = np.zeros((2, L, G))
    resid = np.zeros((2, L, G))
    for fit, ev in ((0, 1), (1, 0)):
        for li in range(L):
            for g in range(G):
                a, b = A[fit, li, g], B[fit, li, g]
                M = np.linalg.solve(a + lam * np.trace(a) / D * np.eye(D), b)
                ae, be, ce = A[ev, li, g], B[ev, li, g], C[ev, li, g]
                sig = max(float(np.trace(ce)), 1e-30)
                raw[ev, li, g] = np.sqrt(max(np.trace(ce) - 2 * np.trace(be)
                                             + np.trace(ae), 0) / sig)
                resid[ev, li, g] = np.sqrt(max(np.trace(ce) - 2 * np.trace(M.T @ be)
                                               + np.trace(M.T @ ae @ M), 0) / sig)
    valid = raw > min_rel
    ratio = np.where(valid, resid / np.maximum(raw, 1e-30), np.nan)
    return {"raw_rel": raw, "resid_rel": resid, "ratio": ratio, "valid": valid}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lam", type=float, nargs="+", default=[1e-2, 1e-1, 1.0])
    args = ap.parse_args()

    met, arrs, n = merge(args.shards)
    out_dir = REPO / "outputs" / args.out
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    cells = {c: np.array(met["cells"][c]["ade"], float) for c in CELLS}
    A00, A01, A10, A11 = (cells[c] for c in CELLS)

    # ---- A0 integrity
    l0 = float(np.max(met["cells"]["layer0_max_dk"]))
    a0i = l0 == 0.0
    # ---- A1 the causal question
    simple_dense = paired(A01, A00)     # expert cut on the dense cache
    simple_prun = paired(A11, A10)      # expert cut on the pruned cache
    inter = paired(simple_prun["d"], simple_dense["d"])   # I, per clip
    cache_only = paired(A10, A00)       # what the VLM cut alone costs
    total = paired(A11, A10)            # what the report calls combined - dual

    gates = {
        "A0_layer0_identical": {"max_abs_dk": l0, "pass": a0i},
        "A1_interaction": {k: inter[k] for k in ("med", "lo", "hi", "mean", "mlo", "mhi", "p", "sig")},
        "simple_effect_dense_cache": {k: simple_dense[k] for k in ("med", "mean", "lo", "hi", "sig")},
        "simple_effect_pruned_cache": {k: simple_prun[k] for k in ("med", "mean", "lo", "hi", "sig")},
        "vlm_cut_alone": {k: cache_only[k] for k in ("med", "mean", "lo", "hi", "sig")},
        "combined_minus_dual": {k: total[k] for k in ("med", "mean", "lo", "hi", "sig")},
        "slim_floor_used": SLIM_FLOOR,
    }

    # ---- B maps
    div = {k[len("div_"):]: v for k, v in arrs.items() if k.startswith("div_")}
    L, G = div["all_cos_k"].shape
    # ---- B2 attribution: does the expert cut sit where the cache moved?
    b2 = {}
    try:
        meta_imp = met.get("importance")
        z = dict(np.load(REPO / "outputs" / meta_imp / "importance.npz"))
        # expert Q head h is fed by VLM KV group h // 2 (GQA: group g -> expert heads [2g, 2g+2))
        eq_score = z["traj_exp_q"]                          # (36, 16) higher = keep
        cut_rank = eq_score.argsort(1).argsort(1)           # 0 = first to cut
        per_group = cut_rank.reshape(eq_score.shape[0], G, -1).mean(2).mean(0)  # (G,)
        moved = 1.0 - div["all_cos_k"].mean(0)              # (G,) higher = moved more
        rho, p = spearmanr(per_group, moved)
        b2 = {"spearman_keeprank_vs_move": float(rho), "p": float(p),
              "per_group_move": moved.tolist(), "per_group_keeprank": per_group.tolist()}
    except Exception as e:                                   # noqa: BLE001
        b2 = {"error": repr(e)}

    # ---- C correctability
    corr = {}
    for tag in ("k", "v"):
        A, B, C = (arrs[f"mom_{tag}_{x}"] for x in ("A", "B", "C"))
        for lam in args.lam:
            r = correctability(A, B, C, lam)
            with np.errstate(invalid="ignore"):
                per_layer = np.nanmean(np.where(r["valid"], r["ratio"], np.nan), axis=(0, 2))
            corr[f"{tag}_lam{lam:g}"] = {
                "mean_residual_ratio": float(np.nanmean(r["ratio"])),
                "raw_rel_mean": float(r["raw_rel"][r["valid"]].mean()),
                "resid_rel_mean": float(r["resid_rel"][r["valid"]].mean()),
                "n_valid_cells": int(r["valid"].sum()), "n_cells": int(r["valid"].size),
                "per_layer": [None if np.isnan(x) else float(x) for x in per_layer]}

    metrics = {"n_clips": n, "clip_ids": met["clip_ids"], "shards": args.shards,
               "cells_mean": {c: float(v.mean()) for c, v in cells.items()},
               "nll": {k: float(np.mean(v)) for k, v in met["cells"]["nll"].items()},
               "gates": gates, "b2": b2, "correctability": corr,
               "divergence": {k: v.tolist() for k, v in div.items()}}
    # never "metrics.json": --out may be a shard dir, and that file is the
    # collector's input (racfit uses the same *_analysis.json convention)
    (out_dir / "metrics_analysis.json").write_text(json.dumps(metrics, indent=2))

    # ---- plots
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    depth = np.arange(L)
    ax[0].plot(depth, 1 - div["all_cos_k"].mean(1), color=C1, label="K (1-cos)")
    ax[0].plot(depth, div["all_rel_v"].mean(1), color=C2, label="V (rel L2)")
    ax[0].set_xlabel("VLM layer"); ax[0].set_ylabel("divergence"); ax[0].legend()
    ax[0].set_title("cache shift grows with depth")
    for s, c in (("vision", C1), ("text", C2), ("hist", C3), ("sink", C4), ("coc", MUTED)):
        if f"{s}_cos_k" in div:
            ax[1].plot(depth, 1 - div[f"{s}_cos_k"].mean(1), color=c, label=s)
    ax[1].set_xlabel("VLM layer"); ax[1].set_ylabel("1 - cos(K)"); ax[1].legend(fontsize=7)
    ax[1].set_title("by token span")
    im = ax[2].imshow(1 - div["all_cos_k"], aspect="auto", cmap="magma")
    ax[2].set_xlabel("KV group"); ax[2].set_ylabel("layer"); ax[2].set_title("1 - cos(K)")
    fig.colorbar(im, ax=ax[2])
    fig.tight_layout(); fig.savefig(out_dir / "plots" / "cachediff_map.png", dpi=150)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    labels = ["expert cut\non dense cache", "expert cut\non pruned cache", "interaction I"]
    vals = [simple_dense["med"], simple_prun["med"], inter["med"]]
    errs = [[v["med"] - v["lo"] for v in (simple_dense, simple_prun, inter)],
            [v["hi"] - v["med"] for v in (simple_dense, simple_prun, inter)]]
    ax[0].bar(labels, vals, color=[C1, C2, C3], yerr=errs, capsize=4)
    ax[0].axhline(0, color=MUTED, lw=1); ax[0].set_ylabel("paired median dminADE@6")
    ax[0].set_title("A1 -- is the cost extra on the pruned cache?")
    for tag, c in (("k", C1), ("v", C2)):
        key = f"{tag}_lam{args.lam[0]:g}"
        y = np.array([np.nan if v is None else v for v in corr[key]["per_layer"]], float)
        ax[1].plot(depth, y, color=c, label=f"{tag.upper()}")
    ax[1].axhline(0.5, color=MUTED, ls="--", lw=1)
    ax[1].set_xlabel("VLM layer"); ax[1].set_ylabel("residual after linear map")
    ax[1].set_ylim(0, 1.05); ax[1].legend(); ax[1].set_title("C -- correctable share")
    fig.tight_layout(); fig.savefig(out_dir / "plots" / "cachediff_gates.png", dpi=150)

    lines = [f"cache-diff diagnostic -- {n} clips, {len(args.shards)} shard(s)", "",
             f"A0  layer-0 cache identical: {a0i} (max |dK| = {l0:.3e})",
             f"    A00 {A00.mean():.4f}  A10 {A10.mean():.4f}  "
             f"A01 {A01.mean():.4f}  A11 {A11.mean():.4f}   (minADE@6 mean)", "",
             "A1  paired medians [95% CI]",
             f"    expert cut | dense cache   {simple_dense['med']:+.4f} "
             f"[{simple_dense['lo']:+.4f},{simple_dense['hi']:+.4f}]",
             f"    expert cut | pruned cache  {simple_prun['med']:+.4f} "
             f"[{simple_prun['lo']:+.4f},{simple_prun['hi']:+.4f}]",
             f"    interaction I              {inter['med']:+.4f} "
             f"[{inter['lo']:+.4f},{inter['hi']:+.4f}]  sig={inter['sig']}",
             f"    VLM cut alone (A10-A00)    {cache_only['med']:+.4f}", "",
             "C   residual after the best linear cache map (fit/score on disjoint folds)"]
    for k, v in corr.items():
        lines.append(f"    {k:10s} ratio {v['mean_residual_ratio']:.3f}  "
                     f"(raw {v['raw_rel_mean']:.3f} -> {v['resid_rel_mean']:.3f} of |K_D|; "
                     f"{v['n_valid_cells']}/{v['n_cells']} cells)")
    if b2 and "spearman_keeprank_vs_move" in b2:
        lines += ["", f"B2  Spearman(expert keep-rank, group cache move) = "
                      f"{b2['spearman_keeprank_vs_move']:+.3f} (p={b2['p']:.3f})"]
    (out_dir / "cachediff_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("saved ->", out_dir)


if __name__ == "__main__":
    main()
