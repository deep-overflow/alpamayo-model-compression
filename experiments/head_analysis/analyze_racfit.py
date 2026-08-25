"""Gate verdicts, per-layer curves and a budget-matched allocation from run_racfit.

plans/2026-08-25_cot-reconstruction.md. Reads outputs/<exp>/racfit.npz (several
shards merge on the layer axis) and answers the two questions the run was built for:

  how far can each layer be pruned before its output stops being reconstructable,
  and does that depend on which tokens the reconstruction was fitted to.

Everything is held out: `err` is read on the fold that did not fit the Hessian, so
a --folds 1 run (the G0 reproduction mode) must not be analysed here -- it is
fit == eval and, on a Hessian whose participation-ratio rank is ~40, fits itself
to ~0.  The script refuses such a run.

Reported gates (thresholds pre-registered in the plan):
  G1 premise      median Spearman(diag H_V, diag H_D) < 0.95 and top-k energy
                  overlap < 0.90
  G2 main         err_D(nat) - err_D(d10) at the u40 level, paired bootstrap over
                  layers, CI excluding 0 and median improvement > 2%
  G2b deployed    same against VT, the mixture that IS the shipped Tyr Hessian
  G2c naive       err_D(VT) - err_D(nat): what the paper's literal concatenation
                  buys in Alpamayo, where the CoC is 0.5% of the token mass
  G3 selection    kept-set overlap below the calibration noise floor (Q 0.860,
                  MLP 0.782)
  G4 cost         err_V(d10) - err_V(nat), CI upper bound < +0.02
  G5 allocation   std over layers of r*(eps=0.10) > 0.05

Usage:
  python analyze_racfit.py --exp-id racfit_v1_l00 racfit_v1_l09 ... --out racfit_summary
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
BG, INK, MUTED, GRID = "#FAF9F5", "#29261B", "#6B6555", "#E8E6DC"
COL = {"VT": "#6B6555", "nat": "#2a78d6", "d10": "#1f9d55", "d100": "#eda100",
       "d1000": "#e05252", "Donly": "#e87ba4"}
SLBL = {"V": "vision + history", "T": "prompt text", "D": "own CoC (decode)"}
MODNAME = {"o": "attention o_proj (Q heads)", "m": "MLP down_proj (channels)"}

P_HEAD, P_MLPC = 2 * 4096 * 128, 3 * 4096      # run_grid.py:68
U40_BUDGET = 2_657_452_032                     # the u40_v2 removed-parameter total
U40_KEEP = {"o": 19, "m": 7390}
N_UNITS = {"o": 32, "m": 12288}
P_UNIT = {"o": P_HEAD, "m": P_MLPC}
NOISE_FLOOR = {"o": 0.860, "m": 0.782}         # calib_100 50:50 kept-set overlap floor
EPS = (0.05, 0.10, 0.20)

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})


def load(exp_ids):
    """Merge shards on the layer axis; every shard must share the sweep axes."""
    parts = [dict(np.load(REPO / "outputs" / e / "racfit.npz", allow_pickle=True))
             for e in exp_ids]
    cfg = json.loads((REPO / "outputs" / exp_ids[0] / "config.json").read_text())
    for p in parts[1:]:
        for k in ("mixes", "streams", "keeps_q", "keeps_m", "fold_pairs"):
            assert (p[k] == parts[0][k]).all(), f"shards disagree on {k}"
    order = np.argsort(np.concatenate([p["layers"] for p in parts]))
    out = {k: parts[0][k] for k in ("mixes", "streams", "keeps_q", "keeps_m",
                                    "fold_pairs")}
    for k in parts[0]:
        if k in out:
            continue
        out[k] = np.concatenate([p[k] for p in parts])[order]
    out["mixes"] = [str(x) for x in out["mixes"]]
    out["streams"] = [str(x) for x in out["streams"]]
    return out, cfg


def boot(d, n=10000, seed=20260825):
    """Median of a paired difference with a percentile bootstrap CI over layers."""
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    med = np.median(rng.choice(d, size=(n, d.size), replace=True), axis=1)
    p = wilcoxon(d).pvalue if np.any(d != 0) else float("nan")
    return float(np.median(d)), float(np.percentile(med, 2.5)), \
        float(np.percentile(med, 97.5)), float(p)


def monotone(e):
    """Running max along the removal axis: error can only be re-read as worse."""
    return np.maximum.accumulate(np.nan_to_num(e, nan=np.inf), axis=-1)


def rstar(frac, err, eps):
    """Largest removal fraction whose (monotonised) error is still <= eps."""
    e = np.maximum.accumulate(np.nan_to_num(err, nan=np.inf))
    if e[0] > eps:
        return 0.0
    idx = np.searchsorted(e, eps, side="right")
    if idx >= len(e):
        return float(frac[-1])
    lo, hi = idx - 1, idx
    if e[hi] == e[lo]:
        return float(frac[hi])
    t = (eps - e[lo]) / (e[hi] - e[lo])
    return float(frac[lo] + t * (frac[hi] - frac[lo]))


def waterfill(curves, budget, step):
    """Budget-matched per-layer allocation by Lagrangian bisection.

    curves[tag] is (L, n_levels) error and removed[tag] the matching removed-unit
    counts. The problem is "remove at least `budget` parameters at least total
    error", so each (layer, module) independently picks the x minimising
    err(x) - lam * x * p_unit -- the reward form; the penalty form err + lam*x is
    always minimised at x = 0 because err only grows. Sweeping lam traces each
    curve's convex envelope, and bisecting it to the budget gives the optimal
    separable allocation. Returns removed units per layer per module.
    """
    grids, errs = {}, {}
    for tag, (rem, e) in curves.items():
        g = np.arange(0, int(rem[-1]) + 1, step[tag])
        grids[tag] = g
        errs[tag] = np.stack([np.interp(g, rem, monotone(e)[i]) for i in range(len(e))])

    def total(lam):
        pick = {}
        for tag in curves:
            cost = errs[tag] - lam * grids[tag][None, :] * P_UNIT[tag]
            pick[tag] = grids[tag][np.argmin(cost, axis=1)]
        params = sum(int(pick[t].sum()) * P_UNIT[t] for t in pick)
        return params, pick

    lo, hi = 1e-16, 1e-2          # params(lo) ~ 0, params(hi) = everything
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        params, _ = total(mid)
        if params > budget:
            hi = mid
        else:
            lo = mid
    params, pick = total(lo)
    return pick, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", nargs="+", default=["racfit_v1"])
    ap.add_argument("--out", default=None, help="output dir (default: first --exp-id)")
    ap.add_argument("--primary-mix", default="d10")
    ap.add_argument("--alloc-mix", default=None, help="default: --primary-mix")
    ap.add_argument("--mlp-step", type=int, default=64)
    args = ap.parse_args()

    d, cfg = load(args.exp_id)
    if cfg.get("folds", 2) != 2:
        raise SystemExit("this run is fit==eval (--folds 1); held-out gates need --folds 2")
    out_dir = REPO / "outputs" / (args.out or args.exp_id[0])
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    plots = out_dir / "plots"

    mixes, streams = d["mixes"], d["streams"]
    layers = d["layers"]
    keeps = {"o": d["keeps_q"], "m": d["keeps_m"]}
    removed = {t: (N_UNITS[t] - keeps[t]).astype(float) for t in ("o", "m")}
    frac = {t: removed[t] / N_UNITS[t] for t in ("o", "m")}
    iu40 = {t: int(np.where(keeps[t] == U40_KEEP[t])[0][0]) for t in ("o", "m")}
    im = {m: mixes.index(m) for m in mixes}
    isr = {s: streams.index(s) for s in streams}

    # (L, M, Lv, S), averaged over the two held-out directions A->B and B->A
    err = {t: np.nanmean(d[f"err_recon_{t}"], axis=2) for t in ("o", "m")}
    msk = {t: np.nanmean(d[f"err_mask_{t}"], axis=2) for t in ("o", "m")}
    dual = {t: np.nanmean(d[f"err_dual_{t}"], axis=2) for t in ("o", "m")}
    kept = {t: d[f"kept_{t}"] for t in ("o", "m")}                # (L,M,P,Lv,U)

    L = ["=" * 78, "RAC fit sweep -- per-layer output preservation vs calibration stream",
         "=" * 78,
         (f"experiments {args.exp_id}   layers {layers.min()}..{layers.max()} "
          f"(n={len(layers)})   clips {cfg['num_clips']}   K seeds {cfg['k_seeds']}"),
         f"mixes {mixes}   damp {cfg['damp']}   folds {cfg['folds']} (held out)",
         ""]

    # ---- token mass and CoC diversity -------------------------------------
    tok = json.loads((REPO / "outputs" / args.exp_id[0] / "metrics.json").read_text())
    tk = tok["tokens_per_key"][str(int(layers[0]))]
    tot = sum(tk.values())
    L.append("token mass accumulated (all folds):")
    for s in streams:
        n = tk.get(f"A/{s}", 0) + tk.get(f"B/{s}", 0)
        L.append(f"  {s} {SLBL[s]:<20s} {n:>8d}  {100 * n / tot:5.2f}%")
    # what each mixture actually weights, given the realised token counts
    nat = {s: (tk.get(f"A/{s}", 0) + tk.get(f"B/{s}", 0)) / tot for s in streams}
    L.append("effective stream share of each mixture H(w) = sum_s w_s H_s / N_s:")
    for m in mixes:
        mm = cfg["mix_mult"][m]
        mult = ((0.0, 0.0, 1.0) if mm is None else
                (1.0, 1.0, float(mm)) if isinstance(mm, (int, float)) else tuple(mm))
        w = {s: mult[i] * nat[s] for i, s in enumerate(streams)}
        sw = sum(w.values()) or 1.0
        L.append(f"  {m:<8s}" + "  ".join(f"{s} {100 * w[s] / sw:6.2f}%" for s in streams))
    L.append("")
    roll_p = REPO / "outputs" / (cfg.get("rollouts_from") or args.exp_id[0]) / "rollouts.json"
    if roll_p.exists():
        rj = json.loads(roll_p.read_text())
        uniq = [len({tuple(c) for c in v["coc"]}) for v in rj["clips"].values()]
        L.append(f"  CoC rollouts: K={rj['k_seeds']} seeds/clip, unique sequences per clip "
                 f"mean {np.mean(uniq):.2f} (1.00 = sampling collapsed)")
    L.append("")

    # ---- G1 premise -------------------------------------------------------
    L += ["G1  premise: are the vision and decode streams the same distribution?"]
    g1 = {}
    for t in ("o", "m"):
        sp = np.nanmedian(d[f"diag_spearman_VD_{t}"])
        ov = np.nanmedian(d[f"diag_overlap_VD_{t}"])
        g1[t] = (sp < 0.95) and (ov < 0.90)
        L.append(f"  {MODNAME[t]:<32s} median Spearman(diag) {sp:6.3f} (<0.95), "
                 f"top-k energy overlap {ov:6.3f} (<0.90) "
                 f"-> {'PASS' if g1[t] else 'FAIL'}")
    L.append(f"  VERDICT G1: {'PASS' if all(g1.values()) else 'FAIL'}"
             + ("" if all(g1.values()) else "  -- the streams coincide; RAC has no room here"))
    L.append("")

    # ---- G2 / G2b / G2c ---------------------------------------------------
    def gate_delta(t, a, b, s):
        return err[t][:, im[a], iu40[t], isr[s]] - err[t][:, im[b], iu40[t], isr[s]]

    pm = args.primary_mix
    L += [("G2  main: does adding the decode stream cut held-out decode error? "
           "(u40 level, per layer)")]
    g2 = {}
    for label, a, b, s in (("G2  nat  - " + pm, "nat", pm, "D"),
                           ("G2b VT   - " + pm, "VT", pm, "D"),
                           ("G2c VT   - nat", "VT", "nat", "D")):
        for t in ("o", "m"):
            med, lo, hi, p = boot(gate_delta(t, a, b, s))
            ok = lo > 0 and med > 0.02
            g2[(label, t)] = (med, lo, hi, p, ok)
            L.append(f"  {label:<18s} {MODNAME[t]:<32s} median {med:+.4f} "
                     f"[{lo:+.4f}, {hi:+.4f}] p={p:.2g} -> {'PASS' if ok else 'fail'}")
    g2_pass = all(g2[(k, t)][4] for k in [f"G2  nat  - {pm}"] for t in ("o", "m"))
    L.append(f"  VERDICT G2 (pre-registered, nat vs {pm}): "
             f"{'PASS' if g2_pass else 'FAIL'}")
    L.append("")

    # full mixture x eval-stream table at u40
    L.append(f"held-out rel_err at the u40 level (keep {U40_KEEP['o']}/32 heads, "
             f"{U40_KEEP['m']}/12288 channels), mean over layers:")
    for t in ("o", "m"):
        L.append(f"  {MODNAME[t]}")
        L.append("    fit \\ eval  " + "".join(f"{SLBL[s][:12]:>14s}" for s in streams)
                 + "     (mask-only)")
        for m in mixes:
            row = [np.nanmedian(err[t][:, im[m], iu40[t], isr[s]]) for s in streams]
            mk = np.nanmedian(msk[t][:, im[m], iu40[t], isr["D"]])
            L.append(f"    {m:<11s}" + "".join(f"{v:14.4f}" for v in row)
                     + f"      D={mk:.4f}")
    L.append("")

    # the headline: does fitting the refit on prefill tokens make the DECODE path
    # worse than not refitting at all?
    L.append("does the reconstruction help or hurt, per eval stream? "
             "(refit error - mask-only error, u40, median over layers, "
             "negative = reconstruction helps)")
    for t in ("o", "m"):
        L.append(f"  {MODNAME[t]}")
        for m in mixes:
            cells = []
            for s in streams:
                dd = (err[t][:, im[m], iu40[t], isr[s]]
                      - msk[t][:, im[m], iu40[t], isr[s]])
                med, lo, hi, _ = boot(dd)
                mark = "HURTS" if lo > 0 else ("helps" if hi < 0 else "  ~  ")
                cells.append(f"{s} {med:+.4f} {mark}")
            L.append(f"    {m:<9s}" + "   ".join(cells))
    L.append("")

    # dual-fixed selection: reconstruction effect with selection held constant
    if np.isfinite(dual["m"]).any():
        L.append("dual-fixed selection (units from max(rank I_traj, rank I_CoC), "
                 "reconstruction only):")
        for t in ("o", "m"):
            L.append(f"  {MODNAME[t]}")
            for m in mixes:
                row = [np.nanmedian(dual[t][:, im[m], isr[s]]) for s in streams]
                L.append(f"    {m:<11s}" + "".join(f"{v:14.4f}" for v in row))
        L.append("")

    # ---- G3 selection movement -------------------------------------------
    L += ["G3  does the mixture move the SELECTION, or only the reconstruction?",
          "    (a pass means the kept sets differ by more than calibration noise, so part",
          "     of G2 is a selection effect; a fail attributes G2 purely to the refit)"]
    ov_mat = {}
    for t in ("o", "m"):
        k = kept[t][:, :, :, iu40[t], :]                         # (L, M, P, U)
        n_keep = k[:, 0, 0].sum(-1).astype(float)
        mat = np.zeros((len(mixes), len(mixes)))
        for i in range(len(mixes)):
            for j in range(len(mixes)):
                inter = (k[:, i] & k[:, j]).sum(-1).mean(-1)     # mean over fold pairs
                mat[i, j] = np.mean(inter / n_keep)
        ov_mat[t] = mat
        vs = mat[im["VT"], im[pm]]
        L.append(f"  {MODNAME[t]:<32s} VT vs {pm}: {vs:.3f} "
                 f"(noise floor {NOISE_FLOOR[t]:.3f}) -> "
                 f"{'PASS (selection moved)' if vs < NOISE_FLOOR[t] else 'fail (within noise)'}")
    g3 = all(ov_mat[t][im["VT"], im[pm]] < NOISE_FLOOR[t] for t in ("o", "m"))
    L.append(f"  VERDICT G3: {'PASS' if g3 else 'FAIL'}"
             + ("" if g3 else "  -- G2 is attributable to the REFIT, not to a"
                              " different set of units being kept"))
    L.append("")

    # ---- G4 cost on the vision stream ------------------------------------
    L += ["G4  cost: does up-weighting the decode stream damage the vision path?"]
    g4 = {}
    for t in ("o", "m"):
        med, lo, hi, p = boot(gate_delta(t, pm, "nat", "V"))
        g4[t] = hi < 0.02
        L.append(f"  {MODNAME[t]:<32s} err_V({pm}) - err_V(nat) median {med:+.4f} "
                 f"[{lo:+.4f}, {hi:+.4f}] -> {'PASS' if g4[t] else 'fail'}")
    L.append(f"  VERDICT G4: {'PASS' if all(g4.values()) else 'FAIL'}")
    L.append("")

    # ---- G5 per-layer profile --------------------------------------------
    L += [(f"G5  per-layer removal limit r*(eps) under mix `{pm}`, "
           "error averaged over eval streams")]
    rst = {}
    for t in ("o", "m"):
        e_mean = np.nanmean(err[t][:, im[pm]], axis=-1)           # (L, Lv)
        rst[t] = {eps: np.array([rstar(frac[t], e_mean[i], eps) for i in range(len(layers))])
                  for eps in EPS}
        for eps in EPS:
            r = rst[t][eps]
            L.append(f"  {MODNAME[t]:<32s} eps={eps:.2f}  mean {r.mean():.3f}  "
                     f"std {r.std():.3f}  min {r.min():.3f} (L{layers[r.argmin()]})  "
                     f"max {r.max():.3f} (L{layers[r.argmax()]})")
    g5 = all(rst[t][0.10].std() > 0.05 for t in ("o", "m"))
    L.append(f"  VERDICT G5: {'PASS (non-uniform)' if g5 else 'FAIL (flat -> uniform is fine)'}")
    L.append("")

    # ---- budget-matched allocation ---------------------------------------
    amix = args.alloc_mix or pm
    curves = {t: (removed[t], np.nanmean(err[t][:, im[amix]], axis=-1)) for t in ("o", "m")}
    step = {"o": 1, "m": args.mlp_step}
    n_l = len(layers)
    # per-axis budgets: exactly what u40 removes on each axis, so the reallocation is
    # a pure DEPTH move and stays comparable to run_grid.allocations()
    ax_budget = {"o": n_l * 13 * P_UNIT["o"], "m": n_l * 4898 * P_UNIT["m"]}
    alloc = {}
    for mode in ("free", "per_axis"):
        if mode == "free":
            pick, got = waterfill(curves, U40_BUDGET, step)
        else:
            pick, got = {}, 0
            for t in ("o", "m"):
                p, g = waterfill({t: curves[t]}, ax_budget[t], step)
                pick[t] = p[t]
                got += g
        alloc[mode] = (pick["o"] / N_UNITS["o"], pick["m"] / N_UNITS["m"], got)
    rq, rm, got = alloc["per_axis"]
    np.savez(out_dir / "alloc.npz", rq=rq, rm=rm, layers=layers, mix=amix,
             removed=got, budget=U40_BUDGET,
             rq_free=alloc["free"][0], rm_free=alloc["free"][1],
             removed_free=alloc["free"][2],
             uniform=np.full(n_l, 0.3985632694))
    L += [f"budget-matched allocation from the `{amix}` curves (u40 = {U40_BUDGET:,} params)"]
    for mode in ("per_axis", "free"):
        q, m, g = alloc[mode]
        tag = ("per-axis (each axis keeps its own u40 budget -- a pure depth move)"
               if mode == "per_axis" else
               "free (budget may move between the head and channel axes)")
        L += [(f"  {tag}: realised {g:,}, "
               f"shortfall {100 * (U40_BUDGET - g) / U40_BUDGET:.3f}%"),
              (f"    Q-head  prune fraction mean {q.mean():.3f} std {q.std():.3f} "
               f"range [{q.min():.3f}, {q.max():.3f}]"),
              (f"    MLP-ch  prune fraction mean {m.mean():.3f} std {m.std():.3f} "
               f"range [{m.min():.3f}, {m.max():.3f}]")]
    L += [("  uniform reference 0.3986 on both axes; alloc.npz holds (rq, rm) in "
           "run_grid.allocations() form (per-axis is the primary)"), ""]

    (out_dir / "racfit_summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    make_plots(plots, d, err, msk, ov_mat, rst, frac, iu40, im, isr, mixes, streams,
               layers, rq, rm, pm)
    (out_dir / "metrics_analysis.json").write_text(json.dumps({
        "gates": {"G1": bool(all(g1.values())), "G2": bool(g2_pass), "G3": bool(g3),
                  "G4": bool(all(g4.values())), "G5": bool(g5)},
        "primary_mix": pm, "alloc_mix": amix, "alloc_removed": int(got),
        "n_layers": len(layers),
        "gate_deltas": {f"{k}|{t}": list(v[:4]) for (k, t), v in g2.items()},
    }, indent=2))
    print("saved ->", out_dir)


def make_plots(plots, d, err, msk, ov_mat, rst, frac, iu40, im, isr, mixes, streams,
               layers, rq, rm, pm):
    # 1. mean error curves, mixture x eval stream
    fig, ax = plt.subplots(2, 3, figsize=(13, 7.5))
    for r, t in enumerate(("o", "m")):
        for c, s in enumerate(streams):
            a = ax[r, c]
            for m in mixes:
                a.plot(frac[t], np.nanmedian(err[t][:, im[m], :, isr[s]], axis=0),
                       "-o", ms=3, color=COL.get(m, MUTED), label=m, lw=1.6)
            a.plot(frac[t], np.nanmedian(msk[t][:, im["VT"], :, isr[s]], axis=0),
                   "--", color=INK, lw=1.2, label="no reconstruction")
            a.set_title(f"{MODNAME[t].split(' (')[0]} -> eval {SLBL[s]}")
            a.set_xlabel("removal fraction")
            a.set_ylabel("held-out rel. output error")
            # ill-conditioned mixtures overshoot 1 by an order of magnitude, so a
            # linear axis would flatten every arm that matters
            a.set_yscale("symlog", linthresh=0.01)
            a.axhline(1.0, color=MUTED, lw=0.8, ls=":")
            a.grid(color=GRID, lw=0.6)
    ax[0, 0].legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(plots / "mean_curves.png", dpi=150)
    plt.close(fig)

    # 2. per-layer heatmap under the primary mixture, eval = decode
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for i, t in enumerate(("o", "m")):
        z = err[t][:, im[pm], :, isr["D"]]
        h = ax[i].imshow(z, aspect="auto", origin="lower", cmap="magma",
                         vmin=0, vmax=float(np.nanpercentile(z, 98)),
                         extent=[frac[t][0], frac[t][-1], layers[0], layers[-1]])
        ax[i].set_title(f"{MODNAME[t]} -- err on decode, mix `{pm}`")
        ax[i].set_xlabel("removal fraction")
        ax[i].set_ylabel("layer")
        fig.colorbar(h, ax=ax[i])
    fig.tight_layout()
    fig.savefig(plots / "layer_heat.png", dpi=150)
    plt.close(fig)

    # 3. per-layer removal limit
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for i, t in enumerate(("o", "m")):
        for eps, col in zip(EPS, ("#1f9d55", "#2a78d6", "#eda100")):
            ax[i].plot(layers, rst[t][eps], "-o", ms=3, color=col, label=f"eps={eps}")
        ax[i].axhline(0.3986, color=INK, ls="--", lw=1.1, label="uniform u40")
        ax[i].set_title(f"{MODNAME[t]} -- max removal at tolerance")
        ax[i].set_xlabel("layer")
        ax[i].set_ylabel("r*")
        ax[i].grid(color=GRID, lw=0.6)
        ax[i].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(plots / "rstar_profile.png", dpi=150)
    plt.close(fig)

    # 4. fit-mixture x eval-stream matrix at u40
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for i, t in enumerate(("o", "m")):
        z = np.array([[np.nanmedian(err[t][:, im[m], iu40[t], isr[s]]) for s in streams]
                      for m in mixes])
        h = ax[i].imshow(z, cmap="viridis")
        ax[i].set_xticks(range(len(streams)), [SLBL[s] for s in streams], rotation=20)
        ax[i].set_yticks(range(len(mixes)), mixes)
        for a in range(z.shape[0]):
            for b in range(z.shape[1]):
                ax[i].text(b, a, f"{z[a, b]:.3f}", ha="center", va="center",
                           color="w", fontsize=8)
        ax[i].set_title(f"{MODNAME[t]} -- rel_err at u40")
        fig.colorbar(h, ax=ax[i])
    fig.tight_layout()
    fig.savefig(plots / "stream_matrix.png", dpi=150)
    plt.close(fig)

    # 5. kept-set overlap between mixtures
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for i, t in enumerate(("o", "m")):
        h = ax[i].imshow(ov_mat[t], cmap="cividis", vmin=0.4, vmax=1.0)
        ax[i].set_xticks(range(len(mixes)), mixes, rotation=20)
        ax[i].set_yticks(range(len(mixes)), mixes)
        for a in range(len(mixes)):
            for b in range(len(mixes)):
                ax[i].text(b, a, f"{ov_mat[t][a, b]:.2f}", ha="center", va="center",
                           color="w", fontsize=8)
        ax[i].set_title(f"{MODNAME[t]} -- kept-set overlap "
                        f"(noise floor {NOISE_FLOOR[t]})")
        fig.colorbar(h, ax=ax[i])
    fig.tight_layout()
    fig.savefig(plots / "overlap.png", dpi=150)
    plt.close(fig)

    # 6. conditioning
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for i, t in enumerate(("o", "m")):
        for s, col in zip(streams, ("#2a78d6", "#eda100", "#e05252")):
            ax[i].semilogy(layers, d[f"diag_pr_rank_{s}_{t}"], "-o", ms=3, color=col,
                           label=SLBL[s])
        ax[i].set_title(f"{MODNAME[t]} -- Hessian effective rank per stream")
        ax[i].set_xlabel("layer")
        ax[i].set_ylabel("participation-ratio rank")
        ax[i].grid(color=GRID, lw=0.6)
        ax[i].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(plots / "conditioning.png", dpi=150)
    plt.close(fig)

    # 7. allocation
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(layers, rq, "-o", ms=3, color="#2a78d6", label="Q heads")
    ax.plot(layers, rm, "-o", ms=3, color="#1f9d55", label="MLP channels")
    ax.axhline(0.3986, color=INK, ls="--", lw=1.1, label="uniform u40")
    ax.set_xlabel("layer")
    ax.set_ylabel("prune fraction")
    ax.set_title("budget-matched allocation from the reconstruction curves")
    ax.grid(color=GRID, lw=0.6)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(plots / "alloc.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
