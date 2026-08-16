"""Read the importance-calibration run and judge the pre-registered gates.

Units: predicted first-order damage (the |dL/dg| the criteria rank, plus its signed
form) against realized single-unit ablation loss, paired per clip. Groups: realized
group loss against (a) the sum of its members' realized singles -- pure additivity --
and (b) the summed first-order prediction, which is the calibration that matters at
the operating point.

Gates (pre-registered in plans/2026-08-15_importance-calibration.md):
  G-CAL-A  pooled within-layer Spearman(pred_abs, realized) >= 0.7, per axis x objective
  G-CAL-B  log-log Pearson r >= 0.7 over units with realized dL > 0
  G-CAL-C  actual u40 Q cut (bottom-13): median |log2(group / sum-of-singles)| > 1
           means first-order additivity is broken at the operating point

Layer 35 is a structural zero for the trajectory objective -- its width units feed only
the final hidden state, which the expert never reads (it reads the KV cache) -- so that
layer is excluded from traj pooling and reported as its own consistency check.

Usage:
  .venv/bin/python experiments/head_analysis/analyze_imp_calib.py
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr

REPO = Path(__file__).resolve().parents[2]

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})

GATE_RHO = 0.7      # G-CAL-A / G-CAL-B threshold, same as the quant track's Gate G0
GATE_LOG2R = 1.0    # G-CAL-C: >2x deviation from additivity
ZERO = 1e-12        # structural-zero detection for the layer-35 traj check

OBJS = (("coc", "dnll", "pred_coc", "pred_coc_abs"),
        ("traj", "dfm", "pred_traj", "pred_traj_abs"))


def load(out_dir):
    rows = []
    for f in sorted(out_dir.glob("records_s*.json")):
        rows.extend(json.loads(f.read_text()))
    metas = []
    for f in sorted(out_dir.glob("meta_s*.json")):
        metas.extend(json.loads(f.read_text()))
    if not rows:
        raise SystemExit(f"no records under {out_dir}")
    return rows, metas


def aggregate(rows):
    """clip-mean per (layer, axis, kind, name); returns dict key -> record."""
    agg = {}
    for r in rows:
        key = (r["layer"], r["axis"], r["kind"], r["name"])
        a = agg.setdefault(key, {"layer": r["layer"], "axis": r["axis"], "kind": r["kind"],
                                 "name": r["name"], "ids": r["ids"], "n_ids": r["n_ids"],
                                 "dnll": [], "dfm": [], "pred_coc": [], "pred_coc_abs": [],
                                 "pred_traj": [], "pred_traj_abs": []})
        for f in ("dnll", "dfm", "pred_coc", "pred_coc_abs", "pred_traj", "pred_traj_abs"):
            a[f].append(r[f])
    for a in agg.values():
        a["n_clips"] = len(a["dnll"])
        for f in ("dnll", "dfm", "pred_coc", "pred_coc_abs", "pred_traj", "pred_traj_abs"):
            a[f"{f}_clips"] = a[f]
            a[f] = float(np.mean(a[f]))
        a["neg_frac_nll"] = float(np.mean(np.array(a["dnll_clips"]) < 0))
        a["neg_frac_fm"] = float(np.mean(np.array(a["dfm_clips"]) < 0))
    return agg


def boot_ci(x, y, fn, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    stats = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        try:
            stats.append(fn(x[i], y[i]))
        except ValueError:
            continue
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def pooled_spearman(units, real_f, pred_f, layers):
    """Within-layer ranks pooled across layers, then Pearson on the ranks."""
    rp, rr = [], []
    per_layer = {}
    for li in layers:
        sub = [u for u in units if u["layer"] == li]
        if len(sub) < 3:
            continue
        p = np.array([u[pred_f] for u in sub])
        r = np.array([u[real_f] for u in sub])
        per_layer[li] = float(spearmanr(p, r).statistic)
        rp.append(rankdata(p) / len(sub))
        rr.append(rankdata(r) / len(sub))
    rp, rr = np.concatenate(rp), np.concatenate(rr)
    rho = float(pearsonr(rp, rr).statistic)
    lo, hi = boot_ci(rp, rr, lambda a, b: pearsonr(a, b).statistic)
    return rho, (lo, hi), len(rp), per_layer


def split_half(units, clips_f, layers):
    """Reliability of the realized target itself: odd- vs even-clip means, within-layer
    ranks pooled. Any predictor's observable correlation is bounded by roughly the
    square root of this, so a near-zero value means the gate cannot convict the score --
    the single-unit damage is not a stable quantity at this clip count."""
    ra, rb = [], []
    for li in layers:
        sub = [u for u in units if u["layer"] == li]
        if len(sub) < 3:
            continue
        a = np.array([np.mean(u[clips_f][0::2]) for u in sub])
        b = np.array([np.mean(u[clips_f][1::2]) for u in sub])
        ra.append(rankdata(a) / len(sub))
        rb.append(rankdata(b) / len(sub))
    return float(pearsonr(np.concatenate(ra), np.concatenate(rb)).statistic)


def loglog_pearson(units, real_f, pred_f):
    real = np.array([u[real_f] for u in units])
    pred = np.array([u[pred_f] for u in units])
    pos = (real > 0) & (pred > 0)
    if pos.sum() < 3:
        return None
    lx, ly = np.log10(pred[pos]), np.log10(real[pos])
    r = float(pearsonr(lx, ly).statistic)
    lo, hi = boot_ci(lx, ly, lambda a, b: pearsonr(a, b).statistic)
    slope = float(np.polyfit(lx, ly, 1)[0])
    return {"r": r, "ci": [lo, hi], "n_pos": int(pos.sum()),
            "frac_nonpos_real": float(np.mean(real <= 0)), "slope": slope}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="imp_calib_v1")
    args = ap.parse_args()
    out_dir = REPO / "outputs" / args.exp_id
    plots = out_dir / "plots"
    plots.mkdir(exist_ok=True)

    rows, metas = load(out_dir)
    agg = aggregate(rows)
    n_clips = max(a["n_clips"] for a in agg.values())
    non_det = [m["clip_id"] for m in metas if not m.get("deterministic", True)]
    layers = sorted({a["layer"] for a in agg.values()})
    units_all = [a for a in agg.values() if a["kind"] == "unit"]

    # layer-35 structural zero for traj: confirm, then exclude from traj pooling
    l35 = [u for u in units_all if u["layer"] == 35]
    l35_check = {"max_abs_dfm": max((abs(u["dfm"]) for u in l35), default=0.0),
                 "max_pred_traj_abs": max((u["pred_traj_abs"] for u in l35), default=0.0)}
    traj_layers = [li for li in layers
                   if any(abs(u["dfm"]) > ZERO for u in units_all if u["layer"] == li)]

    res = {"n_clips": n_clips, "non_deterministic_clips": non_det,
           "layers": layers, "traj_layers": traj_layers, "layer35_check": l35_check,
           "gA": {}, "gB": {}, "gB_signed": {}, "sign_stats": {}}

    for axis in ("q", "mlp"):
        for obj, real_f, pred_s, pred_a in OBJS:
            units = [u for u in units_all if u["axis"] == axis]
            ls = traj_layers if obj == "traj" else layers
            cell = f"{axis}_{obj}"
            rho, ci, n, per_layer = pooled_spearman(units, real_f, pred_a, ls)
            res["gA"][cell] = {"rho": rho, "ci": ci, "n": n, "per_layer": per_layer,
                               "pass": rho >= GATE_RHO}
            sub = [u for u in units if u["layer"] in ls]
            gb = loglog_pearson(sub, real_f, pred_a)
            if gb:
                gb["pass"] = gb["r"] >= GATE_RHO
            res["gB"][cell] = gb
            res["gB_signed"][cell] = loglog_pearson(sub, real_f, pred_s)
            real = np.array([u[real_f] for u in sub])
            negf = np.array([u["neg_frac_nll" if obj == "coc" else "neg_frac_fm"]
                             for u in sub])
            res["sign_stats"][cell] = {
                "frac_units_mean_neg": float(np.mean(real < 0)),
                "frac_units_sign_mixed": float(np.mean((negf >= 0.2) & (negf <= 0.8)))}
            res.setdefault("target_reliability", {})[cell] = split_half(
                units, f"{real_f}_clips", ls)

    # ---- groups: additivity (G-CAL-C) and operating-point prediction ----
    unit_by_key = {(a["layer"], a["axis"], a["name"]): a for a in units_all}

    def singles_sum(g, real_f):
        vals = [unit_by_key[(g["layer"], g["axis"], f"{g['axis']}{i}")][real_f]
                for i in g["ids"]]
        return float(np.sum(vals))

    groups = [a for a in agg.values() if a["kind"] == "group"]
    gtab = []
    for g in groups:
        row = {"layer": g["layer"], "name": g["name"], "axis": g["axis"],
               "n_ids": g["n_ids"], "dnll": g["dnll"], "dfm": g["dfm"],
               "pred_coc": g["pred_coc"], "pred_traj": g["pred_traj"]}
        if g["ids"] is not None:
            row["sum_dnll"] = singles_sum(g, "dnll")
            row["sum_dfm"] = singles_sum(g, "dfm")
            for obj, num, den in (("nll", g["dnll"], row["sum_dnll"]),
                                  ("fm", g["dfm"], row["sum_dfm"])):
                row[f"log2R_{obj}"] = (float(np.log2(num / den))
                                       if num > 0 and den > 0 else None)
        for obj, num, den in (("nll", g["dnll"], g["pred_coc"]),
                              ("fm", g["dfm"], g["pred_traj"])):
            row[f"log2P_{obj}"] = (float(np.log2(num / den))
                                   if num > 0 and den > 0 else None)
        gtab.append(row)
    res["groups"] = gtab

    def med_abs(vals):
        v = [abs(x) for x in vals if x is not None]
        return (float(np.median(v)), len(v)) if v else (None, 0)

    qcut = [r for r in gtab if r["name"].startswith("qcut_")]
    gC = {}
    for obj in ("nll", "fm"):
        rows_o = qcut if obj == "nll" else [r for r in qcut if r["layer"] in traj_layers]
        med, n = med_abs([r[f"log2R_{obj}"] for r in rows_o])
        gC[obj] = {"median_abs_log2R": med, "n": n,
                   "n_sign_violated": sum(1 for r in rows_o if r[f"log2R_{obj}"] is None),
                   "pass_broken": (med is not None and med > GATE_LOG2R)}
    res["gC"] = gC
    res["additivity_by_size"] = {}
    for label, sel in (("k8", [r for r in gtab if r["name"].endswith("8")]),
                       ("k13_qcut", qcut),
                       ("k32", [r for r in gtab if r["name"].endswith("32")])):
        res["additivity_by_size"][label] = {
            "nll": med_abs([r.get("log2R_nll") for r in sel])[0],
            "fm": med_abs([r.get("log2R_fm") for r in sel if r["layer"] in traj_layers])[0]}
    res["opoint_pred_ratio"] = {
        "mcut_nll": med_abs([r["log2P_nll"] for r in gtab
                             if r["name"].startswith("mcut_")])[0],
        "mcut_fm": med_abs([r["log2P_fm"] for r in gtab if r["name"].startswith("mcut_")
                            and r["layer"] in traj_layers])[0],
        "qcut_nll": med_abs([r["log2P_nll"] for r in qcut])[0],
        "qcut_fm": med_abs([r["log2P_fm"] for r in qcut
                            if r["layer"] in traj_layers])[0]}

    # realized damage of each criterion's actual MLP cut, the direct criterion contrast
    res["mcut_table"] = [
        {"layer": r["layer"], "crit": r["name"].split("_", 1)[1],
         "dnll": r["dnll"], "dfm": r["dfm"]}
        for r in gtab if r["name"].startswith("mcut_")]

    # J-lens secondary: ordering only, against realized CoC damage
    jl_path = REPO / "outputs" / "jlens_v2" / "jlens.npz"
    if jl_path.exists():
        jl = dict(np.load(jl_path))
        jsec = {}
        for axis, key in (("q", "q_j"), ("mlp", "mlp_j")):
            units = [dict(u, j=float(jl[key][u["layer"], u["ids"][0]]))
                     for u in units_all if u["axis"] == axis]
            rho, ci, n, _ = pooled_spearman(units, "dnll", "j", layers)
            jsec[axis] = {"rho": rho, "ci": ci, "n": n}
        res["j_secondary"] = jsec

    # ---- plots ----
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.4))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(layers)))
    lcol = dict(zip(layers, cmap))
    for row_i, axis in enumerate(("q", "mlp")):
        for col_i, (obj, real_f, _ps, pred_a) in enumerate(OBJS):
            ax = axes[row_i][col_i]
            ls = traj_layers if obj == "traj" else layers
            for li in ls:
                sub = [u for u in units_all if u["axis"] == axis and u["layer"] == li]
                p = np.array([u[pred_a] for u in sub])
                r = np.array([u[real_f] for u in sub])
                pos = (p > 0) & (r > 0)
                ax.scatter(p[pos], r[pos], s=12, color=lcol[li], label=f"L{li}", alpha=0.8)
            lims = ax.get_xlim()
            grid = np.geomspace(max(lims[0], 1e-12), lims[1], 10)
            ax.plot(grid, grid, ls="--", lw=0.8, color=MUTED)
            ax.set_xscale("log")
            ax.set_yscale("log")
            g = res["gB"][f"{axis}_{obj}"]
            ax.set_title(f"{axis} / {obj} — ρ={res['gA'][f'{axis}_{obj}']['rho']:.2f}, "
                         f"log-log r={g['r']:.2f}" if g else f"{axis} / {obj}")
            ax.set_xlabel("predicted |dL| (Taylor)")
            ax.set_ylabel("realized dL")
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle("Predicted vs realized single-unit ablation loss", y=0.995)
    fig.tight_layout()
    fig.savefig(plots / "pred_vs_real.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for ax, obj, title in ((axes[0], "nll", "CoC NLL"), (axes[1], "fm", "FM MSE")):
        pts = {"k8": [], "k13_qcut": [], "k32": []}
        for r in gtab:
            v = r.get(f"log2R_{obj}")
            if v is None or (obj == "fm" and r["layer"] not in traj_layers):
                continue
            if r["name"].startswith("qcut_"):
                pts["k13_qcut"].append(v)
            elif r["name"].endswith("32"):
                pts["k32"].append(v)
            elif r["name"].endswith("8"):
                pts["k8"].append(v)
        keys = [k for k in ("k8", "k13_qcut", "k32") if pts[k]]
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.axhline(GATE_LOG2R, color=C3, lw=0.8, ls="--")
        ax.axhline(-GATE_LOG2R, color=C3, lw=0.8, ls="--")
        for i, k in enumerate(keys):
            x = np.full(len(pts[k]), i) + np.random.default_rng(1).uniform(
                -0.12, 0.12, len(pts[k]))
            ax.scatter(x, pts[k], s=14, color=C1, alpha=0.7)
        ax.set_xticks(range(len(keys)), keys)
        ax.set_ylabel("log2(group / Σ singles)")
        ax.set_title(f"additivity — {title}")
    fig.tight_layout()
    fig.savefig(plots / "additivity.png", dpi=150)
    plt.close(fig)

    fig, axm = plt.subplots(figsize=(9.6, 3.8))
    crits = ("traj", "coc", "dual", "jtraj")
    ccol = dict(zip(crits, (C4, C3, C1, C2)))
    width = 0.19
    mcut = {(r["layer"], r["crit"]): r["dnll"] for r in res["mcut_table"]}
    for i, crit in enumerate(crits):
        xs = np.arange(len(layers)) + (i - 1.5) * width
        axm.bar(xs, [mcut[(li, crit)] for li in layers], width,
                color=ccol[crit], label=crit)
    axm.axhline(0, color=MUTED, lw=0.8)
    axm.set_xticks(range(len(layers)), [f"L{li}" for li in layers])
    axm.set_ylabel("realized dNLL of the actual MLP cut")
    axm.set_title("per-layer damage of each criterion's real bottom-40% MLP cut (CoC NLL)")
    axm.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(plots / "mcut_damage.png", dpi=150)
    plt.close(fig)

    (out_dir / "calib_metrics.json").write_text(json.dumps(res, indent=2))

    lines = [f"importance calibration — {args.exp_id}",
             f"clips {n_clips}, layers {layers}, non-deterministic clips: {non_det or 'none'}",
             (f"layer-35 traj structural zero: max|dfm|={l35_check['max_abs_dfm']:.2e} "
              f"max pred={l35_check['max_pred_traj_abs']:.2e}"), ""]
    for cell in res["gA"]:
        a, b = res["gA"][cell], res["gB"][cell]
        lines.append(
            f"G-CAL-A {cell:9s} rho={a['rho']:+.3f} [{a['ci'][0]:+.3f},{a['ci'][1]:+.3f}] "
            f"n={a['n']} -> {'PASS' if a['pass'] else 'FAIL'}")
        if b:
            lines.append(
                f"G-CAL-B {cell:9s} r={b['r']:+.3f} [{b['ci'][0]:+.3f},{b['ci'][1]:+.3f}] "
                f"slope={b['slope']:.2f} n+={b['n_pos']} "
                f"nonpos={b['frac_nonpos_real']:.0%} -> {'PASS' if b['pass'] else 'FAIL'}")
    for obj, g in res["gC"].items():
        lines.append(f"G-CAL-C {obj}: median|log2R|="
                     f"{g['median_abs_log2R']:.3f} n={g['n']} "
                     f"sign-violated={g['n_sign_violated']} -> "
                     f"{'BROKEN (>1)' if g['pass_broken'] else 'within 2x'}"
                     if g["median_abs_log2R"] is not None else f"G-CAL-C {obj}: no data")
    rel = res.get("target_reliability", {})
    lines.append("target split-half reliability (realized dL, within-layer ranks): "
                 + ", ".join(f"{k}={v:+.3f}" for k, v in rel.items()))
    lines.append("  -> near zero means A/B cannot convict the score: single-unit damage "
                 "is not a stable quantity at this clip count")
    lines.append(f"additivity by size: {json.dumps(res['additivity_by_size'])}")
    lines.append(f"operating-point pred ratio: {json.dumps(res['opoint_pred_ratio'])}")
    if "j_secondary" in res:
        lines.append(f"J-lens ordering vs realized dNLL: "
                     f"q rho={res['j_secondary']['q']['rho']:+.3f}, "
                     f"mlp rho={res['j_secondary']['mlp']['rho']:+.3f}")
    text = "\n".join(lines)
    (out_dir / "summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
