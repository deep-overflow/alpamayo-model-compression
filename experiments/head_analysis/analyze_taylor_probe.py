"""Stage 1 gates: is the first-order Taylor score comparable across the two axes?

Reads run_taylor_probe.py's rows and judges S0-S4 from
plans/2026-08-30_axis-taylor-comparability.md:

  S0  integrity: the gate pass's own loss reproduces the mask path's baseline (the two
      differ only by the gates' float32 upcast), and every probe's measurement is a
      difference within one path
  S1  slope, the main reading: per axis, OLS through the origin of measured dL on
      predicted dL over the SINGLE-UNIT probes (one Q head; one MLP channel), clip
      bootstrap. beta_q / beta_mlp inside [0.5, 2] means the raw cross-axis ratios can be
      read quantitatively; outside, that ratio IS the correction factor. Reported beside:
      Pearson r and the sign-agreement rate, which say whether first order is predictive
      at all on that axis
  S1b parameter-matched: the same fit for one head against one 85-channel block, so the
      axis effect is not read off perturbations of different parameter size
  S2  set additivity: measured/predicted for the real arm cuts, and -- separately -- the
      sum of a cut's 36 per-layer effects against the same cut applied to all 36 layers,
      which is cross-layer additivity measured directly
  S3  ordering: do the five arm_full cuts rank by measured loss (and dminADE) the way the
      first-order mass says they should?
  S4  robustness: S1 recomputed against sum_s |dL_s/dg| instead of |sum_s|

Also emits the Stage 2a raw cross-axis table for BOTH towers (expert from the raw
step-sum importance, VLM from importance_v2), which is the comparison the whole plan is
about; it needs no GPU and no probe.

Usage:
  .venv/bin/python experiments/head_analysis/analyze_taylor_probe.py \
      [--probe taylor_probe_expert] [--out axis_taylor_analysis]
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

BG, INK, MUTED, GRID = "#FAF9F5", "#29261B", "#6B6555", "#E8E6DC"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.size": 9, "axes.grid": True, "axes.axisbelow": True,
})
REPO = Path(__file__).resolve().parents[2]
# per-unit parameter cost, from the shipped weight shapes (check_vlm_axis.py verifies)
P = {"vlm": (2 * 4096 * 128, 3 * 4096), "expert": (2 * 2048 * 128, 3 * 2048)}
AXIS_LABEL = {"q": "Q head", "mlp": "MLP channel"}


def load_rows(exp_id):
    rows = []
    for f in sorted(glob.glob(str(REPO / "outputs" / exp_id / "rows_s*of*.json"))):
        rows.extend(json.loads(Path(f).read_text()))
    return rows


def table(rows):
    """Flatten to arrays, one entry per (clip, probe)."""
    cols = {k: [] for k in ("clip", "axis", "kind", "n_units", "params", "pred",
                            "pred_abs", "raw_I", "dloss", "dminADE", "name")}
    for ci, r in enumerate(rows):
        for p in r["probes"]:
            cols["clip"].append(ci)
            for k in ("axis", "kind", "n_units", "params", "pred", "pred_abs", "dloss",
                      "name"):
                cols[k].append(p[k])
            cols["raw_I"].append(p.get("raw_I", np.nan))
            cols["dminADE"].append(p.get("dminADE", np.nan))
    out = {k: np.asarray(v) for k, v in cols.items()}
    for k in ("n_units", "params", "pred", "pred_abs", "raw_I", "dloss", "dminADE"):
        out[k] = out[k].astype(float)
    return out


def slope_ci(pred, meas, clip, n_boot=4000, seed=0):
    """OLS through the origin with a clip bootstrap: beta = sum(x y) / sum(x x)."""
    def fit(x, y):
        d = float((x * x).sum())
        return float((x * y).sum() / d) if d > 0 else float("nan")

    beta = fit(pred, meas)
    clips = np.unique(clip)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(clips, len(clips), replace=True)
        idx = np.concatenate([np.flatnonzero(clip == c) for c in pick])
        boots.append(fit(pred[idx], meas[idx]))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    resid = meas - beta * pred
    ss = float(((meas - meas.mean()) ** 2).sum())
    return {"n": len(pred), "beta": beta, "lo": float(lo), "hi": float(hi),
            "r2_origin": float(1 - (resid ** 2).sum() / ss) if ss > 0 else float("nan"),
            "pearson": float(np.corrcoef(pred, meas)[0, 1]) if len(pred) > 2 else float("nan"),
            "sign_agree": float(np.mean(np.sign(pred) == np.sign(meas))),
            "median_ratio": float(np.median(meas[pred != 0] / pred[pred != 0]))}


def ratio_ci(num, den, n_boot=4000, seed=0):
    """CI on beta_num / beta_den, resampling clips jointly so the pair stays paired."""
    def fit(x, y, idx):
        d = float((x[idx] * x[idx]).sum())
        return float((x[idx] * y[idx]).sum() / d) if d > 0 else float("nan")

    (pn_, mn, cn), (pd_, md, cd) = num, den
    point = fit(pn_, mn, np.arange(len(pn_))) / fit(pd_, md, np.arange(len(pd_)))
    clips = np.unique(np.concatenate([cn, cd]))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(clips, len(clips), replace=True)
        i1 = np.concatenate([np.flatnonzero(cn == c) for c in pick])
        i2 = np.concatenate([np.flatnonzero(cd == c) for c in pick])
        b1, b2 = fit(pn_, mn, i1), fit(pd_, md, i2)
        boots.append(b1 / b2 if b2 != 0 else np.nan)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"ratio": float(point), "lo": float(lo), "hi": float(hi)}


def subset(t, axis, kinds, n_units=None):
    m = (t["axis"] == axis) & np.isin(t["kind"], kinds)
    if n_units is not None:
        m &= t["n_units"] == n_units
    return t["pred"][m], t["dloss"][m], t["clip"][m], t["pred_abs"][m]


def bottom_share(a, frac=0.5):
    """Mean over layers of the mass held by the lowest-scoring `frac` of a layer's units.

    A layer whose whole axis scores exactly 0 (VLM layer 35 carries no trajectory
    importance at all) has no share to take, so it is dropped rather than made a nan.
    """
    tot = a.sum(1)
    keep = tot > 0
    k = int(a.shape[1] * frac)
    return float(np.mean(np.sort(a[keep], 1)[:, :k].sum(1) / tot[keep]))


def raw_cross_axis(expert_run="importance_stepexp_sum", vlm_run="importance_v2"):
    """The comparison itself: raw first-order mass per unit and per parameter, both towers.

    Only the RAW score can carry this -- znorm and dual are per-layer normalisations, so
    they have no cross-axis scale by construction.
    """
    out = {}
    for tower, run, keys in (
            ("expert", expert_run, {"traj": ("traj_exp_q", "traj_exp_mlp")}),
            ("vlm", vlm_run, {"traj": ("traj_vlm_q", "traj_vlm_mlp"),
                              "coc": ("coc_vlm_q", "coc_vlm_mlp")})):
        z = np.load(REPO / "outputs" / run / "importance.npz")
        p_head, p_chan = P[tower]
        out[tower] = {"run": run, "p_head": p_head, "p_chan": p_chan,
                      "param_ratio": p_head / p_chan, "obj": {}}
        for obj, (kq, km) in keys.items():
            q, m = z[kq].astype(float), z[km].astype(float)
            out[tower]["obj"][obj] = {
                "median_q": float(np.median(q)), "median_mlp": float(np.median(m)),
                "unit_ratio": float(np.median(q) / np.median(m)),
                "param_ratio": float((np.median(q) / p_head) / (np.median(m) / p_chan)),
                "layer_mass_q": float(q.sum(1).mean()),
                "layer_mass_mlp": float(m.sum(1).mean()),
                "layer_mass_ratio": float(q.sum() / m.sum()),
                "param_share_q": float(q.shape[1] * p_head
                                       / (q.shape[1] * p_head + m.shape[1] * p_chan)),
                # what the bottom half of each axis holds -- the expert-axis mechanism plot
                "bottom50_share_q": bottom_share(q),
                "bottom50_share_mlp": bottom_share(m),
            }
    return out


def plot_slopes(t, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
    for ax, (axis, kinds, lbl) in zip(axes, (
            ("q", ("unit",), "Q head (1 unit)"), ("mlp", ("unit",), "MLP channel (1 unit)"))):
        pr, dl, cl, _ = subset(t, axis, kinds)
        if not len(pr):
            continue
        col = C1 if axis == "q" else C2
        ax.scatter(pr, dl, s=7, alpha=0.35, color=col, edgecolors="none")
        s = slope_ci(pr, dl, cl, n_boot=400)
        xs = np.linspace(pr.min(), pr.max(), 10)
        ax.plot(xs, s["beta"] * xs, color=INK, lw=1.2,
                label=f"slope {s['beta']:.2f} [{s['lo']:.2f},{s['hi']:.2f}]")
        ax.plot(xs, xs, color=MUTED, lw=1, ls="--", label="perfect first order")
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.axvline(0, color=MUTED, lw=0.6)
        ax.set_xlabel(r"predicted $\Delta L = -\partial L/\partial g$")
        ax.set_ylabel(r"measured $\Delta L$")
        ax.set_title(f"{lbl}   r={s['pearson']:.2f}, sign agree {s['sign_agree']:.0%}")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "probe_slope.png", dpi=150)
    plt.close(fig)


def plot_magnitude(t, out):
    """Same-parameter perturbations side by side: one head vs one 85-channel block."""
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    for axis, kinds, col, lbl in (("q", ("unit",), C1, "1 Q head (524,288 par)"),
                                  ("mlp", ("block",), C2, "85 MLP ch (522,240 par)"),
                                  ("mlp", ("unit",), C3, "1 MLP channel")):
        pr, dl, _cl, _pa = subset(t, axis, kinds)
        if not len(pr):
            continue
        ax.scatter(np.abs(pr) + 1e-12, np.abs(dl) + 1e-12, s=8, alpha=0.35, color=col,
                   edgecolors="none", label=lbl)
    lim = [1e-10, 1e-1]
    ax.plot(lim, lim, color=MUTED, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$|$predicted $\Delta L|$")
    ax.set_ylabel(r"$|$measured $\Delta L|$")
    ax.set_title("first order vs truth, matched parameter cost")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "probe_magnitude.png", dpi=150)
    plt.close(fig)


def plot_arms(t, out):
    m = t["kind"] == "arm_full"
    if not m.any():
        return
    names = sorted(set(t["name"][m]))
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    for k, name in enumerate(names):
        s = m & (t["name"] == name)
        col = C1 if t["axis"][s][0] == "q" else C2
        axes[0].bar(k - 0.2, np.mean(t["pred"][s]), 0.38, color=col, alpha=0.45)
        axes[0].bar(k + 0.2, np.mean(t["dloss"][s]), 0.38, color=col)
        axes[1].bar(k, np.mean(t["dminADE"][s]), 0.6, color=col)
    for ax, ttl, yl in ((axes[0], "predicted (pale) vs measured dL", r"$\Delta L$"),
                        (axes[1], "measured dminADE@6", r"$\Delta$minADE")):
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace("arm_", "").replace("_full", "") for n in names],
                           fontsize=8)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(ttl)
        ax.set_ylabel(yl)
    fig.tight_layout()
    fig.savefig(out / "probe_arms.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="taylor_probe_expert")
    ap.add_argument("--out", default="axis_taylor_analysis")
    ap.add_argument("--expert-importance", default="importance_stepexp_sum")
    ap.add_argument("--vlm-importance", default="importance_v2")
    args = ap.parse_args()
    out = REPO / "outputs" / args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)

    raw = raw_cross_axis(args.expert_importance, args.vlm_importance)
    lines = [("axis comparability -- can a Q head and an MLP channel be compared "
              "on one Taylor scale?"), "",
             "Stage 2a: raw first-order mass (the only scale that carries across axes)"]
    for tower, d in raw.items():
        share = next(iter(d["obj"].values()))["param_share_q"]
        lines.append(f"  [{tower}] head {d['p_head']:,} par, channel {d['p_chan']:,} par "
                     f"({d['param_ratio']:.1f}x); Q holds {share:.1%} of the two "
                     f"axes' parameters")
        for obj, v in d["obj"].items():
            lines.append(f"    {obj:5s} per-unit Q/MLP {v['unit_ratio']:9.1f}x | "
                         f"PER PARAMETER {v['param_ratio']:7.2f}x | layer mass Q/MLP "
                         f"{v['layer_mass_ratio']:.3f} | bottom-50% share "
                         f"Q {v['bottom50_share_q']:.2%} MLP {v['bottom50_share_mlp']:.2%}")
    lines.append("")

    rows = load_rows(args.probe)
    metrics = {"raw_cross_axis": raw, "n_clips": len(rows)}
    if not rows:
        lines.append(f"no probe rows in outputs/{args.probe} yet -- Stage 1 gates skipped")
        (out / "summary.txt").write_text("\n".join(lines) + "\n")
        (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
        print("\n".join(lines))
        return

    t = table(rows)
    # S0. The gate pass upcasts the o_proj/down_proj input to float32 (the gates are
    # float32) while the mask path stays in the activation dtype, so the two paths' losses
    # are not bit-identical. That offset cancels in every probe -- both terms of
    # dloss = L(S) - L(baseline) come from the mask path -- so what has to be small is the
    # ABSOLUTE gap next to the probe effects being resolved, not its ratio to a loss that
    # is itself ~1e-3 on easy clips. The pre-registered 1e-3 RELATIVE bound was specified
    # without that in view; both readings are reported and the relative one is kept
    # visible rather than quietly dropped.
    gap = np.array([abs(r["grad_loss"] - r["base_loss"]) for r in rows])
    rel = np.array([g / r["base_loss"] for g, r in zip(gap, rows)])
    eff = np.median(np.abs(t["dloss"][t["kind"] == "unit"])) if len(rows) else float("nan")
    metrics["s0"] = {"n_clips": len(rows), "median_rel_gap": float(np.median(rel)),
                     "max_rel_gap": float(np.max(rel)),
                     "median_abs_gap": float(np.median(gap)),
                     "max_abs_gap": float(np.max(gap)),
                     "median_unit_effect": float(eff),
                     "gap_over_effect": float(np.median(gap) / eff),
                     "pass_relative_1e3": bool(np.max(rel) < 1e-3)}
    lines += [(f"S0 gate path vs mask path: abs gap median {np.median(gap):.2e} "
               f"max {np.max(gap):.2e} (relative median {np.median(rel):.2e}, "
               f"max {np.max(rel):.2e})"),
              ("   the pre-registered form asked for relative < 1e-3, which this misses "
               f"({'meets' if metrics['s0']['pass_relative_1e3'] else 'MISSES'}) -- but "
               "that bound was mis-specified: the offset is a per-clip level difference "
               "between a float32-gated and a bf16-masked forward, and BOTH terms of "
               "dloss = L(S) - L(baseline) come from the mask path, so it cancels "
               "exactly in every probe."),
              (f"   what it could still bias is the gradient's evaluation point; the "
               f"empirical check is beta_q below (1.0 would mean the two paths agree to "
               f"first order). Median single-unit probe effect {eff:.2e} sits near the "
               f"offset, so read the MLP single-channel fit with that in view."),
              f"   {len(rows)} clips, {len(rows[0]['probes'])} probes each", ""]

    fits, abs_fits = {}, {}
    for tag, axis, kinds in (("q_unit", "q", ("unit",)),
                             ("mlp_unit", "mlp", ("unit",)),
                             ("mlp_block85", "mlp", ("block",)),
                             ("q_arm", "q", ("arm_layer",)),
                             ("mlp_arm", "mlp", ("arm_layer",))):
        pr, dl, cl, pa = subset(t, axis, kinds)
        if len(pr) > 2:
            fits[tag] = slope_ci(pr, dl, cl)
            abs_fits[tag] = slope_ci(pa, dl, cl)
    metrics["s1_fits"] = fits
    metrics["s4_fits_sumabs"] = abs_fits
    lines.append("S1 slope of measured on predicted (OLS through origin, clip bootstrap)")
    for tag, f in fits.items():
        lines.append(f"  {tag:12s} n={f['n']:6d} beta {f['beta']:+8.3f} "
                     f"[{f['lo']:+.3f},{f['hi']:+.3f}]  r={f['pearson']:+.3f}  "
                     f"sign agree {f['sign_agree']:.0%}  median ratio {f['median_ratio']:+.2f}")
    if {"q_unit", "mlp_unit"} <= set(fits):
        r = ratio_ci(subset(t, "q", ("unit",))[:3], subset(t, "mlp", ("unit",))[:3])
        metrics["s1_ratio_unit"] = r
        ok = 0.5 <= r["lo"] and r["hi"] <= 2.0
        metrics["s1_pass"] = bool(ok)
        verdict = ("inside [0.5,2]: raw ratios readable" if ok
                   else "OUTSIDE [0.5,2]: this IS the correction factor")
        lines.append(f"  S1 beta_q / beta_mlp (single units) = {r['ratio']:.3f} "
                     f"[{r['lo']:.3f},{r['hi']:.3f}] -> {verdict}")
    if {"q_unit", "mlp_block85"} <= set(fits):
        r = ratio_ci(subset(t, "q", ("unit",))[:3], subset(t, "mlp", ("block",))[:3])
        metrics["s1b_ratio_parammatched"] = r
        lines.append(f"  S1b beta_q(1 head) / beta_mlp(85 ch, same params) = {r['ratio']:.3f} "
                     f"[{r['lo']:.3f},{r['hi']:.3f}]")
    lines.append("")

    # S2 -- set additivity, two ways
    lines.append("S2 set additivity")
    add = {}
    for name in sorted({n for n in t["name"] if n.endswith("_full")}):
        base = name[:-5]
        full = t["kind"] == "arm_full"
        fm = full & (t["name"] == name)
        per_layer = np.isin(t["kind"], ["arm_layer"]) & np.char.startswith(t["name"], base)
        if not fm.any():
            continue
        per_clip_sum, per_clip_full, per_clip_pred = [], [], []
        for ci in np.unique(t["clip"][fm]):
            per_clip_sum.append(t["dloss"][per_layer & (t["clip"] == ci)].sum())
            per_clip_full.append(t["dloss"][fm & (t["clip"] == ci)].sum())
            per_clip_pred.append(t["pred"][fm & (t["clip"] == ci)].sum())
        s_, f_, p_ = (np.array(per_clip_sum), np.array(per_clip_full),
                      np.array(per_clip_pred))
        add[name] = {"sum_of_layers": float(s_.mean()), "all_layers": float(f_.mean()),
                     "layer_additivity": float(f_.mean() / s_.mean()) if s_.mean() else None,
                     "predicted": float(p_.mean()),
                     "measured_over_predicted": float(f_.mean() / p_.mean()) if p_.mean() else None,
                     "dminADE": float(np.nanmean(t["dminADE"][fm]))}
        lines.append(f"  {name:20s} pred {p_.mean():+.3e}  measured {f_.mean():+.3e}  "
                     f"(meas/pred {add[name]['measured_over_predicted']:+8.2f})  "
                     f"sum of 36 single-layer cuts {s_.mean():+.3e} "
                     f"(full/sum {add[name]['layer_additivity']:+.2f})  "
                     f"dminADE {add[name]['dminADE']:+.4f}")
    metrics["s2"] = add
    lines.append("")

    # S3 -- does the first-order mass rank the arms the way the measurement does?
    if add:
        names = list(add)
        short = {n: str(n).replace("arm_", "").replace("_full", "") for n in names}
        pred_rank = [short[n] for n in sorted(names, key=lambda n: -add[n]["predicted"])]
        meas_rank = [short[n] for n in sorted(names, key=lambda n: -add[n]["all_layers"])]
        ade_rank = [short[n] for n in sorted(names, key=lambda n: -add[n]["dminADE"])]
        rho = float(spearmanr([add[n]["predicted"] for n in names],
                              [add[n]["all_layers"] for n in names]).statistic)
        rho_a = float(spearmanr([add[n]["predicted"] for n in names],
                                [add[n]["dminADE"] for n in names]).statistic)
        metrics["s3"] = {"pred_rank": pred_rank, "meas_rank": meas_rank,
                         "ade_rank": ade_rank, "spearman_loss": rho, "spearman_ade": rho_a}
        lines += ["S3 ordering of the five arm cuts",
                  f"  predicted {pred_rank}",
                  f"  measured  {meas_rank}   spearman {rho:+.2f}",
                  f"  dminADE   {ade_rank}   spearman {rho_a:+.2f}", ""]

    lines.append("S4 same fits against sum_s|dL_s/dg| instead of |sum_s|")
    for tag, f in abs_fits.items():
        lines.append(f"  {tag:12s} beta {f['beta']:+8.3f} [{f['lo']:+.3f},{f['hi']:+.3f}]  "
                     f"r={f['pearson']:+.3f}")

    plot_slopes(t, out / "plots")
    plot_magnitude(t, out / "plots")
    plot_arms(t, out / "plots")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("saved ->", out)


if __name__ == "__main__":
    main()
