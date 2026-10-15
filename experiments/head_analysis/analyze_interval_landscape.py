"""Interval-deletion landscape per objective, and the two-interval additivity test.

plans/2026-10-13_interval-landscape-pilot.md. Reads the shards run_token_ablation.py --arm-masks
wrote for the masks of make_interval_masks.py. A config's damage is the paired per-clip
difference to the dense model on the same clip, text and noise:

  dADE   minADE over K denoisings          (trajectory, sampled)
  rFM    fm / fm_dense - 1                 (trajectory, GT-anchored loss)
  dNLL   CoC NLL of the dense rollout      (reasoning, teacher-forced)

Medians over clips are the reading (minADE deltas are heavy-tailed); means with a bootstrap CI
are written alongside.

  G0  layer 35 bypass leaves fm and minADE untouched on every clip; the layers the stored
      150-clip single-layer run flags (0, 1, 6, 15, 16) are the pilot's top five by dADE
  H1  [23, 34]: median rFM < 5% and median dNLL > 0.10 nat
  H2  longest interval with median dADE < 0.05 m: starting at >= 17 vs ending at <= 16
  H3  D(B|A) = D(A u B) - D(A) against D(B): MAE, Pearson, Spearman per A and metric

Usage:
  python experiments/head_analysis/analyze_interval_landscape.py --masks ivl_masks_v1 \
      --shards ivl_pilot_v1_s0 ivl_pilot_v1_s1 ivl_pilot_v1_s2 ivl_pilot_v1_s3 --out ivl_pilot_v1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).parent))

from analyze_gradient_anatomy import C1, C3, MUTED, plt
from analyze_token_ablation import merge
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parents[2]
METRICS = ("dADE", "rFM", "dNLL")
LABEL = {"dADE": "ΔminADE (m)", "rFM": "FM loss, relative change", "dNLL": "ΔCoC NLL (nat)"}
FLAGGED = [0, 1, 6, 15, 16]
TOL_ADE = 0.05
DAMAGE = LinearSegmentedColormap.from_list("damage", ["#F3F1E9", "#E9B79F", "#D97757", "#7A3521", "#2B130C"])


def stats(x, rng, n_boot):
    x = np.asarray(x, float)
    boot = rng.choice(x, (n_boot, len(x)), replace=True).mean(1)  # (n_boot,)
    return {"median": float(np.median(x)), "mean": float(x.mean()),
            "mean_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "frac_worse": float((x > 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True)
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    per, ids, buckets = merge(args.shards)
    meta = json.loads((REPO / "outputs" / args.masks / "intervals.json").read_text())
    cfgs, bounds = meta["configs"], meta["bounds"]
    n = len(ids)
    rng = np.random.default_rng(0)
    dense = {k: np.array(per["dense"][k], float) for k in ("fm", "nll", "ade")}  # each (n,)

    raw = {}  # name -> metric -> (n,) paired difference to dense
    for name in cfgs:
        r = per[f"arm_{name}"]
        raw[name] = {"dADE": np.array(r["ade"], float) - dense["ade"],
                     "rFM": np.array(r["fm"], float) / dense["fm"] - 1.0,
                     "dNLL": np.array(r["nll"], float) - dense["nll"]}
    table = {name: {m: stats(raw[name][m], rng, args.n_boot) for m in METRICS} for name in cfgs}
    med = {name: {m: table[name][m]["median"] for m in METRICS} for name in cfgs}

    def one(a, b):
        return f"i{a}_{b}"

    # ---- G0
    l35 = raw[one(35, 35)]
    g0a = {"max_abs_rFM": float(np.abs(l35["rFM"]).max()), "max_abs_dADE": float(np.abs(l35["dADE"]).max())}
    single_ade = np.array([med[one(li, li)]["dADE"] for li in range(36)])  # (36,)
    top5 = [int(i) for i in np.argsort(-single_ade)[:5]]
    stored = json.loads((REPO / "outputs" / "depth_ablation_summary" / "metrics.json").read_text())["layers"]
    stored_ade = np.array([stored[str(li)]["dADE_median"] for li in range(36)])
    g0 = {"layer35": g0a, "layer35_pass": g0a["max_abs_rFM"] == 0.0 and g0a["max_abs_dADE"] == 0.0,
          "top5_single_dADE": top5, "flagged_in_top5": len(set(top5) & set(FLAGGED)),
          "top5_pass": len(set(top5) & set(FLAGGED)) >= 4,
          "spearman_vs_stored_150clip": float(spearmanr(single_ade, stored_ade)[0])}
    g0["pass"] = g0["layer35_pass"] and g0["top5_pass"]

    # ---- H1
    late = med[one(23, 34)]
    h1 = {"interval": [23, 34], "rFM_median": late["rFM"], "dNLL_median": late["dNLL"],
          "dADE_median": late["dADE"], "pass": abs(late["rFM"]) < 0.05 and late["dNLL"] > 0.10}

    # ---- H2
    solo = {name: c["intervals"][0] for name, c in cfgs.items() if c["group"] != "pair"}
    tol = {name: iv for name, iv in solo.items() if med[name]["dADE"] < TOL_ADE}

    def longest(pred):
        cand = [(b - a + 1, [a, b]) for a, b in tol.values() if pred(a, b)]
        return max(cand) if cand else (0, None)

    early_len, early_iv = longest(lambda a, b: b <= 16)
    late_len, late_iv = longest(lambda a, b: a >= 17)
    h2 = {"tolerance_m": TOL_ADE, "longest_ending_le16": {"len": early_len, "interval": early_iv},
          "longest_starting_ge17": {"len": late_len, "interval": late_iv},
          "longest_any": dict(zip(("len", "interval"), longest(lambda a, b: True))),
          "pass": late_len > early_len}

    # ---- H3
    pairs = {name: c["intervals"] for name, c in cfgs.items() if c["group"] == "pair"}
    a_set = sorted({tuple(iv[0]) for iv in pairs.values()})
    h3, pts = {}, {}
    for a in a_set:
        key = one(*a)
        h3[key], pts[key] = {}, {}
        for m in METRICS:
            alone, marg = [], []
            for name, (ia, ib) in pairs.items():
                if tuple(ia) != a:
                    continue
                alone.append(float(np.median(raw[one(*ib)][m])))
                marg.append(float(np.median(raw[name][m] - raw[key][m])))  # per-clip marginal, then median
            alone, marg = np.array(alone), np.array(marg)  # each (n_B,)
            pts[key][m] = (alone, marg)
            h3[key][m] = {"n_B": len(alone), "mae": float(np.abs(alone - marg).mean()),
                          "scale": float(np.abs(alone).mean()),
                          "pearson": float(pearsonr(alone, marg)[0]),
                          "spearman": float(spearmanr(alone, marg)[0]),
                          "D_A": med[key][m], "alone": alone.tolist(), "marginal": marg.tolist()}

    # ---- main-run gates M1-M5 (plans/2026-10-13_interval-landscape-pilot.md, "Main run")
    def ci_has_zero(x):
        lo, hi = stats(x, rng, args.n_boot)["mean_ci"]
        return lo <= 0.0 <= hi

    m1_rows = {}
    for s in range(2, 35):
        k = one(s, 34)
        if k in raw:
            ok = (abs(med[k]["rFM"]) < 0.05 and abs(med[k]["dADE"]) < TOL_ADE and ci_has_zero(raw[k]["dADE"]))
            m1_rows[s] = {"rFM": med[k]["rFM"], "dADE": med[k]["dADE"], "dNLL": med[k]["dNLL"], "ok": bool(ok)}
    passing = [s for s, r in m1_rows.items() if r["ok"]]
    s_star = min(passing) if passing else None
    m1 = {"s_star": s_star, "passing_starts": passing, "rows": m1_rows, "pass": s_star in (25, 26, 27)}

    mid, lat = med[one(8, 16)], med[one(26, 34)]
    m2 = {"mid": {"interval": [8, 16], **mid}, "late": {"interval": [26, 34], **lat},
          "pass": bool(mid["dADE"] > 1.0 and mid["dNLL"] < 0.2 and abs(lat["dADE"]) < TOL_ADE and lat["dNLL"] > 2.0)}

    def excess(a, b):
        """Per clip: dADE of the whole interval minus the sum over its single layers. (n,)"""
        return raw[one(a, b)]["dADE"] - sum(raw[one(li, li)]["dADE"] for li in range(a, b + 1))

    ex_mid, ex_late = excess(8, 16), excess(26, 34)
    blocks = sum(med[one(s, s + 2)]["dADE"] for s in (8, 11, 14))
    m3 = {"excess_mid": stats(ex_mid, rng, args.n_boot),
          "p_greater": float(wilcoxon(ex_mid, alternative="greater").pvalue),
          "whole_median": mid["dADE"], "sum_3layer_blocks": blocks, "ratio_to_blocks": mid["dADE"] / blocks,
          "sum_singles": float(sum(med[one(li, li)]["dADE"] for li in range(8, 17))),
          "excess_late": stats(ex_late, rng, args.n_boot), "late_ci_has_zero": ci_has_zero(ex_late)}
    m3["pass"] = bool(m3["excess_mid"]["median"] > 0 and m3["p_greater"] < 0.01
                      and m3["ratio_to_blocks"] >= 2 and m3["late_ci_has_zero"])

    late_b = [(23, 25), (26, 28), (29, 31), (32, 34), (23, 28), (29, 34)]
    a13 = one(8, 13)
    marg = np.mean([raw[f"{a13}+{one(*b)}"]["dADE"] - raw[a13]["dADE"] for b in late_b], 0)  # (n,)
    m4 = {**stats(marg, rng, args.n_boot), "p_two_sided": float(wilcoxon(marg).pvalue)}
    m4["pass"] = bool(abs(m4["median"]) <= 0.05)
    m4["repair_reproduced"] = bool(m4["median"] < -0.05 and m4["p_two_sided"] < 0.01)

    band = {"dADE": 0.05, "rFM": 0.05, "dNLL": 0.05}
    m5 = {}
    for key in h3:
        m5[key] = {}
        for m in METRICS:
            alone, mg = pts[key][m]
            use = np.abs(alone) >= band[m]
            rel = float(np.abs(mg[use] - alone[use]).mean() / np.abs(alone[use]).mean()) if use.any() else None
            m5[key][m] = {"n_B_used": int(use.sum()), "rel_mae": rel,
                          "additive": None if rel is None else bool(rel < 0.25)}
    m0 = {"layer35_pass": g0["layer35_pass"], "top5_single_dADE": top5,
          "spearman_vs_stored_150clip": g0["spearman_vs_stored_150clip"],
          "pass": bool(g0["layer35_pass"] and set(top5) == set(FLAGGED)
                       and g0["spearman_vs_stored_150clip"] >= 0.8)}
    main_gates = {"M0": m0, "M1": m1, "M2": m2, "M3": m3, "M4": m4, "M5": m5}

    out = REPO / "outputs" / args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)

    # ---- plot 1: single-layer diagonal
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 7.2), sharex=True)
    for ax, m in zip(axes, METRICS):
        v = np.array([med[one(li, li)][m] for li in range(36)])
        ax.bar(range(36), v, color=[C3 if li in FLAGGED else C1 for li in range(36)], width=0.8)
        ax.set_yscale("symlog", linthresh={"dADE": 0.05, "rFM": 0.05, "dNLL": 0.01}[m])
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_ylabel(LABEL[m])
    axes[0].set_title(f"single-layer bypass, median over {n} clips (pink: flagged by the stored 150-clip run)")
    axes[-1].set_xlabel("layer bypassed")
    fig.tight_layout()
    fig.savefig(out / "plots" / "single_layer.png", dpi=150)
    plt.close(fig)

    # ---- plot 2: (start, end) landscape on the stride grid
    starts, ends = bounds[:-1], [b - 1 for b in bounds[1:]]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    for ax, m in zip(axes, METRICS):
        z = np.full((len(ends), len(starts)), np.nan)  # (end, start)
        for name, (a, b) in solo.items():
            if a in starts and b in ends:
                z[ends.index(b), starts.index(a)] = med[name][m]
        vmax = {"dADE": 1.0, "rFM": 1.0, "dNLL": 2.0}[m]
        im = ax.imshow(z, origin="lower", cmap=DAMAGE, vmin=0, vmax=vmax, aspect="auto")
        if len(starts) <= 13:  # the stride-1 map is too dense to annotate
            for (ei, si), val in np.ndenumerate(z):
                if not np.isnan(val):
                    ax.text(si, ei, f"{val:.2f}" if abs(val) < 10 else f"{val:.0f}", ha="center", va="center",
                            fontsize=7.5, color="white" if val > 0.5 * vmax else "black")
        step = 1 if len(starts) <= 13 else 3
        ax.set_xticks(range(0, len(starts), step), starts[::step])
        ax.set_yticks(range(0, len(ends), step), ends[::step])
        ax.set_xlabel("first layer bypassed")
        ax.set_ylabel("last layer bypassed")
        ax.set_title(f"{LABEL[m]} (colour clipped at {vmax:g})")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"interval bypass [first, last], median over {n} clips")
    fig.tight_layout()
    fig.savefig(out / "plots" / "landscape.png", dpi=150)
    plt.close(fig)

    # ---- plot 2b: which objective tolerates each interval (median below the noise band)
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    cats = np.full((len(ends), len(starts)), np.nan)  # (end, start): 0 neither, 1 traj, 2 nll, 3 both
    for name, (a, b) in solo.items():
        if a in starts and b in ends:
            t, c = med[name]["dADE"] < TOL_ADE, med[name]["dNLL"] < 0.05
            cats[ends.index(b), starts.index(a)] = int(t) + 2 * int(c)
    cols = ["#E4DFD3", "#D97757", C1, "#4F7A4C"]
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    ax.imshow(cats, origin="lower", cmap=ListedColormap(cols), vmin=-0.5, vmax=3.5, aspect="auto")
    step = 1 if len(starts) <= 13 else 3
    ax.set_xticks(range(0, len(starts), step), starts[::step])
    ax.set_yticks(range(0, len(ends), step), ends[::step])
    ax.set_xlabel("first layer bypassed")
    ax.set_ylabel("last layer bypassed")
    ax.legend(handles=[Patch(color=cols[1], label=f"trajectory only (ΔminADE < {TOL_ADE} m)"),
                       Patch(color=cols[2], label="CoC only (ΔNLL < 0.05 nat)"),
                       Patch(color=cols[3], label="both"), Patch(color=cols[0], label="neither")],
              frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title(f"which objective tolerates the interval (median, {n} clips)")
    fig.tight_layout()
    fig.savefig(out / "plots" / "tolerated.png", dpi=150)
    plt.close(fig)

    # ---- plot 3: fixed-length slices
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    for ax, m in zip(axes, METRICS):
        for length, col in zip((3, 6, 9, 12), (C1, "#008300", "#eda100", C3)):
            xs = [a for a, b in solo.values() if b - a + 1 == length]
            ys = [med[one(a, a + length - 1)][m] for a in xs]
            order = np.argsort(xs)
            ax.plot(np.array(xs)[order], np.array(ys)[order], "o-", color=col, label=f"{length} layers", ms=4)
        ax.set_yscale("symlog", linthresh={"dADE": 0.05, "rFM": 0.05, "dNLL": 0.01}[m])
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_xlabel("first layer bypassed")
        ax.set_ylabel(LABEL[m])
    axes[0].legend(frameon=False)
    fig.suptitle("damage against position at fixed interval length")
    fig.tight_layout()
    fig.savefig(out / "plots" / "length_slices.png", dpi=150)
    plt.close(fig)

    # ---- plot 4: additivity
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, m in zip(axes, METRICS):
        lo, hi = np.inf, -np.inf
        for key, col in zip(pts, (C1, C3, "#eda100")):
            alone, marg = pts[key][m]
            ax.scatter(alone, marg, color=col, s=28,
                       label=f"A={key[1:].replace('_', '-')}  ρ={h3[key][m]['spearman']:+.2f}")
            lo, hi = min(lo, alone.min(), marg.min()), max(hi, alone.max(), marg.max())
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8)
        ax.set_xlabel(f"D(B) alone — {LABEL[m]}")
        ax.set_ylabel("D(B | A)")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("additivity: marginal damage of B after A against B alone (line = additive)")
    fig.tight_layout()
    fig.savefig(out / "plots" / "additivity.png", dpi=150)
    plt.close(fig)

    cfg0 = json.loads((REPO / "outputs" / args.shards[0] / "config.json").read_text())
    (out / "config.json").write_text(json.dumps({
        "plan": "plans/2026-10-13_interval-landscape-pilot.md", "masks": args.masks,
        "shards": args.shards, "n_clips": n, "k_samples": cfg0["k_samples"], "gpu": cfg0["gpu"],
        "manifest": cfg0["manifest"], "n_boot": args.n_boot, "tolerance_m": TOL_ADE}, indent=2))
    (out / "metrics.json").write_text(json.dumps({
        "n_clips": n, "clip_ids": ids, "buckets": buckets,
        "dense": {k: {"median": float(np.median(v)), "mean": float(v.mean())} for k, v in dense.items()},
        "gates": {"G0": g0, "H1": h1, "H2": h2, "H3": h3, **main_gates},
        "configs": {name: {**cfgs[name], **table[name]} for name in cfgs}}, indent=2))

    lines = [f"interval-deletion landscape: {n} clips, {len(cfgs)} configs, {cfg0['gpu']}",
             (f"dense: minADE median {np.median(dense['ade']):.3f}  fm {np.median(dense['fm']):.4f}  "
             f"nll {np.median(dense['nll']):.4f}"), "",
             (f"G0 {'PASS' if g0['pass'] else 'FAIL'}: layer 35 max|rFM| {g0a['max_abs_rFM']:.2e} "
             f"max|dADE| {g0a['max_abs_dADE']:.2e}; top-5 single dADE {top5} "
             f"({g0['flagged_in_top5']}/5 flagged); Spearman vs stored {g0['spearman_vs_stored_150clip']:+.3f}"),
             (f"H1 {'PASS' if h1['pass'] else 'FAIL'}: [23,34] rFM {h1['rFM_median']:+.3f}  "
             f"dNLL {h1['dNLL_median']:+.3f}  dADE {h1['dADE_median']:+.3f}"),
             (f"H2 {'PASS' if h2['pass'] else 'FAIL'}: longest tolerated (dADE<{TOL_ADE}) ending<=16 "
             f"{early_iv} ({early_len}), starting>=17 {late_iv} ({late_len})"), "H3 additivity:"]
    for key in h3:
        for m in METRICS:
            s = h3[key][m]
            lines.append(f"  A={key:7s} {m:5s} D(A)={s['D_A']:+.4f}  MAE {s['mae']:.4f} (scale {s['scale']:.4f})  "
                         f"pearson {s['pearson']:+.3f}  spearman {s['spearman']:+.3f}")
    def verdict(g):
        return "PASS" if g["pass"] else "FAIL"

    lines += ["", "main-run gates:",
              (f"M0 {verdict(m0)}: layer 35 exact zero {m0['layer35_pass']}; top-5 {top5}; "
               f"Spearman vs stored {m0['spearman_vs_stored_150clip']:+.3f}"),
              f"M1 {verdict(m1)}: s* = {s_star}; starts passing {passing}",
              (f"M2 {verdict(m2)}: [8,16] dADE {mid['dADE']:+.3f} dNLL {mid['dNLL']:+.3f} | "
               f"[26,34] dADE {lat['dADE']:+.3f} dNLL {lat['dNLL']:+.3f}"),
              (f"M3 {verdict(m3)}: [8,16] whole - sum of singles: median {m3['excess_mid']['median']:+.3f} "
               f"mean {m3['excess_mid']['mean']:+.3f} p={m3['p_greater']:.2e}; whole/blocks "
               f"{m3['ratio_to_blocks']:.2f} ({mid['dADE']:+.3f} / {blocks:+.3f}); [26,34] excess mean "
               f"{m3['excess_late']['mean']:+.3f} CI {m3['excess_late']['mean_ci']}"),
              (f"M4 {verdict(m4)}: late B after [8,13], pooled: median {m4['median']:+.3f} "
               f"mean {m4['mean']:+.3f} p={m4['p_two_sided']:.3f}; repair reproduced: {m4['repair_reproduced']}"),
              "M5 additivity by value (rel MAE over B's outside the noise band; additive < 0.25):"]
    for key, by_metric in m5.items():
        lines.append(f"  A={key:7s} " + "  ".join(
            f"{m} {'n/a' if v['rel_mae'] is None else format(v['rel_mae'], '.2f')} (n={v['n_B_used']})"
            for m, v in by_metric.items()))
    for st in sorted(m1_rows):
        r = m1_rows[st]
        lines.append(f"  [{st},34] rFM {r['rFM']:+.3f} dADE {r['dADE']:+.3f} dNLL {r['dNLL']:+.3f} "
                     f"{'ok' if r['ok'] else ''}")
    n_head = len(lines)
    lines += ["", f"{'config':16s} {'group':7s} {'dADE med':>9s} {'mean':>8s} {'rFM med':>8s} {'dNLL med':>9s}"]
    for name in cfgs:
        t = table[name]
        lines.append(f"{name:16s} {cfgs[name]['group']:7s} {t['dADE']['median']:+9.4f} {t['dADE']['mean']:+8.3f} "
                     f"{t['rFM']['median']:+8.3f} {t['dNLL']['median']:+9.4f}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:n_head]))
    print("saved ->", out)


if __name__ == "__main__":
    main()
