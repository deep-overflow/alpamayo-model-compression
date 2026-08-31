"""G1 of plans/2026-08-29_cache-targeted-reconstruction.md across arms.

Reads run_cacheproxy.py outputs for several slim arms (same clips, same seeds), and
judges: sensitivity-weighted cache shift per arm (median over clips, paired ratio vs the
reference arm) and the cache-only cost A10-A00 (paired median dminADE@K with CI, Wilcoxon).

Usage:
  .venv/bin/python experiments/head_analysis/analyze_cacheproxy.py \
      --arms dual=cacheproxy_dual dualr=cacheproxy_dualr dualrc_s16=cacheproxy_dualrc_s16 \
      --ref dual --out cacheproxy_analysis
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

BG, INK, MUTED, GRID = "#FAF9F5", "#29261B", "#6B6555", "#E8E6DC"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.size": 9, "axes.grid": True, "axes.axisbelow": True,
})
REPO = Path(__file__).resolve().parents[2]


def med_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    b = np.median(x[rng.integers(0, len(x), (n, len(x)))], 1)
    return float(np.median(x)), *[float(q) for q in np.percentile(b, [2.5, 97.5])]


def load(exp):
    m = json.loads((REPO / "outputs" / exp / "metrics.json").read_text())
    z = np.load(REPO / "outputs" / exp / "cacheproxy.npz")
    n = m["n_clips"]
    per = {k: np.array(m[k][:n]) for k in ("ade_A00", "ade_A10", "fde_A00", "fde_A10",
                                             "wshift_v", "wshift_k", "shift_v_mean",
                                             "nll_dense", "nll_slim")}
    return m["clip_ids"][:n], per, z["rel_v"], m["slim_config"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="name=exp-id")
    ap.add_argument("--ref", default="dual")
    ap.add_argument("--out", default="cacheproxy_analysis")
    ap.add_argument("--val-arms", nargs="*", default=[],
                    help="paper_numbers ARMS keys to summarise vs baseline (G2)")
    ap.add_argument("--sets", nargs="*", default=["indist"],
                    help="sets for --val-arms: indist (val500), test, oodval")
    args = ap.parse_args()
    out = REPO / "outputs" / args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)
    arms = {}
    for spec in args.arms:
        name, exp = spec.split("=")
        arms[name] = load(exp)
    ids = sorted(set.intersection(*[set(a[0]) for a in arms.values()]))
    idx = {name: [a[0].index(i) for i in ids] for name, a in arms.items()}
    res, lines = {"n_clips": len(ids), "ref": args.ref, "arms": {}}, [
        f"cache proxy (G1) -- {len(ids)} common clips, ref = {args.ref}", ""]
    ref = arms[args.ref]
    for name, (cids, per, rel_v, cfg) in arms.items():
        sel = idx[name]
        d = per["ade_A10"][sel] - per["ade_A00"][sel]
        try:
            p = float(wilcoxon(d).pvalue)
        except ValueError:
            p = float("nan")
        r = {"config": cfg, "A10_minus_A00": med_ci(d), "A10_minus_A00_mean": float(d.mean()),
             "wilcoxon_p": p, "wshift_v": med_ci(per["wshift_v"][sel]),
             "shift_v_mean": med_ci(per["shift_v_mean"][sel]),
             "nll_slim_minus_dense": med_ci(per["nll_slim"][sel] - per["nll_dense"][sel]),
             "rel_v_by_band": [float(rel_v[a:b].mean()) for a, b in
                               ((1, 8), (8, 16), (16, 24), (24, 36))]}
        if name != args.ref:
            rsel = idx[args.ref]
            ratio = per["wshift_v"][sel] / np.maximum(ref[1]["wshift_v"][rsel], 1e-12)
            r["wshift_ratio_vs_ref"] = med_ci(ratio)
            dd = d - (ref[1]["ade_A10"][rsel] - ref[1]["ade_A00"][rsel])
            r["cache_cost_minus_ref"] = med_ci(dd)
            r["G1_shift_pass"] = r["wshift_ratio_vs_ref"][2] < 0.5
            r["G1_cost_pass"] = r["A10_minus_A00"][1] <= 0 <= r["A10_minus_A00"][2]
        res["arms"][name] = r
        lines.append(
            f"{name:12s} ({cfg}): A10-A00 median {r['A10_minus_A00'][0]:+.4f} "
            f"[{r['A10_minus_A00'][1]:+.4f},{r['A10_minus_A00'][2]:+.4f}] mean "
            f"{r['A10_minus_A00_mean']:+.4f} p={p:.2g} | weighted V shift median "
            f"{r['wshift_v'][0]:.4f} | plain V shift {r['shift_v_mean'][0]:.3f} | rel_v by band "
            f"{' '.join(f'{v:.3f}' for v in r['rel_v_by_band'])} | dNLL {r['nll_slim_minus_dense'][0]:+.4f}"
            + (f" | vs {args.ref}: shift ratio {r['wshift_ratio_vs_ref'][0]:.3f} "
               f"[{r['wshift_ratio_vs_ref'][1]:.3f},{r['wshift_ratio_vs_ref'][2]:.3f}] "
               f"(G1 shift {'PASS' if r['G1_shift_pass'] else 'FAIL'}), cache cost - ref "
               f"{r['cache_cost_minus_ref'][0]:+.4f} [{r['cache_cost_minus_ref'][1]:+.4f},"
               f"{r['cache_cost_minus_ref'][2]:+.4f}] (G1 cost {'PASS' if r['G1_cost_pass'] else 'FAIL'})"
               if name != args.ref else ""))
    if args.val_arms:
        import sys
        sys.path.insert(0, str(REPO / "experiments" / "evaluation"))
        import paper_numbers as pn
        for s_name in args.sets:
          base = pn.load(*pn.ARMS["baseline"][s_name])
          key = "val500" if s_name == "indist" else s_name
          ba = np.array([pn.at6(r, "ade_rollout_k") for r in base.values()])
          bf = np.array([pn.at6(r, "fde_rollout_k") for r in base.values()])
          res[key] = {"baseline": {"n": len(base), "minADE6": float(ba.mean()),
                                    "minFDE6": float(bf.mean())}}
          lines.append("")
          lines.append(f"{key} baseline     n={len(base)} ADE {ba.mean():.4f} FDE {bf.mean():.4f}")
          for arm in args.val_arms:
            spec = pn.ARMS.get(arm, {}).get(s_name)
            rows = pn.load(*spec) if spec else {}
            if not rows:
                lines.append(f"{key} {arm}: no rows yet")
                continue
            cids = sorted(set(base) & set(rows))
            a = np.array([pn.at6(rows[i], "ade_rollout_k") for i in cids])
            b = np.array([pn.at6(base[i], "ade_rollout_k") for i in cids])
            f = np.array([pn.at6(rows[i], "fde_rollout_k") for i in cids])
            d = a - b
            try:
                p = float(wilcoxon(d).pvalue)
            except ValueError:
                p = float("nan")
            degen = float(np.mean([rows[i]["coc_degenerate"] for i in cids]))
            res[key][arm] = {"n": len(cids), "minADE6": float(a.mean()),
                             "minFDE6": float(f.mean()), "d_ade": med_ci(d),
                             "d_ade_mean": float(d.mean()), "p": p, "degen": degen}
            r = res[key][arm]
            lines.append(f"{key} {arm:12s} n={r['n']} ADE {r['minADE6']:.4f} FDE {r['minFDE6']:.4f} "
                         f"dADE {r['d_ade'][0]:+.4f} [{r['d_ade'][1]:+.4f},{r['d_ade'][2]:+.4f}] "
                         f"mean {r['d_ade_mean']:+.4f} p={p:.2g} degen {degen:.3f}")
    text = "\n".join(lines)
    print(text)
    (out / "cacheproxy_summary.txt").write_text(text + "\n")
    (out / "metrics_analysis.json").write_text(json.dumps(res, indent=1))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = list(arms)
    for k, name in enumerate(names):
        r = res["arms"][name]
        m, lo, hi = r["A10_minus_A00"]
        axes[0].errorbar(k, m, yerr=[[m - lo], [hi - m]], fmt="o", color=C1, capsize=4)
        m, lo, hi = r["wshift_v"]
        axes[1].errorbar(k, m, yerr=[[m - lo], [hi - m]], fmt="o", color=C2, capsize=4)
    for ax, t in zip(axes, ("cache-only cost: A10 - A00 (paired median dminADE)",
                            "sensitivity-weighted V shift (median over clips)")):
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_title(t)
    axes[0].axhline(0, color=INK, lw=0.8)
    fig.tight_layout()
    fig.savefig(out / "plots" / "cacheproxy_arms.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for name, color in zip(names, (C1, C2, C3, C4, MUTED)):
        ax.plot(range(36), arms[name][2].mean(1), "o-", ms=3, color=color, label=name)
    ax.set_xlabel("VLM layer")
    ax.set_ylabel("||dV||/||V|| (group mean)")
    ax.set_title("cache shift by layer, per arm")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "plots" / "cacheproxy_shift_by_layer.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
