"""Phase P2 analysis: which importance criterion survives pruning best?

The comparison that matters is paired: every config is evaluated on the same
clips, so criteria are compared with a paired test on per-clip ADE rather than by
eyeballing two means whose spread is dominated by clip difficulty.

Usage: python analyze_ablation.py --exp-id ablation_v1
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

REPO = Path("/workspace/alpamayo-model-compression")
BG, INK, MUTED, GRID = "#FAF9F5", "#29261B", "#6B6555", "#E8E6DC"
COL = {"traj": "#2a78d6", "coc": "#e87ba4", "magnitude": "#eda100", "random": "#6B6555"}
LABEL = {"traj": "trajectory (cache)", "coc": "CoC NLL (output)",
         "magnitude": "weight magnitude", "random": "random"}

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})


def paired(per_clip, a, b, key="ade"):
    """Median paired difference a-b and a Wilcoxon p-value."""
    x = np.array(per_clip[a][key])
    y = np.array(per_clip[b][key])
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    d = x - y
    p = wilcoxon(x, y).pvalue if n > 5 and np.any(d != 0) else float("nan")
    return float(np.median(d)), float(p), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", type=str, default="ablation_v1")
    args = ap.parse_args()
    d_in = REPO / "outputs" / args.exp_id
    met = json.loads((d_in / "metrics.json").read_text())
    cfg = json.loads((d_in / "config.json").read_text())
    rows = {r["name"]: r for r in met["rows"]}
    per_clip = met["per_clip"]
    ratios = cfg["ratios"]
    plots = d_in / "plots"
    plots.mkdir(exist_ok=True)
    n = met["n_clips"]

    base = rows["baseline"]
    lines = [f"P2 ablation -- {args.exp_id}, {n} {cfg['eval_split']} clips",
             f"  protocol: {cfg['protocol']}",
             f"  baseline ADE {base['ade_mean']:.4f}  CoC NLL {base['nll_mean']:.4f}", ""]

    # ---- structural freebie ----
    r = rows["drop_last_layer"]
    md, p, _ = paired(per_clip, "drop_last_layer", "baseline")
    lines += ["Layer 35 attention+MLP removed (P1: exactly zero trajectory importance)",
              f"  ADE {r['ade_mean']:.4f} ({r['ade_delta']:+.4f})   paired median "
              f"{md:+.5f}  p={p:.3g}",
              f"  CoC NLL {r['nll_mean']:.4f} ({r['nll_delta']:+.4f})", ""]

    # ---- criterion comparison ----
    for scope in ("all", "late"):
        lines.append(f"Width pruning, scope = {scope} layers")
        lines.append(f"  {'ratio':>6}" + "".join(f"{LABEL[c]:>22}" for c in COL))
        for ratio in ratios:
            cells = []
            for c in COL:
                nm = f"{c}_{scope}_r{int(ratio * 100)}"
                cells.append(f"{rows[nm]['ade_delta']:+.4f}" if nm in rows else "  --  ")
            lines.append(f"  {int(ratio * 100):>5}%" + "".join(f"{v:>22}" for v in cells))
        lines.append("  (values are mean ADE minus baseline; lower is better)")
        lines.append("")
        lines.append(f"  paired traj vs coc, scope={scope}:")
        for ratio in ratios:
            a = f"traj_{scope}_r{int(ratio * 100)}"
            b = f"coc_{scope}_r{int(ratio * 100)}"
            if a in per_clip and b in per_clip:
                md, p, nn = paired(per_clip, a, b)
                verdict = ("traj better" if md < 0 else "coc better") if p < 0.05 else "n.s."
                lines.append(f"    r={int(ratio * 100):>3}%  median dADE {md:+.4f}  "
                             f"p={p:.3g}  -> {verdict}")
        lines.append("")

    # ---- KV groups ----
    lines.append("KV group removal (per layer, weakest first)")
    lines.append(f"  {'n_drop':>7}" + "".join(f"{LABEL[c]:>22}" for c in ("traj", "coc", "random")))
    for k in (1, 2, 3):
        cells = [f"{rows[f'kv_{c}_drop{k}']['ade_delta']:+.4f}"
                 if f"kv_{c}_drop{k}" in rows else "  --  " for c in ("traj", "coc", "random")]
        lines.append(f"  {k:>7}" + "".join(f"{v:>22}" for v in cells))
    lines.append("")

    text = "\n".join(lines)
    (d_in / "p2_summary.txt").write_text(text + "\n")
    print(text)
    make_plots(rows, per_clip, ratios, plots, n, base)
    print("plots ->", plots)


def make_plots(rows, per_clip, ratios, plots, n, base):
    xs = [int(r * 100) for r in ratios]

    # 1) criterion curves, one panel per scope
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for ax, scope, title in [(axes[0], "all", "all 36 layers pruned"),
                             (axes[1], "late", "only layers 22-35 pruned")]:
        for c in COL:
            ys, es = [], []
            for r in ratios:
                nm = f"{c}_{scope}_r{int(r * 100)}"
                ys.append(rows[nm]["ade_delta"] if nm in rows else np.nan)
                es.append(rows[nm]["ade_se"] if nm in rows else 0)
            ax.errorbar(xs, ys, yerr=es, color=COL[c], lw=2, marker="o", ms=4,
                        capsize=3, label=LABEL[c])
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xlabel("pruning ratio (%)"); ax.set_title(title)
        ax.grid(axis="y", color=GRID, lw=0.7)
    axes[0].set_ylabel("ADE increase vs baseline (m)")
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle(f"Which importance criterion survives pruning? (n={n} val clips)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(plots / "criterion_curves.png", dpi=150); plt.close(fig)

    # 2) same but log-scale, since the high ratios blow up
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for ax, scope, title in [(axes[0], "all", "all 36 layers"), (axes[1], "late", "layers 22-35")]:
        for c in COL:
            ys = [rows[f"{c}_{scope}_r{int(r * 100)}"]["ade_delta"]
                  if f"{c}_{scope}_r{int(r * 100)}" in rows else np.nan for r in ratios]
            ax.plot(xs, np.clip(ys, 1e-4, None), color=COL[c], lw=2, marker="o", ms=4,
                    label=LABEL[c])
        ax.set_yscale("log"); ax.set_xlabel("pruning ratio (%)"); ax.set_title(title)
        ax.grid(axis="y", color=GRID, lw=0.7)
    axes[0].set_ylabel("ADE increase (m, log)")
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.tight_layout(); fig.savefig(plots / "criterion_curves_log.png", dpi=150); plt.close(fig)

    # 3) trajectory quality vs CoC quality -- the two-objective trade-off
    fig, ax = plt.subplots(figsize=(7.6, 5))
    for c in COL:
        for scope, mk in [("all", "o"), ("late", "s")]:
            pts = [(rows[f"{c}_{scope}_r{int(r * 100)}"]["nll_delta"],
                    rows[f"{c}_{scope}_r{int(r * 100)}"]["ade_delta"])
                   for r in ratios if f"{c}_{scope}_r{int(r * 100)}" in rows]
            if pts:
                ax.plot(*zip(*pts), color=COL[c], marker=mk, ms=5, lw=1.2, alpha=0.85,
                        label=f"{LABEL[c]} ({scope})")
    d = rows["drop_last_layer"]
    ax.scatter([d["nll_delta"]], [d["ade_delta"]], s=90, color="#008300", zorder=5,
               marker="*", label="drop layer 35 attn+MLP")
    ax.set_xlabel("CoC NLL increase"); ax.set_ylabel("ADE increase (m)")
    ax.set_xscale("symlog", linthresh=0.05); ax.set_yscale("symlog", linthresh=0.02)
    ax.set_title("Reasoning cost vs action cost of the same pruning budget")
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    ax.grid(color=GRID, lw=0.7)
    fig.tight_layout(); fig.savefig(plots / "ade_vs_nll.png", dpi=150); plt.close(fig)

    # 4) KV groups
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    w = 0.26
    for i, c in enumerate(("traj", "coc", "random")):
        ys = [rows[f"kv_{c}_drop{k}"]["ade_delta"] if f"kv_{c}_drop{k}" in rows else np.nan
              for k in (1, 2, 3)]
        es = [rows[f"kv_{c}_drop{k}"]["ade_se"] if f"kv_{c}_drop{k}" in rows else 0
              for k in (1, 2, 3)]
        ax.bar(np.arange(3) + (i - 1) * w, ys, w, yerr=es, capsize=3, color=COL[c],
               label=LABEL[c])
    ax.set_xticks(range(3))
    ax.set_xticklabels(["drop 1 of 8", "drop 2 of 8", "drop 3 of 8"])
    ax.set_ylabel("ADE increase (m)"); ax.axhline(0, color=MUTED, lw=1)
    ax.set_title("KV group removal (joint unit across both towers)")
    ax.legend(frameon=False, fontsize=9); ax.grid(axis="y", color=GRID, lw=0.7)
    fig.tight_layout(); fig.savefig(plots / "kv_groups.png", dpi=150); plt.close(fig)

    # 5) paired per-clip difference at the most informative ratio
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, scope in zip(axes, ("all", "late")):
        best = None
        for r in ratios:
            a, b = f"traj_{scope}_r{int(r * 100)}", f"coc_{scope}_r{int(r * 100)}"
            if a in per_clip and b in per_clip:
                gap = abs(rows[a]["ade_delta"] - rows[b]["ade_delta"])
                if best is None or gap > best[0]:
                    best = (gap, r, a, b)
        if best is None:
            continue
        _, r, a, b = best
        x = np.array(per_clip[a]["ade"]); y = np.array(per_clip[b]["ade"])
        m = min(len(x), len(y))
        ax.scatter(y[:m], x[:m], s=34, color=COL["traj"], edgecolor=BG, lw=1, zorder=3)
        lim = [0, max(x[:m].max(), y[:m].max()) * 1.05]
        ax.plot(lim, lim, color=MUTED, lw=1.2, ls="--")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("ADE with CoC criterion (m)")
        ax.set_ylabel("ADE with trajectory criterion (m)")
        ax.set_title(f"scope={scope}, ratio={int(r * 100)}%  (below the line = traj better)")
        ax.grid(color=GRID, lw=0.7, zorder=0)
    fig.tight_layout(); fig.savefig(plots / "paired_scatter.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
