"""H1 test: do the CoC (output) and trajectory (cache) rankings actually differ?

Correlations are computed WITHIN each layer. A global correlation over all layers
is confounded by layer depth -- both objectives grow with depth, which manufactures
a high correlation even when the per-layer orderings are unrelated. That confound
already bit us once in the attention analysis, so it is controlled here by design.

Usage: python analyze_importance.py --exp-id importance_v1
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

REPO = Path("/workspace/alpamayo-model-compression")
BG, INK, MUTED, GRID = "#FAF9F5", "#29261B", "#6B6555", "#E8E6DC"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})

UNITS = [("vlm_q", "VLM Q head", C1), ("vlm_mlp", "VLM MLP channel", C2),
         ("kv_k", "KV group (K)", C3), ("kv_v", "KV group (V)", C4)]


def overlap_at_k(a, b, frac):
    """Fraction of the top-frac set shared by two score vectors (per layer)."""
    k = max(int(round(len(a) * frac)), 1)
    ta = set(np.argsort(-a)[:k])
    tb = set(np.argsort(-b)[:k])
    return len(ta & tb) / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", type=str, default="importance_v1")
    args = ap.parse_args()
    d_in = REPO / "outputs" / args.exp_id
    d = np.load(d_in / "importance.npz")
    n_clips = json.loads((d_in / "metrics.json").read_text())["n_clips"]
    plots = d_in / "plots"
    plots.mkdir(exist_ok=True)

    lines = [f"H1 test -- CoC (output) vs trajectory (cache) importance rankings",
             f"  {args.exp_id}, n={n_clips} calibration clips", ""]
    result = {}

    for key, label, _ in UNITS:
        coc, traj = d[f"coc_{key}"], d[f"traj_{key}"]  # (L, U)
        n_layers = coc.shape[0]
        rhos, ov20, ov50 = [], [], []
        for li in range(n_layers):
            a, b = coc[li], traj[li]
            if a.std() == 0 or b.std() == 0:
                continue  # e.g. the last VLM layer carries no trajectory gradient
            rhos.append(spearmanr(a, b).statistic)
            ov20.append(overlap_at_k(a, b, 0.20))
            ov50.append(overlap_at_k(a, b, 0.50))
        glob = spearmanr(coc.ravel(), traj.ravel()).statistic
        result[key] = {
            "within_layer_spearman_mean": float(np.mean(rhos)),
            "within_layer_spearman_median": float(np.median(rhos)),
            "within_layer_spearman_min": float(np.min(rhos)),
            "within_layer_spearman_max": float(np.max(rhos)),
            "top20_overlap_mean": float(np.mean(ov20)),
            "top50_overlap_mean": float(np.mean(ov50)),
            "global_spearman_confounded": float(glob),
            "n_layers_scored": len(rhos),
        }
        r = result[key]
        lines += [
            f"{label}",
            f"  within-layer Spearman   mean {r['within_layer_spearman_mean']:+.3f} "
            f"median {r['within_layer_spearman_median']:+.3f} "
            f"range [{r['within_layer_spearman_min']:+.3f}, {r['within_layer_spearman_max']:+.3f}]",
            f"  top-20% overlap         {r['top20_overlap_mean'] * 100:.1f}%   "
            f"(random baseline 20.0%)",
            f"  top-50% overlap         {r['top50_overlap_mean'] * 100:.1f}%   "
            f"(random baseline 50.0%)",
            f"  global Spearman         {glob:+.3f}  <- confounded by layer depth, "
            f"do not use",
            f"  layers scored           {r['n_layers_scored']}/{n_layers}",
            "",
        ]

    # depth-resolved agreement: the aggregate mean hides a strong depth gradient,
    # so band it rather than collapsing to one number
    bands = [(0, 6, "0-5"), (6, 18, "6-17"), (18, 30, "18-29"), (30, 36, "30-35")]
    lines.append("Depth-resolved within-layer Spearman")
    lines.append(f"  {'band':<10}" + "".join(f"{lbl:>16}" for _, lbl, _ in UNITS))
    band_tab = {}
    for lo, hi, name in bands:
        row = []
        for key, _, _ in UNITS:
            coc, traj = d[f"coc_{key}"], d[f"traj_{key}"]
            rs = [spearmanr(coc[i], traj[i]).statistic for i in range(lo, min(hi, coc.shape[0]))
                  if coc[i].std() > 0 and traj[i].std() > 0]
            row.append(float(np.mean(rs)) if rs else float("nan"))
        band_tab[name] = row
        lines.append(f"  {name:<10}" + "".join(f"{v:>+16.3f}" for v in row))
    lines.append("")
    result["_depth_bands"] = {"bands": [n for _, _, n in bands],
                              "units": [k for k, _, _ in UNITS], "values": band_tab}

    q = result["vlm_q"]["within_layer_spearman_mean"]
    m = result["vlm_mlp"]["within_layer_spearman_mean"]
    late = np.mean([band_tab["30-35"][0], band_tab["30-35"][1]])
    early = np.mean([band_tab["0-5"][0], band_tab["0-5"][1]])
    lines += ["VERDICT",
              f"  aggregate within-layer rho: Q head {q:+.3f}, MLP {m:+.3f}",
              f"  early layers (0-5)  rho = {early:+.3f}",
              f"  late  layers (30-35) rho = {late:+.3f}",
              f"  early-to-late drop  = {early - late:+.3f}"]
    if late < 0.4 and early - late > 0.3:
        lines.append("  -> H1 SUPPORTED IN THE LATE LAYERS: the objectives agree early and")
        lines.append("     diverge with depth, which is where importance is largest. A")
        lines.append("     cache-specific criterion should be tested there first.")
    elif max(q, m) < 0.5:
        lines.append("  -> H1 SUPPORTED: the two objectives rank units differently throughout.")
    elif min(q, m) > 0.8:
        lines.append("  -> H1 REJECTED: the rankings are near-identical; "
                     "a cache-specific criterion buys nothing.")
    else:
        lines.append("  -> INCONCLUSIVE: partial overlap with no clear depth structure.")

    text = "\n".join(lines)
    (d_in / "h1_summary.txt").write_text(text + "\n")
    (d_in / "h1_metrics.json").write_text(json.dumps(result, indent=2))
    print(text)
    make_plots(d, plots, result, n_clips)
    print("plots ->", plots)


def make_plots(d, plots, result, n_clips):
    # 1) within-layer correlation per unit type, by depth
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for key, label, col in UNITS:
        coc, traj = d[f"coc_{key}"], d[f"traj_{key}"]
        xs, ys = [], []
        for li in range(coc.shape[0]):
            if coc[li].std() == 0 or traj[li].std() == 0:
                continue
            xs.append(li)
            ys.append(spearmanr(coc[li], traj[li]).statistic)
        ax.plot(xs, ys, color=col, lw=1.8, marker="o", ms=3, label=label)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("layer"); ax.set_ylabel("within-layer Spearman (CoC vs trajectory)")
    ax.set_title(f"Do the two objectives agree? Per-layer rank correlation (n={n_clips} clips)")
    ax.legend(frameon=False, fontsize=8.5, ncol=2)
    ax.grid(axis="y", color=GRID, lw=0.7)
    fig.tight_layout(); fig.savefig(plots / "within_layer_agreement.png", dpi=150); plt.close(fig)

    # 2) importance vs depth for each objective (shows the confound directly)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, label in [(axes[0], "vlm_q", "VLM Q head"), (axes[1], "vlm_mlp", "VLM MLP channel")]:
        for obj, col, name in [("coc", C1, "CoC NLL"), ("traj", C3, "trajectory")]:
            a = d[f"{obj}_{key}"]
            m = a.mean(1) / max(a.mean(), 1e-12)
            ax.plot(np.arange(a.shape[0]), m, color=col, lw=2, label=name)
        ax.set_xlabel("layer"); ax.set_ylabel("mean importance (normalised)")
        ax.set_title(label); ax.legend(frameon=False, fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.7)
    fig.suptitle("Both objectives trend with depth -- why the global correlation is confounded",
                 fontsize=10, color=MUTED)
    fig.tight_layout(); fig.savefig(plots / "importance_vs_depth.png", dpi=150); plt.close(fig)

    # 3) top-k overlap
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(UNITS))
    ax.bar(x - 0.2, [result[k]["top20_overlap_mean"] * 100 for k, _, _ in UNITS], 0.38,
           color=C1, label="top-20% overlap")
    ax.bar(x + 0.2, [result[k]["top50_overlap_mean"] * 100 for k, _, _ in UNITS], 0.38,
           color=C3, label="top-50% overlap")
    ax.axhline(20, color=C1, ls="--", lw=1.2)
    ax.axhline(50, color=C3, ls=":", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels([lbl for _, lbl, _ in UNITS], fontsize=9)
    ax.set_ylabel("% shared"); ax.set_title("Overlap of the two rankings (dashed = chance level)")
    ax.legend(frameon=False, fontsize=9); ax.grid(axis="y", color=GRID, lw=0.7)
    fig.tight_layout(); fig.savefig(plots / "topk_overlap.png", dpi=150); plt.close(fig)

    # 4) KV group heatmaps -- only 8 groups, so show them directly
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, obj, name in [(axes[0], "coc", "CoC NLL"), (axes[1], "traj", "trajectory")]:
        a = d[f"{obj}_kv_k"] + d[f"{obj}_kv_v"]
        a = a / a.max(1, keepdims=True).clip(1e-12)
        im = ax.imshow(a.T, aspect="auto", cmap="magma", origin="lower")
        ax.set_xlabel("layer"); ax.set_ylabel("KV group")
        ax.set_title(f"KV group importance ({name}), row-normalised")
        fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout(); fig.savefig(plots / "kv_group_heatmap.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
