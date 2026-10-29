"""Per-layer concentration of first-order Taylor importance, VLM vs expert.

The 2026-08-30 axis report (reports/evaluation/2026-08-30_axis-taylor-comparability.html,
section 3.1) pooled every layer and found the VLM MLP flat (Gini 0.369, bottom 50% holds
27.4% of the mass) against an expert MLP that is concentrated (Gini 0.972, 0.44%). This
script asks the same question one layer at a time, because a pooled statistic hides
structure that lives in a few layers.

Per (tower, objective, axis, layer):
  gini         Gini of the layer's unit scores
  b50          share of the layer's mass held by its bottom 50% of units
  top1         share held by its top 1% (at least one unit)
  n50 / n90    fewest units that hold 50% / 90% of the mass
  cut          share held by the bottom round(RATIO * n) units, i.e. what dual_u40_v2's
               uniform per-layer cut removes (VLM Q 13/32, MLP 4898/12288)
plus, for the VLM, how many of the top-32 units traj and CoC share per layer.

Layers whose scores are all zero (VLM layer 35 under traj, structurally) are reported as
null and left out of the medians.

Usage:
  .venv/bin/python cvpr27/code/analyze_layer_concentration.py \
      [--vlm importance_v2] [--expert importance_stepexp_sum] \
      [--exp-id cvpr27/layer_concentration]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG, INK, MUTED, GRID = "#FAF9F5", "#29261B", "#6B6555", "#E8E6DC"
C_Q, C_MLP = "#2a78d6", "#008300"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.size": 9, "axes.grid": True, "axes.axisbelow": True,
})
REPO = Path(__file__).resolve().parents[2]
RATIO = 0.3985632694  # dual_u40_v2 per-layer cut ratio (axis report section 5)
TOPK = 32
# (tower, objective, axis) -> npz key; expert scores come from the raw step-sum file,
# VLM from importance_v2, the same pair the axis report used
SERIES = {
    ("expert", "traj", "q"): "traj_exp_q", ("expert", "traj", "mlp"): "traj_exp_mlp",
    ("vlm", "traj", "q"): "traj_vlm_q", ("vlm", "traj", "mlp"): "traj_vlm_mlp",
    ("vlm", "coc", "q"): "coc_vlm_q", ("vlm", "coc", "mlp"): "coc_vlm_mlp",
}


def layer_stats(x):
    """x: (n,) nonnegative scores of one layer."""
    s = np.sort(x)  # ascending
    n, tot = len(s), s.sum()
    if tot == 0:
        return None
    desc = np.cumsum(s[::-1]) / tot
    return {
        "gini": float((2 * np.arange(1, n + 1) - n - 1) @ s / (n * tot)),
        "b50": float(s[: n // 2].sum() / tot),
        "top1": float(s[-max(1, n // 100):].sum() / tot),
        "n50": int(np.searchsorted(desc, 0.5) + 1),
        "n90": int(np.searchsorted(desc, 0.9) + 1),
        "cut": float(s[: round(RATIO * n)].sum() / tot),
    }


def median_of(rows, k):
    return float(np.median([r[k] for r in rows if r is not None]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="importance_v2")
    ap.add_argument("--expert", default="importance_stepexp_sum")
    ap.add_argument("--exp-id", default="cvpr27/layer_concentration")
    args = ap.parse_args()

    src = {"vlm": np.load(REPO / "outputs" / args.vlm / "importance.npz"),
           "expert": np.load(REPO / "outputs" / args.expert / "importance.npz")}
    out = REPO / "outputs" / args.exp_id
    out.mkdir(parents=True, exist_ok=True)

    per_layer, medians = {}, {}
    for (tower, obj, axis), key in SERIES.items():
        arr = np.abs(src[tower][key])  # (36, n_units)
        rows = [layer_stats(arr[layer]) for layer in range(arr.shape[0])]
        name = f"{tower}/{obj}/{axis}"
        per_layer[name] = rows
        medians[name] = {k: median_of(rows, k) for k in ("gini", "b50", "top1", "n50", "cut")}

    overlap = {}
    for axis in ("q", "mlp"):
        t, c = src["vlm"][f"traj_vlm_{axis}"], src["vlm"][f"coc_vlm_{axis}"]  # (36, n)
        k = min(TOPK, t.shape[1] // 2)
        overlap[axis] = {"k": k, "per_layer": [
            None if t[layer].sum() == 0 else
            len(set(np.argsort(t[layer])[-k:]) & set(np.argsort(c[layer])[-k:]))
            for layer in range(t.shape[0])]}

    config = {"vlm_importance": args.vlm, "expert_importance": args.expert,
              "cut_ratio": RATIO, "topk_overlap": TOPK, "series": {
                  f"{t}/{o}/{a}": k for (t, o, a), k in SERIES.items()}}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    (out / "metrics.json").write_text(json.dumps(
        {"median": medians, "per_layer": per_layer, "traj_coc_topk_overlap": overlap}, indent=2))

    # summary: medians, then the VLM MLP layer by layer
    lines = ["Per-layer concentration of raw Taylor importance",
             f"VLM: outputs/{args.vlm}, expert: outputs/{args.expert}",
             f"cut = mass in the bottom round({RATIO:.4f} * n) units (dual_u40_v2's per-layer cut)",
             "", "Median over layers (all-zero layers excluded):",
             f"{'series':<18}{'gini':>7}{'b50':>8}{'top1':>8}{'n50':>8}{'cut':>8}"]
    for name, m in medians.items():
        lines.append(f"{name:<18}{m['gini']:7.3f}{m['b50'] * 100:7.2f}%{m['top1'] * 100:7.2f}%"
                     f"{m['n50']:8.0f}{m['cut'] * 100:7.2f}%")
    lines += ["", f"VLM MLP by layer (traj | coc), top-{overlap['mlp']['k']} overlap traj/coc:",
              (f"{'L':>3}{'gini':>7}{'n50':>7}{'n90':>7}{'cut':>8} |{'gini':>6}{'n50':>7}"
               f"{'cut':>8} |{'ovl':>5}")]
    for layer in range(36):
        t, c = per_layer["vlm/traj/mlp"][layer], per_layer["vlm/coc/mlp"][layer]
        tt = (f"{t['gini']:7.2f}{t['n50']:7d}{t['n90']:7d}{t['cut'] * 100:7.2f}%"
              if t else f"{'(all zero)':>28}")
        ov = overlap["mlp"]["per_layer"][layer]
        lines.append(f"{layer:3d}{tt} |{c['gini']:6.2f}{c['n50']:7d}{c['cut'] * 100:7.2f}% |"
                     f"{'-' if ov is None else ov:>5}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # plot: Gini and dual-cut mass by layer; colour = axis, line style = tower
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    style = {"expert": "-", "vlm": "--"}
    colour = {"q": C_Q, "mlp": C_MLP}
    label = {"q": "Q head", "mlp": "MLP ch"}
    for ax, k, ylabel in ((axes[0], "gini", "Gini of unit scores"),
                          (axes[1], "cut", f"mass in bottom {RATIO:.0%} of units (%)")):
        for tower in ("expert", "vlm"):
            for axis in ("q", "mlp"):
                rows = per_layer[f"{tower}/traj/{axis}"]
                y = [np.nan if r is None else r[k] * (100 if k == "cut" else 1) for r in rows]
                ax.plot(range(36), y, style[tower], color=colour[axis], lw=2,
                        label=f"{tower} {label[axis]}")
        ax.set_xlabel("layer")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.5, 35.5)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("Concentration per layer (traj objective)", fontsize=9, loc="left")
    axes[1].set_title(f"What a uniform {RATIO:.0%} cut removes", fontsize=9, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=4, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "layer_concentration.png", dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
