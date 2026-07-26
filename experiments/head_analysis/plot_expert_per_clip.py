"""Visualize per-clip expert cache attention with prompt-text and CoC separated.

Usage: python plot_expert_per_clip.py --exp-id expert_per_clip_v1
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/workspace/alpamayo-model-compression")
BG, INK, MUTED, GRID = "#FAF9F5", "#29261B", "#6B6555", "#E8E6DC"
# validated categorical palette (light mode)
C = {
    "vision": "#2a78d6", "prompt_text": "#008300", "coc": "#e87ba4",
    "hist": "#eda100", "own": "#1baf7a", "sink": "#eb6834",
}
LABEL = {
    "vision": "vision", "prompt_text": "prompt text", "coc": "generated CoC",
    "hist": "traj history", "own": "own 64 tokens", "sink": "sink",
}

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})
REGIONS = ["vision", "prompt_text", "coc", "hist", "sink", "own"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", type=str, default="expert_per_clip_v1")
    args = ap.parse_args()
    out_dir = REPO / "outputs" / args.exp_id
    plots = out_dir / "plots"
    plots.mkdir(exist_ok=True)
    d = np.load(out_dir / "expert_per_clip.npz")
    recs = json.load(open(out_dir / "metrics.json"))
    N, L, H = d["mass_vision"].shape
    layers = np.arange(L)

    # ---- 1) per-layer mass with per-clip spread ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for r in REGIONS:
        m = d[f"mass_{r}"].mean(2)  # (N, L) mean over heads
        med = np.median(m, 0)
        axes[0].plot(layers, med, color=C[r], lw=2, label=LABEL[r])
        axes[0].fill_between(layers, np.percentile(m, 10, 0), np.percentile(m, 90, 0),
                             color=C[r], alpha=0.15, lw=0)
    axes[0].set_xlabel("expert layer"); axes[0].set_ylabel("attention mass")
    axes[0].set_title(f"Expert attention mass by region (median, 10–90% band, n={N})")
    axes[0].legend(frameon=False, fontsize=8.5, ncol=2)
    axes[0].grid(axis="y", color=GRID, lw=0.7)

    # zoom: text regions only
    for r in ["prompt_text", "coc"]:
        m = d[f"mass_{r}"].mean(2)
        axes[1].plot(layers, np.median(m, 0), color=C[r], lw=2, label=LABEL[r])
        axes[1].fill_between(layers, np.percentile(m, 10, 0), np.percentile(m, 90, 0),
                             color=C[r], alpha=0.15, lw=0)
    axes[1].set_xlabel("expert layer"); axes[1].set_ylabel("attention mass")
    axes[1].set_title("prompt text vs generated CoC (the split that was merged before)")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].grid(axis="y", color=GRID, lw=0.7)
    fig.tight_layout(); fig.savefig(plots / "region_split.png", dpi=150); plt.close(fig)

    # ---- 2) per-token normalized mass (size-corrected) ----
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for r in REGIONS:
        m = d[f"pertoken_{r}"].mean(2)
        ax.plot(layers, np.median(m, 0), color=C[r], lw=2, label=LABEL[r])
    ax.set_yscale("log")
    ax.set_xlabel("expert layer"); ax.set_ylabel("attention mass per key token (log)")
    ax.set_title("Size-corrected attention: mass divided by region token count")
    ax.legend(frameon=False, fontsize=8.5, ncol=2)
    ax.grid(axis="y", color=GRID, lw=0.7)
    fig.tight_layout(); fig.savefig(plots / "per_token_mass.png", dpi=150); plt.close(fig)

    # ---- 3) per-clip layer-0 breakdown, sorted by CoC mass ----
    l0 = {r: d[f"mass_{r}"][:, 0, :].mean(1) for r in REGIONS}
    order = np.argsort(l0["coc"])[::-1]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    bottom = np.zeros(N)
    for r in ["vision", "prompt_text", "coc", "hist", "sink", "own"]:
        v = l0[r][order]
        ax.bar(np.arange(N), v, bottom=bottom, color=C[r], width=0.86,
               label=LABEL[r], edgecolor=BG, linewidth=0.5)
        bottom += v
    ax.set_xlabel(f"clip (sorted by layer-0 CoC mass, n={N})")
    ax.set_ylabel("attention mass")
    ax.set_title("Expert layer 0: per-clip region breakdown")
    ax.legend(frameon=False, fontsize=8.5, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.set_xlim(-0.7, N - 0.3)
    fig.tight_layout(); fig.savefig(plots / "layer0_per_clip.png", dpi=150); plt.close(fig)

    # ---- 4) CoC mass vs CoC length ----
    coc_len = np.array([r["coc_len"] for r in recs])
    coc_mass_mean = d["mass_coc"].mean((1, 2))
    coc_mass_l0 = d["mass_coc"][:, 0, :].mean(1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].scatter(coc_len, coc_mass_mean, s=42, color=C["coc"],
                    edgecolor=BG, linewidth=1.2, zorder=3)
    axes[0].set_xlabel("generated CoC length (tokens)")
    axes[0].set_ylabel("mean CoC attention mass (all layers)")
    axes[0].set_title("Longer CoC → more CoC attention?")
    axes[0].grid(color=GRID, lw=0.7, zorder=0)
    axes[1].scatter(coc_len, coc_mass_l0, s=42, color=C["prompt_text"],
                    edgecolor=BG, linewidth=1.2, zorder=3)
    axes[1].set_xlabel("generated CoC length (tokens)")
    axes[1].set_ylabel("layer-0 CoC attention mass")
    axes[1].set_title("Layer 0 only")
    axes[1].grid(color=GRID, lw=0.7, zorder=0)
    for ax, y in zip(axes, [coc_mass_mean, coc_mass_l0]):
        if len(coc_len) > 2 and coc_len.std() > 0:
            r = np.corrcoef(coc_len, y)[0, 1]
            ax.annotate(f"pearson r = {r:+.2f}", xy=(0.04, 0.92), xycoords="axes fraction",
                        fontsize=9, color=MUTED)
    fig.tight_layout(); fig.savefig(plots / "coc_len_vs_mass.png", dpi=150); plt.close(fig)

    # ---- console summary ----
    print(f"n_clips={N}, layers={L}, heads={H}")
    print("\n=== mean attention mass over all layers/heads ===")
    for r in REGIONS:
        v = d[f"mass_{r}"].mean()
        print(f"  {LABEL[r]:16s} {v:.4f}")
    print("\n=== layer 0 (the layer that looked text-heavy) ===")
    for r in REGIONS:
        v = d[f"mass_{r}"][:, 0, :].mean()
        sd = d[f"mass_{r}"][:, 0, :].mean(1).std()
        print(f"  {LABEL[r]:16s} {v:.4f}  (per-clip sd {sd:.4f})")
    print("\n=== earlier merged 'text/CoC' value reproduced ===")
    merged0 = (d["mass_prompt_text"][:, 0, :] + d["mass_coc"][:, 0, :]).mean()
    print(f"  layer0 prompt_text + coc = {merged0:.4f}")
    print(f"    of which prompt text: {d['mass_prompt_text'][:, 0, :].mean() / merged0 * 100:.1f}%")
    print(f"    of which CoC        : {d['mass_coc'][:, 0, :].mean() / merged0 * 100:.1f}%")
    print("\n=== clips with the highest layer-0 CoC mass ===")
    for i in order[:5]:
        r = recs[i]
        print(f"  {r['clip_id']}  coc_len={r['coc_len']:2d}  cocmass={l0['coc'][i]:.4f}  "
              f"ADE={r['minADE']:.2f}  CoC={r['coc_text'][:70]!r}")
    print("\n=== clips with the lowest layer-0 CoC mass ===")
    for i in order[-3:]:
        r = recs[i]
        print(f"  {r['clip_id']}  coc_len={r['coc_len']:2d}  cocmass={l0['coc'][i]:.4f}  "
              f"ADE={r['minADE']:.2f}  CoC={r['coc_text'][:70]!r}")
    print("\nplots ->", plots)


if __name__ == "__main__":
    main()
