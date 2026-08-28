"""Report figures for the cache-shift diagnostic (cells / V map / attention + Stage C).

Reads the shard metrics.json files and cachediff_v1/metrics_analysis.json written by
run_cachediff.py / analyze_cachediff.py; writes into outputs/cachediff_v1/plots/.

Usage:
  .venv/bin/python experiments/head_analysis/plot_cachediff.py
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "cachediff_v1" / "plots"
BG, INK, MUTED, GRID = "#FAF9F5", "#29261B", "#6B6555", "#E8E6DC"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.size": 9, "axes.grid": True, "axes.axisbelow": True,
})
CELLS = ["A00_denseC_denseE", "A01_denseC_prunE", "A10_prunC_denseE", "A11_prunC_prunE"]
cells = {c: [] for c in CELLS}
attn = {}
for s in ("cachediff_v1_s0", "cachediff_v1_s100"):
    m = json.loads((REPO / "outputs" / s / "metrics.json").read_text())
    for c in CELLS:
        cells[c] += m["cells"][c]["ade"]
    for tag, d in m["attn"].items():
        for k, v in d.items():
            attn.setdefault(tag, {}).setdefault(k, []).append(np.array(v))
cells = {c: np.array(v) for c, v in cells.items()}
attn = {t: {k: np.mean(v, 0) for k, v in d.items()} for t, d in attn.items()}
ana = json.loads((REPO / "outputs" / "cachediff_v1" / "metrics_analysis.json").read_text())
div = {k: np.array(v) for k, v in ana["divergence"].items()}

def med_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    b = [np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(n)]
    return np.median(d), *np.percentile(b, [2.5, 97.5])

# ---- fig 1: the 2x2 and the interaction, per-clip
A00, A01, A10, A11 = (cells[c] for c in CELLS)
dD, dP = A01 - A00, A11 - A10
I = dP - dD
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
x = np.arange(2)
means = [[A00.mean(), A01.mean()], [A10.mean(), A11.mean()]]
ax[0].bar(x - 0.18, means[0], 0.36, color=C1, label="dense cache")
ax[0].bar(x + 0.18, means[1], 0.36, color=C3, label="pruned cache (dual u40)")
ax[0].set_xticks(x); ax[0].set_xticklabels(["dense expert", "pruned expert\n(znorm r25)"])
ax[0].set_ylabel("minADE@6 mean (m)"); ax[0].set_ylim(0, 1.22); ax[0].legend(fontsize=8, loc="lower left")
ax[0].set_title("2×2 cells (200 clips)")
for i, (row, col) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
    ax[0].text(col + (-0.18 if row == 0 else 0.18), means[row][col] + 0.01,
               f"{means[row][col]:.3f}", ha="center", fontsize=8)
lo = min(dD.min(), dP.min()); hi = max(dD.max(), dP.max())
bins = np.linspace(-0.6, 0.6, 49)
ax[1].hist(np.clip(dD, -0.6, 0.6), bins, alpha=0.55, color=C1, label="on dense cache")
ax[1].hist(np.clip(dP, -0.6, 0.6), bins, alpha=0.55, color=C3, label="on pruned cache")
for d, c in ((dD, C1), (dP, C3)):
    ax[1].axvline(np.median(d), color=c, lw=1.5)
ax[1].set_xlabel("per-clip ΔminADE@6 from the expert cut (m, clipped ±0.6)")
ax[1].set_ylabel("clips"); ax[1].legend(fontsize=8)
ax[1].set_title("expert-cut cost, per clip")
m, l, h = med_ci(I)
ax[2].hist(np.clip(I, -0.6, 0.6), bins, color=C2, alpha=0.7)
ax[2].axvline(0, color=MUTED, lw=1)
ax[2].axvline(m, color=C2, lw=2, label=f"median {m:+.4f}\n95% CI [{l:+.4f}, {h:+.4f}]")
ax[2].set_xlabel("per-clip interaction I = (A11−A10) − (A01−A00)  (m)")
ax[2].set_ylabel("clips"); ax[2].legend(fontsize=8, loc="upper left")
ax[2].set_title("A1: is the cut dearer on the pruned cache?")
fig.tight_layout(); fig.savefig(OUT / "cachediff_cells.png", dpi=150)

# ---- fig 2: V heatmap + span table as bars
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
im = ax[0].imshow(div["all_rel_v"], aspect="auto", cmap="magma")
ax[0].set_xlabel("KV group"); ax[0].set_ylabel("VLM layer"); ax[0].set_title("V relative L2  ‖ΔV‖/‖V‖")
fig.colorbar(im, ax=ax[0])
spans = ["vision", "text", "hist", "coc", "sink"]
kk = [1 - div[f"{s}_cos_k"].mean() for s in spans]
vv = [div[f"{s}_rel_v"].mean() for s in spans]
xs = np.arange(len(spans))
ax[1].bar(xs - 0.18, kk, 0.36, color=C1, label="K: 1 − cos")
ax[1].set_xticks(xs); ax[1].set_xticklabels(spans); ax[1].set_ylabel("K divergence")
ax1b = ax[1].twinx(); ax1b.bar(xs + 0.18, vv, 0.36, color=C2, label="V: rel L2")
ax1b.set_ylabel("V divergence"); ax1b.grid(False)
ax[1].set_title("by token span (all layers)")
h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = ax1b.get_legend_handles_labels()
ax[1].legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
depth = np.arange(36)
for s, c in (("vision", C1), ("text", C2), ("hist", C3), ("coc", MUTED)):
    ax[2].plot(depth, div[f"{s}_rel_v"].mean(1), color=c, label=s)
ax[2].set_xlabel("VLM layer"); ax[2].set_ylabel("‖ΔV‖/‖V‖"); ax[2].legend(fontsize=8)
ax[2].set_title("V shift by span and depth (sink omitted: off-scale)")
fig.tight_layout(); fig.savefig(OUT / "cachediff_v.png", dpi=150)

# ---- fig 3: expert attention under each cache + Stage C vs lambda
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
keys = ["mass_vision", "mass_text", "mass_hist", "mass_sink", "mass_own"]
dv = [attn["dense"][k].mean() for k in keys]; pv = [attn["pruned"][k].mean() for k in keys]
xs = np.arange(len(keys))
ax[0].bar(xs - 0.18, dv, 0.36, color=C1, label="dense cache")
ax[0].bar(xs + 0.18, pv, 0.36, color=C3, label="pruned cache")
ax[0].set_xticks(xs); ax[0].set_xticklabels([k[5:] for k in keys])
ax[0].set_ylabel("expert attention mass (mean over layers×heads)")
ax[0].set_yscale("log"); ax[0].legend(fontsize=8)
ax[0].set_title("where the pruned expert looks — nearly unchanged")
for i, (a, b) in enumerate(zip(dv, pv)):
    ax[0].text(i, max(a, b) * 1.25, f"{(b - a) * 100:+.2f}pp", ha="center", fontsize=8)
corr = ana["correctability"]
lams = [0.01, 0.1, 1.0]
for tag, c in (("k", C1), ("v", C2)):
    ax[1].plot(lams, [corr[f"{tag}_lam{l:g}"]["mean_residual_ratio"] for l in lams],
               "o-", color=c, label=f"{tag.upper()}: residual / raw shift")
ax[1].axhline(1.0, color=MUTED, ls="--", lw=1, label="identity map (no correction)")
ax[1].axhline(0.5, color=C4, ls=":", lw=1, label="C1 gate (<0.5 = correctable)")
ax[1].set_xscale("log"); ax[1].set_xlabel("ridge damping λ (× mean diag)")
ax[1].set_ylabel("cross-fold residual ratio"); ax[1].set_ylim(0, 3.3); ax[1].legend(fontsize=8)
ax[1].set_title("C: best linear cache map barely helps")
fig.tight_layout(); fig.savefig(OUT / "cachediff_attn_corr.png", dpi=150)
print("wrote", [p.name for p in OUT.glob("*.png")])
print(f"I median {m:+.4f} [{l:+.4f},{h:+.4f}]  | share of clips with I>0: {(I>0).mean():.2f}")
print(f"power: sd(I)={I.std(ddof=1):.3f}; n for 80% power at |I|={abs(np.mean(I)):.4f}: {(2.8*I.std(ddof=1)/abs(np.mean(I)))**2:,.0f}")
