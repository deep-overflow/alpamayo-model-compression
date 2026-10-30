"""How the VLM's MLP importance changes with depth: concentration, objective agreement, calibration stability.

Three per-layer readouts of the raw first-order Taylor scores of the VLM MLP channels (12,288 per layer):

  (a) concentration  Gini of the layer's channel scores, per objective (I_traj = trajectory FM loss,
                     I_CoC = CoC NLL), from importance_v2 (calib_100)
  (b) agreement      Spearman rho between I_traj and I_CoC over the layer's channels; the measure of
                     Finding 1 (paper-analysis/sec2_rank_analysis.tex), which put the split at layer 22
  (c) stability      share of the layer's top-64 channels that an independent calibration draw also
                     ranks top-64, against importance_v2; median over five draws (band = min-max)

Layer 35's I_traj is structurally zero (no path to the expert's cache), so it is left out of every
I_traj readout and of (b).

Writes outputs/<exp-id>/{config.json, metrics.json, summary.txt, depth_structure.png, depth_structure.pdf}.

Usage:
  .venv/bin/python cvpr27/code/plot_depth_structure.py [--exp-id cvpr27/depth_structure]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

BG, INK, MUTED, GRID = "#FAF9F5", "#29261B", "#6B6555", "#E8E6DC"
C_TRAJ, C_COC = "#2a78d6", "#008300"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.size": 9, "axes.grid": True, "axes.axisbelow": True,
})
REPO = Path(__file__).resolve().parents[2]
REF = "importance_v2"
DRAWS = ("importance_rd100_a", "importance_se100_b", "importance_tr500_c",
         "importance_nt500_c500", "importance_st4000_c2000")  # independent calibration draws
TOPK = 64
N_LAYERS = 36
BANDS = {"0": [0], "1-4": [1, 2, 3, 4], "5": [5], "6-21": list(range(6, 22)), "22-34": list(range(22, 35))}


def gini(x):
    s = np.sort(x)
    n = len(s)
    return float((2 * np.arange(1, n + 1) - n - 1) @ s / (n * s.sum())) if s.sum() > 0 else float("nan")


def top_overlap(a, b, k=TOPK):
    """a, b: (n,) scores -> share of a's top-k that is also in b's top-k"""
    if a.sum() == 0 or b.sum() == 0:
        return float("nan")
    return len(set(np.argsort(a)[-k:]) & set(np.argsort(b)[-k:])) / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="cvpr27/depth_structure")
    args = ap.parse_args()
    ref = np.load(REPO / "outputs" / REF / "importance.npz")
    draws = {d: np.load(REPO / "outputs" / d / "importance.npz") for d in DRAWS}
    objs = {"traj": "traj_vlm_mlp", "coc": "coc_vlm_mlp"}

    gin = {o: [gini(ref[k][li]) for li in range(N_LAYERS)] for o, k in objs.items()}
    rho = [float("nan") if ref["traj_vlm_mlp"][li].sum() == 0 else
           float(spearmanr(ref["traj_vlm_mlp"][li], ref["coc_vlm_mlp"][li])[0]) for li in range(N_LAYERS)]
    stab = {o: np.array([[top_overlap(ref[k][li], draws[d][k][li]) for li in range(N_LAYERS)]
                         for d in DRAWS]) for o, k in objs.items()}  # obj -> (n_draws, 36)

    out = REPO / "outputs" / args.exp_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({
        "reference": REF, "draws": DRAWS, "topk": TOPK, "axis": "VLM MLP channels (12,288 per layer)",
        "score": "raw first-order Taylor, I_traj (FM loss) and I_CoC (CoC NLL)",
        "finding1": "paper-analysis/sec2_rank_analysis.tex (split at layer 22)"}, indent=2))
    per_layer = {"gini": gin, "spearman_traj_coc": rho,
                 "stability_median": {o: np.nanmedian(v, 0).tolist() for o, v in stab.items()},
                 "stability_min": {o: np.nanmin(v, 0).tolist() for o, v in stab.items()},
                 "stability_max": {o: np.nanmax(v, 0).tolist() for o, v in stab.items()}}
    bands = {}
    for b, ls in BANDS.items():
        bands[b] = {"gini_traj": float(np.nanmedian([gin["traj"][li] for li in ls])),
                    "gini_coc": float(np.nanmedian([gin["coc"][li] for li in ls])),
                    "spearman": float(np.nanmedian([rho[li] for li in ls])),
                    "stability_traj": float(np.nanmedian(stab["traj"][:, ls])),
                    "stability_coc": float(np.nanmedian(stab["coc"][:, ls]))}
    (out / "metrics.json").write_text(json.dumps({"per_layer": per_layer, "bands": bands}, indent=2))

    lines = ["VLM MLP importance by depth (median over the band's layers)", "",
             (f"{'layers':<8}{'Gini traj':>11}{'Gini CoC':>10}{'rho traj-CoC':>14}"
              f"{'top-64 kept traj':>18}{'top-64 kept CoC':>17}")]
    for b, v in bands.items():
        lines.append(f"{b:<8}{v['gini_traj']:>11.2f}{v['gini_coc']:>10.2f}{v['spearman']:>14.2f}"
                     f"{v['stability_traj'] * 100:>17.0f}%{v['stability_coc'] * 100:>16.0f}%")
    lines += ["", f"stability = share of calib_100's top-{TOPK} channels that an independent calibration draw",
              f"also ranks top-{TOPK}; median over {len(DRAWS)} draws ({', '.join(DRAWS)}).",
              "Layer 35 excluded from I_traj readouts (structurally zero)."]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    x = np.arange(N_LAYERS)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), sharex=True)
    for ax in axes:
        ax.axvspan(0.5, 4.5, color=GRID, alpha=0.9, lw=0, zorder=0)
        ax.set_xlim(-0.5, 35.5)
        ax.set_xlabel("layer")
    ax = axes[0]
    for o, c, lab in (("traj", C_TRAJ, "I_traj (trajectory)"), ("coc", C_COC, "I_CoC (reasoning)")):
        ax.plot(x, gin[o], color=c, lw=2, label=lab)
    ax.set_ylim(0, 1)
    ax.set_title("(a) Concentration: Gini of channel scores", fontsize=9, loc="left")
    ax = axes[1]
    ax.plot(x, rho, color=INK, lw=2)
    ax.axvline(21.5, color=MUTED, lw=1, ls="--")
    ax.annotate("Finding 1 split", xy=(21.7, 0.97), fontsize=7, color=MUTED, va="top")
    ax.set_ylim(0, 1)
    ax.set_title("(b) Agreement: Spearman ρ(I_traj, I_CoC)", fontsize=9, loc="left")
    ax = axes[2]
    for o, c in (("traj", C_TRAJ), ("coc", C_COC)):
        v = stab[o] * 100
        ax.fill_between(x, np.nanmin(v, 0), np.nanmax(v, 0), color=c, alpha=0.15, lw=0)
        ax.plot(x, np.nanmedian(v, 0), color=c, lw=2)
    ax.set_ylim(0, 100)
    ax.set_title(f"(c) Stability: top-{TOPK} kept across calib draws (%)", fontsize=9, loc="left")
    axes[0].annotate("shaded: layers 1–4", xy=(0.8, 0.04), ha="left", fontsize=7, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=2, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "depth_structure.png", dpi=200)
    fig.savefig(out / "depth_structure.pdf")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
