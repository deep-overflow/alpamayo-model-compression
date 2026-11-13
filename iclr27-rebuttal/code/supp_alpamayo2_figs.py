"""Appendix C.4 figures (Alpamayo 1.5 top row, Alpamayo 2 Super bottom row), redrawn for the paper as vector PDFs.

  supp_alpamayo2_vlm.pdf     (a) corrected within-layer rank agreement I_FM vs I_CE, Q heads (All + per token type),
                             (b) I_FM share by token type, (c) I_CE share by token type
  supp_alpamayo2_expert.pdf  (a, b) Spearman rank similarity between denoising steps, Q heads / MLP channels,
                             (c) cumulative importance C(r)

Nothing is recomputed: every curve, matrix and number is read from the stored outputs of the 2026-11-08 runs
  outputs/iclr27-rebuttal/a2_fig3_token_type/metrics.json   (a2_fig3_token_type.py; gates G4-G7 PASS)
  outputs/iclr27-rebuttal/a2_fig5_expert/metrics.json       (a2_fig5_expert.py; gate G8 PASS)
so the values are those of the earlier overview PNGs (2026-11-08_a2_fig3_overview_q_head.png, 2026-11-08_a2_fig5_overview.png).
Kept from those figures: data, axis ranges, colours, legend order, the grey band and the dotted step line. Changed: the footer
"Q heads; grey = late band, dotted = I_traj/I_CoC step" is removed (it goes to the caption), the canvas is the ICLR \\linewidth
(5.5 in) so the PDF is placed at 100% and the text stays readable (7-8 pt), and both figures use the main-paper rcParams
(experiments/paper/fig_criterion.py: DejaVu Sans, TrueType-embedded, light y grid, semibold panel titles) and its palette.

Usage: .venv/bin/python iclr27-rebuttal/code/supp_alpamayo2_figs.py [--exp-id iclr27-rebuttal/supp_alpamayo2_figs]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, "/home/cvlab21/project/chan/alpamayo2-distill/experiments/importance")
import paper_figs_a2 as pfa  # imports paper_style -> fig_criterion rcParams

ps = pfa.ps
MODELS = pfa.MODELS  # name, n layers, band (grey), tau (dotted), ticks -- the earlier figures' definitions
WIDTH = 5.5  # in, ICLR \linewidth
TICK, LAB, TITLE, LEG, NOTE = 7, 7.5, 8, 7.5, 6.5  # pt
COLORS = {"All": "black", "Vision": ps.BLUE, "Prompt text": ps.YELLOW, "CoC": pfa.PURPLE, "Ego history": ps.BRICK}
TYPE_ROW = {"Vision": pfa.VIS, "Prompt text": pfa.PT, "CoC": pfa.COC, "Ego history": pfa.HIST}
C_Q, C_MLP, C_UNIFORM = "#0072B2", "#D55E00", ps.GREY  # main Fig. 5c palette
R_MARK = 0.9
SRC3 = REPO / "outputs" / "iclr27-rebuttal" / "a2_fig3_token_type" / "metrics.json"
SRC5 = REPO / "outputs" / "iclr27-rebuttal" / "a2_fig5_expert" / "metrics.json"


def depth_marks(ax, m):
    """grey late band + dotted step line, as paper_figs_a2.depth_marks (Q-head tau in every VLM panel)"""
    ax.axvspan(*m["band"], color=ps.GREY, alpha=0.13, lw=0)
    ax.axvline(m["tau"]["q"] - 0.5, color="black", lw=0.7, ls=(0, (1.2, 1.6)))
    ax.set_xlim(-0.5, m["n"] - 0.5)
    ax.set_xticks(m["ticks"])


def style(ax, grid=True):
    ax.tick_params(axis="both", which="both", labelsize=TICK, length=2.5, pad=2)
    ax.xaxis.label.set_fontsize(LAB)
    ax.yaxis.label.set_fontsize(LAB)
    if grid:
        ax.grid(axis="y")


def vlm_figure(m3, out):
    pl = m3["per_layer"]
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 3.75))
    for row, key in enumerate(("a15", "a2")):
        m = MODELS[key]
        agree, share = pl[f"{key}/q"]["agreement"], pl[f"{key}/q"]["share"]
        last = m["n"] - 1
        ax = axes[row, 0]
        depth_marks(ax, m)
        for label, color in COLORS.items():
            ax.plot(np.arange(last), agree[label]["corrected"], color=color, lw=1.15 if label == "All" else 0.8,
                    label=label)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylim(-0.25, 1.05)
        ax.set_yticks([-0.25, 0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylabel(f"{m['name']}\nCorrected ρ")
        for col, obj, n_layers in ((1, "traj", last), (2, "coc", m["n"])):
            ax = axes[row, col]
            depth_marks(ax, m)
            s = np.asarray(share[obj])  # (5, L)
            for label, ty in TYPE_ROW.items():
                ax.plot(np.arange(n_layers), s[ty][:n_layers], color=COLORS[label], lw=0.9, label=label)
            ax.set_ylim(0, 1.05)
        axes[row, 1].set_ylabel("Share of layer importance")
        for ax in axes[row]:
            ax.set_xlabel("VLM layer index")
            style(ax)
    for ax, t in zip(axes[0], ("(a) Rank correlation", r"(b) $I_{\mathrm{FM}}$ by token type",
                               r"(c) $I_{\mathrm{CE}}$ by token type"), strict=True):
        ax.set_title(t, fontsize=TITLE, pad=4)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, fontsize=LEG,
               handlelength=1.4, handletextpad=0.4, columnspacing=1.1)
    fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.8, rect=(0, 0, 1, 0.94))
    save(fig, out, "supp_alpamayo2_vlm")


def expert_figure(m5, out):
    res = m5["results"]
    fig = plt.figure(figsize=(WIDTH, 3.8))
    # heatmap, heatmap, shared colourbar, spacer (room for the colourbar label and (c)'s y label), curve
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 0.05, 0.36, 1.12], wspace=0.28, hspace=0.55,
                          left=0.1, right=0.975, top=0.885, bottom=0.105)
    for row, key in enumerate(("a15", "a2")):
        name = MODELS[key]["name"]
        for col, axis in enumerate(("q", "mlp")):
            ax = fig.add_subplot(gs[row, col])
            r = res[f"{key}/{axis}"]
            mat = np.asarray(r["matrix"])  # (S, S)
            edges = np.arange(mat.shape[0] + 1) - 0.5
            # vector cells (imshow would embed a 10 x 10 bitmap that PDF viewers smooth); step 0 at the top, as imshow
            im = ax.pcolormesh(edges, edges, mat, vmin=0, vmax=1, cmap="viridis", shading="flat", edgecolors="face",
                               linewidth=0)
            ax.set_xlim(edges[0], edges[-1])
            ax.set_ylim(edges[-1], edges[0])
            ax.set_aspect("equal")
            ax.set_xticks([0, 3, 6, 9])
            ax.set_yticks([0, 3, 6, 9])
            for sp in ax.spines.values():
                sp.set_visible(True)
            ax.set_xlabel("Denoising step")
            if col == 0:  # (b) shares (a)'s step axis; its label would sit on (a) at this width
                ax.set_ylabel(f"{name}\nDenoising step")
            ax.text(0.97, 0.03, f"mean {r['mean_pair_rho']:.3f}\nmin {r['min_pair_rho']:.3f}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=NOTE - 0.5, color="white", linespacing=1.0)
            style(ax, grid=False)
            if row == 0:  # the earlier titles, broken over two lines to fit a 1/3-\linewidth panel
                ax.set_title(("(a) Q head\nrank similarity", "(b) MLP channel\nrank similarity")[col], fontsize=TITLE,
                             pad=4)
        cax = fig.add_subplot(gs[row, 2])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Spearman's ρ", fontsize=LAB)
        cb.ax.tick_params(labelsize=TICK, length=2, pad=1.5)
        cb.outline.set_linewidth(0.5)
        ax = fig.add_subplot(gs[row, 4])
        ax.plot([0, 1], [0, 1], color=C_UNIFORM, lw=0.7, label="uniform")
        for axis, col_, lab in (("q", C_Q, "Q head"), ("mlp", C_MLP, "MLP channel")):
            r = res[f"{key}/{axis}"]
            curve = np.asarray(r["curve"])
            grid = np.arange(len(curve)) / (len(curve) - 1)
            x = r["k_mark"] / (len(curve) - 1)
            ax.plot(grid, curve, color=col_, lw=1.2, label=lab)
            ax.plot(x, r["C_mark"], "o", color=col_, ms=2.8)
            ax.annotate(f"{r['C_mark'] * 100:.1f}%", (x, r["C_mark"]), textcoords="offset points", xytext=(-21, 5),
                        color=col_, fontsize=NOTE, bbox={"fc": "white", "ec": "none", "pad": 0.4, "alpha": 0.85})
        ax.axvline(R_MARK, color="0.4", lw=0.7, ls=":")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
        ax.set_xlabel("Fraction of units removed")
        ax.set_ylabel("Removed importance fraction")
        ax.legend(fontsize=NOTE, frameon=False, loc="upper left", handlelength=1.3, handletextpad=0.4, borderaxespad=0.2,
                  labelspacing=0.25)
        style(ax)
        if row == 0:
            ax.set_title("(c) Cumulative\nimportance", fontsize=TITLE, pad=4)
    save(fig, out, "supp_alpamayo2_expert")


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"  {name}.pdf / .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="iclr27-rebuttal/supp_alpamayo2_figs")
    ap.add_argument("--assets", default=str(REPO / "iclr27-rebuttal" / "paper" / "fig" / "assets"),
                    help="where the two PDFs are copied for the paper (fig/assets/)")
    args = ap.parse_args()
    out = REPO / "outputs" / args.exp_id
    out.mkdir(parents=True, exist_ok=True)
    m3, m5 = json.loads(SRC3.read_text()), json.loads(SRC5.read_text())
    assert all(m3["gates_ok"].values()) and m5["gates_ok"], "source runs must have passed their gates"
    vlm_figure(m3, out)
    expert_figure(m5, out)
    assets = Path(args.assets)
    assets.mkdir(parents=True, exist_ok=True)
    for name in ("supp_alpamayo2_vlm", "supp_alpamayo2_expert"):
        shutil.copy2(out / f"{name}.pdf", assets / f"{name}.pdf")

    caption = {
        "grey_band": {k: {"layers_shaded": [int(np.ceil(m["band"][0])), int(np.floor(m["band"][1]))],
                          "definition": ("the paper's late band (Fig. 3)" if k == "a15" else
                                         "the A1.5 band moved by relative depth (22/36 x 64 = 39)")}
                      for k, m in MODELS.items()},
        "dotted_step": {k: {"tau_q_heads": m["tau"]["q"], "tau_mlp": m["tau"]["mlp"], "drawn_at": m["tau"]["q"] - 0.5,
                            "definition": "changepoint of a two-level least-squares step fit to log(sum_u I_traj / "
                                          "sum_u I_CoC) over layers 0..L-2; the line sits between layers tau-1 and tau"}
                        for k, m in MODELS.items()},
        "expert_C09": {k: round(m5["results"][k]["C_mark"], 5) for k in m5["results"]},
        "sources": {"vlm": str(SRC3), "expert": str(SRC5)}, "width_in": WIDTH,
        "font_pt": {"tick": TICK, "label": LAB, "title": TITLE, "legend": LEG, "note": NOTE}}
    (out / "config.json").write_text(json.dumps(caption, indent=2))
    print(json.dumps(caption, indent=2))
    print(f"wrote {out} and {assets}")


if __name__ == "__main__":
    main()
