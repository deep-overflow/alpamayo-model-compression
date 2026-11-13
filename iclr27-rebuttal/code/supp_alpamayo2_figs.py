"""Appendix C.4 figures (Alpamayo 1.5 top row, Alpamayo 2 Super bottom row), redrawn for the paper as vector PDFs.

  supp_alpamayo2_vlm.pdf     (a) corrected within-layer rank agreement I_FM vs I_CE, Q heads (All + per token type),
                             (b) I_FM share by token type, (c) I_CE share by token type
  supp_alpamayo2_expert.pdf  (a, b) Spearman rank similarity between denoising steps, Q heads / MLP channels,
                             (c) cumulative importance C(r)

Nothing is recomputed: every curve, matrix and number is read from the stored outputs of the 2026-11-08 runs
  outputs/iclr27-rebuttal/a2_fig3_token_type/metrics.json   (a2_fig3_token_type.py; gates G4-G7 PASS)
  outputs/iclr27-rebuttal/a2_fig5_expert/metrics.json       (a2_fig5_expert.py; gate G8 PASS)
so the values are those of the earlier overview PNGs (2026-11-08_a2_fig3_overview_q_head.png, 2026-11-08_a2_fig5_overview.png).

Layout follows the paper's main Fig. 3 / Fig. 5 as typeset (user reference, 2026-11-13): plot text in DejaVu Sans
(experiments/paper/fig_criterion.py rcParams, TrueType-embedded), panel titles as subcaptions UNDER each column in the paper's
serif (STIX = Times metrics; I_FM / I_CE as italic I with upright subscript), model names as serif row labels, one colour bar
per heat map with "Spearman's rho" on it, and Fig. 5c's palette (Q head #618BC8, MLP #ECAE3C, uniform grey, solid black line at
r = 0.9, square MLP marker). Kept from the earlier figures: data, axis ranges, the token-type colours and legend order, the grey
band and the dotted step line. Removed: the footer "Q heads; grey = late band, dotted = I_traj/I_CoC step" (now in the caption).
Canvas = ICLR \\linewidth (5.5 in), so the PDF is placed at 100%: plot text 7-7.5 pt, subcaptions 9 pt.

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

plt.rcParams["mathtext.fontset"] = "stix"  # only the subcaptions use math; STIX matches the paper's Times
ps = pfa.ps
MODELS = pfa.MODELS  # name, n layers, band (grey), tau (dotted), ticks -- the earlier figures' definitions
WIDTH = 5.5  # in, ICLR \linewidth
TICK, LAB, LEG, NOTE, CAP = 7, 7.5, 7.5, 6.5, 9  # pt; CAP = subcaptions and row labels (serif)
SERIF = "STIXGeneral"
COLORS = {"All": "black", "Vision": ps.BLUE, "Prompt text": ps.YELLOW, "CoC": pfa.PURPLE, "Ego history": ps.BRICK}
TYPE_ROW = {"Vision": pfa.VIS, "Prompt text": pfa.PT, "CoC": pfa.COC, "Ego history": pfa.HIST}
C_Q, C_MLP, C_UNIFORM = ps.BLUE, ps.YELLOW, ps.GREY  # main Fig. 5c as typeset
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


def captions(fig, bottom_axes, row_axes, titles, x_row=0.008, y_cap=0.012, x_centres=None):
    """serif subcaption under each column (centred on the bottom-row axes, or on x_centres in figure fraction) and serif
    model name left of each row"""
    for i, (ax, t) in enumerate(zip(bottom_axes, titles, strict=True)):
        p = ax.get_position()
        x = (p.x0 + p.x1) / 2 if x_centres is None else x_centres[i]
        fig.text(x, y_cap, t, ha="center", va="bottom", fontfamily=SERIF, fontsize=CAP)
    for ax, key in zip(row_axes, ("a15", "a2"), strict=True):
        p = ax.get_position()
        fig.text(x_row, (p.y0 + p.y1) / 2, MODELS[key]["name"], rotation=90, ha="left", va="center", fontfamily=SERIF,
                 fontsize=CAP)


def vlm_figure(m3, out):
    pl = m3["per_layer"]
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 4.0))
    fig.subplots_adjust(left=0.135, right=0.99, top=0.905, bottom=0.175, wspace=0.42, hspace=0.5)
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
        ax.set_ylabel("Corrected ρ")
        for col, obj, n_layers in ((1, "traj", last), (2, "coc", m["n"])):
            ax = axes[row, col]
            depth_marks(ax, m)
            s = np.asarray(share[obj])  # (5, L)
            for label, ty in TYPE_ROW.items():
                ax.plot(np.arange(n_layers), s[ty][:n_layers], color=COLORS[label], lw=0.9, label=label)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Share of layer importance")
        for ax in axes[row]:
            ax.set_xlabel("VLM layer index")
            style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.56, 1.0), ncol=5, frameon=False, fontsize=LEG,
               handlelength=1.4, handletextpad=0.4, columnspacing=1.1)
    captions(fig, axes[1], axes[:, 0], ("(a) Rank correlation", r"(b) $\mathit{I}_{\mathrm{FM}}$ by token type",
                                        r"(c) $\mathit{I}_{\mathrm{CE}}$ by token type"))
    save(fig, out, "supp_alpamayo2_vlm")


def expert_figure(m5, out):
    """Manual placement in inches: two heat maps with their own colour bars (as the paper's Fig. 5a, b) and the C(r) panel
    must share 5.5 in at the same 7-7.5 pt text as the VLM figure, which automatic layout cannot fit."""
    res = m5["results"]
    s, wc = 0.92, 0.98  # heat-map side, C(r) panel width
    top, mid, bot = 0.12, 0.47, 0.6  # above row 1; row-1 x labels; row-2 x labels + subcaptions
    W, H = WIDTH, top + s + mid + s + bot
    lab_w, cb_gap, cb_w, cb_lab_w, gap, c_gap, c_lab_w = 0.32, 0.04, 0.06, 0.36, 0.1, 0.22, 0.42  # in
    x_hm = [0.16 + lab_w]
    x_hm.append(x_hm[0] + s + cb_gap + cb_w + cb_lab_w + gap + lab_w)
    x_c = x_hm[1] + s + cb_gap + cb_w + cb_lab_w + c_gap + c_lab_w
    # subcaptions centred under each subfigure block (y label .. colour-bar label, or y label .. C(r) panel), as LaTeX does
    centres = [(x - lab_w + x + s + cb_gap + cb_w + cb_lab_w) / 2 / W for x in x_hm] + [(x_c - c_lab_w + x_c + wc) / 2 / W]
    y_rows = [H - top - s, H - top - s - mid - s]  # top row, bottom row (axes bottoms)

    def box(x, y, w, h):
        return [x / W, y / H, w / W, h / H]

    fig = plt.figure(figsize=(W, H))
    bottom, first = [], []
    for row, key in enumerate(("a15", "a2")):
        y = y_rows[row]
        for col, axis in enumerate(("q", "mlp")):
            ax = fig.add_axes(box(x_hm[col], y, s, s))
            r = res[f"{key}/{axis}"]
            mat = np.asarray(r["matrix"])  # (S, S)
            edges = np.arange(mat.shape[0] + 1) - 0.5
            # vector cells (imshow would embed a 10 x 10 bitmap that PDF viewers smooth); step 0 at the top, as imshow
            im = ax.pcolormesh(edges, edges, mat, vmin=0, vmax=1, cmap="viridis", shading="flat", edgecolors="face",
                               linewidth=0)
            ax.set_xlim(edges[0], edges[-1])
            ax.set_ylim(edges[-1], edges[0])
            ax.set_xticks([0, 3, 6, 9])
            ax.set_yticks([0, 3, 6, 9])
            for sp in ax.spines.values():
                sp.set_visible(True)
            ax.set_xlabel("Denoising step")
            ax.set_ylabel("Denoising step")
            ax.text(0.97, 0.03, f"mean {r['mean_pair_rho']:.3f}\nmin {r['min_pair_rho']:.3f}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=NOTE - 0.5, color="white", linespacing=1.0)
            style(ax, grid=False)
            cax = fig.add_axes(box(x_hm[col] + s + cb_gap, y, cb_w, s))
            cb = fig.colorbar(im, cax=cax)
            cb.set_label("Spearman's ρ", fontsize=LAB, labelpad=2)
            cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            cb.ax.tick_params(labelsize=TICK, length=2, pad=1.5)
            cb.outline.set_linewidth(0.5)
            if row == 1:
                bottom.append(ax)
            if col == 0:
                first.append(ax)
        ax = fig.add_axes(box(x_c, y, wc, s))
        ax.plot([0, 1], [0, 1], color=C_UNIFORM, lw=0.6, label="uniform")
        for axis, col_, lab, mk, off in (("q", C_Q, "Q head", "o", (-17, 12)), ("mlp", C_MLP, "MLP channel", "s", (-31, 6))):
            r = res[f"{key}/{axis}"]
            curve = np.asarray(r["curve"])
            grid = np.arange(len(curve)) / (len(curve) - 1)
            x = r["k_mark"] / (len(curve) - 1)
            ax.plot(grid, curve, color=col_, lw=1.2, label=lab)
            ax.plot(x, r["C_mark"], mk, color=col_, ms=2.8)
            ax.annotate(f"{r['C_mark'] * 100:.1f}%", (x, r["C_mark"]), textcoords="offset points", xytext=off, color=col_,
                        fontsize=NOTE, bbox={"fc": "white", "ec": "none", "pad": 0.3, "alpha": 0.85})
        ax.axvline(R_MARK, color="black", lw=0.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("Fraction of units removed")
        ax.set_ylabel("Removed importance\nfraction")  # two lines: one is taller than the 0.95 in panel
        ax.legend(fontsize=NOTE - 0.5, frameon=False, loc="upper left", handlelength=1.1, handletextpad=0.35,
                  borderaxespad=0.2, labelspacing=0.25)
        style(ax)
        if row == 1:
            bottom.append(ax)
    captions(fig, bottom, first, ("(a) Q head rank similarity", "(b) MLP channel rank similarity",
                                  "(c) Cumulative importance"), y_cap=0.01, x_centres=centres)
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
        "font_pt": {"tick": TICK, "label": LAB, "legend": LEG, "note": NOTE, "subcaption_serif": CAP}}
    (out / "config.json").write_text(json.dumps(caption, indent=2))
    print(f"wrote {out} and {assets}")


if __name__ == "__main__":
    main()
