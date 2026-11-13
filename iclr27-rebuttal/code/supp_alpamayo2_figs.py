"""Appendix C.4 figures (Alpamayo 1.5 top row, Alpamayo 2 Super bottom row) built with the SAME code paths as the paper's main panels.

  Fig. 3 panels  paper-analysis/scripts/paper_same_position.py (corrected mode) and paper_token_share.py (split mode),
                 exported like paper_style.export (3.5 x 2.8 in, 12 pt axis labels, 10 pt text, solid lines, y grid)
  Fig. 5 panels  experiments/paper/fig_criterion.py figure2 drawn as written, then its own --paper-style path
                 (fig_criterion.paper_style + fig_criterion.save: Spearman's rho colour bar, ticks 0/3/6/9, palette
                 #618BC8 / #ECAE3C, all lines solid, 69% / 1.8% labels)
Each of the 12 panels (2 models x 6) is written as its own 3.5 x 2.8 in PDF (fig/assets/supp_alpamayo2_panels/), and the two supp
PDFs are those panels placed 2 x 3 with serif subcaptions under each column and serif model names left of each row
(supp_alpamayo2_compose.py, pypdf via uvx: vector panels placed unchanged). At \\linewidth the panels come out at ~0.5x, the same
scale as the main figures, and the 20 pt subcaptions at ~10 pt.

Values. Alpamayo 1.5 rows reproduce the main panels exactly: (a) follows paper_same_position.py's corrected path (stored P2 series
of gradanat_v1, Ego history recomputed with the same rng order), (b, c) are gradanat_v1's additive token shares (equal to the
stored 2026-11-08 values to 0), the expert panels use stepimp_fm_perstep_v2 through the stored 2026-11-08 matrices / curves
(gate G8: equal to the paper text). Alpamayo 2 Super rows come from the stored 2026-11-08 outputs
(outputs/iclr27-rebuttal/a2_fig3_token_type, a2_fig5_expert; same estimators). Supp-only additions kept from the earlier figure:
each model's grey late band and the dotted I_traj/I_CoC step line (its meaning goes to the caption).

Usage: .venv/bin/python iclr27-rebuttal/code/supp_alpamayo2_figs.py [--exp-id iclr27-rebuttal/supp_alpamayo2_figs]
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "paper-analysis" / "scripts"))
sys.path.insert(0, "/home/cvlab21/project/chan/alpamayo2-distill/experiments/importance")
import paper_figs_a2 as pfa  # MODELS: band, tau, ticks per model
import paper_style as ps  # applies fig_criterion's rcParams

fc = ps.fc  # experiments/paper/fig_criterion.py
MODELS = pfa.MODELS
VIS, HIST, PT, COC = 0, 1, 2, 3
PURPLE = "#8E6BB3"
COLORS = {"All": "black", "Vision": ps.BLUE, "Prompt text": ps.YELLOW, "CoC": PURPLE, "Ego history": ps.BRICK}
SRC3 = REPO / "outputs" / "iclr27-rebuttal" / "a2_fig3_token_type" / "metrics.json"
SRC5 = REPO / "outputs" / "iclr27-rebuttal" / "a2_fig5_expert" / "metrics.json"
GRAD15 = REPO / "outputs" / "gradanat_v1"
VLM_PANELS = ("rank_correlation", "share_fm", "share_ce")
EXP_PANELS = ("steps_q_head", "steps_mlp", "mass_curve")


# ---------------------------------------------------------------- Fig. 3 (paper_same_position.py / paper_token_share.py)
def rho_layers(a, b):
    return pfa.rho_layers(a, b)


def split_half(t, c, n_split, rng):
    """paper_same_position.split_half"""
    n = len(t)
    st, sc_, cr = [], [], []
    for _ in range(n_split):
        p = rng.permutation(n)
        a, b = p[:n // 2], p[n // 2:]
        ta, tb, ca, cb = t[a].mean(0), t[b].mean(0), c[a].mean(0), c[b].mean(0)
        st.append(rho_layers(ta, tb))
        sc_.append(rho_layers(ca, cb))
        cr.append(0.5 * (rho_layers(ta, cb) + rho_layers(tb, ca)))
    return np.nanmean(st, 0), np.nanmean(sc_, 0), np.nanmean(cr, 0)


def corrected(st, sc_, cr):
    """paper_same_position.corrected"""
    ok = (st > 0.2) & (sc_ > 0.2)
    return np.where(ok, cr / np.sqrt(np.where(ok, st * sc_, 1.0)), np.nan)


def a15_corrected_series():
    """paper_same_position.py, axis q, corrected mode, verbatim order (so the A1.5 row equals the main Fig. 3a)"""
    last, n_split = 35, 50
    m = json.loads((GRAD15 / "metrics_analysis.json").read_text())["P2"]["q"]
    stored = {"All": "pooled (the shipped score)", "Vision": "both at VISION positions",
              "Prompt text": "both at prompt-text positions only", "CoC": "both at CoC positions only", "Ego history": None}
    series = {}
    with np.load(GRAD15 / "anatomy_perclip_q.npz") as pq:
        fm, ce = pq["fm_full"], pq["ce"]  # (N, 5, 36, 32)
        rng = np.random.default_rng(0)
        split_half(np.abs(fm[:, VIS])[:, :last], np.abs(ce[:, VIS])[:, :last], n_split, rng)  # same rng order
        for label, key in stored.items():
            if key in m:
                series[label] = corrected(*(np.array(m[key]["per_layer"][k]) for k in ("self_traj", "self_coc", "cross")))
            else:
                ty = {"Prompt text": PT, "Ego history": HIST}[label]
                series[label] = corrected(*split_half(np.abs(fm[:, ty])[:, :last], np.abs(ce[:, ty])[:, :last], n_split, rng))
    return series


def depth_marks(ax, m):
    """grey band as the main panels' axvspan (per model) + the supp-only dotted step line (kept dotted at export)"""
    ax.axvspan(*m["band"], color=ps.GREY, alpha=0.13, lw=0)
    ax.axvline(m["tau"]["q"] - 0.5, color="black", lw=0.9, ls=(0, (1.5, 2)), gid="step")


def export_panel(fig, out_dir, name):
    """paper_style.export, except that the step line keeps its dots"""
    for ax in fig.axes:
        for t in (ax.xaxis.label, ax.yaxis.label):
            t.set_fontsize(ps.LABEL)
        ax.tick_params(axis="both", which="both", labelsize=ps.TEXT)
        for t in ax.texts:
            t.set_fontsize(ps.TEXT)
        for line in ax.lines:
            if line.get_gid() != "step":
                line.set_linestyle("-")
        ax.grid(axis="y")
    fig.tight_layout(pad=0.8)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    margin = 3 * fig.dpi / 72
    for ax in fig.axes:  # keep the long rotated y-label inside the fixed canvas (paper_style.export)
        label = ax.yaxis.label
        box = label.get_window_extent(renderer)
        shift = min(0, fig.bbox.height - margin - box.y1)
        shift += max(0, margin - box.y0 - shift)
        if shift:
            label.set_y(label.get_position()[1] + shift / ax.bbox.height)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)


def rank_correlation_panel(series, m, out_dir, name):
    last = m["n"] - 1
    L = np.arange(last)
    fig, ax = plt.subplots(figsize=ps.SINGLE)
    depth_marks(ax, m)
    for label, y in series.items():
        ax.plot(L, np.asarray(y, dtype=float)[:last], color=COLORS[label], lw=1.8 if label == "All" else 1.3, label=label)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("VLM layer index")
    ax.set_ylabel(r"Corrected $\rho$")
    ax.set_xlim(-0.5, m["n"] - 0.5)
    ax.set_xticks(m["ticks"])
    ax.set_ylim(-0.25, 1.05)
    ax.set_yticks([-0.25, 0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=ps.TEXT, handlelength=1.0,
              handletextpad=0.4, columnspacing=0.7, borderaxespad=0.2)
    export_panel(fig, out_dir, name)


def share_panel(s, m, obj, out_dir, name):
    """paper_token_share.py, split mode"""
    s = np.asarray(s)  # (5, L): vision, hist, prompt, coc, sink
    last = m["n"] - 1 if obj == "traj" else m["n"]  # I_traj is identically zero at the last layer
    L = np.arange(last)
    fig, ax = plt.subplots(figsize=ps.SINGLE)
    depth_marks(ax, m)
    ax.plot(L, s[VIS][:last], color=ps.BLUE, lw=1.5, label="Vision")
    ax.plot(L, s[PT][:last], color=ps.YELLOW, lw=1.5, label="Prompt text")
    ax.plot(L, s[COC][:last], color=PURPLE, lw=1.5, label="CoC")
    ax.plot(L, s[HIST][:last], color=ps.BRICK, lw=1.5, label="Ego history")
    ax.set_xlabel("VLM layer index")
    ax.set_ylabel("Share of layer importance")
    ax.set_xlim(-0.5, m["n"] - 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(m["ticks"])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=ps.TEXT, handlelength=1.2,
              handletextpad=0.45, columnspacing=1.2, borderaxespad=0.2)
    export_panel(fig, out_dir, name)


# ---------------------------------------------------------------- Fig. 5 (fig_criterion.figure2 + --paper-style)
def steps_panel(M, nm, fname):
    """fig_criterion.figure2 heat map, as written"""
    n_step = M.shape[0]
    fig, ax = plt.subplots(figsize=fc.SINGLE)
    im = ax.imshow(M, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(n_step))
    ax.set_yticks(range(n_step))
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("Denoising step")
    for sp in ax.spines.values():
        sp.set_visible(True)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"rank agreement $\rho$", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fc.save(fig, fname)


def mass_panel(rq, rm):
    """fig_criterion.figure2 mass curve, as written; curve = [0, C(1/U), ..., C(1)] = np.r_[0, cq]"""
    TRAJ, COC_, GREY = fc.TRAJ, fc.COC, fc.GREY
    cq, cm = np.asarray(rq["curve"])[1:], np.asarray(rm["curve"])[1:]
    fig, ax = plt.subplots(figsize=fc.SINGLE)
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.8, ls="--", label="uniform")
    ax.plot(np.arange(len(cq) + 1) / len(cq), np.r_[0, cq], color=TRAJ, lw=1.5, label="Q head")
    ax.plot(np.arange(len(cm) + 1) / len(cm), np.r_[0, cm], color=COC_, lw=1.5, label="MLP channel")
    ax.axvline(0.9, color="black", lw=0.7, ls=":")
    q90, m90 = cq[int(0.9 * len(cq)) - 1], cm[int(0.9 * len(cm)) - 1]
    assert abs(q90 - rq["C_mark"]) < 1e-12 and abs(m90 - rm["C_mark"]) < 1e-12
    xq, xm = int(0.9 * len(cq)) / len(cq), int(0.9 * len(cm)) / len(cm)
    ax.scatter([xq], [q90], color=TRAJ, s=18, zorder=4)
    ax.scatter([xm], [m90], color=COC_, marker="s", s=18, zorder=4)
    ax.annotate(f"{100 * q90:.0f}%", (xq, q90), textcoords="offset points", xytext=(-30, 2), color=TRAJ, fontsize=7.5)
    ax.annotate(f"{100 * m90:.1f}%", (xm, m90), textcoords="offset points", xytext=(-32, 6), color=COC_, fontsize=7.5)
    ax.set_xlabel("Fraction of units removed")
    ax.set_ylabel("Importance removed")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fc.save(fig, "fig2_mass_curve")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="iclr27-rebuttal/supp_alpamayo2_figs")
    ap.add_argument("--assets", default=str(REPO / "iclr27-rebuttal" / "paper" / "fig" / "assets"))
    args = ap.parse_args()
    out = REPO / "outputs" / args.exp_id
    assets = Path(args.assets)
    panels = assets / "supp_alpamayo2_panels"
    panels.mkdir(parents=True, exist_ok=True)
    m3, m5 = json.loads(SRC3.read_text()), json.loads(SRC5.read_text())
    assert all(m3["gates_ok"].values()) and m5["gates_ok"], "source runs must have passed their gates"

    # Fig. 3 panels
    a15 = a15_corrected_series()
    stored15 = m3["per_layer"]["a15/q"]["agreement"]
    diff15 = {k: float(np.nanmax(np.abs(np.asarray(stored15[k]["corrected"], dtype=float) - v))) for k, v in a15.items()}
    for key in ("a15", "a2"):
        m, pl = MODELS[key], m3["per_layer"][f"{key}/q"]
        series = a15 if key == "a15" else {k: v["corrected"] for k, v in pl["agreement"].items()}
        rank_correlation_panel(series, m, panels, f"{key}_rank_correlation")
        share_panel(pl["share"]["traj"], m, "traj", panels, f"{key}_share_fm")
        share_panel(pl["share"]["coc"], m, "coc", panels, f"{key}_share_ce")

    # Fig. 5 panels through fig_criterion's own paper-style save
    fc.PAPER_STYLE, fc.EXPORT_PANELS = True, None
    fc.TRAJ, fc.COC = "#618BC8", "#ECAE3C"  # fig_criterion.main() sets these for --paper-style
    for key in ("a15", "a2"):
        tmp = out / f"fc_{key}"
        tmp.mkdir(parents=True, exist_ok=True)
        fc.OUT = tmp
        steps_panel(np.asarray(m5["results"][f"{key}/q"]["matrix"]), "Q head", "fig2_steps_q_head")
        steps_panel(np.asarray(m5["results"][f"{key}/mlp"]["matrix"]), "MLP channel", "fig2_steps_mlp")
        mass_panel(m5["results"][f"{key}/q"], m5["results"][f"{key}/mlp"])
        for src, dst in (("fig2_steps_q_head", "steps_q_head"), ("fig2_steps_mlp", "steps_mlp"), ("fig2_mass_curve", "mass_curve")):
            for ext in ("pdf", "png"):
                (panels / f"{key}_{dst}.{ext}").write_bytes((tmp / f"{src}.{ext}").read_bytes())

    # PNG previews of the panels go to outputs; fig/assets keeps only the PDFs the paper includes
    (out / "panels_png").mkdir(parents=True, exist_ok=True)
    for png in panels.glob("*.png"):
        shutil.move(png, out / "panels_png" / png.name)  # outputs/ is another file system (no rename)

    # captions layer (serif, sized to read ~10 pt at \linewidth) and composition
    spec = {"vlm": {"panels": VLM_PANELS, "titles": ["(a) Rank correlation", r"(b) $\mathit{I}_{\mathrm{FM}}$ by token type",
                                                      r"(c) $\mathit{I}_{\mathrm{CE}}$ by token type"]},
            "expert": {"panels": EXP_PANELS, "titles": ["(a) Q head rank similarity", "(b) MLP channel rank similarity",
                                                         "(c) Cumulative importance"]}}
    pw, ph = fc.SINGLE  # 3.5 x 2.8 in
    row_w, cap_h, cap_pt = 0.42, 0.5, 20
    W, H = row_w + 3 * pw, 2 * ph + cap_h
    plt.rcParams["mathtext.fontset"] = "stix"
    for fig_name, sp in spec.items():
        layer = plt.figure(figsize=(W, H))
        for col, t in enumerate(sp["titles"]):
            layer.text((row_w + (col + 0.5) * pw) / W, 0.18 / H, t, ha="center", va="bottom", fontfamily="STIXGeneral",
                       fontsize=cap_pt)
        for row, key in enumerate(("a15", "a2")):
            layer.text(0.04 / W, (cap_h + (1.5 - row) * ph) / H, MODELS[key]["name"], rotation=90, ha="left", va="center",
                       fontfamily="STIXGeneral", fontsize=cap_pt)
        layer.savefig(out / f"layer_{fig_name}.pdf")
        plt.close(layer)
        layout = {"page_in": [W, H], "panel_in": [pw, ph],
                  "place": [{"pdf": str(panels / f"{key}_{p}.pdf"), "x_in": row_w + col * pw, "y_in": cap_h + (1 - row) * ph}
                            for row, key in enumerate(("a15", "a2")) for col, p in enumerate(sp["panels"])]}
        (out / f"layout_{fig_name}.json").write_text(json.dumps(layout, indent=2))
        subprocess.run(["uvx", "--with", "pypdf", "python", str(REPO / "iclr27-rebuttal" / "code" / "supp_alpamayo2_compose.py"),
                        str(out / f"layer_{fig_name}.pdf"), str(out / f"layout_{fig_name}.json"),
                        str(assets / f"supp_alpamayo2_{fig_name}.pdf")], check=True)

    caption = {
        "grey_band": {k: {"layers_shaded": [int(np.ceil(m["band"][0])), int(np.floor(m["band"][1]))],
                          "definition": ("the paper's late band (Fig. 3)" if k == "a15" else
                                         "the A1.5 band moved by relative depth (22/36 x 64 = 39)")}
                      for k, m in MODELS.items()},
        "dotted_step": {k: {"tau_q_heads": m["tau"]["q"], "tau_mlp": m["tau"]["mlp"], "drawn_at": m["tau"]["q"] - 0.5}
                        for k, m in MODELS.items()},
        "expert_C09": {k: round(m5["results"][k]["C_mark"], 5) for k in m5["results"]},
        "a15_rank_correlation_vs_stored_2026_11_08_max_abs_diff": diff15,
        "page_in": [W, H], "panel_in": [pw, ph], "subcaption_pt": cap_pt}
    (out / "config.json").write_text(json.dumps(caption, indent=2))
    print(json.dumps(caption, indent=2))
    print(f"wrote {panels} and {assets}/supp_alpamayo2_{{vlm,expert}}.pdf")


if __name__ == "__main__":
    main()
