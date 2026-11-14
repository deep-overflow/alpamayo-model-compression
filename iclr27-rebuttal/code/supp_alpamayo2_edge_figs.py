"""Appendix C.4 panels for Alpamayo 2 Edge, drawn with supp_alpamayo2_figs.py's own panel functions.

Writes into paper/fig/assets/:
  supp_alpamayo2_panels/edge_{rank_correlation, share_fm, share_ce, steps_q_head, steps_mlp, mass_curve}.pdf
  supp_alpamayo2_{vlm,expert}_3row.pdf   (A1.5 / A2 Super / A2 Edge; the paper's two-row PDFs are left untouched)
The A1.5 and A2 Super panels are the existing files in supp_alpamayo2_panels/ (placed unchanged).

Edge values: outputs/iclr27-rebuttal/edge_paper_analysis/metrics.json (edge_paper_analysis.py, gated against the stored
A2 numbers). Grey band = the A1.5 late band moved by relative depth (22/36 x 28 = 17 -> layers 17-27), dotted line =
Edge's own I_FM / I_CE step tau (Q heads), as for A2 Super.

Usage: .venv/bin/python iclr27-rebuttal/code/supp_alpamayo2_edge_figs.py [--exp-id iclr27-rebuttal/supp_alpamayo2_edge_figs]
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

sys.path.insert(0, str(Path(__file__).parent))
import supp_alpamayo2_figs as saf

REPO = saf.REPO
SRC = REPO / "outputs" / "iclr27-rebuttal" / "edge_paper_analysis" / "metrics.json"
fc = saf.fc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="iclr27-rebuttal/supp_alpamayo2_edge_figs")
    ap.add_argument("--assets", default=str(REPO / "iclr27-rebuttal" / "paper" / "fig" / "assets"))
    args = ap.parse_args()
    out = REPO / "outputs" / args.exp_id
    out.mkdir(parents=True, exist_ok=True)
    assets = Path(args.assets)
    panels = assets / "supp_alpamayo2_panels"
    me = json.loads(SRC.read_text())
    assert me["gates_ok"], "edge_paper_analysis gates must pass"
    m = {**me["model"], "band": tuple(me["model"]["band"])}  # name, n, band, ticks, tau

    pl = {ax: me["per_layer"][f"edge/{ax}"] for ax in ("q", "mlp")}
    series = {k: v["corrected"] for k, v in pl["q"]["agreement"].items()}
    tmp_png = out / "panels_png"
    tmp_png.mkdir(exist_ok=True)
    saf.rank_correlation_panel(series, m, panels, "edge_rank_correlation")
    saf.share_panel(pl["q"]["share"]["traj"], m, "traj", panels, "edge_share_fm")
    saf.share_panel(pl["q"]["share"]["coc"], m, "coc", panels, "edge_share_ce")

    fc.PAPER_STYLE, fc.EXPORT_PANELS = True, None
    fc.TRAJ, fc.COC = "#618BC8", "#ECAE3C"
    tmp = out / "fc_edge"
    tmp.mkdir(parents=True, exist_ok=True)
    fc.OUT = tmp
    ex = {ax: me["expert"][f"edge/{ax}"] for ax in ("q", "mlp")}
    saf.steps_panel(np.asarray(ex["q"]["matrix"]), "Q head", "fig2_steps_q_head")
    saf.steps_panel(np.asarray(ex["mlp"]["matrix"]), "MLP channel", "fig2_steps_mlp")
    saf.mass_panel(ex["q"], ex["mlp"])
    for src, dst in (("fig2_steps_q_head", "steps_q_head"), ("fig2_steps_mlp", "steps_mlp"),
                     ("fig2_mass_curve", "mass_curve")):
        for ext in ("pdf", "png"):
            (panels / f"edge_{dst}.{ext}").write_bytes((tmp / f"{src}.{ext}").read_bytes())
    for png in panels.glob("edge_*.png"):  # fig/assets keeps only PDFs
        shutil.move(png, tmp_png / png.name)

    # three-row composites: A1.5 / A2 Super panels as they are in supp_alpamayo2_panels, Edge below
    names = {"a15": "Alpamayo 1.5", "a2": "Alpamayo 2 Super", "edge": "Alpamayo 2 Edge"}
    rows = ("a15", "a2", "edge")
    spec = {"vlm": {"panels": saf.VLM_PANELS, "titles": ["(a) Rank correlation", r"(b) $\mathit{I}_{\mathrm{FM}}$ by token type",
                                                         r"(c) $\mathit{I}_{\mathrm{CE}}$ by token type"]},
            "expert": {"panels": saf.EXP_PANELS, "titles": ["(a) Q head rank similarity",
                                                            "(b) MLP channel rank similarity", "(c) Cumulative importance"]}}
    pw, ph = fc.SINGLE
    row_w, cap_h, cap_pt = 0.42, 0.5, 20
    W, H = row_w + 3 * pw, len(rows) * ph + cap_h
    plt.rcParams["mathtext.fontset"] = "stix"
    for fig_name, sp in spec.items():
        layer = plt.figure(figsize=(W, H))
        for col, t in enumerate(sp["titles"]):
            layer.text((row_w + (col + 0.5) * pw) / W, 0.18 / H, t, ha="center", va="bottom",
                       fontfamily="STIXGeneral", fontsize=cap_pt)
        for row, key in enumerate(rows):
            layer.text(0.04 / W, (cap_h + (len(rows) - 0.5 - row) * ph) / H, names[key], rotation=90, ha="left",
                       va="center", fontfamily="STIXGeneral", fontsize=cap_pt)
        layer.savefig(out / f"layer_{fig_name}_3row.pdf")
        plt.close(layer)
        layout = {"page_in": [W, H], "panel_in": [pw, ph],
                  "place": [{"pdf": str(panels / f"{key}_{p}.pdf"), "x_in": row_w + col * pw,
                             "y_in": cap_h + (len(rows) - 1 - row) * ph}
                            for row, key in enumerate(rows) for col, p in enumerate(sp["panels"])]}
        (out / f"layout_{fig_name}_3row.json").write_text(json.dumps(layout, indent=2))
        subprocess.run(["uvx", "--with", "pypdf", "python", str(Path(__file__).parent / "supp_alpamayo2_compose.py"),
                        str(out / f"layer_{fig_name}_3row.pdf"), str(out / f"layout_{fig_name}_3row.json"),
                        str(assets / f"supp_alpamayo2_{fig_name}_3row.pdf")], check=True)
    cfg = {"source": str(SRC), "model": me["model"],
           "grey_band": {"layers_shaded": [int(np.ceil(m["band"][0])), int(np.floor(m["band"][1]))],
                         "definition": "the A1.5 band moved by relative depth (22/36 x 28 = 17)"},
           "dotted_step": {"tau_q_heads": m["tau"]["q"], "tau_mlp": m["tau"]["mlp"], "drawn_at": m["tau"]["q"] - 0.5},
           "expert_C09": {ax: round(ex[ax]["C_mark"], 5) for ax in ex}, "page_in_3row": [W, H]}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg, indent=2))
    print(f"wrote edge panels to {panels} and {assets}/supp_alpamayo2_{{vlm,expert}}_3row.pdf")


if __name__ == "__main__":
    main()
