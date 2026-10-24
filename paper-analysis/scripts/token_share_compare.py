"""Replication table for the Finding-1 panels: band-mean token shares (additive split, MLP and Q)
and rank-agreement band means for every gradient-anatomy run given, as Markdown, with the
difference from the reference run (first argument) flagged when |delta| > 0.10 (the
pre-registered window in plans/2026-09-24_token-share-replication.md).
Usage: python token_share_compare.py gradanat_v1 gradanat_val100 gradanat_ood gradanat_test100"""
import json
import sys

import numpy as np

from paper_style import REPO, fc

VIS, HIST, PT, COC = 0, 1, 2, 3
BANDS = {"6-17": slice(6, 18), "27-34": slice(27, 35)}
RBANDS = {"0-6": slice(0, 7), "7-21": slice(7, 22), "22-34": slice(22, 35)}
TOL = 0.10
runs = sys.argv[1:] or ["gradanat_v1"]


def shares(run, axis):
    m = json.loads((REPO / "outputs" / run / "metrics_analysis.json").read_text())["additive_by_type"]
    out = {}
    for obj, key in (("I_traj", f"{axis}_fm"), ("I_CoC", f"{axis}_ce")):
        s = np.array(m[key]["share_by_type_and_layer"])  # (5, 36)
        for band, sl in BANDS.items():
            for tname, t in (("vision", s[VIS]), ("text", s[PT] + s[COC]), ("hist", s[HIST])):
                out[(obj, band, tname)] = float(t[sl].mean())
    return out


def rank(run):
    d = REPO / "outputs" / run
    if (d / "importance.npz").exists():
        with np.load(d / "importance.npz") as z:
            rq = fc.rho_per_layer(z["traj_vlm_q"], z["coc_vlm_q"])
            rm = fc.rho_per_layer(z["traj_vlm_mlp"], z["coc_vlm_mlp"])
    else:
        with np.load(d / "anatomy.npz") as z:
            rq = fc.rho_per_layer(z["fm_full_q"], z["ce_full_q"])
            rm = fc.rho_per_layer(z["fm_full_mlp"], z["ce_full_mlp"])
    return {(a, b): float(np.nanmean(r[sl])) for a, r in (("Q", rq), ("MLP", rm)) for b, sl in RBANDS.items()}


def table(title, rows, cols, values):
    print(f"\n**{title}** (reference = {runs[0]}; `*` marks |delta| > {TOL:.2f})\n")
    print("| " + " | ".join(["cell"] + cols) + " |")
    print("|" + "---|" * (len(cols) + 1))
    flagged = 0
    for r in rows:
        cells = []
        for c in cols:
            v, ref = values[c][r], values[cols[0]][r]
            flag = "*" if c != cols[0] and abs(v - ref) > TOL else ""
            cells.append(f"{v:.2f}{flag}")
            flagged += bool(flag)
        print("| " + " | ".join([" / ".join(r)] + cells) + " |")
    print(f"\ncells outside the window: {flagged}")


for axis, label in (("mlp", "MLP channels"), ("q", "Q heads")):
    vals = {run: shares(run, axis) for run in runs}
    rows = [(o, b, t) for o in ("I_traj", "I_CoC") for b in BANDS for t in ("vision", "text", "hist")]
    table(f"Token shares, {label}", rows, runs, vals)
vals = {run: rank(run) for run in runs}
table("Rank agreement (Spearman rho, band means)", [(a, b) for a in ("Q", "MLP") for b in RBANDS], runs, vals)
