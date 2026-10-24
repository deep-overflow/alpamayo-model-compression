"""Importance per token, by token type: the typed layer score divided by the number of tokens of
that type -- computed per clip (CoC length varies) and averaged, so a 2,880-token type is not
credited for its count. Two rows per axis (I_traj / I_CoC), log y; the ratio I_traj / I_CoC is
unchanged by this normalisation (same divisor top and bottom), so only absolute values are shown.
Data: outputs/gradanat_v1/anatomy_perclip_{q,mlp}.npz (signed per-clip typed gradients) and
metrics.json (per-clip n_tok)."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/home/cvlab21/project/chan/alpamayo-model-compression")
sys.path.insert(0, str(REPO / "experiments" / "paper"))
import fig_criterion as fc  # noqa: E402
import fig_why_differs as fw  # noqa: E402

OUT = Path(__file__).resolve().parent / "figs"
fc.OUT = OUT
TYPES = ("vision", "hist", "prompt_text", "coc", "sink")
VIS, HIST, PT, COC, SINK = range(5)
LINES = ((VIS, "vision tokens", fc.TRAJ, "-"), (PT, "prompt-text tokens", fw.PURPLE, (0, (4, 2))),
         (COC, "CoC tokens", fc.COC, (0, (1, 1))), (HIST, "ego-history tokens", fw.GREEN, (0, (3, 1, 1, 1))))
L = np.arange(fw.N_L)

recs = json.loads((REPO / "outputs" / "gradanat_v1" / "metrics.json").read_text())["per_clip"]
n_tok = np.array([[r["n_tok"][t] for t in TYPES] for r in recs], dtype=np.float64)  # (N, 5)
print("tokens per clip (mean):", {t: round(float(n_tok[:, i].mean()), 1) for i, t in enumerate(TYPES)})

for name, fname, fkey, ckey in (("q", "fig4_pertoken_q_head", "fm_full", "ce"), ("mlp", "fig4_pertoken_mlp", "fm_type", "ce_type")):
    with np.load(REPO / "outputs" / "gradanat_v1" / f"anatomy_perclip_{name}.npz") as z:
        fm, ce = z[fkey], z[ckey]  # (N, 5, 36, U) signed
    assert fm.shape[0] == n_tok.shape[0]
    with np.load(REPO / "outputs" / "gradanat_v1" / "anatomy.npz") as a:  # consistency with the stored E|G|
        chk = np.abs(fm.astype(np.float64)).mean(0).sum(-1)[:, :fw.LAST]
        ref = a[f"fm_type_{name}"].sum(-1)[:, :fw.LAST]
        print(f"{name}: per-clip |FM| mean vs stored typed score, max rel diff {np.max(np.abs(chk - ref) / ref):.1e}")
    per_tok = {}
    for loss, G in ((r"$I_{\mathrm{traj}}$", fm), (r"$I_{\mathrm{CoC}}$", ce)):
        s = np.abs(G.astype(np.float64)).sum(-1)  # (N, 5, 36) typed layer sums per clip
        per_tok[loss] = (s / n_tok[:, :, None]).mean(0)  # (5, 36): importance per token, clip-averaged
    fig, axes = plt.subplots(2, 1, figsize=(fc.SINGLE[0], fc.SINGLE[1] * 1.85), sharex=True)
    print(f"== {name}  per-token importance, band means L6-17 / L27-34 (relative to vision per token)")
    for ax, (loss, P) in zip(axes, per_tok.items()):
        last = fw.LAST if "traj" in loss else fw.N_L
        ax.axvspan(21.5, 35.5, color=fc.GREY, alpha=0.13, lw=0)
        for ti, label, col, ls in LINES:
            ax.plot(L[:last], P[ti][:last], color=col, lw=1.2 if ti != VIS else 1.4, ls=ls, label=label)
            e, l = P[ti][6:18].mean(), P[ti][27:35].mean()
            ev, lv = P[VIS][6:18].mean(), P[VIS][27:35].mean()
            print(f"  {loss:18s} {label:20s} {e:.2e} ({e / ev:6.1f}x vision) -> {l:.2e} ({l / lv:6.1f}x vision)")
        ax.set_yscale("log")
        ax.set_xlim(-0.5, 35.5)
        ax.set_ylabel(f"{loss} per token")
        ax.text(0.02, 0.95, loss, transform=ax.transAxes, ha="left", va="top", fontsize=8)
    axes[1].set_xlabel("VLM layer")
    axes[1].set_xticks([0, 7, 14, 21, 28, 35])
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6,
                   handlelength=1.8, columnspacing=1.0, borderaxespad=0.2)
    fig.subplots_adjust(hspace=0.12)
    fc.save(fig, fname)
    print(f"  -> {OUT / (fname + '.png')}")
