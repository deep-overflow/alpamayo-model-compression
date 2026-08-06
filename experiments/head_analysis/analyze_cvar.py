"""Does a tail-risk aggregate of Taylor importance rank units differently from the mean?

Every structured-pruning criterion aggregates per-sample sensitivity as an expectation,
I_u = E_x[|dL/dg_u|]. Our own closed-loop result argued that is the wrong functional: the
damage from compression showed up in the tail (lead-vehicle time-headway 5%ile -1.19 s,
p=0.0043) while the mean scene score stayed inside noise. This script asks the cheap version
of the question first, before any eval GPU time is spent:

    if we aggregate with CVaR_alpha (mean over the worst alpha fraction of calibration clips)
    instead of the mean, does the ranking actually change?

If the two rankings agree, the idea is dead and costs nothing more. If they diverge -- and
especially if they diverge in the late layers where the two objectives already disagree --
then a risk-averse criterion is selecting genuinely different units and is worth evaluating.

Also verifies that the per-clip terms average back to the aggregate the run itself stored,
and (with --ref) that they reproduce a previous run's importance.npz. Since importance_cvar
uses the same calibration clips and seed as importance_v1, that second check doubles as a
regression test on the rebuilt environment.

Usage: python analyze_cvar.py --exp-id importance_cvar [--ref importance_v1]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]

BG = "#FAF9F5"
INK = "#29261B"
MUTED = "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

BANDS = [(0, 5), (6, 17), (18, 29), (30, 35)]
ALPHAS = (0.1, 0.2, 0.5)
UNITS = ["vlm_q", "vlm_mlp", "kv_k", "kv_v", "exp_q", "exp_mlp"]


def cvar(per_clip, alpha):
    """Mean over the worst-alpha fraction of clips, per unit. (C, L, U) -> (L, U)."""
    n = per_clip.shape[0]
    k = max(1, round(alpha * n))
    part = np.partition(per_clip, n - k, axis=0)[n - k:]  # top-k along the clip axis
    return part.mean(0)


def layer_agreement(a, b):
    """Within-layer Spearman and top-20% overlap between two score arrays. (L, U)."""
    rho, ov = [], []
    for li in range(a.shape[0]):
        x, y = a[li], b[li]
        if np.allclose(x, 0) or np.allclose(y, 0):
            rho.append(np.nan)
            ov.append(np.nan)
            continue
        rho.append(spearmanr(x, y).statistic)
        k = max(1, round(0.2 * len(x)))
        ov.append(len(set(np.argsort(-x)[:k]) & set(np.argsort(-y)[:k])) / k)
    return np.array(rho, dtype=float), np.array(ov, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="importance_cvar")
    ap.add_argument("--ref", default="importance_v1", help="previous run to reproduce; '' to skip")
    args = ap.parse_args()

    out = REPO / "outputs" / args.exp_id
    (out / "plots").mkdir(parents=True, exist_ok=True)
    pcz = np.load(out / "importance_perclip.npz")
    aggz = np.load(out / "importance.npz")
    n_clips = pcz[next(iter(pcz.files))].shape[0]

    lines = [f"Tail-risk (CVaR) vs mean aggregation of Taylor importance -- {args.exp_id}",
             f"  {n_clips} calibration clips", ""]

    # ---- consistency: per-clip mean must reproduce the stored aggregate
    lines.append("Consistency of the per-clip decomposition (max |mean(per_clip) - stored|)")
    consistency = {}
    for key in sorted(pcz.files):
        err = float(np.abs(pcz[key].mean(0) - aggz[key]).max())
        scale = float(np.abs(aggz[key]).max())
        consistency[key] = {"max_abs_err": err, "scale": scale}
        lines.append(f"  {key:14s} err {err:.3e}   scale {scale:.3e}   "
                     f"{'OK' if err <= 1e-6 * max(scale, 1e-12) + 1e-9 else 'MISMATCH'}")

    if args.ref:
        refz = np.load(REPO / "outputs" / args.ref / "importance.npz")
        lines += ["", f"Reproduction of {args.ref} (same calib clips and seed)"]
        for key in sorted(pcz.files):
            if key not in refz.files:
                continue
            a, b = pcz[key].mean(0), refz[key]
            denom = max(float(np.abs(b).max()), 1e-12)
            rel = float(np.abs(a - b).max() / denom)
            rho = float(spearmanr(a.ravel(), b.ravel()).statistic)
            lines.append(f"  {key:14s} max rel diff {rel:.3e}   global rho {rho:+.4f}")

    # ---- does CVaR rank differently from the mean?
    metrics = {"n_clips": n_clips, "consistency": consistency, "alphas": list(ALPHAS), "units": {}}
    lines += ["", "Does CVaR rank units differently from the mean?",
              "  rho = within-layer Spearman(CVaR, mean); ov = top-20% overlap",
              "  low rho => the tail selects different units => the criterion is new information", ""]
    for obj in ("coc", "traj"):
        for unit in UNITS:
            key = f"{obj}_{unit}"
            if key not in pcz.files:
                continue
            pc = pcz[key]
            if np.allclose(pc, 0):
                continue
            mean = pc.mean(0)
            row = {"bands": {}}
            head = f"  {key:14s}"
            for alpha in ALPHAS:
                rho, ov = layer_agreement(cvar(pc, alpha), mean)
                row[f"alpha{alpha}"] = {"rho_mean": float(np.nanmean(rho)),
                                        "ov_mean": float(np.nanmean(ov)),
                                        "rho_by_layer": np.where(np.isnan(rho), None, rho).tolist()}
                row["bands"][f"alpha{alpha}"] = {
                    f"{a}-{b}": float(np.nanmean(rho[a:b + 1])) for a, b in BANDS}
                head += f"   a={alpha:.1f}: rho {np.nanmean(rho):+.3f} ov {np.nanmean(ov):.3f}"
            metrics["units"][key] = row
            lines.append(head)

    # ---- why: is the importance mass clip-concentrated, and is the concentration shared?
    lines += ["", "Is importance clip-concentrated, and do the SAME clips dominate every unit?",
              f"  uniform would give max-share {1 / n_clips:.3f} and top5-share {5 / n_clips:.3f};",
              "  a modal-clip share near 1/n means each unit has its own worst clip (independent),",
              "  a large modal share means one clip scales the whole layer at once",
              (f"  {'key':16s}{'median max share':>18s}{'top5 share':>12s}"
               f"{'modal-clip share':>18s}{'distinct/layer':>16s}")]
    conc = {}
    for key in ("coc_vlm_q", "coc_vlm_mlp", "traj_vlm_q", "traj_vlm_mlp", "traj_exp_q"):
        if key not in pcz.files:
            continue
        pc = pcz[key]
        if np.allclose(pc, 0):
            continue
        tot = np.maximum(pc.sum(0), 1e-30)
        share = (pc / tot).max(0).ravel()
        top5 = (np.sort(pc, axis=0)[-5:].sum(0) / tot).ravel()
        top = pc.argmax(0)
        modal, ndist = [], []
        for li in range(top.shape[0]):
            _, cnt = np.unique(top[li], return_counts=True)
            modal.append(cnt.max() / top.shape[1])
            ndist.append(len(cnt))
        conc[key] = {"median_max_share": float(np.median(share)),
                     "median_top5_share": float(np.median(top5)),
                     "modal_clip_share": float(np.mean(modal)),
                     "distinct_dominant_clips_per_layer": float(np.mean(ndist))}
        lines.append(f"  {key:16s}{np.median(share):>18.3f}{np.median(top5):>12.3f}"
                     f"{np.mean(modal):>18.3f}{np.mean(ndist):>16.1f}")
    metrics["concentration"] = conc

    # ---- the allocation-level version of the same question
    lines += ["", "Does tail-weighting change the layerwise budget, even if within-layer ranks hold?",
              "  (rank-based selection is invariant to a monotone rescaling of a layer, so the",
              "   layer-level mass profile is where a shared-difficulty effect could still show up)",
              f"  {'key':16s}{'rho(layer mass)':>17s}{'max |share diff|':>18s}   band shares mean -> CVaR@0.1"]
    profile = {}
    for key in ("coc_vlm_q", "coc_vlm_mlp", "traj_vlm_q", "traj_vlm_mlp"):
        if key not in pcz.files:
            continue
        pc = pcz[key]
        if np.allclose(pc, 0):
            continue
        m = pc.mean(0).sum(1)
        c = cvar(pc, 0.1).sum(1)
        m, c = m / m.sum(), c / c.sum()
        profile[key] = {"rho": float(spearmanr(m, c).statistic),
                        "max_share_diff": float(np.abs(m - c).max()),
                        "bands": {f"{a}-{b}": [float(m[a:b + 1].sum()), float(c[a:b + 1].sum())]
                                  for a, b in BANDS}}
        bands = "  ".join(f"{a}-{b}:{m[a:b + 1].sum():.3f}->{c[a:b + 1].sum():.3f}" for a, b in BANDS)
        lines.append(f"  {key:16s}{profile[key]['rho']:>17.4f}"
                     f"{profile[key]['max_share_diff']:>18.4f}   {bands}")
    metrics["layer_profile"] = profile

    # ---- depth profile of the divergence, for the two units the configs actually prune
    lines += ["", "Depth profile of rho(CVaR@0.1, mean) -- where does the tail disagree most?",
              f"  {'band':10s}" + "".join(f"{f'{o}_{u}':>18s}"
                                          for o in ("coc", "traj") for u in ("vlm_q", "vlm_mlp"))]
    for a, b in BANDS:
        row = f"  {a:2d}-{b:2d}     "
        for o in ("coc", "traj"):
            for u in ("vlm_q", "vlm_mlp"):
                key = f"{o}_{u}"
                v = metrics["units"].get(key, {}).get("bands", {}).get("alpha0.1", {}).get(f"{a}-{b}")
                row += f"{'n/a':>18s}" if v is None else f"{v:>+18.3f}"
        lines.append(row)

    (out / "cvar_analysis.json").write_text(json.dumps(metrics, indent=2))
    (out / "cvar_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- figure
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    layers = np.arange(aggz["coc_vlm_q"].shape[0])

    ax = axes[0]
    for key, col in [("coc_vlm_q", C1), ("traj_vlm_q", C3), ("coc_vlm_mlp", C2), ("traj_vlm_mlp", C4)]:
        r = metrics["units"].get(key, {}).get("alpha0.1", {}).get("rho_by_layer")
        if r:
            ax.plot(layers, [np.nan if v is None else v for v in r], color=col, lw=1.5, label=key)
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("layer"); ax.set_ylabel(r"Spearman(CVaR$_{0.1}$, mean)")
    ax.set_title("does the tail rank differently?")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    pc = pcz["traj_vlm_mlp"]
    mean, cv = pc.mean(0), cvar(pc, 0.1)
    li = aggz["coc_vlm_q"].shape[0] - 3
    ax.scatter(mean[li], cv[li], s=4, alpha=0.3, color=C1, edgecolors="none")
    lim = [0, max(mean[li].max(), cv[li].max()) * 1.05]
    ax.plot(lim, lim, color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("mean importance"); ax.set_ylabel(r"CVaR$_{0.1}$ importance")
    ax.set_title(f"traj_vlm_mlp, layer {li}")

    ax = axes[2]
    share = (pc / np.maximum(pc.sum(0, keepdims=True), 1e-30)).max(0).ravel()
    ax.hist(share, bins=60, color=C2)
    ax.axvline(1.0 / n_clips, color=MUTED, lw=1.2, ls="--", label=f"uniform (1/{n_clips})")
    ax.set_xlabel("largest single-clip share of a unit's total importance")
    ax.set_ylabel("units")
    ax.set_title("is importance clip-concentrated?")
    ax.legend(frameon=False, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out / "plots" / "cvar_vs_mean.png", dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
