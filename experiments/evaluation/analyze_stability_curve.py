"""How many calibration clips before the pruning selection stops moving?

`2026-09-03_calib-draw-variance` established that a 100-clip draw is a gamble (SD 0.363
over six draws, reproduced on val500) but could not say whether 500 is safe: it had one
n=500 arm, so that arm's variance was unmeasurable.

This answers the question without evaluating anything. If two disjoint draws of size n
select the same units, the calibration draw stops mattering by construction -- no minADE
run needed. `mean(per-clip) == importance.npz` holds to 0.0 on every array, so a subset's
mean is exactly what a separate run over those clips would have produced, and one 4,000-
clip importance pass yields every point on the curve.

Two cautions the numbers have to be read with:

- Selection convergence is a CONSERVATIVE criterion. Units that churn may be
  interchangeable, in which case swapping them costs nothing -- G0b measured exactly
  that (a 2.8% Q-head move from changing GPU cost +0.0012, p=0.68). So the n this
  reports is an upper bound.
- That same G0b figure is what makes the question answerable at all: it is a measured
  churn budget. "Identical" is unreachable (extrapolating the earlier fit, 1% MLP
  disagreement needs ~1.1M clips, seven times the whole official train split), so the
  target is the churn already shown to be harmless, not zero.

Usage:
  python experiments/evaluation/analyze_stability_curve.py --importance importance_st4000
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C2, C3 = "#2a78d6", "#008300", "#D97757"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})

# the churn G0b showed costs nothing measurable: same clips, same seeds, different GPU
G0B_CHURN = {"q": 0.0278, "mlp": 0.0271}
G0B_COST = "+0.0012 [-0.0042, +0.0080], p=0.68"


def load(importance):
    d = REPO / "outputs" / importance
    per = dict(np.load(d / "importance_perclip.npz"))
    full = dict(np.load(d / "importance.npz"))
    n = next(iter(per.values())).shape[0]
    # the identity the whole derivation rests on, re-checked on this run
    worst = max(float(np.abs(full[k] - per[k].astype(np.float64).mean(0)).max())
                for k in full)
    return per, full, n, worst


def selector(full):
    import mask_lib as ml
    import tyr_lib as tyr
    from make_slim import allocations

    ref = json.loads(
        (REPO / "outputs" / "slim_integrated_mag" / "slim_meta.json").read_text())
    allocs, _ = allocations(full, ref, 36, 32, 12288, 0.5)
    rq, rm = allocs["uniform"]

    def sel(imp):
        sq, sm = tyr.dual_scores(imp)
        return ml.select_mask_ratios(sq, rq), ml.select_mask_ratios(sm, rm)
    return sel


def curve(per, sel, n_clips, sizes, draws, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for n in sizes:
        if 2 * n > n_clips:
            continue
        q, m = [], []
        for _ in range(draws):
            p = rng.permutation(n_clips)
            a = sel({k: v[p[:n]].astype(np.float64).mean(0) for k, v in per.items()})
            b = sel({k: v[p[n:2 * n]].astype(np.float64).mean(0) for k, v in per.items()})
            q.append(float((a[0] * b[0]).sum() / a[0].sum()))
            m.append(float((a[1] * b[1]).sum() / a[1].sum()))
        out[n] = {"q": float(np.mean(q)), "q_sd": float(np.std(q)),
                  "mlp": float(np.mean(m)), "mlp_sd": float(np.std(m)),
                  # the fraction that DISagrees is what the fit and the report read;
                  # stored rather than recomputed so both say the same thing
                  "q_disagree": float(1 - np.mean(q)),
                  "mlp_disagree": float(1 - np.mean(m)),
                  "draws": len(q)}
        print(f"  n={n:5d}  Q {np.mean(q):.4f}+-{np.std(q):.4f}   "
              f"MLP {np.mean(m):.4f}+-{np.std(m):.4f}", flush=True)
    return out


def fit(cur, axis, lo=0):
    """Power-law fit of disagreement (1 - overlap) against n, over points with n >= lo."""
    ns = np.array([n for n in sorted(cur) if n >= lo], dtype=float)
    d = np.array([1 - cur[int(n)][axis] for n in ns])
    if len(ns) < 3:
        return None
    s, c = np.polyfit(np.log(ns), np.log(d), 1)
    resid = np.log(d) - (s * np.log(ns) + c)
    need = float(np.exp((np.log(G0B_CHURN[axis]) - c) / s))
    return {"exponent": float(s), "intercept": float(c),
            "rms_log_resid": float(np.sqrt((resid ** 2).mean())),
            "n_for_g0b_churn": need, "fitted_from": int(ns[0])}


def plot(cur, fits, out):
    ns = sorted(cur)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for axis, col, lab in (("q", C1, "Q head"), ("mlp", C2, "MLP channel")):
        y = [1 - cur[n][axis] for n in ns]
        ax.plot(ns, y, "o", color=col, label=f"{lab} (measured)")
        f = fits.get(axis)
        if f:
            xs = np.array([min(ns), max(ns) * 20], dtype=float)
            ax.plot(xs, np.exp(f["intercept"]) * xs ** f["exponent"], "-", color=col,
                    lw=1, alpha=.6, label=f"{lab}  n^{f['exponent']:.3f}")
        ax.axhline(G0B_CHURN[axis], color=col, ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("clips per draw (two disjoint draws)")
    ax.set_ylabel("fraction of the kept set that disagrees")
    ax.set_title("does the selection settle, and where?")
    ax.legend(fontsize=8)
    fig.text(0.5, 0.005, "dotted: the churn a GPU change caused, which cost "
             f"{G0B_COST}", ha="center", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "stability.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--importance", default="importance_st4000")
    ap.add_argument("--out", default="outputs/calib_stability")
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = REPO / args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)

    per, full, n_clips, worst = load(args.importance)
    print(f"{args.importance}: {n_clips} clips, "
          f"max |mean(per-clip) - importance.npz| = {worst:.3e}")
    sizes = [10, 25, 50, 100, 150, 250, 400, 600, 800, 1000, 1500, 2000]
    cur = curve(per, selector(full), n_clips, sizes, args.draws, args.seed)

    # fitted twice: over everything, and over the large-n half only. If the exponent
    # steepens on the second fit the curve is bending and the extrapolation from small n
    # was pessimistic -- that is H1's falsifier.
    fits = {}
    for axis in ("q", "mlp"):
        fits[axis] = fit(cur, axis)
        fits[axis + "_large_n"] = fit(cur, axis, lo=250)
    m = {"clips": n_clips, "identity_max_diff": worst, "curve": cur, "fits": fits,
         "g0b_churn": G0B_CHURN, "g0b_cost": G0B_COST}
    plot(cur, fits, out / "plots")
    (out / "metrics.json").write_text(json.dumps(m, indent=2))

    lines = [f"== selection stability, {n_clips} calibration clips ==",
             f"identity check max|diff| {worst:.3e}"]
    for axis in ("q", "mlp"):
        a, b = fits[axis], fits[axis + "_large_n"]
        lines.append(f"{axis}: all points n^{a['exponent']:+.3f} "
                     f"(rms log resid {a['rms_log_resid']:.3f}) -> "
                     f"n={a['n_for_g0b_churn']:,.0f} for {G0B_CHURN[axis]:.1%} churn")
        if b:
            lines.append(f"{axis}: n>=250    n^{b['exponent']:+.3f} -> "
                         f"n={b['n_for_g0b_churn']:,.0f}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print("->", out, flush=True)


if __name__ == "__main__":
    main()
