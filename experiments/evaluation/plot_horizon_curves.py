"""minADE@K / minFDE@K at every waypoint, from runs evaluated with --save-pred.

`run_baseline.py` stores the full-horizon scalars plus two truncations (1.6 s / 3.2 s),
and those three points cannot be interpolated into a curve: `min` over the K samples
picks a DIFFERENT sample at each horizon, so an aggregate at 6.4 s carries no information
about 3 s. The curve has to be re-reduced from the paths, which is what `--save-pred`
stores as `pred_xy_k` (K, 64, 2).

That distinction is the reason this script exists. It was written after a waypoint-weight
study read three different verdicts off three different aggregates
(reports/evaluation/2026-09-13_waypoint-weight-and-quant-composition.html): 6.4 s alone
said "no effect", minADE at 1.6 s gave a perfect rho = -1.000 ladder over three arms, and
minFDE at the same horizon scattered it. Three points are monotone by chance one time in
three. Only the full curve settled it.

GT comes from the clip cache at the manifest's t0, the same resolution run_baseline used,
so nothing here depends on a stored GT copy.

    .venv/bin/python experiments/evaluation/plot_horizon_curves.py \
        --arms "dual=sp_dual_indist" "dualw_lin=sp_dualw_lin_indist" \
        --out outputs/earlyweight_eval

The first --arms entry is the reference the delta panel subtracts.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "evaluation"))
import sample_cache as sc

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
PALETTE = [INK, "#2a78d6", "#008300", "#e87ba4", "#eda100", "#7b5bd6", "#b0552f"]
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})


def load_rows(exp):
    """clip_id -> row, keeping only rows that actually carry the saved paths."""
    rows = {}
    for f in glob.glob(str(REPO / "outputs" / exp / "*_s*of*.json")):
        for r in json.loads(Path(f).read_text()):
            if "pred_xy_k" in r:
                rows[r["clip_id"]] = r
    return rows


def curves(rows, ids, gt, k, n_tok):
    """(minADE(t), minFDE(t)) means over clips, reduced at every waypoint."""
    ade = np.zeros((len(ids), n_tok))
    fde = np.zeros((len(ids), n_tok))
    for j, cid in enumerate(ids):
        p = np.asarray(rows[cid]["pred_xy_k"], dtype=float)[:k]  # (K, n_tok, 2)
        err = np.linalg.norm(p - gt[cid][None], axis=2)  # (K, n_tok)
        run = np.cumsum(err, axis=1) / np.arange(1, n_tok + 1)[None]
        ade[j] = run.min(0)  # min over samples AT each horizon, not a fixed argmin
        fde[j] = err.min(0)
    return ade.mean(0), fde.mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True,
                    help="label=exp_id, first one is the reference for the delta panel")
    ap.add_argument("--manifest", default="indist_500")
    ap.add_argument("--cache", default="eval")
    ap.add_argument("--k", type=int, default=6, help="samples the min is taken over")
    ap.add_argument("--mark", type=float, default=1.9,
                    help="vertical marker, default the measured effective plan horizon")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    arms = [a.split("=", 1) for a in args.arms]
    data = {}
    for label, exp in arms:
        r = load_rows(exp)
        if not r:
            raise SystemExit(f"{exp}: no rows with pred_xy_k -- was it run with --save-pred?")
        data[label] = r
    ids = sorted(set.intersection(*[set(v) for v in data.values()]))
    n_tok = len(data[arms[0][0]][ids[0]]["pred_xy_k"][0])
    print(f"{len(ids)} clips, {len(arms)} arms, {n_tok} waypoints")

    man = pd.read_parquet(REPO / "outputs" / "eval_sets" / f"{args.manifest}.parquet")
    t0_of = dict(zip(man["clip_id"], man["t0_us"]))
    gt = {}
    for cid in ids:
        z = np.load(sc.path_for(args.cache, cid, int(t0_of[cid])), allow_pickle=True)
        gt[cid] = np.asarray(z["ego_future_xyz"])[0, 0, :n_tok, :2]

    cur = {lab: curves(data[lab], ids, gt, args.k, n_tok) for lab, _ in arms}
    t = np.arange(1, n_tok + 1) * 0.1
    plots = args.out / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    for fname, title, ref in (
            ("horizon_curves_full.png", "", None),
            ("horizon_curves_delta.png", ": arm minus reference  (below 0 = better)",
             arms[0][0])):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
        for i, (lab, _) in enumerate(arms):
            if ref is not None and lab == ref:
                continue
            for ax, m in ((axes[0], 0), (axes[1], 1)):
                y = cur[lab][m] - (cur[ref][m] if ref else 0.0)
                ax.plot(t, y, color=PALETTE[i % len(PALETTE)],
                        ls="-" if i == 0 else "--", lw=1.7, label=lab)
        for ax, lab in ((axes[0], f"minADE@{args.k}"), (axes[1], f"minFDE@{args.k}")):
            if ref is not None:
                ax.axhline(0, color=MUTED, lw=0.9)
            ax.axvline(args.mark, color="#e87ba4", lw=0.9, ls=":")
            ax.set_xlabel("horizon  (s)")
            ax.set_ylabel(f"{lab}  mean  (m)")
            ax.set_title(f"{lab}{title}")
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{args.manifest}, n={len(ids)}, all {n_tok} waypoints "
                     f"(dotted: {args.mark} s)", fontsize=10, color=MUTED)
        fig.tight_layout()
        fig.savefig(plots / fname, dpi=150)
        plt.close(fig)
    print("wrote", plots / "horizon_curves_full.png", "and horizon_curves_delta.png")

    ref = arms[0][0]
    (args.out / "metrics.json").write_text(json.dumps(
        {"n_clips": len(ids), "k": args.k, "manifest": args.manifest,
         "arms": {lab: {"ade": cur[lab][0].tolist(), "fde": cur[lab][1].tolist()}
                  for lab, _ in arms}}, indent=2))

    print(f"\n{'t (s)':>6s} " + " ".join(f"{lab[:13]:>13s}" for lab, _ in arms[1:]))
    for k in (0, 4, 9, 18, 31, 47, n_tok - 1):
        print(f"{t[k]:6.1f} " +
              " ".join(f"{cur[lab][0][k] - cur[ref][0][k]:+13.5f}" for lab, _ in arms[1:]))


if __name__ == "__main__":
    main()
