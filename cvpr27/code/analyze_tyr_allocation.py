"""Does tyr_off's per-layer allocation follow the first-order Taylor importance?

tyr_off (cvpr27/reports/2026-11-01_tyr-official-a100.md) chose how much to cut per layer by an
evolutionary search on text KL, and which units by OSSCAR on calib_100 Hessians -- no Taylor score
anywhere. This compares its choices with importance_v2 (raw first-order Taylor, traj = flow matching,
coc = CoC cross-entropy, calib_100), on three levels:

  between layers   Spearman rho across layers of tyr's cut fraction against the layer's Taylor mass
                   and its concentration (Gini; fewest units holding 90% of the mass)
  allocation       a Taylor allocation with tyr's budget: keep the globally top-ranked units by raw score
                   (raw is the comparable scale across layers, reports/evaluation/2026-08-30) with the
                   same totals as tyr (684 heads, 266,040 channels); compare per-layer kept counts
  within layers    overlap of tyr's kept units with the layer's Taylor top-k (k = tyr's kept count),
                   against chance k/n; Taylor mass tyr keeps vs MoOP (dual_u40_v2) and vs the top-k optimum

Layer 35 has identically zero traj scores (structural) and is left out of traj statistics.

  .venv/bin/python cvpr27/code/analyze_tyr_allocation.py
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG, INK, MUTED, GRID = "#FAF9F5", "#29261B", "#6B6555", "#E8E6DC"
C_TYR, C_TRAJ, C_COC, C_DUAL, C_MOOP = "#d0632c", "#2a78d6", "#eda100", "#008300", "#6B6555"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.size": 9, "axes.grid": True, "axes.axisbelow": True,
})
REPO = Path(__file__).resolve().parents[2]
N_LAYERS, N_HEADS, N_CH = 36, 32, 12288


def rankdata(x):
    return np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)


def spearman(a, b):
    a, b = rankdata(np.asarray(a)), rankdata(np.asarray(b))
    return float(np.corrcoef(a, b)[0, 1])


def gini(x):
    s = np.sort(x)
    n = len(s)
    return float((2 * np.arange(1, n + 1) - n - 1) @ s / (n * s.sum())) if s.sum() > 0 else float("nan")


def n90_frac(x):
    s = np.sort(x)[::-1]
    c = np.cumsum(s) / s.sum()
    return float((np.searchsorted(c, 0.9) + 1) / len(x))


def kept_masks(meta_path):
    m = json.loads(Path(meta_path).read_text())
    q = np.zeros((N_LAYERS, N_HEADS), bool)
    ml = np.zeros((N_LAYERS, N_CH), bool)
    for i, layer in enumerate(m["vlm"]):
        q[i, layer["q"]] = True
        ml[i, layer["mlp"]] = True
    return q, ml


def global_alloc(score, n_keep, valid_layers):
    """Per-layer kept counts when the n_keep globally highest raw scores are kept."""
    s = score.copy()
    s[~valid_layers] = np.inf          # layers without a score are kept whole, off the budget
    n_fixed = int((~valid_layers).sum() * s.shape[1])
    flat = np.argsort(s, axis=None)[::-1][: n_keep + n_fixed]
    keep = np.zeros(s.size, bool)
    keep[flat] = True
    return keep.reshape(s.shape).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--importance", default="importance_v2")
    ap.add_argument("--tyr", default="cvpr27/slim_tyr_off")
    ap.add_argument("--moop", default="slim_dual_u40_v2")
    ap.add_argument("--exp-id", default="cvpr27/tyr_off_allocation")
    args = ap.parse_args()
    out = REPO / "outputs" / args.exp_id
    out.mkdir(parents=True, exist_ok=True)

    z = np.load(REPO / "outputs" / args.importance / "importance.npz")
    imp = {("traj", "q"): z["traj_vlm_q"], ("coc", "q"): z["coc_vlm_q"],
           ("traj", "mlp"): z["traj_vlm_mlp"], ("coc", "mlp"): z["coc_vlm_mlp"]}
    tq, tm = kept_masks(REPO / "outputs" / args.tyr / "slim_meta.json")
    mq, mm = kept_masks(REPO / "outputs" / args.moop / "slim_meta.json")
    kept = {"q": tq, "mlp": tm}
    moop = {"q": mq, "mlp": mm}
    n_units = {"q": N_HEADS, "mlp": N_CH}

    res = {"between_layers": {}, "allocation": {}, "within_layers": {}, "mass_kept": {}, "per_layer": {}}
    for axis in ("q", "mlp"):
        k_tyr = kept[axis].sum(1)
        cut = 1 - k_tyr / n_units[axis]
        res["per_layer"][f"tyr_kept_{axis}"] = k_tyr.tolist()
        for obj in ("traj", "coc"):
            sc = imp[(obj, axis)]
            valid = sc.sum(1) > 0
            L = np.flatnonzero(valid)
            mass = sc.sum(1)
            g = np.array([gini(sc[i]) if valid[i] else np.nan for i in range(N_LAYERS)])
            n90 = np.array([n90_frac(sc[i]) if valid[i] else np.nan for i in range(N_LAYERS)])
            res["per_layer"][f"{obj}_{axis}_mass"] = mass.tolist()
            res["per_layer"][f"{obj}_{axis}_gini"] = g.tolist()
            res["between_layers"][f"{obj}/{axis}"] = {
                "rho_cut_vs_mass": spearman(cut[L], mass[L]),
                "rho_cut_vs_gini": spearman(cut[L], g[L]),
                "rho_cut_vs_n90": spearman(cut[L], n90[L]),
                "n_layers": len(L),
            }
            # Taylor allocation with tyr's totals
            alloc = global_alloc(sc, int(k_tyr[L].sum()), valid)
            res["per_layer"][f"taylor_{obj}_kept_{axis}"] = alloc.tolist()
            res["allocation"][f"{obj}/{axis}"] = {
                "rho_kept_tyr_vs_taylor": spearman(k_tyr[L], alloc[L]),
                "mean_abs_diff_units": float(np.abs(k_tyr[L] - alloc[L]).mean()),
            }
            # within layers: overlap with the layer's Taylor top-k
            ov, ch = [], []
            for i in L:
                k = int(k_tyr[i])
                top = np.argsort(sc[i])[::-1][:k]
                ov.append(kept[axis][i, top].mean() if k else np.nan)
                ch.append(k / n_units[axis])
            res["within_layers"][f"{obj}/{axis}"] = {
                "mean_overlap_with_taylor_topk": float(np.nanmean(ov)),
                "mean_chance": float(np.mean(ch)),
            }
            # Taylor mass kept (raw, layers with scores)
            tot = sc[L].sum()
            res["mass_kept"][f"{obj}/{axis}"] = {
                "tyr_off": float(sc[L][kept[axis][L]].sum() / tot),
                "moop": float(sc[L][moop[axis][L]].sum() / tot),
                "taylor_topk_same_counts_as_tyr": float(sum(np.sort(sc[i])[::-1][: int(k_tyr[i])].sum()
                                                           for i in L) / tot),
                "taylor_global_same_total": float(sum(np.sort(sc[i])[::-1][: int(alloc[i])].sum()
                                                      for i in L) / tot),
            }
        # agreement with MoOP's kept units, per layer, on the units both could keep
        agree = [(kept[axis][i] & moop[axis][i]).sum() / max(kept[axis][i].sum(), 1) for i in range(N_LAYERS)]
        res["within_layers"][f"tyr_in_moop/{axis}"] = {
            "mean_share_of_tyr_kept_also_kept_by_moop": float(np.mean(agree)),
            "moop_keep_share": float(moop[axis].sum() / moop[axis].size),
        }

    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    (out / "metrics.json").write_text(json.dumps(res, indent=2))

    lines = ["tyr_off allocation vs first-order Taylor importance (importance_v2, raw)", ""]
    lines.append("between layers: Spearman rho of tyr cut fraction vs layer Taylor stats")
    for k, v in res["between_layers"].items():
        lines.append(f"  {k:9s} mass {v['rho_cut_vs_mass']:+.2f}  gini {v['rho_cut_vs_gini']:+.2f}  "
                     f"n90 {v['rho_cut_vs_n90']:+.2f}  (n={v['n_layers']})")
    lines.append("allocation: tyr kept counts vs a global raw-Taylor top-k with the same totals")
    for k, v in res["allocation"].items():
        lines.append(f"  {k:9s} rho {v['rho_kept_tyr_vs_taylor']:+.2f}  mean |diff| {v['mean_abs_diff_units']:.1f} units")
    lines.append("within layers: share of the layer's Taylor top-k that tyr kept (chance = k/n)")
    for k, v in res["within_layers"].items():
        if "overlap" in str(v):
            lines.append(f"  {k:9s} {v['mean_overlap_with_taylor_topk']:.3f} (chance {v['mean_chance']:.3f})")
        else:
            lines.append(f"  {k:9s} {v['mean_share_of_tyr_kept_also_kept_by_moop']:.3f} of tyr's kept units are "
                         f"MoOP-kept (MoOP keeps {v['moop_keep_share']:.3f})")
    lines.append("Taylor mass kept (share of raw mass):")
    for k, v in res["mass_kept"].items():
        lines.append(f"  {k:9s} tyr {v['tyr_off']:.3f}  MoOP {v['moop']:.3f}  top-k at tyr counts "
                     f"{v['taylor_topk_same_counts_as_tyr']:.3f}  global top-k {v['taylor_global_same_total']:.3f}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    x = np.arange(N_LAYERS)
    for ax, axis, title in ((axes[0], "q", "Q heads kept per layer"), (axes[1], "mlp", "MLP channels kept per layer")):
        pl = res["per_layer"]
        ax.bar(x, pl[f"tyr_kept_{axis}"], color=C_TYR, width=0.7, label="tyr_off")
        ax.plot(x, pl[f"taylor_traj_kept_{axis}"], "o-", color=C_TRAJ, ms=3, lw=1.2, label="Taylor traj, same total")
        ax.plot(x, pl[f"taylor_coc_kept_{axis}"], "s-", color=C_COC, ms=3, lw=1.2, label="Taylor CoC, same total")
        ax.plot(x, moop[axis].sum(1), "--", color=C_MOOP, lw=1.2, label="MoOP (uniform)")
        ax.set_title(title, loc="left")
        ax.set_ylim(0, n_units[axis] * 1.02)
    axes[0].legend(ncol=4, frameon=False, fontsize=8, loc="lower left")
    axes[1].set_xlabel("VLM layer")
    fig.tight_layout()
    fig.savefig(out / "allocation_vs_taylor.png", dpi=150)
    print("->", out)


if __name__ == "__main__":
    main()
