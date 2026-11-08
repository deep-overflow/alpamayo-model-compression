"""Paper Figure 5 (action expert: step-wise rank stability and importance concentration),
Alpamayo 1.5 vs Alpamayo 2 Super, recomputed from the per-step FM importance with the paper's definitions.

  (a, b) Spearman correlation between the rankings of step s and step s' (I_FM(l, u, s), |Taylor| per step),
         per layer, averaged over layers; Q heads (a) and MLP channels (b)
  (c)    C(r) of Eq. 5: share of a layer's step-averaged importance (Eq. 4, plain mean over S = 10 steps)
         held by its lowest floor(r U) units, averaged over layers

Independent of alpamayo2-distill/experiments/importance/paper_figs_a2.py (which rendered the same panels
in September); the A1.5 values are gated against the paper text.

Data (per-step |gate x grad|, calib_100):
  A1.5  outputs/stepimp_fm_perstep_v2/step_importance.npz   (36 layers, Q 16, MLP 8,256)
  A2    alpamayo2-distill/outputs/a2_importance_v1/step_importance.npz   (64 layers, Q 16, MLP 6,144)

Usage:
  .venv/bin/python iclr27-rebuttal/code/a2_fig5_expert.py [--exp-id iclr27-rebuttal/a2_fig5_expert]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[2]
MODELS = {"a15": {"name": "Alpamayo 1.5", "steps": REPO / "outputs" / "stepimp_fm_perstep_v2" / "step_importance.npz"},
          "a2": {"name": "Alpamayo 2 Super",
                 "steps": Path("/home/cvlab21/project/chan/alpamayo2-distill/outputs/a2_importance_v1/step_importance.npz")}}
KEYS = {"q": "q_abs_step", "mlp": "mlp_abs_step"}
R_MARK = 0.9
# G8: paper text (analysis_v2 / Fig. 5): mean step-pair rho, C(0.9)
G_PAPER = {"q": {"rho": 0.668, "c90": 0.685}, "mlp": {"rho": 0.874, "c90": 0.018}}
C_Q, C_MLP = "#0072B2", "#D55E00"


def step_matrix(a):
    """a: (S, L, U) per-step scores -> (S, S) Spearman between steps, per layer then averaged over layers"""
    r = rankdata(a, axis=-1)  # (S, L, U)
    r = (r - r.mean(-1, keepdims=True)) / r.std(-1, keepdims=True)
    m = np.einsum("slu,tlu->stl", r, r) / a.shape[-1]  # (S, S, L) Pearson of ranks = Spearman
    return m.mean(-1), m  # (S, S), (S, S, L)


def cum_curve(a):
    """a: (S, L, U) -> r grid (U+1,), C(r) on it, and per-layer C(R_MARK) (Eq. 4-5)"""
    bar = a.mean(0)  # (L, U)  Eq. 4
    s = np.sort(bar, axis=-1)
    U = s.shape[-1]
    cum = np.concatenate([np.zeros((s.shape[0], 1)), np.cumsum(s, -1)], -1) / s.sum(-1, keepdims=True)  # (L, U+1)
    k = int(np.floor(R_MARK * U))
    return np.arange(U + 1) / U, cum.mean(0), cum[:, k], k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="iclr27-rebuttal/a2_fig5_expert")
    args = ap.parse_args()
    out = REPO / "outputs" / args.exp_id
    out.mkdir(parents=True, exist_ok=True)

    res = {}
    for key, m in MODELS.items():
        z = np.load(m["steps"])
        for axis, k in KEYS.items():
            a = z[k].astype(np.float64)  # (S, L, U)
            mat, per_layer = step_matrix(a)
            S = mat.shape[0]
            off = mat[~np.eye(S, dtype=bool)]
            by_dist = [float(np.mean([mat[i, i + d] for i in range(S - d)])) for d in range(1, S)]
            grid, curve, c_layer, kk = cum_curve(a)
            res[(key, axis)] = {"shape": list(a.shape), "matrix": mat, "mean_pair": float(off.mean()),
                                "min_pair": float(off.min()), "by_distance": by_dist,
                                "layer_mean_pair": per_layer[~np.eye(S, dtype=bool)].mean(0),
                                "grid": grid, "curve": curve, "c_mark": float(curve[kk]), "k_mark": kk,
                                "c_mark_layers": c_layer}
    gates = {f"a15/{axis}/{q}": {"expected": G_PAPER[axis][q],
                                 "got": round(res[("a15", axis)]["mean_pair" if q == "rho" else "c_mark"], 4),
                                 "ok": abs(res[("a15", axis)]["mean_pair" if q == "rho" else "c_mark"]
                                           - G_PAPER[axis][q]) <= 0.0006}
             for axis in KEYS for q in ("rho", "c90")}

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
    for row, key in enumerate(MODELS):
        for col, axis in enumerate(("q", "mlp")):
            ax = axes[row, col]
            r = res[(key, axis)]
            im = ax.imshow(r["matrix"], vmin=0, vmax=1, cmap="viridis")
            ax.set_xticks([0, 3, 6, 9])
            ax.set_yticks([0, 3, 6, 9])
            ax.set_xlabel("Denoising step")
            ax.set_ylabel(f"{MODELS[key]['name']}\nDenoising step" if col == 0 else "Denoising step")
            ax.text(0.98, 0.02, f"mean {r['mean_pair']:.3f}\nmin {r['min_pair']:.3f}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8, color="white")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman's ρ")
        ax = axes[row, 2]
        ax.plot([0, 1], [0, 1], color="0.5", lw=0.8, label="uniform")
        for axis, col_, lab in (("q", C_Q, "Q head"), ("mlp", C_MLP, "MLP channel")):
            r = res[(key, axis)]
            ax.plot(r["grid"], r["curve"], color=col_, lw=1.6, label=lab)
            ax.plot(r["k_mark"] / (len(r["grid"]) - 1), r["c_mark"], "o", color=col_, ms=4)
            ax.annotate(f"{r['c_mark'] * 100:.1f}%", (r["k_mark"] / (len(r["grid"]) - 1), r["c_mark"]),
                        textcoords="offset points", xytext=(-30, 6), color=col_, fontsize=8)
        ax.axvline(R_MARK, color="0.4", lw=0.8, ls=":")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Fraction of units removed")
        ax.set_ylabel("Removed importance fraction")
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    for ax, t in zip(axes[0], ("(a) Q head rank similarity", "(b) MLP channel rank similarity",
                               "(c) Cumulative importance"), strict=True):
        ax.set_title(t, fontsize=9)
    fig.tight_layout()
    for ext_ in ("png", "pdf"):
        fig.savefig(out / f"fig5_overview.{ext_}", dpi=200)
    plt.close(fig)

    (out / "config.json").write_text(json.dumps({
        "sources": {k: str(m["steps"]) for k, m in MODELS.items()}, "keys": KEYS, "r_mark": R_MARK,
        "definitions": "paper Eq. 4 (plain step mean) and Eq. 5 (C(r), floor(rU), mean over layers)"}, indent=2))
    (out / "metrics.json").write_text(json.dumps({
        "gates": gates, "gates_ok": all(v["ok"] for v in gates.values()),
        "results": {f"{k}/{a}": {"shape_S_L_U": r["shape"], "mean_pair_rho": r["mean_pair"],
                                 "min_pair_rho": r["min_pair"], "rho_by_step_distance": r["by_distance"],
                                 "layer_mean_pair_rho": r["layer_mean_pair"].tolist(), "k_mark": r["k_mark"],
                                 "C_mark": r["c_mark"], "C_mark_per_layer": r["c_mark_layers"].tolist(),
                                 "matrix": r["matrix"].tolist(), "curve": r["curve"].tolist()}
                    for (k, a), r in res.items()}}, indent=2))

    L = ["Paper Figure 5 on Alpamayo 1.5 and Alpamayo 2 Super (expert, per-step FM importance)", "",
         f"G8 (A1.5 = paper text): {'PASS' if all(v['ok'] for v in gates.values()) else 'FAIL'}"]
    L += [f"   {k}: expected {v['expected']} got {v['got']}" for k, v in gates.items()]
    L += ["", (f"{'model':<7}{'axis':<5}{'(S,L,U)':>16}{'mean rho':>10}{'min rho':>9}{'C(0.9)':>9}"
          f"{'layers C(0.9) min-max':>24}{'k':>6}")]
    for (k, a), r in res.items():
        cl = r["c_mark_layers"]
        L.append(f"{k:<7}{a:<5}{tuple(r['shape'])!s:>16}{r['mean_pair']:>10.3f}{r['min_pair']:>9.3f}"
                 f"{r['c_mark'] * 100:>8.1f}%{cl.min() * 100:>13.1f}% - {cl.max() * 100:.1f}%{r['k_mark']:>6}")
    L += ["", "rho by step distance |s - s'| = 1 .. 9 (mean over pairs at that distance):"]
    for (k, a), r in res.items():
        L.append(f"{k:<7}{a:<5}" + " ".join(f"{v:.2f}" for v in r["by_distance"]))
    (out / "summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
