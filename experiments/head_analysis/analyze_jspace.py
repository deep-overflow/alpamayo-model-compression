"""Gates G1/G2 and the Stage-C retro-diagnostic for the J-lens run.

G1  layer bands: excess kurtosis of the lens readout should show a middle-layer
    plateau (the paper's workspace band at 38-92% of depth), and the CKA between
    per-layer dictionaries should be block structured.

G2  is the J-space unit score new information, and is it aimed at reasoning?
      (a) novelty  : median |rho(j, magnitude)| < 0.9   -- else it is magnitude
      (b) direction: rho(j, coc) > rho(j, traj)         -- hypothesis H's prediction
    Both are pre-registered in the plan; (b) failing falsifies H.

Stage C  how much J-mass did each shipped slim config actually delete, relative
    to how much total write mass it deleted? Mass is summed in squares because
    the scores are norms, so squares are the additive energy.

    CONFOUND: cocsafe and integrated_mag differ in criterion AND in layerwise
    allocation, so the per-config comparison here is descriptive only. It is not
    evidence for H until the 2x3 grid (criterion x allocation) is filled.

Usage:
  python analyze_jspace.py --jlens-exp jlens_v1 --importance-exp importance_v1
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

BG = "#FAF9F5"
INK = "#29261B"
MUTED = "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})

SLIM_CONFIGS = ["slim_cocsafe_r20", "slim_cocsafe_r30", "slim_integrated_mag"]


def per_layer_rho(a, b):
    """Spearman per layer between two (L, n_units) score arrays."""
    out = []
    for li in range(a.shape[0]):
        x, y = a[li], b[li]
        if np.std(x) == 0 or np.std(y) == 0:
            out.append(np.nan)
        else:
            out.append(float(spearmanr(x, y).statistic))
    return np.array(out)


def gate2(jl, imp, unit):
    """Correlation panel for one unit family ('q' or 'mlp')."""
    j = jl[f"{unit}_j"]
    w = jl[f"{unit}_w"]
    jfrac = j / np.clip(w, 1e-12, None)
    mag = jl[f"mag_{unit}"]
    coc = imp[f"coc_vlm_{unit}"]
    traj = imp[f"traj_vlm_{unit}"]
    res = {}
    for name, score in (("j", j), ("jfrac", jfrac)):
        res[name] = {
            "vs_mag": per_layer_rho(score, mag),
            "vs_coc": per_layer_rho(score, coc),
            "vs_traj": per_layer_rho(score, traj),
        }
    res["sanity_w_vs_mag"] = per_layer_rho(w, mag)
    res["coc_vs_traj"] = per_layer_rho(coc, traj)
    return res


def stage_c(jl, cfg_name, n_heads, n_mlp):
    """Fraction of J-mass vs total write mass removed by a shipped slim config."""
    meta = json.loads((REPO / "outputs" / cfg_name / "slim_meta.json").read_text())
    kvonly = set(meta.get("kvonly_layers", []))
    n_layers = len(meta["vlm"])
    acc = {}
    for unit, n_units, key in (("q", n_heads, "q"), ("mlp", n_mlp, "mlp")):
        j2 = jl[f"{unit}_j"] ** 2
        w2 = jl[f"{unit}_w"] ** 2
        rm_j = np.zeros(n_layers)
        rm_w = np.zeros(n_layers)
        for li in range(n_layers):
            kept = set(meta["vlm"][li].get(key, []))
            if li in kvonly:
                kept = set()
            removed = np.array([u for u in range(n_units) if u not in kept], dtype=int)
            if len(removed):
                rm_j[li] = j2[li, removed].sum()
                rm_w[li] = w2[li, removed].sum()
        # Per layer, then aggregate. A global mass fraction is unreadable here:
        # massive activations put >95% of the squared write mass in a single layer,
        # so the pooled number is that one layer's answer wearing a disguise.
        pl_j = rm_j / np.clip(j2.sum(1), 1e-30, None)
        pl_w = rm_w / np.clip(w2.sum(1), 1e-30, None)
        ok = pl_w > 1e-6  # layers this config actually prunes
        ratio = pl_j[ok] / pl_w[ok]
        acc[unit] = {
            "j_frac": float(np.mean(pl_j[ok])) if ok.any() else float("nan"),
            "w_frac": float(np.mean(pl_w[ok])) if ok.any() else float("nan"),
            "ratio": float(np.median(ratio)) if ok.any() else float("nan"),
            "ratio_iqr": [float(np.percentile(ratio, 25)),
                          float(np.percentile(ratio, 75))] if ok.any() else [],
            "n_layers_pruned": int(ok.sum()),
            "per_layer_j": pl_j.tolist(), "per_layer_w": pl_w.tolist(),
        }
    return acc


def plot_corr(res, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, unit, title in zip(axes, ("q", "mlp"), ("Q heads", "MLP channels")):
        r = res[unit]["j"]
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.plot(r["vs_mag"], color=C4, label="vs magnitude")
        ax.plot(r["vs_coc"], color=C1, label="vs CoC Taylor")
        ax.plot(r["vs_traj"], color=C3, label="vs traj Taylor")
        ax.set_title(f"{title}: Spearman of J-score")
        ax.set_xlabel("layer")
        ax.set_ylabel("rho")
        ax.set_ylim(-1.05, 1.05)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "corr_layers.png", dpi=150)
    plt.close(fig)


def plot_bands(kurt, cka, out_dir):
    n = len(kurt)
    lo, hi = int(0.38 * n), int(0.92 * n)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].axvspan(lo, hi, color=C2, alpha=0.12, label="paper workspace band")
    axes[0].plot(kurt, color=C1)
    axes[0].set_title("G1: excess kurtosis of the lens readout")
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("excess kurtosis")
    axes[0].legend(frameon=False, fontsize=8)
    im = axes[1].imshow(cka, cmap="Oranges", interpolation="nearest")
    axes[1].set_title("G1: CKA between per-layer J-lens dictionaries")
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("layer")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "layer_bands.png", dpi=150)
    plt.close(fig)


def plot_stage_c(sc, out_dir):
    cfgs = list(sc.keys())
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(cfgs))
    for ax, unit, title in zip(axes, ("q", "mlp"), ("Q heads", "MLP channels")):
        w = [sc[c][unit]["w_frac"] for c in cfgs]
        j = [sc[c][unit]["j_frac"] for c in cfgs]
        ax.bar(x - 0.2, w, 0.4, color=MUTED, label="total write mass removed")
        ax.bar(x + 0.2, j, 0.4, color=C1, label="J-space mass removed")
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("slim_", "") for c in cfgs], rotation=12, fontsize=8)
        ax.set_title(f"Stage C ({title}) -- descriptive only, see confound")
        ax.set_ylabel("fraction of squared mass")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "stagec_mass.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jlens-exp", default="jlens_v1")
    ap.add_argument("--importance-exp", default="importance_v1",
                    help="the run the shipped masks were built from")
    ap.add_argument("--exp-id", default="jspace_v1")
    args = ap.parse_args()

    jl = dict(np.load(REPO / "outputs" / args.jlens_exp / "jlens.npz"))
    imp = dict(np.load(REPO / "outputs" / args.importance_exp / "importance.npz"))
    out_dir = REPO / "outputs" / args.exp_id
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    n_heads, n_mlp = jl["q_j"].shape[1], jl["mlp_j"].shape[1]
    res = {u: gate2(jl, imp, u) for u in ("q", "mlp")}
    sc = {}
    for cfg in SLIM_CONFIGS:
        if (REPO / "outputs" / cfg / "slim_meta.json").exists():
            sc[cfg] = stage_c(jl, cfg, n_heads, n_mlp)

    plot_corr(res, out_dir)
    plot_bands(jl["kurtosis"], jl["cka"], out_dir)
    if sc:
        plot_stage_c(sc, out_dir)

    lines = ["J-space gates", "=" * 60, ""]
    verdicts = {}
    for unit, title in (("q", "Q heads"), ("mlp", "MLP channels")):
        r = res[unit]
        med = {k: float(np.nanmedian(v)) for k, v in r["j"].items()}
        medf = {k: float(np.nanmedian(v)) for k, v in r["jfrac"].items()}
        novel = abs(med["vs_mag"]) < 0.9
        direction = med["vs_coc"] > med["vs_traj"]
        verdicts[unit] = {"novelty": bool(novel), "direction": bool(direction),
                          "median_j": med, "median_jfrac": medf}
        lines += [
            f"{title}",
            f"  j     : vs mag {med['vs_mag']:+.3f}   vs coc {med['vs_coc']:+.3f}   "
            f"vs traj {med['vs_traj']:+.3f}",
            f"  jfrac : vs mag {medf['vs_mag']:+.3f}   vs coc {medf['vs_coc']:+.3f}   "
            f"vs traj {medf['vs_traj']:+.3f}",
            f"  G2(a) novelty   |rho(j,mag)| < 0.9 : {'PASS' if novel else 'FAIL'}",
            f"  G2(b) direction rho(coc) > rho(traj): {'PASS' if direction else 'FAIL'} "
            f"(margin {med['vs_coc'] - med['vs_traj']:+.3f})",
            f"  sanity  rho(write_norm, magnitude) = {np.nanmedian(r['sanity_w_vs_mag']):+.3f} "
            "(should be high; low means an activation-stats bug)",
            f"  context rho(coc, traj) = {np.nanmedian(r['coc_vs_traj']):+.3f}",
            "",
        ]

    kurt = jl["kurtosis"]
    n = len(kurt)
    lo, hi = int(0.38 * n), int(0.92 * n)
    lines += [
        "G1 layer bands",
        f"  predicted workspace band {lo}-{hi} of {n} layers",
        f"  mean excess kurtosis  sensory={kurt[:lo].mean():.2f}  "
        f"workspace={kurt[lo:hi].mean():.2f}  motor={kurt[hi:].mean():.2f}",
        f"  plateau present: {kurt[lo:hi].mean() > max(kurt[:lo].mean(), kurt[hi:].mean())}",
        "",
    ]
    if sc:
        lines += ["Stage C -- per-layer, DESCRIPTIVE ONLY (criterion x allocation confound)",
                  "  ratio > 1 = removed J-mass out of proportion to total write mass"]
        for cfg, a in sc.items():
            lines.append(f"  {cfg}")
            for unit in ("q", "mlp"):
                d = a[unit]
                iqr = d["ratio_iqr"]
                lines.append(f"    {unit:<4} write {d['w_frac']:.3f}  J {d['j_frac']:.3f}  "
                             f"ratio {d['ratio']:.3f} [{iqr[0]:.2f},{iqr[1]:.2f}] "
                             f"over {d['n_layers_pruned']} layers")
        lines.append("")

    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    (out_dir / "config.json").write_text(json.dumps({
        "purpose": "G1/G2 gates + Stage C retro-diagnostic for the J-lens run",
        "jlens_exp": args.jlens_exp, "importance_exp": args.importance_exp,
        "slim_configs": list(sc.keys()),
        "pre_registered": {"G2a": "median |rho(j, magnitude)| < 0.9",
                           "G2b": "median rho(j, coc) > median rho(j, traj)"},
    }, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps({
        "verdicts": verdicts,
        "per_layer": {u: {k: {kk: vv.tolist() for kk, vv in v.items()}
                          if isinstance(v, dict) else v.tolist()
                          for k, v in res[u].items()} for u in res},
        "stage_c": sc,
        "kurtosis": kurt.tolist(),
    }, indent=2))
    print("\n".join(lines), flush=True)
    print("saved ->", out_dir, flush=True)


if __name__ == "__main__":
    main()
