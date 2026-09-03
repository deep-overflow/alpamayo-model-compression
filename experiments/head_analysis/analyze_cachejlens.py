"""G0 of plans/2026-08-30_cache-jlens-criterion.md: is the cache-Jacobian score a usable,
DIFFERENT signal?

  stability   split-half (even/odd clips) Spearman within layer and kept-set overlap at the
              u40 uniform ratio -- pass >= 0.95 (the jlens_coc32 bar)
  novelty     kept-set overlap of cacheonly / cachedual against traj / dual / coc at the
              same ratio -- the experiment is moot if cachedual overlaps dual >= 0.95
  profile     where the cache-moving units sit (layer mass), Spearman of I_cache with
              I_traj per layer

Usage:
  .venv/bin/python experiments/head_analysis/analyze_cachejlens.py --run cachejlens_v1 \
      --importance importance_v2 --out cachejlens_v1
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from run_cocsafe import rank_norm  # noqa: E402

BG, INK, MUTED, GRID = "#FAF9F5", "#29261B", "#6B6555", "#E8E6DC"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.size": 9, "axes.grid": True, "axes.axisbelow": True,
})
REPO = Path(__file__).resolve().parents[2]
RATIO = 0.3985632694  # the u40_v2 family's matched uniform ratio (make_slim / run_grid)


def kept(scores, ratio=RATIO):
    """Per-layer kept mask at the uniform ratio: mask_lib.select_mask_ratios' rule."""
    n = scores.shape[1]
    k = round(ratio * n)
    m = np.ones_like(scores, dtype=bool)
    for li in range(scores.shape[0]):
        m[li, np.argsort(scores[li])[:k]] = False
    return m


def overlap(a, b):
    return float((a & b).sum() / a.sum())


def layer_spearman(a, b):
    """Within-layer Spearman, averaged over layers that are not constant (the last
    layer's cache score is identically zero -- its gates reach no cache)."""
    vals = [spearmanr(a[li], b[li])[0] for li in range(a.shape[0])
            if a[li].std() > 0 and b[li].std() > 0]
    return float(np.mean(vals))


def load_runs(runs):
    """Probe-weighted mean of several run_cache_jlens outputs (more probes, same clips)."""
    zs = [dict(np.load(REPO / "outputs" / r / "importance.npz")) for r in runs]
    n = np.array([float(z["n_probes"]) for z in zs])
    out = {}
    for key in zs[0]:
        if key in ("sensitivity",):
            out[key] = zs[0][key]
        elif key == "n_probes":
            out[key] = n.sum()
        else:
            out[key] = sum(z[key] * w for z, w in zip(zs, n)) / n.sum()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", default=["cachejlens_v1"],
                    help="one or more run_cache_jlens exp-ids (merged, probe-weighted)")
    ap.add_argument("--importance", default="importance_v2")
    ap.add_argument("--out", default="cachejlens_v1")
    args = ap.parse_args()
    out = REPO / "outputs" / args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)
    z = load_runs(args.run)
    if len(args.run) > 1:  # the merged criterion is what make_slim reads
        np.savez(out / "importance.npz", **z)
    imp = dict(np.load(REPO / "outputs" / args.importance / "importance.npz"))
    res = {"n_probes": int(z["n_probes"]), "runs": args.run}
    lines = [f"cache-Jacobian criterion G0 -- {args.run} ({int(z['n_probes'])} probes)", ""]

    for kind, key, ref_q in (("q", "cache_vlm_q", "vlm_q"), ("mlp", "cache_vlm_mlp", "vlm_mlp")):
        c, ce, co = z[key], z[f"{key}_even"], z[f"{key}_odd"]
        traj, coc = imp[f"traj_{ref_q}"], imp[f"coc_{ref_q}"]
        dual = np.maximum(rank_norm(traj), rank_norm(coc))
        cachedual = np.maximum(rank_norm(c), rank_norm(coc))
        r = {
            "splithalf_spearman": layer_spearman(ce, co),
            "splithalf_overlap": overlap(kept(ce), kept(co)),
            "spearman_vs_traj": layer_spearman(c, traj),
            "spearman_vs_coc": layer_spearman(c, coc),
            "overlap_cacheonly_vs_traj": overlap(kept(c), kept(traj)),
            "overlap_cacheonly_vs_dual": overlap(kept(c), kept(dual)),
            "overlap_cacheonly_vs_coc": overlap(kept(c), kept(coc)),
            "overlap_cachedual_vs_dual": overlap(kept(cachedual), kept(dual)),
            "overlap_cachedual_vs_traj": overlap(kept(cachedual), kept(traj)),
            "layer_mass": (c.sum(1) / c.sum()).tolist(),
            "traj_layer_mass": (traj.sum(1) / traj.sum()).tolist(),
            "cv_median": float(np.median(np.sqrt(np.maximum(z[f"{key}_var"], 0))
                                         / np.maximum(c, 1e-30))),
        }
        r["stability_pass"] = r["splithalf_spearman"] >= 0.95 and r["splithalf_overlap"] >= 0.95
        r["novel_pass"] = r["overlap_cachedual_vs_dual"] < 0.95
        res[kind] = r
        lines += [
            (f"[{kind}] split-half Spearman {r['splithalf_spearman']:.3f}, kept overlap "
             f"{r['splithalf_overlap']:.3f} -> stability "
             f"{'PASS' if r['stability_pass'] else 'FAIL'}"),
            f"      Spearman vs traj {r['spearman_vs_traj']:+.3f}, vs coc {r['spearman_vs_coc']:+.3f}",
            (f"      kept overlap cacheonly vs traj {r['overlap_cacheonly_vs_traj']:.3f}, vs dual "
             f"{r['overlap_cacheonly_vs_dual']:.3f}, vs coc {r['overlap_cacheonly_vs_coc']:.3f}"),
            (f"      kept overlap cachedual vs dual {r['overlap_cachedual_vs_dual']:.3f}, vs traj "
             f"{r['overlap_cachedual_vs_traj']:.3f} -> novelty "
             f"{'PASS' if r['novel_pass'] else 'FAIL'}"),
            f"      per-probe CV median {r['cv_median']:.2f}",
        ]
    res["G0_pass"] = all(res[k]["stability_pass"] and res[k]["novel_pass"] for k in ("q", "mlp"))
    lines.append(f"G0 {'PASS' if res['G0_pass'] else 'FAIL'}")
    text = "\n".join(lines)
    print(text)
    (out / "cachejlens_summary.txt").write_text(text + "\n")
    (out / "metrics_analysis.json").write_text(json.dumps(res, indent=1))

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, kind in zip(axes, ("q", "mlp")):
        ax.plot(res[kind]["layer_mass"], "o-", ms=3, color=C1, label="I_cache")
        ax.plot(res[kind]["traj_layer_mass"], "s--", ms=3, color=C3, label="I_traj")
        ax.set_xlabel("VLM layer")
        ax.set_ylabel("share of importance mass")
        ax.set_title(f"{kind}: where the cache-moving units sit")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "plots" / "cachejlens_layer_mass.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
