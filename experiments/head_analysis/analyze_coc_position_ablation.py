"""Gates D1-D3 of plans/2026-09-25_coc-position-content.md: double dissociation at generated-CoC
positions, from run_coc_position_ablation.py shards. Every config switches a head set off at the
CoC positions only; d = config - dense per clip (same text, noise and seeds).

  D1  language: dNLL(C_all) > dNLL(T_all) and > every R_all (paired one-sided Wilcoxon, p < 0.01)
  D2  action:   dFM(T_trunk) > dFM(C_trunk) and > every R_trunk (p < 0.01); all-layer sets reported
  D3  reported: dminADE in the same directions (sampled configs)

Usage:
  python experiments/head_analysis/analyze_coc_position_ablation.py --shards coc_posabl_v1_s0 ... --out coc_posabl_v1
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
rng = np.random.default_rng(0)


def ci(x, n=3000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def wil(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    return float(wilcoxon(d, alternative="greater")[1]) if len(d) > 5 and (d != 0).any() else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = REPO / "outputs" / args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)
    clips, buckets, per = [], [], {}
    for s in args.shards:
        m = json.loads((REPO / "outputs" / s / "metrics.json").read_text())
        clips += m["clip_ids"]
        buckets += m["buckets"]
        for cfg, v in m["per_clip"].items():
            for k, arr in v.items():
                per.setdefault(cfg, {}).setdefault(k, []).extend(arr)
    buckets = np.array(buckets)
    dense = {k: np.array(v, float) for k, v in per["dense"].items()}
    d = {c: {k: np.array(per[c][k], float) - dense[k] for k in ("nll", "fm", "ade")} for c in per if c != "dense"}
    base = {k: float(np.nanmean(dense[k])) for k in ("nll", "fm", "ade")}
    lines = [f"CoC-position double dissociation -- {len(clips)} clips, shards {', '.join(args.shards)}",
             f"dense: NLL {base['nll']:.4f}  FM {base['fm']:.4f}  minADE {base['ade']:.3f}", "",
             f"{'config':10s} {'dNLL mean [CI]':>28s} {'rel':>7s} {'dFM mean [CI]':>28s} {'rel':>7s} {'dminADE mean [CI]':>26s}   rel dFM by manoeuvre (cruise/accel/decel_stop/turn)"]
    res = {}
    for c in ("T_all", "C_all", "R0_all", "R1_all", "R2_all", "T_trunk", "C_trunk", "R0_trunk", "R1_trunk", "R2_trunk"):
        if c not in d:
            continue
        n_, f, a = ci(d[c]["nll"]), ci(d[c]["fm"]), (ci(d[c]["ade"]) if np.isfinite(d[c]["ade"]).any() else None)
        byb = " ".join(f"{np.nanmean(d[c]['fm'][buckets == b]) / np.nanmean(dense['fm'][buckets == b]):+.1%}" for b in ("cruise", "accel", "decel_stop", "turn"))
        med = {k: float(np.nanmedian(d[c][k])) for k in ("nll", "fm", "ade")}
        res[c] = {"nll": n_, "fm": f, "ade": a, "median": med}
        lines.append(f"{c:10s} {n_[0]:+.4f} [{n_[1]:+.4f},{n_[2]:+.4f}] {n_[0] / base['nll']:+6.1%} {f[0]:+.4f} [{f[1]:+.4f},{f[2]:+.4f}] {f[0] / base['fm']:+6.1%} "
                     + (f"{a[0]:+.3f} [{a[1]:+.3f},{a[2]:+.3f}]" if a else f"{'-':>26s}") + f"   {byb}"
                     + f"   | medians dNLL {med['nll']:+.4f} dFM {med['fm']:+.4f}" + (f" dminADE {med['ade']:+.3f}" if a else ""))
    gates = {}
    for band in ("all", "trunk"):
        T, C = f"T_{band}", f"C_{band}"
        R = [f"R{i}_{band}" for i in range(3) if f"R{i}_{band}" in d]
        if T not in d or C not in d:
            continue
        p1 = wil(d[C]["nll"], d[T]["nll"]); p1r = max(wil(d[C]["nll"], d[r]["nll"]) for r in R)
        p2 = wil(d[T]["fm"], d[C]["fm"]); p2r = max(wil(d[T]["fm"], d[r]["fm"]) for r in R)
        p3 = wil(d[T]["ade"], d[C]["ade"]) if np.isfinite(d[T]["ade"]).any() and np.isfinite(d[C]["ade"]).any() else 1.0
        p3c = wil(d[C]["ade"], d[T]["ade"]) if np.isfinite(d[T]["ade"]).any() and np.isfinite(d[C]["ade"]).any() else 1.0
        gates[band] = {"D1_nll_C_gt_T_p": p1, "D1_nll_C_gt_R_maxp": p1r, "D1": p1 < 0.01 and p1r < 0.01,
                       "D2_fm_T_gt_C_p": p2, "D2_fm_T_gt_R_maxp": p2r, "D2": p2 < 0.01 and p2r < 0.01,
                       "D3_ade_T_gt_C_p": p3, "D3_ade_C_gt_T_p": p3c}
        lines.append(f"\n[{band}] D1 language (C > T, C > random): p={p1:.2g}, max p vs random {p1r:.2g} -> {'PASS' if gates[band]['D1'] else 'FAIL'} | "
                     f"D2 action (T > C, T > random): p={p2:.2g}, max p vs random {p2r:.2g} -> {'PASS' if gates[band]['D2'] else 'FAIL'} | "
                     f"D3 minADE: T > C p={p3:.2g}, C > T p={p3c:.2g}")
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
    names = [c for c in ("T_all", "C_all", "R0_all", "R1_all", "R2_all", "T_trunk", "C_trunk", "R0_trunk", "R1_trunk", "R2_trunk") if c in res]
    cols = ["#2a78d6" if c.startswith("T") else "#e87ba4" if c.startswith("C") else "#bbbbbb" for c in names]
    for ax, key, lab in zip(axes, ("nll", "fm"), ("dNLL / dense NLL", "dFM / dense FM")):
        vals = [res[c][key] for c in names]
        ax.bar(range(len(names)), [v[0] / base[key] for v in vals], color=cols,
               yerr=[[(v[0] - v[1]) / base[key] for v in vals], [(v[2] - v[0]) / base[key] for v in vals]], capsize=2)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=60, fontsize=7); ax.axhline(0, color="black", lw=0.6); ax.set_ylabel(lab)
    fig.tight_layout(); fig.savefig(out / "plots" / "dissociation.png", dpi=150); plt.close(fig)
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    (out / "metrics.json").write_text(json.dumps({"gates": gates, "results": res, "n_clips": len(clips)}, indent=1, default=float))
    (out / "config.json").write_text(json.dumps({"shards": args.shards, "plan": "plans/2026-09-25_coc-position-content.md"}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
