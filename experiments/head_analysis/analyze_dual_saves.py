"""Gates of plans/2026-09-24_dual-saves-trunk.md: do the trunk units that the dual criterion keeps
but a single criterion drops carry the OTHER objective's function?

Reads the shards of run_token_ablation.py for the per-axis sets (--per-axis) and, optionally, for
the joint both-axis masks (--joint, run with --arm-masks). Every config is paired per clip with
the dense model on the same clip, text and noise: dFM (GT-anchored flow-matching loss), dNLL (CoC
NLL of the dense rollout) and, for the sampled configs, dminADE.

  G1  Straj (dual kept, trajectory-only dropped): dFM above the score-matched control MStraj and
      above every size-matched random set (paired one-sided Wilcoxon, p < 0.01)
  G2  Straj: dminADE mean CI excludes 0 and dminADE > MStraj (paired Wilcoxon, p < 0.05)
  G3  Straj: relative dFM larger on decel_stop + turn clips than on cruise + accel clips
      (one-sided Mann-Whitney, p < 0.05)
  G4  Scoc (dual kept, CoC-only dropped): dFM above MScoc and every random set (p < 0.01), and
      dNLL not above the mean of the random sets (p < 0.05 one-sided, i.e. no language cost)
  G5  reported, not gated: dNLL of Straj vs random (these are I_CoC-high units)

Usage:
  python experiments/head_analysis/analyze_dual_saves.py --per-axis tokabl_trunk_v1_s0 ... \
      [--joint tokabl_trunkjoint_v1_s0 ...] --out dual_saves_trunk_v1
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

REPO = Path(__file__).resolve().parents[2]
rng = np.random.default_rng(0)


def merge(shards):
    clips, buckets, per = [], [], {}
    for s in shards:
        m = json.loads((REPO / "outputs" / s / "metrics.json").read_text())
        clips += m["clip_ids"]
        buckets += m["buckets"]
        for cfg, v in m["per_clip"].items():
            for k, arr in v.items():
                per.setdefault(cfg, {}).setdefault(k, []).extend(arr)
    return clips, np.array(buckets), {c: {k: np.array(a, float) for k, a in v.items()} for c, v in per.items()}


def ci(x, n=3000):
    x = np.asarray(x, float)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def wil(a, b=None, alt="greater"):
    d = np.asarray(a) - (0 if b is None else np.asarray(b))
    d = d[np.isfinite(d)]
    if len(d) < 5 or not (d != 0).any():
        return 1.0
    return float(wilcoxon(d, alternative=alt)[1])


def analyse(clips, buckets, per, prefixes, lines, gates, tag):
    dense = per["dense"]
    d = {c: {k: per[c][k] - dense[k] for k in ("nll", "fm", "ade") if k in per[c] and np.isfinite(per[c][k]).any()}
         for c in per if c != "dense"}
    base = {k: float(np.nanmean(dense[k])) for k in ("nll", "fm", "ade")}
    lines.append(f"\n[{tag}] dense: FM {base['fm']:.4f} NLL {base['nll']:.4f} minADE {base['ade']:.3f}; n = {len(clips)} clips")
    lines.append(f"{'config':24s} {'dFM mean [CI]':>28s} {'rel':>7s} {'dNLL mean [CI]':>28s} {'rel':>7s} {'dminADE mean [CI]':>26s}   rel dFM by manoeuvre (cruise/accel/decel_stop/turn)")
    for pre in prefixes:
        for name in ("Straj", "MStraj", "RStraj0", "RStraj1", "RStraj2", "Scoc", "MScoc", "RScoc0", "RScoc1", "RScoc2"):
            c = f"{pre}_{name}"
            if c not in d:
                continue
            f, n_ = ci(d[c]["fm"]), ci(d[c]["nll"])
            a = ci(d[c]["ade"]) if "ade" in d[c] else None
            byb = " ".join(f"{np.nanmean(d[c]['fm'][buckets == b]) / np.nanmean(dense['fm'][buckets == b]):+.1%}" for b in ("cruise", "accel", "decel_stop", "turn"))
            lines.append(f"{c:24s} {f[0]:+.4f} [{f[1]:+.4f},{f[2]:+.4f}] {f[0] / base['fm']:+6.1%} {n_[0]:+.4f} [{n_[1]:+.4f},{n_[2]:+.4f}] {n_[0] / base['nll']:+6.1%} "
                         + (f"{a[0]:+.3f} [{a[1]:+.3f},{a[2]:+.3f}]" if a else f"{'-':>26s}") + f"   {byb}")
        # gates
        for side, S, M, R in (("traj", "Straj", "MStraj", ("RStraj0", "RStraj1", "RStraj2")), ("coc", "Scoc", "MScoc", ("RScoc0", "RScoc1", "RScoc2"))):
            cS, cM = f"{pre}_{S}", f"{pre}_{M}"
            if cS not in d:
                continue
            rmean_nll = np.mean([d[f"{pre}_{r}"]["nll"] for r in R], 0)
            p_fm_M = wil(d[cS]["fm"], d[cM]["fm"])
            p_fm_R = max(wil(d[cS]["fm"], d[f"{pre}_{r}"]["fm"]) for r in R)
            g = {"fm_vs_matched_p": p_fm_M, "fm_vs_random_max_p": p_fm_R}
            if side == "traj":
                a_ci = ci(d[cS]["ade"]) if "ade" in d[cS] else (np.nan,) * 3
                p_ade_M = wil(d[cS]["ade"], d[cM]["ade"], "greater") if "ade" in d[cS] and "ade" in d[cM] else 1.0
                hard = np.isin(buckets, ["decel_stop", "turn"])
                rel = d[cS]["fm"] / np.maximum(dense["fm"], 1e-6)
                p_man = float(mannwhitneyu(rel[hard], rel[~hard], alternative="greater")[1]) if hard.sum() > 3 and (~hard).sum() > 3 else 1.0
                g.update({"G1": p_fm_M < 0.01 and p_fm_R < 0.01, "ade_ci": a_ci, "ade_vs_matched_p": p_ade_M,
                          "G2": (a_ci[1] > 0) and p_ade_M < 0.05, "manoeuvre_p": p_man, "G3": p_man < 0.05,
                          "G5_nll_vs_random_p": wil(d[cS]["nll"], rmean_nll)})
                lines.append(f"  gates {pre} traj-side: G1 {'PASS' if g['G1'] else 'FAIL'} (FM vs matched p={p_fm_M:.2g}, vs random max p={p_fm_R:.2g}) | "
                             f"G2 {'PASS' if g['G2'] else 'FAIL'} (dminADE {a_ci[0]:+.3f} [{a_ci[1]:+.3f},{a_ci[2]:+.3f}], vs matched p={p_ade_M:.2g}) | "
                             f"G3 {'PASS' if g['G3'] else 'FAIL'} (decel_stop+turn > cruise+accel, p={p_man:.2g}) | G5 dNLL vs random p={g['G5_nll_vs_random_p']:.2g}")
            else:
                p_nll = wil(d[cS]["nll"], rmean_nll, "greater")
                g.update({"G4_fm": p_fm_M < 0.01 and p_fm_R < 0.01, "nll_vs_random_p": p_nll, "G4_nll": p_nll >= 0.05,
                          "G4": p_fm_M < 0.01 and p_fm_R < 0.01 and p_nll >= 0.05})
                lines.append(f"  gates {pre} coc-side: G4 {'PASS' if g['G4'] else 'FAIL'} (FM vs matched p={p_fm_M:.2g}, vs random max p={p_fm_R:.2g}; dNLL vs random one-sided p={p_nll:.2g}, pass if >= 0.05)")
            gates[f"{tag}:{pre}:{side}"] = g
    return d, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-axis", nargs="+", required=True)
    ap.add_argument("--joint", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = REPO / "outputs" / args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)
    lines, gates, results = ["dual-saves trunk ablation -- " + ", ".join(args.per_axis)], {}, {}
    clips, buckets, per = merge(args.per_axis)
    prefixes = sorted({c.rsplit("_", 1)[0] for c in per if c != "dense"})
    d, base = analyse(clips, buckets, per, prefixes, lines, gates, "per-axis")
    results["per_axis"] = {c: {k: ci(v) for k, v in dd.items()} for c, dd in d.items()}
    if args.joint:
        cj, bj, pj = merge(args.joint)
        pj = {("dense" if c == "dense" else c.replace("arm_", "")): v for c, v in pj.items()}
        prefixes_j = sorted({c.rsplit("_", 1)[0] for c in pj if c != "dense"})
        dj, _ = analyse(cj, bj, pj, prefixes_j, lines, gates, "joint")
        results["joint"] = {c: {k: ci(v) for k, v in dd.items()} for c, dd in dj.items()}
    # plot: dFM per config, grouped by prefix
    fig, axes = plt.subplots(1, len(prefixes), figsize=(3.2 * len(prefixes), 3.4), squeeze=False)
    for ax, pre in zip(axes[0], prefixes):
        names = [n for n in ("Straj", "MStraj", "RStraj0", "RStraj1", "RStraj2", "Scoc", "MScoc", "RScoc0", "RScoc1", "RScoc2") if f"{pre}_{n}" in d]
        vals = [ci(d[f"{pre}_{n}"]["fm"]) for n in names]
        cols = ["#2a78d6" if n == "Straj" else "#9ec3f0" if n == "MStraj" else "#e87ba4" if n == "Scoc" else "#f5c0d3" if n == "MScoc" else "#bbbbbb" for n in names]
        ax.bar(range(len(names)), [v[0] / base["fm"] for v in vals], color=cols,
               yerr=[[(v[0] - v[1]) / base["fm"] for v in vals], [(v[2] - v[0]) / base["fm"] for v in vals]], capsize=2)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=60, fontsize=7); ax.set_title(pre, fontsize=9)
        ax.axhline(0, color="black", lw=0.6); ax.set_ylabel("dFM / dense FM")
    fig.tight_layout(); fig.savefig(out / "plots" / "dfm_by_config.png", dpi=150); plt.close(fig)
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    (out / "metrics.json").write_text(json.dumps({"gates": gates, "results": results, "n_clips": len(clips)}, indent=1, default=float))
    (out / "config.json").write_text(json.dumps({"per_axis": args.per_axis, "joint": args.joint, "plan": "plans/2026-09-24_dual-saves-trunk.md"}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
