"""Gates B1-B5 of plans/2026-09-21_importance-causal-validation.md: token-targeted ablation.

Every config of run_token_ablation.py removes one unit set from the dense model; its damage is
the paired per-clip difference to the dense model on the same clip, text and noise, on two
losses: dFM (the GT-anchored flow-matching loss) and dNLL (the CoC NLL of the dense rollout).

  B1  late layers: dFM(T) > dFM(C), and dFM(T) above every random set
  B2  late layers: dNLL(C) > dNLL(T), and dNLL(C) above every random set
      (paired one-sided Wilcoxon, p < 0.01; for the token-resolved and the pooled sets, per axis)
  B3  control, layers 6-17: T and C do not dissociate -- the log-ratio of their dFM/dNLL
      differs by less than the spread of that log-ratio over the random sets
  B4  token resolution: T-tok is at least as action-specific as T-pool (dFM no smaller, dNLL
      no larger), and symmetrically for C
  B5  what the dual criterion saves: removing S-coc (units CoC-only dropped, dual kept) raises
      the FM loss more than every size-matched random set and the NLL no more than their mean;
      S-traj the other way round

Shards are merged by clip id. Usage:
  python experiments/head_analysis/analyze_token_ablation.py --shards tokabl_v1_s0 tokabl_v1_s1 \
      tokabl_v1_s2 tokabl_v1_s3 --out tokabl_v1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).parent))

from analyze_gradient_anatomy import C1, C3, MUTED, plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def merge(shards):
    per, ids, buckets = {}, [], []
    for s in shards:
        m = json.loads((REPO / "outputs" / s / "metrics.json").read_text())
        ids += m["clip_ids"]
        buckets += m["buckets"]
        for cfg, r in m["per_clip"].items():
            for k, v in r.items():
                per.setdefault(cfg, {}).setdefault(k, []).extend(v[: m["n_clips"]])
    assert len(set(ids)) == len(ids), "a clip appears in two shards"
    return per, ids, buckets


def greater(x, y):
    """One-sided paired Wilcoxon p that x > y, and the median paired difference."""
    d = np.asarray(x) - np.asarray(y)
    if np.allclose(d, 0):
        return 1.0, 0.0
    return float(wilcoxon(x, y, alternative="greater").pvalue), float(np.median(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    per, ids, _ = merge(args.shards)
    cfg0 = json.loads((REPO / "outputs" / args.shards[0] / "config.json").read_text())
    meta = {c["name"]: c for c in cfg0["configs"]}
    n = len(ids)
    rng = np.random.default_rng(0)
    dense = {k: np.array(per["dense"][k], float) for k in ("fm", "nll", "ade")}
    dfm = {c: np.array(r["fm"], float) - dense["fm"] for c, r in per.items() if c != "dense"}
    dnll = {c: np.array(r["nll"], float) - dense["nll"] for c, r in per.items() if c != "dense"}
    dade = {c: np.array(r["ade"], float) - dense["ade"] for c, r in per.items()
            if c != "dense" and r["ade"][0] is not None}

    def ci(x):
        b = rng.choice(x, (args.n_boot, len(x))).mean(1)
        return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

    out = {"n_clips": n, "dense": {k: float(v.mean()) for k, v in dense.items()}, "configs": {}}
    lines = [f"token-targeted ablation -- {n} held-out clips ({cfg0['manifest']}), shards {', '.join(args.shards)}",
             f"dense: FM loss {dense['fm'].mean():.4f}  CoC NLL {dense['nll'].mean():.4f}  minADE@{cfg0['k_samples']} "
             f"{dense['ade'].mean():.3f} (median {np.median(dense['ade']):.3f})", "",
             f"{'config':20s} {'rm/layer':>9s} {'dFM mean [95% CI]':>28s} {'dNLL mean [95% CI]':>28s} {'dminADE mean':>13s}"]
    for c in dfm:
        lo_f, hi_f = ci(dfm[c])
        lo_n, hi_n = ci(dnll[c])
        out["configs"][c] = {"dfm_mean": float(dfm[c].mean()), "dfm_ci": [lo_f, hi_f], "dfm_median": float(np.median(dfm[c])),
                             "dfm_rel": float(dfm[c].mean() / dense["fm"].mean()),
                             "dnll_mean": float(dnll[c].mean()), "dnll_ci": [lo_n, hi_n],
                             "dnll_median": float(np.median(dnll[c])),
                             "dnll_rel": float(dnll[c].mean() / dense["nll"].mean()),
                             "dade_mean": float(dade[c].mean()) if c in dade else None,
                             "dade_median": float(np.median(dade[c])) if c in dade else None}
        lines.append(f"{c:20s} {meta[c]['removed_per_layer_mean']:9.1f} {dfm[c].mean():+10.4f} [{lo_f:+.4f},{hi_f:+.4f}] "
                     f"{dnll[c].mean():+10.4f} [{lo_n:+.4f},{hi_n:+.4f}] "
                     + (f"{dade[c].mean():+13.4f}" if c in dade else f"{'-':>13s}")
                     + f"   rel dFM {dfm[c].mean() / dense['fm'].mean():+6.1%} dNLL {dnll[c].mean() / dense['nll'].mean():+6.1%}")
    lines.append("")
    # the scale of each band's effect, whatever the set: the mean over that band's random sets
    out["band_scale"] = {}
    for ax in ("q", "mlp"):
        for band in ("late", "trunk"):
            rs = [f"{ax}_{band}_R{i}" for i in range(5)]
            out["band_scale"][f"{ax}|{band}"] = {
                "dfm_rel": float(np.mean([out["configs"][r]["dfm_rel"] for r in rs])),
                "dnll_rel": float(np.mean([out["configs"][r]["dnll_rel"] for r in rs]))}
            s = out["band_scale"][f"{ax}|{band}"]
            lines.append(f"scale, five random sets, {ax:3s} {band:5s}: FM loss {s['dfm_rel']:+.1%} of dense, NLL {s['dnll_rel']:+.1%}")
    lines.append("")

    gates = {}
    for ax in ("q", "mlp"):
        rand = [f"{ax}_late_R{i}" for i in range(5)]
        for fam in ("tok", "pool"):
            t, c = f"{ax}_late_T{fam}", f"{ax}_late_C{fam}"
            p1, m1 = greater(dfm[t], dfm[c])
            p2, m2 = greater(dnll[c], dnll[t])
            r1 = [greater(dfm[t], dfm[r]) for r in rand]
            r2 = [greater(dnll[c], dnll[r]) for r in rand]
            gates[f"B1|{ax}|{fam}"] = {"T_gt_C": {"p": p1, "median_diff": m1},
                                       "T_gt_random": [{"p": p, "median_diff": m} for p, m in r1],
                                       "pass_cross": bool(p1 < 0.01),
                                       "pass_random": bool(all(p < 0.01 for p, _ in r1))}
            gates[f"B2|{ax}|{fam}"] = {"C_gt_T": {"p": p2, "median_diff": m2},
                                       "C_gt_random": [{"p": p, "median_diff": m} for p, m in r2],
                                       "pass_cross": bool(p2 < 0.01),
                                       "pass_random": bool(all(p < 0.01 for p, _ in r2))}
            lines.append(f"B1 {ax:3s} {fam:4s}: dFM(T) > dFM(C) p={p1:.1e} (median diff {m1:+.4f}) "
                         f"{'PASS' if p1 < 0.01 else 'FAIL'} | above every random set: "
                         f"{sum(p < 0.01 for p, _ in r1)}/5 {'PASS' if all(p < 0.01 for p, _ in r1) else 'FAIL'}")
            lines.append(f"B2 {ax:3s} {fam:4s}: dNLL(C) > dNLL(T) p={p2:.1e} (median diff {m2:+.4f}) "
                         f"{'PASS' if p2 < 0.01 else 'FAIL'} | above every random set: "
                         f"{sum(p < 0.01 for p, _ in r2)}/5 {'PASS' if all(p < 0.01 for p, _ in r2) else 'FAIL'}")

        # B3: the trunk control, on mean damages (a per-clip ratio of two small numbers is not usable)
        def spec(cfg):
            a, b = dfm[cfg].mean(), dnll[cfg].mean()
            return float(np.log(a / b)) if a > 0 and b > 0 else float("nan")
        for band in ("late", "trunk"):
            rs = np.array([spec(f"{ax}_{band}_R{i}") for i in range(5)])
            for fam in ("tok", "pool"):
                gap = spec(f"{ax}_{band}_T{fam}") - spec(f"{ax}_{band}_C{fam}")
                spread = float(np.nanmax(rs) - np.nanmin(rs))
                verdict = ("undefined (a mean damage is <= 0)" if np.isnan(gap) else
                           "dissociates" if abs(gap) > spread else "no dissociation")
                gates[f"B3|{ax}|{band}|{fam}"] = {"log_ratio_gap_T_minus_C": None if np.isnan(gap) else gap,
                                                  "random_spread": spread, "verdict": verdict}
                lines.append(f"B3 {ax:3s} {band:5s} {fam:4s}: log(dFM/dNLL) of T minus C = {gap:+.2f}; spread over the "
                             f"random sets {spread:.2f} -> {verdict}"
                             + (f" -> {'PASS' if verdict == 'no dissociation' else 'FAIL'}" if band == "trunk" else ""))

        # descriptive, the same two paired tests inside the trunk band: a per-clip reading of B3
        for fam in ("tok", "pool"):
            t, c = f"{ax}_trunk_T{fam}", f"{ax}_trunk_C{fam}"
            p1, m1 = greater(dfm[t], dfm[c])
            p2, m2 = greater(dnll[c], dnll[t])
            gates[f"trunk_pairs|{ax}|{fam}"] = {"dFM_T_gt_C": {"p": p1, "median_diff": m1},
                                                "dNLL_C_gt_T": {"p": p2, "median_diff": m2}}
            lines.append(f"   {ax:3s} trunk {fam:4s} (descriptive): dFM(T) > dFM(C) p={p1:.1e} ({m1:+.4f}); "
                         f"dNLL(C) > dNLL(T) p={p2:.1e} ({m2:+.4f})")

        # B4: token-resolved vs pooled selection
        for side, own, other in (("T", dfm, dnll), ("C", dnll, dfm)):
            tok, pool = f"{ax}_late_{side}tok", f"{ax}_late_{side}pool"
            p_own, m_own = greater(own[pool], own[tok])      # is pooled MORE damaging on its own channel?
            p_oth, m_oth = greater(other[tok], other[pool])  # is token-resolved MORE damaging on the other?
            gates[f"B4|{ax}|{side}"] = {"own_channel_pool_gt_tok": {"p": p_own, "median_diff": m_own},
                                        "other_channel_tok_gt_pool": {"p": p_oth, "median_diff": m_oth},
                                        "pass": bool(p_own >= 0.01 and p_oth >= 0.01)}
            lines.append(f"B4 {ax:3s} {side}: own-channel damage pool > tok p={p_own:.1e} ({m_own:+.4f}); other-channel "
                         f"damage tok > pool p={p_oth:.1e} ({m_oth:+.4f}) -> "
                         f"{'PASS (token set at least as specific)' if p_own >= 0.01 and p_oth >= 0.01 else 'FAIL'}")

        # B5: what the dual criterion saves
        for s, own, other, own_n, oth_n in (("Scoc", dfm, dnll, "dFM", "dNLL"), ("Straj", dnll, dfm, "dNLL", "dFM")):
            cfg, rs = f"{ax}_late_{s}", [f"{ax}_late_R{s}{i}" for i in range(3)]
            above = [greater(own[cfg], own[r]) for r in rs]
            r_mean = np.mean([other[r] for r in rs], 0)
            p_oth, m_oth = greater(other[cfg], r_mean)
            gates[f"B5|{ax}|{s}"] = {"own_gt_random": [{"p": p, "median_diff": m} for p, m in above],
                                     "other_gt_random_mean": {"p": p_oth, "median_diff": m_oth},
                                     "pass": bool(all(p < 0.01 for p, _ in above) and p_oth >= 0.01)}
            lines.append(f"B5 {ax:3s} {s:5s}: {own_n} above size-matched random sets {sum(p < 0.01 for p, _ in above)}/3 "
                         f"(mean {own[cfg].mean():+.4f} vs {np.mean([own[r].mean() for r in rs]):+.4f}); {oth_n} above their "
                         f"mean p={p_oth:.1e} (mean {other[cfg].mean():+.4f} vs {r_mean.mean():+.4f}) -> "
                         f"{'PASS' if gates[f'B5|{ax}|{s}']['pass'] else 'FAIL'}")
        lines.append("")
    out["gates"] = gates

    # first-order prediction vs measured damage, over every config
    names = list(dfm)
    fo_t = np.array([meta[c]["first_order"]["traj"] for c in names])
    fo_c = np.array([meta[c]["first_order"]["coc"] for c in names])
    m_t, m_c = np.array([dfm[c].mean() for c in names]), np.array([dnll[c].mean() for c in names])
    out["first_order"] = {}
    lines.append("first-order prediction (summed score of the removed units) vs measured mean damage, Spearman over configs")
    for ax in ("q", "mlp"):
        for band in ("late", "trunk"):
            sel = np.array([c.startswith(f"{ax}_{band}_") for c in names])
            rt, rc = spearmanr(fo_t[sel], m_t[sel])[0], spearmanr(fo_c[sel], m_c[sel])[0]
            out["first_order"][f"{ax}|{band}"] = {"n": int(sel.sum()), "traj": float(rt), "coc": float(rc)}
            lines.append(f"  {ax:3s} {band:5s} ({int(sel.sum()):2d} configs): I_traj -> dFM {rt:+.2f}   I_CoC -> dNLL {rc:+.2f}")

    out_dir = REPO / "outputs" / args.out
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    for i, ax in enumerate(("q", "mlp")):
        for j, band in enumerate(("late", "trunk")):
            a = axs[i, j]
            for c in names:
                if not c.startswith(f"{ax}_{band}_"):
                    continue
                s = meta[c]["set"]
                col = C1 if s.startswith(("T", "Scoc")) else C3 if s.startswith(("C", "Straj")) else MUTED
                a.errorbar(dfm[c].mean(), dnll[c].mean(),
                           xerr=np.abs(np.array(out["configs"][c]["dfm_ci"])[:, None] - dfm[c].mean()),
                           yerr=np.abs(np.array(out["configs"][c]["dnll_ci"])[:, None] - dnll[c].mean()),
                           fmt="o" if not s.startswith("S") else "s", color=col, ms=5, lw=0.8)
                if not s.startswith("R"):
                    a.annotate(s, (dfm[c].mean(), dnll[c].mean()), fontsize=8, xytext=(4, 4),
                               textcoords="offset points")
            a.axhline(0, color=MUTED, lw=0.6)
            a.axvline(0, color=MUTED, lw=0.6)
            a.set_title(f"{'Q heads' if ax == 'q' else 'MLP channels'}, layers {'22-34' if band == 'late' else '6-17'}")
            a.set_xlabel("FM loss: set removed - dense")
            a.set_ylabel("CoC NLL: set removed - dense")
    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=C1, label="trajectory-side sets (T, S-coc)"),
                        plt.Line2D([], [], marker="o", ls="", color=C3, label="CoC-side sets (C, S-traj)"),
                        plt.Line2D([], [], marker="o", ls="", color=MUTED, label="random sets")],
               loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_dir / "plots" / "dissociation.png", dpi=150)
    plt.close(fig)

    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    (out_dir / "config.json").write_text(json.dumps({
        "plan": "plans/2026-09-21_importance-causal-validation.md (part B)", "shards": args.shards,
        "n_clips": n, "clip_ids": ids, "sets": cfg0["sets"], "manifest": cfg0["manifest"],
        "k_samples": cfg0["k_samples"], "n_boot": args.n_boot}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
