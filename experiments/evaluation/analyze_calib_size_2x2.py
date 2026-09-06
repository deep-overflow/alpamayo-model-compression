"""Calibration size x criterion, 2x2 (plans/2026-09-06_calibration-size-2x2.md).

The dual -> maxstep11 gain was measured at n=100, where draw noise is several times the
criterion effect. This asks whether it survives at n=2,000, where selection churn reaches
the level G0b showed costs nothing measurable.

Pre-registered on test500 paired median dminADE@6, threshold 0.01 (eight times the noise
floor G0b measured). val500 and ood_val are read for direction only.

  H1  size    : st2000 - calib_100, same criterion
  H2  criterion: maxstep11 - dual, same calibration -- survives at n=2,000?
  H3  interaction: does the size effect differ by criterion

The criterion contrasts sit inside one box each (both calib_100 arms on cvlab21, both
st2000 arms on cvlab20); the size contrasts cross boxes, where only the unpruned baseline
path has been shown bit-identical.

Usage:
  python experiments/evaluation/analyze_calib_size_2x2.py
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, wilcoxon

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})

REPO = Path(__file__).resolve().parents[2]
O = REPO / "outputs"
K = 6
THRESHOLD = 0.01

ARMS = {
    ("dual", "calib_100"): "dual_u40_v2_ps_{s}",
    ("maxstep11", "calib_100"): "maxstep11_u40_v2_{s}",
    ("dual", "st2000"): "dual_u40_st2000_{s}",
    ("maxstep11", "st2000"): "maxstep11_u40_st2000_{s}",
}
SETS = {"test": "test", "indist": "indist", "oodval": "oodval"}
# the ps_ arms name their in-distribution set differently
ALIAS = {("dual", "calib_100", "indist"): "dual_u40_v2_ps_indist",
         ("dual", "calib_100", "oodval"): "dual_u40_v2_ps_ood"}


def load(tag):
    rows = []
    for f in sorted(glob.glob(str(O / tag / "*_s*of*.json"))):
        rows.extend(json.loads(Path(f).read_text()))
    if not rows:
        for f in sorted(glob.glob(str(O / tag / "*.json"))):
            if "config" in f:
                continue
            try:
                rows.extend(json.loads(Path(f).read_text()))
            except Exception:                      # noqa: BLE001 -- a metrics file
                pass
    return {r["clip_id"]: r for r in rows
            if isinstance(r, dict) and "ade_rollout_k" in r}


def a6(r, key="ade_rollout_k"):
    return float(np.min(np.asarray(r[key], dtype=float)[:K]))


def paired(a, b):
    ids = sorted(set(a) & set(b))
    if not ids:
        return None
    d = np.array([a6(a[i]) - a6(b[i]) for i in ids])
    rng = np.random.default_rng(0)
    boot = [np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(10000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n": len(ids), "median": float(np.median(d)), "lo": float(lo),
            "hi": float(hi), "p": float(wilcoxon(d).pvalue) if np.any(d) else 1.0,
            "sig": bool(lo > 0 or hi < 0)}


# every dual arm that differs only in its calibration set -> (label, n, pool)
DRAWS = {
    "dual_u40_v2_ps_test": ("calib_100", 100, "natural"),
    "dual_nt_a_test": ("nt_a", 100, "natural"),
    "dual_nt_b_test": ("nt_b", 100, "natural"),
    "dual_nt_c_test": ("nt_c", 100, "natural"),
    "dual_nt_d_test": ("nt_d", 100, "natural"),
    "dual_nt_e_test": ("nt_e", 100, "natural"),
    "dual_nt_c200_test": ("nt_c+100", 200, "natural"),
    "dual_nt_c300_test": ("nt_c+200", 300, "natural"),
    "dual_nt_c500_test": ("nt_c+400", 500, "natural"),
    "dual_u40_st2000_test": ("st2000", 2000, "natural"),
}


def draw_context(m):
    """H1 compares st2000 against calib_100, which is the best of six matched draws. The
    same test clips already carry the other five and a nested ladder, so the comparison
    can be made against the draw DISTRIBUTION instead of against its luckiest member."""
    vals = {}
    for tag, (lab, n, pool) in DRAWS.items():
        r = load(tag)
        if len(r) >= 400:
            vals[lab] = {"n": n, "pool": pool,
                         "minADE6": float(np.mean([a6(x) for x in r.values()]))}
    m["draw_context"] = vals
    if not vals:
        return
    print("\n== 같은 기준(dual), 캘리브레이션만 다름 ==")
    for lab, v in sorted(vals.items(), key=lambda kv: (kv[1]["n"], kv[0])):
        print(f"  {lab:10s} n={v['n']:5d}  {v['minADE6']:.4f}")
    n100 = [v["minADE6"] for v in vals.values() if v["n"] == 100]
    st = vals.get("st2000")
    if len(n100) > 2 and st:
        mu, sd = float(np.mean(n100)), float(np.std(n100, ddof=1))
        better = sum(x < st["minADE6"] for x in n100)
        m["draw_context"]["_summary"] = {
            "n100_mean": mu, "n100_sd": sd, "n100_count": len(n100),
            "st2000_rank_among_n100": better,
            "st2000_minus_n100_mean": st["minADE6"] - mu}
        print(f"  n=100 추출 {len(n100)}개: 평균 {mu:.4f} SD {sd:.4f}")
        print(f"  st2000은 그 중 {better}개보다 나쁨 — 평균 대비 "
              f"{st['minADE6'] - mu:+.4f}, 운 좋은 calib_100 대비 "
              f"{st['minADE6'] - vals['calib_100']['minADE6']:+.4f}")


def union_diversity(m):
    """max over K ranks is a UNION, and a union only buys coverage while its members
    disagree. Averaging over 2,000 clips instead of 100 should make the ten per-step
    gradients converge onto each other, collapsing maxstep11's union toward a single
    ranking -- while dual's two members stay apart, being different objectives rather
    than two noisy looks at one thing. That asymmetry is what H3 asks about."""
    out = {}
    runs = {"calib_100": ("importance_v2", "importance_stepvlm_v1"),
            "st2000": ("importance_st4000_c2000", "importance_stepvlm_st2000")}
    for cal, (imp_run, step_run) in runs.items():
        ip = O / imp_run / "importance.npz"
        sp = O / step_run / "step_importance_vlm.npz"
        if not ip.exists():
            continue
        z = np.load(ip)
        step = np.load(sp) if sp.exists() else None
        for axis, tk, ck, sk in (("q", "traj_vlm_q", "coc_vlm_q", "q_abs_step"),
                                 ("mlp", "traj_vlm_mlp", "coc_vlm_mlp", "mlp_abs_step")):
            t, c = np.asarray(z[tk], float), np.asarray(z[ck], float)
            keep = round(0.3985632694 * t.shape[1])
            for crit in ("dual", "maxstep11"):
                if crit == "maxstep11" and step is None:
                    continue
                rhos, unions = [], []
                for lyr in range(t.shape[0]):
                    rows_ = ([t[lyr], c[lyr]] if crit == "dual"
                             else list(np.asarray(step[sk], float)[:, lyr, :]) + [c[lyr]])
                    good = [r for r in rows_ if np.std(r) > 0]
                    if len(good) < 2:
                        continue
                    rhos.append(np.nanmean([spearmanr(good[i], good[j]).statistic
                                            for i in range(len(good))
                                            for j in range(i + 1, len(good))]))
                    sel = set()
                    for r in good:
                        sel |= set(np.argsort(-r)[:keep])
                    unions.append(len(sel) / keep)
                out[f"{crit}_{cal}_{axis}"] = {
                    "mean_pairwise_rho": float(np.mean(rhos)),
                    "union_size": float(np.mean(unions)), "layers": len(rhos)}
    m["union_diversity"] = out
    if not out:
        return
    print("\n== 기준 내부 순위들의 불일치 (max = 합집합) ==")
    print(f"  {'':22s} {'평균 상관':>10s} {'합집합':>8s}")
    for k in sorted(out):
        v = out[k]
        print(f"  {k:22s} {v['mean_pairwise_rho']:+10.4f} {v['union_size']:8.3f}x")


def plots(m, out):
    """Two figures carry the argument.

    (1) st2000 against the DRAW DISTRIBUTION rather than against calib_100 alone -- the
        pre-registered control is the best of six, so the single comparison reads as a
        loss while the distribution reads as a median.
    (2) the union collapse that explains why maxstep11 loses four times as much."""
    out.mkdir(parents=True, exist_ok=True)
    dc = {k: v for k, v in m.get("draw_context", {}).items() if not k.startswith("_")}
    if dc:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        n100 = [v["minADE6"] for v in dc.values() if v["n"] == 100]
        lad = sorted([(v["n"], v["minADE6"]) for k, v in dc.items()
                      if k.startswith("nt_c")] + [(100, dc["nt_c"]["minADE6"])])
        ax.scatter([100] * len(n100), n100, s=46, color=C3, zorder=3,
                   label=f"{len(n100)} draws at n=100 (SD {np.std(n100, ddof=1):.3f})")
        ax.plot([x for x, _ in lad], [y for _, y in lad], "-o", color=C1, lw=1.6, ms=5,
                zorder=4, label="nested ladder (nt_c extended)")
        if "calib_100" in dc:
            ax.scatter([100], [dc["calib_100"]["minADE6"]], s=90, marker="*",
                       color=C2, zorder=5, label="calib_100 (best of six)")
        if "st2000" in dc:
            ax.scatter([2000], [dc["st2000"]["minADE6"]], s=90, marker="D",
                       color=C4, zorder=5, label="st2000")
        ax.axhline(np.mean(n100), color=MUTED, ls=":", lw=1)
        ax.text(2050, np.mean(n100), " mean of n=100", va="center", color=MUTED, fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("calibration clips")
        ax.set_ylabel("test500 minADE@6")
        ax.set_title("same criterion (dual), calibration set varied")
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(out / "draw_vs_size.png", dpi=150)
        plt.close(fig)

    ud = m.get("union_diversity", {})
    if ud:
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
        for ax, axis in zip(axes, ("q", "mlp")):
            for i, crit in enumerate(("dual", "maxstep11")):
                ys = [ud.get(f"{crit}_{c}_{axis}", {}).get("union_size", np.nan)
                      for c in ("calib_100", "st2000")]
                ax.plot([0, 1], ys, "-o", color=(C1, C4)[i], lw=2, ms=6, label=crit)
                for x, y in zip((0, 1), ys):
                    if np.isfinite(y):
                        ax.annotate(f"{y:.3f}x", (x, y), textcoords="offset points",
                                    xytext=(0, 7), ha="center", fontsize=8)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["calib_100", "st2000"])
            ax.set_ylabel("union size (x budget)")
            ax.set_title(f"{axis} axis")
            ax.legend(fontsize=8, frameon=False)
        fig.suptitle("max is a union: it collapses as its members converge", y=1.0)
        fig.tight_layout()
        fig.savefig(out / "union_collapse.png", dpi=150)
        plt.close(fig)
    print("plots ->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/calib_size_2x2")
    args = ap.parse_args()

    rows = {}
    for (crit, cal), pat in ARMS.items():
        for s in SETS:
            tag = ALIAS.get((crit, cal, s), pat.format(s=s))
            r = load(tag)
            if r:
                rows[(crit, cal, s)] = r

    print("== 절대값 minADE@6 ==")
    print(f"{'':22s} " + "".join(f"{s:>12s}" for s in SETS))
    absolute = {}
    for crit in ("dual", "maxstep11"):
        for cal in ("calib_100", "st2000"):
            line = f"{crit + ' @ ' + cal:22s} "
            for s in SETS:
                r = rows.get((crit, cal, s))
                if r:
                    v = float(np.mean([a6(x) for x in r.values()]))
                    absolute[f"{crit}|{cal}|{s}"] = {"minADE6": v, "n": len(r)}
                    line += f"{v:12.4f}"
                else:
                    line += f"{'—':>12s}"
            print(line)

    m = {"absolute": absolute, "H2_criterion": {}, "H1_size": {}, "threshold": THRESHOLD}

    print("\n== H2: maxstep11 − dual (같은 캘리브레이션, 같은 서버) ==")
    for cal in ("calib_100", "st2000"):
        for s in SETS:
            a, b = rows.get(("maxstep11", cal, s)), rows.get(("dual", cal, s))
            if not (a and b):
                continue
            d = paired(a, b)
            m["H2_criterion"][f"{cal}|{s}"] = d
            print(f"  {cal:10s} {s:7s} {d['median']:+.4f}{'*' if d['sig'] else ' '} "
                  f"[{d['lo']:+.4f},{d['hi']:+.4f}] p={d['p']:.2g} n={d['n']}")

    print("\n== H1: st2000 − calib_100 (같은 기준, 서버 교차) ==")
    for crit in ("dual", "maxstep11"):
        for s in SETS:
            a, b = rows.get((crit, "st2000", s)), rows.get((crit, "calib_100", s))
            if not (a and b):
                continue
            d = paired(a, b)
            m["H1_size"][f"{crit}|{s}"] = d
            print(f"  {crit:10s} {s:7s} {d['median']:+.4f}{'*' if d['sig'] else ' '} "
                  f"[{d['lo']:+.4f},{d['hi']:+.4f}] p={d['p']:.2g} n={d['n']}")

    h2 = m["H2_criterion"].get("st2000|test")
    if h2:
        v = ("H2 채택 — 기준 차이가 n=2,000에서도 남는다" if h2["median"] < -THRESHOLD
             else "H2 기각 — 기준 차이가 사라진다" if abs(h2["median"]) < THRESHOLD
             else "역전 — st2000에서는 dual이 낫다")
        m["verdict"] = v
        print(f"\n판정 (test500): {v}")

    draw_context(m)
    union_diversity(m)

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    plots(m, out / "plots")
    (out / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False))
    print("->", out)


if __name__ == "__main__":
    main()
