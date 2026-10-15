"""Which calibration set holds the multimodal / hard clips? Reads `run_coc_multimode.py`.

Six n=100 calibration sets drawn by the same greedy rule (calib_100 and the nt blocks
a-e), the unpruned model, 6 CoC rollouts x 6 denoising seeds per clip = 36 paths.

Per clip, with P[j, k] the path under CoC j and noise k (6, 6, 64, 2):

  vs GT        ADE / FDE of all 36 paths: their SD and range, and the same split into
               within-CoC (SD over k, averaged over j) and between-CoC (SD over j of the
               per-CoC mean).
  path space   law of total variance on positions, no GT involved:
                 total   = E_jk |P_jk - P..|^2
                 within  = E_jk |P_jk - P_j.|^2      same sentence, different noise
                 between = E_j  |P_j. - P..|^2       different sentence
               reported as RMS metres at the 6.4 s endpoint and averaged over the path,
               plus `between_share = between / total`. No within/K correction is applied
               to the between term: the K denoising seeds are shared by every CoC, so
               the noise main effect cancels in P_j. - P.. and what is left is the CoC
               main effect plus 1/K of the CoC x noise interaction.
  confidence   mean per-token NLL of the 6 sampled CoCs (temperature 1, teacher-forced),
               number of distinct sentences and of distinct 2-word heads (the head clause
               is the meta-action, see coc_action_consistency.py).
  difficulty   minADE@6 under CoC 0 (the protocol number), mean ADE over 36, minADE@36.

A clip is flagged on an indicator when it is in the pooled top 20% of all 600 clips, so a
set's count has expectation 20 under "sets are exchangeable"; chi-square tests that.

G0: CoC 0 must reproduce `run_baseline.py --k 6 --save-pred` on the same clips bitwise
(`--g0`, needs outputs/mm_<manifest>).

Usage:
  python experiments/evaluation/analyze_coc_multimode.py [--out outputs/coc_multimode] [--g0]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency, kruskal, spearmanr

REPO = Path(__file__).resolve().parents[2]

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})

SETS = [("calib_100", "calib_100"), ("a", "calib_nt100_a"), ("b", "calib_nt100_b"),
        ("c", "calib_nt100_c"), ("d", "calib_nt100_d"), ("e", "calib_nt100_e")]
# test500 minADE@6 of the dual_u40_v2 arm calibrated on each set
# (reports/evaluation/2026-09-03_calib-draw-variance.html)
ARM = {"calib_100": 0.9498, "a": 1.0219, "b": 1.0602, "c": 1.3479, "d": 1.7742, "e": 1.7178}
BUCKETS = ["cruise", "decel_stop", "accel", "turn"]
EXPECT, BOOT, TOP = 100, 10000, 0.20
# (key, label, unit) -- the per-clip quantities every table is built from
VSGT = [("ade_sd", "ADE SD (36)", "m"), ("ade_range", "ADE range (36)", "m"),
        ("fde_sd", "FDE SD (36)", "m"), ("fde_range", "FDE range (36)", "m"),
        ("ade_sd_within", "ADE SD within CoC", "m"), ("ade_sd_between", "ADE SD between CoC", "m"),
        ("fde_sd_within", "FDE SD within CoC", "m"), ("fde_sd_between", "FDE SD between CoC", "m")]
PATH = [("end_rms_total", "endpoint RMS total", "m"), ("end_rms_within", "endpoint RMS within", "m"),
        ("end_rms_between", "endpoint RMS between", "m"),
        ("path_rms_within", "path RMS within", "m"), ("path_rms_between", "path RMS between", "m"),
        ("end_share_between", "between share (endpoint)", "")]
CONF = [("nll", "CoC NLL / token", "nat"), ("nll_sum", "CoC NLL / sentence", "nat"),
        ("n_text", "distinct CoC of 6", ""), ("n_head", "distinct heads of 6", "")]
DIFF = [("minade6", "minADE@6 (CoC 0)", "m"), ("ade_mean36", "mean ADE (36)", "m"),
        ("minade36", "minADE@36", "m"), ("minfde6", "minFDE@6 (CoC 0)", "m")]
HARD = ["nll", "n_head", "end_rms_between", "end_rms_within", "ade_mean36", "minade6"]


def load(stem):
    rows = []
    for p in sorted((REPO / "outputs" / f"cocmm_{stem}").glob("rows_s*of*.json")):
        rows += json.loads(p.read_text())
    assert len({r["clip_id"] for r in rows}) == len(rows), stem
    return rows


def head(text):
    return " ".join(str(text).lower().split()[:2])


def clip_metrics(r):
    P = np.array([c["pred_xy_k"] for c in r["cocs"]])      # (J, K, 64, 2)
    A = np.array([c["ade_k"] for c in r["cocs"]])          # (J, K)
    F = np.array([c["fde_k"] for c in r["cocs"]])          # (J, K)
    m = {"bucket": r["bucket"], "clip_id": r["clip_id"]}
    for nm, X in (("ade", A), ("fde", F)):
        m[f"{nm}_sd"] = X.std(ddof=1)
        m[f"{nm}_range"] = X.max() - X.min()
        m[f"{nm}_sd_within"] = X.std(1, ddof=1).mean()
        m[f"{nm}_sd_between"] = X.mean(1).std(ddof=1)
    Pj = P.mean(1, keepdims=True)                          # (J, 1, 64, 2) per-CoC mean path
    Pg = P.mean((0, 1), keepdims=True)                     # (1, 1, 64, 2)
    within = ((P - Pj) ** 2).sum(-1).mean((0, 1))          # (64,)
    between = ((Pj - Pg) ** 2).sum(-1).mean((0, 1))        # (64,)
    m["end_rms_within"], m["end_rms_between"] = np.sqrt(within[-1]), np.sqrt(between[-1])
    m["end_rms_total"] = np.sqrt(within[-1] + between[-1])
    m["path_rms_within"], m["path_rms_between"] = np.sqrt(within.mean()), np.sqrt(between.mean())
    m["end_var_within"], m["end_var_between"] = within[-1], between[-1]
    tot = between[-1] + within[-1]
    m["end_share_between"] = between[-1] / tot if tot > 0 else 0.0
    texts = [c["text"] for c in r["cocs"]]
    m["nll"] = float(np.mean([c["nll"] for c in r["cocs"]]))
    m["nll_sum"] = float(np.mean([c["nll_sum"] for c in r["cocs"]]))
    m["n_text"], m["n_head"] = len(set(texts)), len({head(t) for t in texts})
    m["degen"] = float(np.mean([c["degenerate"] for c in r["cocs"]]))
    m["minade6"], m["minfde6"] = A[0].min(), F[0].min()
    m["ade_mean36"], m["minade36"] = A.mean(), A.min()
    return m


def boot_ci(x, rng):
    idx = rng.integers(0, len(x), (BOOT, len(x)))
    return [float(v) for v in np.percentile(x[idx].mean(1), [2.5, 97.5])]


def g0():
    ok = True
    for name, stem in SETS:
        p = REPO / "outputs" / f"mm_{stem}" / "baseline_s0of1.json"
        if not p.exists():
            print(f"G0 {name}: no single-CoC run")
            continue
        ref = {r["clip_id"]: r for r in json.loads(p.read_text())}
        rows = load(stem)
        same = sum(r["cocs"][0]["ade_k"] == ref[r["clip_id"]]["ade_rollout_k"]
                   and r["cocs"][0]["pred_xy_k"] == ref[r["clip_id"]]["pred_xy_k"]
                   and r["cocs"][0]["text"] == ref[r["clip_id"]]["gen_coc"] for r in rows)
        print(f"G0 {name}: {same}/{len(rows)} clips bit-identical under CoC 0")
        ok &= same == len(rows)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/coc_multimode")
    ap.add_argument("--g0", action="store_true")
    args = ap.parse_args()
    out = Path(args.out) if Path(args.out).is_absolute() else REPO / args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    g0_ok = g0() if args.g0 else None
    names = [n for n, _ in SETS]
    per = {}
    for name, stem in SETS:
        rows = load(stem)
        assert len(rows) == EXPECT, (name, len(rows))
        cm = [clip_metrics(r) for r in rows]
        per[name] = {k: np.array([c[k] for c in cm]) for k in cm[0] if k not in ("bucket", "clip_id")}
        per[name]["bucket"] = np.array([c["bucket"] for c in cm])
        per[name]["clip_id"] = np.array([c["clip_id"] for c in cm])

    allk = [k for k, _, _ in VSGT + PATH + CONF + DIFF]
    metrics = {"sets": {}, "g0_bit_identical": g0_ok, "n_clips_per_set": EXPECT,
               "design": "6 CoC x 6 traj per clip, unpruned, Ada"}
    for n in names:
        s = {}
        for k in allk:
            v = per[n][k].astype(float)
            s[k] = {"mean": float(v.mean()), "ci": boot_ci(v, rng), "median": float(np.median(v)),
                    "p90": float(np.percentile(v, 90))}
        # pooled variance share: ratio of set-level sums, not a mean of per-clip ratios
        w, b = per[n]["end_var_within"].sum(), per[n]["end_var_between"].sum()
        s["pooled_between_share"] = float(b / (b + w))
        s["n_between_gt_within"] = int((per[n]["end_rms_between"] > per[n]["end_rms_within"]).sum())
        s["n_between_gt_1m"] = int((per[n]["end_rms_between"] > 1.0).sum())
        s["frac_one_coc"] = float((per[n]["n_text"] == 1).mean())
        s["frac_one_head"] = float((per[n]["n_head"] == 1).mean())
        s["degen"] = float(per[n]["degen"].mean())
        s["buckets"] = {bk: int((per[n]["bucket"] == bk).sum()) for bk in BUCKETS}
        metrics["sets"][n] = s

    def table(title, spec, stat="mean"):
        say(f"\n== {title} (set {stat})")
        say(f"{'':34s}" + "".join(f"{n:>11s}" for n in names) + "   largest   KW p")
        for k, label, _ in spec:
            vals = [metrics["sets"][n][k][stat] for n in names]
            p = kruskal(*[per[n][k] for n in names]).pvalue
            metrics.setdefault("kruskal_p", {})[k] = float(p)
            say(f"{label:34s}" + "".join(f"{v:11.3f}" for v in vals)
                + f"   {names[int(np.argmax(vals))]:9s} {p:.3f}")

    for stat in ("mean", "median"):
        table("variation of ADE / FDE vs GT", VSGT, stat)
        table("path-space variance split", PATH, stat)
        table("CoC confidence and diversity", CONF, stat)
        table("difficulty", DIFF, stat)

    say("\n== set-level shares")
    say(f"{'':34s}" + "".join(f"{n:>11s}" for n in names))
    for k, label in (("pooled_between_share", "between share, pooled variance"),
                     ("frac_one_coc", "clips with one CoC text"),
                     ("frac_one_head", "clips with one head"), ("degen", "CoC degenerate rate"),
                     ("n_between_gt_within", "clips: between > within"),
                     ("n_between_gt_1m", "clips: between RMS > 1 m")):
        say(f"{label:34s}" + "".join(f"{metrics['sets'][n][k]:11.3f}" for n in names))
    for bk in BUCKETS:
        say(f"{'bucket ' + bk:34s}" + "".join(f"{metrics['sets'][n]['buckets'][bk]:11d}" for n in names))

    # hard clips: pooled top-20% on each indicator
    say(f"\n== clips in the pooled top {TOP:.0%} per indicator (expectation {EXPECT * TOP:.0f} per set)")
    say(f"{'':34s}" + "".join(f"{n:>11s}" for n in names) + "   chi2 p")
    flags = {n: np.zeros(EXPECT, int) for n in names}
    metrics["hard"] = {}
    for k in HARD:
        pool = np.concatenate([per[n][k] for n in names]).astype(float)
        thr = np.quantile(pool, 1 - TOP)
        # ties at the threshold (n_head is an integer) are flagged only above it
        cnt = [int((per[n][k] > thr).sum()) for n in names]
        for n in names:
            flags[n] += per[n][k] > thr
        p = chi2_contingency([cnt, [EXPECT - c for c in cnt]])[1] if 0 < sum(cnt) < 6 * EXPECT else 1.0
        metrics["hard"][k] = {"threshold": float(thr), "count": dict(zip(names, cnt)), "chi2_p": float(p)}
        say(f"{k + f' > {thr:.3f}':34s}" + "".join(f"{c:11d}" for c in cnt) + f"   {p:.3f}")
    for need in (2, 3, 4):
        cnt = [int((flags[n] >= need).sum()) for n in names]
        p = chi2_contingency([cnt, [EXPECT - c for c in cnt]])[1]
        metrics["hard"][f"flags_ge{need}"] = {"count": dict(zip(names, cnt)), "chi2_p": float(p)}
        say(f"{f'flagged on >= {need} of {len(HARD)}':34s}" + "".join(f"{c:11d}" for c in cnt) + f"   {p:.3f}")

    # the tail: mass carried by a set's five most extreme clips
    say("\n== share of a set's total carried by its top 5 clips")
    say(f"{'':34s}" + "".join(f"{n:>11s}" for n in names))
    for k in ("end_var_between", "end_var_within", "ade_mean36"):
        sh = [float(np.sort(per[n][k])[-5:].sum() / per[n][k].sum()) for n in names]
        metrics.setdefault("top5_share", {})[k] = dict(zip(names, sh))
        say(f"{k:34s}" + "".join(f"{v:11.3f}" for v in sh))

    # pooled clip-level structure
    pool = {k: np.concatenate([per[n][k] for n in names]).astype(float) for k in allk}
    pb = np.concatenate([per[n]["bucket"] for n in names])
    say("\n== pooled over 600 clips: Spearman between clip-level measures")
    metrics["pooled_rho"] = {}
    for a, b in (("nll", "n_head"), ("nll", "end_rms_between"), ("nll", "end_rms_within"),
                 ("nll", "minade6"), ("n_head", "end_rms_between"),
                 ("end_rms_between", "minade6"), ("end_rms_within", "minade6"),
                 ("end_rms_within", "end_rms_between")):
        r, p = spearmanr(pool[a], pool[b])
        metrics["pooled_rho"][f"{a}~{b}"] = [float(r), float(p)]
        say(f"{a:18s} ~ {b:18s} rho {r:+.3f}  p {p:.1e}")
    say("\n== pooled, by bucket (mean)")
    say(f"{'':12s}{'n':>5s}" + "".join(f"{k:>17s}" for k in ("nll", "n_head", "end_rms_within",
                                                              "end_rms_between", "minade6")))
    metrics["by_bucket"] = {}
    for bk in BUCKETS:
        sel = pb == bk
        row = {k: float(pool[k][sel].mean()) for k in ("nll", "n_head", "end_rms_within",
                                                        "end_rms_between", "minade6")}
        metrics["by_bucket"][bk] = {"n": int(sel.sum()), **row}
        say(f"{bk:12s}{sel.sum():5d}" + "".join(f"{v:17.3f}" for v in row.values()))

    # does any set-level property line up with what the calibrated arm then scored? n=6
    say("\n== Spearman(set mean, downstream arm test500 minADE@6), n=6 -- descriptive only")
    y = [ARM[n] for n in names]
    metrics["rho_vs_arm"] = {}
    for k, label, _ in VSGT[:4] + PATH + CONF + DIFF:
        r, p = spearmanr([metrics["sets"][n][k]["mean"] for n in names], y)
        metrics["rho_vs_arm"][k] = [float(r), float(p)]
        say(f"{label:38s} rho {r:+.3f}  p {p:.3f}")
    say("18 statistics at n=6: |rho| >= 0.83 is needed for p < 0.05, and about one such hit "
        "is expected by chance.")

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    (out / "config.json").write_text(json.dumps({
        "sets": dict(SETS), "runs": [f"cocmm_{s}" for _, s in SETS], "n_coc": 6, "k": 6,
        "top_fraction": TOP, "hard_indicators": HARD, "boot": BOOT, "arm_minade": ARM}, indent=2))

    # plots
    xs = np.arange(len(names))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (k, label, unit) in zip(axes.ravel(), [VSGT[1], VSGT[3], PATH[1], PATH[2], CONF[0], CONF[3]]):
        mu = [metrics["sets"][n][k]["mean"] for n in names]
        lo = [mu[i] - metrics["sets"][n][k]["ci"][0] for i, n in enumerate(names)]
        hi = [metrics["sets"][n][k]["ci"][1] - mu[i] for i, n in enumerate(names)]
        ax.bar(xs, mu, color=[C4] + [C1] * 5, width=0.65)
        ax.errorbar(xs, mu, yerr=[lo, hi], fmt="none", ecolor=INK, capsize=3, lw=1)
        ax.plot(xs, [metrics["sets"][n][k]["median"] for n in names], "o", color=INK, ms=4)
        ax.set_xticks(xs, names)
        ax.set_title(f"{label}" + (f" [{unit}]" if unit else ""))
    fig.suptitle("Set mean with 95% bootstrap CI (bars), set median (dots)", fontsize=10, color=MUTED)
    fig.tight_layout()
    fig.savefig(out / "plots" / "set_means.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), sharex=True, sharey=True)
    for ax, n in zip(axes.ravel(), names):
        nh = per[n]["n_head"]
        sc = ax.scatter(per[n]["end_rms_within"], per[n]["end_rms_between"], c=nh, cmap="viridis",
                        vmin=1, vmax=6, s=16, alpha=0.85)
        lim = 26
        ax.plot([0, lim], [0, lim], color=MUTED, lw=0.8, ls="--")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        above = int((per[n]["end_rms_between"] > per[n]["end_rms_within"]).sum())
        ax.set_title(f"{n}: {above}/100 above the diagonal")
    for ax in axes[-1]:
        ax.set_xlabel("endpoint RMS within CoC [m]")
    for ax in axes[:, 0]:
        ax.set_ylabel("endpoint RMS between CoC [m]")
    fig.colorbar(sc, ax=axes, label="distinct CoC heads of 6", shrink=0.8)
    fig.savefig(out / "plots" / "within_vs_between.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.13
    for i, k in enumerate(HARD):
        ax.bar(xs + (i - 2.5) * w, [metrics["hard"][k]["count"][n] for n in names], width=w,
               label=k, color=[C1, C2, C3, C4, MUTED, INK][i])
    ax.axhline(EXPECT * TOP, color=MUTED, lw=0.8, ls="--")
    ax.set_xticks(xs, names)
    ax.set_ylabel(f"clips in pooled top {TOP:.0%}")
    ax.legend(ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "plots" / "hard_counts.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, k in zip(axes, ("nll", "end_rms_between", "end_rms_within")):
        for n, c in zip(names, [C4, C1, C2, C3, MUTED, INK]):
            v = np.sort(per[n][k])
            ax.plot(v, np.linspace(0, 1, len(v)), color=c, lw=1.4, label=n)
        ax.set_xlabel(k)
        ax.set_ylabel("cumulative share of clips")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "plots" / "cdfs.png", dpi=150)
    plt.close(fig)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
