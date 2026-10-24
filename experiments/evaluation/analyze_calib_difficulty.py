"""How is per-clip difficulty distributed INSIDE each calibration set?

No GPU: reads the `run_coc_multimode.py` rows of the six n=100 sets (calib_100, nt blocks
a-e) and takes CoC 0 with its 6 denoised paths -- the standard protocol, and what
`run_importance` conditions on (G0: bit-identical to `run_baseline.py --k 6`).

Per clip, five difficulty readings:

  nll        mean per-token NLL of the sampled CoC (teacher-forced, temperature 1):
             how unsure the VLM is about the sentence it wrote
  end_rms    RMS spread of the 6 endpoints (6.4 s) about their mean, no GT involved:
             multimodality of the action distribution
  ade_sd     SD of the 6 ADEs against the GT: the same spread as the metric sees it
  meanade    mean ADE of the 6 paths: how far a typical sample is from the GT
  minade     minADE@6: how far the best sample is

Each reading is then asked two questions per set:

  spread     quantiles, IQR, robust CV (IQR / median), p90 / p10: how wide is the range
             of difficulties the set contains
  balance    the 600 clips are pooled and cut into quintiles; a set that mirrors the pool
             holds 20 clips in each. Normalised entropy of the occupancy (1 = exactly
             balanced), chi-square against 20 x 5, and KS against the other five sets.

A joint view crosses the reasoning axis (nll) with the action axis (minade) on pooled
tertiles: a 3 x 3 grid with 11.1 expected per cell under independence-and-balance.

Usage:
  python experiments/evaluation/analyze_calib_difficulty.py [--out outputs/calib_difficulty]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chisquare, kruskal, ks_2samp, levene, spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

from analyze_coc_multimode import ARM, SETS, load

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})

METRICS = [("nll", "CoC NLL / token", "nat"), ("end_rms", "endpoint spread of 6 paths", "m"),
           ("ade_sd", "SD of 6 ADEs", "m"), ("meanade", "meanADE@6", "m"),
           ("minade", "minADE@6", "m")]
QS = [10, 25, 50, 75, 90]
N_Q = 5
EXPECT = 100


def clip_metrics(r):
    c = r["cocs"][0]
    P = np.array(c["pred_xy_k"])                           # (6, 64, 2)
    A = np.array(c["ade_k"])                               # (6,)
    end = P[:, -1, :]                                      # (6, 2)
    return {"nll": c["nll"], "ade_sd": A.std(ddof=1), "meanade": A.mean(), "minade": A.min(),
            "end_rms": float(np.sqrt(((end - end.mean(0)) ** 2).sum(-1).mean())),
            "bucket": r["bucket"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/calib_difficulty")
    args = ap.parse_args()
    out = Path(args.out) if Path(args.out).is_absolute() else REPO / args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    names = [n for n, _ in SETS]
    per = {}
    for name, stem in SETS:
        cm = [clip_metrics(r) for r in load(stem)]
        assert len(cm) == EXPECT, (name, len(cm))
        per[name] = {k: np.array([c[k] for c in cm], float) for k, _, _ in METRICS}
        per[name]["bucket"] = np.array([c["bucket"] for c in cm])
    pool = {k: np.concatenate([per[n][k] for n in names]) for k, _, _ in METRICS}
    metrics = {"sets": {n: {} for n in names}, "tests": {}, "n_per_set": EXPECT,
               "protocol": "CoC 0, 6 denoising seeds (standard); unpruned; Ada"}

    # ---- spread ----
    for k, label, unit in METRICS:
        say(f"\n== {label} [{unit}]: distribution inside each set")
        say(f"{'set':10s}{'mean':>8s}{'sd':>8s}" + "".join(f"{'p' + str(q):>8s}" for q in QS)
            + f"{'IQR':>8s}{'IQR/med':>9s}{'p90/p10':>9s}{'max':>8s}")
        for n in names + ["pooled"]:
            v = pool[k] if n == "pooled" else per[n][k]
            q = np.percentile(v, QS)
            row = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                   **{f"p{p}": float(x) for p, x in zip(QS, q)},
                   "iqr": float(q[3] - q[1]), "rcv": float((q[3] - q[1]) / q[2]),
                   "p90_p10": float(q[4] / q[0]), "max": float(v.max())}
            if n != "pooled":
                metrics["sets"][n][k] = row
            else:
                metrics.setdefault("pooled", {})[k] = row
            say(f"{n:10s}{row['mean']:8.3f}{row['sd']:8.3f}" + "".join(f"{x:8.3f}" for x in q)
                + f"{row['iqr']:8.3f}{row['rcv']:9.2f}{row['p90_p10']:9.1f}{row['max']:8.2f}")
        groups = [per[n][k] for n in names]
        # location (Kruskal) and scale (Brown-Forsythe on logs: the readings are right-skewed)
        kw = kruskal(*groups).pvalue
        bf = levene(*[np.log(g + 1e-6) for g in groups], center="median").pvalue
        metrics["tests"][k] = {"kruskal_p": float(kw), "brown_forsythe_log_p": float(bf)}
        say(f"across sets: location Kruskal p={kw:.3f}; spread Brown-Forsythe (log) p={bf:.3f}")

    # ---- balance over pooled quintiles ----
    say(f"\n== balance: clips per pooled quintile (Q1 easiest .. Q5 hardest; {EXPECT // N_Q} each "
        "if the set mirrors the pool)")
    occ = {}
    for k, label, _ in METRICS:
        edges = np.quantile(pool[k], np.linspace(0, 1, N_Q + 1)[1:-1])
        say(f"\n{label}  (cut points {', '.join(f'{e:.3f}' for e in edges)})")
        say(f"{'set':10s}" + "".join(f"{'Q' + str(i + 1):>6s}" for i in range(N_Q))
            + f"{'entropy':>9s}{'chi2 p':>8s}{'KS p':>8s}{'max |dev|':>10s}")
        for n in names:
            cnt = np.bincount(np.searchsorted(edges, per[n][k], side="right"), minlength=N_Q)
            p = cnt / cnt.sum()
            ent = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(N_Q))
            chi = float(chisquare(cnt).pvalue)
            rest = np.concatenate([per[m][k] for m in names if m != n])
            ks = float(ks_2samp(per[n][k], rest).pvalue)
            occ[(n, k)] = cnt
            metrics["sets"][n][k].update({"quintile_counts": cnt.tolist(), "entropy": ent,
                                          "chi2_p": chi, "ks_p_vs_rest": ks})
            say(f"{n:10s}" + "".join(f"{c:6d}" for c in cnt)
                + f"{ent:9.4f}{chi:8.3f}{ks:8.3f}{int(np.abs(cnt - EXPECT // N_Q).max()):10d}")

    # ---- composite difficulty ----
    # mean of the five pooled percentile ranks: a clip's overall standing among the 600
    rank = {k: np.argsort(np.argsort(pool[k])) / (len(pool[k]) - 1) for k, _, _ in METRICS}
    comp = np.mean([rank[k] for k, _, _ in METRICS], axis=0)
    off = np.cumsum([0] + [EXPECT] * len(names))
    say("\n== composite difficulty (mean pooled percentile of the five readings)")
    say(f"{'set':10s}{'mean':>8s}{'sd':>8s}" + "".join(f"{'p' + str(q):>8s}" for q in QS)
        + f"{'easy<.2':>9s}{'hard>.8':>9s}" + "".join(f"{'Q' + str(i + 1):>5s}" for i in range(N_Q))
        + f"{'entropy':>9s}")
    cedges = np.quantile(comp, np.linspace(0, 1, N_Q + 1)[1:-1])
    for i, n in enumerate(names):
        v = comp[off[i]:off[i + 1]]
        per[n]["comp"] = v
        cnt = np.bincount(np.searchsorted(cedges, v, side="right"), minlength=N_Q)
        p = cnt / cnt.sum()
        ent = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(N_Q))
        metrics["sets"][n]["composite"] = {
            "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "n_easy": int((v < 0.2).sum()), "n_hard": int((v > 0.8).sum()),
            "quintile_counts": cnt.tolist(), "entropy": ent,
            "chi2_p": float(chisquare(cnt).pvalue)}
        say(f"{n:10s}{v.mean():8.3f}{v.std(ddof=1):8.3f}"
            + "".join(f"{x:8.3f}" for x in np.percentile(v, QS))
            + f"{int((v < 0.2).sum()):9d}{int((v > 0.8).sum()):9d}"
            + "".join(f"{c:5d}" for c in cnt) + f"{ent:9.4f}")
    kw = kruskal(*[per[n]["comp"] for n in names]).pvalue
    bf = levene(*[per[n]["comp"] for n in names], center="median").pvalue
    metrics["tests"]["composite"] = {"kruskal_p": float(kw), "brown_forsythe_p": float(bf)}
    say(f"across sets: location Kruskal p={kw:.3f}; spread Brown-Forsythe p={bf:.3f}")

    # ---- joint: reasoning difficulty x action difficulty ----
    say("\n== joint grid: CoC NLL tertile (rows, low->high) x minADE@6 tertile (cols); "
        "11.1 per cell if balanced")
    te = {k: np.quantile(pool[k], [1 / 3, 2 / 3]) for k in ("nll", "minade")}
    for n in names:
        g = np.zeros((3, 3), int)
        np.add.at(g, (np.searchsorted(te["nll"], per[n]["nll"], side="right"),
                      np.searchsorted(te["minade"], per[n]["minade"], side="right")), 1)
        p = g.ravel() / g.sum()
        ent = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(9))
        metrics["sets"][n]["joint_grid"] = {"counts": g.tolist(), "entropy": ent,
                                            "min_cell": int(g.min()), "max_cell": int(g.max())}
        say(f"{n:10s} " + " | ".join(" ".join(f"{c:3d}" for c in row) for row in g)
            + f"   entropy {ent:.4f}  min {g.min()}  max {g.max()}")
    say("\n== pooled Spearman between the readings (600 clips)")
    ks_ = [k for k, _, _ in METRICS]
    metrics["pooled_rho"] = {}
    say(f"{'':10s}" + "".join(f"{k:>10s}" for k in ks_))
    for a in ks_:
        row = [spearmanr(pool[a], pool[b])[0] for b in ks_]
        metrics["pooled_rho"][a] = dict(zip(ks_, map(float, row)))
        say(f"{a:10s}" + "".join(f"{x:10.3f}" for x in row))

    # ---- by driving bucket (what makes a clip hard) ----
    pb = np.concatenate([per[n]["bucket"] for n in names])
    say("\n== pooled, by bucket (median)")
    say(f"{'bucket':12s}{'n':>5s}" + "".join(f"{k:>10s}" for k in ks_) + f"{'composite':>11s}")
    metrics["by_bucket"] = {}
    for bk in ("cruise", "decel_stop", "accel", "turn"):
        s = pb == bk
        row = [float(np.median(pool[k][s])) for k in ks_] + [float(np.median(comp[s]))]
        metrics["by_bucket"][bk] = {"n": int(s.sum()), **dict(zip(ks_ + ["composite"], row))}
        say(f"{bk:12s}{s.sum():5d}" + "".join(f"{x:10.3f}" for x in row[:-1]) + f"{row[-1]:11.3f}")

    # ---- set-level summaries against what each set's arm then scored (n=6) ----
    say("\n== Spearman(set statistic, downstream arm test500 minADE@6), n=6 -- descriptive only")
    y = [ARM[n] for n in names]
    metrics["rho_vs_arm"] = {}
    stats = [(f"{k} {s}", [metrics["sets"][n][k][s] for n in names])
             for k, _, _ in METRICS for s in ("p50", "iqr", "entropy")]
    stats += [(f"composite {s}", [metrics["sets"][n]["composite"][s] for n in names])
              for s in ("mean", "sd", "entropy", "n_hard", "n_easy")]
    stats += [("joint grid entropy", [metrics["sets"][n]["joint_grid"]["entropy"] for n in names])]
    for label, x in stats:
        r, p = spearmanr(x, y)
        metrics["rho_vs_arm"][label] = [float(r), float(p)]
        say(f"{label:24s} rho {r:+.3f}  p {p:.3f}")
    say(f"{len(stats)} statistics at n=6: |rho| >= 0.83 is needed for p < 0.05, and about "
        f"{len(stats) * 0.05:.0f} such hit(s) are expected by chance.")

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    (out / "config.json").write_text(json.dumps({
        "sets": dict(SETS), "source": [f"cocmm_{s}" for _, s in SETS], "coc_index": 0, "k": 6,
        "metrics": [k for k, _, _ in METRICS], "n_quantiles": N_Q, "arm_minade": ARM}, indent=2))

    # plots
    cols = [C4, C1, C2, C3, MUTED, INK]
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.9))
    for ax, (k, label, unit) in zip(axes, METRICS):
        bp = ax.boxplot([per[n][k] for n in names], patch_artist=True, widths=0.6,
                        whis=(10, 90), showfliers=False, medianprops={"color": INK})
        for b, c in zip(bp["boxes"], cols):
            b.set_facecolor(c)
            b.set_alpha(0.75)
        ax.set_xticks(range(1, 7), names, fontsize=8)
        ax.set_yscale("log")
        ax.set_title(f"{label} [{unit}]", fontsize=10)
    fig.suptitle("Box = IQR, whiskers = p10-p90, log scale", fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(out / "plots" / "distributions.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 6, figsize=(17, 3.4), sharey=True)
    keys = [k for k, _, _ in METRICS]
    for ax, k in zip(axes, keys + ["comp"]):
        m = np.array([occ[(n, k)] if k != "comp"
                      else metrics["sets"][n]["composite"]["quintile_counts"] for n in names])
        im = ax.imshow(m, cmap="RdBu_r", vmin=8, vmax=32, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, int(m[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if abs(m[i, j] - 20) > 7 else INK)
        ax.set_xticks(range(N_Q), [f"Q{i + 1}" for i in range(N_Q)], fontsize=8)
        ax.set_yticks(range(len(names)), names)
        ax.set_title("composite" if k == "comp" else k, fontsize=10)
    fig.colorbar(im, ax=axes, label="clips in pooled quintile (20 = balanced)", shrink=0.9)
    fig.savefig(out / "plots" / "quintile_occupancy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    for ax, n, c in zip(axes.ravel(), names, cols):
        ax.scatter(per[n]["nll"], per[n]["minade"], s=14, color=c, alpha=0.8)
        for e in te["nll"]:
            ax.axvline(e, color=MUTED, lw=0.6, ls="--")
        for e in te["minade"]:
            ax.axhline(e, color=MUTED, lw=0.6, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        g = metrics["sets"][n]["joint_grid"]
        ax.set_title(f"{n}: grid entropy {g['entropy']:.3f}, cells {g['min_cell']}-{g['max_cell']}")
    for ax in axes[-1]:
        ax.set_xlabel("CoC NLL / token")
    for ax in axes[:, 0]:
        ax.set_ylabel("minADE@6 [m]")
    fig.tight_layout()
    fig.savefig(out / "plots" / "joint_grid.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for n, c in zip(names, cols):
        v = np.sort(per[n]["comp"])
        ax.plot(v, np.linspace(0, 1, len(v)), color=c, lw=1.5, label=n)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("composite difficulty (pooled percentile)")
    ax.set_ylabel("cumulative share of the set")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "plots" / "composite_cdf.png", dpi=150)
    plt.close(fig)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
