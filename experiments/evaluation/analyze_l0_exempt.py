"""Does keeping all of layer 0's Q heads remove the draw / seed collapses? test500.

`make_l0_exempt.py` variants: eight dual_u40_v2 kept sets -- the shipped arm, three seed
arms on calib_100 (ref, c3, n1) and four other calibration blocks (nt_b..e) -- each with
layer 0's 32 Q heads restored and nothing else changed. Every variant is paired, over all
500 test clips, with its own source arm, with the shipped arm and with the unpruned model.

Pre-stated readings:
  H1  the spread across the eight arms shrinks: SD of arm means was 0.34 before.
  H2  the collapsed arms (n1, nt_c, nt_d, nt_e) land within the pruning cost of the
      shipped arm.
  H3  the shipped arm itself does not get worse (exemption is free where nothing was wrong).

The exempt variants remove 0.51% fewer parameters (13 heads of one layer), so "exempt
beats own" is not a matched-budget claim; the shipped-vs-shipped row sizes what that
0.51% buys on its own.

Usage:
  python experiments/evaluation/analyze_l0_exempt.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C3, C4 = "#2a78d6", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})
OWN = {"ship": "dual_u40_v2_ps_test", "ref": "dual_seed_ref_test", "c3": "dual_seed_c3_test",
       "n1": "dual_seed_n1_test", **{f"nt{x}": f"dual_nt_{x}_test" for x in "bcde"}}
BUCKETS = ["cruise", "decel_stop", "accel", "turn"]
BOOT, K, EXPECT = 20000, 6, 500


def load_run(exp):
    rows = {}
    for p in sorted((REPO / "outputs" / exp).glob("*.json")):
        if not p.name.startswith("config"):
            for r in json.loads(p.read_text()):
                rows[r["clip_id"]] = r
    return rows


def main():
    out = REPO / "outputs" / "swap_masks"
    (out / "plots").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    base = load_run("baseline_ada_ps_test")
    ids = sorted(base)
    assert len(ids) == EXPECT
    bucket = np.array([base[c]["bucket"] for c in ids])

    def vec(rows, key="ade_rollout_k"):
        return np.array([min(rows[c][key][:K]) for c in ids])

    def ci(d):
        idx = rng.integers(0, len(d), (BOOT, len(d)))
        lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
        nz = d[d != 0]
        return {"mean": float(d.mean()), "ci": [float(lo), float(hi)],
                "median": float(np.median(d)),
                "p": float(wilcoxon(nz).pvalue) if len(nz) > 10 else 1.0,
                "trim10": float(np.sort(d)[len(d) // 10: -len(d) // 10].mean())}

    def f(s):
        star = "*" if s["ci"][0] > 0 or s["ci"][1] < 0 else " "
        return f"{s['mean']:+.4f} [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}]{star}"

    b = vec(base)
    own, ex, raw = {}, {}, {}
    for n, e in OWN.items():
        own[n] = vec(load_run(e))
        rows = load_run(f"swap_{n}_L0full_500")
        if all(c in rows for c in ids):
            ex[n], raw[n] = vec(rows), rows
        else:
            say(f"missing {n}_L0full ({len(rows)}/{EXPECT})")
    ship = own["ship"]
    metrics = {"n_clips": EXPECT, "unpruned": float(b.mean()), "arms": {}}

    say(f"unpruned {b.mean():.4f} (median {np.median(b):.4f}); shipped {ship.mean():.4f}")
    say("\n== test500 minADE@6: each arm as built, and with layer 0's Q heads all kept")
    say(f"{'arm':6s}{'own':>8s}{'exempt':>8s}{'med own':>9s}{'med ex':>8s}{'degen':>7s}   "
        f"exempt - own [95% CI]            p        exempt - shipped [95% CI]        "
        f"exempt - unpruned (mean / median)")
    for n in list(ex):
        d1, d2, d3 = ci(ex[n] - own[n]), ci(ex[n] - ship), ci(ex[n] - b)
        deg = float(np.mean([raw[n][c]["coc_degenerate"] for c in ids]))
        metrics["arms"][n] = {"own": float(own[n].mean()), "exempt": float(ex[n].mean()),
                              "own_median": float(np.median(own[n])),
                              "exempt_median": float(np.median(ex[n])), "degen": deg,
                              "exempt_minus_own": d1, "exempt_minus_shipped": d2,
                              "exempt_minus_unpruned": d3,
                              "minfde": float(vec(raw[n], "fde_rollout_k").mean())}
        say(f"{n:6s}{own[n].mean():8.4f}{ex[n].mean():8.4f}{np.median(own[n]):9.4f}"
            f"{np.median(ex[n]):8.4f}{deg:7.3f}   {f(d1)} {d1['p']:.1e}   {f(d2)}   "
            f"{d3['mean']:+.4f} / {d3['median']:+.4f}")

    names = list(ex)
    for label, src in (("own", own), ("exempt", ex)):
        m = [float(src[n].mean()) for n in names]
        md = [float(np.median(src[n])) for n in names]
        metrics[f"spread_{label}"] = {"sd_means": float(np.std(m, ddof=1)),
                                      "range_means": [min(m), max(m)],
                                      "sd_medians": float(np.std(md, ddof=1))}
        say(f"\n{label:7s} across {len(names)} arms: means {min(m):.4f}-{max(m):.4f}, "
            f"SD {np.std(m, ddof=1):.4f}; SD of medians {np.std(md, ddof=1):.4f}")

    say("\n== by bucket: mean minADE@6 minus unpruned, own -> exempt")
    say(f"{'arm':6s}" + "".join(f"{bk + f' ({int((bucket == bk).sum())})':>24s}" for bk in BUCKETS))
    metrics["by_bucket"] = {}
    for n in names:
        row = {}
        cells = []
        for bk in BUCKETS:
            s = bucket == bk
            a, e = float((own[n] - b)[s].mean()), float((ex[n] - b)[s].mean())
            row[bk] = [a, e]
            cells.append(f"{a:+.3f} -> {e:+.3f}")
        metrics["by_bucket"][n] = row
        say(f"{n:6s}" + "".join(f"{c:>24s}" for c in cells))

    # pairwise among exempt arms: are they one model now?
    say("\n== exempt arms against each other: paired mean difference (row - column)")
    say(f"{'':6s}" + "".join(f"{n:>9s}" for n in names))
    pw = {}
    for a in names:
        cells = []
        for c in names:
            if a == c:
                cells.append(f"{'':>9s}")
                continue
            d = ex[a] - ex[c]
            pw[f"{a}-{c}"] = float(d.mean())
            cells.append(f"{d.mean():+9.3f}")
        say(f"{a:6s}" + "".join(cells))
    metrics["pairwise_exempt"] = pw
    worst = max(pw.items(), key=lambda kv: abs(kv[1]))
    d = ci(ex[worst[0].split("-")[0]] - ex[worst[0].split("-")[1]])
    metrics["largest_pair"] = {worst[0]: d}
    say(f"largest pair {worst[0]}: {f(d)} p={d['p']:.1e}")

    # ---- budget-matched variants: layer 0 kept in full, 1,109 more MLP channels cut ----
    mx, mraw = {}, {}
    for n in names:
        rows = load_run(f"swap_{n}_L0fullM_500")
        if all(c in rows for c in ids):
            mx[n], mraw[n] = vec(rows), rows
    if mx:
        say("\n== budget-matched (layer 0 kept, 1,109 more MLP channels cut; 4,096 params "
            "from the source arm's budget)")
        say(f"{'arm':6s}{'own':>8s}{'exempt':>8s}{'matched':>9s}{'med':>8s}{'degen':>7s}   "
            f"matched - exempt [95% CI]        p        matched - own [95% CI]           "
            f"matched - shipped [95% CI]")
        metrics["matched"] = {}
        for n in list(mx):
            d1, d2, d3 = ci(mx[n] - ex[n]), ci(mx[n] - own[n]), ci(mx[n] - ship)
            deg = float(np.mean([mraw[n][c]["coc_degenerate"] for c in ids]))
            metrics["matched"][n] = {"mean": float(mx[n].mean()),
                                     "median": float(np.median(mx[n])), "degen": deg,
                                     "matched_minus_exempt": d1, "matched_minus_own": d2,
                                     "matched_minus_shipped": d3,
                                     "matched_minus_unpruned": ci(mx[n] - b),
                                     "minfde": float(vec(mraw[n], "fde_rollout_k").mean())}
            say(f"{n:6s}{own[n].mean():8.4f}{ex[n].mean():8.4f}{mx[n].mean():9.4f}"
                f"{np.median(mx[n]):8.4f}{deg:7.3f}   {f(d1)} {d1['p']:.1e}   {f(d2)}   {f(d3)}")
        mn = list(mx)
        m = [float(mx[n].mean()) for n in mn]
        md = [float(np.median(mx[n])) for n in mn]
        metrics["spread_matched"] = {"sd_means": float(np.std(m, ddof=1)),
                                     "range_means": [min(m), max(m)],
                                     "sd_medians": float(np.std(md, ddof=1))}
        say(f"matched across {len(mn)} arms: means {min(m):.4f}-{max(m):.4f}, "
            f"SD {np.std(m, ddof=1):.4f}; SD of medians {np.std(md, ddof=1):.4f}")
        # what the 1,109 channels cost, pooled: per-clip mean over arms of (matched - exempt)
        pooled = np.mean([mx[n] - ex[n] for n in mn], axis=0)
        s = ci(pooled)
        metrics["matched_minus_exempt_pooled"] = s
        say(f"cost of the 1,109 channels, per-clip mean over arms: {f(s)} "
            f"median {s['median']:+.4f} p={s['p']:.1e}")
        say("\nby bucket, matched: mean minADE@6 minus unpruned  (and matched - shipped [95% CI])")
        metrics["by_bucket_matched"] = {}
        for n in mn:
            cells, row = [], {}
            for bkt in BUCKETS:
                sel = bucket == bkt
                d = ci((mx[n] - ship)[sel])
                row[bkt] = {"vs_unpruned": float((mx[n] - b)[sel].mean()), "vs_shipped": d}
                cells.append(f"{(mx[n] - b)[sel].mean():+.3f} ({d['mean']:+.3f}"
                             f"{'*' if d['ci'][0] > 0 or d['ci'][1] < 0 else ' '})")
            metrics["by_bucket_matched"][n] = row
            say(f"{n:6s}" + "".join(f"{c:>20s}" for c in cells))
        say(f"{'shipped':6s}" + "".join(
            f"{(ship - b)[bucket == bkt].mean():+20.3f}" for bkt in BUCKETS))
        # the stop bucket of the shipped arm, flagged in the unmatched batch
        for lab, src in (("exempt", ex), ("matched", mx)):
            if "ship" in src:
                d = ci((src["ship"] - ship)[bucket == "decel_stop"])
                metrics[f"ship_stop_{lab}"] = d
                say(f"shipped arm, decel_stop, {lab} - as built: {f(d)} p={d['p']:.2g}")

        fig, ax = plt.subplots(figsize=(10, 4.4))
        xs = np.arange(len(mn))
        ax.bar(xs - 0.27, [own[n].mean() for n in mn], width=0.26, color=C3, label="as built")
        ax.bar(xs, [ex[n].mean() for n in mn], width=0.26, color=C1,
               label="layer 0 kept (0.51% less removed)")
        ax.bar(xs + 0.27, [mx[n].mean() for n in mn], width=0.26, color=C4,
               label="layer 0 kept, budget matched")
        ax.axhline(b.mean(), color=INK, lw=0.9, ls="--", label="unpruned")
        ax.axhline(ship.mean(), color=MUTED, lw=0.9, ls=":", label="shipped")
        ax.set_xticks(xs, mn)
        ax.set_ylabel("test500 minADE@6, mean")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "plots" / "l0_exempt_matched.png", dpi=150)
        plt.close(fig)

    (out / "metrics_exempt.json").write_text(json.dumps(metrics, indent=2))
    (out / "summary_exempt.txt").write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    xs = np.arange(len(names))
    for ax, key, title in ((axes[0], "mean", "mean"), (axes[1], "median", "median")):
        fn = np.mean if key == "mean" else np.median
        ax.bar(xs - 0.2, [fn(own[n]) for n in names], width=0.38, color=C3, label="as built")
        ax.bar(xs + 0.2, [fn(ex[n]) for n in names], width=0.38, color=C1,
               label="layer-0 heads all kept")
        ax.axhline(fn(b), color=INK, lw=0.9, ls="--", label="unpruned")
        ax.axhline(fn(ship), color=MUTED, lw=0.9, ls=":", label="shipped")
        ax.set_xticks(xs, names)
        ax.set_title(f"test500 minADE@6, {title}")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "plots" / "l0_exempt.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
