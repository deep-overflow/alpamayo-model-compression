"""Does keeping all of layer 0's Q heads remove the draw / seed collapses?

`make_l0_exempt.py` variants of eight dual_u40_v2 kept sets -- the shipped arm, three seed
arms on calib_100 (ref, c3, n1) and four other calibration blocks (nt_b..e):

  exempt    layer 0's 32 Q heads restored, nothing else changed (0.51% fewer params removed)
  matched   the same, with 1,109 more MLP channels cut to pay the budget back

Each is paired, clip by clip, with the unpruned model and the shipped arm on the same set,
and with its own source arm where an evaluation of that arm is stored (all eight on
test500; the shipped arm and the nt blocks on val500; the shipped arm only on OOD-val).

Pre-stated readings (test500; val500 and OOD-val are the replication):
  H1  the spread across the eight arms shrinks: SD of arm means was 0.34 before.
  H2  the collapsed arms (n1, nt_c, nt_d, nt_e) land within the pruning cost of the
      shipped arm.
  H3  the shipped arm itself does not get worse.

Usage:
  python experiments/evaluation/analyze_l0_exempt.py [--set test|indist|ood]
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
BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C3, C4 = "#2a78d6", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})
NAMES = ["ship", "ref", "c3", "n1", "ntb", "ntc", "ntd", "nte"]
SETS = {
    "test": {"label": "test500", "base": "baseline_ada_ps_test", "ship": "dual_u40_v2_ps_test",
             "suffix": "500", "n": 500,
             "own": {"ship": "dual_u40_v2_ps_test", "ref": "dual_seed_ref_test",
                     "c3": "dual_seed_c3_test", "n1": "dual_seed_n1_test",
                     **{f"nt{x}": f"dual_nt_{x}_test" for x in "bcde"}}},
    "indist": {"label": "val500", "base": "baseline_ada_ps_indist",
               "ship": "dual_u40_v2_ps_indist", "suffix": "val500", "n": 500,
               "own": {"ship": "dual_u40_v2_ps_indist",
                       **{f"nt{x}": f"dual_nt_{x}_indist" for x in "bcde"}}},
    # the shipped arm's OOD run covers all 1,533 clips; rows are indexed by clip id, so
    # pairing on the 262 OOD-val clips of the baseline run needs no separate file
    "ood": {"label": "OOD-val", "base": "baseline_ada_ps_oodval", "ship": "dual_u40_v2_ps_ood",
            "suffix": "oodval", "n": 262, "own": {"ship": "dual_u40_v2_ps_ood"}},
}
BUCKETS = ["cruise", "decel_stop", "accel", "turn"]
BOOT, K = 20000, 6


def load_run(exp):
    rows = {}
    for p in sorted((REPO / "outputs" / exp).glob("*.json")):
        if not p.name.startswith("config"):
            for r in json.loads(p.read_text()):
                rows[r["clip_id"]] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="which", default="test", choices=list(SETS))
    args = ap.parse_args()
    S = SETS[args.which]
    out = REPO / "outputs" / "swap_masks"
    (out / "plots").mkdir(parents=True, exist_ok=True)
    tag = "" if args.which == "test" else f"_{S['suffix']}"
    rng = np.random.default_rng(0)
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    base = load_run(S["base"])
    ids = sorted(base)
    assert len(ids) == S["n"], len(ids)
    bucket = np.array([base[c]["bucket"] for c in ids])

    def vec(rows, key="ade_rollout_k"):
        return np.array([min(rows[c][key][:K]) for c in ids])

    def ci(d):
        idx = rng.integers(0, len(d), (BOOT, len(d)))
        lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
        nz = d[d != 0]
        return {"mean": float(d.mean()), "ci": [float(lo), float(hi)],
                "median": float(np.median(d)),
                "p": float(wilcoxon(nz).pvalue) if len(nz) > 10 else 1.0}

    def f(s):
        star = "*" if s["ci"][0] > 0 or s["ci"][1] < 0 else " "
        return f"{s['mean']:+.4f} [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}]{star}"

    def complete(exp):
        p = REPO / "outputs" / exp
        rows = load_run(exp) if p.exists() else {}
        return rows if all(c in rows for c in ids) else None

    b, ship = vec(base), vec(load_run(S["ship"]))
    own, var, raw = {}, {"exempt": {}, "matched": {}}, {"exempt": {}, "matched": {}}
    for n in NAMES:
        if n in S["own"]:
            r = complete(S["own"][n])
            if r is not None:
                own[n] = vec(r)
        for kind, suf in (("exempt", "L0full"), ("matched", "L0fullM")):
            r = complete(f"swap_{n}_{suf}_{S['suffix']}")
            if r is not None:
                var[kind][n], raw[kind][n] = vec(r), r
    metrics = {"set": S["label"], "n_clips": S["n"], "unpruned": float(b.mean()),
               "shipped": float(ship.mean()), "arms": {}, "spread": {}}
    say(f"{S['label']}: unpruned {b.mean():.4f} (median {np.median(b):.4f}); shipped "
        f"{ship.mean():.4f} (median {np.median(ship):.4f}); pruning cost "
        f"{f(ci(ship - b))} median {np.median(ship - b):+.4f}")

    def num(x):
        return f"{x:8.4f}" if x is not None else f"{'-':>8s}"

    say(f"\n== {S['label']} minADE@6")
    say(f"{'arm':6s}{'own':>8s}{'exempt':>8s}{'matched':>8s}{'med m':>8s}{'degen':>7s}   "
        f"matched - shipped [95% CI]        matched - own [95% CI]           "
        f"matched - exempt [95% CI]")
    for n in NAMES:
        o, e, m = own.get(n), var["exempt"].get(n), var["matched"].get(n)
        rec = {"own": None if o is None else float(o.mean()),
               "exempt": None if e is None else float(e.mean()),
               "matched": None if m is None else float(m.mean())}
        cells = []
        best = m if m is not None else e
        if best is None:
            continue
        kind = "matched" if m is not None else "exempt"
        rec["degen"] = float(np.mean([raw[kind][n][c]["coc_degenerate"] for c in ids]))
        rec["median"] = float(np.median(best))
        rec["vs_shipped"] = ci(best - ship)
        rec["vs_unpruned"] = ci(best - b)
        cells.append(f(rec["vs_shipped"]))
        if o is not None:
            rec["vs_own"] = ci(best - o)
            cells.append(f(rec["vs_own"]))
        else:
            cells.append(f"{'-':^29s}")
        if m is not None and e is not None:
            rec["matched_minus_exempt"] = ci(m - e)
            cells.append(f(rec["matched_minus_exempt"]))
        if e is not None:
            rec["exempt_vs_shipped"] = ci(e - ship)
        metrics["arms"][n] = rec
        say(f"{n:6s}{num(rec['own'])}{num(rec['exempt'])}{num(rec['matched'])}"
            f"{rec['median']:8.4f}{rec['degen']:7.3f}   " + "   ".join(cells))

    for label, src in (("own", own), ("exempt", var["exempt"]), ("matched", var["matched"])):
        if len(src) < 3:
            continue
        m = [float(v.mean()) for v in src.values()]
        md = [float(np.median(v)) for v in src.values()]
        metrics["spread"][label] = {"n": len(m), "sd_means": float(np.std(m, ddof=1)),
                                    "range_means": [min(m), max(m)],
                                    "sd_medians": float(np.std(md, ddof=1))}
        say(f"{label:8s} across {len(m)} arms: means {min(m):.4f}-{max(m):.4f}, "
            f"SD {np.std(m, ddof=1):.4f}; SD of medians {np.std(md, ddof=1):.4f}")
    mx, ex = var["matched"], var["exempt"]
    both = [n for n in mx if n in ex]
    if both:
        s = ci(np.mean([mx[n] - ex[n] for n in both], axis=0))
        metrics["matched_minus_exempt_pooled"] = s
        say(f"cost of the 1,109 channels, per-clip mean over {len(both)} arms: {f(s)} "
            f"median {s['median']:+.4f}")

    final = mx if mx else ex
    say(f"\n== by bucket ({'matched' if mx else 'exempt'}): mean minADE@6 minus unpruned, "
        "with (variant - shipped) and * where its 95% CI excludes 0")
    say(f"{'arm':8s}" + "".join(f"{bk + f' ({int((bucket == bk).sum())})':>22s}" for bk in BUCKETS))
    metrics["by_bucket"] = {}
    n_sig = 0
    for n, v in final.items():
        cells, row = [], {}
        for bk in BUCKETS:
            sel = bucket == bk
            d = ci((v - ship)[sel])
            sig = d["ci"][0] > 0 or d["ci"][1] < 0
            n_sig += sig and n != "ship"
            row[bk] = {"vs_unpruned": float((v - b)[sel].mean()), "vs_shipped": d}
            cells.append(f"{(v - b)[sel].mean():+.3f} ({d['mean']:+.3f}{'*' if sig else ' '})")
        metrics["by_bucket"][n] = row
        say(f"{n:8s}" + "".join(f"{c:>22s}" for c in cells))
    say(f"{'shipped':8s}" + "".join(f"{(ship - b)[bucket == bk].mean():+22.3f}" for bk in BUCKETS))
    if own:
        say("own arms, minus unpruned:")
        for n, v in own.items():
            say(f"{n:8s}" + "".join(f"{(v - b)[bucket == bk].mean():+22.3f}" for bk in BUCKETS))
    metrics["n_bucket_cells_sig_vs_shipped"] = int(n_sig)

    if len(final) > 1:
        names = list(final)
        pw = {f"{a}-{c}": float((final[a] - final[c]).mean()) for a in names for c in names if a < c}
        worst = max(pw.items(), key=lambda kv: abs(kv[1]))
        a, c = worst[0].split("-")
        d = ci(final[a] - final[c])
        metrics["largest_pair"] = {worst[0]: d}
        say(f"\nlargest pairwise difference among the variants, {worst[0]}: {f(d)} p={d['p']:.1e}")

    (out / f"metrics_exempt{tag}.json").write_text(json.dumps(metrics, indent=2))
    (out / f"summary_exempt{tag}.txt").write_text("\n".join(lines) + "\n")

    names = [n for n in NAMES if n in final]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    xs = np.arange(len(names))
    for off, src, col, lab in ((-0.27, own, C3, "as built"),
                               (0.0, ex, C1, "layer 0 kept (0.51% less removed)"),
                               (0.27, mx, C4, "layer 0 kept, budget matched")):
        pts = [(x + off, src[n].mean()) for x, n in zip(xs, names) if n in src]
        if pts:
            ax.bar([p[0] for p in pts], [p[1] for p in pts], width=0.26, color=col, label=lab)
    ax.axhline(b.mean(), color=INK, lw=0.9, ls="--", label="unpruned")
    ax.axhline(ship.mean(), color=MUTED, lw=0.9, ls=":", label="shipped")
    ax.set_xticks(xs, names)
    ax.set_ylabel(f"{S['label']} minADE@6, mean")
    ax.set_ylim(bottom=min(0.7, b.mean() - 0.1))
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "plots" / f"l0_exempt_matched{tag}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
