"""How much does the calibration draw move a pruned model? (G0b/G1/G2/G3/G4)

`2026-08-19_calibration-source.html` had one in-distribution control and put +0.0941 on
the draw. This re-reads that comparison under the current frozen protocol (rollout-only,
minADE@6) and adds five disjoint official-train blocks, so the point estimate becomes a
spread. Every arm is the same `dual_u40_v2` recipe -- same criterion, same
0.3985632694 budget, same 2,657,452,032 removed parameters, VLM only -- and differs only
in the clips the Taylor scores were measured on.

The three 2026-08-19 arms are re-scored, not re-run: `dual_u40_ctl/ood/mix_test` all
carry `ade_rollout_k` with k=8 at seed 42 on Ada with `deterministic`, the same protocol
as `dual_u40_v2_ps_test`, so @6 is a re-read of stored rows and costs no GPU.

Arms not yet evaluated are skipped, so this runs while the queue is still filling.

Usage:
  python experiments/evaluation/analyze_calib_variance.py [--out outputs/calib_variance]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, wilcoxon

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))

import paper_numbers as pn

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})

# Two block families, and they are not interchangeable:
#   nt  natural official train, t0=5.1 s -- drawn the way calib_100 was, so nt vs
#       calib_100 is the one-factor draw contrast this study is about
#   tr  the first pass, drawn from pre_processed/train, which turned out to be a hard
#       scenario-balanced recovery mix (minADE 2.20 vs 0.84 natural, 20% per bucket).
#       Kept because it answers a different question -- what calibrating on long-tail
#       clips costs -- not because it is a draw replicate.
def _family(tag, label, n_map):
    out = {}
    for b in "abcde":
        out[f"{tag}_{b}"] = (f"{label} {b}", 100,
                             {"test": f"dual_{tag}_{b}_test",
                              "indist": f"dual_{tag}_{b}_indist"})
    for k, lab in n_map.items():
        out[f"{tag}_{k}"] = (f"{label} {lab}", k, {"test": f"dual_{tag}_c{k}_test"})
    return out


LADDER = {200: "a+b", 300: "a+b+c", 500: "a..e"}
ARMS = {
    "baseline":     ("무압축",                   None, {"test": "baseline_ada_ps_test",
                                                        "indist": "baseline_ada_ps_indist"}),
    "calib_100":    ("calib_100 (train, BW)",     100, {"test": "dual_u40_v2_ps_test",
                                                        "indist": "dual_u40_v2_ps_indist"}),
    "calib_100_ada": ("calib_100 (train, Ada)",   100, {"test": "dual_u40_v2_ada_test"}),
    **_family("nt", "자연추출", LADDER),
    **_family("tr", "하드풀", LADDER),
    # the 2026-08-19 arms, re-scored at @6
    "calib_val100": ("calib_val100 (val)",        100, {"test": "dual_u40_ctl_test"}),
    "calib_ood100": ("calib_ood_100 (OOD)",       100, {"test": "dual_u40_ood_test"}),
    "pooled_200":   ("pooled 200 (in+OOD)",       200, {"test": "dual_u40_mix_test"}),
}
FAMILY = "nt"                              # which family G1-G4 judge
BLOCKS = [f"{FAMILY}_{b}" for b in "abcde"]
DRAWS100 = ["calib_100"] + BLOCKS          # the six n=100 draws G1 measures
LADDER_ARMS = [f"{FAMILY}_{k}" for k in LADDER]
TR_BLOCKS = [f"tr_{b}" for b in "abcde"]
EXPECT = {"test": 500, "indist": 500}


def load_arms(sets):
    rows = {}
    for arm, (_, _, dirs) in ARMS.items():
        for s in sets:
            if s not in dirs:
                continue
            try:
                r = pn.load(dirs[s], False)
            except (SystemExit, ValueError):
                continue
            if len(r) >= EXPECT[s]:
                rows.setdefault(arm, {})[s] = r
    return rows


def paired_delta(rows, a, b, s):
    """median (arm a - arm b) minADE@6 with a bootstrap CI and Wilcoxon."""
    if a not in rows or b not in rows or s not in rows[a] or s not in rows[b]:
        return None
    ids = sorted(set(rows[a][s]) & set(rows[b][s]))
    d = np.array([pn.at6(rows[a][s][i], "ade_rollout_k")
                  - pn.at6(rows[b][s][i], "ade_rollout_k") for i in ids])
    rng = np.random.default_rng(0)
    boot = [np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(10000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = float(wilcoxon(d).pvalue) if np.any(d) else 1.0
    return {"n": len(ids), "median": float(np.median(d)), "lo": float(lo),
            "hi": float(hi), "p": p, "sig": bool(lo > 0 or hi < 0)}


def mean_ade(rows, arm, s):
    return float(np.mean(pn.stats(rows[arm][s])[0])) if arm in rows and s in rows[arm] \
        else None


def kept_sets(importance_dirs):
    """dual_u40_v2 kept masks for each importance run, for the overlap analyses."""
    import mask_lib as ml
    import tyr_lib as tyr
    from make_slim import allocations

    ref = json.loads(
        (REPO / "outputs" / "slim_integrated_mag" / "slim_meta.json").read_text())
    out = {}
    for name, d in importance_dirs.items():
        p = REPO / "outputs" / d / "importance.npz"
        if not p.exists():
            continue
        imp = dict(np.load(p))
        allocs, _ = allocations(imp, ref, 36, 32, 12288, 0.5)
        rq, rm = allocs["uniform"]
        sq, sm = tyr.dual_scores(imp)
        out[name] = (ml.select_mask_ratios(sq, rq), ml.select_mask_ratios(sm, rm))
    return out


def overlap(x, y):
    return float((x * y).sum() / x.sum())


def stability_curve(importance, sizes, draws=20, seed=0):
    """Kept-set agreement between two DISJOINT subsets of n clips each.

    Two disjoint halves of one run stand in for two independent draws of that size, so
    this is the block-to-block overlap at a fraction of the cost -- and it reaches sizes
    no pair of real blocks could without more runs. `mean(per-clip) == importance.npz`
    holds to 0.0 on every array, so a subset's mean is exactly the mean a separate run
    over those clips would have produced.
    """
    import mask_lib as ml
    import tyr_lib as tyr
    from make_slim import allocations

    p = REPO / "outputs" / importance / "importance_perclip.npz"
    if not p.exists():
        return {}
    per = dict(np.load(p))
    full = dict(np.load(REPO / "outputs" / importance / "importance.npz"))
    ref = json.loads(
        (REPO / "outputs" / "slim_integrated_mag" / "slim_meta.json").read_text())
    allocs, _ = allocations(full, ref, 36, 32, 12288, 0.5)
    rq, rm = allocs["uniform"]
    n_clips = next(iter(per.values())).shape[0]

    def masks(idx):
        imp = {k: v[idx].astype(np.float64).mean(axis=0) for k, v in per.items()}
        sq, sm = tyr.dual_scores(imp)
        return ml.select_mask_ratios(sq, rq), ml.select_mask_ratios(sm, rm)

    rng = np.random.default_rng(seed)
    out = {}
    for n in sizes:
        if 2 * n > n_clips:
            continue
        q, m = [], []
        for _ in range(draws):
            pm = rng.permutation(n_clips)
            a, b = masks(pm[:n]), masks(pm[n:2 * n])
            q.append(overlap(a[0], b[0]))
            m.append(overlap(a[1], b[1]))
        out[str(n)] = {"q": float(np.mean(q)), "q_sd": float(np.std(q)),
                       "mlp": float(np.mean(m)), "mlp_sd": float(np.std(m))}
        print(f"  n={n:4d}  Q {np.mean(q):.4f}+-{np.std(q):.4f}  "
              f"MLP {np.mean(m):.4f}+-{np.std(m):.4f}", flush=True)
    return out


def plot_stability(curve, card_only, out):
    if not curve:
        return
    ns = sorted(int(k) for k in curve)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for key, c, lab in (("q", C1, "Q head"), ("mlp", C2, "MLP channel")):
        y = [curve[str(n)][key] for n in ns]
        e = [curve[str(n)][f"{key}_sd"] for n in ns]
        ax.errorbar(ns, y, yerr=e, fmt="o-", color=c, capsize=3, label=lab)
    if card_only:
        ax.axhline(card_only, color=MUTED, ls="--", lw=1,
                   label=f"card change only {card_only:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("clips per draw (two disjoint draws)")
    ax.set_ylabel("kept-set overlap")
    ax.set_title("selection converges slowly in the number of clips")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "stability.png", dpi=150)
    plt.close(fig)


def plot_draws(rows, out):
    """The six n=100 draws on one axis, with the noise floor marked."""
    have = [a for a in DRAWS100 if a in rows and "test" in rows[a]]
    if len(have) < 2:
        return
    vals = [mean_ade(rows, a, "test") for a in have]
    base = mean_ade(rows, "baseline", "test")
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(range(len(have)), vals, color=[C1 if a == "calib_100" else C2 for a in have])
    if base:
        ax.axhline(base, color=MUTED, ls="--", lw=1, label=f"unpruned {base:.4f}")
    ax.axhline(float(np.mean(vals)), color=C4, lw=1,
               label=f"draw mean {np.mean(vals):.4f} (SD {np.std(vals, ddof=1):.4f})")
    ax.set_xticks(range(len(have)))
    ax.set_xticklabels([ARMS[a][0] for a in have], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("test500 minADE@6")
    ax.set_ylim(min(vals) - 0.05, max(vals) + 0.05)
    ax.set_title("same recipe, only the 100 calibration clips differ")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "draws.png", dpi=150)
    plt.close(fig)


def plot_ladder(rows, out):
    pts = [(ARMS[a][1], mean_ade(rows, a, "test"))
           for a in LADDER_ARMS if a in rows and "test" in rows[a]]
    blocks = [mean_ade(rows, a, "test") for a in BLOCKS if a in rows and "test" in rows[a]]
    if not pts or not blocks:
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.errorbar([100], [np.mean(blocks)],
                yerr=[[np.std(blocks, ddof=1)], [np.std(blocks, ddof=1)]],
                fmt="o", color=C2, capsize=4, label="n=100 block mean +- SD")
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=C1, label="cumulative ladder")
    ax.set_xlabel("calibration clips")
    ax.set_ylabel("test500 minADE@6")
    ax.set_title("does more calibration data help?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "ladder.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/calib_variance")
    ap.add_argument("--importance", default="importance_tr500")
    args = ap.parse_args()
    out = REPO / args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)

    rows = load_arms(("test", "indist"))
    print("arms with complete rows:",
          {a: sorted(v) for a, v in sorted(rows.items())}, flush=True)

    m = {"absolute": {}, "vs_baseline": {}, "gates": {}, "pairs": {}, "overlap": {},
         "anchor_2026_08_19": {}}
    for arm in ARMS:
        for s in ("test", "indist"):
            v = mean_ade(rows, arm, s)
            if v is not None:
                m["absolute"].setdefault(arm, {})[s] = v
                d = paired_delta(rows, arm, "baseline", s)
                if d:
                    m["vs_baseline"].setdefault(arm, {})[s] = d

    # G0b: same clips, same seed, different card
    m["gates"]["G0b_card"] = paired_delta(rows, "calib_100_ada", "calib_100", "test")

    # the 2026-08-19 contrasts, restated at @6 so the +0.0941 anchor sits on this axis
    for arm in ("calib_val100", "calib_ood100", "pooled_200"):
        d = paired_delta(rows, arm, "calib_100", "test")
        if d:
            m["anchor_2026_08_19"][arm] = d

    # G1: spread of the six n=100 draws, and all 15 pairwise contrasts
    have = [a for a in DRAWS100 if a in rows and "test" in rows[a]]
    if len(have) >= 2:
        vals = [mean_ade(rows, a, "test") for a in have]
        m["gates"]["G1_spread"] = {
            "draws": have, "minADE": vals, "sd": float(np.std(vals, ddof=1)),
            "range": float(max(vals) - min(vals)),
        }
        for i, a in enumerate(have):
            for b in have[i + 1:]:
                m["pairs"].setdefault(a, {})[b] = paired_delta(rows, a, b, "test")

    # G1b: does the spread reproduce on the second set? val500 is drawn from official
    # val, disjoint from test_500 and from every calibration set, so it is an
    # independent read of the same six models -- not a re-test of the same clips.
    have_v = [a for a in DRAWS100 if a in rows and "indist" in rows[a]]
    if len(have_v) >= 2:
        vals_v = [mean_ade(rows, a, "indist") for a in have_v]
        m["gates"]["G1b_spread_val500"] = {
            "draws": have_v, "minADE": vals_v, "sd": float(np.std(vals_v, ddof=1)),
            "range": float(max(vals_v) - min(vals_v)),
        }
        for a in have_v:
            d = paired_delta(rows, a, "calib_100", "indist")
            if d:
                m["pairs"].setdefault(a, {})["calib_100|val500"] = d
        # the two sets must rank the draws the same way, or "the draw matters" is a
        # statement about test_500 rather than about the model
        common = [a for a in have_v if a in have]
        if len(common) >= 3:
            t = [mean_ade(rows, a, "test") for a in common]
            v = [mean_ade(rows, a, "indist") for a in common]
            r = spearmanr(t, v)
            m["gates"]["G1b_spread_val500"]["rank_agreement"] = {
                "arms": common, "spearman": float(r.statistic), "p": float(r.pvalue)}

    # G2: is calib_100 a lucky draw?
    if "G1_spread" in m["gates"]:
        vals = m["gates"]["G1_spread"]["minADE"]
        if "calib_100" in have:
            m["gates"]["G2_rank_of_calib_100"] = {
                "rank": int(np.argsort(np.argsort(vals))[have.index("calib_100")]) + 1,
                "of": len(have)}
        blocks = [mean_ade(rows, a, "test") for a in BLOCKS if a in rows and "test" in rows[a]]
        if blocks and "calib_100" in have:
            m["gates"]["G2_blockmean_minus_calib100"] = \
                float(np.mean(blocks) - mean_ade(rows, "calib_100", "test"))

    # G3: does n help?
    blocks = [mean_ade(rows, a, "test") for a in BLOCKS if a in rows and "test" in rows[a]]
    if blocks and f"{FAMILY}_500" in rows:
        m["gates"]["G3_ladder"] = {
            "block_mean": float(np.mean(blocks)),
            "block_sd": float(np.std(blocks, ddof=1)),
            "n500": mean_ade(rows, f"{FAMILY}_500", "test"),
            "gain": float(np.mean(blocks) - mean_ade(rows, f"{FAMILY}_500", "test")),
        }

    # H5 (not pre-registered, found while diagnosing the first pass): calibrating on the
    # hard scenario-balanced recovery mix. Reported against the natural blocks at the
    # same n, and against the OOD arm, because the interesting claim is that "OOD
    # calibration is worse" may really be "hard-clip calibration is worse".
    tr = [mean_ade(rows, a, "test") for a in TR_BLOCKS if a in rows and "test" in rows[a]]
    if tr:
        m["gates"]["H5_hard_pool"] = {
            "n_blocks": len(tr), "mean": float(np.mean(tr)),
            "sd": float(np.std(tr, ddof=1)) if len(tr) > 1 else None,
            "vs_natural_blocks": (float(np.mean(tr) - np.mean(blocks)) if blocks else None),
            "ood_arm": mean_ade(rows, "calib_ood100", "test"),
        }
        for a in TR_BLOCKS:
            d = paired_delta(rows, a, "calib_100", "test")
            if d:
                m["pairs"].setdefault(a, {})["calib_100"] = d
        # the same n-ladder on the hard pool. Cumulative unions, so a rung differs from
        # its neighbour in BOTH the clip count and which blocks are in it -- read the
        # direction, not the slope.
        lad = {k: mean_ade(rows, f"tr_{k}", "test") for k in LADDER
               if f"tr_{k}" in rows and "test" in rows[f"tr_{k}"]}
        if lad:
            m["gates"]["H5_hard_pool"]["ladder"] = lad
            m["gates"]["H5_hard_pool"]["ladder_vs_block_mean"] = {
                str(k): float(v - np.mean(tr)) for k, v in lad.items()}
            for k in lad:
                d = paired_delta(rows, f"tr_{k}", "tr_200", "test")
                if d:
                    m["pairs"].setdefault(f"tr_{k}", {})["tr_200"] = d

    # G4: does kept-set overlap predict the paired delta?
    imp_dirs = {"calib_100": "importance_v2", "calib_100_ada": "importance_v2_ada"}
    imp_dirs.update({f"{FAMILY}_{b}": f"{args.importance}_{b}" for b in "abcde"})
    imp_dirs.update({f"{FAMILY}_{k}": f"{args.importance}_c{k}" for k in LADDER})
    masks = kept_sets(imp_dirs)
    ov, dl = [], []
    for i, a in enumerate(have):
        for b in have[i + 1:]:
            if a in masks and b in masks and b in m["pairs"].get(a, {}):
                o = 0.5 * (overlap(masks[a][0], masks[b][0])
                           + overlap(masks[a][1], masks[b][1]))
                m["overlap"].setdefault(a, {})[b] = o
                ov.append(o)
                dl.append(abs(m["pairs"][a][b]["median"]))
    if len(ov) >= 4:
        r = spearmanr(ov, dl)
        m["gates"]["G4_overlap_vs_delta"] = {"n": len(ov), "spearman": float(r.statistic),
                                             "p": float(r.pvalue)}
    if "calib_100" in masks and "calib_100_ada" in masks:
        m["overlap"]["card_only"] = 0.5 * (
            overlap(masks["calib_100"][0], masks["calib_100_ada"][0])
            + overlap(masks["calib_100"][1], masks["calib_100_ada"][1]))

    print("stability curve (disjoint draws of n clips each):", flush=True)
    m["stability"] = stability_curve(args.importance, (10, 25, 50, 100, 150, 250))

    plot_draws(rows, out / "plots")
    plot_ladder(rows, out / "plots")
    plot_stability(m["stability"], m["overlap"].get("card_only"), out / "plots")
    (out / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False))

    lines = ["== test500 minADE@6, dual_u40_v2 with the calibration clips varied =="]
    for arm, (label, n, _) in ARMS.items():
        v = m["absolute"].get(arm, {}).get("test")
        if v is None:
            continue
        d = m["vs_baseline"].get(arm, {}).get("test")
        tail = (f"  vs baseline {d['median']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}]"
                f"{'*' if d['sig'] else ' '}") if d else ""
        lines.append(f"  {label:26s} n={n!s:>4s}  {v:.4f}{tail}")
    lines.append("-- 2026-08-19 contrasts vs calib_100, restated at @6 --")
    for k, d in m["anchor_2026_08_19"].items():
        lines.append(f"  {k:32s} {d['median']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}]"
                     f"{'*' if d['sig'] else ' '} p={d['p']:.2g}")
    for k, v in m["gates"].items():
        lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print("->", out, flush=True)


if __name__ == "__main__":
    main()
