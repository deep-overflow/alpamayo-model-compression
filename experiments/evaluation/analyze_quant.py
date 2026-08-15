"""Read the quantization arms and decide whether the CoC criterion beats uniform bits.

Three arms over the same VLM pool (VLM text Linear + lm_head + ViT Linear, 8.155 B) at
the same projected storage budget, with the expert, `embed_tokens` and the action
projections left in bf16 in every arm:

  baseline     unquantized, re-measured on the same cards as the arms
  uniform_w8   every row at 8 bits -- the control
  qvla_coc_b8  rows allocated from the CoC-only directional sensitivity, budget matched
               to what uniform_w8 costs

The trajectory loss never enters the criterion, so a trajectory difference between the
two quantized arms is what a reasoning-only signal bought or cost on the driving task.

Gate G1 (pre-registered in plans/2026-08-13_qvla-coc-vlm-only.md): pooled val+test,
paired `qvla_coc_b8 - uniform_w8`. The primary reading is the **median** with a
bootstrap CI on the median, because minADE is heavy-tailed and one broken clip lands at
25 m; the mean and Wilcoxon are reported beside it, not trusted alone.

Both quantized arms are also read against the baseline, since "does an 8-bit budget cost
anything at all" is the first-order question and answers itself from those two contrasts.

Usage:
  python experiments/evaluation/analyze_quant.py
  python experiments/evaluation/analyze_quant.py --sets indist test   # before OOD lands
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
sys.path.insert(0, str(Path(__file__).parent))

import eval_lib as el
from analyze_baseline import BUCKETS, describe, load_rows

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})

GATE_G1_M = 0.05        # smallest minADE shift this protocol resolves
GATE_DEGEN = 0.05       # CoC severe-collapse threshold (baseline runs 0.006-0.008)

# arm tag -> (exp-id prefix written by launch_arms.sh, rows-file tag from run_baseline)
ARMS = {
    "baseline": ("baseline_ada_", "baseline"),
    "uniform_w8": ("uniform_w8_", "uniform_w8"),
    "qvla_coc_b8": ("qvla_coc_b8_", "qvla_coc_b8"),
    "uniform_w4": ("uniform_w4_", "uniform_w4"),
    "qvla_coc_b4": ("qvla_coc_b4_", "qvla_coc_b4"),
}
ARM_ORDER = ["baseline", "uniform_w8", "qvla_coc_b8", "uniform_w4", "qvla_coc_b4"]
COLORS = {"baseline": MUTED, "uniform_w8": C4, "qvla_coc_b8": C1,
          "uniform_w4": "#b07d00", "qvla_coc_b4": "#7b5bd6"}
# One budget per pair. `coc_vs_uniform` is Gate G1 at that budget -- same projected bytes,
# different distribution -- and the two vs-baseline rows say what the budget itself cost.
BUDGETS = [8, 4]
CONTRASTS = [(f"w{b}_{n}", a, c) for b in BUDGETS for n, a, c in (
    ("uniform_vs_baseline", "baseline", f"uniform_w{b}"),
    ("coc_vs_baseline", "baseline", f"qvla_coc_b{b}"),
    ("coc_vs_uniform", f"uniform_w{b}", f"qvla_coc_b{b}"),
)]


def median_ci(d, n_boot=10000, seed=0, alpha=0.05):
    """Bootstrap CI for the MEDIAN of a paired difference -- what Gate G1 reads."""
    d = np.asarray(d, dtype=float)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), size=(n_boot, len(d)))
    boots = np.median(d[idx], axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.median(d)), float(lo), float(hi)


def paired(a_rows, b_rows, metric):
    """Per-clip b - a over the clips both arms evaluated. Positive = b is worse."""
    a = {r["clip_id"]: r for r in a_rows}
    b = {r["clip_id"]: r for r in b_rows}
    ids = sorted(set(a) & set(b))
    ids = [i for i in ids if metric in a[i] and metric in b[i]]
    if not ids:
        return None
    d = np.array([b[i][metric] - a[i][metric] for i in ids])
    mean, mlo, mhi = el.paired_bootstrap_ci(d)
    med, dlo, dhi = median_ci(d)
    return {"n": len(ids), "mean": mean, "mean_ci": [mlo, mhi],
            "median": med, "median_ci": [dlo, dhi],
            "p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
            "frac_worse": float(np.mean(d > 0)), "frac_equal": float(np.mean(d == 0))}


def pooled(rows_by_set, sets, a, b, metric):
    merged = {arm: [r for s in sets if s in rows_by_set for r in rows_by_set[s].get(arm, [])]
              for arm in (a, b)}
    if not merged[a] or not merged[b]:
        return None
    return paired(merged[a], merged[b], metric)


def arm_stats(rows):
    out = {"n": len(rows)}
    for m in ("minADE_rollout", "minFDE_rollout", "nll_self"):
        out[m] = describe([r[m] for r in rows])
    if "minADE_tf" in rows[0]:
        for m in ("minADE_tf", "nll_gtcoc"):
            out[m] = describe([r[m] for r in rows])
    out["coc_degen"] = float(np.mean([r["coc_degenerate"] for r in rows]))
    out["coc_len_median"] = float(np.median([r["coc_len"] for r in rows]))
    out["by_bucket"] = {
        b: describe([r["minADE_rollout"] for r in rows if r["bucket"] == b]) or {}
        for b in BUCKETS}
    return out


def fmt(x, unit="m"):
    if x is None:
        return "n/a"
    return (f"med {x['median']:+.4f}{unit} [{x['median_ci'][0]:+.4f},{x['median_ci'][1]:+.4f}]"
            f"  mean {x['mean']:+.4f} [{x['mean_ci'][0]:+.4f},{x['mean_ci'][1]:+.4f}]"
            f"  p={x['p']:.2e}  worse {x['frac_worse'] * 100:.0f}%"
            f"  same {x['frac_equal'] * 100:.0f}%  n={x['n']}")


def quant_meta(arm):
    """The `quant` block run_baseline.py stored, so the report states the real budget."""
    for s in ("indist", "test", "ood"):
        p = REPO / "outputs" / f"{ARMS[arm][0]}{s}" / "config.json"
        if p.exists():
            q = json.loads(p.read_text()).get("quant")
            if q:
                return q
    return None


def plots(res, rows_by_set, sets, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = [a for a in ARM_ORDER if any(a in rows_by_set[s] for s in sets)]

    # 1. minADE and minFDE per arm per set
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    w, xs = 0.8 / max(len(arms), 1), np.arange(len(sets))
    for j, (metric, lab) in enumerate((("minADE_rollout", "median minADE@8 (m)"),
                                       ("minFDE_rollout", "median minFDE@8 (m)"))):
        for i, a in enumerate(arms):
            med = [res["per_arm"].get(s, {}).get(a, {}).get(metric, {}).get("median", np.nan)
                   for s in sets]
            axes[j].bar(xs + i * w, med, w, label=a, color=COLORS[a])
        axes[j].set_xticks(xs + w * (len(arms) - 1) / 2)
        axes[j].set_xticklabels(sets)
        axes[j].set_ylabel(lab)
    axes[0].set_title("Trajectory error, VLM-only quantization")
    axes[1].set_title("Final-point error")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "minade_by_arm.png", dpi=150)
    plt.close(fig)

    # 2. paired delta vs baseline, median with CI, for both error metrics
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for j, (metric, lab) in enumerate((("minADE_rollout", "minADE@8"),
                                       ("minFDE_rollout", "minFDE@8"))):
        ax = axes[j]
        labels, meds, los, his, cols = [], [], [], [], []
        for name, a, b in CONTRASTS:
            for s in sets:
                x = res["contrasts"].get(s, {}).get(name, {}).get(metric)
                if not x:
                    continue
                labels.append(f"{name}\n{s}")
                meds.append(x["median"])
                los.append(x["median"] - x["median_ci"][0])
                his.append(x["median_ci"][1] - x["median"])
                cols.append(C2 if name == "qvla_coc_vs_uniform" else MUTED)
        if labels:
            ax.errorbar(range(len(labels)), meds, yerr=[los, his], fmt="o", capsize=3,
                        ecolor=MUTED, mfc="none", linestyle="none")
            for i, c in enumerate(cols):
                ax.plot(i, meds[i], "o", color=c)
            ax.axhline(0, color=MUTED, lw=0.8)
            for y in (GATE_G1_M, -GATE_G1_M):
                ax.axhline(y, color=C3, lw=0.8, ls="--")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
            ax.set_ylabel(f"paired median d{lab} (m)")
            ax.set_title(f"{lab}: positive = second arm worse")
    fig.tight_layout()
    fig.savefig(out_dir / "paired_delta.png", dpi=150)
    plt.close(fig)

    # 3. reasoning channel: CoC degeneracy and NLL per arm
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for j, (key, lab) in enumerate((("coc_degen", "CoC degeneracy"),
                                    ("nll_self", "NLL of own CoC"))):
        for i, a in enumerate(arms):
            v = []
            for s in sets:
                st = res["per_arm"].get(s, {}).get(a)
                if not st:
                    v.append(np.nan)
                elif key == "coc_degen":
                    v.append(st["coc_degen"])
                else:
                    v.append(st["nll_self"]["median"])
            axes[j].bar(np.arange(len(sets)) + i * w, v, w, label=a, color=COLORS[a])
        axes[j].set_xticks(np.arange(len(sets)) + w * (len(arms) - 1) / 2)
        axes[j].set_xticklabels(sets)
        axes[j].set_title(lab)
        if key == "coc_degen":
            axes[j].axhline(GATE_DEGEN, color=C3, ls="--", lw=0.8)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "reasoning_channel.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["indist", "test", "ood"])
    ap.add_argument("--out", default="quant_summary")
    args = ap.parse_args()

    rows_by_set = {}
    for s in args.sets:
        rows_by_set[s] = {}
        for arm, (prefix, tag) in ARMS.items():
            d = REPO / "outputs" / f"{prefix}{s}"
            if not d.exists():
                continue
            r = load_rows(d, tag)
            if r:
                rows_by_set[s][arm] = r

    res = {"sets": args.sets, "per_arm": {}, "contrasts": {},
           "quant": {a: quant_meta(a) for a in ARM_ORDER if a != "baseline"}}
    for s in args.sets:
        res["per_arm"][s] = {a: arm_stats(r) for a, r in rows_by_set[s].items()}
        res["contrasts"][s] = {}
        for name, a, b in CONTRASTS:
            if a in rows_by_set[s] and b in rows_by_set[s]:
                res["contrasts"][s][name] = {
                    m: paired(rows_by_set[s][a], rows_by_set[s][b], m)
                    for m in ("minADE_rollout", "minFDE_rollout", "nll_self",
                              "minADE_tf", "nll_gtcoc")
                    if m in rows_by_set[s][a][0] and m in rows_by_set[s][b][0]}

    indist = [s for s in args.sets if s in ("indist", "test")]
    res["pooled_indist"] = {}
    for name, a, b in CONTRASTS:
        res["pooled_indist"][name] = {
            m: pooled(rows_by_set, indist, a, b, m)
            for m in ("minADE_rollout", "minFDE_rollout")}

    # Gate G1 is evaluated once per budget: same projected bytes, different distribution
    res["gates"] = {}
    for b in BUDGETS:
        g = res["pooled_indist"].get(f"w{b}_coc_vs_uniform", {}).get("minADE_rollout")
        res["gates"][f"G1_w{b}"] = None if not g else {
            "n": g["n"], "median": g["median"], "median_ci": g["median_ci"],
            "pass": bool(g["median"] <= -GATE_G1_M and g["median_ci"][1] < 0),
            "rule": f"median(qvla_coc_b{b} - uniform_w{b}) <= -{GATE_G1_M} m "
                    "AND bootstrap CI upper < 0",
        }
    degen = {s: {a: st["coc_degen"] for a, st in res["per_arm"][s].items()}
             for s in args.sets}
    res["gates"]["degeneracy_ok"] = all(v < GATE_DEGEN for s in degen for v in degen[s].values())

    out_dir = REPO / "outputs" / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2))
    plots(res, rows_by_set, args.sets, out_dir / "plots")

    L = []
    for a in ARM_ORDER:
        q = res["quant"].get(a)
        if q:
            L.append(f"{a}: {q['effective_bits']:.3f} eff bits over {q['pool_tensors']} "
                     f"tensors, pool {q['pool_bf16_gb']:.2f} -> {q['projected_pool_gb']:.2f} GB "
                     f"(projected), rows {q['row_bit_hist']}")
    for s in args.sets:
        L.append(f"\n=== {s} ===")
        for a in ARM_ORDER:
            st = res["per_arm"][s].get(a)
            if not st:
                continue
            m, f = st["minADE_rollout"], st["minFDE_rollout"]
            L.append(f"  {a:14s} n={st['n']:5d}  minADE med {m['median']:.4f} "
                     f"mean {m['mean']:.4f}  |  minFDE med {f['median']:.4f} "
                     f"mean {f['mean']:.4f}  |  NLL(own) med {st['nll_self']['median']:.4f}  "
                     f"degen {st['coc_degen']:.4f}")
            if "minADE_tf" in st:
                t = st["minADE_tf"]
                L.append(f"  {'':14s}          minADE_tf med {t['median']:.4f} "
                         f"mean {t['mean']:.4f}  NLL(GT CoC) med "
                         f"{st['nll_gtcoc']['median']:.4f}")
        for name, _, _ in CONTRASTS:
            c = res["contrasts"][s].get(name)
            if not c:
                continue
            for m in ("minADE_rollout", "minFDE_rollout", "minADE_tf", "nll_gtcoc"):
                if c.get(m):
                    unit = "" if m.startswith("nll") else "m"
                    L.append(f"    {name:22s} {m:14s} {fmt(c[m], unit)}")
    for b in BUDGETS:
        L.append(f"\n=== Gate G1 @ W{b} (pooled val+test, qvla_coc_b{b} - uniform_w{b}) ===")
        p = res["pooled_indist"].get(f"w{b}_coc_vs_uniform", {})
        for m in ("minADE_rollout", "minFDE_rollout"):
            L.append(f"  {m:14s} {fmt(p.get(m))}")
        g = res["gates"].get(f"G1_w{b}")
        if g:
            L.append(f"  rule: {g['rule']}")
            L.append(f"  PASS: {g['pass']}")
    L.append(f"\nCoC degeneracy below {GATE_DEGEN} everywhere: {res['gates']['degeneracy_ok']}")
    txt = "\n".join(L)
    (out_dir / "summary.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
