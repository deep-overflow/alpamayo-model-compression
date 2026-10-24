"""What is a correct CoC worth, and for which clips? Reads run_coc_swap.py + run_coc_ablation.py.

Five contexts per clip, same denoising seeds: trajprompt (no CoC, native mode), rollout
(the model's own CoC), gt (curated), gt_swap (another clip's curated CoC), gen_swap (the
CoC the model wrote for another clip). minADE@6 / minFDE@6 means, clip-bootstrap 95% CI,
Wilcoxon alongside.

Clips are stratified by how right the model's own CoC is, fixed before the run
(Qwen3-32B judge total, 0-4): wrong <= 1.0, partial (1.0, 3.5], right > 3.5.

Pre-registered (plans/2026-10-07_coc-vs-nococ-ood.md, "CoC 교체 실험"):
  Q1 gt - rollout on `wrong`: negative, CI excluding 0        (the main measurement)
  Q2 gt - rollout on `right`: zero
  Q3 gt_swap - gen_swap: style alone, content unrelated in both
  Q4 gen_swap - trajprompt, gt_swap - trajprompt: is an unrelated CoC worse than none
  Q5 gt - trajprompt: net value of the correct CoC over writing none

Usage:
  python experiments/evaluation/analyze_coc_swap.py
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

import eval_lib as el

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})
K = 6
CONDS = ["trajprompt", "rollout", "gt", "gt_swap", "gen_swap"]
PAIRS = [("gt", "rollout"), ("gt", "trajprompt"), ("rollout", "trajprompt"),
         ("gt_swap", "gen_swap"), ("gen_swap", "trajprompt"), ("gt_swap", "trajprompt"),
         ("gt", "gt_swap"), ("rollout", "gen_swap")]
STRATA = ["wrong", "partial", "right"]


def load(d):
    rows = {}
    for f in sorted(d.glob("baseline_s*of*.json")):
        for r in json.loads(f.read_text()):
            rows[r["clip_id"]] = r
    return rows


def paired(d):
    d = np.asarray(d, dtype=float)
    mean, lo, hi = el.paired_bootstrap_ci(d)
    return {"n": int(len(d)), "mean": mean, "ci": [lo, hi], "median": float(np.median(d)),
            "wilcoxon_p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
            "frac_better": float((d < -0.05).mean()), "frac_worse": float((d > 0.05).mean())}


def fmt(s):
    star = "*" if (s["ci"][0] > 0 or s["ci"][1] < 0) else " "
    return (f"{s['mean']:+.4f} [{s['ci'][0]:+.4f},{s['ci'][1]:+.4f}]{star} med {s['median']:+.4f} "
            f"p={s['wilcoxon_p']:.2g} better/worse {s['frac_better']:.0%}/{s['frac_worse']:.0%}")


def strata_of(total):
    return np.where(total <= 1.0, "wrong", np.where(total > 3.5, "right", "partial"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="coc_swap_ood")
    ap.add_argument("--src", default="coc_ablation_ood")
    ap.add_argument("--ref", default="baseline_ada_ood")
    ap.add_argument("--expect", type=int, default=1533)
    args = ap.parse_args()
    out = REPO / "outputs" / args.exp_id
    (out / "plots").mkdir(exist_ok=True)
    sw, ab, ref = load(out), load(REPO / "outputs" / args.src), load(REPO / "outputs" / args.ref)
    src = REPO / "outputs" / args.src
    j32 = json.loads((src / "coc_similarity_q32.json").read_text())
    j8 = json.loads((src / "coc_similarity.json").read_text())
    ids = [c for c in j32["clip_id"] if c in sw]
    assert len(ids) == args.expect, f"{len(ids)} clips, expected {args.expect}"
    pos = {c: i for i, c in enumerate(j32["clip_id"])}
    sel = np.array([pos[c] for c in ids])

    def red(cond, metric="ade", h="", k=K):
        src_rows = ab if cond in ("trajprompt", "rollout", "empty") else sw
        return np.array([min(src_rows[c][f"{metric}_{cond}_k{h}"][:k]) for c in ids])  # (N,)

    tot32 = (np.array(j32["judge_action_own"]) + np.array(j32["judge_cause_own"]))[sel]
    tot8 = (np.array(j8["judge_action_own"]) + np.array(j8["judge_cause_own"]))[sel]
    shuf32 = (np.array(j32["judge_action_shuffled"]) + np.array(j32["judge_cause_shuffled"]))[sel]
    st = strata_of(tot32)
    split = np.array([sw[c]["split"] for c in ids])
    scopes = {"all": np.ones(len(ids), bool), "train": split == "train", "val": split == "val",
              **{s: st == s for s in STRATA}}

    M = {"n": {k: int(v.sum()) for k, v in scopes.items()}, "abs": {}, "delta": {}}
    L = [f"CoC swap on OOD, {len(ids)} clips, minADE@{K} / minFDE@{K}; * = CI excludes 0",
         f"strata by Qwen3-32B judge total: {M['n']}",
         f"partner's generated CoC judged against this clip's GT: mean {shuf32.mean():.3f} "
         f"(own {tot32.mean():.3f}); <=1.0 on {(shuf32 <= 1.0).mean():.1%}", ""]

    # reproduction of the stored GT-CoC rung, k=8
    g8 = red("gt", k=8)
    old = np.array([ref[c]["minADE_tf"] for c in ids])
    M["gt_reproduced_k8"] = int((np.abs(g8 - old) < 1e-5).sum())
    L += [f"gt @8 reproduces stored {args.ref} minADE_tf on {M['gt_reproduced_k8']}/{len(ids)} "
          f"clips (mean {g8.mean():.4f} vs {old.mean():.4f})", ""]

    L.append("-- absolute means --")
    L.append(f"  {'scope':8s} {'n':>5s} " + " ".join(f"{c:>20s}" for c in CONDS) + "   (ADE / FDE)")
    for sn, k in scopes.items():
        M["abs"][sn] = {c: {"minADE": float(red(c)[k].mean()),
                            "minFDE": float(red(c, "fde")[k].mean())} for c in CONDS}
        L.append(f"  {sn:8s} {k.sum():5d} " + " ".join(
            f"{M['abs'][sn][c]['minADE']:9.4f} /{M['abs'][sn][c]['minFDE']:8.4f} " for c in CONDS))
    L.append("")

    for a, b in PAIRS:
        key = f"{a}-{b}"
        M["delta"][key] = {}
        L.append(f"-- {a} - {b} --")
        for sn, k in scopes.items():
            e = {"minADE": paired(red(a)[k] - red(b)[k]),
                 "minFDE": paired(red(a, "fde")[k] - red(b, "fde")[k])}
            M["delta"][key][sn] = e
            L.append(f"  {sn:8s} ADE {fmt(e['minADE'])}")
            L.append(f"  {'':8s} FDE {fmt(e['minFDE'])}")
        L.append("")

    # interaction: is the gain of GT over the own CoC larger where the own CoC is wrong
    rng = np.random.RandomState(0)
    d = red("gt") - red("rollout")
    w, r = d[st == "wrong"], d[st == "right"]
    bs = [w[rng.randint(0, len(w), len(w))].mean() - r[rng.randint(0, len(r), len(r))].mean()
          for _ in range(10000)]
    M["interaction_wrong_minus_right"] = {"mean": float(w.mean() - r.mean()),
                                          "ci": [float(np.percentile(bs, 2.5)),
                                                 float(np.percentile(bs, 97.5))]}
    x = M["interaction_wrong_minus_right"]
    L += [f"interaction, (gt - rollout | wrong) - (gt - rollout | right): {x['mean']:+.4f} "
          f"[{x['ci'][0]:+.4f},{x['ci'][1]:+.4f}]", ""]

    # how the main number is built: tail, trimming, horizon
    M["q1_structure"] = {}
    L.append("-- structure of gt - rollout on `wrong` --")
    ws = np.sort(w)
    t = int(0.1 * len(ws))
    order = np.argsort(np.abs(w))[::-1]
    M["q1_structure"] = {"trim10_mean": float(ws[t:len(ws) - t].mean()),
                         "top5pct_share_of_sum": float(w[order[:max(1, len(w) // 20)]].sum() / w.sum()),
                         "relative_to_rollout": float(w.mean() / red("rollout")[st == "wrong"].mean())}
    q = M["q1_structure"]
    L.append(f"  10% trimmed mean {q['trim10_mean']:+.4f}; the 5% of clips with the largest |delta| "
             f"carry {q['top5pct_share_of_sum']:.0%} of the sum; mean is "
             f"{q['relative_to_rollout']:+.1%} of rollout's minADE there")
    for hn, h in (("1.6s", "_h16"), ("3.2s", "_h32"), ("6.4s", "")):
        e = paired((red("gt", h=h) - red("rollout", h=h))[st == "wrong"])
        M["q1_structure"][hn] = e
        L.append(f"  horizon {hn}: {fmt(e)}")
    L.append("")

    # robustness of the strata definition
    M["strata_robustness"] = {}
    L.append("-- gt - rollout under other definitions of wrong / right --")
    emb = np.array(j32["emb_own"])[sel]
    lo_e, hi_e = np.quantile(emb, [1 / 3, 2 / 3])
    alt = {"judge 8B (<=1 / >3.5)": strata_of(tot8),
           "judge 32B strict (<=0.5 / >3.9)": np.where(tot32 <= 0.5, "wrong",
                                                       np.where(tot32 > 3.9, "right", "partial")),
           "embedding terciles": np.where(emb <= lo_e, "wrong",
                                          np.where(emb > hi_e, "right", "partial")),
           "both judges agree": np.where((tot32 <= 1) & (tot8 <= 1), "wrong",
                                         np.where((tot32 > 3.5) & (tot8 > 3.5), "right", "partial"))}
    for name, lab in alt.items():
        M["strata_robustness"][name] = {s: paired(d[lab == s]) for s in ("wrong", "right")}
        e = M["strata_robustness"][name]
        L.append(f"  {name:32s} wrong n={e['wrong']['n']:4d} {e['wrong']['mean']:+.4f} "
                 f"[{e['wrong']['ci'][0]:+.3f},{e['wrong']['ci'][1]:+.3f}]   right n={e['right']['n']:4d} "
                 f"{e['right']['mean']:+.4f} [{e['right']['ci'][0]:+.3f},{e['right']['ci'][1]:+.3f}]")

    (out / "metrics.json").write_text(json.dumps(M, indent=2))
    (out / "summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    names = ["all"] + STRATA
    show = [("rollout", C1, "own CoC"), ("gt", C2, "GT CoC"),
            ("gt_swap", C4, "another clip's GT CoC"), ("gen_swap", C3, "another clip's own CoC")]
    for j, (c, col, lab) in enumerate(show):
        e = [M["delta"][f"{c}-trajprompt"][s]["minADE"] if f"{c}-trajprompt" in M["delta"]
             else paired(red(c)[scopes[s]] - red("trajprompt")[scopes[s]]) for s in names]
        ax.bar(np.arange(len(names)) + (j - 1.5) * 0.2, [v["mean"] for v in e], 0.2,
               yerr=[[v["mean"] - v["ci"][0] for v in e], [v["ci"][1] - v["mean"] for v in e]],
               color=col, label=lab, error_kw={"ecolor": INK, "lw": 0.8})
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(np.arange(len(names)),
                  [f"all\n(n={M['n']['all']})"] + [f"own CoC {s}\n(n={M['n'][s]})" for s in STRATA])
    ax.set_ylabel(f"Δ minADE@{K} vs no CoC (m), <0: better")
    ax.set_title("What the cache's CoC does, relative to writing none")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout(), fig.savefig(out / "plots" / "swap_vs_nococ.png", dpi=150), plt.close(fig)


if __name__ == "__main__":
    main()
