"""Is the CoC worth more when it is closer to the curated one?

Joins score_coc_similarity.py's per-clip similarities with run_coc_ablation.py's paired
trajectory results. delta = minADE@6(trajprompt) - minADE@6(rollout); > 0 means writing the
CoC helped on that clip.

Similarity S, three ways: `judge` (Qwen3-8B action + cause, 0-4), `emb` (mpnet cosine), `nll`
(-nll_gtcoc from the stored GT-CoC run: how plausible the model finds the curated sentence --
a property of the clip, not of the sentence it generated).

Pre-registered (plans/2026-10-07_coc-vs-nococ-ood.md, 2026-10-08 section):
  H1  Spearman(S, delta) > 0 with a bootstrap CI excluding 0
  H2  S is higher on clips where rollout won (delta > +0.05) than where it lost (< -0.05)
  neg Spearman(S, empty - trajprompt) must include 0 -- neither side has a CoC
  pos Spearman(S, GT - rollout @8) > 0 -- a CoC already close to GT gains nothing from GT
  partial rank correlation controlling for trajprompt's own minADE (clip difficulty)

Usage:
  python experiments/evaluation/analyze_coc_similarity.py
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, rankdata, spearmanr

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
MEAS = ["judge", "judge_action", "judge_cause", "emb", "nll"]


def load(exp_dir):
    rows = {}
    for f in sorted(exp_dir.glob("baseline_s*of*.json")):
        for r in json.loads(f.read_text()):
            rows[r["clip_id"]] = r
    return rows


def red(rows, cond, metric="ade", k=K):
    return np.array([min(r[f"{metric}_{cond}_k"][:k]) for r in rows])  # (N,)


def rho_ci(x, y, n_boot=2000, seed=0):
    r = float(spearmanr(x, y)[0])
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(x), size=(n_boot, len(x)))
    b = [spearmanr(x[i], y[i])[0] for i in idx]
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"rho": r, "ci": [float(lo), float(hi)], "p": float(spearmanr(x, y)[1])}


def partial_rho(x, y, z):
    """Spearman of x and y with the rank of z regressed out of both."""
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    A = np.c_[np.ones_like(rz), rz]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def f(s):
    star = "*" if (s["ci"][0] > 0 or s["ci"][1] < 0) else " "
    return f"{s['rho']:+.3f} [{s['ci'][0]:+.3f},{s['ci'][1]:+.3f}]{star}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="coc_ablation_ood")
    ap.add_argument("--ref", default="baseline_ada_ood")
    ap.add_argument("--tag", default=None,
                    help="read coc_similarity_<tag>.json (a second judge) and suffix the outputs")
    args = ap.parse_args()
    out = REPO / "outputs" / args.exp_id
    sfx = f"_{args.tag}" if args.tag else ""
    by = load(out)
    ref = load(REPO / "outputs" / args.ref)
    sim = json.loads((out / f"coc_similarity{sfx}.json").read_text())
    ids = sim["clip_id"]
    assert set(ids) == set(by) and len(ids) == 1533
    rows = [by[c] for c in ids]
    split = np.array([r["split"] for r in rows])

    S = {"judge_action": np.array(sim["judge_action_own"]),
         "judge_cause": np.array(sim["judge_cause_own"]),
         "emb": np.array(sim["emb_own"]),
         "nll": -np.array([ref[c]["nll_gtcoc"] for c in ids])}
    S["judge"] = S["judge_action"] + S["judge_cause"]
    ro, tp, em = red(rows, "rollout"), red(rows, "trajprompt"), red(rows, "empty")
    Y = {"d_trajprompt": tp - ro, "d_empty": em - ro,
         "d_trajprompt_fde": red(rows, "trajprompt", "fde") - red(rows, "rollout", "fde"),
         "neg_empty_minus_trajprompt": em - tp,
         "pos_gt_minus_rollout_k8": np.array([ref[c]["minADE_tf"] for c in ids])
         - red(rows, "rollout", k=8)}
    Y["gt_minus_trajprompt_k8"] = (np.array([ref[c]["minADE_tf"] for c in ids])
                                   - red(rows, "trajprompt", k=8))

    M = {"n": len(ids), "controls": {}, "agreement": {}, "h1": {}, "h2": {}, "bins": {}}
    L = [f"CoC similarity to GT vs benefit of the CoC, OOD {len(ids)} clips, minADE@{K}",
         "delta = trajprompt - rollout (>0: the CoC helped); * = CI excludes 0", ""]

    L.append("-- validity of the two text measures (mean over clips) --")
    for name, key in (("judge action", "judge_action"), ("judge cause", "judge_cause"),
                      ("emb cosine", "emb")):
        c = {k: float(np.mean(sim[f"{key}_{k}"])) for k in ("own", "shuffled", "identity")}
        own, sh = np.array(sim[f"{key}_own"]), np.array(sim[f"{key}_shuffled"])
        c["own_gt_shuffled_frac"] = float((own > sh).mean())
        c["auc_own_vs_shuffled"] = float(mannwhitneyu(own, sh).statistic / len(own) ** 2)
        M["controls"][key] = c
        L.append(f"  {name:13s} own {c['own']:.3f}  other clip's CoC {c['shuffled']:.3f}  GT itself "
                 f"{c['identity']:.3f}  AUC(own vs other) {c['auc_own_vs_shuffled']:.3f}")
    L.append("")

    if args.tag:
        # same prompt, same pairs, other judge: agreement is the only reliability estimate
        # available without human labels
        base = json.loads((out / "coc_similarity.json").read_text())
        assert base["clip_id"] == ids
        M["judge_agreement"] = {"this": sim["judge_model"], "other": base["judge_model"]}
        L.append(f"-- judge agreement: {sim['judge_model']} vs {base['judge_model']} --")
        for qn in ("action", "cause"):
            a, b = np.array(sim[f"judge_{qn}_own"]), np.array(base[f"judge_{qn}_own"])
            ra, rb = np.rint(a).astype(int), np.rint(b).astype(int)
            tab = [[int(((ra == i) & (rb == j)).sum()) for j in range(3)] for i in range(3)]
            M["judge_agreement"][qn] = {
                "spearman": float(spearmanr(a, b)[0]), "exact": float((ra == rb).mean()),
                "mean_this": float(a.mean()), "mean_other": float(b.mean()), "table": tab}
            L.append(f"  {qn:6s} Spearman {spearmanr(a, b)[0]:+.3f}  same rounded score "
                     f"{(ra == rb).mean():.1%}  mean {a.mean():.3f} vs {b.mean():.3f}  "
                     f"rows=this 0/1/2, cols=other: {tab}")
        ta = np.array(sim["judge_action_own"]) + np.array(sim["judge_cause_own"])
        tb = np.array(base["judge_action_own"]) + np.array(base["judge_cause_own"])
        M["judge_agreement"]["total_spearman"] = float(spearmanr(ta, tb)[0])
        M["judge_agreement"]["full_match_this"] = float((ta > 3.5).mean())
        M["judge_agreement"]["full_match_other"] = float((tb > 3.5).mean())
        L.append(f"  total  Spearman {spearmanr(ta, tb)[0]:+.3f}  scored >3.5 (full match): "
                 f"{(ta > 3.5).mean():.1%} vs {(tb > 3.5).mean():.1%}")
        L.append("")

    L.append("-- agreement between measures (Spearman) --")
    for i, a in enumerate(MEAS):
        for b in MEAS[i + 1:]:
            M["agreement"][f"{a}|{b}"] = float(spearmanr(S[a], S[b])[0])
    L.append("  " + "  ".join(f"{k} {v:+.2f}" for k, v in M["agreement"].items()))
    L.append("")

    L.append("-- H1 dose-response: Spearman(S, outcome) [95% CI] --")
    L.append(f"  {'measure':13s} {'d trajprompt (H1)':27s} {'partial|difficulty':>18s}  "
             f"{'d empty':27s} {'d trajprompt FDE':27s}")
    for m in MEAS:
        M["h1"][m] = {y: rho_ci(S[m], Y[y]) for y in Y}
        M["h1"][m]["partial_d_trajprompt"] = partial_rho(S[m], Y["d_trajprompt"], tp)
        for sp in ("train", "val"):
            k = split == sp
            M["h1"][m][f"d_trajprompt_{sp}"] = rho_ci(S[m][k], Y["d_trajprompt"][k])
        h = M["h1"][m]
        L.append(f"  {m:13s} {f(h['d_trajprompt']):27s} {h['partial_d_trajprompt']:+18.3f}  "
                 f"{f(h['d_empty']):27s} {f(h['d_trajprompt_fde']):27s}")
    L.append("  by split (d trajprompt):")
    for m in MEAS:
        L.append(f"  {m:13s} train {f(M['h1'][m]['d_trajprompt_train'])}   "
                 f"val {f(M['h1'][m]['d_trajprompt_val'])}")
    L.append("")
    L.append("-- controls: Spearman(S, outcome) --")
    L.append(f"  {'measure':13s} {'NEG empty - trajprompt (want 0)':34s} "
             f"{'POS GT - rollout @8 (want > 0)':34s}")
    for m in MEAS:
        h = M["h1"][m]
        L.append(f"  {m:13s} {f(h['neg_empty_minus_trajprompt']):34s} "
                 f"{f(h['pos_gt_minus_rollout_k8']):34s}")
    L.append("")

    d = Y["d_trajprompt"]
    grp = {"rollout wins (d>+0.05)": d > 0.05, "tie (|d|<=0.05)": np.abs(d) <= 0.05,
           "rollout loses (d<-0.05)": d < -0.05}
    L.append("-- H2: similarity by who won (rollout vs trajprompt) --")
    L.append(f"  {'group':26s} {'n':>5s} " + " ".join(f"{m:>13s}" for m in MEAS))
    for g, k in grp.items():
        M["h2"][g] = {"n": int(k.sum()), **{m: float(S[m][k].mean()) for m in MEAS}}
        L.append(f"  {g:26s} {k.sum():5d} " + " ".join(f"{S[m][k].mean():13.4f}" for m in MEAS))
    w, lo = grp["rollout wins (d>+0.05)"], grp["rollout loses (d<-0.05)"]
    M["h2"]["wins_minus_loses"], M["h2"]["mwu_p"] = {}, {}
    for m in MEAS:
        rng = np.random.RandomState(0)
        bs = [S[m][w][rng.randint(0, w.sum(), w.sum())].mean()
              - S[m][lo][rng.randint(0, lo.sum(), lo.sum())].mean() for _ in range(5000)]
        M["h2"]["wins_minus_loses"][m] = {"mean": float(S[m][w].mean() - S[m][lo].mean()),
                                          "ci": [float(np.percentile(bs, 2.5)),
                                                 float(np.percentile(bs, 97.5))]}
        M["h2"]["mwu_p"][m] = float(mannwhitneyu(S[m][w], S[m][lo]).pvalue)
    L.append(f"  {'wins - loses':26s} {'':5s} " + " ".join(
        f"{M['h2']['wins_minus_loses'][m]['mean']:+13.4f}" for m in MEAS))
    L.append(f"  {'Mann-Whitney p':26s} {'':5s} " + " ".join(
        f"{M['h2']['mwu_p'][m]:13.3g}" for m in MEAS))
    L.append("")

    L.append("-- delta by similarity quintile (mean [95% CI], m) --")
    for m in ("judge", "emb", "nll"):
        q = np.minimum((rankdata(S[m], method="ordinal") - 1) * 5 // len(ids), 4).astype(int)
        M["bins"][m] = []
        L.append(f"  {m}")
        for b in range(5):
            k = q == b
            e = {"lo": float(S[m][k].min()), "hi": float(S[m][k].max()), "n": int(k.sum()),
                 "rollout": float(ro[k].mean()), "trajprompt": float(tp[k].mean())}
            for y in ("d_trajprompt", "neg_empty_minus_trajprompt", "pos_gt_minus_rollout_k8",
                      "gt_minus_trajprompt_k8"):
                e[y] = dict(zip(("mean", "lo", "hi"), el.paired_bootstrap_ci(Y[y][k])))
            M["bins"][m].append(e)
            x = e["d_trajprompt"]
            L.append(f"    Q{b + 1} S in [{e['lo']:+.3f},{e['hi']:+.3f}] n={e['n']}  rollout "
                     f"{e['rollout']:.3f}  d {x['mean']:+.4f} [{x['lo']:+.3f},{x['hi']:+.3f}]  "
                     f"neg {e['neg_empty_minus_trajprompt']['mean']:+.4f}  "
                     f"GT-rollout {e['pos_gt_minus_rollout_k8']['mean']:+.4f}  "
                     f"GT-trajprompt {e['gt_minus_trajprompt_k8']['mean']:+.4f}")
    L.append("")

    # POST HOC (added after seeing the quintile means): the registered statistic is a rank
    # correlation, but these deltas are heavy-tailed and the protocol's headline is the mean,
    # so a mean effect carried by a tail of clips is invisible to it. Slope per SD of S and
    # the top-minus-bottom quintile gap, both on means, clip bootstrap.
    M["posthoc_mean"] = {}
    L.append("-- POST HOC, mean-based: slope per 1 SD of S, and Q5 - Q1 (m) [95% CI] --")
    rng = np.random.RandomState(0)
    idx = rng.randint(0, len(ids), size=(5000, len(ids)))
    outs = ["d_trajprompt", "neg_empty_minus_trajprompt", "pos_gt_minus_rollout_k8",
            "gt_minus_trajprompt_k8"]
    for m in ("judge", "emb", "nll"):
        z = (S[m] - S[m].mean()) / S[m].std()
        q = np.minimum((rankdata(S[m], method="ordinal") - 1) * 5 // len(ids), 4).astype(int)
        M["posthoc_mean"][m] = {}
        L.append(f"  {m}")
        for y in outs:
            v = Y[y]
            sl = float(np.polyfit(z, v, 1)[0])
            bsl = [np.polyfit(z[i], v[i], 1)[0] for i in idx[:2000]]
            gap = float(v[q == 4].mean() - v[q == 0].mean())
            bg = [v[i][q[i] == 4].mean() - v[i][q[i] == 0].mean() for i in idx]
            e = {"slope": sl, "slope_ci": [float(np.percentile(bsl, 2.5)),
                                            float(np.percentile(bsl, 97.5))],
                 "q5_minus_q1": gap, "gap_ci": [float(np.percentile(bg, 2.5)),
                                                float(np.percentile(bg, 97.5))]}
            M["posthoc_mean"][m][y] = e
            L.append(f"    {y:28s} slope {sl:+.4f} [{e['slope_ci'][0]:+.4f},{e['slope_ci'][1]:+.4f}]"
                     f"   Q5-Q1 {gap:+.4f} [{e['gap_ci'][0]:+.4f},{e['gap_ci'][1]:+.4f}]")
    L.append("")

    h1 = {m: M["h1"][m]["d_trajprompt"]["ci"][0] > 0 for m in ("judge", "emb", "nll")}
    neg = {m: M["h1"][m]["neg_empty_minus_trajprompt"]["ci"][0] <= 0
           <= M["h1"][m]["neg_empty_minus_trajprompt"]["ci"][1] for m in ("judge", "emb", "nll")}
    M["verdict"] = {"H1": h1, "neg_control_clean": neg}
    L.append(f"VERDICT  H1 (rho>0, CI excl. 0): {h1}   negative control clean: {neg}")

    rng = np.random.RandomState(1)
    L += ["", "-- face validity: random pairs --"]
    for i in rng.choice(len(ids), 8, replace=False):
        L.append(f"  act {S['judge_action'][i]:.2f} cause {S['judge_cause'][i]:.2f} emb "
                 f"{S['emb'][i]:.2f} | GT: {rows[i]['gt_coc']} | GEN: {rows[i]['gen_coc']}")

    (out / f"metrics_similarity{sfx}.json").write_text(json.dumps(M, indent=2))
    (out / f"summary_similarity{sfx}.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
    lab = {"judge": f"LLM judge {sim['judge_model'].split('/')[-1]} (0-4)", "emb": "embedding cosine",
           "nll": "-NLL of GT CoC"}
    for ax, m in zip(axes, ("judge", "emb", "nll")):
        for y, c, name in (("d_trajprompt", C1, "trajprompt - rollout"),
                           ("neg_empty_minus_trajprompt", MUTED, "empty - trajprompt (neg. control)")):
            e = [b[y] for b in M["bins"][m]]
            ax.errorbar(np.arange(5) + (0.06 if y[0] == "n" else -0.06), [v["mean"] for v in e],
                        yerr=[[v["mean"] - v["lo"] for v in e], [v["hi"] - v["mean"] for v in e]],
                        color=c, marker="o", ms=4, lw=1.3, capsize=2, label=name)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(np.arange(5), ["Q1\nleast similar", "Q2", "Q3", "Q4", "Q5\nmost similar"])
        r = M["h1"][m]["d_trajprompt"]
        ax.set_title(f"{lab[m]}\nrho {r['rho']:+.3f} [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]")
    axes[0].set_ylabel(f"Δ minADE@{K} (m), >0: CoC helped")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(), fig.savefig(out / "plots" / f"sim_dose_response{sfx}.png", dpi=150), plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, m in zip(axes, ("judge", "emb", "nll")):
        vals = [S[m][k] for k in grp.values()]
        ms = [v.mean() for v in vals]
        ci = [1.96 * v.std(ddof=1) / np.sqrt(len(v)) for v in vals]
        ax.bar(range(3), ms, yerr=ci, color=[C2, MUTED, C3], error_kw={"ecolor": INK, "lw": 0.8})
        ax.set_xticks(range(3), [f"rollout wins\n(n={len(vals[0])})", f"tie\n(n={len(vals[1])})",
                                 f"rollout loses\n(n={len(vals[2])})"])
        ax.set_ylim(min(ms) - 4 * max(ci), max(ms) + 4 * max(ci))
        ax.set_title(f"{lab[m]}\nwins vs loses p={M['h2']['mwu_p'][m]:.2g}")
    axes[0].set_ylabel("similarity to GT CoC (mean ± 95% CI)")
    fig.tight_layout(), fig.savefig(out / "plots" / f"sim_by_winner{sfx}.png", dpi=150), plt.close(fig)


if __name__ == "__main__":
    main()
