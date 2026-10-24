"""Paired reading of run_coc_ablation.py: is the action better when the CoC is written?

Every no-CoC condition is read as a per-clip delta against `rollout` on the same clips and
the same denoising seeds (delta > 0 = removing the CoC hurts = the CoC helps). Primary
statistic is the mean of minADE@6 with a bootstrap CI; median and Wilcoxon alongside,
because the deltas are heavy-tailed.

Reported per scope (all 1,533 / OOD-train / OOD-val): minADE@6 / minFDE@6 at 1.6 / 3.2 /
6.4 s, the sample-mean ADE (separates accuracy from diversity -- a min over 6 rewards
spread), and the pre-registered gates of plans/2026-10-07_coc-vs-nococ-ood.md.
Subgroups (bucket, cluster, CoC length, whether the generated CoC's head clause matches
the curated one) are exploratory.

`--ref` names a stored run_baseline.py OOD run with the GT-CoC condition; its minADE_tf
(@8, never re-run here) is joined on clip id as a third rung next to @8 values of ours.

Usage:
  python experiments/evaluation/analyze_coc_ablation.py --exp-id coc_ablation_ood
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, wilcoxon

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

CONDS = ["rollout", "empty", "skip", "trajprompt"]
NOCOC = CONDS[1:]
COL = {"rollout": C1, "empty": C3, "skip": C4, "trajprompt": C2}
BUCKETS = ["decel_stop", "turn", "accel", "cruise"]
HORIZ = {"1.6s": "_h16", "3.2s": "_h32", "6.4s": ""}
K = 6


def load_rows(exp_dir):
    rows = {}
    for f in sorted(exp_dir.glob("baseline_s*of*.json")):
        for r in json.loads(f.read_text()):
            rows[r["clip_id"]] = r
    return list(rows.values())


def vec(rows, cond, metric="ade", h="", red="min", k=K):
    """(N,) per-clip reduction over the first k samples."""
    a = np.array([r[f"{metric}_{cond}_k{h}"][:k] for r in rows])  # (N, k)
    return a.min(1) if red == "min" else a.mean(1)


def paired(d):
    d = np.asarray(d, dtype=float)
    mean, lo, hi = el.paired_bootstrap_ci(d)
    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return {"n": int(len(d)), "mean": mean, "ci": [lo, hi], "median": float(np.median(d)),
            "wilcoxon_p": p, "frac_worse": float((d > 0.05).mean()),
            "frac_better": float((d < -0.05).mean())}


def head(text):
    return " ".join(str(text).lower().split()[:1])  # the verb; GT is free-form past it


def fmt(s):
    return (f"{s['mean']:+.4f} [{s['ci'][0]:+.4f},{s['ci'][1]:+.4f}] med {s['median']:+.4f} "
            f"p={s['wilcoxon_p']:.2g} worse/better {s['frac_worse']:.1%}/{s['frac_better']:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="coc_ablation_ood")
    ap.add_argument("--ref", default="baseline_ada_ood")
    ap.add_argument("--expect", type=int, default=1533)
    args = ap.parse_args()
    out = REPO / "outputs" / args.exp_id
    (out / "plots").mkdir(exist_ok=True)
    rows = load_rows(out)
    assert len(rows) == args.expect, f"{len(rows)} clips, expected {args.expect}"
    scopes = {"all": rows, "train": [r for r in rows if r["split"] == "train"],
              "val": [r for r in rows if r["split"] == "val"]}
    scopes = {s: v for s, v in scopes.items() if v}  # a partial run may have no val yet
    splits = [s for s in ("train", "val") if s in scopes]
    M = {"n": {s: len(v) for s, v in scopes.items()}, "k": K, "abs": {}, "delta": {}}
    L = [f"CoC ablation on OOD, {len(rows)} clips ({M['n']}),"
         f" minADE@{K}; delta = no-CoC - rollout, >0 means the CoC helps", ""]

    for sname, rs in scopes.items():
        M["abs"][sname], M["delta"][sname] = {}, {}
        L.append(f"== {sname} (n={len(rs)}) ==")
        for c in CONDS:
            a = {}
            for hn, h in HORIZ.items():
                for met in ("ade", "fde"):
                    v = vec(rs, c, met, h)
                    a[f"min{met.upper()}@{K}_{hn}"] = {"mean": float(v.mean()),
                                                      "median": float(np.median(v))}
            v = vec(rs, c, red="mean")
            a[f"meanADE@{K}_6.4s"] = {"mean": float(v.mean()), "median": float(np.median(v))}
            v = vec(rs, c, k=1)
            a["ADE@1_6.4s"] = {"mean": float(v.mean()), "median": float(np.median(v))}
            a["sec"] = float(np.mean([r[f"sec_{c}"] for r in rs]))
            a["nll_forced"] = float(np.mean([r[f"nll_{c}"] for r in rs]))
            M["abs"][sname][c] = a
            L.append(f"  {c:10s} minADE {a[f'minADE@{K}_6.4s']['mean']:.4f} "
                     f"(med {a[f'minADE@{K}_6.4s']['median']:.4f})  minFDE "
                     f"{a[f'minFDE@{K}_6.4s']['mean']:.4f}  meanADE "
                     f"{a[f'meanADE@{K}_6.4s']['mean']:.4f}  ADE@1 {a['ADE@1_6.4s']['mean']:.4f}")
        for c in NOCOC:
            d = {}
            for hn, h in HORIZ.items():
                for met in ("ade", "fde"):
                    d[f"min{met.upper()}@{K}_{hn}"] = paired(
                        vec(rs, c, met, h) - vec(rs, "rollout", met, h))
            d[f"meanADE@{K}_6.4s"] = paired(vec(rs, c, red="mean") - vec(rs, "rollout", red="mean"))
            d["ADE@1_6.4s"] = paired(vec(rs, c, k=1) - vec(rs, "rollout", k=1))
            M["delta"][sname][c] = d
            L.append(f"  d {c:10s} minADE  {fmt(d[f'minADE@{K}_6.4s'])}")
            L.append(f"    {'':10s} minFDE  {fmt(d[f'minFDE@{K}_6.4s'])}")
            L.append(f"    {'':10s} meanADE {fmt(d[f'meanADE@{K}_6.4s'])}")
            L.append(f"    {'':10s} @1.6s   {fmt(d[f'minADE@{K}_1.6s'])}")
        L.append("")

    # gates
    key = f"minADE@{K}_6.4s"
    g1 = all(M["delta"][s]["empty"][key]["ci"][0] > 0 for s in splits)
    g2 = M["delta"]["all"]["empty"][key]["mean"] >= 0.05
    sk, em = M["delta"]["all"]["skip"][key], M["delta"]["all"]["empty"][key]
    g3 = (np.sign(sk["mean"]) == np.sign(em["mean"])) and (sk["ci"][0] > 0 or sk["ci"][1] < 0)
    # same two rules on the native no-CoC mode (plan amendment, made before the full run)
    g1n = all(M["delta"][s]["trajprompt"][key]["ci"][0] > 0 for s in splits)
    g2n = M["delta"]["all"]["trajprompt"][key]["mean"] >= 0.05
    M["gates"] = {"G1_ci_excludes_0_both_splits": bool(g1), "G2_mean_ge_0.05": bool(g2),
                  "G3_skip_agrees": bool(g3), "G1n_trajprompt": bool(g1n),
                  "G2n_trajprompt": bool(g2n)}
    ok = {True: "PASS", False: "FAIL"}
    L += [f"GATES (empty)  G1 (CoC helps, both splits): {ok[g1]}   G2 (>=0.05 m): {ok[g2]}   "
          f"G3 (skip agrees): {ok[bool(g3)]}",
          f"GATES (trajprompt, native no-CoC mode)  G1': {ok[g1n]}   G2': {ok[g2n]}", ""]

    # subgroups (exploratory), delta of `empty` and `trajprompt` on all clips
    d_all = {c: vec(rows, c) - vec(rows, "rollout") for c in NOCOC}
    sub = {}

    def group(name, labels, order=None):
        labels = np.asarray(labels)
        sub[name] = {}
        L.append(f"-- by {name} (exploratory) --")
        for g in (order or sorted(set(labels.tolist()))):
            m = labels == g
            if m.sum() < 20:
                continue
            sub[name][str(g)] = {"n": int(m.sum()),
                                 "rollout": float(vec(rows, "rollout")[m].mean()),
                                 **{c: paired(d_all[c][m]) for c in NOCOC}}
            e = sub[name][str(g)]["empty"]
            L.append(f"  {str(g)[:38]:38s} n={m.sum():4d} rollout {sub[name][str(g)]['rollout']:.3f}"
                     f"  d empty {e['mean']:+.4f} [{e['ci'][0]:+.3f},{e['ci'][1]:+.3f}]"
                     f"  d trajprompt {sub[name][str(g)]['trajprompt']['mean']:+.4f}")
        L.append("")

    group("bucket", [r["bucket"] for r in rows], BUCKETS)
    top = [c for c, _ in Counter(r["cluster"] for r in rows).most_common(12)]
    group("cluster", [r["cluster"] if r["cluster"] in top else "~other" for r in rows])
    gl = np.array([r["gen_len"] for r in rows])
    q = np.quantile(gl, [1 / 3, 2 / 3])
    group("coc_len", np.where(gl <= q[0], f"short (<={q[0]:.0f} tok)",
                              np.where(gl <= q[1], f"mid (<={q[1]:.0f})", "long")))
    group("head_match", ["gen head == GT head" if head(r["gen_coc"]) == head(r["gt_coc"])
                         else "gen head != GT head" for r in rows])
    group("coc_degenerate", ["degenerate" if r["coc_degenerate"] else "healthy" for r in rows])

    M["subgroups"] = sub
    M["coc"] = {"gen_len_mean": float(gl.mean()), "gen_len_median": float(np.median(gl)),
                "degenerate_rate": float(np.mean([r["coc_degenerate"] for r in rows])),
                "head_match_rate": float(np.mean([head(r["gen_coc"]) == head(r["gt_coc"])
                                                  for r in rows])),
                "sec_generate_mean": float(np.mean([r["sec_generate"] for r in rows])),
                "spearman_delta_empty_vs_gen_len": float(spearmanr(d_all["empty"], gl)[0]),
                "spearman_delta_empty_vs_nll_empty": float(
                    spearmanr(d_all["empty"], [r["nll_empty"] for r in rows])[0])}
    L.append(f"CoC: {M['coc']['gen_len_mean']:.1f} tokens mean, degenerate "
             f"{M['coc']['degenerate_rate']:.2%}, head clause matches GT "
             f"{M['coc']['head_match_rate']:.1%}, generate() {M['coc']['sec_generate_mean']:.2f} s"
             f" (prefill + decode, sdpa path, not a latency benchmark)")

    # how far a clip moves between two contexts, same noise: the scale against which a
    # ~0 mean has to be read (the CoC does change the trajectory, in no consistent direction)
    M["abs_move"] = {}
    L += ["", "-- per-clip |delta minADE@6| between two contexts (same noise) --"]
    for a, b in [("rollout", "empty"), ("rollout", "trajprompt"), ("empty", "skip"),
                 ("empty", "trajprompt")]:
        ad = np.abs(vec(rows, a) - vec(rows, b))
        M["abs_move"][f"{a}|{b}"] = {"median_abs": float(np.median(ad)),
                                     "frac_gt_0.05": float((ad > 0.05).mean())}
        L.append(f"  {a + ' | ' + b:24s} median {np.median(ad):.4f}  >0.05 m on {(ad > 0.05).mean():.1%}")

    # stored GT-CoC rung, @8 on both sides
    ref_dir = REPO / "outputs" / args.ref
    ref = {}
    for f in ref_dir.glob("baseline_s*of*.json"):
        for r in json.loads(f.read_text()):
            ref[r["clip_id"]] = r
    if all(r["clip_id"] in ref and "minADE_tf" in ref[r["clip_id"]] for r in rows):
        M["ref_k8"] = {}
        L += ["", f"-- three rungs at k=8 (GT CoC from stored {args.ref}, not re-run) --"]
        for sname, rs in scopes.items():
            ro8, em8 = vec(rs, "rollout", k=8), vec(rs, "empty", k=8)
            gt8 = np.array([ref[r["clip_id"]]["minADE_tf"] for r in rs])
            old = np.array([ref[r["clip_id"]]["minADE_rollout"] for r in rs])
            M["ref_k8"][sname] = {
                "empty": float(em8.mean()), "rollout": float(ro8.mean()), "gt_coc": float(gt8.mean()),
                "empty_minus_rollout": paired(em8 - ro8), "gt_minus_rollout": paired(gt8 - ro8),
                "gt_minus_empty": paired(gt8 - em8),
                "rollout_reproduced": int((np.abs(ro8 - old) < 1e-5).sum())}
            x = M["ref_k8"][sname]
            L.append(f"  {sname:5s} empty {x['empty']:.4f}  rollout {x['rollout']:.4f}  GT CoC "
                     f"{x['gt_coc']:.4f} | GT-rollout {x['gt_minus_rollout']['mean']:+.4f} "
                     f"[{x['gt_minus_rollout']['ci'][0]:+.4f},{x['gt_minus_rollout']['ci'][1]:+.4f}]"
                     f" | GT-empty {x['gt_minus_empty']['mean']:+.4f} "
                     f"[{x['gt_minus_empty']['ci'][0]:+.4f},{x['gt_minus_empty']['ci'][1]:+.4f}]"
                     f" | rollout bit-reproduced {x['rollout_reproduced']}/{len(rs)}")

    (out / "metrics.json").write_text(json.dumps(M, indent=2))
    (out / "summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    # plots
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    w = 0.2
    for j, c in enumerate(CONDS):
        ms, los, his = [], [], []
        for rs in scopes.values():
            m, lo, hi = el.paired_bootstrap_ci(vec(rs, c))
            ms.append(m), los.append(m - lo), his.append(hi - m)
        ax.bar(np.arange(len(scopes)) + (j - 1.5) * w, ms, w, yerr=[los, his], color=COL[c], label=c,
               error_kw={"ecolor": INK, "lw": 0.8})
    ax.set_xticks(np.arange(len(scopes)), [f"{s}\n(n={len(v)})" for s, v in scopes.items()])
    ax.set_ylabel(f"minADE@{K} (m), mean ± 95% CI")
    ax.set_title("Open-loop accuracy by what the cache holds")
    ax.legend(frameon=False, ncol=4, fontsize=8)
    fig.tight_layout(), fig.savefig(out / "plots" / "abs_by_split.png", dpi=150), plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(3)
    for j, c in enumerate(NOCOC):
        for i, (sname, ls) in enumerate(zip(splits, ["-", "--"])):
            s = [M["delta"][sname][c][f"minADE@{K}_{hn}"] for hn in HORIZ]
            off = (j - 1) * 0.08 + (i - 0.5) * 0.03
            ax.errorbar(x + off, [v["mean"] for v in s],
                        yerr=[[v["mean"] - v["ci"][0] for v in s],
                              [v["ci"][1] - v["mean"] for v in s]],
                        color=COL[c], ls=ls, marker="o", ms=4, lw=1.2, capsize=2,
                        label=f"{c} ({sname})")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axhline(0.05, color=MUTED, lw=0.6, ls=":")
    ax.set_xticks(x, list(HORIZ))
    ax.set_xlabel("horizon")
    ax.set_ylabel(f"Δ minADE@{K} vs rollout (m)")
    ax.set_title("Cost of removing the CoC (>0: the CoC helps)")
    ax.legend(frameon=False, ncol=3, fontsize=7)
    fig.tight_layout(), fig.savefig(out / "plots" / "delta_by_horizon.png", dpi=150), plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bins = np.linspace(-1.5, 1.5, 61)
    for c in NOCOC:
        ax.hist(np.clip(d_all[c], -1.5, 1.5), bins=bins, histtype="step", color=COL[c], lw=1.3,
                label=f"{c} (mean {d_all[c].mean():+.3f}, med {np.median(d_all[c]):+.3f})")
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel(f"per-clip Δ minADE@{K} vs rollout (m), clipped to ±1.5")
    ax.set_ylabel("clips")
    ax.set_title("Per-clip paired difference, all 1,533")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(), fig.savefig(out / "plots" / "delta_hist.png", dpi=150), plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    names = [b for b in BUCKETS if b in sub["bucket"]]
    for j, c in enumerate(NOCOC):
        s = [sub["bucket"][b][c] for b in names]
        ax.bar(np.arange(len(names)) + (j - 1) * 0.26, [v["mean"] for v in s], 0.26,
               yerr=[[v["mean"] - v["ci"][0] for v in s], [v["ci"][1] - v["mean"] for v in s]],
               color=COL[c], label=c, error_kw={"ecolor": INK, "lw": 0.8})
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(np.arange(len(names)), [f"{b}\n(n={sub['bucket'][b]['n']})" for b in names])
    ax.set_ylabel(f"Δ minADE@{K} vs rollout (m)")
    ax.set_title("By manoeuvre bucket (exploratory)")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout(), fig.savefig(out / "plots" / "delta_by_bucket.png", dpi=150), plt.close(fig)


if __name__ == "__main__":
    main()
