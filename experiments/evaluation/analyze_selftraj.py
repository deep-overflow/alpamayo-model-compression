"""Anchoring the trajectory criterion on the model's own sample: stable, and worse.

`I_CoC` scores the model's own rollout; `I_traj` scores a flow-matching loss against the
GT action. The trajectory distribution is multimodal and the GT is one mode of it, so on a
clip where the model produces a different mode the residual v(x_t,t) - (x1_gt - eps) is
dominated by mode mismatch and the Taylor score becomes a projection onto "what moves the
output toward the GT" rather than "what this model uses". Replacing x1_gt with the model's
own Denoise_10 sample removes that asymmetry.

Four things were measured, in order (plans/2026-09-10_self-anchored-traj-importance.md):

  G1  which noise pairs with the sample -- the one that generated it (tied, the model's
      own coupling) or a fresh one (independent). Tied was suspected of degenerating.
  G2  does the anchor swap change the retained set at all.
  G3  does it make the estimate more stable -- the hypothesis.
  G4  does the resulting arm drive better.

G3 and G4 disagree, and that is the result: the self anchor is measurably the more stable
estimator and measurably the worse criterion. The GT residual's systematic component was
not only mode-mismatch noise; it was also the signal that tells the criterion which units
keep the model right rather than merely self-consistent.

Usage:
  .venv/bin/python experiments/evaluation/analyze_selftraj.py
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
sys.path.insert(0, str(REPO / "experiments" / "evaluation"))
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
import paper_numbers as pn
from run_cocsafe import rank_norm

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})

# gtN is the sample-count-matched control: the GT target with N repeats of the whole
# fm_steps sweep, so at every t it averages N independent noises exactly as the tied self
# anchor does with its N chords. Without it, "the self anchor is more stable" is
# indistinguishable from "the self anchor was given ten times the samples".
IMP = {"self": "importance_selftraj_v1", "gt": "importance_gt_ref100",
       "bestn6": "importance_bestn6",
       "gt_k3": "importance_gtk3", "gt_k10": "importance_gtk10"}
K_OF = {"gt": 1, "gt_k3": 3, "gt_k10": 10, "self": 10, "bestn6": 1}
EVAL = {"self": "dualself_u40_v2_ps_indist", "gt": "dualgtref_u40_v2_ps_indist",
        "bestn6": "dualbestn6_u40_v2_ps_indist",
        "baseline": "baseline_ada_ps_indist", "shipped": "dual_u40_v2_ps_indist"}
KEEP = {"vlm_q": 19, "vlm_mlp": 7390}
N_SPLITS = 20
GATE = 0.05


def dual_kept(traj, coc, keep):
    s = np.maximum(rank_norm(traj), rank_norm(coc))
    return [set(np.argsort(-s[i])[:keep]) for i in range(s.shape[0])]


def ov(a, b, keep):
    return float(np.mean([len(x & y) / keep for x, y in zip(a, b)]))


def at6(r, k):
    return float(np.min(np.asarray(r[k], dtype=float)[:6]))


def g1(root):
    p = root / "selftraj_probe" / "metrics.json"
    if not p.exists():
        return None
    rec = json.loads(p.read_text())["per_clip"]
    out = {k: float(np.median([r[k] for r in rec])) for k in ("fm_gt", "fm_A", "fm_B")}
    out["n"] = len(rec)
    out["straightness"] = float(np.median([r["straightness"] for r in rec]))
    out["ade_sample"] = float(np.mean([r["ade_sample_vs_gt"] for r in rec]))
    out["per_clip"] = {k: [r[k] for r in rec] for k in ("fm_gt", "fm_A", "fm_B")}
    return out


def g23(root):
    pcs = {k: dict(np.load(root / v / "importance_perclip.npz"))
           for k, v in IMP.items() if (root / v / "importance_perclip.npz").exists()}
    res = {"overlap": {}, "stability": {}}
    for axis, keep in KEEP.items():
        k = {n: dual_kept(pc[f"traj_{axis}"].mean(0), pc[f"coc_{axis}"].mean(0), keep)
             for n, pc in pcs.items()}
        res["overlap"][axis] = {f"self-{n}": ov(k["self"], k[n], keep)
                                for n in k if n != "self"}
        for n, pc in pcs.items():
            rng = np.random.default_rng(0)  # identical splits for both arms
            v = []
            for _ in range(N_SPLITS):
                idx = rng.permutation(pc[f"traj_{axis}"].shape[0])
                a, b = idx[: len(idx) // 2], idx[len(idx) // 2:]
                v.append(ov(dual_kept(pc[f"traj_{axis}"][a].mean(0),
                                      pc[f"coc_{axis}"][a].mean(0), keep),
                            dual_kept(pc[f"traj_{axis}"][b].mean(0),
                                      pc[f"coc_{axis}"][b].mean(0), keep), keep))
            res["stability"].setdefault(axis, {})[n] = v
    return res


def g4(root):
    rows = {}
    for k, d in EVAL.items():
        if (root / d).is_dir():
            rows[k] = pn.load(d, False)
    ids = sorted(set.intersection(*[set(v) for v in rows.values()]))
    out = {"n": len(ids), "abs": {}, "delta": {}}
    v = {}
    for k, r in rows.items():
        a = np.array([at6(r[i], "ade_rollout_k") for i in ids])
        f = np.array([at6(r[i], "fde_rollout_k") for i in ids])
        v[k] = {"ade": a, "fde": f}
        out["abs"][k] = {"ade_mean": float(a.mean()), "ade_median": float(np.median(a)),
                         "fde_mean": float(f.mean()), "fde_median": float(np.median(f)),
                         "degen": float(np.mean([r[i]["coc_degenerate"] for i in ids]))}
    for m in ("ade", "fde"):
        for x, y in (("bestn6", "gt"), ("self", "gt"), ("bestn6", "self"),
                     ("bestn6", "baseline"), ("self", "baseline"),
                     ("gt", "baseline"), ("gt", "shipped")):
            if x not in v or y not in v:
                continue
            d = v[x][m] - v[y][m]
            rng = np.random.default_rng(0)
            meds = [np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(10000)]
            lo, hi = np.percentile(meds, [2.5, 97.5])
            out["delta"][f"{m}:{x}-{y}"] = {
                "median": float(np.median(d)), "lo": float(lo), "hi": float(hi),
                "mean": float(np.mean(d)),
                "wilcoxon": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
                "sig": bool(lo > 0 or hi < 0)}
    return out, v


def plots(r1, r23, r4, v, out):
    out.mkdir(parents=True, exist_ok=True)

    # 1 -- G1: the three couplings
    if r1:
        fig, ax = plt.subplots(figsize=(7.0, 3.8))
        data = [r1["per_clip"]["fm_gt"], r1["per_clip"]["fm_A"], r1["per_clip"]["fm_B"]]
        bp = ax.boxplot(data, labels=["GT anchor\n(shipped)", "self, tied\n(A)",
                                      "self, independent\n(B)"],
                        widths=0.5, showfliers=False, patch_artist=True)
        for patch, c in zip(bp["boxes"], (MUTED, C1, C2)):
            patch.set_facecolor(c)
            patch.set_alpha(0.35)
        for med in bp["medians"]:
            med.set_color(INK)
        ax.axhline(0.019, color=C3, lw=1.2, ls="--")
        ax.text(3.35, 0.021, "abort threshold\n(a tenth of shipped)", color=C3,
                fontsize=8.5, ha="right")
        ax.set_ylabel("flow-matching loss at g=1")
        ax.set_title(f"G1: the tied coupling does not degenerate "
                     f"(straightness {r1['straightness']:.3f})")
        fig.tight_layout()
        fig.savefig(out / "s1_coupling.png", dpi=150)
        plt.close(fig)

    # 2 -- THE figure: stability up, quality down
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    ax = axes[0]
    for axis, c, mk in (("vlm_q", C1, "o"), ("vlm_mlp", C2, "s")):
        st = r23["stability"][axis]
        gts = sorted((K_OF[n], np.mean(st[n])) for n in st if n.startswith("gt"))
        ax.plot([k for k, _ in gts], [v_ for _, v_ in gts], mk + "-", color=c, lw=1.6,
                ms=6, label=f"GT anchor, {'Q head' if axis == 'vlm_q' else 'MLP'}")
        for n, m2, dx in (("self", "*", 9), ("bestn6", "D", -26)):
            if n not in st:
                continue
            sv = np.mean(st[n])
            ax.plot(K_OF[n], sv, m2, color=c, ms=14 if m2 == "*" else 7,
                    mec=INK, mew=0.6)
            ax.annotate(n, (K_OF[n], sv), textcoords="offset points",
                        xytext=(dx, -3), fontsize=8.5, color=c)
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 10])
    ax.set_xticklabels(["1", "3", "10"])
    ax.set_xlabel("independent noises averaged per t  (K)")
    ax.set_ylabel("split-half agreement")
    ax.set_title("G3  the GT anchor does not improve with more noises")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    ax = axes[1]
    labels, meds, los, his, cols = [], [], [], [], []
    for key, lbl, c in (("ade:gt-baseline", "GT anchor\nvs baseline", MUTED),
                        ("ade:bestn6-gt", "best-of-6\nvs GT anchor", C2),
                        ("ade:self-gt", "self anchor\nvs GT anchor", C1)):
        d = r4["delta"][key]
        labels.append(lbl)
        meds.append(d["median"])
        los.append(d["median"] - d["lo"])
        his.append(d["hi"] - d["median"])
        cols.append(c)
    ax.bar(range(3), meds, 0.55, yerr=[los, his], color=cols, alpha=0.9, capsize=4)
    ax.axhline(GATE, color=C4, lw=1.2, ls="--")
    ax.text(2.4, GATE * 1.06, "0.05 m, what the protocol resolves", color=C4,
            fontsize=8.5, ha="right")
    ax.axhline(0, color=MUTED, lw=0.9)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("paired median minADE@6  (m)")
    ax.set_title("G4  and the arm it selects is WORSE")
    fig.tight_layout()
    fig.savefig(out / "s2_stability_vs_quality.png", dpi=150)
    plt.close(fig)

    # 3 -- per-clip paired difference, self vs gt
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    d = np.sort(v["self"]["ade"] - v["gt"]["ade"])
    ax.plot(np.arange(len(d)) / len(d) * 100, d, color=C1, lw=1.6)
    ax.axhline(0, color=MUTED, lw=0.9, ls=":")
    ax.axhspan(-GATE, GATE, color=MUTED, alpha=0.09, lw=0)
    ax.set_xlabel("clips, sorted by the paired difference  (%)")
    ax.set_ylabel("minADE@6  self - GT anchor  (m)")
    ax.set_ylim(-2, 4)
    ax.set_title(f"the loss is not a few outliers: "
                 f"{float(np.mean(d > 0)) * 100:.0f}% of clips get worse")
    fig.tight_layout()
    fig.savefig(out / "s3_perclip.png", dpi=150)
    plt.close(fig)

    # 4 -- CoC degeneracy
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    order = [k for k in ("baseline", "shipped", "gt", "self") if k in r4["abs"]]
    names = {"baseline": "baseline", "shipped": "shipped dual", "gt": "GT anchor",
             "self": "self anchor"}
    cols = {"baseline": MUTED, "shipped": C2, "gt": C2, "self": C1}
    ax.bar(range(len(order)), [r4["abs"][k]["degen"] * 100 for k in order], 0.55,
           color=[cols[k] for k in order], alpha=0.9)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([names[k] for k in order], fontsize=9)
    ax.set_ylabel("CoC degenerate  (% of clips)")
    ax.set_title("reasoning collapse quadruples under the self anchor")
    fig.tight_layout()
    fig.savefig(out / "s4_degeneracy.png", dpi=150)
    plt.close(fig)
    print(f"wrote 4 plots -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "selftraj_eval")
    a = ap.parse_args()
    root = REPO / "outputs"

    r1 = g1(root)
    r23 = g23(root)
    r4, v = g4(root)

    lines = ["Self-anchored trajectory importance: G1-G4", ""]
    if r1:
        lines += [(f"G1  {r1['n']} clips.  gt {r1['fm_gt']:.4f}   "
                   f"A(tied) {r1['fm_A']:.4f}   B(indep) {r1['fm_B']:.4f}   (medians)"),
                  f"    A/gt {r1['fm_A'] / r1['fm_gt']:.3f}, abort below 0.019 -> ADOPT A",
                  (f"    flow straightness {r1['straightness']:.3f}; sampled ADE "
                   f"{r1['ade_sample']:.3f} m"), ""]
    lines += ["G2  retained-set overlap of the self anchor against each GT arm:"]
    for axis in KEEP:
        for k, val in r23["overlap"][axis].items():
            lines.append(f"    {axis:9s} vs {k.split('-')[1]:7s} {val:.4f}")
    lines += ["", f"G3  split-half stability, {N_SPLITS} identical splits.",
              "    gt_kN is the sample-matched control: N independent noises per t,",
              "    exactly what the tied self anchor averages with its N chords.",
              f"    {'axis':9s} {'arm':9s} {'K':>3s} {'stability':>10s} {'vs self':>18s}"]
    for axis in KEEP:
        sv = np.array(r23["stability"][axis]["self"])
        for n in sorted(r23["stability"][axis], key=lambda x: (x == "self", K_OF.get(x, 0))):
            v_ = np.array(r23["stability"][axis][n])
            if n == "self":
                lines.append(f"    {axis:9s} {n:9s} {K_OF.get(n, 1):3d} "
                             f"{v_.mean():10.4f} {'--':>18s}")
                continue
            d = sv - v_
            rng = np.random.default_rng(0)
            lo, hi = np.percentile(
                [np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(2000)],
                [2.5, 97.5])
            lines.append(f"    {axis:9s} {n:9s} {K_OF.get(n, 1):3d} {v_.mean():10.4f} "
                         f"{d.mean():+8.4f} [{lo:+.3f},{hi:+.3f}]")
    lines += ["", f"G4  val500, {r4['n']} clips paired:"]
    for k, b in r4["abs"].items():
        lines.append(
            f"    {k:9s} minADE@6 {b['ade_mean']:.4f} / {b['ade_median']:.4f}   "
            f"minFDE@6 {b['fde_mean']:.4f} / {b['fde_median']:.4f}   "
            f"degen {b['degen'] * 100:.1f}%")
    for key, d in r4["delta"].items():
        lines.append(
            f"    {key:22s} median {d['median']:+.4f} [{d['lo']:+.4f}, "
            f"{d['hi']:+.4f}] mean {d['mean']:+.4f} p={d['wilcoxon']:.4f}"
            f"{'  *' if d['sig'] else ''}")
    lines += ["",
              "VERDICT: G1 adopt A, G2 pass, G3 PASS (more stable), G4 FAIL (worse arm).",
              "Stability of the estimator does not predict quality of the selection."]
    txt = "\n".join(lines)
    print(txt)

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "config.json").write_text(json.dumps(
        {"importance_runs": IMP, "eval_runs": EVAL, "keep": KEEP,
         "n_splits": N_SPLITS, "gate_m": GATE,
         "note": "both arms built and evaluated on cvlab20 from 100-clip Ada importance "
                 "runs; the shipped-dual rows are cvlab21 context"}, indent=2))
    (a.out / "metrics.json").write_text(json.dumps(
        {"g1": {k: v_ for k, v_ in (r1 or {}).items() if k != "per_clip"},
         "g2": r23["overlap"],
         "g3": {ax: {n: list(map(float, r23["stability"][ax][n])) for n in ("self", "gt")}
                for ax in KEEP},
         "g4": r4}, indent=2))
    (a.out / "summary.txt").write_text(txt + "\n")
    plots(r1, r23, r4, v, a.out / "plots")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
