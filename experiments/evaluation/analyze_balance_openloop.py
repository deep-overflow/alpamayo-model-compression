"""Does moving the FM/CE balance off equal say cost anything open-loop?

`reports/evaluation/2026-09-09_criterion-balance.html` settled the SELECTION side: the
knob `max(rank I_traj, rank I_CoC - delta)` cannot be tuned at n=100 because the
calibration draw displaces the retained set 1.7-3.8x further than the knob does, and
delta = 0 is the modal optimum of eleven disjoint draws. That is an argument about which
units get kept. It is not a measurement of what keeping different units costs.

This runs the measurement. Two points on the knob were built at dual_u40_v2's exact
budget (2,657,452,032 params removed, 19/32 heads and 7390/12288 channels per layer,
expert and KV untouched, importance_v2), so the only factor is how much say each
objective gets:

  dualfm10  delta = +0.10   displaces  5.6% of dual's Q heads, 3.8% of its MLP channels
  dualfm20  delta = +0.20   displaces  8.2% / 6.9%

Positive delta favours the trajectory (action) objective: at +0.10 the reasoning half
decides 25.9% of retained Q heads instead of 46.5%, at +0.20 it decides 14.0%.

Frozen protocol: rollout only, minADE@6 / minFDE@6 (min over the first 6 of 8 stored
samples), median + paired bootstrap CI primary, mean and Wilcoxon reported alongside.
Every arm on Ada, clip-derived seeds, val500 / test500 / OOD-val 262.

Pre-registered reading, stated before the runs:
  the protocol resolves 0.05 m, and dualfm10's displacement is below the calibration
  noise floor (9.0% Q / 14.6% MLP), so a null was expected. A null therefore CONFIRMS
  the selection-side verdict rather than merely failing to reject it; a significant
  effect would have falsified it.

Usage:
  .venv/bin/python experiments/evaluation/analyze_balance_openloop.py
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
import paper_numbers as pn

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})

# arm -> set -> (outputs dir, keep only the OOD val split)
ARMS = {
    "baseline": {"val500": ("baseline_ada_ps_indist", False),
                 "test500": ("baseline_ada_ps_test", False),
                 "OOD-val": ("baseline_ada_ps_oodval", False)},
    "dual": {"val500": ("dual_u40_v2_ps_indist", False),
             "test500": ("dual_u40_v2_ps_test", False),
             "OOD-val": ("dual_u40_v2_ps_ood", True)},
    "dualfm10": {"val500": ("dualfm10_u40_v2_ps_indist", False),
                 "test500": ("dualfm10_u40_v2_ps_test", False),
                 "OOD-val": ("dualfm10_u40_v2_ps_oodval", False)},
    "dualfm20": {"val500": ("dualfm20_u40_v2_ps_indist", False),
                 "test500": ("dualfm20_u40_v2_ps_test", False),
                 "OOD-val": ("dualfm20_u40_v2_ps_oodval", False)},
}
SETS = ("val500", "test500", "OOD-val")
DELTA = {"dual": 0.0, "dualfm10": 0.10, "dualfm20": 0.20}
BOOT = 10000
GATE = 0.05  # the smallest minADE shift this protocol resolves


def at6(r, key):
    return float(np.min(np.asarray(r[key], dtype=float)[:6]))


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(0)
    meds = [np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(BOOT)]
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return {"median": float(np.median(d)), "lo": float(lo), "hi": float(hi),
            "mean": float(np.mean(d)),
            "wilcoxon": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
            "changed": float(np.mean(d != 0)), "n": len(d),
            "sig": bool(lo > 0 or hi < 0)}


def collect(root):
    rows = {}
    for arm, sets in ARMS.items():
        for s, (d, ood) in sets.items():
            if not (root / d).is_dir():
                continue  # an arm that has not been run on this set yet
            r = pn.load(d, ood)
            if r:
                rows[(arm, s)] = r
    return rows


def analyse(rows):
    res = {}
    for s in SETS:
        have = [a for a in ARMS if (a, s) in rows]
        if len(have) < 2:
            continue
        ids = sorted(set.intersection(*[set(rows[(a, s)]) for a in have]))
        cur = {"n": len(ids), "arms": have, "abs": {}, "delta": {}}
        v = {}
        for a in have:
            ade = np.array([at6(rows[(a, s)][i], "ade_rollout_k") for i in ids])
            fde = np.array([at6(rows[(a, s)][i], "fde_rollout_k") for i in ids])
            v[a] = {"ade": ade, "fde": fde}
            cur["abs"][a] = {
                "ade_mean": float(ade.mean()), "ade_median": float(np.median(ade)),
                "fde_mean": float(fde.mean()), "fde_median": float(np.median(fde)),
                "degen": float(np.mean([rows[(a, s)][i]["coc_degenerate"] for i in ids]))}
        for m in ("ade", "fde"):
            for x in ("dualfm10", "dualfm20"):
                if x in v:
                    cur["delta"][f"{m}:{x}-dual"] = paired(v[x][m], v["dual"][m])
                    cur["delta"][f"{m}:{x}-baseline"] = paired(v[x][m], v["baseline"][m])
            cur["delta"][f"{m}:dual-baseline"] = paired(v["dual"][m], v["baseline"][m])
        res[s] = cur
    return res


def plots(res, out):
    out.mkdir(parents=True, exist_ok=True)

    # 1 -- dose response on the set where every point exists
    s = "val500"
    arms = [a for a in ("dual", "dualfm10", "dualfm20") if a in res[s]["abs"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = [DELTA[a] for a in arms]
    med = [0.0] + [res[s]["delta"][f"ade:{a}-dual"]["median"] for a in arms[1:]]
    lo = [0.0] + [res[s]["delta"][f"ade:{a}-dual"]["lo"] for a in arms[1:]]
    hi = [0.0] + [res[s]["delta"][f"ade:{a}-dual"]["hi"] for a in arms[1:]]
    ax.axhspan(-GATE, GATE, color=MUTED, alpha=0.09, lw=0)
    ax.axhline(0, color=MUTED, lw=1.0, ls=":")
    ax.errorbar(x, med, yerr=[np.array(med) - np.array(lo), np.array(hi) - np.array(med)],
                fmt="o-", color=C1, lw=1.6, ms=7, capsize=4)
    for xi, mi, a in zip(x, med, arms):
        ax.annotate(a, (xi, mi), textcoords="offset points", xytext=(6, 9),
                    fontsize=8.5, color=MUTED)
    ax.text(0.005, GATE * 0.55, "the band this protocol cannot resolve (0.05 m)",
            fontsize=8.5, color=MUTED)
    ax.set_xlabel("delta   (0 = dual; larger favours the trajectory objective)")
    ax.set_ylabel("paired median minADE@6 vs dual  (m)")
    ax.set_title("val500: significant at delta=+0.2, and still a fifth of the gate")
    fig.tight_layout()
    fig.savefig(out / "o1_dose_response.png", dpi=150)
    plt.close(fig)

    # 2 -- dualfm10 across all three sets, both metrics
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ys, labels = [], []
    for mi, (m, mn) in enumerate((("ade", "minADE@6"), ("fde", "minFDE@6"))):
        for si, s in enumerate(SETS):
            k = f"{m}:dualfm10-dual"
            if s not in res or k not in res[s]["delta"]:
                continue
            d = res[s]["delta"][k]
            y = -(mi * 3.6 + si)
            ys.append(y)
            labels.append(f"{mn}  {s}")
            ax.plot([d["lo"], d["hi"]], [y, y], color=C1 if mi == 0 else C2, lw=2.4,
                    solid_capstyle="round")
            ax.plot(d["median"], y, "o", color=C1 if mi == 0 else C2, ms=7)
    ax.axvline(0, color=MUTED, lw=1.0, ls=":")
    ax.axvspan(-GATE, GATE, color=MUTED, alpha=0.09, lw=0)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("paired median vs dual, 95% bootstrap CI  (m)")
    ax.set_title("delta = +0.10 is indistinguishable from dual on every set")
    fig.tight_layout()
    fig.savefig(out / "o2_forest_dualfm10.png", dpi=150)
    plt.close(fig)

    # 3 -- CoC degeneracy
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    w, order = 0.2, ["baseline", "dual", "dualfm10", "dualfm20"]
    cols = {"baseline": MUTED, "dual": C1, "dualfm10": C2, "dualfm20": C4}
    for i, a in enumerate(order):
        xs = [j + (i - 1.5) * w for j, s in enumerate(SETS)
              if s in res and a in res[s]["abs"]]
        vs = [res[s]["abs"][a]["degen"] * 100 for s in SETS
              if s in res and a in res[s]["abs"]]
        ax.bar(xs, vs, w, color=cols[a], label=a, alpha=0.9)
    ax.set_xticks(range(len(SETS)))
    ax.set_xticklabels(SETS)
    ax.set_ylabel("CoC degenerate  (% of clips)")
    ax.set_title("dualfm10 is below dual on all three sets -- by 2, 1 and 7 clips")
    ax.legend(frameon=False, fontsize=9, ncol=4)
    fig.tight_layout()
    fig.savefig(out / "o3_coc_degeneracy.png", dpi=150)
    plt.close(fig)
    print(f"wrote 3 plots -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "balance_openloop")
    a = ap.parse_args()

    rows = collect(REPO / "outputs")
    res = analyse(rows)
    lines = []
    for s in SETS:
        if s not in res:
            lines.append(f"### {s}: not available")
            continue
        r = res[s]
        lines.append(f"### {s}  paired on {r['n']} clips  ({', '.join(r['arms'])})")
        lines.append(f"{'arm':10s} {'ADE mean':>9s} {'median':>8s} {'FDE mean':>9s} "
                     f"{'median':>8s} {'degen':>7s}")
        for arm in r["arms"]:
            b = r["abs"][arm]
            lines.append(f"{arm:10s} {b['ade_mean']:9.4f} {b['ade_median']:8.4f} "
                         f"{b['fde_mean']:9.4f} {b['fde_median']:8.4f} "
                         f"{b['degen'] * 100:6.1f}%")
        for k, d in r["delta"].items():
            lines.append(f"  {k:26s} median {d['median']:+.4f} "
                         f"[{d['lo']:+.4f}, {d['hi']:+.4f}] mean {d['mean']:+.4f} "
                         f"p={d['wilcoxon']:.4f} changed {d['changed'] * 100:.1f}%"
                         f"{'  *' if d['sig'] else ''}")
        lines.append("")
    txt = "\n".join(lines)
    print(txt)

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "config.json").write_text(json.dumps(
        {"arms": {k: {s: v[s][0] for s in v} for k, v in ARMS.items()},
         "delta": DELTA, "metric": "min over the first 6 of 8 rollout samples",
         "primary": "paired median + bootstrap CI; mean and Wilcoxon alongside",
         "gate_m": GATE, "boot": BOOT}, indent=2))
    (a.out / "metrics.json").write_text(json.dumps(res, indent=2))
    (a.out / "summary.txt").write_text(txt)
    plots(res, a.out / "plots")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
