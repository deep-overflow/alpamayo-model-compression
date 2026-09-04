"""Local reconstruction error vs what actually propagates, and vs capability.

plans/2026-09-04_stream-error-decomposition.md, part 2. Reads run_streamprop.py's
metrics.json (propagated) and run_streamerr.py's (local).

Three questions:
  P1  does the local per-sublayer error predict the propagated divergence?
  P2  does the energy-weighted verdict on the CoC trade survive propagation?
  P3  does either predict capability (val500 minADE@6, LingoQA)?

Capability numbers are pasted from the shipped runs rather than recomputed, so this
script stays cheap; each is a 500-clip / 500-question evaluation already on disk.

Usage:
  .venv/bin/python experiments/head_analysis/analyze_streamprop.py
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
BG, CARD, TEXT, MUTED = "#FAF9F5", "#FFFFFF", "#29261B", "#6B6555"
ACCENT, GOOD, WARN, BLUE = "#D97757", "#008300", "#eda100", "#2F6FBF"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": CARD,
                     "axes.edgecolor": "#E8E6DC", "text.color": TEXT,
                     "axes.labelcolor": TEXT, "xtick.color": MUTED,
                     "ytick.color": MUTED, "font.size": 9, "axes.titlesize": 10})
ARM_COLOR = {"dualr": MUTED, "dualr_rep": "#8C8878", "dualr_w": ACCENT, "dualr_wl": BLUE}
LABEL = {"vision": "vision", "prompt_text": "prompt text", "hist": "ego history",
         "sink": "sink", "coc": "own CoC"}
QLAB = {"h": "hidden state", "k": "cache K", "v": "cache V"}
# already-published capability of the same four checkpoints
CAP = {"dualr": (0.8143, 41.8), "dualr_rep": (0.8437, 52.2),
       "dualr_w": (0.8702, 49.0), "dualr_wl": (0.8271, 72.6)}
BOOT = 10000


def med_ci(d, seed=0):
    g = np.random.default_rng(seed)
    b = np.median(np.asarray(d)[g.integers(0, len(d), (BOOT, len(d)))], axis=1)
    return (float(np.median(d)), *np.percentile(b, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prop", default="streamprop_v1")
    ap.add_argument("--local", default="streamerr_v1")
    ap.add_argument("--out", default="streamprop_v1")
    args = ap.parse_args()
    P = json.loads((REPO / "outputs" / args.prop / "metrics.json").read_text())
    L = json.loads((REPO / "outputs" / args.local / "metrics.json").read_text())
    S, A = P["streams"], P["arms"]
    out = REPO / "outputs" / args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)
    res, lines = {}, []

    def pr(a, q, s):
        return np.array(P["rel"][a][f"{q}_{s}"])

    def lo(a, k, s):
        return np.array(L["err"][a][f"{k}_{s}"])

    def wts(den, keys):
        tot = sum(sum(den[k]) for k in keys)
        return {k: sum(den[k]) / tot for k in keys}

    lines.append(f"propagated divergence, {P['n_clips']} held-out clips")
    for q in ("h", "k", "v"):
        lines += ["", f"{QLAB[q]} -- relative L2, median over layers"]
        lines.append(f"{'arm':11s} " + " ".join(f"{s:>13s}" for s in S))
        for a in A:
            lines.append(f"{a:11s} " + " ".join(
                f"{np.nanmedian(pr(a, q, s)):13.4f}" for s in S))

    # --- energy shares differ per quantity ------------------------------------
    lines += ["", "dense energy share per stream (weights for the verdict):"]
    W = {}
    for q in ("h", "k", "v"):
        W[q] = wts(P["den"], [f"{q}_{s}" for s in S])
        lines.append(f"  {QLAB[q]:13s} "
                     + "  ".join(f"{s} {100 * W[q][f'{q}_{s}']:5.2f}%" for s in S))
    for k, lab in (("o", "o_proj"), ("m", "down_proj")):
        W[k] = wts(L["den"], [f"{k}_{s}" for s in S])
        lines.append(f"  {lab + ' (local)':13s} "
                     + "  ".join(f"{s} {100 * W[k][f'{k}_{s}']:5.2f}%" for s in S))
    res["energy_share"] = {q: {s: W[q][f"{q}_{s}"] for s in S} for q in W}

    # --- P2: does the verdict survive propagation? ----------------------------
    lines += ["", ("P2 energy-weighted net change of the CoC trade "
                   "(dualr_w - dualr_rep; negative = the trade paid off):")]
    verdict = {}
    for q in ("h", "k", "v"):
        verdict[QLAB[q]] = float(sum(
            W[q][f"{q}_{s}"] * (np.median(pr("dualr_w", q, s)) ** 2
                                - np.median(pr("dualr_rep", q, s)) ** 2) for s in S))
    for k, lab in (("o", "o_proj"), ("m", "down_proj")):
        verdict[f"{lab} (local)"] = float(sum(
            W[k][f"{k}_{s}"] * (np.median(lo("dualr_w", k, s)) ** 2
                                - np.median(lo("dualr_rep", k, s)) ** 2) for s in S))
    for k, v in verdict.items():
        lines.append(f"  {k:18s} {v:+.6f}  ({'이득' if v < 0 else '손해'})")
    res["verdict"] = verdict

    # --- P1: local vs propagated ----------------------------------------------
    lines += ["", "P1 local vs propagated, 20 (arm x stream) cells:"]
    res["p1"] = {}
    hv = [np.median(pr(a, "h", s)) for a in A for s in S]
    for k, lab in (("o", "o_proj"), ("m", "down_proj")):
        lv = [np.median(lo(a, k, s)) for a in A for s in S]
        r = stats.spearmanr(lv, hv)
        res["p1"][f"{lab}_vs_hidden"] = {"rho": float(r.statistic), "p": float(r.pvalue)}
        lines.append(f"  {lab:10s} vs hidden  rho {r.statistic:+.3f} (p={r.pvalue:.1e})")
    lines.append("  amplification (propagated hidden / local o_proj), median over layers:")
    lines.append(f"  {'arm':11s} " + " ".join(f"{s:>12s}" for s in S))
    for a in A:
        lines.append(f"  {a:11s} " + " ".join(
            f"{np.median(pr(a, 'h', s)) / np.median(lo(a, 'o', s)):12.3f}" for s in S))

    # --- P3: capability --------------------------------------------------------
    agg = {}
    for q in ("h", "k", "v"):
        agg[QLAB[q]] = {a: float(np.sqrt(sum(
            W[q][f"{q}_{s}"] * np.median(pr(a, q, s)) ** 2 for s in S))) for a in A}
    for k, lab in (("o", "o_proj"), ("m", "down_proj")):
        agg[f"{lab} (local)"] = {a: float(np.sqrt(sum(
            W[k][f"{k}_{s}"] * np.median(lo(a, k, s)) ** 2 for s in S))) for a in A}
    res["aggregate"], res["p3"] = agg, {}
    lines += ["", "P3 energy-weighted overall divergence vs published capability:"]
    lines.append(f"  {'arm':11s} {'val500':>8s} {'LingoQA':>8s} "
                 + " ".join(f"{m:>16s}" for m in agg))
    for a in A:
        lines.append(f"  {a:11s} {CAP[a][0]:8.4f} {CAP[a][1]:7.1f}% "
                     + " ".join(f"{agg[m][a]:16.4f}" for m in agg))
    lines.append("  Spearman (positive = less divergence goes with better capability):")
    for m in agg:
        v = [agg[m][a] for a in A]
        r1 = stats.spearmanr(v, [CAP[a][0] for a in A])
        r2 = stats.spearmanr(v, [-CAP[a][1] for a in A])
        res["p3"][m] = {"val500_rho": float(r1.statistic),
                        "lingo_rho": float(r2.statistic)}
        lines.append(f"    {m:18s} val500 {r1.statistic:+.2f}   LingoQA {r2.statistic:+.2f}")
    lines.append("  (n=4 arms: read the ordering, not the coefficient)")

    # --- plots -----------------------------------------------------------------
    fig, axes = plt.subplots(3, len(S), figsize=(3.1 * len(S), 8.4), sharex=True)
    for row, q in enumerate(("h", "k", "v")):
        for col, s in enumerate(S):
            ax = axes[row, col]
            for a in A:
                ax.plot(pr(a, q, s), lw=1.3, color=ARM_COLOR[a],
                        label=a if (row == 0 and col == 0) else None)
            ax.set_title(f"{QLAB[q]} · {LABEL[s]}", fontsize=9)
            if row == 2:
                ax.set_xlabel("VLM layer")
            if col == 0:
                ax.set_ylabel("relative divergence")
            ax.set_ylim(0, None)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "plots" / "prop_layers.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    for a in A:
        ax.scatter([np.median(lo(a, "o", s)) for s in S],
                   [np.median(pr(a, "h", s)) for s in S], s=40,
                   color=ARM_COLOR[a], label=a)
    lim = ax.get_xlim()
    ax.plot(lim, lim, "--", color=MUTED, lw=1)
    ax.set_xlabel("local o_proj error")
    ax.set_ylabel("propagated hidden divergence")
    ax.set_title(f"local vs propagated (ρ={res['p1']['o_proj_vs_hidden']['rho']:+.2f});"
                 " all below y=x")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    ks = list(verdict)
    cols = [GOOD if verdict[k] < 0 else ACCENT for k in ks]
    ax.bar(range(len(ks)), [verdict[k] for k in ks], 0.6, color=cols)
    ax.axhline(0, color=TEXT, lw=0.8)
    ax.set_xticks(range(len(ks)), ks, rotation=20, ha="right")
    ax.set_ylabel("energy-weighted Δ (w − rep)")
    ax.set_title("the CoC trade: local says loss, propagated says gain")
    fig.tight_layout()
    fig.savefig(out / "plots" / "local_vs_prop.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, (idx, nm, inv) in zip(axes, ((0, "val500 minADE@6 (lower better)", False),
                                         (1, "LingoQA accuracy % (higher better)", True))):
        for a in A:
            ax.scatter(agg["hidden state"][a], CAP[a][idx], s=70, color=ARM_COLOR[a])
            ax.annotate(a, (agg["hidden state"][a], CAP[a][idx]), fontsize=8,
                        xytext=(5, 4), textcoords="offset points")
        ax.set_xlabel("energy-weighted hidden-state divergence")
        ax.set_ylabel(nm)
        ax.set_title("preservation does not predict capability" if inv
                     else "preservation does not predict trajectory")
    fig.tight_layout()
    fig.savefig(out / "plots" / "capability.png", dpi=150)
    plt.close(fig)

    (out / "metrics_analysis.json").write_text(json.dumps(res, indent=1))
    text = "\n".join(lines)
    (out / "streamprop_summary.txt").write_text(text + "\n")
    print(text)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
