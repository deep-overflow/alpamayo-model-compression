"""Per-token-type reconstruction error: gates, per-layer curves, and the exchange.

plans/2026-09-04_stream-error-decomposition.md. Reads run_streamerr.py's metrics.json.

The point of the decomposition is that reconstruction is a budget: an arm that buys
one token stream into its Hessian sells another. `analyze_dualrw.py`'s "rel err on
own-CoC tokens" figure could not see that -- it showed only the stream that was bought,
in-sample.

Two readings are kept apart deliberately:
  relative error   per-stream ||dY|| / ||Y||, i.e. how badly that stream is served
  energy share     that stream's share of the dense output energy -- what an error
                   there costs the model. Vision is 95.6% of tokens but its energy
                   share is what decides whether its +1.1% outweighs CoC's -12%.

Usage:
  .venv/bin/python experiments/head_analysis/analyze_streamerr.py
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
ACCENT, GOOD, WARN = "#D97757", "#008300", "#eda100"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": CARD,
                     "axes.edgecolor": "#E8E6DC", "text.color": TEXT,
                     "axes.labelcolor": TEXT, "xtick.color": MUTED,
                     "ytick.color": MUTED, "font.size": 9, "axes.titlesize": 10})
ARM_COLOR = {"dualr": MUTED, "dualr_rep": "#8C8878", "dualr_w": ACCENT,
             "dualr_wl": "#2F6FBF"}
LABEL = {"vision": "vision", "prompt_text": "prompt text (instruction)",
         "hist": "ego history", "sink": "sink", "coc": "own CoC"}
# each arm's H composition, from the supernet metadata the runner copied in
FIT_W = {
    "dualr": {"vision": .25, "prompt_text": .25, "hist": .25, "sink": .25, "coc": 0.0},
    "dualr_rep": {"vision": .7223, "prompt_text": .1656, "hist": .0418, "sink": .0110,
                  "coc": 0.0},
    "dualr_w": {"vision": .7223 * .84, "prompt_text": .1656 * .84, "hist": .0418 * .84,
                "sink": .0110 * .84, "coc": .16},
    "dualr_wl": {"vision": .7223 * .96, "prompt_text": .1656 * .96, "hist": .0418 * .96,
                 "sink": .0110 * .96, "coc": .04},
}
BOOT = 10000


def med_ci(d, seed=0):
    g = np.random.default_rng(seed)
    b = np.median(np.asarray(d)[g.integers(0, len(d), (BOOT, len(d)))], axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(np.median(d)), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="streamerr_v1")
    ap.add_argument("--out", default="streamerr_v1")
    args = ap.parse_args()
    M = json.loads((REPO / "outputs" / args.exp_id / "metrics.json").read_text())
    S, A = M["streams"], M["arms"]
    out = REPO / "outputs" / args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)
    n_layers = len(M["err"][A[0]][f"o_{S[0]}"])
    res, lines = {"gates": {}}, []

    def e(a, k, s):
        return np.array(M["err"][a][f"{k}_{s}"])

    def den(k, s):
        return np.array(M["den"][f"{k}_{s}"])

    lines.append(f"per-token-type reconstruction error, {M['n_clips']} held-out clips, "
                 f"{n_layers} layers")
    lines.append("")
    lines.append(f"{'arm':11s} {'module':10s} " + " ".join(f"{s:>13s}" for s in S))
    for a in A:
        for k, lab in (("o", "o_proj"), ("m", "down_proj")):
            lines.append(f"{a:11s} {lab:10s} "
                         + " ".join(f"{np.median(e(a, k, s)):13.4f}" for s in S))

    # --- energy share: what an error on a stream actually costs -----------------
    lines += ["", "dense output energy share per stream (what an error there costs):"]
    share = {}
    for k, lab in (("o", "o_proj"), ("m", "down_proj")):
        tot = sum(den(k, s).sum() for s in S)
        share[k] = {s: float(den(k, s).sum() / tot) for s in S}
        lines.append(f"  {lab:10s} " + "  ".join(f"{s} {100 * share[k][s]:5.2f}%" for s in S))
    tokshare = {s: M["token_counts"][s] / sum(M["token_counts"].values()) for s in S}
    lines.append("  token share " + "  ".join(f"{s} {100 * tokshare[s]:5.2f}%" for s in S))
    res["energy_share"], res["token_share"] = share, tokshare

    # --- H1: the exchange, CoC share 0 -> 0.16 with everything else held --------
    lines += ["", "H1 exchange: dualr_w - dualr_rep (CoC share 0 -> 0.16), paired over layers"]
    res["gates"]["H1"] = {}
    for k, lab in (("o", "o_proj"), ("m", "down_proj")):
        for s in S:
            m, lo, hi = med_ci(e("dualr_w", k, s) - e("dualr_rep", k, s))
            res["gates"]["H1"][f"{k}_{s}"] = {"med": m, "lo": lo, "hi": hi}
            star = "*" if lo > 0 or hi < 0 else " "
            lines.append(f"  {lab:10s} {s:12s} {m:+.4f} [{lo:+.4f},{hi:+.4f}]{star}")
    # energy-weighted net: sum_s share_s * (err_w^2 - err_rep^2) -- squared because
    # energy adds, not relative error
    lines += ["", "  energy-weighted net change (negative = the trade paid off):"]
    for k, lab in (("o", "o_proj"), ("m", "down_proj")):
        net = sum(share[k][s] * (np.median(e("dualr_w", k, s)) ** 2
                                 - np.median(e("dualr_rep", k, s)) ** 2) for s in S)
        contrib = {s: share[k][s] * (np.median(e("dualr_w", k, s)) ** 2
                                     - np.median(e("dualr_rep", k, s)) ** 2) for s in S}
        res["gates"].setdefault("H1_energy", {})[k] = {"net": float(net),
                                                       "by_stream": contrib}
        lines.append(f"  {lab:10s} net {net:+.5f}  ("
                     + "  ".join(f"{s} {contrib[s]:+.5f}" for s in S) + ")")

    # --- H2: does per-stream error follow the fit weight? -----------------------
    lines += ["", ("H2 within-arm Spearman(fit weight, error) -- negative would mean "
                   "error follows what you put in:")]
    res["gates"]["H2"] = {}
    for a in A:
        for k, lab in (("o", "o_proj"), ("m", "down_proj")):
            r = stats.spearmanr([FIT_W[a][s] for s in S],
                                [np.median(e(a, k, s)) for s in S])
            res["gates"]["H2"][f"{a}_{k}"] = {"rho": float(r.statistic),
                                              "p": float(r.pvalue)}
            lines.append(f"  {a:11s} {lab:10s} rho {r.statistic:+.3f} (p={r.pvalue:.2f})")

    # --- H3: which streams swing most across arms -------------------------------
    lines += ["", "H3 spread across arms (max - min of the per-arm median):"]
    res["gates"]["H3"] = {}
    for s in S:
        vals = [np.median(e(a, k, s)) for a in A for k in ("o", "m")]
        res["gates"]["H3"][s] = {"min": float(min(vals)), "max": float(max(vals)),
                                 "spread": float(max(vals) - min(vals)),
                                 "tokens": M["token_counts"][s]}
        lines.append(f"  {s:12s} tokens {M['token_counts'][s]:7d}  "
                     f"{min(vals):.3f}~{max(vals):.3f}  spread {max(vals) - min(vals):.3f}")

    # --- plots -------------------------------------------------------------------
    fig, axes = plt.subplots(2, len(S), figsize=(3.1 * len(S), 6), sharex=True)
    for row, (k, lab) in enumerate((("o", "o_proj"), ("m", "down_proj"))):
        for col, s in enumerate(S):
            ax = axes[row, col]
            for a in A:
                ax.plot(range(n_layers), e(a, k, s), lw=1.3, color=ARM_COLOR[a],
                        label=a if (row == 0 and col == 0) else None)
            ax.set_title(f"{lab} · {LABEL[s]}", fontsize=9)
            if row == 1:
                ax.set_xlabel("VLM layer")
            if col == 0:
                ax.set_ylabel("relative output error")
            ax.set_ylim(0, None)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "plots" / "layers.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, (k, lab) in zip(axes, (("o", "o_proj"), ("m", "down_proj"))):
        d = [e("dualr_w", k, s) - e("dualr_rep", k, s) for s in S]
        m = [np.median(x) for x in d]
        errs = np.array([[m[i] - med_ci(d[i])[1] for i in range(len(S))],
                         [med_ci(d[i])[2] - m[i] for i in range(len(S))]])
        cols = [ACCENT if v > 0 else GOOD for v in m]
        ax.bar(range(len(S)), m, 0.6, color=cols, yerr=errs, capsize=3, ecolor=MUTED,
               error_kw={"lw": 1})
        ax.axhline(0, color=TEXT, lw=0.8)
        ax.set_xticks(range(len(S)), [LABEL[s] for s in S], rotation=20, ha="right")
        ax.set_ylabel("Δ relative error")
        ax.set_title(f"{lab}: buying CoC into the Hessian (0 → 0.16)")
    fig.tight_layout()
    fig.savefig(out / "plots" / "exchange.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, (k, lab) in zip(axes, (("o", "o_proj"), ("m", "down_proj"))):
        x = np.arange(len(S))
        ax.bar(x - 0.2, [100 * tokshare[s] for s in S], 0.4, color=MUTED, label="token share")
        ax.bar(x + 0.2, [100 * share[k][s] for s in S], 0.4, color=ACCENT, label="energy share")
        ax.set_yscale("log")
        ax.set_xticks(x, [LABEL[s] for s in S], rotation=20, ha="right")
        ax.set_ylabel("% of total (log)")
        ax.set_title(f"{lab}: tokens vs dense output energy")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "plots" / "shares.png", dpi=150)
    plt.close(fig)

    (out / "metrics_analysis.json").write_text(json.dumps(res, indent=1))
    text = "\n".join(lines)
    (out / "streamerr_summary.txt").write_text(text + "\n")
    print(text)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
