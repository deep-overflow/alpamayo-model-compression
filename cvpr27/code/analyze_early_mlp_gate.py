"""Step 1 of the deeper layer 1-4 MLP cut: paired loss deltas and the G1 / G2 gates.

Plan: cvpr27/plans/2026-10-29_early-mlp-cut.md (section 3). Reads run_token_ablation.py's per-clip
CoC NLL, FM loss and minADE (K=8, min over all eight) for dense and the ten configs of
make_early_mlp_masks.py, all paired per clip on calib_rd100_a.

  G1 (descriptive)  is the Taylor ranking causal in layers 1-4: t64 - r64 and r10 - e10 > 0 on both
                    losses, CI excluding 0
  G2 (decides step 2)  e10 / e05 - b0 against the proportional threshold
                    (b0 - dense) * (extra removed / b0 removed), per loss:
                    FREE if CI hi < threshold, REJECT if CI lo >= threshold, else INCONCLUSIVE;
                    UNDEFINED when b0 - dense <= 0 on that loss (no price to compare with).
                    An arm REJECTed on either loss leaves step 2.

Bootstrap: paired over clips, 200,000 draws (gates sit near their thresholds), mean and median.

Usage:
  .venv/bin/python cvpr27/code/analyze_early_mlp_gate.py [--exp-id cvpr27/early_mlp]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG, INK, MUTED, GRID = "#FAF9F5", "#29261B", "#6B6555", "#E8E6DC"
C_PT = "#2a78d6"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.size": 9, "axes.grid": True, "axes.axisbelow": True,
})
REPO = Path(__file__).resolve().parents[2]
METRICS = (("fm", "FM loss"), ("nll", "CoC NLL"), ("ade", "minADE@8"))
ORDER = ("e10", "e05", "r10", "t64", "r64", "e05_l1", "e05_l2", "e05_l3", "e05_l4")
DRAWS = 200_000


def boot_ci(d, rng, draws=DRAWS, chunk=20_000):
    """d: (n,) paired differences -> (lo, hi) 95% CI of the mean."""
    means = []
    for s in range(0, draws, chunk):
        idx = rng.integers(0, len(d), size=(min(chunk, draws - s), len(d)))  # (chunk, n)
        means.append(d[idx].mean(1))
    lo, hi = np.percentile(np.concatenate(means), [2.5, 97.5])
    return float(lo), float(hi)


def verdict(lo, hi, thr):
    # the threshold is the existing cut's average price, which only exists if b0 costs something
    if thr <= 0:
        return "UNDEFINED"
    return "FREE" if hi < thr else "REJECT" if lo >= thr else "INCONCLUSIVE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="cvpr27/early_mlp")
    args = ap.parse_args()
    root = REPO / "outputs" / args.exp_id
    run = json.loads((root / "gate_loss" / "metrics.json").read_text())
    mcfg = json.loads((root / "masks_config.json").read_text())["configs"]
    pc = {n.removeprefix("arm_"): {k: np.array(v[k], dtype=float) for k in ("nll", "fm", "ade")}
          for n, v in run["per_clip"].items()}  # name -> metric -> (n_clips,)
    n = run["n_clips"]
    rng = np.random.default_rng(0)

    absolute = {c: {k: {"mean": float(x.mean()), "median": float(np.median(x))} for k, x in v.items()}
                for c, v in pc.items()}
    deltas = {}
    for c in ORDER:
        deltas[c] = {}
        for k, _ in METRICS:
            d = pc[c][k] - pc["b0"][k]  # (n,)
            lo, hi = boot_ci(d, rng)
            deltas[c][k] = {"mean": float(d.mean()), "median": float(np.median(d)), "lo": lo, "hi": hi}

    g1 = {}
    for name, a, b in (("t64-r64", "t64", "r64"), ("r10-e10", "r10", "e10")):
        g1[name] = {}
        for k in ("fm", "nll"):
            d = pc[a][k] - pc[b][k]
            lo, hi = boot_ci(d, rng)
            g1[name][k] = {"mean": float(d.mean()), "lo": lo, "hi": hi, "positive": lo > 0}

    g2 = {}
    for arm in ("e10", "e05"):
        ratio = mcfg[arm]["extra_over_b0_removed"]
        g2[arm] = {"ratio": ratio}
        for k in ("fm", "nll"):
            cost = absolute["b0"][k]["mean"] - absolute["dense"][k]["mean"]
            thr = cost * ratio
            dd = deltas[arm][k]
            g2[arm][k] = {"b0_minus_dense": cost, "threshold": thr, "delta": dd["mean"],
                          "lo": dd["lo"], "hi": dd["hi"], "verdict": verdict(dd["lo"], dd["hi"], thr)}
        g2[arm]["to_step2"] = all(g2[arm][k]["verdict"] != "REJECT" for k in ("fm", "nll"))

    out = root / "gate_analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({
        "plan": "cvpr27/plans/2026-10-29_early-mlp-cut.md (section 3)", "run": f"{args.exp_id}/gate_loss",
        "n_clips": n, "complete": n == 100, "bootstrap_draws": DRAWS, "bootstrap_seed": 0,
        "metrics": {k: lab for k, lab in METRICS}, "masks": {c: mcfg[c]["path"] for c in mcfg}}, indent=2))
    (out / "metrics.json").write_text(json.dumps(
        {"n_clips": n, "absolute": absolute, "delta_vs_b0": deltas, "g1": g1, "g2": g2}, indent=2))

    lines = [f"Step 1 loss check on calib_rd100_a: {n} clips{'' if n == 100 else ' (PARTIAL)'}",
             "", "Absolute (mean / median):",
             f"{'config':<9}" + "".join(f"{lab:>22}" for _, lab in METRICS)]
    for c in ("dense", "b0") + ORDER:
        lines.append(f"{c:<9}" + "".join(
            f"{absolute[c][k]['mean']:>12.4f} / {absolute[c][k]['median']:<7.4f}" for k, _ in METRICS))
    lines += ["", "Delta vs b0: mean [95% CI] (median)"]
    for c in ORDER:
        lines.append(f"{c:<9}" + "".join(
            f"  {deltas[c][k]['mean']:+.4f} [{deltas[c][k]['lo']:+.4f}, {deltas[c][k]['hi']:+.4f}]"
            f" ({deltas[c][k]['median']:+.4f})" for k, _ in METRICS))
    lines += ["", "G1 (Taylor ranking causal): difference mean [CI], positive = CI lo > 0"]
    for name, v in g1.items():
        lines.append(f"  {name:<8}" + "".join(
            f"  {k} {v[k]['mean']:+.4f} [{v[k]['lo']:+.4f}, {v[k]['hi']:+.4f}] {'positive' if v[k]['positive'] else 'not'}"
            for k in ("fm", "nll")))
    lines += ["", "G2 (proportional threshold = (b0 - dense) * extra/b0 removed):"]
    for arm, v in g2.items():
        for k in ("fm", "nll"):
            g = v[k]
            lines.append(f"  {arm} {k:<3} delta {g['delta']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}] vs threshold "
                         f"{g['threshold']:+.4f} (b0-dense {g['b0_minus_dense']:+.4f} x {v['ratio']}) -> {g['verdict']}")
        lines.append(f"  {arm}: {'goes to step 2' if v['to_step2'] else 'leaves step 2 (REJECT)'}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # dot-and-whisker: delta vs b0 per config, one panel per loss; ticks mark the G2 thresholds
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    ys = np.arange(len(ORDER))[::-1]
    for ax, (k, lab) in zip(axes, METRICS[:2]):
        m = np.array([deltas[c][k]["mean"] for c in ORDER])
        lo = np.array([deltas[c][k]["lo"] for c in ORDER])
        hi = np.array([deltas[c][k]["hi"] for c in ORDER])
        ax.errorbar(m, ys, xerr=[m - lo, hi - m], fmt="o", color=C_PT, ms=5, lw=2, capsize=0)
        for arm in ("e10", "e05"):
            y = ys[ORDER.index(arm)]
            ax.plot([g2[arm][k]["threshold"]] * 2, [y - 0.35, y + 0.35], color=INK, lw=1.5)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_title(f"{lab}: config - b0", fontsize=9, loc="left")
        ax.set_xlabel("paired difference (mean, 95% CI)")
    axes[0].set_yticks(ys, ORDER)
    axes[1].annotate("| = G2 threshold", xy=(1, 1.01), xycoords="axes fraction", ha="right",
                     va="bottom", fontsize=7, color=MUTED)
    fig.tight_layout()
    fig.savefig(out / "gate_loss.png", dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
