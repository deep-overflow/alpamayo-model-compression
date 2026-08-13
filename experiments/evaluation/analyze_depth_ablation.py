"""Read the per-layer ablation and decide whether depth pruning has room.

Every layer is scored against the *same clips and seeds* as the unmasked reference in
the same run, so the paired delta cancels clip difficulty and the GPU architecture.

Gate D (pre-registered): at least three layers with a median dminADE below +0.05 m --
the smallest effect this protocol resolves. dNLL is reported beside it because the
whole point of this project is that trajectory metrics alone hide reasoning damage:
`integrated_mag` looked fine open-loop while its CoC degenerated 86% closed-loop.

minADE deltas are heavy-tailed (a broken layer lands at +30 m), so the median and the
Wilcoxon are the primary readings and the mean is shown alongside rather than trusted.

Usage:
  python experiments/evaluation/analyze_depth_ablation.py
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import sys

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

GATE_D_M = 0.05      # smallest dminADE the protocol resolves
GATE_D_N = 3         # layers needed under it for the axis to stay open


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="depth_ablation")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    exp_dir = REPO / "outputs" / args.exp_id
    out_dir = REPO / "outputs" / (args.out or f"{args.exp_id}_summary")
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    rows = {}
    for p in sorted(exp_dir.glob("rows_s*of*.json")):
        for r in json.loads(p.read_text()):
            rows[r["clip_id"]] = r
    rows = list(rows.values())
    layers = sorted({int(k) for r in rows for k in r["layers"]})
    print(f"{len(rows)} clips x {len(layers)} layers", flush=True)

    base_ade = np.array([r["baseline"]["minADE"] for r in rows])
    base_nll = np.array([r["baseline"]["nll"] for r in rows])

    res = {}
    for li in layers:
        a = np.array([r["layers"][str(li)]["minADE"] for r in rows])
        n = np.array([r["layers"][str(li)]["nll"] for r in rows])
        d_ade, d_nll = a - base_ade, n - base_nll
        mean, lo, hi = el.paired_bootstrap_ci(d_ade)
        res[li] = {
            "dADE_median": float(np.median(d_ade)), "dADE_mean": mean, "dADE_ci": [lo, hi],
            "dADE_p": float(wilcoxon(d_ade).pvalue) if np.any(d_ade != 0) else 1.0,
            "dNLL_median": float(np.median(d_nll)), "dNLL_mean": float(d_nll.mean()),
            "frac_worse": float(np.mean(d_ade > 0)),
            "minADE_median": float(np.median(a)),
        }

    passing = [li for li in layers if res[li]["dADE_median"] < GATE_D_M]
    gate = len(passing) >= GATE_D_N

    order = sorted(layers, key=lambda li: res[li]["dADE_median"])
    lines = [f"per-layer kv-only ablation -- {len(rows)} clips",
             f"baseline minADE median {np.median(base_ade):.3f}  "
             f"NLL median {np.median(base_nll):.3f}", "",
             f"{'layer':>5} {'dADE med':>9} {'dADE mean':>10} {'[95% CI]':>20} "
             f"{'p':>9} {'worse%':>7} {'dNLL med':>9}"]
    for li in order:
        r = res[li]
        ci = r["dADE_ci"]
        lines.append(f"{li:5d} {r['dADE_median']:+9.3f} {r['dADE_mean']:+10.3f} "
                     f"[{ci[0]:+.3f},{ci[1]:+.3f}]".rjust(20)
                     + f" {r['dADE_p']:9.2e} {r['frac_worse'] * 100:6.0f}% "
                       f"{r['dNLL_median']:+9.3f}")
    lines += ["", f"GATE D: {len(passing)} layers with median dADE < +{GATE_D_M} m "
                  f"(need {GATE_D_N}) -> {'PASS' if gate else 'FAIL'}"]
    if passing:
        lines.append("  " + ", ".join(
            f"L{li}(dADE {res[li]['dADE_median']:+.3f}, dNLL {res[li]['dNLL_median']:+.3f})"
            for li in sorted(passing, key=lambda x: res[x]["dADE_median"])))

    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps({
        "n_clips": len(rows), "layers": {str(k): v for k, v in res.items()},
        "baseline": {"minADE_median": float(np.median(base_ade)),
                     "nll_median": float(np.median(base_nll))},
        "gate_d": {"threshold_m": GATE_D_M, "need": GATE_D_N,
                   "passing": passing, "pass": bool(gate)},
    }, indent=2))

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    x = np.array(layers)
    med = np.array([res[li]["dADE_median"] for li in layers])
    ax = axes[0]
    ax.bar(x, np.clip(med, None, 2.0), color=[C2 if m < GATE_D_M else C1 for m in med])
    ax.axhline(GATE_D_M, color=C4, ls="--", lw=1, label=f"gate D  +{GATE_D_M} m")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("median dminADE (m)")
    ax.set_title("cost of removing each layer (clipped at 2 m; green passes gate D)")
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1]
    ax.bar(x, [res[li]["dNLL_median"] for li in layers], color=C3)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("VLM layer")
    ax.set_ylabel("median dNLL")
    ax.set_title("reasoning damage: can the ablated model still predict the intact CoC")
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "per_layer.png", dpi=150)
    plt.close(fig)

    print("\n".join(lines), flush=True)
    print("saved ->", out_dir, flush=True)


if __name__ == "__main__":
    main()
