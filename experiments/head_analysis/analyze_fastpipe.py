"""Aggregate the Blackwell pipeline benches: pruning x CUDA-graph latency matrix.

Reads outputs/fastpipe_{base,slim_int,slim_coc}_bw/metrics.json, builds the stage
table (stock vs graphed per model), the normalized end-to-end matrix, and the stacked
stage plot. Usage: python analyze_fastpipe.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/workspace/alpamayo-model-compression")

BG = "#FAF9F5"
INK = "#29261B"
MUTED = "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

RUNS = {"baseline": "fastpipe_base_bw", "slim_int": "fastpipe_slim_int_bw",
        "slim_coc": "fastpipe_slim_coc_bw"}
REF_STEPS = 16


def load(name):
    m = json.loads((REPO / "outputs" / name / "metrics.json").read_text())
    live = [r for r in m["per_clip"] if not r["warmup"] and not r.get("denoise_capture")]
    med = lambda k: float(np.median([r[k] for r in live]))  # noqa: E731
    out = {}
    for p in ("stock", "fast"):
        steps_off = 0 if p == "stock" else 1  # fast loop runs n_steps-1 graph steps
        tok = float(np.median(
            [r[f"{p}_decode_ms"] / max(r[f"{p}_steps"] - steps_off, 1) for r in live]))
        out[p] = {
            "prefill": med(f"{p}_prefill_ms"), "decode_tok": tok,
            "denoise": med(f"{p}_denoise_ms"),
            "norm": med(f"{p}_prefill_ms") + tok * REF_STEPS + med(f"{p}_denoise_ms"),
        }
    out["n"] = len(live)
    return out


def main():
    out_dir = REPO / "outputs" / "fastpipe_summary"
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    S = {k: load(v) for k, v in RUNS.items()}
    base_stock = S["baseline"]["stock"]["norm"]

    lines = [f"pruning x CUDA-graph latency matrix (Blackwell, norm {REF_STEPS} CoC tok)\n",
             f"{'model':10s} {'path':6s} {'vit+prefill':>11s} {'dec/tok':>8s} "
             f"{'denoise':>8s} {'e2e norm':>9s} {'vs base-stock':>13s}"]
    for name, s in S.items():
        for p in ("stock", "fast"):
            d = s[p]
            lines.append(f"{name:10s} {p:6s} {d['prefill']:11.1f} {d['decode_tok']:8.2f} "
                         f"{d['denoise']:8.1f} {d['norm']:9.1f} "
                         f"{base_stock / d['norm']:12.2f}x")
    lines.append("")
    lines.append(f"graph-only gain (baseline):  "
                 f"{base_stock / S['baseline']['fast']['norm']:.2f}x")
    lines.append(f"pruning-only gain (stock):   "
                 f"{base_stock / S['slim_int']['stock']['norm']:.2f}x (integrated)")
    lines.append(f"combined (slim_int fast):    "
                 f"{base_stock / S['slim_int']['fast']['norm']:.2f}x")
    lines.append(f"combined (slim_coc fast):    "
                 f"{base_stock / S['slim_coc']['fast']['norm']:.2f}x")
    txt = "\n".join(lines)
    print(txt)
    (out_dir / "summary.txt").write_text(txt + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(
        {"ref_steps": REF_STEPS, "stages": S}, indent=2))

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    order = [("baseline", "stock"), ("baseline", "fast"), ("slim_int", "stock"),
             ("slim_int", "fast"), ("slim_coc", "stock"), ("slim_coc", "fast")]
    labels = ["baseline\nstock", "baseline\ngraphed", "slim int.\nstock",
              "slim int.\ngraphed", "slim coc.\nstock", "slim coc.\ngraphed"]
    parts = [("prefill", "ViT+prefill", C1), ("decode_tok", f"decode ({REF_STEPS} tok)", C4),
             ("denoise", "expert denoise", C2)]
    for i, (name, p) in enumerate(order):
        d = S[name][p]
        bottom = 0.0
        for key, lab, c in parts:
            v = d[key] * REF_STEPS if key == "decode_tok" else d[key]
            ax.bar(i, v, 0.62, bottom=bottom, color=c, label=lab if i == 0 else None)
            bottom += v
        ax.text(i, bottom + 15, f"{d['norm']:.0f}\n{base_stock / d['norm']:.2f}x",
                ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"ms (median, {REF_STEPS}-token normalized)")
    ax.set_title("Pruning x CUDA graphs: measured pipeline latency (Blackwell)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "fastpipe_matrix.png", dpi=150)
    print("saved ->", out_dir)


if __name__ == "__main__":
    main()
