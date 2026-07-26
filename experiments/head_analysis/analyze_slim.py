"""Slim-vs-baseline latency analysis: measured stage medians vs the analytic predictions.

Reads the Blackwell profile runs (baseline = existing profile_blackwell_s{0,13}) and
produces the stage table, gather-overhead isolation (identity - baseline), and the
normalized end-to-end speedup. Decode is compared per token and normalized to the
baseline's median CoC length -- the slim integrated model's degenerate CoC rollouts run
longer, which would otherwise inflate its decode stage unfairly.

Usage: python analyze_slim.py
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

RUNS = {
    "baseline": ["profile_blackwell_s0", "profile_blackwell_s13"],
    "identity": ["profile_identity_s0", "profile_identity_s13"],
    "slim_int": ["profile_slim_int_s0", "profile_slim_int_s13"],
    "slim_coc": ["profile_slim_coc_s0", "profile_slim_coc_s13"],
}
PRED = {"slim_int": 965.0, "slim_coc": 1093.0}  # analytic predictions (comprehensive report)
BASELINE_EST = 1336.2


def load(names):
    recs = []
    for n in names:
        m = json.loads((REPO / "outputs" / n / "metrics.json").read_text())
        recs += [r for r in m["per_clip"] if not r.get("warmup")]
    return recs


def stages(recs):
    med = lambda k: float(np.median([r[k] for r in recs]))  # noqa: E731
    per_tok = float(np.median([r["decode_ms"] / r["decode_steps"] for r in recs]))
    # rollout_other is per-decode-step generate-loop overhead -> normalize like decode,
    # else the slim-integrated model's degenerate long CoC rollouts inflate it unfairly
    ovh_tok = float(np.median([r["rollout_other_ms"] / r["decode_steps"] for r in recs]))
    return {
        "vit": med("vit_ms"), "prefill": med("prefill_ms"),
        "decode": med("decode_ms"), "decode_steps": med("decode_steps"),
        "decode_per_tok": per_tok, "expert": med("expert_ms"),
        "ovh_per_tok": ovh_tok, "denoise_other": med("denoise_other_ms"),
        "total_wall": med("total_wall_ms"), "n": len(recs),
        "peak_gb": med("rollout_peak_gb"),
    }


def main():
    out_dir = REPO / "outputs" / "slim_profile_summary"
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    S = {k: stages(load(v)) for k, v in RUNS.items()}
    ref_steps = S["baseline"]["decode_steps"]
    for k, s in S.items():
        # end-to-end normalized to the baseline's median CoC length
        s["overhead"] = s["ovh_per_tok"] * ref_steps + s["denoise_other"]
        s["norm_total"] = (s["vit"] + s["prefill"] + s["decode_per_tok"] * ref_steps
                          + s["expert"] + s["overhead"])
    for k, s in S.items():
        s["speedup"] = S["baseline"]["norm_total"] / s["norm_total"]

    lines = [f"Slim latency (Blackwell, n per run below; decode normalized to "
             f"{ref_steps:.0f} CoC tokens)\n"]
    hdr = (f"{'run':10s} {'n':>3s} {'ViT':>7s} {'prefill':>8s} {'dec/tok':>8s} "
           f"{'expert':>7s} {'ovh':>6s} {'norm total':>10s} {'speedup':>8s} {'peak':>6s}")
    lines.append(hdr)
    for k, s in S.items():
        lines.append(f"{k:10s} {s['n']:3d} {s['vit']:7.1f} {s['prefill']:8.1f} "
                     f"{s['decode_per_tok']:8.2f} {s['expert']:7.1f} {s['overhead']:6.1f} "
                     f"{s['norm_total']:10.1f} {s['speedup']:7.2f}x {s['peak_gb']:5.1f}G")
    lines.append("")
    gather = S["identity"]["norm_total"] - S["baseline"]["norm_total"]
    lines.append(f"gather-path overhead (identity - baseline): {gather:+.1f} ms "
                 f"(prefill {S['identity']['prefill'] - S['baseline']['prefill']:+.1f}, "
                 f"dec/tok {S['identity']['decode_per_tok'] - S['baseline']['decode_per_tok']:+.2f}, "
                 f"expert {S['identity']['expert'] - S['baseline']['expert']:+.1f})")
    for k, p in PRED.items():
        lines.append(f"{k}: measured {S[k]['norm_total']:.0f} ms ({S[k]['speedup']:.2f}x) "
                     f"vs predicted {p:.0f} ms ({BASELINE_EST / p:.2f}x)")
    txt = "\n".join(lines)
    print(txt)
    (out_dir / "summary.txt").write_text(txt + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(
        {"stages": S, "ref_decode_steps": ref_steps, "predictions": PRED}, indent=2))

    # stacked stage bars: baseline / identity / slim_int / slim_coc + predictions
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    keys = ["vit", "prefill", "_decode_norm", "expert", "overhead"]
    labels = ["ViT", "prefill", f"decode ({ref_steps:.0f} tok)", "expert denoise", "overhead"]
    colors = [C3, C1, C4, C2, MUTED]
    names = list(S.keys())
    for i, name in enumerate(names):
        s = dict(S[name])
        s["_decode_norm"] = s["decode_per_tok"] * ref_steps
        bottom = 0.0
        for k, lab, c in zip(keys, labels, colors):
            ax.bar(i, s[k], 0.62, bottom=bottom, color=c,
                   label=lab if i == 0 else None)
            bottom += s[k]
        ax.text(i, bottom + 12, f"{s['norm_total']:.0f} ms\n{s['speedup']:.2f}x",
                ha="center", fontsize=9, color=INK)
    for i, (k, p) in enumerate(PRED.items()):
        xi = names.index(k)
        ax.hlines(p, xi - 0.38, xi + 0.38, color=INK, ls="--", lw=1.2)
        ax.text(xi + 0.4, p, f"pred {p:.0f}", fontsize=8, color=MUTED, va="center")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(["baseline\n(stock)", "identity\n(gather ctrl)",
                        "slim\nintegrated_mag", "slim\ncocsafe_full_r20"])
    ax.set_ylabel("ms (median, CoC-normalized)")
    ax.set_title("Measured stage latency: physical removal vs baseline (Blackwell)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "slim_stages.png", dpi=150)
    print("saved ->", out_dir)


if __name__ == "__main__":
    main()
