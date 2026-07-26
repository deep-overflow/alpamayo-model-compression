"""Aggregate the four stage-profile runs into one comparison + plots.

Two GPU models x two clip shards. Same-hardware shards give the noise floor;
same-shard cross-hardware pairs isolate the hardware effect, which is what
separates compute-bound stages from bandwidth-bound ones.

Usage: python aggregate_profile.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path("/workspace/alpamayo-model-compression")
OUT = REPO / "outputs" / "profile_summary"
BG, INK, MUTED, GRID = "#FAF9F5", "#29261B", "#6B6555", "#E8E6DC"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
C5, C6 = "#1baf7a", "#eb6834"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})

RUNS = ["profile_blackwell_s0", "profile_blackwell_s13", "profile_ada_s0", "profile_ada_s13"]
STAGES = [
    ("vit_ms", "ViT", C4),
    ("prefill_ms", "VLM prefill", C1),
    ("decode_ms", "CoC decode", C3),
    ("expert_ms", "Expert denoise", C2),
    ("rollout_other_ms", "rollout overhead", C5),
    ("denoise_other_ms", "denoise overhead", C6),
]
# analytic estimate made before profiling, for the estimate-vs-measured figure
ESTIMATE_MS = {"ViT": 35.0, "VLM prefill": 162.0, "CoC decode": 73.0, "Expert denoise": 10.0}
ESTIMATE_TFLOP = {"ViT": 10.4, "VLM prefill": 48.5, "CoC decode": 0.23, "Expert denoise": 3.0}


def load():
    runs = {}
    for name in RUNS:
        d = REPO / "outputs" / name
        cfg = json.loads((d / "config.json").read_text())
        met = json.loads((d / "metrics.json").read_text())
        live = [r for r in met["per_clip"] if not r["warmup"]]
        runs[name] = {"cfg": cfg, "live": live, "flops": met.get("flops"),
                      "hw": "Blackwell" if "Blackwell" in cfg["gpu"] else "Ada",
                      "gpu": cfg["gpu"]}
    return runs


def med(live, key):
    return float(np.median([r[key] for r in live]))


def stage_flops(runs):
    """Split measured FLOPs into the four stages (TFLOP)."""
    f = next(r["flops"] for r in runs.values() if r["flops"])
    mods = f["rollout"]["by_module"]

    def total(prefix):
        return sum(sum(ops.values()) for m, ops in mods.items() if m == prefix)

    visual = total("Qwen3VLForConditionalGeneration.model.visual")
    lm = total("Qwen3VLForConditionalGeneration.model.language_model")
    rollout_total = f["rollout"]["total"]
    denoise = f["denoise"]["total"]
    # decode is a handful of single-token forwards; its share of `lm` is
    # n_decode / prefill_len, small enough that any split error is negligible
    live = next(r["live"] for r in runs.values() if r["flops"])
    frac = med(live, "decode_steps") / med(live, "prefill_len")
    decode = lm * frac
    return {
        "ViT": visual / 1e12,
        "VLM prefill": (lm - decode) / 1e12,
        "CoC decode": decode / 1e12,
        "Expert denoise": denoise / 1e12,
        "_rollout_total": rollout_total / 1e12,
        "_denoise_total": denoise / 1e12,
        "_other_rollout": (rollout_total - visual - lm) / 1e12,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(exist_ok=True)
    runs = load()
    fl = stage_flops(runs)

    by_hw = {}
    for hw in ("Blackwell", "Ada"):
        rs = [r for r in runs.values() if r["hw"] == hw]
        live = [c for r in rs for c in r["live"]]
        by_hw[hw] = {
            "gpu": rs[0]["gpu"],
            "n": len(live),
            "stages": {label: med(live, key) for key, label, _ in STAGES},
            "total": med(live, "total_wall_ms"),
            "shards": {r["cfg"]["clip_offset"]: {label: med(r["live"], key)
                                                 for key, label, _ in STAGES} for r in rs},
            "kv_mb": med(live, "kv_cache_mb"),
            "peak_gb": med(live, "rollout_peak_gb"),
            "coc": med(live, "coc_len"),
            "live": live,
        }

    # ---- achieved throughput; prefill is the empirical compute roof ----
    thr = {}
    for hw, d in by_hw.items():
        thr[hw] = {s: fl[s] / (d["stages"][s] / 1e3) for s in ESTIMATE_MS}

    lines = ["Alpamayo-1.5-10B measured stage profile", ""]
    lines.append("FLOPs per inference (measured, hardware-independent):")
    for s in ESTIMATE_MS:
        lines.append(f"  {s:<16}{fl[s]:>8.2f} TFLOP   (estimate was {ESTIMATE_TFLOP[s]:.2f})")
    lines.append(f"  {'TOTAL':<16}{fl['_rollout_total'] + fl['_denoise_total']:>8.2f} TFLOP")
    lines.append("")
    for hw, d in by_hw.items():
        lines.append(f"{d['gpu']}  (n={d['n']} clips, 2 shards)")
        lines.append(f"  {'stage':<18}{'ms':>9}{'%':>7}{'TFLOP/s':>10}{'shard spread':>14}")
        for key, label, _ in STAGES:
            v = d["stages"][label]
            sp = [s[label] for s in d["shards"].values()]
            spread = abs(sp[0] - sp[1]) / max(np.mean(sp), 1e-9) * 100
            t = f"{thr[hw][label]:>10.1f}" if label in thr[hw] else " " * 10
            lines.append(f"  {label:<18}{v:>9.1f}{v / d['total'] * 100:>6.1f}%{t}{spread:>13.1f}%")
        lines.append(f"  {'TOTAL (wall)':<18}{d['total']:>9.1f}{100.0:>6.1f}%")
        lines.append(f"  KV cache {d['kv_mb']:.0f} MB | peak {d['peak_gb']:.1f} GB "
                     f"| CoC {d['coc']:.0f} tok")
        lines.append("")

    b, a = by_hw["Blackwell"]["stages"], by_hw["Ada"]["stages"]
    lines.append("Hardware sensitivity (Ada ms / Blackwell ms) - high = compute bound:")
    for key, label, _ in STAGES:
        lines.append(f"  {label:<18}{a[label] / b[label]:>8.2f}x")
    lines.append("")
    lines.append("Estimate vs measured latency share (Ada):")
    tot_a = by_hw["Ada"]["total"]
    for s in ESTIMATE_MS:
        est_share = ESTIMATE_MS[s] / sum(ESTIMATE_MS.values()) * 100
        lines.append(f"  {s:<16}estimated {est_share:>5.1f}%   measured "
                     f"{by_hw['Ada']['stages'][s] / tot_a * 100:>5.1f}%")
    text = "\n".join(lines)
    (OUT / "summary.txt").write_text(text + "\n")
    print(text)

    json.dump({"flops_tflop": fl,
               "by_hardware": {k: {kk: vv for kk, vv in v.items() if kk != "live"}
                               for k, v in by_hw.items()},
               "achieved_tflops": thr},
              open(OUT / "metrics.json", "w"), indent=2)

    plots(by_hw, fl, thr)
    print("plots ->", OUT / "plots")


def plots(by_hw, fl, thr):
    P = OUT / "plots"
    labels = [lbl for _, lbl, _ in STAGES]
    colors = [c for _, _, c in STAGES]

    # 1) stacked stage breakdown per hardware
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for i, (hw, d) in enumerate(by_hw.items()):
        left = 0
        for lbl, col in zip(labels, colors):
            v = d["stages"][lbl]
            ax.barh(i, v, left=left, color=col, edgecolor=BG, height=0.62,
                    label=lbl if i == 0 else None)
            if v / d["total"] > 0.06:
                ax.text(left + v / 2, i, f"{v:.0f}", ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
            left += v
        ax.text(left + 18, i, f"{d['total']:.0f} ms", va="center", fontsize=9, color=MUTED)
    ax.set_yticks(range(len(by_hw)))
    ax.set_yticklabels([d["gpu"].replace("NVIDIA ", "") for d in by_hw.values()], fontsize=9)
    ax.set_xlabel("latency (ms)")
    ax.set_title("Measured stage breakdown of one Alpamayo-1.5 inference")
    ax.legend(frameon=False, fontsize=8.5, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    ax.set_xlim(0, max(d["total"] for d in by_hw.values()) * 1.12)
    fig.tight_layout(); fig.savefig(P / "stage_breakdown.png", dpi=150); plt.close(fig)

    # 2) FLOPs share vs latency share -- the mismatch that broke the estimate
    keys = list(ESTIMATE_MS)
    ftot = sum(fl[s] for s in keys)
    d = by_hw["Ada"]
    ltot = sum(d["stages"][s] for s in keys)
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - 0.2, [fl[s] / ftot * 100 for s in keys], 0.38, color=C1, label="share of FLOPs")
    ax.bar(x + 0.2, [d["stages"][s] / ltot * 100 for s in keys], 0.38, color=C3,
           label="share of latency")
    ax.set_xticks(x); ax.set_xticklabels(keys, fontsize=9)
    ax.set_ylabel("% of total")
    ax.set_title("FLOPs do not predict latency: work vs time by stage")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.7)
    fig.tight_layout(); fig.savefig(P / "flops_vs_latency.png", dpi=150); plt.close(fig)

    # 3) achieved throughput -- prefill is the empirical compute roof
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, (hw, col) in enumerate([("Blackwell", C1), ("Ada", C4)]):
        ax.bar(np.arange(len(keys)) + (i - 0.5) * 0.38, [thr[hw][s] for s in keys], 0.36,
               color=col, label=hw)
    for hw, col, ls in [("Blackwell", C1, "--"), ("Ada", C4, ":")]:
        ax.axhline(thr[hw]["VLM prefill"], color=col, ls=ls, lw=1.2)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(keys))); ax.set_xticklabels(keys, fontsize=9)
    ax.set_ylabel("achieved TFLOP/s (log)")
    ax.set_title("Achieved throughput per stage; dashed lines = prefill (compute roof)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.7)
    fig.tight_layout(); fig.savefig(P / "achieved_throughput.png", dpi=150); plt.close(fig)

    # 4) hardware sensitivity
    b, a = by_hw["Blackwell"]["stages"], by_hw["Ada"]["stages"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ratios = [a[s] / b[s] for s in keys]
    ax.bar(keys, ratios, color=[C1 if r > 1.3 else C2 for r in ratios], width=0.6)
    ax.axhline(1.0, color=MUTED, lw=1)
    for i, r in enumerate(ratios):
        ax.text(i, r + 0.02, f"{r:.2f}x", ha="center", fontsize=9, color=INK)
    ax.set_ylabel("Ada ms / Blackwell ms")
    ax.set_title("Hardware sensitivity: high = compute bound, ~1 = bandwidth bound")
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(keys, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.7)
    fig.tight_layout(); fig.savefig(P / "hardware_sensitivity.png", dpi=150); plt.close(fig)

    # 5) decode cost scales with CoC length
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for hw, col in [("Blackwell", C1), ("Ada", C4)]:
        live = by_hw[hw]["live"]
        n = np.array([r["decode_steps"] for r in live])
        t = np.array([r["decode_ms"] for r in live])
        ax.scatter(n, t, s=40, color=col, edgecolor=BG, lw=1.1, zorder=3, label=hw)
        if len(set(n.tolist())) > 1:
            k, c0 = np.polyfit(n, t, 1)
            xs = np.linspace(n.min(), n.max(), 10)
            ax.plot(xs, k * xs + c0, color=col, lw=1.4, alpha=0.7, zorder=2)
            ax.annotate(f"{hw}: {k:.1f} ms/token", xy=(0.04, 0.92 - 0.08 * (hw == "Ada")),
                        xycoords="axes fraction", fontsize=9, color=col)
    ax.set_xlabel("CoC tokens generated"); ax.set_ylabel("CoC decode latency (ms)")
    ax.set_title("Decode cost is linear in CoC length")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(color=GRID, lw=0.7, zorder=0)
    fig.tight_layout(); fig.savefig(P / "decode_vs_coclen.png", dpi=150); plt.close(fig)

    # 6) pre-profiling estimate vs measurement
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(keys))
    axes[0].bar(x - 0.2, [ESTIMATE_TFLOP[s] for s in keys], 0.38, color=MUTED, label="estimated")
    axes[0].bar(x + 0.2, [fl[s] for s in keys], 0.38, color=C1, label="measured")
    axes[0].set_ylabel("TFLOP"); axes[0].set_title("FLOPs: estimate was accurate")
    ltot = sum(by_hw["Ada"]["stages"][s] for s in keys)
    etot = sum(ESTIMATE_MS.values())
    axes[1].bar(x - 0.2, [ESTIMATE_MS[s] / etot * 100 for s in keys], 0.38,
                color=MUTED, label="estimated")
    axes[1].bar(x + 0.2, [by_hw["Ada"]["stages"][s] / ltot * 100 for s in keys], 0.38,
                color=C3, label="measured")
    axes[1].set_ylabel("% of latency"); axes[1].set_title("Latency share: estimate was wrong")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(keys, fontsize=8.5, rotation=12)
        ax.legend(frameon=False, fontsize=9); ax.grid(axis="y", color=GRID, lw=0.7)
    fig.tight_layout(); fig.savefig(P / "estimate_vs_measured.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
