"""Analyze the strengthened evaluation: multi-sample minADE/minFDE, buckets, CIs.

Merges the sharded runs, then for every config reports the paired delta vs baseline
with a bootstrap CI, worst-case percentiles, a regression count, and a per-bucket
breakdown. The bucket breakdown is the point: it shows whether a "free" config is
free everywhere or only on easy (cruise) clips.

Usage: python analyze_eval.py --shards eval_val_k8_s0 eval_val_k8_s25 --out eval_val_k8
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import eval_lib as el  # noqa: E402

REPO = Path("/workspace/alpamayo-model-compression")
BG, INK, MUTED, GRID = "#FAF9F5", "#29261B", "#6B6555", "#E8E6DC"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
BUCKETS = ["cruise", "turn", "decel_stop", "accel"]
BCOL = {"cruise": C1, "turn": C4, "decel_stop": C3, "accel": C2}

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})


def merge(shards):
    per_clip, buckets, clip_ids, meta, cfg_order = {}, [], [], {}, None
    for sh in shards:
        m = json.loads((REPO / "outputs" / sh / "metrics.json").read_text())
        if cfg_order is None:
            cfg_order = m["configs"]
            meta = m["meta"]
        for name in m["configs"]:
            per_clip.setdefault(name, {"ade": [], "fde": [], "nll": []})
            for k in ("ade", "fde", "nll"):
                per_clip[name][k].extend(m["per_clip"][name][k])
        buckets.extend(m["buckets"])
        clip_ids.extend(m["clip_ids"])
    return per_clip, np.array(buckets), clip_ids, meta, cfg_order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", type=str, default="eval_summary")
    args = ap.parse_args()
    per_clip, buckets, clip_ids, meta, cfg_order = merge(args.shards)
    out_dir = REPO / "outputs" / args.out
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    n = len(clip_ids)

    base_ade = np.array(per_clip["baseline"]["ade"])
    base_fde = np.array(per_clip["baseline"]["fde"])
    base_nll = np.array(per_clip["baseline"]["nll"])

    lines = [f"Strengthened evaluation -- {args.out}",
             f"  n={n} clips, K-sample minADE/minFDE, buckets from GT geometry",
             f"  bucket counts: " + ", ".join(
                 f"{b}={int((buckets == b).sum())}" for b in BUCKETS),
             f"  baseline minADE {base_ade.mean():.4f}  minFDE {base_fde.mean():.4f}  "
             f"CoC NLL {base_nll.mean():.4f}", ""]

    rows = []
    for name in cfg_order:
        if name == "baseline":
            continue
        ade = np.array(per_clip[name]["ade"])
        fde = np.array(per_clip[name]["fde"])
        nll = np.array(per_clip[name]["nll"])
        d = ade - base_ade
        mean_d, lo, hi = el.paired_bootstrap_ci(d)
        reg = int((d > 0.5).sum())
        per_bucket = {}
        for b in BUCKETS:
            mask = buckets == b
            per_bucket[b] = float(d[mask].mean()) if mask.any() else float("nan")
        rows.append({
            "name": name, **meta[name],
            "minADE_mean": float(ade.mean()), "minADE_delta": mean_d,
            "delta_ci_lo": lo, "delta_ci_hi": hi,
            "delta_p50": float(np.percentile(d, 50)), "delta_p90": float(np.percentile(d, 90)),
            "delta_p95": float(np.percentile(d, 95)),
            "minFDE_delta": float(fde.mean() - base_fde.mean()),
            "nll_delta": float(nll.mean() - base_nll.mean()),
            "regressions_gt0p5": reg, "per_bucket_delta": per_bucket,
        })

    lines.append("Per-config paired delta minADE vs baseline (95% bootstrap CI)")
    lines.append(f"  {'config':22s}{'dADE':>9}{'95% CI':>18}{'p90':>8}{'p95':>8}"
                 f"{'reg>0.5':>8}{'dNLL':>8}")
    for r in rows:
        free = "" if (r["delta_ci_lo"] <= 0 <= r["delta_ci_hi"]) else \
               ("*" if r["minADE_delta"] > 0 else "+")
        lines.append(
            f"  {r['name']:22s}{r['minADE_delta']:+9.4f}"
            f"  [{r['delta_ci_lo']:+.3f},{r['delta_ci_hi']:+.3f}]"
            f"{r['delta_p90']:+8.3f}{r['delta_p95']:+8.3f}"
            f"{r['regressions_gt0p5']:>8}{r['nll_delta']:+8.3f} {free}")
    lines.append("  (* = CI excludes 0 and delta>0 i.e. real regression; blank = within noise)")
    lines.append("")

    lines.append("Per-bucket delta minADE (does 'free' hold on turns / stops?)")
    lines.append(f"  {'config':22s}" + "".join(f"{b:>13}" for b in BUCKETS))
    for r in rows:
        lines.append(f"  {r['name']:22s}" + "".join(
            f"{r['per_bucket_delta'][b]:>+13.4f}" for b in BUCKETS))
    lines.append("")

    text = "\n".join(lines)
    (out_dir / "eval_summary.txt").write_text(text + "\n")
    (out_dir / "metrics.json").write_text(json.dumps({
        "n_clips": n, "bucket_counts": {b: int((buckets == b).sum()) for b in BUCKETS},
        "baseline": {"minADE": float(base_ade.mean()), "minFDE": float(base_fde.mean()),
                     "nll": float(base_nll.mean())},
        "rows": rows, "clip_ids": clip_ids, "buckets": buckets.tolist(),
    }, indent=2))
    print(text)
    make_plots(rows, per_clip, base_ade, buckets, out_dir / "plots", n)
    print("plots ->", out_dir / "plots")


def make_plots(rows, per_clip, base_ade, buckets, plots, n):
    names = [r["name"] for r in rows]
    y = np.arange(len(names))

    # 1) paired delta with CI, sorted as given
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(names) + 1.2))
    for i, r in enumerate(rows):
        col = C2 if (r["delta_ci_lo"] <= 0 <= r["delta_ci_hi"]) else \
              (C3 if r["minADE_delta"] > 0 else C1)
        ax.plot([r["delta_ci_lo"], r["delta_ci_hi"]], [i, i], color=col, lw=2.5, zorder=2)
        ax.scatter([r["minADE_delta"]], [i], color=col, s=30, zorder=3)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("minADE change vs baseline (m), 95% bootstrap CI")
    ax.set_title(f"Multi-sample minADE effect of each config (n={n})")
    ax.grid(axis="x", color=GRID, lw=0.7)
    fig.tight_layout(); fig.savefig(plots / "delta_ci.png", dpi=150); plt.close(fig)

    # 2) per-bucket heatmap
    key = [r for r in rows if r["name"] != "baseline"]
    mat = np.array([[r["per_bucket_delta"][b] for b in BUCKETS] for r in key])
    fig, ax = plt.subplots(figsize=(6.5, 0.42 * len(key) + 1.4))
    vmax = np.nanpercentile(np.abs(mat), 95) or 0.1
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(BUCKETS))); ax.set_xticklabels(BUCKETS, fontsize=9)
    ax.set_yticks(range(len(key))); ax.set_yticklabels([r["name"] for r in key], fontsize=8.5)
    for i in range(len(key)):
        for j in range(len(BUCKETS)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > vmax * 0.6 else INK)
    ax.set_title("minADE change by scenario bucket (red = worse)")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout(); fig.savefig(plots / "bucket_heatmap.png", dpi=150); plt.close(fig)

    # 3) worst-case: p50 vs p90 vs p95 for the key "free" claims
    picks = [nm for nm in ["drop_last_layer", "traj_late_r50", "coc_late_r50",
                           "random_late_r50", "traj_all_r30", "kv_traj_drop1"]
             if nm in [r["name"] for r in rows]]
    rmap = {r["name"]: r for r in rows}
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(picks))
    for j, (pct, col) in enumerate([("delta_p50", C1), ("delta_p90", C4), ("delta_p95", C3)]):
        ax.bar(x + (j - 1) * 0.26, [rmap[p][pct] for p in picks], 0.24, color=col,
               label=pct.replace("delta_", ""))
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(picks, fontsize=8, rotation=15)
    ax.set_ylabel("minADE change (m)")
    ax.set_title("Mean hides the tail: median vs p90 vs p95 of per-clip change")
    ax.legend(frameon=False, fontsize=9); ax.grid(axis="y", color=GRID, lw=0.7)
    fig.tight_layout(); fig.savefig(plots / "worst_case.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
