"""Stage E readout: does the label-free J-score preserve CoC like the labeled dual?

Merges the sharded criterion sweeps and reports paired per-clip statistics. Every
config saw the same clips, the same K sampling seeds, and the same baseline CoC
context, so differences pair exactly and clip difficulty cancels.

Two kinds of comparison, both needed:
  vs baseline     how much did this criterion cost at this ratio
  head-to-head    jspace vs traj (does dropping labels still beat trajectory-only?)
                  jspace vs cocsafe (does dropping labels cost anything?)
                  j_traj vs cocsafe (the label-free analogue of the shipped dual)

minADE deltas are heavy-tailed -- a broken config lands at 25 m and drags any mean
with it -- so the median and the rank-based Wilcoxon are the primary readings and
the mean is reported alongside rather than trusted on its own.

Usage:
  python analyze_jsweep.py --shards jsweep_s0 jsweep_s40 --exp-id jsweep_summary
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

import eval_lib as el  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})

LABEL_FREE = {"magnitude", "traj", "jspace", "j_traj"}
COLORS = {"magnitude": MUTED, "traj": C3, "coc": C4, "cocsafe": C2,
          "jspace": C1, "j_traj": "#7b3fb8"}
HEAD_TO_HEAD = [("jspace", "traj"), ("jspace", "cocsafe"), ("j_traj", "cocsafe"),
                ("j_traj", "traj")]


def load(shards):
    """Concatenate per-clip arrays across shards. Returns (configs, meta, data)."""
    configs, meta, data, clips = None, None, {}, []
    for sh in shards:
        m = json.loads((REPO / "outputs" / sh / "metrics.json").read_text())
        if configs is None:
            configs, meta = m["configs"], m["meta"]
        elif configs != m["configs"]:
            raise RuntimeError(f"shard {sh} has different configs")
        clips += m["clip_ids"]
        n = len(m["clip_ids"])
        for name in configs:
            d = data.setdefault(name, {"ade": [], "nll": []})
            for key in ("ade", "nll"):
                # a shard saves incrementally, so trim to the clips it reports
                d[key] += m["per_clip"][name][key][:n]
    for name in configs:
        for key in ("ade", "nll"):
            data[name][key] = np.array(data[name][key], dtype=float)
    return configs, meta, data, clips


def paired(delta):
    mean, lo, hi = el.paired_bootstrap_ci(delta)
    try:
        p = float(wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.0
    except ValueError:
        p = 1.0
    return {"mean": mean, "ci": [lo, hi], "median": float(np.median(delta)),
            "p": p, "n": int(len(delta))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", default=["jsweep_s0", "jsweep_s40"])
    ap.add_argument("--exp-id", default="jsweep_summary")
    args = ap.parse_args()

    configs, meta, data, clips = load(args.shards)
    out_dir = REPO / "outputs" / args.exp_id
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    base = data["baseline"]

    vs_base = {}
    for name in configs:
        if name == "baseline":
            continue
        vs_base[name] = {
            "ade": paired(data[name]["ade"] - base["ade"]),
            "nll": paired(data[name]["nll"] - base["nll"]),
            "criterion": meta[name].get("criterion"), "ratio": meta[name].get("ratio"),
            "label_free": meta[name].get("criterion") in LABEL_FREE,
        }

    ratios = sorted({v["ratio"] for v in vs_base.values() if v["ratio"] is not None})
    h2h = {}
    for a, b in HEAD_TO_HEAD:
        for r in ratios:
            na, nb = f"{a}_r{int(r * 100)}", f"{b}_r{int(r * 100)}"
            if na in data and nb in data:
                h2h[f"{na} - {nb}"] = {
                    "ade": paired(data[na]["ade"] - data[nb]["ade"]),
                    "nll": paired(data[na]["nll"] - data[nb]["nll"]),
                }

    # ---- report ----
    lines = [f"Stage E criterion sweep -- {len(clips)} clips, shards {', '.join(args.shards)}",
             f"baseline minADE {base['ade'].mean():.3f}  NLL {base['nll'].mean():.3f}", "",
             "vs baseline (paired per clip; median is primary, minADE deltas are heavy-tailed)",
             f"{'config':<16} {'lbl-free':>8} {'dADE med':>9} {'dADE mean':>10} "
             f"{'dNLL med':>9} {'dNLL mean':>10} {'[95% CI]':>18} {'p(NLL)':>8}"]
    for name in configs:
        if name == "baseline":
            continue
        v = vs_base[name]
        ci = v["nll"]["ci"]
        lines.append(
            f"{name:<16} {'yes' if v['label_free'] else 'NO':>8} "
            f"{v['ade']['median']:+9.3f} {v['ade']['mean']:+10.3f} "
            f"{v['nll']['median']:+9.3f} {v['nll']['mean']:+10.3f} "
            f"[{ci[0]:+.3f},{ci[1]:+.3f}]".rjust(18) + f" {v['nll']['p']:8.4f}")

    lines += ["", "head-to-head (negative = first config is better)",
              f"{'comparison':<34} {'dADE med':>9} {'dNLL med':>9} {'dNLL mean':>10} "
              f"{'[95% CI]':>18} {'p(NLL)':>8}"]
    for k, v in h2h.items():
        ci = v["nll"]["ci"]
        lines.append(f"{k:<34} {v['ade']['median']:+9.3f} {v['nll']['median']:+9.3f} "
                     f"{v['nll']['mean']:+10.3f} "
                     f"[{ci[0]:+.3f},{ci[1]:+.3f}]".rjust(18) + f" {v['nll']['p']:8.4f}")

    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    (out_dir / "config.json").write_text(json.dumps({
        "purpose": "Stage E: label-free J-score vs labeled dual at matched ratios",
        "shards": args.shards, "n_clips": len(clips), "clip_ids": clips,
        "label_free_criteria": sorted(LABEL_FREE),
    }, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(
        {"vs_baseline": vs_base, "head_to_head": h2h}, indent=2))

    # ---- frontier: CoC damage vs trajectory damage ----
    fig, axes = plt.subplots(1, len(ratios), figsize=(5.5 * len(ratios), 4.2), squeeze=False)
    for ax, r in zip(axes[0], ratios):
        for name, v in vs_base.items():
            if v["ratio"] != r:
                continue
            c = COLORS.get(v["criterion"], MUTED)
            ax.scatter(v["nll"]["median"], v["ade"]["median"], s=90, color=c,
                       marker="o" if v["label_free"] else "s", zorder=3)
            ax.annotate(v["criterion"], (v["nll"]["median"], v["ade"]["median"]),
                        textcoords="offset points", xytext=(7, 4), fontsize=9, color=c)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_title(f"ratio {r:.0%}  (circle = label-free, square = needs CoC labels)")
        ax.set_xlabel("median dNLL  (CoC damage)")
        ax.set_ylabel("median dminADE  (trajectory damage)")
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "frontier.png", dpi=150)
    plt.close(fig)

    print("\n".join(lines), flush=True)
    print("saved ->", out_dir, flush=True)


if __name__ == "__main__":
    main()
