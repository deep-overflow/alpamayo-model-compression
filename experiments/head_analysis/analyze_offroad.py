"""Why does reconstruction lose in closed loop? Lane-keeping forensics (rollout-free).

plans/2026-08-31_offroad-forensics.md. dualr_wl removed reasoning from the list of
suspects -- it fixed dualr's CoC degeneracy (0.045 -> 0.027) and its LingoQA score
(41.8 -> 72.6) yet scored the same, with offroad worsening 0.077 -> 0.097. This reads
the stored 10 Hz per-rollout time series (`metrics.parquet`) and asks whether the
reconstruction arms simply drive closer to the lane boundary.

Per rollout, from `min_distance_to_lane_boundary_m` / `wrong_lane` / `offroad` /
`dist_to_gt_trajectory` / `plan_deviation`:
  margin_p10, margin_median, frac(margin < 0.5 m), frac(margin < 0.2 m)  -- H1
  offroad episodes (count, total frames, first onset time)
  pre-exit window (5 s before each offroad onset): margin slope, speed proxy
    (dist_traveled derivative), plan_deviation mean                       -- H3
Aggregation mirrors analyze_alpasim / analyze_longitudinal: rollout -> scene mean ->
paired delta vs a reference arm with bootstrap CI and Wilcoxon.

Usage (alpasim venv, for pandas/pyarrow):
  cd /home/cvlab21/project/chan/alpasim && uv run python \
      .../analyze_offroad.py --runs-root /home/cvlab21/project/chan/alpasim-runs \
      --out .../outputs/alpasim_offroad
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from analyze_alpasim import BG, C1, C2, C3, C4, INK, MUTED, boot_ci, wilcoxon_p  # noqa: E402

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": "#E8E6DC",
    "font.size": 9, "axes.grid": True, "axes.axisbelow": True,
})
ARMS = ["baseline", "slim_dual_u40_v2", "slim_dualr_u40", "slim_tyr_u40_r", "slim_dualr_wl_u40"]
LABEL = {"baseline": "baseline", "slim_dual_u40_v2": "dual", "slim_dualr_u40": "dualr",
         "slim_tyr_u40_r": "tyr_r", "slim_dualr_wl_u40": "dualr_wl"}
HZ = 10.0          # rollout metrics are 10 Hz
PRE_S = 5.0        # pre-exit window


def series(df, name):
    """(t_seconds, values) for one metric name, sorted, valid only."""
    d = df[df.name == name]
    if d.empty:
        return None, None
    d = d.sort_values("timestamps_us")
    v = d["values"].to_numpy(dtype=float)
    ok = d["valid"].to_numpy(dtype=bool) if "valid" in d else np.ones(len(v), bool)
    t = d["timestamps_us"].to_numpy(dtype=float) / 1e6
    return t[ok], v[ok]


def episodes(mask):
    """[(start, end)] index runs where mask is True."""
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def rollout_features(path):
    df = pd.read_parquet(path)
    _, margin = series(df, "min_distance_to_lane_boundary_m")
    if margin is None or len(margin) == 0:
        return None
    _, off = series(df, "offroad")
    _, wrong = series(df, "wrong_lane")
    _, dist = series(df, "dist_traveled_m")
    _, dev = series(df, "plan_deviation")
    off = np.zeros_like(margin) if off is None else off[: len(margin)]
    offm = off > 0.5
    clean = margin[~offm] if (~offm).any() else margin
    f = {
        "margin_p10": float(np.percentile(clean, 10)),
        "margin_median": float(np.median(clean)),
        "margin_min": float(clean.min()),
        "frac_lt_050": float(np.mean(clean < 0.5)),
        "frac_lt_020": float(np.mean(clean < 0.2)),
        "frac_offroad": float(offm.mean()),
        "frac_wrong_lane": float(np.mean(wrong > 0.5)) if wrong is not None else np.nan,
        "n_frames": len(margin),
    }
    eps = episodes(offm)
    f["n_episodes"] = len(eps)
    f["first_onset_frac"] = float(eps[0][0] / max(len(margin) - 1, 1)) if eps else np.nan
    slopes, devs, speeds = [], [], []
    for a, _ in eps:
        lo = max(0, a - int(PRE_S * HZ))
        if a - lo < 5:
            continue
        w = margin[lo:a]
        x = np.arange(len(w)) / HZ
        slopes.append(float(np.polyfit(x, w, 1)[0]))            # m/s of margin change
        if dev is not None and a <= len(dev):
            devs.append(float(np.nanmean(dev[lo:a])))
        if dist is not None and a <= len(dist) and a - lo > 1:
            speeds.append(float((dist[a - 1] - dist[lo]) / ((a - 1 - lo) / HZ)))
    f["pre_exit_margin_slope"] = float(np.mean(slopes)) if slopes else np.nan
    f["pre_exit_plan_dev"] = float(np.mean(devs)) if devs else np.nan
    f["pre_exit_speed"] = float(np.mean(speeds)) if speeds else np.nan
    return f


def scene_table(run_dir):
    rows = {}
    for p in sorted(run_dir.glob("rollouts/*/*/metrics.parquet")):
        scene = p.parent.parent.name
        f = rollout_features(p)
        if f:
            rows.setdefault(scene, []).append(f)
    return {s: {k: float(np.nanmean([r[k] for r in v])) for k in v[0]} for s, v in rows.items()}


def paired(a, b, key):
    ids = sorted(set(a) & set(b))
    d = np.array([a[i][key] - b[i][key] for i in ids])
    d = d[~np.isnan(d)]
    if len(d) < 5:
        return None
    mean, lo, hi = boot_ci(d)
    return {"n": len(d), "mean": float(mean), "lo": float(lo), "hi": float(hi),
            "p": float(wilcoxon_p(d))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--prefix", default="m2601_merged_")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--ref", default="slim_dual_u40_v2")
    args = ap.parse_args()
    root, out = Path(args.runs_root), Path(args.out)
    (out / "plots").mkdir(parents=True, exist_ok=True)

    tab = {}
    for a in args.arms:
        d = root / f"{args.prefix}{a}"
        if not d.exists():
            print(f"!! missing {d}")
            continue
        tab[a] = scene_table(d)
        print(f"{a}: {len(tab[a])} scenes", flush=True)

    keys = ["margin_p10", "margin_median", "margin_min", "frac_lt_050", "frac_lt_020",
            "frac_offroad", "frac_wrong_lane", "n_episodes", "first_onset_frac",
            "pre_exit_margin_slope", "pre_exit_plan_dev", "pre_exit_speed"]
    res = {"arms": {}, "paired_vs_ref": {}, "ref": args.ref, "corr": {}}
    lines = [f"lane-keeping forensics -- {len(tab)} arms, ref = {LABEL.get(args.ref, args.ref)}", ""]
    for a, t in tab.items():
        res["arms"][a] = {k: float(np.nanmean([v[k] for v in t.values()])) for k in keys}
        r = res["arms"][a]
        lines.append(f"{LABEL.get(a, a):10s} margin p10 {r['margin_p10']:.3f} med "
                     f"{r['margin_median']:.3f} | <0.5m {r['frac_lt_050']:.3f} <0.2m "
                     f"{r['frac_lt_020']:.3f} | offroad {r['frac_offroad']:.4f} wrong-lane "
                     f"{r['frac_wrong_lane']:.3f} | episodes {r['n_episodes']:.2f} onset "
                     f"{r['first_onset_frac']:.2f} | pre-exit slope {r['pre_exit_margin_slope']:+.3f} "
                     f"m/s dev {r['pre_exit_plan_dev']:.2f} speed {r['pre_exit_speed']:.1f}")
    lines.append("")
    for a in tab:
        if a == args.ref:
            continue
        res["paired_vs_ref"][a] = {}
        for k in ("margin_p10", "margin_median", "frac_lt_050", "frac_offroad", "frac_wrong_lane"):
            c = paired(tab[a], tab[args.ref], k)
            if c:
                res["paired_vs_ref"][a][k] = c
        lines.append(f"{LABEL.get(a, a):10s} vs ref: " + "  ".join(
            f"{k} {v['mean']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}]{'*' if v['lo'] > 0 or v['hi'] < 0 else ''}"
            for k, v in res["paired_vs_ref"][a].items()))

    # H2: does GT-trajectory adherence trade against lane margin, scene by scene?
    from scipy.stats import spearmanr
    for a, t in tab.items():
        d2 = {}
        summ = json.loads((root / f"{args.prefix}{a}" / "aggregate" / "results-summary.json").read_text())
        for rr in summ["rollouts"]:
            v = rr["metrics"].get("dist_to_gt_trajectory")
            if v is not None:
                d2.setdefault(rr["clipgt_id"], []).append(float(v))
        d2 = {s: float(np.mean(v)) for s, v in d2.items()}
        ids = [s for s in t if s in d2 and not np.isnan(t[s]["margin_p10"])]
        if len(ids) > 10:
            rho, p = spearmanr([d2[s] for s in ids], [t[s]["margin_p10"] for s in ids])
            res["corr"][a] = {"spearman_d2gt_vs_margin_p10": float(rho), "p": float(p), "n": len(ids)}
    lines.append("")
    lines.append("H2 Spearman(scene d2gt, margin p10):")
    for a, c in res["corr"].items():
        lines.append(f"  {LABEL.get(a, a):10s} rho {c['spearman_d2gt_vs_margin_p10']:+.3f} "
                     f"(p={c['p']:.3g}, n={c['n']})")

    text = "\n".join(lines)
    print(text)
    (out / "offroad_summary.txt").write_text(text + "\n")
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    (out / "scene_tables.json").write_text(json.dumps(tab))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    names = [a for a in args.arms if a in tab]
    colors = [MUTED, C1, C2, C3, C4][: len(names)]
    for ax, k, title in zip(axes, ("margin_p10", "frac_lt_050", "frac_offroad"),
                            ("차선 여유 10퍼센타일 (m)", "여유 < 0.5 m 시간 비율", "offroad 시간 비율")):
        vals = [[t[k] for t in tab[a].values() if not np.isnan(t[k])] for a in names]
        ax.boxplot(vals, tick_labels=[LABEL.get(a, a) for a in names], showfliers=False)
        for i, (a, c) in enumerate(zip(names, colors)):
            ax.plot(i + 1, np.nanmean(vals[i]), "o", color=c, ms=7)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out / "plots" / "offroad_margins.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
