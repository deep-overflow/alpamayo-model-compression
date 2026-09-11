"""Lateral (lane-keeping / path-tracking) surrogates from the alpasim rollouts.

`analyze_longitudinal.py`'s twin on the other axis, written because the longitudinal
surrogates ruled themselves out: measured against `dual`, `dual+h4` brakes MORE and
travels SLOWER near obstacles, yet loses 0.091 of closed-loop score. Braking judgement
is therefore not the mechanism, and lane keeping is the remaining candidate -- but the
`offroad` gate is a rare event (20 vs 28 hits in 300 rollouts, scene-paired Wilcoxon
p=0.087), exactly the powerlessness that made the longitudinal script necessary.

The signal is `min_distance_to_lane_boundary_m`, which the sim already writes at 10 Hz.
**Zero is not a sentinel**, which had to be established before any of this means
anything: over 24k steps the positive values run continuously down to 0 (p1 = 0.0097 m,
no gap), and `offroad` fires on 3.46% of zero-margin steps against **0.00%** of
positive-margin steps. So the series is the margin to the lane edge, clamped at 0 once
the ego is on or across it -- and the fraction of time at 0 is the continuous precursor
to the `offroad` gate, which is the whole point.

Measures, per rollout, over eval-relevant steps (`eval_relevant` is 0 for the ~9% warm-up
and masking it is why these numbers are not directly comparable to the longitudinal
script's, which does not mask):

  * frac_out_of_lane      fraction of steps at zero margin -- the offroad precursor
  * lane_margin_p05 / median   over in-lane steps only: how much room it keeps
  * frac_margin_below_0p2m     near-edge exposure, in-lane
  * out_of_lane_episodes / longest_excursion_s  many brief touches or few long departures?
  * xtrack_median / xtrack_p95  `dist_to_gt_trajectory`, distance to the nearest point of
                                the GT path -- cross-track error, not the along-track
                                error that `dist_to_gt_location` also carries
  * plan_dev_median / p95       `plan_deviation`: drift from the model's OWN plan, so a
                                self-consistency measure rather than a GT one
  * frac_wrong_lane             time in the wrong lane

Aggregation mirrors analyze_longitudinal.py: per-rollout -> per-scene mean -> paired
delta vs `--reference` with bootstrap CI and Wilcoxon. Bootstrap CI is primary; Wilcoxon
drops ties and many scenes tie exactly.

Usage (this repo's venv is enough -- only pandas is needed, no alpasim_utils):
    .venv/bin/python experiments/head_analysis/analyze_lateral.py \
        --runs-root /home/cvlab21/project/chan/alpasim-runs --prefix m2601_merged_ \
        --reference slim_dual_u40_v2 \
        --configs baseline slim_dual_u40_v2 slim_dual_u40_qcut4_v2 \
        --out .../outputs/headmlp_lateral
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
ORANGE, BLUE, GREEN, RED = "#D97757", "#2a78d6", "#008300", "#b3261e"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

NEAR_EDGE_M = 0.20   # "close to the lane edge" while still inside it
BOOT = 20000

# key -> (label, sign that means SAFER)
KEYS = [
    ("frac_out_of_lane", "time out of lane", -1),
    ("frac_margin_below_0p2m", "time within 0.2 m of edge", -1),
    ("lane_margin_p05", "lane margin p05", +1),
    ("lane_margin_median", "lane margin median", +1),
    ("longest_excursion_s", "longest out-of-lane episode", -1),
    ("out_of_lane_episodes", "out-of-lane episodes", -1),
    ("xtrack_median", "cross-track error median", -1),
    ("xtrack_p95", "cross-track error p95", -1),
    ("plan_dev_median", "plan deviation median", -1),
    ("plan_dev_p95", "plan deviation p95", -1),
    ("frac_wrong_lane", "time in wrong lane", -1),
]


def series(df, name):
    s = df[df["name"] == name]
    if s.empty:
        return None
    return s["values"].to_numpy(dtype=float)


def boot_ci(x, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    if x.size == 0:
        return np.nan, np.nan, np.nan
    g = np.random.default_rng(seed)
    b = x[g.integers(0, len(x), (BOOT, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def wilcoxon_p(d):
    d = np.asarray([v for v in d if np.isfinite(v)], dtype=float)
    d = d[d != 0]
    if d.size < 5:
        return None
    return float(stats.wilcoxon(d).pvalue)


def rollout_metrics(path):
    """Lateral summary for one rollout, or None if the lane series is missing."""
    df = pd.read_parquet(path)
    margin = series(df, "min_distance_to_lane_boundary_m")
    if margin is None or len(margin) < 20:
        return None
    rel = series(df, "eval_relevant")
    t = df[df["name"] == "min_distance_to_lane_boundary_m"]["timestamps_us"].to_numpy(float) / 1e6
    step = float(np.median(np.diff(t))) if len(t) > 1 else 0.1

    n = len(margin) if rel is None else min(len(margin), len(rel))
    margin = margin[:n]
    keep = np.ones(n, bool) if rel is None else (rel[:n] > 0)
    m = margin[keep]
    if m.size == 0:
        return None

    out = {}
    out_of = m == 0.0
    out["frac_out_of_lane"] = float(out_of.mean())

    inlane = m[~out_of]
    out["lane_margin_p05"] = float(np.percentile(inlane, 5)) if inlane.size else np.nan
    out["lane_margin_median"] = float(np.median(inlane)) if inlane.size else np.nan
    out["frac_margin_below_0p2m"] = (float(np.mean(inlane < NEAR_EDGE_M))
                                     if inlane.size else np.nan)

    # episode structure: one long departure and twenty brief kisses of the line are very
    # different failures, and the mean margin cannot tell them apart
    runs, cur = [], 0
    for x in out_of:
        if x:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    out["out_of_lane_episodes"] = float(len(runs))
    out["longest_excursion_s"] = float(max(runs) * step) if runs else 0.0

    for src, dst in (("dist_to_gt_trajectory", "xtrack"), ("plan_deviation", "plan_dev")):
        v = series(df, src)
        if v is None or len(v) == 0:
            out[f"{dst}_median"] = out[f"{dst}_p95"] = np.nan
            continue
        k = keep[:len(v)] if len(v) <= len(keep) else np.ones(len(v), bool)
        vv = v[:len(k)][k]
        vv = vv[np.isfinite(vv)]
        out[f"{dst}_median"] = float(np.median(vv)) if vv.size else np.nan
        out[f"{dst}_p95"] = float(np.percentile(vv, 95)) if vv.size else np.nan

    wl = series(df, "wrong_lane")
    out["frac_wrong_lane"] = (float(np.mean(wl[:len(keep)][keep] > 0))
                              if wl is not None and len(wl) else np.nan)
    out["n_steps"] = int(m.size)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--prefix", default="m2601_merged_")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference", default="baseline",
                    help="config every delta is measured against; must be in --configs")
    ap.add_argument("--configs", nargs="+", required=True)
    args = ap.parse_args()
    cfgs = list(args.configs)
    if args.reference not in cfgs:
        raise SystemExit(f"--reference {args.reference!r} not in --configs {cfgs}")
    args.out.mkdir(parents=True, exist_ok=True)

    per_scene = {}
    n_roll = {}
    for cfg in cfgs:
        run = args.runs_root / f"{args.prefix}{cfg}"
        summ = run / "aggregate" / "results-summary.json"
        if not summ.exists():
            raise SystemExit(f"{summ} missing")
        acc, cnt = {}, 0
        for r in json.loads(summ.read_text())["rollouts"]:
            p = run / "rollouts" / r["clipgt_id"] / r["rollout_id"] / "metrics.parquet"
            if not p.exists():
                continue
            m = rollout_metrics(p)
            if m is None:
                continue
            cnt += 1
            acc.setdefault(r["clipgt_id"], []).append(m)
        per_scene[cfg] = {s: {k: float(np.nanmean([x[k] for x in v]))
                              for k, _, _ in KEYS}
                          for s, v in acc.items()}
        n_roll[cfg] = cnt
        print(f"{cfg}: {cnt} rollouts over {len(acc)} scenes", flush=True)

    scenes = sorted(set.intersection(*[set(per_scene[c]) for c in cfgs]))
    res = {"n_scenes": len(scenes), "reference": args.reference,
           "params": {"near_edge_m": NEAR_EDGE_M, "bootstrap": BOOT},
           "n_rollouts": n_roll, "configs": {}, "paired": {}}

    for cfg in cfgs:
        res["configs"][cfg] = {}
        for k, _, _ in KEYS:
            mu, lo, hi = boot_ci([per_scene[cfg][s][k] for s in scenes])
            res["configs"][cfg][k] = {"mean": mu, "ci_lo": lo, "ci_hi": hi}

    for cfg in cfgs:
        if cfg == args.reference:
            continue
        cell = {}
        for k, _, _ in KEYS:
            d = [per_scene[cfg][s][k] - per_scene[args.reference][s][k] for s in scenes
                 if np.isfinite(per_scene[cfg][s][k])
                 and np.isfinite(per_scene[args.reference][s][k])]
            mu, lo, hi = boot_ci(d)
            cell[k] = {"delta": mu, "ci_lo": lo, "ci_hi": hi,
                       "wilcoxon_p": wilcoxon_p(d), "n": len(d)}
        res["paired"][cfg] = cell

    (args.out / "metrics.json").write_text(json.dumps(res, indent=2))
    # per-scene values are the raw material for joining these against anything else
    # (score deltas, gate outcomes), and recomputing them means re-reading 1,200 parquets
    (args.out / "per_scene.json").write_text(json.dumps(
        {c: {s: v for s, v in per_scene[c].items()} for c in cfgs}, indent=1))

    L = [f"횡방향 대리지표 — {len(scenes)} 씬, 기준 {args.reference}",
         f"(차선 가장자리 근접 기준 {NEAR_EDGE_M} m, eval_relevant 스텝만)", ""]
    L.append(f"{'config':28s} " + " ".join(f"{lab[:13]:>14s}" for _, lab, _ in KEYS[:5]))
    for cfg in cfgs:
        a = res["configs"][cfg]
        L.append(f"{cfg:28s} " + " ".join(f"{a[k]['mean']:14.4f}" for k, _, _ in KEYS[:5]))
    L.append("")
    for cfg, cell in res["paired"].items():
        L.append(f"[{cfg}] vs {args.reference}")
        for k, lab, better in KEYS:
            e = cell[k]
            p = e["wilcoxon_p"]
            sig = p is not None and p < 0.05
            ci_excl = (e["ci_lo"] > 0) or (e["ci_hi"] < 0)
            tag = ""
            if ci_excl:
                tag = "  SAFER" if e["delta"] * better > 0 else "  WORSE"
            L.append(f"  {lab:30s} {e['delta']:+10.4f} [{e['ci_lo']:+.4f},{e['ci_hi']:+.4f}]"
                     f"{'*' if ci_excl else ' '} p={'n/a' if p is None else f'{p:.4f}'}"
                     f"{'*' if sig else ' '}{tag}")
        L.append("")
    (args.out / "summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    plots = args.out / "plots"
    plots.mkdir(exist_ok=True)
    arms = [c for c in cfgs if c != args.reference]
    fig, ax = plt.subplots(1, len(arms), figsize=(5.6 * len(arms), 4.0), squeeze=False)
    for j, cfg in enumerate(arms):
        cell = res["paired"][cfg]
        ys, labs, cols = [], [], []
        for k, lab, better in KEYS:
            base = abs(res["configs"][args.reference][k]["mean"]) or 1.0
            ys.append(100 * cell[k]["delta"] / base)
            labs.append(lab)
            excl = (cell[k]["ci_lo"] > 0) or (cell[k]["ci_hi"] < 0)
            cols.append((GREEN if cell[k]["delta"] * better > 0 else RED) if excl else MUTED)
        y = np.arange(len(ys))
        ax[0][j].barh(y, ys, color=cols, alpha=0.85)
        ax[0][j].axvline(0, color=INK, lw=0.9)
        ax[0][j].set_yticks(y)
        ax[0][j].set_yticklabels(labs, fontsize=8)
        ax[0][j].invert_yaxis()
        ax[0][j].set_xlabel(f"% of {args.reference}'s own level")
        ax[0][j].set_title(f"{cfg} vs {args.reference}\n"
                           "green = safer, red = worse (CI excludes 0), grey = n.s.",
                           fontsize=9)
    fig.tight_layout()
    fig.savefig(plots / "lateral.png", dpi=150)
    plt.close(fig)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
