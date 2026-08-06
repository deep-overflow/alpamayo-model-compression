"""Longitudinal (braking / following) safety metrics from the alpasim matrix rollouts.

At-fault collisions are rare events (8 vs 2), so the "reasoning collapse costs
longitudinal judgement" claim from analyze_collisions.py is underpowered as a count.
This re-analyses the same 240 rollouts as *continuous* surrogate safety measures,
which every timestep contributes to:

  * time headway (gap / speed) — standard surrogate; low = tailgating
  * proximity exposure — fraction of time within 2 / 3 / 5 m of an obstacle
  * braking response — when close to an obstacle and moving, does speed drop over
    the next second? This is the direct test of "lost the braking decision".
  * speed at closest approach

Everything comes from the per-rollout metrics.parquet time series (10 Hz), so no GPU
and no re-simulation. Aggregation mirrors analyze_alpasim.py: per-rollout -> per-scene
mean -> paired delta vs baseline with bootstrap CI and Wilcoxon.

Usage (alpasim venv):
    cd /home/cvlab21/project/chan/alpasim && uv run python \
        .../analyze_longitudinal.py --runs-root /home/cvlab21/project/chan/alpasim-runs \
        --out .../outputs/alpasim_longitudinal
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
from analyze_alpasim import (  # noqa: E402
    BG, C1, C2, C3, C4, CONFIGS, INK, MUTED, boot_ci, wilcoxon_p,
)

COLORS = {c: col for c, col in zip(CONFIGS, [C1, C2, C3, C4])}

CLOSE_M = 6.0        # "near an obstacle" threshold for the braking test
MOVING_MPS = 1.0     # ignore standing still — braking is undefined there
HORIZON_S = 1.0      # how far ahead to look for a speed drop
BRAKE_MPS2 = -0.5    # counts as "actually braked"
SMOOTH = 5           # 0.5 s moving average on speed

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
})


LANE_HALF_W = 1.75   # lateral corridor that counts as "in our lane"
LEAD_MAX_M = 50.0    # ignore vehicles further ahead than this
TTC_DANGER_S = (2.0, 3.0)


def _yaw(q):
    """Yaw from a (w, x, y, z) quaternion."""
    return np.arctan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                      1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2))


async def lead_vehicle_series(asl_path):
    """(t, gap_m, ego_speed, lead_speed) for the nearest in-lane vehicle ahead.

    Poses come from the ASL actor_poses stream; other actors are rotated into the ego
    frame each step and the nearest one with x>0 and |y|<LANE_HALF_W is the lead.
    Gaps are centre-to-centre (no AABB subtraction) — consistent across configs, which
    is what the paired comparison needs.
    """
    from alpasim_utils.logs import async_read_pb_log

    ts, ego_xy, ego_yaw, others = [], [], [], []
    async for e in async_read_pb_log(str(asl_path)):
        if e.WhichOneof("log_entry") != "actor_poses":
            continue
        ap = e.actor_poses
        ego, rest = None, []
        for a in ap.actor_poses:
            v, q = a.actor_pose.vec, a.actor_pose.quat
            p = (v.x, v.y)
            if a.actor_id == "EGO":
                ego = (p, _yaw((q.w, q.x, q.y, q.z)))
            else:
                rest.append((a.actor_id, p))
        if ego is None:
            continue
        ts.append(ap.timestamp_us / 1e6)
        ego_xy.append(ego[0])
        ego_yaw.append(ego[1])
        others.append(rest)

    if len(ts) < 20:
        return None

    ts = np.asarray(ts)
    ego_xy = np.asarray(ego_xy)
    ego_yaw = np.asarray(ego_yaw)

    gap = np.full(len(ts), np.nan)
    lead_id = [None] * len(ts)
    for i, (o, c, yaw) in enumerate(zip(others, ego_xy, ego_yaw)):
        if not o:
            continue
        d = np.asarray([p for _, p in o]) - c
        ca, sa = np.cos(-yaw), np.sin(-yaw)
        x = d[:, 0] * ca - d[:, 1] * sa
        y = d[:, 0] * sa + d[:, 1] * ca
        m = (x > 0) & (x < LEAD_MAX_M) & (np.abs(y) < LANE_HALF_W)
        if m.any():
            j = int(np.argmin(np.where(m, x, np.inf)))
            gap[i] = x[j]
            lead_id[i] = o[j][0]

    ego_speed = np.zeros(len(ts))
    step = np.diff(ts)
    ego_speed[1:] = np.linalg.norm(np.diff(ego_xy, axis=0), axis=1) / np.maximum(step, 1e-6)
    ego_speed[0] = ego_speed[1]
    ego_speed = np.clip(smooth(ego_speed), 0, 40)

    # closing speed from the gap trend (positive = gap shrinking)
    closing = np.full(len(ts), np.nan)
    same = np.asarray([lead_id[i] is not None and lead_id[i] == lead_id[i - 1]
                       for i in range(1, len(ts))])
    dg = np.diff(gap) / np.maximum(step, 1e-6)
    closing[1:] = np.where(same, -dg, np.nan)
    closing = np.where(np.isfinite(closing), closing, np.nan)
    return ts, gap, ego_speed, closing


def lead_metrics(asl_path):
    import asyncio as _aio

    res = _aio.run(lead_vehicle_series(asl_path))
    if res is None:
        return {}
    ts, gap, v, closing = res
    has = np.isfinite(gap)
    moving = v > MOVING_MPS
    out = {"frac_time_with_lead": float(np.mean(has))}

    sel = has & moving
    if sel.any():
        thw = gap[sel] / np.maximum(v[sel], 1e-6)
        out["lead_thw_p05"] = float(np.percentile(thw, 5))
        out["lead_thw_p10"] = float(np.percentile(thw, 10))
        out["lead_thw_p25"] = float(np.percentile(thw, 25))
        out["lead_thw_median"] = float(np.median(thw))
        out["lead_frac_thw_below_1s"] = float(np.mean(thw < 1.0))
        out["lead_min_gap_m"] = float(np.nanmin(gap[has]))
    else:
        out.update({"lead_thw_p05": np.nan, "lead_thw_p10": np.nan,
                    "lead_thw_p25": np.nan, "lead_thw_median": np.nan,
                    "lead_frac_thw_below_1s": np.nan, "lead_min_gap_m": np.nan})

    # TTC only where we are actually closing on the lead
    ttc_sel = has & moving & np.isfinite(closing) & (closing > 0.1)
    if ttc_sel.any():
        ttc = gap[ttc_sel] / closing[ttc_sel]
        out["lead_ttc_p05"] = float(np.percentile(ttc, 5))
        for thr in TTC_DANGER_S:
            out[f"lead_frac_ttc_below_{thr:g}s"] = float(
                np.sum(ttc < thr) / max(int(np.sum(has & moving)), 1))
    else:
        out["lead_ttc_p05"] = np.nan
        for thr in TTC_DANGER_S:
            out[f"lead_frac_ttc_below_{thr:g}s"] = 0.0

    # braking response to a *close lead*: speed change over the next second
    step_s = float(np.median(np.diff(ts))) or 0.1
    h = max(1, int(round(HORIZON_S / step_s)))
    close_lead = has & moving & (gap < CLOSE_M * 2)   # 12 m: a lead this near demands a response
    close_lead[-h:] = False
    if close_lead.any():
        idx = np.nonzero(close_lead)[0]
        dv = (v[idx + h] - v[idx]) / HORIZON_S
        out["lead_brake_accel"] = float(np.mean(dv))
        out["lead_brake_frac"] = float(np.mean(dv < BRAKE_MPS2))
        out["n_close_lead_steps"] = int(idx.size)
    else:
        out.update({"lead_brake_accel": np.nan, "lead_brake_frac": np.nan,
                    "n_close_lead_steps": 0})
    return out


def series(df, name):
    s = df[df["name"] == name]
    if s.empty:
        return None, None
    return (s["timestamps_us"].to_numpy(dtype=float) / 1e6,
            s["values"].to_numpy(dtype=float))


def smooth(x, w=SMOOTH):
    if len(x) < w:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def rollout_metrics(parquet_path):
    """Longitudinal summary for one rollout, or None if the series are unusable."""
    df = pd.read_parquet(parquet_path)
    t, dist = series(df, "dist_traveled_m")
    _, gap = series(df, "min_distance_to_obstacle_m")
    if t is None or gap is None or len(t) < 20:
        return None

    # speed from the cumulative distance; clip the first sample (start transient)
    dt = np.diff(t)
    v = np.zeros_like(dist)
    v[1:] = np.diff(dist) / np.maximum(dt, 1e-6)
    v[0] = v[1]
    v = np.clip(v, 0, 40)          # 144 km/h ceiling kills differentiation spikes
    v = smooth(v)
    n = min(len(v), len(gap))
    v, gap, t = v[:n], gap[:n], t[:n]

    step = float(np.median(np.diff(t))) or 0.1
    horizon = max(1, int(round(HORIZON_S / step)))

    moving = v > MOVING_MPS
    close = gap < CLOSE_M
    sel = moving & close
    sel[-horizon:] = False          # need a future window

    # braking response: speed change over the next HORIZON_S while close & moving
    out = {}
    if sel.any():
        idx = np.nonzero(sel)[0]
        dv = (v[idx + horizon] - v[idx]) / HORIZON_S
        out["brake_accel_when_close"] = float(np.mean(dv))
        out["brake_frac_when_close"] = float(np.mean(dv < BRAKE_MPS2))
        out["n_close_steps"] = int(idx.size)
    else:
        out["brake_accel_when_close"] = np.nan
        out["brake_frac_when_close"] = np.nan
        out["n_close_steps"] = 0

    # time headway over moving steps
    thw = np.where(moving, gap / np.maximum(v, 1e-6), np.nan)
    thw_valid = thw[np.isfinite(thw)]
    out["thw_p05"] = float(np.percentile(thw_valid, 5)) if thw_valid.size else np.nan
    out["thw_median"] = float(np.median(thw_valid)) if thw_valid.size else np.nan
    out["frac_thw_below_1s"] = (float(np.mean(thw_valid < 1.0)) if thw_valid.size else np.nan)

    # proximity exposure and closest approach
    for thr in (2.0, 3.0, 5.0):
        out[f"frac_within_{thr:g}m"] = float(np.mean(gap < thr))
    out["min_gap_m"] = float(np.min(gap))
    out["mean_speed_when_close"] = (float(np.mean(v[close & moving]))
                                    if (close & moving).any() else np.nan)
    j = int(np.argmin(gap))
    out["speed_at_closest_mps"] = float(v[j])
    out["mean_speed_mps"] = float(np.mean(v[moving])) if moving.any() else np.nan
    return out


def main():
    # rebound below from --configs, so a two-config re-run (stored baseline + one new
    # config) works without threading the list through every helper
    global CONFIGS, COLORS
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--prefix", default="matrix_")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--configs", nargs="+", default=CONFIGS,
                    help="run dirs to compare; the first is treated as the baseline")
    args = ap.parse_args()
    CONFIGS = list(args.configs)
    COLORS = {c: col for c, col in zip(CONFIGS, [C1, C2, C3, C4])}
    (args.out / "plots").mkdir(parents=True, exist_ok=True)

    KEYS = ["brake_accel_when_close", "brake_frac_when_close", "thw_p05",
            "frac_thw_below_1s", "frac_within_2m", "frac_within_3m",
            "min_gap_m", "speed_at_closest_mps", "mean_speed_when_close",
            "mean_speed_mps",
            # lead-vehicle specific (from ASL actor poses)
            "frac_time_with_lead", "lead_thw_p05", "lead_thw_p10",
            "lead_thw_p25", "lead_thw_median",
            "lead_frac_thw_below_1s", "lead_min_gap_m", "lead_ttc_p05",
            "lead_frac_ttc_below_2s", "lead_frac_ttc_below_3s",
            "lead_brake_accel", "lead_brake_frac"]

    per_scene = {}       # config -> key -> {scene: value}
    per_rollout = {}     # config -> list of dicts (for distribution plots)
    for cfg in CONFIGS:
        run = args.runs_root / f"{args.prefix}{cfg}"
        summary = json.loads((run / "aggregate" / "results-summary.json").read_text())
        rows = []
        for r in summary["rollouts"]:
            p = run / "rollouts" / r["clipgt_id"] / r["rollout_id"] / "metrics.parquet"
            if not p.exists():
                continue
            m = rollout_metrics(p)
            if m is None:
                continue
            asl = p.parent / "rollout.asl"
            if asl.exists():
                m.update(lead_metrics(asl))
            for k in KEYS:
                m.setdefault(k, np.nan)
            m["scene"] = r["clipgt_id"]
            m["collision_at_fault"] = r["metrics"].get("collision_at_fault", 0.0)
            rows.append(m)
        per_rollout[cfg] = rows
        acc = {}
        for k in KEYS:
            by = {}
            for m in rows:
                by.setdefault(m["scene"], []).append(m[k])
            acc[k] = {s: float(np.nanmean(v)) for s, v in by.items()}
        per_scene[cfg] = acc

    scenes = sorted(set.intersection(*[set(per_scene[c]["min_gap_m"]) for c in CONFIGS]))

    metrics = {"n_scenes": len(scenes), "params": {
        "close_m": CLOSE_M, "moving_mps": MOVING_MPS, "horizon_s": HORIZON_S,
        "brake_mps2": BRAKE_MPS2}, "configs": {}, "paired_vs_baseline": {}}

    for cfg in CONFIGS:
        agg = {}
        for k in KEYS:
            vals = [per_scene[cfg][k][s] for s in scenes if not np.isnan(per_scene[cfg][k][s])]
            mean, lo, hi = boot_ci(vals)
            agg[k] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
        agg["n_rollouts"] = len(per_rollout[cfg])
        metrics["configs"][cfg] = agg

    for cfg in CONFIGS:
        if cfg == "baseline":
            continue
        d = {}
        for k in KEYS:
            deltas = [per_scene[cfg][k][s] - per_scene["baseline"][k][s] for s in scenes
                      if not (np.isnan(per_scene[cfg][k][s])
                              or np.isnan(per_scene["baseline"][k][s]))]
            mean, lo, hi = boot_ci(deltas)
            d[k] = {"delta": mean, "ci_lo": lo, "ci_hi": hi,
                    "wilcoxon_p": wilcoxon_p(deltas), "n": len(deltas)}
        metrics["paired_vs_baseline"][cfg] = d

    # ---- robustness: does the following-distance effect survive dropping the
    # rollouts that actually crashed? (during a crash the gap goes to ~0, which would
    # depress headway as a *consequence* rather than a precursor)
    rob = {}
    for k in ("lead_thw_p05", "lead_thw_median"):
        rob[k] = {}
        for drop in (False, True):
            byc = {}
            for cfg in CONFIGS:
                acc = {}
                for m in per_rollout[cfg]:
                    if drop and m.get("collision_at_fault", 0) > 0:
                        continue
                    v = m.get(k, np.nan)
                    if np.isfinite(v):
                        acc.setdefault(m["scene"], []).append(v)
                byc[cfg] = {s: float(np.mean(v)) for s, v in acc.items()}
            tag = "excl_collision_rollouts" if drop else "all_rollouts"
            rob[k][tag] = {"baseline_mean":
                           float(np.mean(list(byc["baseline"].values()))) if byc["baseline"] else None}
            for cfg in CONFIGS:
                if cfg == "baseline":
                    continue
                sc = sorted(set(byc[cfg]) & set(byc["baseline"]))
                d = [byc[cfg][s] - byc["baseline"][s] for s in sc]
                mu, lo, hi = boot_ci(d)
                rob[k][tag][cfg] = {"delta": mu, "ci_lo": lo, "ci_hi": hi,
                                    "wilcoxon_p": wilcoxon_p(d), "n_scenes": len(sc)}
    metrics["robustness"] = rob

    # ---- plots (English labels: the report font has no Hangul glyphs)
    short = [c.replace("slim_", "").replace("_mag", "") for c in CONFIGS]

    # 1. the headline: median headway is intact, the tail is not
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    levels = [("lead_thw_p05", "5th pct"), ("lead_thw_p10", "10th pct"),
              ("lead_thw_p25", "25th pct"), ("lead_thw_median", "median")]
    xs = np.arange(len(levels))
    for cfg in CONFIGS:
        ys = [metrics["configs"][cfg][k]["mean"] for k, _ in levels]
        axes[0].plot(xs, ys, "o-", color=COLORS[cfg], lw=2, ms=6,
                     label=cfg.replace("slim_", ""))
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([lab for _, lab in levels])
    axes[0].set_ylabel("time headway to lead vehicle (s)")
    axes[0].set_title("Following distance by percentile", fontsize=11)
    axes[0].legend(frameon=False, fontsize=9)

    for i, (k, lab) in enumerate([("lead_thw_median", "median"), ("lead_thw_p05", "5th pct")]):
        for j, cfg in enumerate(CONFIGS):
            if cfg == "baseline":
                continue
            e = metrics["paired_vs_baseline"][cfg][k]
            x = i * 4 + j
            axes[1].bar(x, e["delta"], color=COLORS[cfg], width=0.7)
            axes[1].errorbar(x, e["delta"],
                             yerr=[[max(e["delta"] - e["ci_lo"], 0)],
                                   [max(e["ci_hi"] - e["delta"], 0)]],
                             color=INK, capsize=4, lw=1.2)
    axes[1].axhline(0, color=MUTED, lw=1)
    axes[1].set_xticks([1, 5])
    axes[1].set_xticklabels(["median", "5th pct"])
    axes[1].set_ylabel("Δ headway vs baseline (s)")
    axes[1].set_title("Paired per-scene delta (95% CI)", fontsize=11)
    fig.suptitle("Following distance: the tail moves, the median does not", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out / "plots" / "headway_tail.png", dpi=150)

    # 2. braking response panels
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    panels = [("brake_accel_when_close",
               f"Mean accel. within {CLOSE_M:g} m\n(m/s², negative = braking)"),
              ("brake_frac_when_close", "Fraction of close steps\nfollowed by braking"),
              ("lead_thw_p05", "Lead-vehicle headway\n5th pct (s)")]
    for ax, (k, title) in zip(axes, panels):
        for i, cfg in enumerate(CONFIGS):
            a = metrics["configs"][cfg][k]
            ax.bar(i, a["mean"], color=COLORS[cfg], width=0.62)
            ax.errorbar(i, a["mean"],
                        yerr=[[max(a["mean"] - a["ci_lo"], 0)],
                              [max(a["ci_hi"] - a["mean"], 0)]],
                        color=INK, capsize=4, lw=1.2)
        ax.set_xticks(range(len(CONFIGS)))
        ax.set_xticklabels(short, rotation=20, ha="right", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.axhline(0, color=MUTED, lw=0.8)
    fig.suptitle("Longitudinal control under obstacle proximity", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out / "plots" / "braking_response.png", dpi=150)

    # ---- outputs
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.out / "config.json").write_text(json.dumps({
        "runs_root": str(args.runs_root), "n_scenes": len(scenes),
        "close_m": CLOSE_M, "horizon_s": HORIZON_S, "brake_mps2": BRAKE_MPS2,
        "keys": KEYS}, indent=2))

    L = [f"종방향 안전 지표 — {len(scenes)} 씬 x 2 rollouts, 4 configs",
         f"(근접 기준 {CLOSE_M:g}m, 주행 기준 {MOVING_MPS:g}m/s, 전방 {HORIZON_S:g}s)", ""]
    hdr = (f"{'config':22s} {'제동가속':>9s} {'제동비율':>9s} {'THW p05':>8s} "
           f"{'THW<1s':>7s} {'<2m비율':>8s} {'최소갭':>7s} {'근접시속도':>10s}")
    L += [hdr, "-" * len(hdr)]
    for cfg in CONFIGS:
        a = metrics["configs"][cfg]
        L.append(f"{cfg:22s} {a['brake_accel_when_close']['mean']:9.3f} "
                 f"{a['brake_frac_when_close']['mean']:9.3f} {a['thw_p05']['mean']:8.2f} "
                 f"{a['frac_thw_below_1s']['mean']:7.3f} {a['frac_within_2m']['mean']:8.3f} "
                 f"{a['min_gap_m']['mean']:7.2f} {a['mean_speed_when_close']['mean']:10.2f}")
    L.append("")
    L.append("선행차 특정 지표 (ASL actor poses, 전방 ±1.75m 차선 내 최근접 차량):")
    lhdr = (f"{'config':22s} {'선행차비율':>10s} {'THW p05':>8s} {'THW<1s':>7s} "
            f"{'TTC p05':>8s} {'TTC<2s':>7s} {'제동가속':>9s} {'제동비율':>9s}")
    L += [lhdr, "-" * len(lhdr)]
    for cfg in CONFIGS:
        a = metrics["configs"][cfg]
        L.append(f"{cfg:22s} {a['frac_time_with_lead']['mean']:10.3f} "
                 f"{a['lead_thw_p05']['mean']:8.2f} {a['lead_frac_thw_below_1s']['mean']:7.3f} "
                 f"{a['lead_ttc_p05']['mean']:8.2f} {a['lead_frac_ttc_below_2s']['mean']:7.4f} "
                 f"{a['lead_brake_accel']['mean']:9.3f} {a['lead_brake_frac']['mean']:9.3f}")
    L.append("")
    L.append("baseline 대비 씬-페어드 차이 (95% CI, Wilcoxon):")
    for cfg, d in metrics["paired_vs_baseline"].items():
        L.append(f"  [{cfg}]")
        for k in ["brake_accel_when_close", "brake_frac_when_close", "thw_p05",
                  "frac_thw_below_1s", "frac_within_2m", "mean_speed_when_close",
                  "lead_thw_p05", "lead_frac_thw_below_1s", "lead_ttc_p05",
                  "lead_frac_ttc_below_2s", "lead_brake_accel", "lead_brake_frac"]:
            e = d[k]
            star = "*" if (e["wilcoxon_p"] is not None and e["wilcoxon_p"] < 0.05) else " "
            pv = f"{e['wilcoxon_p']:.4f}" if e["wilcoxon_p"] is not None else "n/a"
            L.append(f"    {k:26s} {e['delta']:+8.4f} [{e['ci_lo']:+.4f}, {e['ci_hi']:+.4f}] "
                     f"p={pv}{star}")
    L.append("")
    L.append("강건성 — 충돌 rollout 제외 시에도 유지되는가 (선행차 THW):")
    for k, byt in metrics["robustness"].items():
        for tag, e in byt.items():
            L.append(f"  [{k} / {tag}] baseline {e['baseline_mean']:.2f}s")
            for cfg in CONFIGS:
                if cfg == "baseline":
                    continue
                v = e[cfg]
                pv = f"{v['wilcoxon_p']:.4f}" if v["wilcoxon_p"] is not None else "n/a"
                star = "*" if (v["wilcoxon_p"] is not None and v["wilcoxon_p"] < 0.05) else " "
                L.append(f"      {cfg:22s} {v['delta']:+7.3f}s "
                         f"[{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}] p={pv}{star} "
                         f"(n={v['n_scenes']}씬)")
    (args.out / "summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
