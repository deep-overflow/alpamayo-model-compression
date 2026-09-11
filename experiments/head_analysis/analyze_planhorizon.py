"""Effective plan horizon: how much of the 6.4 s prediction the sim actually realises.

Open loop scores the whole 6.4 s trajectory; closed loop re-queries the policy every
0.1 s (`control_timestep_us = 100_000`) and advances one control step. That raises the
question of whether closed loop can see criterion differences at all, if those
differences live in the far horizon. This script answers the measurable half of it --
what fraction of a plan is realised -- from logs that already exist, with no GPU.

Every policy step logs its full plan (`driver_return.trajectory`: 65 poses, 0.1 s apart,
span exactly 6.4 s) and the sim logs the realised ego pose (`controller_return.states`,
10 Hz). Both are poses of rig in the *local* frame -- `events/controller.py` passes the
driver trajectory to the controller as `reference_trajectory_of_rig_in_local` without a
transform -- so plan and realisation are directly comparable. V1 below checks that
rather than trusting it.

Three curves, all indexed by lead time k (0.1 s per step, k = 0 .. 64):

  d(k)  realisation gap   ||P_t[k] - R(tau_k)||    averaged over every plan
  c(k)  plan churn        ||P_t[k+1] - P_{t+1}[k]|| between consecutive plans
  v(k)  constant-velocity null: the same gap for a straight-line extrapolation of the
        ego state at the plan's issue time

d(k) grows for two reasons -- the plan is superseded 0.1 s later, and the model is
simply wrong that far out. c(k) isolates the second, so the pair separates "unused"
from "unpredictable".

v(k) is what keeps d(k) honest. A small d(k) means the plan agrees with what happened,
which is not the same as the plan having caused it: driving straight at constant speed
makes any sane plan agree for seconds. Read d(k)/v(k), not d(k) alone -- where the ratio
approaches 1 the agreement is free.

None of the three observes the controller's internal lookahead, which lives in a closed
service; this measures which part of the plan is realised, not which part is read.

Pre-registered gates (plans/2026-09-11_effective-plan-horizon.md):
  H1  median H(0.5 m) <= 10 steps  -> the short-horizon premise holds
      median H(0.5 m) >= 30 steps  -> closed loop does use the long horizon
  H2  arm d(k) curves coincide at small k and separate at large k
      -> method differences sit where nothing is executed

Runs under **alpasim's** venv (needs `alpasim_utils` to read the protobuf logs):

    cd /home/cvlab21/project/chan/alpasim && uv run python \
        <repo>/experiments/head_analysis/analyze_planhorizon.py \
        --runs-root /home/cvlab21/project/chan/alpasim-runs \
        --configs baseline slim_dual_u40_v2 --n-scenes 30 \
        --out <abs>/outputs/planhorizon
"""

import argparse
import asyncio
import json
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
PALETTE = [MUTED, C1, C2, C3, C4, "#7b5bd6", "#b0552f"]
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})

STEP_US = 100_000          # control timestep and plan spacing alike
N_PLAN = 65                # poses per plan: current + 64 waypoints
THRESHOLDS = (0.5, 1.0, 2.0)


def colour_map(configs):
    return {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(configs)}


def load_rollouts(run_dir):
    d = json.loads((run_dir / "aggregate" / "results-summary.json").read_text())
    return [{"scene": r["clipgt_id"], "rollout_id": r["rollout_id"],
             "score": float(r["score"]) if r["score"] is not None else np.nan}
            for r in d["rollouts"]]


async def _read_rollout(asl_path):
    """-> (plans, realised) with plans [(ts array, xy array)] and realised {ts: xy}."""
    from alpasim_utils.logs import async_read_pb_log

    plans, realised = [], {}
    async for entry in async_read_pb_log(str(asl_path)):
        if entry.HasField("driver_return"):
            poses = entry.driver_return.trajectory.poses
            if not poses:
                continue
            ts = np.fromiter((p.timestamp_us for p in poses), dtype=np.int64, count=len(poses))
            xy = np.array([[p.pose.vec.x, p.pose.vec.y] for p in poses], dtype=float)
            plans.append((ts, xy))
        elif entry.HasField("controller_return"):
            for st in entry.controller_return.states:
                v = st.pose_local_to_rig.vec
                realised[int(st.timestamp_us)] = (v.x, v.y)
    return plans, realised


def rollout_curves(plans, realised):
    """Per-rollout d(k), c(k), v(k) sums and counts, plus the V1/V2 integrity numbers."""
    d_sum, d_cnt = np.zeros(N_PLAN), np.zeros(N_PLAN, dtype=int)
    c_sum, c_cnt = np.zeros(N_PLAN), np.zeros(N_PLAN, dtype=int)
    v_sum, v_cnt = np.zeros(N_PLAN), np.zeros(N_PLAN, dtype=int)
    grid_ok = True

    if realised:
        r_ts = np.array(sorted(realised), dtype=np.int64)
        r_xy = np.array([realised[t] for t in r_ts], dtype=float)
    else:
        r_ts = np.zeros(0, dtype=np.int64)
        r_xy = np.zeros((0, 2))

    for ts, xy in plans:
        if len(ts) != N_PLAN or ts[-1] - ts[0] != STEP_US * (N_PLAN - 1):
            grid_ok = False
        # d(k): compare to the realised pose at the same timestamp. Interpolating
        # rather than nearest-matching keeps a half-step offset from masquerading as
        # divergence; plan steps past the rollout end are dropped, not extrapolated.
        if len(r_ts) >= 2:
            inside = (ts >= r_ts[0]) & (ts <= r_ts[-1])
            if inside.any():
                k = np.nonzero(inside)[0]
                px = np.interp(ts[inside], r_ts, r_xy[:, 0])
                py = np.interp(ts[inside], r_ts, r_xy[:, 1])
                dist = np.hypot(xy[inside, 0] - px, xy[inside, 1] - py)
                d_sum[k] += dist
                d_cnt[k] += 1

                # constant-velocity null from the realised state at the issue instant.
                # Velocity is a finite difference of the realised path itself, so it
                # stays in the same frame and needs no dynamic_state field.
                t0 = ts[0]
                if t0 - STEP_US >= r_ts[0]:
                    p_now = np.array([np.interp(t0, r_ts, r_xy[:, 0]),
                                      np.interp(t0, r_ts, r_xy[:, 1])])
                    p_prev = np.array([np.interp(t0 - STEP_US, r_ts, r_xy[:, 0]),
                                       np.interp(t0 - STEP_US, r_ts, r_xy[:, 1])])
                    vel = (p_now - p_prev) / (STEP_US / 1e6)
                    lead = (ts[inside] - t0) / 1e6
                    cvx = p_now[0] + vel[0] * lead
                    cvy = p_now[1] + vel[1] * lead
                    v_sum[k] += np.hypot(cvx - px, cvy - py)
                    v_cnt[k] += 1

    # c(k): consecutive plans on their shared timestamps. Plans are issued one control
    # step apart, so plan i's pose k+1 and plan i+1's pose k name the same instant.
    for (ts_a, xy_a), (ts_b, xy_b) in pairwise(plans):
        if ts_b[0] - ts_a[0] != STEP_US:
            continue
        n = min(len(ts_a) - 1, len(ts_b))
        dist = np.hypot(xy_a[1:n + 1, 0] - xy_b[:n, 0], xy_a[1:n + 1, 1] - xy_b[:n, 1])
        c_sum[:n] += dist
        c_cnt[:n] += 1

    # V1: plan pose 0 sits at the query instant, so it must coincide with the realised
    # pose there. A frame mismatch would show up here as a large, systematic offset.
    v1 = float(d_sum[0] / d_cnt[0]) if d_cnt[0] else np.nan
    return d_sum, d_cnt, c_sum, c_cnt, v_sum, v_cnt, v1, grid_ok


READOUT_K = (1, 3, 5, 10, 20, 30, 45, 64)


def h2_section(res, per_rollout, configs, scores):
    """H2: where the arms separate, and whether the gap tracks the driving score.

    The within-arm correlation between d(1 s) and score is confounded -- a hard scene
    raises the plan gap and lowers the score on its own -- so the paired column against
    the baseline is the control, exactly as analyze_lateral_score_join found for lane
    metrics. Report both; believe the paired one.
    """
    from scipy.stats import spearmanr

    base = configs[0]
    out = ["arm separation: (arm - baseline) / baseline, by lead time", ""]
    out.append(f"{'lead':>6s} " + " ".join(f"{c.replace('slim_', '')[:13]:>13s}"
                                           for c in configs[1:]))
    for k in READOUT_K:
        b = res[base]["d"][k]
        out.append(f"{k * STEP_US / 1e6:5.1f}s "
                   + " ".join(f"{(res[c]['d'][k] - b) / b:+13.3f}" for c in configs[1:]))
    spreads = []
    for k in READOUT_K:
        vals = [res[c]["d"][k] for c in configs]
        spreads.append((k, (max(vals) - min(vals)) / float(np.mean(vals))))
    out += ["", "spread across arms / mean, by lead time:",
            "  " + "  ".join(f"{k * STEP_US / 1e6:.1f}s {s:.3f}" for k, s in spreads)]
    peak = max(spreads, key=lambda x: x[1])
    out.append(f"  peak separation at {peak[0] * STEP_US / 1e6:.1f} s "
               f"({peak[1]:.3f}), 6.4 s value {spreads[-1][1]:.3f}")
    out.append("H2 predicted the arms separate in the FAR horizon; "
               + ("they do" if peak[0] >= 45 else "they do NOT -- separation peaks early"))

    by = {}
    for r in per_rollout:
        by.setdefault(r["config"], {})[r["scene"]] = r
    common = sorted(set.intersection(*[set(v) for v in by.values()]))
    header = (f"{'config':24s} {'score':>7s} {'d(1s)':>8s} {'rho within':>11s} "
              f"{'p':>9s} {'rho paired':>11s} {'p':>9s} {'med delta':>10s}")
    out += ["", f"plan gap vs closed-loop score, {len(common)} shared scenes", header]
    for c in configs:
        d1 = np.array([by[c][s]["d_10"] for s in common], dtype=float)
        sc = np.array([by[c][s]["score"] for s in common], dtype=float)
        ok = np.isfinite(d1) & np.isfinite(sc)
        rw, pw = spearmanr(d1[ok], sc[ok])
        if c == base:
            rp = pp = md = float("nan")
        else:
            dd = d1 - np.array([by[base][s]["d_10"] for s in common], dtype=float)
            ds = sc - np.array([by[base][s]["score"] for s in common], dtype=float)
            ok2 = np.isfinite(dd) & np.isfinite(ds)
            rp, pp = spearmanr(dd[ok2], ds[ok2])
            md = float(np.median(dd[ok2]))
        out.append(f"{c:24s} {scores.get(c, float('nan')):7.4f} {res[c]['d'][10]:8.4f} "
                   f"{rw:11.3f} {pw:9.2e} {rp:11.3f} {pp:9.2e} {md:10.4f}")
    note_a = ("'rho within' is confounded by scene difficulty; 'rho paired' is the "
              "control.")
    note_b = ("Arm-level: does a more self-consistent planner drive better? Compare "
              "the score and d(1s) columns.")
    out += ["", note_a, note_b, ""]
    return out


def horizon(curve, thresh):
    """First k with curve(k) > thresh, in steps; N_PLAN if it never crosses."""
    over = np.nonzero(np.asarray(curve) > thresh)[0]
    return int(over[0]) if len(over) else N_PLAN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--prefix", default="m2601_merged_")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--n-scenes", type=int, default=30)
    ap.add_argument("--rollouts-per-scene", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    out = args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)
    colours = colour_map(args.configs)

    # same scenes for every arm, so the curves are paired
    per_cfg_rows, scores = {}, {}
    for cfg in args.configs:
        rows = load_rollouts(args.runs_root / f"{args.prefix}{cfg}")
        # the published arm score uses every rollout, not the subset d(k) is read on
        scores[cfg] = float(np.nanmean([r["score"] for r in rows]))
        by_scene = {}
        for r in rows:
            by_scene.setdefault(r["scene"], []).append(r)
        per_cfg_rows[cfg] = by_scene
    common = sorted(set.intersection(*[set(v) for v in per_cfg_rows.values()]))
    scenes = common[: args.n_scenes]
    print(f"{len(common)} shared scenes, using {len(scenes)}")

    res, per_rollout = {}, []
    for cfg in args.configs:
        d_sum, d_cnt = np.zeros(N_PLAN), np.zeros(N_PLAN, dtype=int)
        c_sum, c_cnt = np.zeros(N_PLAN), np.zeros(N_PLAN, dtype=int)
        v_sum, v_cnt = np.zeros(N_PLAN), np.zeros(N_PLAN, dtype=int)
        v1s, n_plans, grid_bad = [], 0, 0
        run_dir = args.runs_root / f"{args.prefix}{cfg}"
        for sc in scenes:
            for r in sorted(per_cfg_rows[cfg][sc],
                            key=lambda x: x["rollout_id"])[: args.rollouts_per_scene]:
                asl = run_dir / "rollouts" / sc / r["rollout_id"] / "rollout.asl"
                if not asl.exists():
                    continue
                plans, realised = asyncio.run(_read_rollout(asl))
                ds, dc, cs, cc, vs, vc, v1, ok = rollout_curves(plans, realised)
                d_sum += ds
                d_cnt += dc
                c_sum += cs
                c_cnt += cc
                v_sum += vs
                v_cnt += vc
                n_plans += len(plans)
                grid_bad += 0 if ok else 1
                if np.isfinite(v1):
                    v1s.append(v1)
                rd = np.divide(ds, dc, out=np.full(N_PLAN, np.nan), where=dc > 0)
                per_rollout.append({
                    "config": cfg, "scene": sc, "rollout_id": r["rollout_id"],
                    "score": r["score"], "n_plans": len(plans),
                    "h05": horizon(np.nan_to_num(rd, nan=np.inf), 0.5),
                    "d_1": float(rd[1]) if np.isfinite(rd[1]) else None,
                    "d_10": float(rd[10]) if np.isfinite(rd[10]) else None,
                    "d_64": float(rd[64]) if np.isfinite(rd[64]) else None,
                })
        d = np.divide(d_sum, d_cnt, out=np.full(N_PLAN, np.nan), where=d_cnt > 0)
        c = np.divide(c_sum, c_cnt, out=np.full(N_PLAN, np.nan), where=c_cnt > 0)
        v = np.divide(v_sum, v_cnt, out=np.full(N_PLAN, np.nan), where=v_cnt > 0)
        res[cfg] = {"d": d, "c": c, "v": v, "d_cnt": d_cnt, "c_cnt": c_cnt,
                    "v1_mean": float(np.mean(v1s)) if v1s else None,
                    "n_plans": n_plans, "grid_bad": grid_bad,
                    "H": {str(t): horizon(np.nan_to_num(d, nan=np.inf), t) for t in THRESHOLDS},
                    "Hv": {str(t): horizon(np.nan_to_num(v, nan=np.inf), t) for t in THRESHOLDS}}
        print(f"  {cfg:22s} plans={n_plans:6d} V1={res[cfg]['v1_mean']:.4f} "
              f"H(0.5m)={res[cfg]['H']['0.5']} (null {res[cfg]['Hv']['0.5']})  "
              f"d(1)={d[1]:.3f} d(10)={d[10]:.3f} d(64)={d[64]:.3f}  "
              f"d/v @1s={d[10] / v[10]:.2f} @6.4s={d[64] / v[64]:.2f}")

    t = np.arange(N_PLAN) * STEP_US / 1e6

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for cfg in args.configs:
        axes[0].plot(t, res[cfg]["d"], color=colours[cfg], lw=1.6, label=cfg)
        axes[1].plot(t[:-1], res[cfg]["c"][:-1], color=colours[cfg], lw=1.6, label=cfg)
    axes[0].plot(t, res[args.configs[0]]["v"], color=INK, lw=1.2, ls="-.",
                 label="constant-velocity null")
    for ax, ttl, yl in ((axes[0], "d(k): plan vs what actually happened", "gap  (m)"),
                        (axes[1], "c(k): change between consecutive plans", "churn  (m)")):
        ax.set_xlabel("lead time  (s)")
        ax.set_ylabel(yl)
        ax.set_title(ttl)
        ax.legend(frameon=False, fontsize=8)
    for th in THRESHOLDS:
        axes[0].axhline(th, color=C3, lw=0.8, ls=":")
    axes[0].axvline(0.1, color=INK, lw=0.9, ls="--")
    axes[0].text(0.13, axes[0].get_ylim()[1] * 0.92, "executed\n(0.1 s)",
                 fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(out / "plots" / "p1_horizon_curves.png", dpi=150)
    plt.close(fig)

    # arm separation as a fraction of the baseline curve -- H2 reads this panel
    base = args.configs[0]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for cfg in args.configs[1:]:
        ax.plot(t, res[cfg]["d"] - res[base]["d"], color=colours[cfg], lw=1.6, label=cfg)
    ax.axhline(0, color=MUTED, lw=0.9)
    ax.axvline(0.1, color=INK, lw=0.9, ls="--")
    ax.set_xlabel("lead time  (s)")
    ax.set_ylabel(f"d(k) - d(k) of {base}  (m)")
    ax.set_title("H2: do the arms separate only in the far horizon?")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "plots" / "p2_arm_separation.png", dpi=150)
    plt.close(fig)

    # p3 -- the headline negative: self-consistency does not order the arms
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for cfg in args.configs:
        ax.scatter(res[cfg]["d"][10], scores[cfg], s=70, color=colours[cfg], zorder=3)
        ax.annotate(cfg.replace("slim_", ""), (res[cfg]["d"][10], scores[cfg]),
                    textcoords="offset points", xytext=(7, -3), fontsize=8.5, color=INK)
    ax.set_xlabel("d(1 s): plan self-consistency  (m, lower = more consistent)")
    ax.set_ylabel("closed-loop score")
    ax.set_title("A more self-consistent planner does not drive better")
    ax.margins(x=0.22, y=0.18)
    fig.tight_layout()
    fig.savefig(out / "plots" / "p3_consistency_vs_score.png", dpi=150)
    plt.close(fig)

    # p4 -- the confound: the same association, within-arm vs paired against baseline
    from scipy.stats import spearmanr

    by = {}
    for r in per_rollout:
        by.setdefault(r["config"], {})[r["scene"]] = r
    common = sorted(set.intersection(*[set(v) for v in by.values()]))
    base_d = np.array([by[base][s]["d_10"] for s in common], dtype=float)
    base_s = np.array([by[base][s]["score"] for s in common], dtype=float)
    rw, rp = [], []
    for cfg in args.configs[1:]:
        d1 = np.array([by[cfg][s]["d_10"] for s in common], dtype=float)
        sc = np.array([by[cfg][s]["score"] for s in common], dtype=float)
        ok = np.isfinite(d1) & np.isfinite(sc)
        rw.append(spearmanr(d1[ok], sc[ok])[0])
        ok2 = np.isfinite(d1 - base_d) & np.isfinite(sc - base_s)
        rp.append(spearmanr((d1 - base_d)[ok2], (sc - base_s)[ok2])[0])
    x = np.arange(len(args.configs) - 1)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.bar(x - 0.19, rw, width=0.36, color=C3, label="within arm (confounded)")
    ax.bar(x + 0.19, rp, width=0.36, color=C1, label="paired vs baseline (control)")
    ax.axhline(0, color=MUTED, lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("slim_", "") for c in args.configs[1:]],
                       rotation=12, fontsize=8.5)
    ax.set_ylabel("Spearman rho with closed-loop score")
    ax.set_title("Scene difficulty, not a link: the association does not survive pairing")
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "plots" / "p4_confound.png", dpi=150)
    plt.close(fig)

    metrics = {cfg: {"d": res[cfg]["d"].tolist(), "c": res[cfg]["c"].tolist(),
                     "v_null": res[cfg]["v"].tolist(),
                     "d_count": res[cfg]["d_cnt"].tolist(),
                     "H_steps": res[cfg]["H"], "H_null_steps": res[cfg]["Hv"],
                     "n_plans": res[cfg]["n_plans"],
                     "v1_mean_m": res[cfg]["v1_mean"], "grid_bad": res[cfg]["grid_bad"]}
               for cfg in args.configs}
    (out / "metrics.json").write_text(json.dumps(
        {"per_config": metrics, "per_rollout": per_rollout,
         "step_s": STEP_US / 1e6, "n_scenes": len(scenes), "scenes": scenes}, indent=2))
    (out / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    lines = ["Effective plan horizon (closed-loop logs, no GPU)", ""]
    lines.append(f"scenes={len(scenes)}  rollouts/scene={args.rollouts_per_scene}")
    lines.append("")
    lines.append(f"{'config':24s} {'plans':>7s} {'V1(m)':>7s} "
                 + " ".join(f"H({t}m)".rjust(8) for t in THRESHOLDS)
                 + f" {'d(0.1s)':>8s} {'d(1s)':>7s} {'d(6.4s)':>8s}")
    for cfg in args.configs:
        r = res[cfg]
        lines.append(f"{cfg:24s} {r['n_plans']:7d} {r['v1_mean']:7.4f} "
                     + " ".join(f"{r['H'][str(t)]:8d}" for t in THRESHOLDS)
                     + f" {r['d'][1]:8.3f} {r['d'][10]:7.3f} {r['d'][64]:8.3f}")
    lines += ["", "H(x) is the first lead step whose mean gap exceeds x metres;",
              f"one step is {STEP_US / 1e6:.1f} s, so H=10 means 1.0 s.", ""]
    lines += h2_section(res, per_rollout, args.configs, scores)
    lines.append("plan vs constant-velocity null (d / v; 1.0 = the plan adds nothing):")
    lines.append(f"{'config':24s} {'H_null(0.5m)':>13s} {'d/v @0.5s':>10s} "
                 f"{'@1s':>7s} {'@3s':>7s} {'@6.4s':>7s}")
    for cfg in args.configs:
        r = res[cfg]
        lines.append(f"{cfg:24s} {r['Hv']['0.5']:13d} {r['d'][5] / r['v'][5]:10.2f} "
                     f"{r['d'][10] / r['v'][10]:7.2f} {r['d'][30] / r['v'][30]:7.2f} "
                     f"{r['d'][64] / r['v'][64]:7.2f}")
    lines.append("")
    h05 = res[args.configs[0]]["H"]["0.5"]
    lines.append(f"H1: baseline H(0.5 m) = {h05} steps = {h05 * STEP_US / 1e6:.1f} s -> "
                 + ("short-horizon premise HOLDS" if h05 <= 10 else
                    "premise FAILS (closed loop uses the long horizon)" if h05 >= 30 else
                    "INCONCLUSIVE (between the pre-registered bounds)"))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
