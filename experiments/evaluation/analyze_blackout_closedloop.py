"""Per-step t0 blackout in closed loop (alpasim, Origin150) vs the stored baseline run.

Plan and gates: plans/2026-10-19_camera-blackout-closedloop.md.

The blackout run (`m2601_merged_bkrand_baseline`) is the unpruned model with one random
camera's newest frame replaced by black on every policy step
(`driver.inference.blackout_t0_random_camera=true`). The reference is the stored
`m2601_merged_baseline` (same sim code, scene files, cards and thread count). alpasim has
no seed control, so every comparison is paired by scene.

  G0  at scale: every driver_return of the blackout run carries `blackout_cam`, the camera
      recomputes from (session seed, inference count), and no plan is empty.
  G1  scene-score delta (blackout - baseline), bootstrap CI of the mean.

Also reported: gates (offroad, at-fault collision), progress / plan-deviation / lane margin
per rollout, and plan churn between consecutive plans split by which camera was blacked out
on the later step (exploratory: does blacking out the front tele camera move the plan more?).

Runs under alpasim's venv (reads the ASL protobuf logs):

    /mnt/nvme1n1/ad_vla/venvs/alpasim-chan/bin/python \
        experiments/evaluation/analyze_blackout_closedloop.py
"""

import argparse
import asyncio
import collections
import json
import pickle
import random
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
RUNS = Path("/home/cvlab21/project/chan/alpasim-runs")

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})

CAMS = ["camera_cross_left_120fov", "camera_front_wide_120fov",
        "camera_cross_right_120fov", "camera_front_tele_30fov"]
SHORT = {c: c.replace("camera_", "").replace("fov", "") for c in CAMS}
COLORS = dict(zip(CAMS, (C1, C4, C2, C3)))
METRICS = ("progress_clipped_rel", "dist_traveled_m", "plan_deviation",
           "min_distance_to_lane_boundary_m", "dist_to_gt_trajectory")
GATES = ("offroad", "collision_at_fault", "offroad_or_collision_at_fault", "collision_any")
STEP_US = 100_000
CHURN_K = {"0.5s": 5, "1.0s": 10, "3.0s": 30}
N_BOOT = 200_000


def boot_ci(d, n_boot=N_BOOT, seed=0, chunk=10_000):
    d = np.asarray(d, dtype=float)
    if len(d) == 0:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    means = []
    for s in range(0, n_boot, chunk):
        idx = rng.randint(0, len(d), size=(min(chunk, n_boot - s), len(d)))
        means.append(d[idx].mean(1))
    lo, hi = np.percentile(np.concatenate(means), [2.5, 97.5])
    return float(lo), float(hi)


def paired(base, x):
    b, x = np.asarray(base, float), np.asarray(x, float)
    d = x - b
    lo, hi = boot_ci(d)
    return {"n": len(d), "base_mean": float(b.mean()), "blackout_mean": float(x.mean()),
            "delta_mean": float(d.mean()), "ci": [lo, hi], "delta_median": float(np.median(d)),
            "wilcoxon_p": float(wilcoxon(d).pvalue) if len(d) > 1 and np.any(d != 0) else 1.0,
            "neg_pos_zero": [int((d < 0).sum()), int((d > 0).sum()), int((d == 0).sum())]}


def load_summary(run):
    s = json.loads((run / "aggregate" / "results-summary.json").read_text())
    return s["rollouts"]


def by_scene(rollouts, key):
    acc = collections.defaultdict(list)
    for r in rollouts:
        v = r["score"] if key == "score" else r["metrics"].get(key)
        if v is not None:
            acc[r["clipgt_id"]].append(float(v))
    return {k: float(np.mean(v)) for k, v in acc.items()}


async def _read(asl):
    from alpasim_utils.logs import async_read_pb_log

    seed, plans, cams, empty = None, [], [], 0
    async for e in async_read_pb_log(str(asl)):
        f = e.WhichOneof(e.DESCRIPTOR.oneofs[0].name)
        if f == "driver_session_request":
            seed = e.driver_session_request.random_seed
        elif f == "driver_return":
            poses = e.driver_return.trajectory.poses
            blob = e.driver_return.debug_info.unstructured_debug_info
            cams.append(pickle.loads(blob).get("blackout_cam") if blob else None)
            if not poses:
                empty += 1
                plans.append(None)
                continue
            ts = np.fromiter((p.timestamp_us for p in poses), dtype=np.int64, count=len(poses))
            xy = np.array([[p.pose.vec.x, p.pose.vec.y] for p in poses], dtype=float)
            plans.append((ts, xy))
    return seed, plans, cams, empty


def read_rollout(asl):
    seed, plans, cams, empty = asyncio.run(_read(asl))
    expect = [random.Random(f"blackout:{seed}:{i}").choice(CAMS) for i in range(1, len(cams) + 1)]
    # churn between consecutive plans, attributed to the camera blacked out on the later one
    churn = []
    for i, (a, b) in enumerate(pairwise(plans)):
        if a is None or b is None or b[0][0] - a[0][0] != STEP_US:
            continue
        row = {"cam": cams[i + 1]}
        for name, k in CHURN_K.items():
            if k + 1 < len(a[0]) and k < len(b[0]):
                row[name] = float(np.hypot(*(a[1][k + 1] - b[1][k])))
        churn.append(row)
    return {"asl": str(asl), "scene": Path(asl).parent.parent.name, "seed": seed,
            "n_returns": len(cams), "empty": empty,
            "n_with_cam": sum(c is not None for c in cams),
            "recompute_match": sum(a == b for a, b in zip(cams, expect)),
            "cam_counts": dict(collections.Counter(c for c in cams if c)), "churn": churn}


def read_run(run, workers):
    asls = sorted(str(p) for p in run.glob("rollouts/*/*/rollout.asl"))
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(read_rollout, asls))


def scene_mean_churn(rows, name):
    acc = collections.defaultdict(list)
    for r in rows:
        acc[r["scene"]] += [c[name] for c in r["churn"] if name in c]
    return {s: float(np.mean(v)) for s, v in acc.items() if v}


def churn_tables(base_rows, bk_rows):
    """Per-scene mean churn (between runs) and per-camera churn within the blackout run."""
    out = {"between_runs": {}, "by_camera": {}, "tele_vs_rest_within_rollout": {}}
    for name in CHURN_K:
        b, x = scene_mean_churn(base_rows, name), scene_mean_churn(bk_rows, name)
        common = sorted(set(b) & set(x))
        out["between_runs"][name] = paired([b[s] for s in common], [x[s] for s in common])
        # per camera: rollout-level mean churn on steps where that camera was blacked out
        per_cam = {}
        for cam in CAMS:
            vals = [np.mean([c[name] for c in r["churn"] if c["cam"] == cam and name in c])
                    for r in bk_rows if any(c["cam"] == cam and name in c for c in r["churn"])]
            if not vals:
                continue
            lo, hi = boot_ci(vals)
            per_cam[SHORT[cam]] = {"n_rollouts": len(vals), "mean": float(np.mean(vals)),
                                   "ci": [lo, hi], "median": float(np.median(vals))}
        out["by_camera"][name] = per_cam
        # within rollout: tele steps minus other steps (removes scene and rollout effects)
        d = []
        for r in bk_rows:
            t = [c[name] for c in r["churn"] if c["cam"] == CAMS[3] and name in c]
            o = [c[name] for c in r["churn"] if c["cam"] != CAMS[3] and name in c]
            if t and o:
                d.append(np.mean(t) - np.mean(o))
        if len(d) < 2:
            continue
        lo, hi = boot_ci(d)
        out["tele_vs_rest_within_rollout"][name] = {
            "n_rollouts": len(d), "delta_mean": float(np.mean(d)), "ci": [lo, hi],
            "delta_median": float(np.median(d)), "wilcoxon_p": float(wilcoxon(d).pvalue)}
    return out


def plots(res, base_sc, bk_sc, out):
    common = sorted(set(base_sc) & set(bk_sc))
    b = np.array([base_sc[s] for s in common])
    x = np.array([bk_sc[s] for s in common])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    jit = np.random.RandomState(0).uniform(-0.01, 0.01, size=(2, len(b)))
    ax.plot(b + jit[0], x + jit[1], "o", ms=4, alpha=0.6, color=C1)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8)
    ax.set_xlabel("baseline scene score")
    ax.set_ylabel("blackout scene score")
    ax.set_title(f"per scene (n={len(common)}), jittered")
    ax = axes[1]
    ax.hist(x - b, bins=np.linspace(-1, 1, 41), color=C1, alpha=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    p = res["score"]
    ax.set_title(f"scene delta: mean {p['delta_mean']:+.3f} [{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}]")
    ax.set_xlabel("blackout - baseline scene score")
    ax.set_ylabel("scenes")
    fig.tight_layout()
    fig.savefig(out / "score_paired.png", dpi=150)
    plt.close(fig)

    ch = res["churn"]
    fig, axes = plt.subplots(1, len(CHURN_K), figsize=(5 * len(CHURN_K), 4.2))
    for ax, name in zip(axes, CHURN_K):
        names = list(ch["by_camera"][name])
        m = [ch["by_camera"][name][n]["mean"] for n in names]
        lo = [m[i] - ch["by_camera"][name][n]["ci"][0] for i, n in enumerate(names)]
        hi = [ch["by_camera"][name][n]["ci"][1] - m[i] for i, n in enumerate(names)]
        ax.errorbar(range(len(names)), m, yerr=[lo, hi], fmt="o", capsize=4,
                    color=C1, label="blackout run, by blacked camera")
        br = ch["between_runs"][name]
        ax.axhline(br["base_mean"], color=MUTED, ls="--", lw=1, label="baseline run (no blackout)")
        ax.set_xticks(range(len(names)), names, rotation=15)
        ax.set_ylabel(f"plan churn at {name} lead (m)")
        ax.set_title(f"consecutive-plan change, {name}")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "churn_by_camera.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, default=RUNS / "m2601_merged_baseline")
    ap.add_argument("--blackout", type=Path, default=RUNS / "m2601_merged_bkrand_baseline")
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "blackout_closedloop")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    (args.out / "plots").mkdir(parents=True, exist_ok=True)

    base_r, bk_r = load_summary(args.baseline), load_summary(args.blackout)
    res = {"n_rollouts": {"baseline": len(base_r), "blackout": len(bk_r)},
           "status": {"baseline": dict(collections.Counter(r["status"] for r in base_r)),
                      "blackout": dict(collections.Counter(r["status"] for r in bk_r))}}
    base_sc, bk_sc = by_scene(base_r, "score"), by_scene(bk_r, "score")
    common = sorted(set(base_sc) & set(bk_sc))
    res["n_scenes_paired"] = len(common)
    res["score"] = paired([base_sc[s] for s in common], [bk_sc[s] for s in common])
    res["gates"] = {}
    for g in GATES:
        b, x = by_scene(base_r, g), by_scene(bk_r, g)
        c = sorted(set(b) & set(x))
        res["gates"][g] = {**paired([b[s] for s in c], [x[s] for s in c]),
                           "rollouts_hit": [int(sum(r["metrics"].get(g, 0) > 0 for r in base_r)),
                                            int(sum(r["metrics"].get(g, 0) > 0 for r in bk_r))]}
    res["metrics"] = {}
    for m in METRICS:
        b, x = by_scene(base_r, m), by_scene(bk_r, m)
        c = sorted(set(b) & set(x))
        res["metrics"][m] = paired([b[s] for s in c], [x[s] for s in c])

    bk_rows = read_run(args.blackout, args.workers)
    base_rows = read_run(args.baseline, args.workers)
    tot = sum(r["n_returns"] for r in bk_rows)
    res["G0"] = {"rollouts": len(bk_rows), "driver_returns": tot,
                 "with_blackout_cam": sum(r["n_with_cam"] for r in bk_rows),
                 "recompute_match": sum(r["recompute_match"] for r in bk_rows),
                 "empty_plans": sum(r["empty"] for r in bk_rows),
                 "camera_counts": dict(sum((collections.Counter(r["cam_counts"]) for r in bk_rows),
                                           collections.Counter())),
                 "baseline_rows_with_cam": sum(r["n_with_cam"] for r in base_rows)}
    g0 = res["G0"]
    res["G0"]["status"] = "PASS" if (g0["with_blackout_cam"] == tot == g0["recompute_match"]
                                     and g0["empty_plans"] == 0
                                     and g0["baseline_rows_with_cam"] == 0) else "FAIL"
    res["churn"] = churn_tables(base_rows, bk_rows)
    d = res["score"]
    res["G1"] = "PASS" if d["ci"][1] < 0 else ("FAIL" if d["ci"][0] <= 0 <= d["ci"][1] else "POSITIVE")
    plots(res, base_sc, bk_sc, args.out / "plots")

    (args.out / "config.json").write_text(json.dumps({
        "baseline": str(args.baseline), "blackout": str(args.blackout), "n_boot": N_BOOT,
        "churn_leads": CHURN_K, "plan": "plans/2026-10-19_camera-blackout-closedloop.md",
        "intervention": "driver.inference.blackout_t0_random_camera=true",
    }, indent=2))
    (args.out / "metrics.json").write_text(json.dumps(res, indent=2))

    def fmt(p):
        return (f"{p['delta_mean']:+.4f} [{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}] "
                f"(base {p['base_mean']:.4f} -> {p['blackout_mean']:.4f}; median {p['delta_median']:+.4f}, "
                f"Wilcoxon p={p['wilcoxon_p']:.3g}, scenes neg/pos/zero {p['neg_pos_zero']})")
    lines = ["per-step t0 blackout (1 random camera) vs stored baseline, alpasim Origin150",
             f"rollouts {res['n_rollouts']} | status {res['status']} | paired scenes {len(common)}",
             f"G0: {res['G0']}", f"G1 (score CI excludes 0): {res['G1']}",
             f"score: {fmt(res['score'])}", "gates (per-scene rate; rollouts hit base/blackout):"]
    for g, v in res["gates"].items():
        lines.append(f"  {g:30s} {fmt(v)} hits {v['rollouts_hit']}")
    lines.append("rollout metrics (per-scene mean):")
    for m, v in res["metrics"].items():
        lines.append(f"  {m:32s} {fmt(v)}")
    for name in CHURN_K:
        ch = res["churn"]
        lines.append(f"plan churn {name}: between runs {fmt(ch['between_runs'][name])}")
        for n, v in ch["by_camera"][name].items():
            lines.append(f"    blacked {n:18s} rollouts {v['n_rollouts']} mean {v['mean']:.4f} "
                         f"[{v['ci'][0]:.4f}, {v['ci'][1]:.4f}] median {v['median']:.4f}")
        t = ch["tele_vs_rest_within_rollout"].get(name)
        if t is None:
            continue
        lines.append(f"    tele - rest within rollout: {t['delta_mean']:+.4f} [{t['ci'][0]:+.4f}, "
                     f"{t['ci'][1]:+.4f}] median {t['delta_median']:+.4f} p={t['wilcoxon_p']:.3g} "
                     f"(n={t['n_rollouts']})")
    (args.out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
