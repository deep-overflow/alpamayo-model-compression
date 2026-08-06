"""Per-collision forensics for the alpasim closed-loop matrix.

For every at-fault collision rollout, extracts the collision timestamp from the
per-rollout metrics time series and the driver's Chain-of-Causation text around that
moment from the ASL log, then compares:

  1. CoC health in the pre-collision window vs the rollout's overall rate
     (does reasoning collapse *concentrate* before a crash?),
  2. what the other configs did on the same scene (scene-difficulty control),
  3. collision type (front / lateral / rear) per config.

Usage (alpasim venv):
    cd /home/cvlab21/project/chan/alpasim && uv run python \
        .../analyze_collisions.py --runs-root /home/cvlab21/project/chan/alpasim-runs \
        --out .../outputs/alpasim_collisions
"""

import argparse
import asyncio
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analyze_alpasim import coc_stats  # noqa: E402

# the label-free j_traj checkpoint joins the shipped four; kept here rather than
# imported so adding a config never silently changes analyze_alpasim's default
CONFIGS = ["baseline", "slim_cocsafe_r20", "slim_cocsafe_r30", "slim_integrated_mag",
           "slim_j_traj_r20"]

PRE_WINDOW_S = 5.0  # how far back from the collision to inspect reasoning


def metric_series(df, name):
    """(timestamps_us, values) for one metric of a per-rollout metrics.parquet."""
    s = df[df["name"] == name]
    if s.empty:
        return None, None
    return s["timestamps_us"].to_numpy(dtype=float), s["values"].to_numpy(dtype=float)


def first_event_us(df, name):
    """Timestamp at which a 0/1 metric first fires, or None."""
    ts, v = metric_series(df, name)
    if ts is None:
        return None
    hit = np.nonzero(v > 0)[0]
    return float(ts[hit[0]]) if hit.size else None


def first_at_fault_us(df):
    """At-fault collision = front or lateral. (collision_at_fault itself is derived
    during aggregation and absent from the per-rollout time series.)"""
    times = [t for t in (first_event_us(df, "collision_front"),
                         first_event_us(df, "collision_lateral")) if t is not None]
    return min(times) if times else None


async def read_coc_timed(asl_path):
    """[(time_us, reasoning_text)] in log order, pairing requests with responses."""
    from alpasim_utils.logs import async_read_pb_log

    out, now_us = [], None
    async for e in async_read_pb_log(str(asl_path)):
        w = e.WhichOneof("log_entry")
        if w == "driver_request":
            now_us = float(e.driver_request.time_now_us)
        elif w == "driver_return":
            blob = e.driver_return.debug_info.unstructured_debug_info
            if not blob:
                continue
            try:
                dbg = pickle.loads(blob)
            except Exception:
                continue
            t = dbg.get("reasoning_text")
            stamp = now_us
            if stamp is None and e.driver_return.trajectory.poses:
                stamp = float(e.driver_return.trajectory.poses[0].timestamp_us)
            if stamp is not None:
                out.append((stamp, "" if t is None else str(t)))
    return out


def classify(text):
    """healthy / empty / soup, mirroring analyze_alpasim.coc_stats thresholds."""
    st = coc_stats([text])
    if st["empty_frac"] == 1.0:
        return "empty"
    return "soup" if st["soup_frac"] == 1.0 else "healthy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--prefix", default="matrix_")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- per-rollout table across configs
    rollouts = {}
    for cfg in CONFIGS:
        run = args.runs_root / f"{args.prefix}{cfg}"
        summary = json.loads((run / "aggregate" / "results-summary.json").read_text())
        rows = []
        for r in summary["rollouts"]:
            d = run / "rollouts" / r["clipgt_id"] / r["rollout_id"]
            rows.append({
                "scene": r["clipgt_id"], "rollout_id": r["rollout_id"], "dir": d,
                "score": r["score"], "passed": r["passed"],
                **{k: r["metrics"].get(k, 0.0) for k in
                   ["collision_at_fault", "collision_any", "collision_front",
                    "collision_lateral", "collision_rear", "offroad",
                    "dist_traveled_m", "progress_clipped_rel"]},
            })
        rollouts[cfg] = rows

    # ---- collision forensics
    cases = []
    for cfg in CONFIGS:
        for r in rollouts[cfg]:
            if r["collision_at_fault"] <= 0:
                continue
            df = pd.read_parquet(r["dir"] / "metrics.parquet")
            t_col = first_at_fault_us(df)
            coc = asyncio.run(read_coc_timed(r["dir"] / "rollout.asl"))
            labels = [(t, classify(x)) for t, x in coc]
            overall = {k: sum(lab == k for _, lab in labels) / max(len(labels), 1)
                       for k in ["healthy", "empty", "soup"]}
            pre, at_text = {}, None
            if t_col is not None and labels:
                lo = t_col - PRE_WINDOW_S * 1e6
                win = [(t, lab) for t, lab in labels if lo <= t <= t_col]
                pre = {k: sum(lab == k for _, lab in win) / max(len(win), 1)
                       for k in ["healthy", "empty", "soup"]}
                pre["n_steps"] = len(win)
                before = [x for t, x in coc if t <= t_col]
                at_text = before[-1] if before else None
            cases.append({
                "config": cfg, "scene": r["scene"], "rollout_id": r["rollout_id"],
                "collision_time_s": None if t_col is None else round(t_col / 1e6, 2),
                "type": {k: r[f"collision_{k}"] for k in ["front", "lateral", "rear"]},
                "score": r["score"], "dist_traveled_m": round(r["dist_traveled_m"], 1),
                "coc_overall": overall, "coc_pre_collision": pre,
                "coc_at_collision": at_text,
                "n_coc_steps": len(labels),
            })

    # ---- scene-difficulty control: what did every config do on the collision scenes?
    collision_scenes = sorted({c["scene"] for c in cases})
    control = {}
    for s in collision_scenes:
        control[s] = {}
        for cfg in CONFIGS:
            rs = [r for r in rollouts[cfg] if r["scene"] == s]
            control[s][cfg] = {
                "n_rollouts": len(rs),
                "n_collision_at_fault": int(sum(r["collision_at_fault"] > 0 for r in rs)),
                "n_offroad": int(sum(r["offroad"] > 0 for r in rs)),
                "mean_score": float(np.mean([r["score"] for r in rs])) if rs else None,
            }

    # ---- per-config collision type + rate summary
    per_config = {}
    for cfg in CONFIGS:
        rs = rollouts[cfg]
        n = len(rs)
        per_config[cfg] = {
            "n_rollouts": n,
            "n_collision_at_fault": int(sum(r["collision_at_fault"] > 0 for r in rs)),
            "n_collision_any": int(sum(r["collision_any"] > 0 for r in rs)),
            "front": float(np.mean([r["collision_front"] for r in rs])),
            "lateral": float(np.mean([r["collision_lateral"] for r in rs])),
            "rear": float(np.mean([r["collision_rear"] for r in rs])),
            "scenes_with_collision": len({r["scene"] for r in rs if r["collision_at_fault"] > 0}),
        }

    out = {"pre_window_s": PRE_WINDOW_S, "per_config": per_config,
           "cases": cases, "scene_control": control}
    (args.out / "metrics.json").write_text(json.dumps(out, indent=2, default=str))

    # ---- human-readable summary
    L = [f"충돌 개별 분석 — 과실 충돌 {len(cases)}건 (pre-window {PRE_WINDOW_S}s)", ""]
    L.append(f"{'config':22s} {'col@f':>6s} {'씬수':>5s} {'front':>7s} {'lateral':>8s} {'rear':>7s}")
    L.append("-" * 60)
    for cfg in CONFIGS:
        p = per_config[cfg]
        L.append(f"{cfg:22s} {p['n_collision_at_fault']:6d} {p['scenes_with_collision']:5d} "
                 f"{p['front']:7.3f} {p['lateral']:8.3f} {p['rear']:7.3f}")
    L.append("")
    L.append("충돌별 CoC 상태 (충돌 직전 5초 / 롤아웃 전체):")
    for c in cases:
        pre, ov = c["coc_pre_collision"], c["coc_overall"]
        pre_s = (f"healthy {pre.get('healthy', float('nan')):.2f} empty {pre.get('empty', 0):.2f} "
                 f"soup {pre.get('soup', 0):.2f} (n={pre.get('n_steps', 0)})") if pre else "n/a"
        L.append(f"  [{c['config']}] {c['scene'][7:19]} t={c['collision_time_s']}s "
                 f"score={c['score']:.2f} dist={c['dist_traveled_m']}m")
        L.append(f"      직전: {pre_s}")
        L.append(f"      전체: healthy {ov['healthy']:.2f} empty {ov['empty']:.2f} soup {ov['soup']:.2f}")
        if c["coc_at_collision"] is not None:
            L.append(f"      충돌시점 CoC: {c['coc_at_collision'][:110]}")
    L.append("")
    L.append("충돌 발생 씬에서 각 구성의 결과 (씬 난이도 통제):")
    L.append(f"  {'scene':14s} " + " ".join(f"{c.replace('slim_', '')[:12]:>13s}" for c in CONFIGS))
    for s in collision_scenes:
        cells = []
        for cfg in CONFIGS:
            v = control[s][cfg]
            cells.append(f"{v['n_collision_at_fault']}충/{v['n_offroad']}이/{v['mean_score']:.2f}")
        L.append(f"  {s[7:19]:14s} " + " ".join(f"{c:>13s}" for c in cells))
    L.append("  (형식: 과실충돌 rollout수 / 이탈 rollout수 / 평균 score, 씬당 2 rollouts)")

    (args.out / "summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
