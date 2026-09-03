"""Model-independent difficulty features for alpasim scenes, read offline from the usdz.

A usdz is a ZIP, so `rig_trajectories.usda` (68 KB, ego GT pose at 10 Hz over the 20 s clip),
`sequence_tracks.json` (~3 MB, other agents' pose tracks + cuboid dims) and `map.xodr` are read by
random access and the 400 MB meshes are never touched. All 913 `public_2601` scenes take ~40 s.

The parse is verified against the simulator: `gt_path_m` here matches the `gt_dist_traveled_m`
recorded in a rollout's metrics at r=1.00000 (median relative error 4.6e-5).

Of the features produced here only ego kinematics predict closed-loop difficulty --
`yaw_total_deg` (rho=-0.41 vs the unpruned scene score), `v_mean` and `gt_path_m` (-0.35 each).
Agent density does not (`n_agents_30m`, rho=+0.02). `make_hard_suite.py` uses
`z(v_mean) + z(yaw_total_deg)`; see `reports/evaluation/2026-09-03_difficulty-stratified-arms.html`
for the out-of-sample validation and for the fact that the score saturates above the 80th
percentile.

Usage:
  python extract_scene_feats.py --out outputs/scene_difficulty [--suite public_2601]
"""

import argparse
import csv
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
USDZ = Path("/mnt/nvme1n1/ad_vla/data/nre-artifacts/all-usdzs")
SUITES = Path("/home/cvlab21/project/chan/alpasim/data/scenes/sim_suites.csv")


def suite_scenes(suite):
    """{scene_id: uuid} for one test suite, in scene_id order."""
    rows = [r for r in csv.DictReader(SUITES.open()) if r["test_suite_id"] == suite]
    return {r["scene_id"]: r["uuid"] for r in sorted(rows, key=lambda r: r["scene_id"])}


def ego_track(z):
    """(t_sec (N,), xy (N,2), heading (N,)) in the rig-start frame, plus the clip's time base."""
    txt = z.read("rig_trajectories.usda").decode()
    off = int(re.search(r"absoluteTimeOffsetMicroSec = (\d+)", txt).group(1))
    end_tc = float(re.search(r"endTimeCode = ([\d.]+)", txt).group(1))
    blk = txt.split('def Xform "sensor_rig_0"')[1]
    tc, pos, hdg = [], [], []
    for t, body in re.findall(r"^\s*([\d.]+):\s*\(\s*(.*?)\s*\)\s*,?\s*$", blk, re.MULTILINE):
        n = [float(v) for v in re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", body)]
        if len(n) != 16:
            continue
        # USD matrix4d rows are (row0, row1, row2, translation); row0 is the local x axis in
        # world, so the heading is atan2 of its y over its x component.
        tc.append(float(t))
        pos.append(n[12:14])
        hdg.append(np.arctan2(n[1], n[0]))
    di = json.loads(z.read("data_info.json"))["pose-range"]
    dur = (di["end-timestamp_us"] - di["start-timestamp_us"]) / 1e6
    fps = end_tc / dur if dur > 0 else 24.0
    return np.array(tc) / fps, np.array(pos), np.unwrap(np.array(hdg)), off, dur


def agent_tracks(z):
    """[(class, timestamps_us, xy, cuboid_dims), ...] for every annotated agent."""
    s = json.loads(z.read("sequence_tracks.json"))["dummy_chunk_id"]
    td = s["tracks_data"]
    dims = s.get("cuboidtracks_data", {}).get("cuboids_dims", [])
    out = []
    for i, _ in enumerate(td["tracks_id"]):
        out.append((td["tracks_label_class"][i],
                    np.array(td["tracks_timestamps_us"][i], dtype=np.int64),
                    np.array([q[:2] for q in td["tracks_poses"][i]], dtype=float),
                    dims[i] if i < len(dims) else [4.5, 1.8, 1.5]))
    return out


def feats(scene_id, uuid):
    z = zipfile.ZipFile(USDZ / f"{uuid}.usdz")
    t, xy, hdg, t_us0, dur = ego_track(z)
    dt = np.diff(t)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    v = step / np.maximum(dt, 1e-6)                      # (N-1,) m/s
    a = np.diff(v) / np.maximum(dt[1:], 1e-6)            # (N-2,) m/s^2
    dpsi = np.diff(hdg) / np.maximum(dt, 1e-6)           # (N-1,) rad/s
    curv = np.abs(dpsi) / np.maximum(v, 0.5)             # (N-1,) 1/m
    f = {
        "scene_id": scene_id, "uuid": uuid, "duration_s": float(dur),
        "gt_path_m": float(step.sum()),
        "v_mean": float(v.mean()), "v_max": float(v.max()), "v_min": float(v.min()),
        "v_std": float(v.std()),
        "stop_frac": float((v < 0.5).mean()),
        "decel_max": float(-a.min()) if len(a) else 0.0,
        "accel_max": float(a.max()) if len(a) else 0.0,
        "jerk_rms": float(np.sqrt((np.diff(a) ** 2).mean())) if len(a) > 1 else 0.0,
        "yaw_total_deg": float(np.degrees(np.abs(hdg[-1] - hdg[0]))),
        "yaw_abs_deg": float(np.degrees(np.abs(dpsi * dt).sum())),
        "yawrate_max": float(np.degrees(np.abs(dpsi).max())),
        "curv_max": float(curv.max()),
    }
    tracks = agent_tracks(z)
    n_close = n_veryclose = n_ped_close = 0
    dmin_all, lead_gap = [], []
    for cls, ts, p, _dim in tracks:
        if len(p) < 2:
            continue
        tt = (ts - t_us0) / 1e6
        m = (tt >= t[0]) & (tt <= t[-1])
        if m.sum() < 2:
            continue
        ex = np.interp(tt[m], t, xy[:, 0])
        ey = np.interp(tt[m], t, xy[:, 1])
        eh = np.interp(tt[m], t, hdg)
        dx, dy = p[m, 0] - ex, p[m, 1] - ey
        d = np.hypot(dx, dy)
        fwd = dx * np.cos(eh) + dy * np.sin(eh)          # longitudinal in the ego frame
        lat = -dx * np.sin(eh) + dy * np.cos(eh)
        dmin_all.append(d.min())
        n_close += d.min() < 30
        n_veryclose += d.min() < 8
        n_ped_close += cls == "person" and d.min() < 15
        inlane = (np.abs(lat) < 2.0) & (fwd > 0) & (fwd < 60)
        if inlane.any():
            lead_gap.append(float(fwd[inlane].min()))
    f.update(
        n_agents=len(tracks), n_agents_30m=int(n_close), n_agents_8m=int(n_veryclose),
        n_ped_15m=int(n_ped_close),
        dmin_agent=float(min(dmin_all)) if dmin_all else 999.0,
        lead_gap_min=float(min(lead_gap)) if lead_gap else 999.0,
        n_lead=len(lead_gap),
    )
    try:
        x = z.read("map.xodr").decode("utf-8", "replace")
        f["n_junctions"] = x.count("<junction ")
        f["n_roads"] = x.count("<road ")
        f["n_signals"] = x.count("<signal ")
    except KeyError:
        f["n_junctions"] = f["n_roads"] = f["n_signals"] = -1
    return f


def hard_score(rows):
    """z(v_mean) + z(yaw_total_deg), z-scored over whatever set is passed in."""
    def z(v):
        return (v - v.mean()) / (v.std() + 1e-9)

    return (z(np.array([r["v_mean"] for r in rows]))
            + z(np.array([r["yaw_total_deg"] for r in rows])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="public_2601")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="only the first N scenes (smoke test)")
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else REPO / args.out
    out.mkdir(parents=True, exist_ok=True)

    sc = suite_scenes(args.suite)
    items = list(sc.items())[:args.limit] if args.limit else list(sc.items())
    rows = []
    for i, (sid, u) in enumerate(items):
        try:
            rows.append(feats(sid, u))
        except Exception as e:                                    # noqa: BLE001
            print(f"[skip] {sid}: {type(e).__name__}: {e}", file=sys.stderr)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(items)}", file=sys.stderr, flush=True)

    hs = hard_score(rows)
    for i, r in enumerate(rows):
        r["hard_score"] = float(hs[i])
    rows.sort(key=lambda r: -r["hard_score"])

    # filenames drop the suite's underscore: outputs/scene_difficulty already holds
    # scene_feats_public2601.json under that spelling and other scripts point at it
    tag = args.suite.replace("_", "")
    (out / f"scene_feats_{tag}.json").write_text(json.dumps(rows))
    keys = ["scene_id", "uuid", "hard_score", "v_mean", "yaw_total_deg", "gt_path_m", "v_min",
            "v_max", "stop_frac", "decel_max", "yawrate_max", "curv_max", "n_agents",
            "n_agents_30m", "n_agents_8m", "n_ped_15m", "dmin_agent", "lead_gap_min",
            "n_junctions", "n_signals"]
    with (out / f"scene_difficulty_{tag}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} scenes -> {out}")
    print(f"  hard_score {hs.min():+.2f} .. {hs.max():+.2f} | "
          f"gt_path_m < 5 m (auto scores 1.0): {sum(r['gt_path_m'] < 5 for r in rows)}")


if __name__ == "__main__":
    main()
