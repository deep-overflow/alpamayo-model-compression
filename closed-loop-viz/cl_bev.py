"""Heading-up, ego-centred BEV frames of a stored rollout, on the scene's own map. No GPU.

For each arm (a merged run directory) the WORSE of the scene's two rollouts is drawn, the same choice
replay_camera.py / render_replay.py make, so BEV and camera frames are of one rollout.

What is drawn and from where:
  road surface      union of lane polygons (left edge + reversed right edge) from the scene map
                    (`Artifact(usdz).map`) -- the polygons eval/scorers/offroad.py tests the ego box against
  lane / road edge  the map's lane edges (thin) and road edges (white)
  RED, per frame    only what the scorer flagged at that timestamp: the road edge the ego box touches on
                    `offroad` frames, the traffic box within 0.3 m on `collision_*` frames
  ego, traffic, plan, GT path/car, driven trail     as in render_replay.py (read_rollout)

Off-road (eval/scorers/offroad.py): no nearby lane, nor the union of nearby lanes, contains the ego box,
AND the box touches the closest road edge. So the red edge is the scorer's own reason.

Frame alignment: camera frame i (render request i) ends at the (i+1)-th actor-pose timestamp, so BEV
frame i is drawn at ts[i+1]; event_frame = first frame whose time reaches the event.

Writes <out>/<arm>/bev/%05d.jpg (640x480) and <out>/<arm>/bev_meta.json (verdict, event frame/time, per-
frame CoC as run-length [[first_frame, text], ...]).

  python closed-loop-viz/cl_bev.py --scene clipgt-... --out DIR \
      --arms moop=/path/to/h100_merged_slim_dual_u40_v2 fm=... ce=...
"""
import argparse
import asyncio
import csv
import json
import math
import sys
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
import shapely.ops
from matplotlib.collections import LineCollection, PolyCollection
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_replay import clean_coc, read_rollout

ARTIFACTS = Path("/mnt/nvme1n1/ad_vla/data/nre-artifacts/scenesets")
SUITES_CSV = Path("/home/cvlab21/project/chan/advla/alpasim/data/scenes/sim_suites.csv")
W, H, DPI = 640, 480, 100
LAT, AHEAD, BEHIND = 30.0, 33.0, 12.0          # 60 m x 45 m -> 10.67 px/m on both axes
BG, ROAD, LANE, EDGE, TRAIL = "#15181d", "#373d48", "#59626f", "#b4bcc7", "#e3e6ea"
GT, PLAN, TRAF, PASS, FAIL = "#6fa8ff", "#ff9f43", "#8d939c", "#3fbf8a", "#f0605a"


def usdz_for(scene, suite="public_2601"):
    """scene_id -> usdz path, the way make_sceneset.py resolves it."""
    with open(SUITES_CSV) as fh:
        uuid = next((r["uuid"] for r in csv.DictReader(fh)
                     if r.get("test_suite_id") == suite and r["scene_id"] == scene), None)
    if uuid is None:
        raise SystemExit(f"{scene}: not in {suite} of {SUITES_CSV}")
    p = next(ARTIFACTS.glob(f"*/{uuid}.usdz"), None)
    if p is None:
        raise SystemExit(f"{scene}: usdz {uuid} not found under {ARTIFACTS}")
    return p


def load_map(scene):
    from alpasim_utils.artifact import Artifact
    return Artifact(source=str(usdz_for(scene))).map


def verdict_of(r):
    m = r.get("metrics") or {}
    return "offroad" if float(m.get("offroad") or 0) else ("collision" if float(m.get("collision_at_fault") or 0) else "pass")


def worst_rollout(cfg, scene):
    d = json.loads((Path(cfg) / "aggregate/results-summary.json").read_text())
    return min((r for r in d["rollouts"] if r["clipgt_id"] == scene), key=lambda r: float(r["score"]))


def to_ego(pts, ex, ey, yaw):
    """World -> heading-up ego frame (x right, y forward)."""
    c, s = math.cos(math.pi / 2 - yaw), math.sin(math.pi / 2 - yaw)
    d = np.asarray(pts, dtype=float)[..., :2] - (ex, ey)
    return np.stack([c * d[..., 0] - s * d[..., 1], s * d[..., 0] + c * d[..., 1]], -1)


def box(x, y, yaw, length, width):
    c, s = math.cos(yaw), math.sin(yaw)
    loc = np.array([[length / 2, width / 2], [length / 2, -width / 2], [-length / 2, -width / 2], [-length / 2, width / 2]])
    return np.stack([x + c * loc[:, 0] - s * loc[:, 1], y + s * loc[:, 0] + c * loc[:, 1]], -1)


def map_geometry(vm, lo, hi):
    clip = shapely.box(*lo, *hi)
    polys, lanes = [], []
    for ln in vm.lanes:
        try:
            if ln.left_edge is not None and ln.right_edge is not None:
                p = shapely.Polygon(np.vstack([ln.left_edge.points[:, :2],
                                               np.flip(ln.right_edge.points[:, :2], 0)])).buffer(0)
            else:
                p = shapely.LineString(ln.center.points[:, :2]).buffer(3.7 / 2)
        except (ValueError, shapely.errors.GEOSException):
            continue
        if not p.intersects(clip):
            continue
        polys.append(p)
        lanes += [e.points[:, :2] for e in (ln.left_edge, ln.right_edge) if e is not None]
    road = shapely.ops.unary_union(polys).intersection(clip)
    ext, ints = [], []
    for g in getattr(road, "geoms", [road]):
        if g.geom_type == "Polygon":
            ext.append(np.asarray(g.exterior.coords))
            ints += [np.asarray(i.coords) for i in g.interiors]
    edges = [e.polyline.points[:, :2] for e in vm.road_edges
             if shapely.LineString(e.polyline.points[:, :2]).intersects(clip)]
    return ext, ints, lanes, edges


def runs(texts):
    out = []
    for i, c in enumerate(texts):
        if not out or out[-1][1] != c:
            out.append([i, c])
    return out


def frame_cocs(dat, ts):
    """Reasoning in force at each frame (frame i is drawn at ts[i+1]): the latest NON-empty one issued at or
    before it. Some driver returns carry no reasoning; like replay_camera.py, keep the previous caption then
    rather than blanking it. Frames before the first reasoning get ''."""
    nz = [(t, clean_coc(x)) for t, x in dat["cocs"] if clean_coc(x)]
    nz_t = np.array([t for t, _ in nz], dtype=np.int64)
    out = []
    for i in range(len(ts) - 1):
        k = int(np.searchsorted(nz_t, ts[i + 1], side="right")) - 1
        out.append(nz[k][1] if k >= 0 else "")
    return runs(out)


def render_arm(vm, scene, cfg, out):
    r = worst_rollout(cfg, scene)
    m = r.get("metrics") or {}
    verdict = verdict_of(r)
    rdir = Path(cfg) / "rollouts" / scene / r["rollout_id"]
    dat = asyncio.run(read_rollout(rdir / "rollout.asl"))
    ts = np.array(sorted(dat["frames"]), dtype=np.int64)
    n = len(ts) - 1
    ego = {t: next(f for f in dat["frames"][t] if f[0] == "EGO") for t in ts}
    path = np.array([[ego[t][1], ego[t][2]] for t in ts])

    df = pd.read_parquet(rdir / "metrics.parquet")
    off = df[df.name == "offroad"].sort_values("timestamps_us")
    off_t, off_v = off.timestamps_us.to_numpy(np.int64), off["values"].to_numpy() > 0
    col_df = df[df.name.isin(["collision_front", "collision_lateral", "collision_rear"])]
    col_hit = set(col_df[col_df["values"] > 0].timestamps_us.astype(np.int64).tolist())
    names = {"offroad": ["offroad"], "collision": ["collision_front", "collision_lateral"]}.get(verdict, [])
    hit = df[df.name.isin(names) & (df["values"] > 0)]
    t_hit = int(hit.timestamps_us.min()) if len(hit) else None
    event_frame = max(0, math.ceil((t_hit - ts[1]) / 1e5)) if t_hit else None

    allp = np.vstack([path, dat["gt"]])
    ext, ints, lanes, edges = map_geometry(vm, allp.min(0) - 60, allp.max(0) + 60)
    edge_ls = [shapely.LineString(e) for e in edges]
    length, width = dat["boxes"].get("EGO", (4.5, 2.0))
    col = FAIL if verdict != "pass" else PASS

    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-LAT, LAT)
    ax.set_ylim(-BEHIND, AHEAD)
    ax.set_facecolor(BG)
    ax.axis("off")
    c_road = ax.add_collection(PolyCollection([], facecolor=ROAD, edgecolor="none", zorder=1))
    c_hole = ax.add_collection(PolyCollection([], facecolor=BG, edgecolor="none", zorder=1.1))
    c_lane = ax.add_collection(LineCollection([], colors=LANE, linewidths=0.7, zorder=1.2))
    c_edge = ax.add_collection(LineCollection([], colors=EDGE, linewidths=1.4, zorder=1.3))
    gt_line, = ax.plot([], [], "--", color=GT, lw=1.5, zorder=2)
    trail, = ax.plot([], [], "-", color=TRAIL, lw=2.4, alpha=0.9, zorder=3, solid_capstyle="round")
    c_hit = ax.add_collection(LineCollection([], colors=FAIL, linewidths=4.0, zorder=3.5))
    plan, = ax.plot([], [], "-", color=PLAN, lw=2.2, zorder=4, solid_capstyle="round")
    c_traf = ax.add_collection(PolyCollection([], facecolor=TRAF, edgecolor="#cfd3d8", linewidths=0.6, zorder=5))
    c_traf_hit = ax.add_collection(PolyCollection([], facecolor=FAIL, edgecolor="#ffd9d6", linewidths=1.2, zorder=5.2))
    gt_car = ax.add_collection(PolyCollection([], facecolor="none", edgecolor=GT, linewidths=1.6,
                                              linestyles="--", zorder=5.5))
    c_ego = ax.add_collection(PolyCollection([], facecolor="#ffffff", edgecolor=col, linewidths=2.0, zorder=6))
    ax.plot([LAT - 13, LAT - 3], [-BEHIND + 2.2] * 2, color="#e6e8eb", lw=2, zorder=7)       # 10 m scale bar
    ax.text(LAT - 8, -BEHIND + 2.8, "10 m", color="#e6e8eb", fontsize=10, ha="center", va="bottom", zorder=7)

    out = Path(out)
    (out / "bev").mkdir(parents=True, exist_ok=True)
    plan_t = np.array([p[0] for p in dat["plans"]], dtype=np.int64)
    for i in range(n):
        t = ts[i + 1]
        _, ex, ey, eyaw = ego[t]
        f = partial(to_ego, ex=ex, ey=ey, yaw=eyaw)
        c_road.set_verts([f(p) for p in ext])
        c_hole.set_verts([f(p) for p in ints])
        c_lane.set_segments([f(p) for p in lanes])
        c_edge.set_segments([f(p) for p in edges])
        g = f(dat["gt"])
        gt_line.set_data(g[:, 0], g[:, 1])
        tr = f(path[: i + 2])
        trail.set_data(tr[:, 0], tr[:, 1])
        if len(plan_t):
            pp = f(dat["plans"][int(np.argmin(np.abs(plan_t - t)))][1])
            plan.set_data(pp[:, 0], pp[:, 1])
        others = [box(x, y, yw, *dat["boxes"].get(a, (4.5, 2.0))) for a, x, y, yw in dat["frames"][t] if a != "EGO"]
        ego_box = box(ex, ey, eyaw, length, width)
        ep = shapely.Polygon(ego_box)
        hits = [o for o in others if int(t) in col_hit and shapely.Polygon(o).distance(ep) < 0.3]
        c_traf.set_verts([f(o) for o in others])
        c_traf_hit.set_verts([f(o) for o in hits])
        gj = int(np.argmin(np.abs(dat["gt_t"] - t)))
        gt_car.set_verts([f(box(*dat["gt"][gj], dat["gt_yaw"][gj], length, width))])
        c_ego.set_verts([f(ego_box)])
        j = int(np.argmin(np.abs(off_t - t))) if len(off_t) else -1
        segs = []
        if j >= 0 and off_v[j]:          # frames the scorer flags off-road: the road edge the box touches
            near = ep.centroid.buffer(7.0)
            for ls in edge_ls:
                if ls.distance(ep) < 0.3:
                    cut = ls.intersection(near)
                    segs += [f(np.asarray(g.coords)) for g in getattr(cut, "geoms", [cut])
                             if g.geom_type == "LineString" and not g.is_empty]
        c_hit.set_segments(segs)
        fig.canvas.draw()
        Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).save(out / "bev" / f"{i:05d}.jpg", quality=92)
    plt.close(fig)
    meta = {"scene": scene, "config": str(cfg), "rollout_id": r["rollout_id"], "score": float(r["score"]),
            "verdict": verdict, "n_frames": n, "event_frame": event_frame,
            "event_t": round((t_hit - ts[0]) / 1e6, 2) if t_hit else None, "t_offset": round((ts[1] - ts[0]) / 1e6, 4),
            "dist_traveled_m": float(m.get("dist_traveled_m") or 0),
            "gt_dist_m": float(m.get("gt_dist_traveled_m") or 0), "coc": frame_cocs(dat, ts)}
    (out / "bev_meta.json").write_text(json.dumps(meta))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--arms", nargs="+", required=True, help="name=merged_run_dir, all on this scene")
    ap.add_argument("--out", type=Path, required=True, help="writes <out>/<name>/{bev,bev_meta.json}")
    args = ap.parse_args()
    vm = load_map(args.scene)
    for spec in args.arms:
        name, cfg = spec.split("=", 1)
        m = render_arm(vm, args.scene, cfg, args.out / name)
        print(f"{name}: {m['n_frames']} BEV frames, {m['verdict']}, event frame {m['event_frame']} (t={m['event_t']})",
              flush=True)


if __name__ == "__main__":
    main()
