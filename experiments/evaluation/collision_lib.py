"""Open-loop collision proxy: does a trajectory's footprint overlap a labelled obstacle?

The dataset has no map, lane or drivable-area labels, so offroad cannot be scored
open-loop at all. It does have `labels/obstacle.offline` -- autolabelled 3D tracks -- so
collisions can be, as a proxy.

Frames are the whole problem. An obstacle row is expressed in the rig frame AT ITS OWN
timestamp (`reference_frame_timestamp_us == timestamp_us`), while `ego_future_xyz` /
`ego_future_rot` are future poses in the rig frame AT t0. The transform between the two is
therefore the GT future pose itself:

    c_t0 = R_i @ c_rig(t_i) + p_i        R_i = ego_future_rot[i], p_i = ego_future_xyz[i]

so an obstacle observed at step i lands in the t0 frame where a predicted path also lives,
and the two footprints can be compared there.

Everything is done in 2D (bird's eye). The z axis is dropped: the ego and every labelled
class sit on the same ground plane, and a 3D check would only add height mismatches that
the autolabels are not accurate enough to support.

This is NOT closed-loop collision. Other agents replay their logged motion and do not
react to the trajectory being scored, so a situation they would have avoided still counts.
The absolute rate is an upper bound, not an estimate; arm-vs-arm differences share the
same labels and are the reading this is built for.
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/mnt/nvme1n1/ad_vla/data/physicalai_av")
# classes that are physical obstacles for a ground vehicle. `protruding_object` is kept --
# it is what the label set calls things like open doors and load overhang.
CLASSES = ("automobile", "person", "heavy_truck", "bus", "rider", "trailer",
           "other_vehicle", "stroller", "animal", "protruding_object")
MIN_SPEED = 0.5  # m/s below which a path tangent is too noisy to use as a heading


def load_obstacles(clip_id, chunk):
    """-> DataFrame of that clip's tracks, or None when the chunk was never downloaded."""
    z = DATA / "labels" / "obstacle.offline" / f"obstacle.offline.chunk_{chunk:04d}.zip"
    if not z.exists():
        return None
    member = f"{clip_id}.obstacle.offline.parquet"
    with zipfile.ZipFile(z) as f:
        if member not in f.namelist():
            return None
        d = pd.read_parquet(io.BytesIO(f.read(member)))
    return d[d.label_class.isin(CLASSES)]


def load_ego_size(clip_id, chunk):
    """-> (length, width, rear_axle_to_bbox_center) for this clip."""
    p = DATA / "calibration" / "vehicle_dimensions" / f"vehicle_dimensions.chunk_{chunk:04d}.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    if clip_id not in d.index:
        return None
    r = d.loc[clip_id]
    return float(r.length), float(r.width), float(r.rear_axle_to_bbox_center)


def yaw_of(R):
    """Heading from a rotation matrix stack (..., 3, 3) -> (...,) radians."""
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


def path_headings(xy, fallback):
    """Heading per waypoint from the path tangent.

    The model emits positions, not poses, so the ego box orientation has to be derived.
    Below MIN_SPEED the tangent is dominated by noise, so the previous usable heading is
    carried forward (and `fallback`, the pose at t0, seeds the first step).
    """
    d = np.diff(xy, axis=0, prepend=xy[:1])
    d[0] = xy[1] - xy[0] if len(xy) > 1 else 0.0
    speed = np.linalg.norm(d, axis=1) * 10.0  # 10 Hz
    h = np.arctan2(d[:, 1], d[:, 0])
    out = np.empty(len(xy))
    cur = fallback
    for i in range(len(xy)):
        if speed[i] >= MIN_SPEED:
            cur = h[i]
        out[i] = cur
    return out


def corners(cx, cy, yaw, length, width):
    """(N,) boxes -> (N, 4, 2) corners, counter-clockwise."""
    c, s = np.cos(yaw), np.sin(yaw)
    hl, hw = length / 2.0, width / 2.0
    local = np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]])
    R = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)  # (N, 2, 2)
    return np.einsum("nij,kj->nki", R, local) + np.stack([cx, cy], -1)[:, None, :]


def obb_overlap(a, b):
    """Separating-axis test between two convex quads, (4,2) each. True = overlapping.

    Exact for rectangles and cheap enough at this scale -- the alternative, an inflated
    circle test, would count near-misses as hits and the metric is about contact.
    """
    for poly in (a, b):
        for i in range(4):
            edge = poly[(i + 1) % 4] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            n = np.hypot(*axis)
            if n < 1e-12:
                continue
            axis = axis / n
            pa, pb = a @ axis, b @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def obstacles_at(obs, t_us):
    """Each track's box interpolated to `t_us`, in the rig frame of that instant.

    A track is only used when `t_us` falls inside its observed span -- extrapolating an
    autolabelled track past its last sighting invents obstacles.
    """
    out = []
    for tid, g in obs.groupby("track_id"):
        g = g.sort_values("timestamp_us")
        ts = g.timestamp_us.to_numpy(float)
        if t_us < ts[0] or t_us > ts[-1]:
            continue
        x = np.interp(t_us, ts, g.center_x.to_numpy(float))
        y = np.interp(t_us, ts, g.center_y.to_numpy(float))
        sx = float(np.interp(t_us, ts, g.size_x.to_numpy(float)))
        sy = float(np.interp(t_us, ts, g.size_y.to_numpy(float)))
        qz = np.interp(t_us, ts, g.orientation_z.to_numpy(float))
        qw = np.interp(t_us, ts, g.orientation_w.to_numpy(float))
        yaw = 2.0 * np.arctan2(qz, qw)
        out.append((x, y, yaw, sx, sy, str(g.label_class.iloc[0]), int(tid)))
    return out


def ego_self_tracks(obs, t0_us, ego_size):
    """Track ids the autolabeller put on the ego vehicle itself.

    Found by the GT gate: one calib clip had an `automobile` at (1.00, 0.01) sized
    4.62 x 1.97 against an ego of 4.69 x 2.00 whose box centre sits at (1.31, 0) -- the
    ego, labelled as an obstacle. Nothing can be in contact with you at the instant you
    start, so any track overlapping the ego footprint at t0 is dropped for the whole clip.
    Left in, it fires at step 0 for every trajectory whatever the model predicted, which
    is a constant false positive that dilutes the signal rather than biasing one arm.
    """
    length, width, rear_to_c = ego_size
    ego = corners(np.array([rear_to_c]), np.array([0.0]), np.array([0.0]),
                  length, width)[0]
    bad = set()
    for (ox, oy, oyaw, sx, sy, _cls, tid) in obstacles_at(obs, t0_us):
        box = corners(np.array([ox]), np.array([oy]), np.array([oyaw]), sx, sy)[0]
        if obb_overlap(ego, box):
            bad.add(tid)
    return bad


def score_path(xy, ego_future_xyz, ego_future_rot, obs, t0_us, ego_size,
               dt_us=100_000, yaw0=0.0, drop_tracks=()):
    """Collision statistics for one 2D path in the t0 rig frame.

    `xy` is (T, 2); the GT future pose arrays are (T, 3) / (T, 3, 3) and supply both the
    obstacle transform and, for the GT path itself, the exact heading.
    """
    length, width, rear_to_c = ego_size
    T = len(xy)
    yaws = path_headings(xy, yaw0)
    # the box is centred ahead of the rear axle, which is where the trajectory is anchored
    cx = xy[:, 0] + rear_to_c * np.cos(yaws)
    cy = xy[:, 1] + rear_to_c * np.sin(yaws)
    ego = corners(cx, cy, yaws, length, width)

    hit_step, hit_class, min_dist = None, None, np.inf
    for i in range(T):
        t_us = t0_us + dt_us * i
        R_i, p_i = ego_future_rot[i], ego_future_xyz[i]
        for (ox, oy, oyaw, sx, sy, cls, tid) in obstacles_at(obs, t_us):
            if tid in drop_tracks:
                continue
            # rig-at-t_i -> rig-at-t0
            c = R_i @ np.array([ox, oy, 0.0]) + p_i
            oyaw_t0 = oyaw + yaw_of(R_i)
            box = corners(np.array([c[0]]), np.array([c[1]]),
                          np.array([oyaw_t0]), sx, sy)[0]
            d = float(np.hypot(c[0] - cx[i], c[1] - cy[i]))
            min_dist = min(min_dist, d)
            if hit_step is None and obb_overlap(ego[i], box):
                hit_step, hit_class = i, cls
    return {"collide": hit_step is not None,
            "hit_step": hit_step, "hit_class": hit_class,
            "hit_time_s": None if hit_step is None else hit_step * dt_us / 1e6,
            "min_center_dist": None if min_dist == np.inf else min_dist}
