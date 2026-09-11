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
    """-> DataFrame of that clip's tracks, or None when the labels are not local.

    Two sources, per-clip first: `fetch_obstacles.py` streams single clips into
    labels/obstacle_perclip/ because the evaluation sets span chunks that were never
    downloaded, while the calibration chunks are present as whole zips.
    """
    p = DATA / "labels" / "obstacle_perclip" / f"{clip_id}.parquet"
    if p.exists():
        return pd.read_parquet(p).pipe(lambda d: d[d.label_class.isin(CLASSES)])
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


def obstacles_over_time(obs, times_us):
    """All tracks interpolated onto `times_us` at once -> list per timestep.

    Built once per clip and shared by every path scored on it. The first version
    interpolated inside the per-step loop of every path, so one clip did 9 paths x 64
    steps = 576 groupbys over the same DataFrame and 20 clips took over ten minutes.
    Here the grouping happens once and np.interp is vectorised over the timesteps.

    A track only contributes at times inside its observed span -- extrapolating an
    autolabelled track past its last sighting invents obstacles.
    """
    t = np.asarray(times_us, float)
    out = [[] for _ in range(len(t))]
    for tid, g in obs.groupby("track_id", sort=False):
        g = g.sort_values("timestamp_us")
        ts = g.timestamp_us.to_numpy(float)
        inside = (t >= ts[0]) & (t <= ts[-1])
        if not inside.any():
            continue
        ti = t[inside]
        x = np.interp(ti, ts, g.center_x.to_numpy(float))
        y = np.interp(ti, ts, g.center_y.to_numpy(float))
        sx = np.interp(ti, ts, g.size_x.to_numpy(float))
        sy = np.interp(ti, ts, g.size_y.to_numpy(float))
        qz = np.interp(ti, ts, g.orientation_z.to_numpy(float))
        qw = np.interp(ti, ts, g.orientation_w.to_numpy(float))
        yaw = 2.0 * np.arctan2(qz, qw)
        cls = str(g.label_class.iloc[0])
        for j, i in enumerate(np.flatnonzero(inside)):
            out[i].append((x[j], y[j], yaw[j], sx[j], sy[j], cls, int(tid)))
    return out


def obstacles_at(obs, t_us):
    """Single-timestep convenience wrapper over `obstacles_over_time`."""
    return obstacles_over_time(obs, [t_us])[0]


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


def prepare_clip(obs, ego_future_xyz, ego_future_rot, t0_us, T, dt_us=100_000,
                 drop_tracks=()):
    """Per-clip work that every path scored on this clip shares.

    Interpolating the tracks and transforming them into the t0 frame depends only on the
    clip, not on the path being scored, so it is done once for all 9 (GT + k samples).
    """
    times = [t0_us + dt_us * i for i in range(T)]
    per_step = obstacles_over_time(obs, times)
    boxes = []
    for i, items in enumerate(per_step):
        R_i, p_i, yr = ego_future_rot[i], ego_future_xyz[i], yaw_of(ego_future_rot[i])
        step = []
        for (ox, oy, oyaw, sx, sy, cls, tid) in items:
            if tid in drop_tracks:
                continue
            c = R_i @ np.array([ox, oy, 0.0]) + p_i
            box = corners(np.array([c[0]]), np.array([c[1]]),
                          np.array([oyaw + yr]), sx, sy)[0]
            step.append((c[0], c[1], box, cls))
        boxes.append(step)
    return boxes


def score_path(xy, ego_future_xyz, ego_future_rot, obs, t0_us, ego_size,
               dt_us=100_000, yaw0=0.0, drop_tracks=(), prepared=None):
    """Collision statistics for one 2D path in the t0 rig frame.

    `xy` is (T, 2); the GT future pose arrays are (T, 3) / (T, 3, 3) and supply both the
    obstacle transform and, for the GT path itself, the exact heading. Pass `prepared`
    from `prepare_clip` when scoring several paths on one clip.
    """
    length, width, rear_to_c = ego_size
    T = len(xy)
    if prepared is None:
        prepared = prepare_clip(obs, ego_future_xyz, ego_future_rot, t0_us, T,
                                dt_us, drop_tracks)
    yaws = path_headings(xy, yaw0)
    # the box is centred ahead of the rear axle, which is where the trajectory is anchored
    cx = xy[:, 0] + rear_to_c * np.cos(yaws)
    cy = xy[:, 1] + rear_to_c * np.sin(yaws)
    ego = corners(cx, cy, yaws, length, width)

    hit_step, hit_class, min_dist = None, None, np.inf
    for i in range(T):
        for (bx, by, box, cls) in prepared[i]:
            d = float(np.hypot(bx - cx[i], by - cy[i]))
            min_dist = min(min_dist, d)
            # the separating-axis test is the expensive part; a box whose centre is
            # further than the two half-diagonals cannot overlap, so skip it
            if hit_step is None and d < 12.0 and obb_overlap(ego[i], box):
                hit_step, hit_class = i, cls
    return {"collide": hit_step is not None,
            "hit_step": hit_step, "hit_class": hit_class,
            "hit_time_s": None if hit_step is None else hit_step * dt_us / 1e6,
            "min_center_dist": None if min_dist == np.inf else min_dist}
