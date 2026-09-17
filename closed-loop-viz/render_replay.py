"""Render a stored alpasim rollout as a top-down mp4 -- two arms on one scene, side by side.

Why this exists: every closed-loop number in this repo is a scene-averaged score, and the
gates that decide it are binary. A scene where one arm scores 1.000 and another 0.000 has
a story that no aggregate can show. The logs already hold everything needed to replay it.

What is drawn, and where each piece comes from in `rollout.asl`:

  ego box + trail     `actor_poses` entry whose actor_id == "EGO" (physics truth, and it
                      carries the same box geometry as every other actor, so ego and
                      traffic are drawn by one code path)
  other traffic       the remaining `actor_poses` entries, oriented by their quaternion
  footprints          `rollout_metadata.actor_definitions.actor_aabb`, joined on actor_id
                      (string on BOTH sides -- pose ids print bare but are strings)
  the plan            `driver_return.trajectory.poses`, the 6.4 s / 65-pose plan the model
                      held at that step
  GT reference        `rollout_metadata.ego_rig_recorded_ground_truth_trajectory`
  reasoning           `driver_return.debug_info.unstructured_debug_info`, pickled, key
                      `reasoning_text`

There are NO camera frames: the launchers run with ALPASIM_ASL_SKIP_IMAGES=1, whose
LogWriter drops driver camera frames, and the log holds `batch_render_request` entries
with no matching return. So this is a top-down replay by necessity, not by preference.

Two frame facts the drawing depends on, both checked rather than assumed:
  * ego, actors, plans and GT share one frame. `transform_ego_coords_rig_to_aabb` is a
    pure translation (quat w=1, x=1.4485 m) between the rig origin and the box centre, not
    a change of frame -- so no transform is applied here and the EGO actor entry is used
    for the ego pose.
  * sim timestamps are NOT comparable across rollouts (they differ by ~10^12 us between
    runs of the same scene), so every rollout is replayed against its own t0.

Usage:
  .venv/bin/python closed-loop-viz/render_replay.py \
      --scene clipgt-2431387d-... \
      --arm "dual=slim_dual_u40_v2" --arm "dual+h4=slim_dual_u40_qcut4_v2" \
      --out closed-loop-viz/video/scene.mp4
"""
import argparse
import asyncio
import json
import math
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # must precede the pyplot import; this is why the block is split

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Rectangle

RUNS = Path("/home/cvlab21/project/chan/alpasim-runs")

# Same ink/ground as the repo's plots so a frame grab sits beside the report figures.
BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
ACC, BAD, GOOD = "#D97757", "#b0402a", "#087f5b"
TRAFFIC = "#8A8F98"


def yaw_from_quat(q):
    """Heading about z. The poses are near-planar, so the full matrix is not needed."""
    return np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


async def read_rollout(asl):
    from alpasim_utils.logs import async_read_pb_log

    boxes, gt = {}, []
    frames = {}        # ts -> [(actor_id, x, y, yaw)]
    plans = []         # (ts, xy array)
    cocs = []          # (ts, text)
    async for e in async_read_pb_log(str(asl)):
        if e.HasField("rollout_metadata"):
            md = e.rollout_metadata
            boxes = {a.actor_id: (a.aabb.size_x, a.aabb.size_y)
                     for a in md.actor_definitions.actor_aabb}
            gt = np.array([[p.pose.vec.x, p.pose.vec.y]
                           for p in md.ego_rig_recorded_ground_truth_trajectory.poses])
        elif e.HasField("actor_poses"):
            ap = e.actor_poses
            frames[int(ap.timestamp_us)] = [
                (a.actor_id, a.actor_pose.vec.x, a.actor_pose.vec.y,
                 yaw_from_quat(a.actor_pose.quat)) for a in ap.actor_poses]
        elif e.HasField("driver_return"):
            dr = e.driver_return
            ps = dr.trajectory.poses
            if ps:
                plans.append((int(ps[0].timestamp_us),
                              np.array([[p.pose.vec.x, p.pose.vec.y] for p in ps])))
                blob = dr.debug_info.unstructured_debug_info
                txt = ""
                if blob:
                    try:
                        t = pickle.loads(blob).get("reasoning_text")
                        txt = str(t) if t else ""
                    except Exception:
                        txt = ""
                cocs.append((int(ps[0].timestamp_us), txt))
    return {"boxes": boxes, "gt": gt, "frames": frames, "plans": plans, "cocs": cocs}


def run_dir(config):
    """Resolve an arm spec to its merged-run directory.

    A bare name is one of our own runs and follows the `m2601_merged_<config>` convention.
    An absolute path is taken as the run directory itself, which is how an arm that lives
    in someone else's runs_root gets in -- soowon's LLM-Pruner baseline is at
    /mnt/nvme1n1/ad_vla/outputs/soowon/alpasim-analysis/runs_root/lp_r50 and covers the
    same 150 scenes, but does not sit under our prefix.
    """
    p = Path(config)
    return p if p.is_absolute() else RUNS / f"m2601_merged_{config}"


def pick_rollout(config, scene, want_worst):
    """Choose which of the scene's two rollouts to show, and return its score and gates."""
    d = json.loads((run_dir(config) / "aggregate/results-summary.json").read_text())
    rs = [r for r in d["rollouts"] if r["clipgt_id"] == scene]
    if not rs:
        raise SystemExit(f"{config}: {scene} 없음")
    rs.sort(key=lambda r: float(r["score"]))
    r = rs[0] if want_worst else rs[-1]
    m = r.get("metrics") or {}
    return (r["rollout_id"], float(r["score"]),
            {k: float(m.get(k) or 0) for k in ("collision_at_fault", "offroad")},
            float(m.get("dist_traveled_m") or 0), float(m.get("gt_dist_traveled_m") or 0))


def clean_coc(t):
    s = t.strip()
    if s.startswith("['") and s.endswith("']"):
        s = s[2:-2]
    return s.strip()


def wrap(s, width=62, lines=2):
    out, cur = [], ""
    for w in s.split():
        if len(cur) + len(w) + 1 > width:
            out.append(cur)
            cur = w
            if len(out) == lines:
                break
        else:
            cur = f"{cur} {w}".strip()
    if len(out) < lines and cur:
        out.append(cur)
    txt = "\n".join(out)
    if len(" ".join(out)) < len(s):
        txt += " …"
    return txt


def draw_box(ax, x, y, yaw, w, h, **kw):
    r = Rectangle((-w / 2, -h / 2), w, h, **kw)
    tr = (matplotlib.transforms.Affine2D().rotate(yaw).translate(x, y) + ax.transData)
    r.set_transform(tr)
    ax.add_patch(r)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--arm", action="append", required=True,
                    help="label=config, repeatable. `config` is either one of our run "
                         "names (resolved as m2601_merged_<config>) or an absolute path "
                         "to a merged run directory elsewhere.")
    ap.add_argument("--worst", action="store_true", default=True,
                    help="show each arm's worse rollout (default; the failure is the point)")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    arms = []
    for spec in args.arm:
        label, cfg = spec.split("=", 1)
        rid, score, gates, dist, gtdist = pick_rollout(cfg, args.scene, args.worst)
        asl = run_dir(cfg) / "rollouts" / args.scene / rid / "rollout.asl"
        data = asyncio.run(read_rollout(asl))
        data.update(label=label, config=cfg, score=score, gates=gates,
                    dist=dist, gtdist=gtdist, rid=rid)
        ts = sorted(data["frames"])
        data["ts"] = ts
        data["t0"] = ts[0]
        arms.append(data)
        print(f"{label:8s} rollout {rid[:8]} score {score:.3f} "
              f"offroad {gates['offroad']:.0f} frames {len(ts)} "
              f"plans {len(data['plans'])} CoC {sum(1 for _, c in data['cocs'] if c)}")

    # the two arms must be replaying the same scene geometry, or the panels are not
    # comparable; the GT path is the one thing that cannot differ between them
    g0 = arms[0]["gt"]
    for a in arms[1:]:
        n = min(len(g0), len(a["gt"]))
        d = float(np.abs(g0[:n] - a["gt"][:n]).max())
        print(f"GT 일치 검사 {arms[0]['label']} vs {a['label']}: 최대 차이 {d:.4f} m")
        assert d < 0.01, "arm 간 GT 경로가 다릅니다 -- 같은 씬이 아닙니다"

    # one shared extent so the panels are at the same scale
    pts = [a["gt"] for a in arms]
    for a in arms:
        ego = np.array([[x, y] for ts in a["ts"]
                        for i, x, y, _ in [next((f for f in a["frames"][ts] if f[0] == "EGO"),
                                                (None, np.nan, np.nan, 0))]])
        pts.append(ego[~np.isnan(ego[:, 0])])
    allp = np.vstack(pts)
    pad = 7.0
    xlim = (allp[:, 0].min() - pad, allp[:, 0].max() + pad)
    ylim = (allp[:, 1].min() - pad, allp[:, 1].max() + pad)
    span = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    cx, cy = np.mean(xlim), np.mean(ylim)
    xlim = (cx - span / 2, cx + span / 2)
    ylim = (cy - span / 2, cy + span / 2)

    n_frames = min(len(a["ts"]) for a in arms)
    # Panels are square: one shared extent keeps every arm at the same scale, and the
    # figure is sized to that square plus room for title and caption (a taller figure just
    # letterboxes the data box in white). Past three arms a single row is too wide to
    # read, so the panels wrap into a grid.
    ncol = len(arms) if len(arms) <= 3 else math.ceil(len(arms) / 2)
    nrow = math.ceil(len(arms) / ncol)
    fig, axgrid = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 6.1 * nrow + 0.6),
                               facecolor=BG, squeeze=False)
    axes = list(axgrid.ravel()[:len(arms)])
    for extra in axgrid.ravel()[len(arms):]:
        extra.axis("off")
    fig.suptitle(f"{args.scene}      same scene, worse rollout of each arm",
                 color=INK, fontsize=11, y=0.985)

    state = []
    for ax, a in zip(axes, arms):
        ax.set_facecolor(BG)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(MUTED)
        ax.plot(a["gt"][:, 0], a["gt"][:, 1], "--", color=MUTED, lw=1.2,
                label="GT reference path", zorder=1)
        col = BAD if a["gates"]["offroad"] or a["gates"]["collision_at_fault"] else GOOD
        a["col"] = col
        trail, = ax.plot([], [], "-", color=col, lw=2.2, zorder=4, label="driven path")
        plan, = ax.plot([], [], "-", color=ACC, lw=1.6, alpha=0.95, zorder=5,
                        label="plan held now (6.4 s)")
        verdict = ("OFFROAD" if a["gates"]["offroad"] else
                   ("COLLISION" if a["gates"]["collision_at_fault"] else "pass"))
        ax.set_title(f"{a['label']}   score {a['score']:.3f}   {verdict}\n"
                     f"{a['config']}   ·   drove {a['dist']:.0f} m of {a['gtdist']:.0f} m",
                     color=col if verdict != "pass" else INK, fontsize=10, pad=10)
        cap = ax.text(0.5, -0.035, "", transform=ax.transAxes, ha="center", va="top",
                      color=INK, fontsize=7.4, wrap=True)
        clock = ax.text(0.02, 0.975, "", transform=ax.transAxes, ha="left", va="top",
                        color=MUTED, fontsize=9, family="monospace")
        if not state:          # legend on the first panel only
            ax.legend(loc="lower right", fontsize=7.5, framealpha=0.85,
                      facecolor=BG, edgecolor=MUTED)
        state.append({"ax": ax, "trail": trail, "plan": plan, "cap": cap,
                      "clock": clock, "patches": []})

    def update(k):
        arts = []
        for a, s in zip(arms, state):
            ts = a["ts"][k]
            for p in s["patches"]:
                p.remove()
            s["patches"] = []

            ego_xy = []
            for t in a["ts"][:k + 1]:
                e = next((f for f in a["frames"][t] if f[0] == "EGO"), None)
                if e:
                    ego_xy.append((e[1], e[2]))
            if ego_xy:
                arr = np.array(ego_xy)
                s["trail"].set_data(arr[:, 0], arr[:, 1])

            for aid, x, y, yaw in a["frames"][ts]:
                w, h = a["boxes"].get(aid, (4.5, 2.0))
                if aid == "EGO":
                    s["patches"].append(draw_box(s["ax"], x, y, yaw, w, h,
                                                 facecolor=a["col"], edgecolor=INK,
                                                 lw=1.0, zorder=6))
                else:
                    s["patches"].append(draw_box(s["ax"], x, y, yaw, w, h,
                                                 facecolor=TRAFFIC, edgecolor=MUTED,
                                                 lw=0.5, alpha=0.75, zorder=3))

            # the plan whose first pose is nearest this frame's time
            if a["plans"]:
                j = int(np.argmin([abs(p[0] - ts) for p in a["plans"]]))
                xy = a["plans"][j][1]
                s["plan"].set_data(xy[:, 0], xy[:, 1])
                txt = clean_coc(a["cocs"][j][1]) if j < len(a["cocs"]) else ""
                s["cap"].set_text(wrap(txt) if txt else "(empty reasoning output)")
            s["clock"].set_text(f"t = {(ts - a['t0']) / 1e6:5.1f} s")
            arts += [s["trail"], s["plan"], s["cap"], s["clock"], *s["patches"]]
        return arts

    args.out.parent.mkdir(parents=True, exist_ok=True)
    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    fig.subplots_adjust(left=0.02, right=0.98,
                        top=0.885 if nrow == 1 else 0.925,
                        bottom=0.125 if nrow == 1 else 0.055,
                        wspace=0.05, hspace=0.26)
    # libx264 with yuv420p needs EVEN pixel dimensions, and the figure size that follows
    # from the panel grid does not always give them (a 2-arm figure came out 1080x669 and
    # ffmpeg exited 1). Rounding down to even in ffmpeg is robust to any grid.
    anim.save(str(args.out), writer=animation.FFMpegWriter(
        fps=args.fps, bitrate=2400, codec="libx264",
        extra_args=["-pix_fmt", "yuv420p",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"]))
    print(f"\n-> {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB, "
          f"{n_frames} frames @ {args.fps} fps = {n_frames / args.fps:.0f} s)")


if __name__ == "__main__":
    main()
