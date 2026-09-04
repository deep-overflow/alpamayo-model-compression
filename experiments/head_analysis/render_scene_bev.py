"""Top-down animation of an alpasim scene, drawn from the usdz GT data alone.

The usdz is a *neural* reconstruction (`checkpoint.ckpt` + `volume.nurec`), so it stores no
camera frames -- rendering imagery needs the NuRec renderer and a GPU. What it does store is
exact geometry, and that is enough to show what makes a scene hard: `rig_trajectories.usda`
gives the ego GT pose at 10 Hz over the 20 s clip and `sequence_tracks.json` gives every
annotated agent. Both are read by `extract_scene_feats`, so this costs no GPU and no network.

Defaults render in real time (`stride=1`, `fps=None` -> the poses' own 10 Hz), which is what
lets the result sit next to the real camera video from `fetch_scene_camera.py`; see
`make_scene_reel.py`.

Two framing details worth keeping, both found by looking at a rendered frame:
  * the view padding scales with the route (a fixed margin leaves a 69 m U-turn adrift in an
    empty frame), and
  * the axes are stretched to fill the figure -- matplotlib's default subplot margins eat ~23%
    of it, which is the larger reason a route looks smaller than it is.

Usage:
  python render_scene_bev.py --scenes outputs/scene_difficulty/hard100_suite.csv \
      --out outputs/scene_difficulty/media/full --dpi 200 --limit 10
  python render_scene_bev.py --scene-id clipgt-<uuid> --out /tmp/one
"""

import argparse
import csv
import json
import zipfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import extract_scene_feats as esf
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[2]

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
EGO, CAR, PED = "#D97757", "#2a78d6", "#008300"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9,
})


def _agents_on_ego_clock(tracks, t, t_us0):
    """Each agent resampled onto the ego timestamps; NaN outside its own observed span."""
    out = []
    for cls, ts, p, _dim in tracks:
        if len(p) < 2:
            continue
        tt = (ts - t_us0) / 1e6
        if tt[-1] < t[0] or tt[0] > t[-1]:
            continue
        ax_ = np.interp(t, tt, p[:, 0], left=np.nan, right=np.nan)
        ay_ = np.interp(t, tt, p[:, 1], left=np.nan, right=np.nan)
        gap = (t < tt[0]) | (t > tt[-1])
        ax_[gap] = np.nan
        ay_[gap] = np.nan
        if np.isfinite(ax_).sum() >= 2:
            out.append((cls, ax_, ay_))
    return out


def render(scene_id, uuid, out_path, title_extra="", stride=1, fps=None, dpi=100):
    """Write an mp4. fps=None uses the poses' own rate, i.e. real time."""
    z = zipfile.ZipFile(esf.USDZ / f"{uuid}.usdz")
    t, xy, hdg, t_us0, _dur = esf.ego_track(z)
    agents = _agents_on_ego_clock(esf.agent_tracks(z), t, t_us0)

    v = np.r_[0, np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.maximum(np.diff(t), 1e-6)]
    idx = list(range(0, len(t), stride))
    if fps is None:
        fps = (len(idx) - 1) / (t[-1] - t[0]) if t[-1] > t[0] else 10.0

    xs, ys = xy[:, 0], xy[:, 1]
    extent = max(np.ptp(xs), np.ptp(ys))
    half = extent / 2 + max(10.0, 0.15 * extent)
    cx, cy = (xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    ax.plot(xs, ys, color=MUTED, lw=0.8, ls=":", zorder=1)      # whole GT route
    trail, = ax.plot([], [], color=EGO, lw=2.2, zorder=3)        # driven so far
    car_sc = ax.scatter([], [], s=26, color=CAR, zorder=4)
    ped_sc = ax.scatter([], [], s=30, color=PED, marker="^", zorder=4)
    ego_patch = Rectangle((0, 0), 4.6, 2.0, color=EGO, zorder=5)
    ax.add_patch(ego_patch)
    ax.text(0.5, 0.975, f"{scene_id.replace('clipgt-', '')[:8]}   {title_extra}",
            transform=ax.transAxes, ha="center", va="top", fontsize=11, color=INK)
    hud = ax.text(0.03, 0.045, "", transform=ax.transAxes, va="bottom", fontsize=11,
                  family="monospace", color=INK)

    def frame(k):
        i = idx[k]
        trail.set_data(xs[:i + 1], ys[:i + 1])
        cars = np.array([[a[1][i], a[2][i]] for a in agents if a[0] != "person"])
        peds = np.array([[a[1][i], a[2][i]] for a in agents if a[0] == "person"])
        cars = cars[np.isfinite(cars).all(1)] if len(cars) else np.empty((0, 2))
        peds = peds[np.isfinite(peds).all(1)] if len(peds) else np.empty((0, 2))
        car_sc.set_offsets(cars)
        ped_sc.set_offsets(peds)
        c, s = np.cos(hdg[i]), np.sin(hdg[i])
        ego_patch.set_xy((xs[i] - 2.3 * c + 1.0 * s, ys[i] - 2.3 * s - 1.0 * c))
        ego_patch.angle = np.degrees(hdg[i])
        hud.set_text(f"t={t[i]:5.1f}s   {v[i] * 3.6:5.1f} km/h   cars {len(cars)}")
        return trail, car_sc, ped_sc, ego_patch, hud

    an = animation.FuncAnimation(fig, frame, frames=len(idx), interval=1000 / fps, blit=False)
    an.save(str(out_path), dpi=dpi,
            writer=animation.FFMpegWriter(fps=fps, bitrate=900,
                                          extra_args=["-pix_fmt", "yuv420p"]))
    plt.close(fig)
    return Path(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=Path,
                    help="suite CSV with scene_id,uuid (e.g. from make_hard_suite.py)")
    ap.add_argument("--scene-id", help="render a single scene instead")
    ap.add_argument("--suite", default="public_2601", help="used to resolve --scene-id's uuid")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=100, help="200 gives 1280x1280")
    ap.add_argument("--stride", type=int, default=1, help=">1 speeds the animation up")
    ap.add_argument("--fps", type=float, default=None, help="default: real time")
    ap.add_argument("--feats", type=Path,
                    default=REPO / "outputs/scene_difficulty/scene_feats_public2601.json",
                    help="adds speed/turn/length to each title when present")
    args = ap.parse_args()

    if args.scenes:
        rows = [(r["scene_id"], r["uuid"]) for r in csv.DictReader(args.scenes.open())]
    elif args.scene_id:
        rows = [(args.scene_id, esf.suite_scenes(args.suite)[args.scene_id])]
    else:
        raise SystemExit("--scenes 또는 --scene-id 중 하나가 필요합니다")
    if args.limit:
        rows = rows[:args.limit]

    feats = {}
    if args.feats.exists():
        feats = {r["scene_id"]: r for r in json.loads(args.feats.read_text())}

    out = args.out if args.out.is_absolute() else REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    for i, (sid, uuid) in enumerate(rows, 1):
        f = feats.get(sid)
        extra = ("" if f is None else
                 f"{f['v_mean'] * 3.6:.0f} km/h  turn {f['yaw_total_deg']:.0f}deg  "
                 f"{f['gt_path_m']:.0f} m")
        p = out / f"bev_{sid.replace('clipgt-', '')[:8]}.mp4"
        render(sid, uuid, p, extra, stride=args.stride, fps=args.fps, dpi=args.dpi)
        print(f"[{i}/{len(rows)}] {p.name}  {p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
