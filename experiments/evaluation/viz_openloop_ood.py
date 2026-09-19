"""Per-clip qualitative view of the OOD-val open-loop rollouts.

The tables report minADE@6 / minFDE@6 over the 262 OOD-val clips. This renders one
clip at a time so a number can be looked at: the front camera frame nearest t0, the
rolled-out trajectories against ground truth in bird's-eye view, and the generated
CoC against the curated reference.

Only arms with a `<arm>_pred_oodval` dump carry `pred_xy_k`, written under
`run_baseline.py --save-pred`, so only those can be drawn. The plain `*_oodval` /
`*_ood` dirs hold the same metrics and CoC text but no waypoints. The teacher-forced
paths are never stored, only their ADE/FDE, so the rollout is the one path to draw.

`pred_xy_k` is (K=8, T=64, 2) in metres at mm precision, and ground truth is
`ego_future_xyz[..., :2]` from the clip cache: both are the same 6.4 s / 10 Hz horizon
in the same ego rig frame at t0, so no transform is needed between them.

That frame is forward-left-up: `x` forward, `y` left. Confirmed against the camera on
the hardest right turn in the set (8d9851c9, y = -51 m at the final step, road visibly
bending right), so the BEV panel inverts the horizontal axis and left in the plot is
left in the world. Camera index 1 of `camera_indices = [0, 1, 2, 6]` was likewise
confirmed by eye to be the forward view.

`t0_us` is not in the rows, so the clip id is joined against the manifest to build the
cache filename.

The evaluation protocol reduces the K=8 samples to the first 6 (best-of-6), so the
highlighted sample and the reported minADE are taken over `ade_rollout_k[:6]`.

Usage:
  python experiments/evaluation/viz_openloop_ood.py --clip 002b195e-3e0d-4a20-92dd-0e9920e74461
  python experiments/evaluation/viz_openloop_ood.py --n 8 --sort-by worst --arms baseline dual_u40_v2
"""

import argparse
import io
import json
import textwrap
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from PIL import Image

# outputs/ is an untracked symlink and is absent inside a git worktree, so the
# defaults are absolute; --out-root / --cache override them off-box.
OUT = Path("/mnt/nvme1n1/ad_vla/outputs/chan")
CACHE = Path("/mnt/nvme1n1/ad_vla/data/physicalai_av/pre_processed/ood/samples")

FRONT_CAMERA = 1   # camera_indices is [0, 1, 2, 6]; 1 is the front camera
K_EVAL = 6         # best-of-6, the reduction the published tables use

ARM_COLOR = {"baseline": "#1f77b4", "dual_u40_v2": "#d62728",
             "tyrK": "#2ca02c", "tyr_u40_r": "#9467bd"}


def load_arm(arm, out_root=OUT):
    """clip_id -> row, merging the 4 shards of `<arm>_pred_oodval`."""
    d = Path(out_root) / f"{arm}_pred_oodval"
    if not d.is_dir():
        have = sorted(p.name.replace("_pred_oodval", "")
                      for p in Path(out_root).glob("*_pred_oodval"))
        raise SystemExit(f"no prediction dump for arm {arm!r}: {d} missing.\n"
                         f"arms with a dump: {have}")
    rows = {}
    for p in sorted(d.glob("*.json")):
        if p.name == "config.json":
            continue
        for r in json.loads(p.read_text()):
            rows[r["clip_id"]] = r
    return rows


def load_clip(clip_id, t0_us, cache=CACHE):
    """Front camera frame nearest t0 plus the ego history and ground-truth future."""
    z = np.load(Path(cache) / f"{clip_id}__t0_{t0_us}.npz", allow_pickle=True)
    n_frame = int(z["n_frame"])
    c = z["camera_indices"].tolist().index(FRONT_CAMERA)
    # jpeg_bytes is camera-major, index c * n_frame + f (see sample_cache.load_cached)
    frame = Image.open(io.BytesIO(z["jpeg_bytes"][c * n_frame + (n_frame - 1)]))
    return {"frame": frame,
            "hist": z["ego_history_xyz"][0, 0][:, :2],    # (16, 2)
            "gt": z["ego_future_xyz"][0, 0][:, :2],       # (64, 2)
            "gt_coc": str(z["gt_coc"]) if "gt_coc" in z.files else ""}


def best_of_k(row, k=K_EVAL):
    """Index and value of the best sample under the best-of-k reduction."""
    ade = np.asarray(row["ade_rollout_k"][:k])
    fde = np.asarray(row["fde_rollout_k"][:k])
    i = int(np.argmin(ade))
    return i, float(ade[i]), float(fde.min())


def bev_limits(clip, rows, pad=2.0):
    """One axis box covering every arm, so the panels of a clip stay comparable.

    Letting each panel autoscale would draw an arm that overshoots by 20 m at the
    same visual size as one that tracks the reference.
    """
    xy = np.concatenate([clip["hist"], clip["gt"]]
                        + [np.asarray(r["pred_xy_k"]).reshape(-1, 2) for r in rows])
    # Straight clips span metres laterally against tens forward; hold a minimum
    # half-width so the equal-aspect box does not collapse to an unreadable sliver.
    lo, hi = xy[:, 1].min() - pad, xy[:, 1].max() + pad
    mid, half = (lo + hi) / 2, max((hi - lo) / 2, 4.0)
    return (mid + half, mid - half), (xy[:, 0].min() - pad, xy[:, 0].max() + pad)


def draw_bev(ax, clip, row, arm, xlim, ylim, k=K_EVAL):
    """Ego history, ground truth and the K rollout samples for one arm."""
    pred = np.asarray(row["pred_xy_k"])              # (8, 64, 2)
    i_best, ade, fde = best_of_k(row, k)
    color = ARM_COLOR.get(arm, "#d62728")
    first_other = 1 if i_best == 0 else 0

    ax.plot(clip["hist"][:, 1], clip["hist"][:, 0], color="0.55", lw=1.6,
            ls=":", label="ego history")
    for j in range(len(pred)):
        if j == i_best:
            continue
        ax.plot(pred[j, :, 1], pred[j, :, 0], color=color, lw=0.8,
                alpha=0.35 if j < k else 0.12,
                label=f"other samples (K={len(pred)})" if j == first_other else None)
    ax.plot(clip["gt"][:, 1], clip["gt"][:, 0], color="k", lw=2.4, label="ground truth")
    ax.plot(pred[i_best, :, 1], pred[i_best, :, 0], color=color, lw=2.0,
            label=f"best-of-{k} (k={i_best})")
    ax.plot(0, 0, marker="s", ms=5, color="k")

    # Equal aspect with adjustable="box" shrinks the axes to the data instead of
    # padding the lateral range out to the cell width, which on a straight clip
    # would stretch a +-2 m rollout across +-40 m of axis.
    ax.set_xlim(*xlim)                                        # reversed: +y is left
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("W")              # left-align the box, leaving room for the legend
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    # Left-aligned: the equal-aspect box is narrow, so a centred title would run
    # back across the camera panel.
    ax.set_title(f"{arm}   minADE@{k} {ade:.3f}   minFDE@{k} {fde:.3f}",
                 fontsize=9, loc="left")
    ax.set_xlabel("lateral y (m)   ← left", fontsize=8, loc="left")
    ax.set_ylabel("forward x (m)", fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=7)
    # The equal-aspect box is often a narrow sliver, so the legend goes beside it
    # rather than on top of the trajectories.
    ax.legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.03, 0.5),
              framealpha=0.85, borderaxespad=0)


def render(clip_id, arms, rows_by_arm, manifest, out_png, cache=CACHE, k=K_EVAL):
    """One figure: shared camera panel on the left, one BEV per arm on the right."""
    t0 = int(manifest.loc[clip_id, "t0_us"])
    clip = load_clip(clip_id, t0, cache)
    head = rows_by_arm[arms[0]][clip_id]

    # The CoC block is one line per arm plus the reference; size it in inches so the
    # bottom margin tracks the arm count instead of leaving a blank band.
    n = len(arms)
    line_in, panel_in = 0.30, 3.4
    text_in = line_in * (n + 1) + 0.35
    fig_h = max(4.2, panel_in * n) + text_in
    fig = plt.figure(figsize=(13.0, fig_h))
    gs = fig.add_gridspec(n, 2, width_ratios=[1.45, 1.0],
                          left=0.035, right=0.90, top=1 - 0.80 / fig_h,
                          bottom=text_in / fig_h, hspace=0.36, wspace=0.16)

    ax_cam = fig.add_subplot(gs[:, 0])
    ax_cam.imshow(clip["frame"])
    ax_cam.set_axis_off()
    ax_cam.set_anchor("N")          # keep the frame against the top of its span
    ax_cam.set_title(f"front camera @ t0   ({clip['frame'].size[0]}x{clip['frame'].size[1]})",
                     fontsize=9)

    xlim, ylim = bev_limits(clip, [rows_by_arm[a][clip_id] for a in arms])
    for i, arm in enumerate(arms):
        draw_bev(fig.add_subplot(gs[i, 1]), clip, rows_by_arm[arm][clip_id], arm,
                 xlim, ylim, k)

    fig.suptitle(f"{clip_id}    bucket={head['bucket']}    cluster={head['cluster']}    "
                 f"split={head.get('split', '?')}    t0_us={t0}",
                 fontsize=10, y=1 - 0.26 / fig_h)

    lines = [("GT CoC", clip["gt_coc"] or head.get("gt_coc", ""), "k")]
    for arm in arms:
        r = rows_by_arm[arm][clip_id]
        flag = "  [DEGENERATE]" if r.get("coc_degenerate") else ""
        lines.append((arm, (r["gen_coc"] or "<empty>") + flag, ARM_COLOR.get(arm, "#d62728")))

    y = (text_in - 0.42) / fig_h
    for label, text, color in lines:
        body = textwrap.shorten(text.replace("\n", " "), width=185, placeholder=" ...")
        fig.text(0.035, y, f"{label}:", fontsize=8.5, fontweight="bold", color=color)
        fig.text(0.145, y, body, fontsize=8.5, color=color)
        y -= line_in / fig_h

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


def check_ade(clip, row):
    """Recompute ADE from pred_xy_k against the stored ade_rollout_k. Max abs diff."""
    pred = np.asarray(row["pred_xy_k"])
    d = np.linalg.norm(pred - clip["gt"][None], axis=-1).mean(axis=-1)   # (8,)
    return float(np.abs(d - np.asarray(row["ade_rollout_k"])).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["baseline"])
    ap.add_argument("--clip", nargs="*", default=None, help="explicit clip_ids")
    ap.add_argument("--n", type=int, default=1, help="how many clips when --clip is absent")
    ap.add_argument("--sort-by", choices=["worst", "best", "random"], default="worst",
                    help="rank by the first arm's minADE@6")
    ap.add_argument("--bucket", default=None, help="restrict to one bucket")
    ap.add_argument("--k", type=int, default=K_EVAL)
    ap.add_argument("--exp-id", default="viz_oodval")
    ap.add_argument("--out-root", default=str(OUT))
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    manifest = pd.read_parquet(out_root / "eval_sets" / "ood_val.parquet").set_index("clip_id")
    rows_by_arm = {a: load_arm(a, out_root) for a in args.arms}

    common = set.intersection(*(set(r) for r in rows_by_arm.values())) & set(manifest.index)
    if args.clip:
        missing = [c for c in args.clip if c not in common]
        if missing:
            raise SystemExit(f"not in OOD-val for every requested arm: {missing}")
        clips = list(args.clip)
    else:
        pool = sorted(common)
        head = rows_by_arm[args.arms[0]]
        if args.bucket:
            pool = [c for c in pool if head[c]["bucket"] == args.bucket]
        if not pool:
            raise SystemExit(f"no clips left after bucket={args.bucket!r}")
        if args.sort_by == "random":
            clips = list(np.random.default_rng(args.seed).permutation(pool)[:args.n])
        else:
            pool.sort(key=lambda c: best_of_k(head[c], args.k)[1],
                      reverse=args.sort_by == "worst")
            clips = pool[:args.n]

    out_dir = out_root / args.exp_id
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({
        "arms": args.arms, "clips": list(clips), "n": len(clips),
        "sort_by": None if args.clip else args.sort_by,
        "bucket": args.bucket, "k_eval": args.k, "seed": args.seed,
        "manifest": str(out_root / "eval_sets" / "ood_val.parquet"),
        "cache": args.cache, "n_oodval": len(manifest),
        "source": "experiments/evaluation/viz_openloop_ood.py",
    }, indent=2))

    t = time.time()
    lines = [f"OOD-val open-loop visualization -- arms {' '.join(args.arms)}, best-of-{args.k}",
             f"{len(clips)} clip(s) of {len(manifest)} in the OOD-val manifest", ""]
    for c in clips:
        png = render(c, args.arms, rows_by_arm, manifest, out_dir / "plots" / f"{c}.png",
                     args.cache, args.k)
        clip = load_clip(c, int(manifest.loc[c, "t0_us"]), args.cache)
        per = []
        for a in args.arms:
            _, ade, fde = best_of_k(rows_by_arm[a][c], args.k)
            per.append(f"{a} ADE {ade:.3f} FDE {fde:.3f} "
                       f"(recheck {check_ade(clip, rows_by_arm[a][c]):.1e})")
        line = f"{c}  {rows_by_arm[args.arms[0]][c]['bucket']:<11} " + " | ".join(per)
        lines.append(line)
        print(line, flush=True)
        print(f"  -> {png}", flush=True)

    lines += ["", f"plots in {out_dir / 'plots'}", f"{time.time() - t:.1f}s"]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"\n{len(clips)} figure(s) -> {out_dir / 'plots'}", flush=True)


if __name__ == "__main__":
    main()
