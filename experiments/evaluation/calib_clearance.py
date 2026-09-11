"""Per-clip clearance on the calibration set, so importance can be weighted by risk.

`I_traj` is the Taylor importance of the flow-matching loss against the GT actions -- an
imitation term. Alpamayo-R1's RL reward is not only that: it is
`r_traj = λ_L2‖x_pred − x_expert‖² + λ_coll·I[collision] + λ_jerk·J`, and our criterion has
been reading the L2 term alone. The collision term cannot enter as a loss here --
`collision_lib`'s test is a separating-axis boolean, with no gradient -- but it does not
have to. `score_path` also returns a continuous `min_center_dist`, so risk can enter as a
per-clip WEIGHT on a loss that is already differentiable:

    I_traj^safe(u) = Σ_c w(d_c)·|∂FM_c/∂g_u| / Σ_c w(d_c),    w(d) = exp(−d/τ)

That needs no new backward and no new importance run -- `importance_perclip.npz` already
stores |∂FM_c/∂g| per clip. This script only produces the d_c.

Clearance is measured on the GT future path, not on a prediction: the weight has to be a
property of the CLIP (how safety-critical this scene is), not of whatever model happens to
be scored. A prediction-dependent weight would make the criterion depend on the model it
is about to prune.

The obstacle labels exist for the calibration chunks only (93/100 clips here); the
evaluation sets have none locally. That asymmetry is why this can run today: importance is
measured on calibration clips, so the missing evaluation labels do not block it.

`collision_lib` is NOT vendored here -- it belongs to the open-loop collision track
(branch `worktree-openloop-collision`) and is still moving, so a copy in this branch would
go stale and could revert that work on merge. This script needs that branch to have landed
on main; the four entry points it uses (`load_obstacles`, `load_ego_size`,
`ego_self_tracks`, `score_path` with its `min_center_dist`) are unchanged at that branch's
head 0288e5b, confirmed by its author on 2026-09-09. Two of its later changes are worth
knowing when re-running this: `score_path` gained an optional `prepared=` that hoists the
per-clip interpolation out of the loop -- irrelevant here, which scores exactly one path
per clip -- and `load_obstacles` now prefers `labels/obstacle_perclip/<clip_id>.parquet`
over the chunk zip, so the labelled count may come out above the 93/100 recorded here.

Usage:
  .venv/bin/python experiments/evaluation/calib_clearance.py
  .venv/bin/python experiments/evaluation/calib_clearance.py --manifest calib_100
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import collision_lib as cl

PRE = Path("/mnt/nvme1n1/ad_vla/data/physicalai_av/pre_processed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="calib_100")
    ap.add_argument("--cache", default="calib", help="pre_processed namespace holding the npz")
    ap.add_argument("--out", default="calib_clearance")
    args = ap.parse_args()

    man = pd.read_parquet(REPO / "outputs" / "eval_sets" / f"{args.manifest}.parquet")
    rows = []
    for r in man.itertuples():
        t0 = int(getattr(r, "t0_us", 5_100_000))
        f = PRE / args.cache / "samples" / f"{r.clip_id}__t0_{t0}.npz"
        rec = {"clip_id": r.clip_id, "clearance_m": None, "gt_collide": None}
        if f.exists():
            obs = cl.load_obstacles(r.clip_id, int(r.chunk))
            size = cl.load_ego_size(r.clip_id, int(r.chunk))
            if obs is not None and size is not None:
                z = np.load(f, allow_pickle=True)
                fut_xyz = z["ego_future_xyz"][0, 0]
                fut_rot = z["ego_future_rot"][0, 0]
                drop = cl.ego_self_tracks(obs, t0, size)
                s = cl.score_path(fut_xyz[:, :2], fut_xyz, fut_rot, obs, t0, size,
                                  drop_tracks=drop)
                rec["clearance_m"] = s["min_center_dist"]
                rec["gt_collide"] = bool(s["collide"])
        rows.append(rec)

    have = [r for r in rows if r["clearance_m"] is not None]
    d = np.array([r["clearance_m"] for r in have])
    out = REPO / "outputs" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "clearance.json").write_text(json.dumps(
        {r["clip_id"]: r["clearance_m"] for r in rows}, indent=1))
    summary = (
        f"clearance on {args.manifest} (GT path vs labelled obstacles)\n"
        f"  labelled {len(have)}/{len(rows)} clips, GT collisions "
        f"{sum(bool(r['gt_collide']) for r in have)}/{len(have)}\n"
        f"  min {d.min():.2f}  p10 {np.percentile(d, 10):.2f}  median {np.median(d):.2f}"
        f"  p90 {np.percentile(d, 90):.2f}  max {d.max():.2f} m\n"
        + "".join(f"  within {t:2d} m: {100 * np.mean(d < t):.1f}%\n" for t in (3, 5, 10, 20)))
    (out / "summary.txt").write_text(summary)
    (out / "config.json").write_text(json.dumps(vars(args), indent=1))
    print(summary, end="")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
