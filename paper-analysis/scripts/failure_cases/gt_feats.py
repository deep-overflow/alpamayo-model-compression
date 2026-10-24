"""Per-clip GT kinematics for test_500 and ood from the sample cache: speed at t0 (last history
step), mean future speed, future path length, net signed heading change, max |curvature| proxy."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/cvlab21/project/chan/alpamayo-model-compression/experiments/evaluation")
import sample_cache as sc
REPO = "/home/cvlab21/project/chan/alpamayo-model-compression"
rows = []
for sname, manifest, cache in (("test", "test_500", "test"), ("ood", "ood", "ood")):
    for cid, t0 in sc.calib_samples(REPO, manifest):
        z = np.load(sc.path_for(cache, cid, t0), allow_pickle=True)
        fut = np.asarray(z["ego_future_xyz"]).reshape(-1, 3)[:, :2]
        vel = np.diff(fut, axis=0) / 0.1; sp = np.linalg.norm(vel, axis=1)
        hist = np.asarray(z["ego_history_xyz"]).reshape(-1, 3)[:, :2] if "ego_history_xyz" in z else None
        v0 = float(np.linalg.norm(hist[-1] - hist[-2]) / 0.1) if hist is not None and len(hist) > 1 else float(sp[0])
        moving = sp > 0.5
        head = np.arctan2(vel[moving, 1], vel[moving, 0]) if moving.sum() >= 2 else np.zeros(2)
        yaw = float(np.rad2deg(np.angle(np.exp(1j * (head[-1] - head[0]))))) if moving.sum() >= 2 else 0.0
        dh = np.abs(np.angle(np.exp(1j * np.diff(head)))) if moving.sum() >= 3 else np.zeros(1)
        rows.append({"set": sname, "clip_id": cid, "v0": v0, "v_mean": float(sp.mean()), "v_end": float(sp[-5:].mean()),
                     "path_len": float(sp.sum() * 0.1), "yaw_net": yaw, "yaw_abs": float(np.rad2deg(dh).sum()),
                     "stops": bool((sp[-10:] < 0.5).all()), "keys": ",".join(sorted(z.files)) if len(rows) == 0 else ""})
        if len(rows) % 250 == 0:
            print(len(rows), flush=True)
pd.DataFrame(rows).to_parquet("/home/cvlab21/project/chan/.claude/jobs/48e21e0f/tmp/afc/gt_feats.parquet", index=False)
print("done", len(rows))
