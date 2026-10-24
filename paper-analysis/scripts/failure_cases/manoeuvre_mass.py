"""Per-manoeuvre importance vs each arm's kept set. From the calibration clips' per-clip scores
(outputs/importance_v2/importance_perclip.npz, 100 clips) build I_traj^m and I_CoC^m = mean over
the calibration clips of manoeuvre m, then ask how much of that mass each shipped arm removed
(traj / coc / dual kept sets from slim_meta.json), per band. Also the per-clip weight of each
manoeuvre in the pooled score (mass share / clip share)."""
import json, sys
import numpy as np
sys.path.insert(0, "experiments/evaluation"); sys.path.insert(0, "experiments/head_analysis")
import sample_cache as sc, eval_lib as el
REPO = "/home/cvlab21/project/chan/alpamayo-model-compression"
z = np.load(f"{REPO}/outputs/importance_v2/importance_perclip.npz")
met = json.load(open(f"{REPO}/outputs/importance_v2/metrics.json"))
clips = sc.calib_samples(REPO, "calib_100")
ids = None
for key in ("clip_ids", "per_clip", "clips"):
    if key in met:
        v = met[key]; ids = [r["clip_id"] if isinstance(r, dict) else r for r in v]; print("per-clip order from metrics.json:", key); break
if ids is None:
    ids = [c for c, _ in clips]; print("per-clip order assumed = manifest order")
t0 = dict(clips)
buckets = np.array([el.bucket(np.asarray(np.load(sc.path_for("calib", cid, t0[cid]), allow_pickle=True)["ego_future_xyz"]).reshape(-1, 3)[:, :2]) for cid in ids])
print("calib buckets:", {b: int((buckets == b).sum()) for b in ("cruise", "accel", "decel_stop", "turn")})
kept = {}
for arm in ("traj", "coc", "dual"):
    m = json.load(open(f"{REPO}/outputs/slim_{arm}_u40_v2/slim_meta.json"))["vlm"]
    kept[arm] = {"q": np.zeros((36, 32), bool), "mlp": np.zeros((36, 12288), bool)}
    for l in range(36):
        kept[arm]["q"][l, m[l]["q"]] = True; kept[arm]["mlp"][l, m[l]["mlp"]] = True
BANDS = {"0-21": slice(0, 22), "22-34": slice(22, 35)}
MAN = ("cruise", "accel", "decel_stop", "turn", "all")
out = {}
for obj in ("traj", "coc"):
    for axis in ("q", "mlp"):
        P = z[f"{obj}_vlm_{axis}"].astype(np.float64)  # (100, 36, U) per-clip |gate grad|
        tot = P.sum((1, 2)); tot[~np.isfinite(tot)] = 0
        w = {m: tot[buckets == m].sum() / tot.sum() / ((buckets == m).mean()) for m in MAN[:-1]}
        print(f"\n## I_{obj}, {axis}: per-clip weight of each manoeuvre in the pooled score (mass share / clip share): " + ", ".join(f"{m} {w[m]:.2f}" for m in w))
        print(f"| manoeuvre | band | removed by traj arm | by coc arm | by dual arm | by own ranking (floor) |")
        print("|---|---|---:|---:|---:|---:|")
        for m in MAN:
            sel = np.ones(len(buckets), bool) if m == "all" else buckets == m
            I = P[sel].mean(0)  # (36, U)
            for b, sl in BANDS.items():
                cells = []
                for arm in ("traj", "coc", "dual"):
                    r = [I[l][~kept[arm][axis][l]].sum() / I[l].sum() for l in range(sl.start, sl.stop) if I[l].sum() > 0]
                    cells.append(np.mean(r)); out[f"{obj}|{axis}|{m}|{b}|{arm}"] = float(np.mean(r))
                k_rm = (~kept["traj"][axis][0]).sum()
                floor = np.mean([np.sort(I[l])[:k_rm].sum() / I[l].sum() for l in range(sl.start, sl.stop) if I[l].sum() > 0])
                print(f"| {m} | {b} | {cells[0]:.3f} | {cells[1]:.3f} | {cells[2]:.3f} | {floor:.3f} |")
json.dump(out, open("/home/cvlab21/project/chan/.claude/jobs/48e21e0f/tmp/afc/manoeuvre_mass.json", "w"), indent=1)
