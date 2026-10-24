import asyncio, csv, json, sys
from pathlib import Path
sys.path.insert(0, "/home/cvlab21/project/chan/alpamayo-model-compression/closed-loop-viz")
from render_replay import read_rollout

S = "clipgt-0f1f60d5-785b-4a47-8436-7046ad1a3d21"
O = Path(f"/home/cvlab21/project/chan/alpamayo-model-compression/closed-loop-viz/eb_qual/matrix150/{S}")
rows, summary = [], {}
for arm_dir in sorted((O / "runs").iterdir()):
    arm = arm_dir.name
    d = json.loads((arm_dir / "results-summary.json").read_text())
    rs = sorted([r for r in d["rollouts"] if r["clipgt_id"] == S], key=lambda r: float(r["score"]))
    summary[arm] = {"run_dir": str((arm_dir / "results-summary.json").resolve().parent.parent),
                    "scene_score_mean": sum(float(r["score"]) for r in rs) / len(rs),
                    "worst_rollout_shown_in_videos": rs[0]["rollout_id"],
                    "rollouts": rs}
    for r in rs:
        m = r.get("metrics") or {}
        rows.append({"arm": arm, "rollout_id": r["rollout_id"], "score": r["score"],
                     "status": r.get("status"), "shown_in_video": r is rs[0],
                     **{k: m.get(k) for k in ("progress_clipped_rel", "dist_traveled_m", "gt_dist_traveled_m",
                        "dist_to_gt_trajectory", "offroad", "collision_at_fault", "collision_any",
                        "wrong_lane", "plan_deviation", "min_distance_to_obstacle_m", "duration_frac_20s")}})
        asl = arm_dir / "rollouts" / r["rollout_id"] / "rollout.asl"
        if not asl.exists():
            continue
        ro = asyncio.run(read_rollout(asl))
        t0 = min(ro["frames"]) if ro["frames"] else 0
        steps = [{"t_s": (ts - t0) / 1e6, "timestamp_us": ts, "reasoning_text": txt,
                  "plan_xy": ro["plans"][i][1].round(3).tolist()}
                 for i, (ts, txt) in enumerate(ro["cocs"])]
        tag = "WORST" if r is rs[0] else "best"
        (O / "reasoning" / f"{arm}__{r['rollout_id']}__{tag}.json").write_text(json.dumps({
            "arm": arm, "rollout_id": r["rollout_id"], "score": r["score"], "t0_us": t0,
            "gt_path_xy": ro["gt"].round(3).tolist(),
            "ego_xy": [[(ts - t0) / 1e6, *[a[1:] for a in ro["frames"][ts] if a[0] == "EGO"][0]] for ts in sorted(ro["frames"])],
            "steps": steps}, indent=1))
(O / "metrics" / "scene_metrics.json").write_text(json.dumps(summary, indent=1))
with open(O / "metrics" / "rollout_metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for arm, s in sorted(summary.items(), key=lambda kv: -kv[1]["scene_score_mean"]):
    print(f"{arm:14s} {s['scene_score_mean']:.3f}")
