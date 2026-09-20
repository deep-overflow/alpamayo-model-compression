"""Which gate fired, per arm, on one scene -- read from the run's own aggregate file.

`score_criteria` carries collision_at_fault and offroad as HARD gates, so a scene scoring
exactly 0 has tripped one of them and the progress term is irrelevant there. Knowing which
one decides what the replay should be pointing at.

The gates live under `rollout["metrics"]`, not at the top level; reading them from the top
level silently yields None for every arm.
"""
import json
import sys
from pathlib import Path

R = Path("/home/cvlab21/project/chan/alpasim-runs")
SCENE = sys.argv[1]
ARMS = sys.argv[2:]

for cfg in ARMS:
    d = json.loads((R / f"m2601_merged_{cfg}/aggregate/results-summary.json").read_text())
    for r in d["rollouts"]:
        if r["clipgt_id"] != SCENE:
            continue
        m = r.get("metrics") or {}
        gates = {k: m.get(k) for k in ("collision_at_fault", "offroad", "wrong_lane")}
        extra = {k: round(float(v), 2) for k, v in m.items()
                 if k in ("progress", "distance_to_gt_m", "gt_dist_traveled_m",
                          "dist_traveled_m") and v is not None}
        print(f"{cfg:28s} score {float(r['score']):.3f}  status {r['status']:5s}  "
              f"rollout {r['rollout_id'][:8]}")
        print(f"{'':28s} gates {gates}")
        if extra:
            print(f"{'':28s} {extra}")
