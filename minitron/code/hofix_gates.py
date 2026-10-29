"""Gates G1-G4 for the AlpaSim hand-over fix (minitron/plans/2026-10-26_alpasim-handover-fix.md §5).

Old = alpasim-runs/m2601_merged_baseline (dense, before the fix), new = hofix_gate_merged_baseline
(dense, after the fix), both on the 20 gate scenes (the first 20 of Origin150). Per rollout, k = 0
is the hand-over step (the first plan), k = 1 the first free step:
  G1  moving starts (>= 2 m/s at k = 0): median |dv_car - dv_human| over k = 0 -> 1 below 0.1 m/s
      (before the fix the car lost a median ~1.06 m/s here while the human did not)
  G2  share of moving starts whose controller-reported acceleration at k = 0 is <= -7.9 m/s2 below 5%
  G3  forced-GT phase (k < 0): ego positions identical old vs new, max |dxy| below 1 cm
  G4  every gate rollout produced a log that reaches the hand-over
Reads rollout.asl, so run it with alpasim's venv:
  /mnt/nvme1n1/ad_vla/venvs/alpasim-chan/bin/python minitron/code/hofix_gates.py
Writes outputs/minitron/hofix_gates/{metrics.json, summary.txt}.
"""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "evaluation"))

import analyze_handover as A

RUNS = Path("/home/cvlab21/project/chan/alpasim-runs")
OLD, NEW = RUNS / "m2601_merged_baseline", RUNS / "hofix_gate_merged_baseline"
SUITE = RUNS / "shard_suites_hofix_gate_baseline.csv"
MOVING, SAT = 2.0, -7.9


def forced_positions(asl):
    """controller_request ego (x, y) by timestamp, up to the first plan (the hand-over)."""
    from alpasim_utils.logs import async_read_pb_log

    async def go():
        pos, t_ho = {}, None
        async for e in async_read_pb_log(str(asl)):
            if e.HasField("driver_return") and t_ho is None and len(e.driver_return.trajectory.poses):
                t_ho = e.driver_return.trajectory.poses[0].timestamp_us
            elif e.HasField("controller_request"):
                s = e.controller_request.state
                if t_ho is not None and s.timestamp_us > t_ho:
                    break
                pos[s.timestamp_us] = (s.pose.vec.x, s.pose.vec.y)
        return pos, t_ho

    pos, t_ho = asyncio.run(go())
    return {t: p for t, p in pos.items() if t_ho is None or t < t_ho}


def main():
    scenes = sorted(pd.read_csv(SUITE)["scene_id"])
    tasks, asls = [], {}
    for label, d in (("old", OLD), ("new", NEW)):
        for sc in scenes:
            for rd in sorted((d / "rollouts" / sc).glob("*")):
                asl = rd / "rollout.asl"
                if asl.exists():
                    tasks.append(((label, "baseline", sc, rd.name), asl, 30))
                    asls.setdefault((label, sc), asl)
    df = A.read_many(tasks, workers=8)
    rows = []
    for (label, sc, ro), g in df.groupby(["set", "scene", "rollout"]):
        g = g.set_index("k")
        if 0 not in g.index or 1 not in g.index:
            continue
        rows.append({"set": label, "scene": sc, "rollout": ro, "v0": g.at[0, "v"],
                     "dv_car": g.at[1, "v"] - g.at[0, "v"], "dv_human": g.at[1, "vh"] - g.at[0, "vh"],
                     "a0": g.at[0, "a_ret"]})
    r = pd.DataFrame(rows)
    r["err"] = (r["dv_car"] - r["dv_human"]).abs()
    mv = r[r["v0"] >= MOVING]
    res = {}
    for label in ("old", "new"):
        m = mv[mv["set"] == label]
        res[label] = {"rollouts": int((r["set"] == label).sum()), "moving": len(m),
                      "median_abs_dv_minus_human": float(m["err"].median()),
                      "median_dv_car": float(m["dv_car"].median()),
                      "median_dv_human": float(m["dv_human"].median()),
                      "share_a0_saturated": float((m["a0"] <= SAT).mean()),
                      "median_a0": float(m["a0"].median())}
    gaps = []
    for sc in scenes:
        if ("old", sc) in asls and ("new", sc) in asls:
            po, pn = forced_positions(asls[("old", sc)]), forced_positions(asls[("new", sc)])
            common = sorted(set(po) & set(pn))
            if common:
                d = [np.hypot(po[t][0] - pn[t][0], po[t][1] - pn[t][1]) for t in common]
                gaps.append({"scene": sc, "n": len(common), "max_dxy_m": float(max(d))})
    n_new_logs = sum(1 for k in asls if k[0] == "new")
    gates = {
        "G1": res["new"]["median_abs_dv_minus_human"] < 0.1,
        "G2": res["new"]["share_a0_saturated"] < 0.05,
        "G3": bool(gaps) and max(g["max_dxy_m"] for g in gaps) < 0.01 and len(gaps) == len(scenes),
        "G4": n_new_logs == len(scenes) and res["new"]["rollouts"] == len(scenes),
    }
    out = REPO / "outputs" / "minitron" / "hofix_gates"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(
        {"scenes": scenes, "hand_over": res, "forced_phase_gap": gaps, "new_logs": n_new_logs,
         "gates": gates}, indent=2))
    lines = [f"gate scenes {len(scenes)} (first 20 of Origin150); moving start >= {MOVING} m/s"]
    for label in ("old", "new"):
        x = res[label]
        lines.append(f"{label}: rollouts {x['rollouts']}, moving {x['moving']} | first free step: car "
                     f"{x['median_dv_car']:+.3f} m/s, human {x['median_dv_human']:+.3f} m/s, "
                     f"median |car - human| {x['median_abs_dv_minus_human']:.3f} | accel at hand-over "
                     f"median {x['median_a0']:+.2f} m/s2, saturated {x['share_a0_saturated']:.0%}")
    if gaps:
        lines.append(f"forced phase: max |dxy| old vs new {max(g['max_dxy_m'] for g in gaps):.4f} m "
                     f"over {len(gaps)} scenes")
    lines.append("gates: " + "  ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in gates.items()))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
