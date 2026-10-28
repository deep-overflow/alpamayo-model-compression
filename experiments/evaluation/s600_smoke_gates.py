"""Smoke gates for the dense 600-scene closed-loop run (plans/2026-10-28_dense-closedloop-full2601.md §4).

Reads alpasim-runs/s600smoke_merged_baseline (16 sampled scenes outside Origin150, two per card, all
eight cards) and judges:
  G1  every scene has a scored rollout and no rollout failed for a reason other than collision / offroad
      (a scene-loading failure from the hardlinked cache, or a crash on a Blackwell card, would show here)
  G2  hand-over still fixed: on moving starts (>= 2 m/s) median |dv_car - dv_human| over the first free
      step below 0.1 m/s, and no controller acceleration at the hand-over at or below -7.9 m/s2
Also records each shard's driver throughput (seconds per 100 inference batches, from the launch log)
against hofix_baseline_sh0-3, the same harness on four Ada cards, so the CPU cost of eight stacks shows.
Reads rollout.asl, so run it with alpasim's venv:
  /mnt/nvme1n1/ad_vla/venvs/alpasim-chan/bin/python experiments/evaluation/s600_smoke_gates.py
Writes outputs/alpasim_dense_s600/smoke/{metrics.json, summary.txt}. Another arm's smoke run is judged the same
way by passing --run / --scenes / --logs / --out, e.g. MoOP (plans/2026-10-28_dual-closedloop-s600.md):
  ... s600_smoke_gates.py --run d600smoke_merged_slim_dual_u40_v2 --logs d600smoke_slim_dual_u40_v2 \
      --scenes outputs/alpasim_dual_s600/scenes_smoke8.csv --out outputs/alpasim_dual_s600/smoke
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import analyze_handover as A
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RUNS = Path("/home/cvlab21/project/chan/alpasim-runs")
MOVING, SAT = 2.0, -7.9
BATCH = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+\].*Inference batches: (\d+) processed")


def batch_rate(log):
    """Seconds per 100 driver inference batches between the first and last progress line of a launch log."""
    pts = [(datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S"), int(m[2]))
           for m in BATCH.finditer(Path(log).read_text(errors="replace"))]
    if len(pts) < 2:
        return None
    (t0, n0), (t1, n1) = pts[0], pts[-1]
    return (t1 - t0).total_seconds() / (n1 - n0) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="s600smoke_merged_baseline", help="merged smoke run under alpasim-runs")
    ap.add_argument("--logs", default="s600smoke_baseline", help="shard launch logs: <logs>_sh<i>.launch.log")
    ap.add_argument("--scenes", type=Path, default=REPO / "outputs" / "alpasim_dense_s600" / "scenes_smoke16.csv")
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "alpasim_dense_s600" / "smoke")
    args = ap.parse_args()
    RUN, OUT = RUNS / args.run, args.out
    scenes = sorted(pd.read_csv(args.scenes)["scene_id"])
    summ = json.loads((RUN / "aggregate" / "results-summary.json").read_text())["rollouts"]
    scored = {r["clipgt_id"] for r in summ if r.get("score") is not None}
    bad = [(r["clipgt_id"], r.get("failure_reason")) for r in summ
           if r.get("failure_reason") not in (None, "collision_at_fault", "offroad")]

    tasks = [((("new", "baseline", p.parent.parent.name, p.parent.name)), p, 30)
             for p in sorted((RUN / "rollouts").glob("*/*/rollout.asl"))]
    df = A.read_many(tasks, workers=8)
    rows = []
    for (sc, ro), g in df.groupby(["scene", "rollout"]):
        g = g.set_index("k")
        if 0 in g.index and 1 in g.index:
            rows.append({"scene": sc, "v0": g.at[0, "v"], "a0": g.at[0, "a_ret"],
                         "err": abs((g.at[1, "v"] - g.at[0, "v"]) - (g.at[1, "vh"] - g.at[0, "vh"]))})
    r = pd.DataFrame(rows)
    mv = r[r["v0"] >= MOVING]

    rate = {f"new_sh{i}": batch_rate(RUNS / f"{args.logs}_sh{i}.launch.log") for i in range(8)}
    rate.update({f"hofix_sh{i}": batch_rate(RUNS / f"hofix_baseline_sh{i}.launch.log") for i in range(4)})

    gates = {
        "G1": scored >= set(scenes) and not bad,
        "G2": bool(len(mv)) and float(mv["err"].median()) < 0.1 and bool((mv["a0"] > SAT).all()),
    }
    res = {"scenes": len(scenes), "scored": len(scored & set(scenes)), "non_gate_failures": bad,
           "moving": len(mv), "median_abs_dv_minus_human": float(mv["err"].median()) if len(mv) else None,
           "n_a0_saturated": int((mv["a0"] <= SAT).sum()), "median_a0": float(mv["a0"].median()) if len(mv) else None,
           "sec_per_100_batches": rate, "gates": gates}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(res, indent=2, default=str))
    bw = [v for k, v in rate.items() if k.startswith("new") and int(k[-1]) < 4 and v]
    ada = [v for k, v in rate.items() if k.startswith("new") and int(k[-1]) >= 4 and v]
    ref = [v for k, v in rate.items() if k.startswith("hofix") and v]
    lines = [f"scenes {len(scenes)}, scored {res['scored']}, non-gate failures {len(bad)} {bad}",
             (f"moving starts {len(mv)}: median |dv_car - dv_human| {res['median_abs_dv_minus_human']:.3f} m/s, "
             f"accel at hand-over median {res['median_a0']:+.2f} m/s2, saturated {res['n_a0_saturated']}"),
             "driver s / 100 batches: Blackwell " + " ".join(f"{v:.0f}" for v in bw)
             + " | Ada " + " ".join(f"{v:.0f}" for v in ada) + " | hofix 4xAda " + " ".join(f"{v:.0f}" for v in ref),
             "gates: " + "  ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in gates.items())]
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
