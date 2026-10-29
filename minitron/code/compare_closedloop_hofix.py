"""Dense Alpamayo 1.5 on Origin150, before vs after the AlpaSim hand-over fix, paired per scene.

Before = alpasim-runs/m2601_merged_baseline, after = hofix_merged_baseline: same 150 scenes (public_2601
first 150) x 2 rollouts. Per rollout -> per-scene mean -> paired delta (after - before), bootstrap 95% CI
(20,000 draws) and Wilcoxon. Also the hand-over itself on every moving start (>= 2 m/s): the car's speed
change over the first free step minus the human's, and the controller acceleration at hand-over.
Reads rollout.asl, so run with alpasim's venv:
  /mnt/nvme1n1/ad_vla/venvs/alpasim-chan/bin/python minitron/code/compare_closedloop_hofix.py
Writes outputs/minitron/closedloop_hofix_compare/{metrics.json, summary.txt}.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "evaluation"))

import analyze_handover as A

RUNS = Path("/home/cvlab21/project/chan/alpasim-runs")
BEFORE, AFTER = "m2601_merged_baseline", "hofix_merged_baseline"
KEYS = ["collision_at_fault", "collision_any", "offroad", "progress_clipped_rel", "progress_rel_to_total",
        "dist_traveled_m", "dist_to_gt_trajectory", "plan_deviation"]


def per_scene(run):
    d = json.loads((RUNS / run / "aggregate" / "results-summary.json").read_text())
    out = {}
    for r in d["rollouts"]:
        rec = out.setdefault(r["clipgt_id"], {k: [] for k in ["score", "passed", *KEYS]})
        rec["score"].append(float(r["score"]) if r["score"] is not None else np.nan)
        rec["passed"].append(float(bool(r["passed"])))
        for k in KEYS:
            v = r.get("metrics", {}).get(k)
            rec[k].append(float(v) if v is not None else np.nan)
    return {s: {k: float(np.nanmean(v)) for k, v in rec.items()} for s, rec in out.items()}


def boot(d, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(d, (n, len(d))).mean(1)
    return [float(x) for x in np.percentile(m, [2.5, 97.5])]


def handover():
    tasks = []
    for label, run in (("before", BEFORE), ("after", AFTER)):
        for asl in sorted((RUNS / run / "rollouts").glob("*/*/rollout.asl")):
            tasks.append(((label, "baseline", asl.parent.parent.name, asl.parent.name), asl, 30))
    df = A.read_many(tasks, workers=12)
    res = {}
    for label, g in df.groupby("set"):
        rows = []
        for _, r in g.groupby(["scene", "rollout"]):
            r = r.set_index("k")
            if 0 in r.index and 1 in r.index and r.at[0, "v"] >= 2.0:
                rows.append((r.at[1, "v"] - r.at[0, "v"], r.at[1, "vh"] - r.at[0, "vh"], r.at[0, "a_ret"]))
        a = np.array(rows)
        res[label] = {"moving": len(a), "median_dv_car": float(np.median(a[:, 0])),
                      "median_dv_human": float(np.median(a[:, 1])),
                      "median_abs_car_minus_human": float(np.median(np.abs(a[:, 0] - a[:, 1]))),
                      "share_a0_saturated": float(np.mean(a[:, 2] <= -7.9))}
    return res


def main():
    b, a = per_scene(BEFORE), per_scene(AFTER)
    scenes = sorted(set(a) & set(b))
    res = {"n_scenes": len(scenes), "metrics": {}}
    for k in ["score", "passed", *KEYS]:
        x = np.array([b[s][k] for s in scenes])
        y = np.array([a[s][k] for s in scenes])
        ok = ~(np.isnan(x) | np.isnan(y))
        d = y[ok] - x[ok]
        p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        res["metrics"][k] = {"before": float(x[ok].mean()), "after": float(y[ok].mean()),
                             "delta": float(d.mean()), "ci95": boot(d), "wilcoxon_p": p, "n": int(ok.sum())}
    res["handover"] = handover()
    out = REPO / "outputs" / "minitron" / "closedloop_hofix_compare"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    lines = [(f"Origin150 dense, before vs after the hand-over fix: {len(scenes)} scenes x 2 rollouts, "
              "per-scene means, delta = after - before"),
             f"{'metric':24s} {'before':>8s} {'after':>8s} {'delta':>8s}  95% CI              Wilcoxon p"]
    for k, m in res["metrics"].items():
        lines.append(f"{k:24s} {m['before']:8.4f} {m['after']:8.4f} {m['delta']:+8.4f}  "
                     f"[{m['ci95'][0]:+.4f}, {m['ci95'][1]:+.4f}]  {m['wilcoxon_p']:.3g}")
    for label in ("before", "after"):
        h = res["handover"][label]
        lines.append(f"hand-over ({label}): moving starts {h['moving']}, first free step car "
                     f"{h['median_dv_car']:+.3f} vs human {h['median_dv_human']:+.3f} m/s, "
                     f"|car - human| {h['median_abs_car_minus_human']:.3f}, accel saturated {h['share_a0_saturated']:.0%}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
