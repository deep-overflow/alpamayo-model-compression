"""The contrast the gate is defined on: em93p75 vs dualr_wl, paired per scene.

analyze_alpasim.py only reports each arm against the unpruned baseline, but the question
here is whether the expert-MLP cut costs anything ON TOP OF dualr_wl. Same convention:
per-rollout -> per-scene mean -> paired delta, bootstrap CI on the mean plus Wilcoxon.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

RUNS = Path("/home/cvlab21/project/chan/alpasim-runs")
ARMS = {
    "baseline": "m2601_merged_baseline",
    "dual": "m2601_merged_slim_dual_u40_v2",
    "dualr_wl": "m2601_merged_slim_dualr_wl_u40",
    "em93p75": "m2601_merged_slim_dualrwl_em93p75_u40",
    "dualexp": "m2601_merged_slim_dualexp_u40_em93p75",
}
KEYS = ["score", "passed", "collision_at_fault", "offroad", "progress_clipped_rel",
        "dist_to_gt_trajectory"]


def per_scene(run):
    d = json.loads((RUNS / run / "aggregate" / "results-summary.json").read_text())
    out = {}
    for r in d["rollouts"]:
        s = r["clipgt_id"]
        rec = out.setdefault(s, {k: [] for k in KEYS})
        rec["score"].append(float(r["score"]) if r["score"] is not None else np.nan)
        rec["passed"].append(float(bool(r["passed"])))
        for k in KEYS[2:]:
            v = r.get("metrics", {}).get(k)
            rec[k].append(float(v) if v is not None else np.nan)
    return {s: {k: float(np.nanmean(v)) for k, v in rec.items()} for s, rec in out.items()}


data = {a: per_scene(r) for a, r in ARMS.items()}
scenes = sorted(set.intersection(*[set(d) for d in data.values()]))
print(f"scenes common to all four arms: {len(scenes)}\n")


def contrast(a, b, key="score"):
    d = np.array([data[a][s][key] - data[b][s][key] for s in scenes])
    d = d[~np.isnan(d)]
    rng = np.random.default_rng(0)
    boots = [np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(10000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = wilcoxon(d).pvalue if np.any(d) else 1.0
    w, l, t = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
    star = "*" if lo > 0 or hi < 0 else " "
    print(f"  {a:9s} - {b:9s} {key:22s} {np.mean(d):+.4f} [{lo:+.4f}, {hi:+.4f}]{star}"
          f"  p={p:.3g}  W/L/T {w}/{l}/{t}")


print("score, paired per scene:")
for a, b in (("dualexp", "dual"), ("dualexp", "em93p75"), ("dualexp", "baseline"),
             ("em93p75", "dualr_wl"), ("em93p75", "baseline"), ("em93p75", "dual"),
             ("dualr_wl", "baseline"), ("dual", "baseline")):
    contrast(a, b)
print("\ndualexp - dual, the other headline metrics:")
for k in KEYS[1:]:
    contrast("dualexp", "dual", k)
