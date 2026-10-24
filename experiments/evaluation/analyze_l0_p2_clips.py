"""How many calibration clips does the propagated signal P2 need? No GPU.

plans/2026-10-14_l0-functional-redundancy.md. Random clip subsets of step 2's per-clip FM loss
(first attempt's 80 checkpointed clips + the 20-clip rerun), re-ranked against groups A and B.

Usage:
  python experiments/evaluation/analyze_l0_p2_clips.py > outputs/l0_redundancy/p2_clip_count.txt
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

O = "outputs"
files = [f"{O}/l0_redundancy/attempt1/l0red_prop_s{s}.metrics.json" for s in range(4)] + \
        [f"{O}/l0red_prop_rest_s{s}/metrics.json" for s in range(4)]
ms = [json.loads(Path(f).read_text()) for f in files]
sj = json.loads(Path(f"{O}/l0_redundancy/sets.json").read_text())
use = sorted({a["set"] for a in sj["arms"] if a["group"] in "AB"})
fm = {n: np.concatenate([np.asarray(m["per_clip"][f"arm_{n}"]["fm"], float) for m in ms]) for n in use}
dense = np.concatenate([np.asarray(m["per_clip"]["dense"]["fm"], float) for m in ms])
print(f"{len(dense)} clips, {len(use)} sets")
rng = np.random.default_rng(0)
for g in "AB":
    ga = [a for a in sj["arms"] if a["group"] == g]
    y = np.array([a["y"] for a in ga])
    full = np.array([fm[a["set"]].mean() for a in ga])
    print(f"\ngroup {g}: full 100 clips rho {spearmanr(full, y)[0]:+.3f}")
    for n in (10, 20, 30, 50, 75):
        rs, agree = [], []
        for _ in range(300):
            idx = rng.choice(len(dense), n, replace=False)
            x = np.array([fm[a["set"]][idx].mean() for a in ga])
            rs.append(spearmanr(x, y)[0])
            agree.append(spearmanr(x, full)[0])
        rs = np.array(rs)
        print(f"  n={n:3d}: rho median {np.median(rs):+.3f}  10th pct {np.percentile(rs, 10):+.3f}  "
              f"P(rho >= thresholds) {np.mean(rs >= (0.45 if g == 'A' else 0.20)):.2f}  "
              f"rank agreement with full {np.median(agree):.3f}")
# relative effect size: how much does FM move between best and worst sets, vs dense?
vals = np.array([fm[n].mean() for n in use])
print(f"\nP2 across the 73 sets: min {vals.min():.4f}  median {np.median(vals):.4f}  max {vals.max():.4f}; "
      f"dense {dense.mean():.4f}")
