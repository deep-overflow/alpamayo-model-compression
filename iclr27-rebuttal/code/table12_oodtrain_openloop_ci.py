"""Paper Table 12, OOD-train columns: 6.4 s minADE@6 / minFDE@6 with 95% CIs per arm (GH200, 1,271 clips).

Point estimates are those of analyze_oodtrain_gh200.py (clip mean of min over the first 6 of 8 samples, rollout).
CIs use ci_openloop_lingoqa.boot_many -- the bootstrap behind the table's Test500 / OOD-Val262 columns (clip-level,
200,000 draws, seed 0, shared draws across arms) -- with seed 1 as a stability check.

  .venv/bin/python iclr27-rebuttal/code/table12_oodtrain_openloop_ci.py
writes outputs/iclr27-rebuttal/table12_oodtrain_openloop/{metrics.json, table.md}
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "iclr27-rebuttal" / "code"))
sys.path.insert(0, str(REPO / "cvpr27" / "code"))
import analyze_oodtrain_gh200 as og
import ci_openloop_lingoqa as ci
from analyze_a100 import load_rows, metric

ROOT = REPO / "outputs" / "iclr27-rebuttal" / "gh200"
OUT = REPO / "outputs" / "iclr27-rebuttal" / "table12_oodtrain_openloop"
ORDER = ["dense", "wanda", "lp_r50", "tyr_u40_r", "fm_only", "ce_only", "moop", "moop_em93p75"]


def main():
    ids, _ = og.ids_and_clusters()
    series = {}
    for k in ORDER:
        r = load_rows(ROOT / f"{k}_oodtrain", og.TAG[k])
        assert all(c in r for c in ids), f"{k}: clips missing"
        for kind in ("ade", "fde"):
            series[f"{k}|{kind}"] = [metric(r[c], kind, "6.4s") for c in ids]  # (1271,)
    b0 = ci.boot_many(series, seed=0)
    b1 = ci.boot_many(series, seed=1)
    res = {}
    lines = ["| arm | clips | minADE@6 6.4 s [95% CI] | minFDE@6 6.4 s [95% CI] | seed-1 CIs (ADE; FDE) |",
             "|---|---|---|---|---|"]
    for k in ORDER:
        e = {}
        for kind in ("ade", "fde"):
            key = f"{k}|{kind}"
            e[kind] = {"mean": float(np.mean(series[key])), "ci": list(b0[key][:2]), "ci_seed1": list(b1[key][:2])}
        res[k] = {"label": og.LABEL[k], "n": len(ids), **e}
        a, f = e["ade"], e["fde"]
        lines.append(f"| {og.LABEL[k]} | {len(ids)} | {a['mean']:.3f} [{a['ci'][0]:.3f}, {a['ci'][1]:.3f}] "
                     f"| {f['mean']:.3f} [{f['ci'][0]:.3f}, {f['ci'][1]:.3f}] "
                     f"| [{a['ci_seed1'][0]:.3f}, {a['ci_seed1'][1]:.3f}]; [{f['ci_seed1'][0]:.3f}, {f['ci_seed1'][1]:.3f}] |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps({"draws": ci.N_BOOT, "arms": res}, indent=1))
    (OUT / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
