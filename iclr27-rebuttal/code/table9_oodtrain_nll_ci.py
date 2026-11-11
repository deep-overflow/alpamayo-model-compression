"""Paper Table 9, column "NLL, OOD-train (ground-truth CoC)": 95% CIs for each arm's mean reference-CoC NLL.

Point estimates are those of analyze_oodtrain_gh200.py (GH200, 1,271 OOD-train clips, clip mean of nll_gtcoc).
CIs use ci_openloop_lingoqa.boot_many -- the bootstrap behind the table's Test500 NLL column (clip-level, 200,000
draws, seed 0, shared draws across arms) -- with seed 1 as a stability check.

  .venv/bin/python iclr27-rebuttal/code/table9_oodtrain_nll_ci.py
writes outputs/iclr27-rebuttal/table9_oodtrain_nll/{metrics.json, table.md}
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

ROOT = REPO / "outputs" / "iclr27-rebuttal" / "gh200"
OUT = REPO / "outputs" / "iclr27-rebuttal" / "table9_oodtrain_nll"
ORDER = ["dense", "wanda", "lp_r50", "tyr_u40_r", "fm_only", "ce_only", "moop", "moop_em93p75"]


def main():
    ids, _ = og.ids_and_clusters()
    rows = {k: og.nll_rows(ROOT, k) for k in ORDER}
    for k, r in rows.items():
        assert all(c in r for c in ids), f"{k}: NLL clips missing"
    series = {k: [rows[k][c]["nll_gtcoc"] for c in ids] for k in ORDER}  # (1271,) each
    b0 = ci.boot_many(series, seed=0)
    b1 = ci.boot_many(series, seed=1)
    res, lines = {}, ["| arm | clips | NLL OOD-train (GT CoC) [95% CI] | seed-1 CI |", "|---|---|---|---|"]
    for k in ORDER:
        m = float(np.mean(series[k]))
        res[k] = {"label": og.LABEL[k], "n": len(ids), "mean": m, "ci": list(b0[k][:2]), "ci_seed1": list(b1[k][:2])}
        lines.append(f"| {og.LABEL[k]} | {len(ids)} | {m:.3f} [{b0[k][0]:.3f}, {b0[k][1]:.3f}] "
                     f"| [{b1[k][0]:.3f}, {b1[k][1]:.3f}] |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps({"draws": ci.N_BOOT, "arms": res}, indent=1))
    (OUT / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
