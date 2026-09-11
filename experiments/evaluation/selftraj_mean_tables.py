"""Mean tables for the self-anchored trajectory-importance study, all three sets.

`analyze_selftraj.py` runs the G1-G4 gates on val500 only, so the report's three-set
mean tables (section 9) were computed by hand and one row went missing: `gtk10` had its
absolute means recorded but not its cost-vs-baseline row, because its test500 and
OOD-val runs finished after that table was written.

This prints every cell of those tables from the stored per-clip rollouts, so the report
has a script to regenerate rather than a hand transcription. Verified 2026-09-11: it
reproduces all seven previously published rows to the digits the report quotes.

    .venv/bin/python experiments/evaluation/selftraj_mean_tables.py

Means only -- the protocol's primary reading is the paired median with a bootstrap CI,
which `analyze_selftraj.py` owns. Mean cost equals the mean paired difference here
because every arm is evaluated on the same clips (the script intersects clip ids).
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "evaluation"))
import paper_numbers as pn

# (directory suffix, keep only the val half of the OOD manifest)
SETS = {"val500": ("_ps_indist", False),
        "test500": ("_ps_test", False),
        "OOD-val": ("_ps_oodval", True)}
ARMS = {"baseline": "baseline_ada", "shipped": "dual_u40_v2",
        "gtref": "dualgtref_u40_v2", "gtk3": "dualgtk3_u40_v2",
        "gtk5": "dualgtk5_u40_v2", "gtk10": "dualgtk10_u40_v2",
        "self": "dualself_u40_v2", "bestn6": "dualbestn6_u40_v2"}
# shipped dual's OOD run predates the _ps_oodval naming
OVERRIDE = {("shipped", "OOD-val"): "dual_u40_v2_ps_ood"}


def at6(rec, key):
    return float(np.min(np.asarray(rec[key], dtype=float)[:6]))


def main():
    for sname, (suffix, ood_val_only) in SETS.items():
        rows = {}
        for arm, stem in ARMS.items():
            d = OVERRIDE.get((arm, sname), stem + suffix)
            if not (REPO / "outputs" / d).is_dir():
                print(f"  [missing] {sname} {arm}: outputs/{d}")
                continue
            rows[arm] = pn.load(d, ood_val_only)
        ids = sorted(set.intersection(*[set(v) for v in rows.values()]))
        ade = {a: np.array([at6(rows[a][i], "ade_rollout_k") for i in ids]) for a in rows}
        fde = {a: np.array([at6(rows[a][i], "fde_rollout_k") for i in ids]) for a in rows}

        print(f"\n### {sname}  (n={len(ids)}, {len(rows)} arms)")
        print(f"  {'arm':9s} {'ADE mean':>9s} {'vs base':>8s} | {'FDE mean':>9s} {'vs base':>8s}")
        for a in ARMS:
            if a not in rows:
                continue
            print(f"  {a:9s} {ade[a].mean():9.4f} {np.mean(ade[a] - ade['baseline']):+8.4f} | "
                  f"{fde[a].mean():9.4f} {np.mean(fde[a] - fde['baseline']):+8.4f}")


if __name__ == "__main__":
    main()
