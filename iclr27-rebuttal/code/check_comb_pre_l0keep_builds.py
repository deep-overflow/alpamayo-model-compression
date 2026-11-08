"""Gate B1 of iclr27-rebuttal/plans/2026-11-08_comb-rules-pre-l0keep-test500.md: the new pre-l0keep raw builds remove exactly
what dual_u40_v2 removes and keep 19 Q heads / 7,390 MLP channels in every VLM layer. Prints each arm's layer-0 Q heads.

Usage: .venv/bin/python iclr27-rebuttal/code/check_comb_pre_l0keep_builds.py
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REF = REPO / "outputs" / "slim_dual_u40_v2" / "slim_meta.json"
ARMS = {"rank-mean": REPO / "outputs" / "slim_dualsum_u40_v2" / "slim_meta.json",
        "raw-mean": REPO / "outputs" / "iclr27-rebuttal" / "slim_rawabsavg_u40_v2" / "slim_meta.json",
        "raw-max": REPO / "outputs" / "iclr27-rebuttal" / "slim_rawabsmean_u40_v2" / "slim_meta.json"}


def main():
    ref = json.loads(REF.read_text())
    ok = True
    print(f"rank-max (dual_u40_v2) layer 0 Q: {sorted(ref['vlm'][0]['q'])}  params {ref['params']}")
    for name, f in ARMS.items():
        m = json.loads(f.read_text())
        shape_ok = all(len(L["q"]) == 19 and len(L["mlp"]) == 7390 for L in m["vlm"])
        params_ok = m["params"] == ref["params"]
        expert_ok = m["expert"] == ref["expert"]
        l0 = sorted(m["vlm"][0]["q"])
        same = len(set(l0) & set(ref["vlm"][0]["q"]))
        q_ovl = sum(len(set(a["q"]) & set(b["q"])) for a, b in zip(m["vlm"], ref["vlm"], strict=True)) / (36 * 19)
        m_ovl = sum(len(set(a["mlp"]) & set(b["mlp"])) for a, b in zip(m["vlm"], ref["vlm"], strict=True)) / (36 * 7390)
        print(f"{name:<10} shape {shape_ok} params {params_ok} expert {expert_ok} | layer 0 Q {l0} "
              f"({same}/19 shared with rank-max) | overlap with rank-max Q {q_ovl:.3f} MLP {m_ovl:.3f}")
        ok &= shape_ok and params_ok and expert_ok
    print("B1", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
