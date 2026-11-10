"""Build gates for the u50 closed loop (iclr27-rebuttal/plans/2026-11-09_u50-closedloop-origin150.md, step 0).

  G0-moop  this code's `make_slim.py --config dual_u40_v2` == outputs/slim_dual_u40_v2 (paper MoOP), unit for unit
  G0-lp    this code's `lp_prune.py --ratio 0.5` == the paper LLM-Pruner driver lp_r50, unit for unit
  B-moop   dual_u50_v2 removes 3,321,888,768 parameters (the u50 budget Tyr's supernet is cut to)
  B-tyr    tyr_u50_r: VLM text tower only (expert untouched), removed within 1% of the u50 budget
           (type-conserving levels keep it equal unless a level kept 0 heads -> zero-output placeholder head)
  B-lp     lp_r60: layers 4-33 only, expert untouched; units it keeps are a subset of lp_r50's (same scores, more cut)
Prints one line per gate and `gates: PASS` / `gates: FAIL`; exit status 1 on FAIL. Writes <out>/gates.json.

  .venv/bin/python iclr27-rebuttal/code/check_u50_builds.py
"""

import json
import sys
from pathlib import Path

REPO = Path("/home/cvlab21/project/chan/alpamayo-model-compression")
OUT = REPO / "outputs" / "iclr27-rebuttal" / "u50_origin150"
DRIVERS = Path("/mnt/nvme1n1/ad_vla/data/alpasim/drivers")
U50 = 3_321_888_768


def meta(d):
    return json.loads((Path(d) / "slim_meta.json").read_text())


def units(m):
    return [(sorted(l["q"]), sorted(l["mlp"])) for l in m["vlm"]], [(sorted(l["q"]), sorted(l["mlp"])) for l in m["expert"]]


def same_units(a, b):
    return units(a) == units(b)


def expert_untouched(m, ref):
    return units(m)[1] == units(ref)[1]


def main():
    res = {}
    paper_moop = meta(REPO / "outputs" / "slim_dual_u40_v2")
    res["G0-moop"] = same_units(meta(OUT / "g0" / "dual_u40_v2"), paper_moop)
    lp50 = meta(DRIVERS / "lp_r50")
    res["G0-lp"] = same_units(meta(OUT / "g0" / "lp_r50"), lp50)

    moop = meta(REPO / "outputs" / "slim_dual_u50_v2")
    res["B-moop"] = moop["params"]["removed"] == U50

    tyr = meta(REPO / "outputs" / "slim_tyr_u50_r")
    tr = tyr["params"]["removed"]
    res["B-tyr"] = expert_untouched(tyr, paper_moop) and abs(tr - U50) <= 0.01 * U50

    lp = meta(REPO / "outputs" / "slim_lp_r60")
    v60, v50 = units(lp)[0], units(lp50)[0]
    in_scope = all(v60[i] == v50[i] for i in list(range(4)) + [34, 35])
    subset = all(set(v60[i][0]) <= set(v50[i][0]) and set(v60[i][1]) <= set(v50[i][1]) for i in range(36))
    res["B-lp"] = expert_untouched(lp, lp50) and in_scope and subset

    info = {"moop_removed": moop["params"]["removed"], "tyr_removed": tr, "lp_removed": lp["params"]["removed"],
            "tyr_placeholder_layers": tyr.get("attn_placeholder_layers", []),
            "tyr_kept_q_per_layer": [len(l["q"]) for l in tyr["vlm"]],
            "lp_kept_q_per_layer": [len(l["q"]) for l in lp["vlm"]]}
    for k, v in res.items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")
    for k, v in info.items():
        print(f"  {k}: {v}")
    ok = all(res.values())
    print(f"gates: {'PASS' if ok else 'FAIL'}")
    (OUT / "gates.json").write_text(json.dumps({"gates": res, **info}, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
