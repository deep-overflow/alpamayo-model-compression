"""The layer-0 head sets of the redundancy test, their outcomes, and one mask per set.

plans/2026-10-14_l0-functional-redundancy.md. Three outcome groups, read from stored runs:

  A  the 50 dual-family u40 arms with an Ada test500 run (analyze_l0_census.py's set):
     mean minADE@6 over 500 clips
  B  the 64 calib_100 half-split arms (clip-influence study): delta minADE@6 on 150 val clips
  C  the single-head map under the shipped mask (make_l0_masks.py ship_L0drop<h>: head h
     dropped, head 20 restored): mean minADE@6 on 60 cruise clips

Every distinct layer-0 dropped set gets a keep mask for run_token_ablation.py --arm-masks
(step 2): the as-built shipped arm (dual_u40_v2) in layers 1-35, that set in layer 0. All
masks remove the same parameters, so step 2 compares layer-0 choices and nothing else.

Writes outputs/l0_redundancy/sets.json and outputs/l0_redundancy/masks/<set>.npz.

Usage:
  python experiments/evaluation/make_l0_redundancy_masks.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_l0_census import FAMILIES, K, O, u40_meta

N_HEADS, INTER = 32, 12288


def load_rows(exp):
    rows = {}
    for p in (O / exp).glob("*.json"):
        if p.name.startswith("config"):
            continue
        try:
            for r in json.loads(p.read_text()):
                if "ade_rollout_k" in r and len(r["ade_rollout_k"]) >= K:
                    rows[r["clip_id"]] = min(r["ade_rollout_k"][:K])
        except (json.JSONDecodeError, TypeError, KeyError):
            continue
    return rows


def main():
    out = O / "l0_redundancy"
    (out / "masks").mkdir(parents=True, exist_ok=True)
    meta = {d.name[5:]: v for d in sorted(O.glob("slim_*")) if not d.name.startswith("slim_swap_")
            for v in [u40_meta(d)] if v is not None}
    drop0 = {n: sorted(set(range(N_HEADS)) - set(v[0]["q"])) for n, v in meta.items()}

    # group A: the census's rule -- Ada, set test, no manifest / quant, all 500 clips
    test = {}
    for cfgp in sorted(O.glob("*/config.json")):
        try:
            c = json.loads(cfgp.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        model, st = str(c.get("model", "")), c.get("set")
        if "slim_" not in model or st != "test" or c.get("manifest") or c.get("quant"):
            continue
        name = Path(model).name[5:]
        if name not in meta or "Ada" not in str(c.get("gpu", "")) or name in test:
            continue
        rows = load_rows(cfgp.parent.name)
        if len(rows) == 500:
            test[name] = (float(np.mean(list(rows.values()))), cfgp.parent.name)
    arms = []
    for fam, names in FAMILIES.items():
        for n in names:
            if n in meta and n in test:
                arms.append({"group": "A", "arm": n, "family": fam, "drop": drop0[n],
                             "y": test[n][0], "run": test[n][1]})

    # group B: half-split arms, y = delta minADE on 150 val clips (clipinfluence/metrics.json)
    ci = json.loads((O / "clipinfluence" / "metrics.json").read_text())
    for a, v in zip(ci["arms"], ci["y_ade"]["values"]):
        n = f"subinf_{a}"
        if n in meta:
            arms.append({"group": "B", "arm": n, "family": "half-split", "drop": drop0[n],
                         "y": float(v), "run": "clipinfluence"})

    # group C: single-head map under the shipped mask
    ship = drop0["dual_u40_v2"]
    l0 = json.loads((O / "swap_masks" / "metrics_l0.json").read_text())
    for h, rec in l0["l0_map"].items():
        d = sorted((set(ship) - {20}) | {int(h)})
        arms.append({"group": "C", "arm": f"ship_L0drop{h}", "family": "single-head map",
                     "drop": d, "y": float(rec["mean"]), "run": "swap_masks/metrics_l0.json"})

    # one mask per distinct set: shipped layers 1-35, this set in layer 0
    base_q = np.zeros((36, N_HEADS), np.float32)
    base_m = np.zeros((36, INTER), np.float32)
    for li, e in enumerate(meta["dual_u40_v2"]):
        base_q[li, e["q"]] = 1.0
        base_m[li, e["mlp"]] = 1.0
    distinct = sorted({tuple(a["drop"]) for a in arms})
    sets = {}
    for i, d in enumerate(distinct):
        name = f"s{i:02d}"
        q = base_q.copy()
        q[0] = 1.0
        q[0, list(d)] = 0.0
        assert q[0].sum() == N_HEADS - 13, (name, d)
        np.savez(out / "masks" / f"{name}.npz", q_mask=q, mlp_mask=base_m)
        sets[name] = {"drop": list(d), "arms": [a["arm"] for a in arms if tuple(a["drop"]) == d]}
    for a in arms:
        a["set"] = next(k for k, v in sets.items() if v["drop"] == a["drop"])
    (out / "sets.json").write_text(json.dumps({"ship_drop": ship, "sets": sets, "arms": arms},
                                              indent=1))
    for g in "ABC":
        ga = [a for a in arms if a["group"] == g]
        print(f"group {g}: {len(ga)} arms, {len({a['set'] for a in ga})} distinct sets")
    print(f"{len(sets)} distinct layer-0 sets -> {out / 'masks'}")


if __name__ == "__main__":
    main()
