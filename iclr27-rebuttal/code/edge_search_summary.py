"""Exploration summary for the Edge MoOP driving-setting search (plans/2026-11-15_a2-edge-moop-driving-search.md).

For every explored setting with all three criteria finished on Test500 and OOD-Val250:
  - per set and pooled over the 750 clips: mean minADE@6 at 6.4 s of MoOP / FM-only / CE-only (analyze_a100.metric),
    paired MoOP - FM-only and MoOP - CE-only with 95% bootstrap CIs (ci_openloop_lingoqa.summarise, 200,000 draws)
  - gap = max(MoOP - FM-only, MoOP - CE-only) on the pooled clips (negative = MoOP beats both halves)
Settings are ranked by gap with the pre-registered tie rule (within 0.005 m: lower sparsity, then fewer Q-head cuts).
Settings that lack OOD-Val (u40-l0keep) are listed but not eligible.

Usage: .venv/bin/python iclr27-rebuttal/code/edge_search_summary.py [--exp-id iclr27-rebuttal/edge_driving_search]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPO / "cvpr27" / "code"))
from analyze_a100 import load_rows, metric
from ci_openloop_lingoqa import summarise

OUT = REPO / "outputs" / "iclr27-rebuttal"
SETS = {"test500": REPO / "outputs" / "eval_sets" / "test_500.parquet", "oodval": OUT / "edge_moop" / "ood_val_250.parquet"}
# setting -> (arm suffix, sparsity pct, Q heads cut per layer)
SETTINGS = {"u40": ("", 40, 6), "u40-l0keep": ("_l0keep", 40, 6), "u40-q4": ("_q4", 40, 4), "u40-q2": ("_q2", 40, 2),
            "u50": ("_u50", 50, 8), "u50-q4": ("_q4_u50", 50, 4), "u50-q2": ("_q2_u50", 50, 2),
            "u60": ("_u60", 60, 10), "u60-q4": ("_q4_u60", 60, 4), "u60-q2": ("_q2_u60", 60, 2),
            "u70-q2": ("_q2_u70", 70, 2)}
CRIT = ("moop", "fm_only", "ce_only")


def rows_for(set_name, crit, sfx):
    d = OUT / f"edge_{set_name}_{crit}{sfx}"
    return load_rows(d, "edge") if d.exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="iclr27-rebuttal/edge_driving_search")
    args = ap.parse_args()
    out = REPO / "outputs" / args.exp_id
    out.mkdir(parents=True, exist_ok=True)
    dense = {s: rows_for(s, "dense", "") for s in SETS}
    ids = {s: list(pd.read_parquet(p)["clip_id"]) for s, p in SETS.items()}
    res = {}
    for name, (sfx, pct, nq) in SETTINGS.items():
        have = {s: all(len(rows_for(s, c, sfx)) == len(ids[s]) for c in CRIT) for s in SETS}
        if not have["test500"]:
            continue
        sets = [s for s in SETS if have[s]]
        series, absolute = {}, {}
        pooled = {c: [] for c in CRIT + ("dense",)}
        for s in sets:
            r = {c: rows_for(s, c, sfx) for c in CRIT}
            r["dense"] = dense[s]
            for c in CRIT + ("dense",):
                v = [metric(r[c][i], "ade", "6.4s") for i in ids[s]]
                absolute[f"{s}|{c}"] = float(np.mean(v))
                pooled[c] += v
            for b in ("fm_only", "ce_only"):
                series[f"{s}|moop-{b}"] = [metric(r["moop"][i], "ade", "6.4s") - metric(r[b][i], "ade", "6.4s") for i in ids[s]]
        st = {}
        for s in sets:  # boot_many needs aligned series: one call per set
            st.update(summarise({k: v for k, v in series.items() if k.startswith(f"{s}|")}))
        entry = {"pct": pct, "q_cut": nq, "sets": sets, "eligible": sets == list(SETS), "absolute": absolute,
                 "per_set": {k: {"mean": v["mean"], "ci": v["ci"], "p": v["p"]} for k, v in st.items()}}
        if entry["eligible"]:
            pst = summarise({f"pooled|moop-{b}": list(np.subtract(pooled["moop"], pooled[b])) for b in ("fm_only", "ce_only")})
            entry["pooled"] = {k: {"mean": v["mean"], "ci": v["ci"], "p": v["p"]} for k, v in pst.items()}
            entry["pooled_abs"] = {c: float(np.mean(v)) for c, v in pooled.items()}
            entry["gap"] = max(entry["pooled"]["pooled|moop-fm_only"]["mean"], entry["pooled"]["pooled|moop-ce_only"]["mean"])
        res[name] = entry
        print(f"  {name} done", flush=True)
    elig = sorted((k for k, v in res.items() if v["eligible"]), key=lambda k: res[k]["gap"])
    # pre-registered tie rule: within 0.005 m of the best gap -> lower sparsity, then fewer Q-head cuts
    if elig:
        best = res[elig[0]]["gap"]
        tied = [k for k in elig if res[k]["gap"] <= best + 0.005]
        pick = min(tied, key=lambda k: (res[k]["pct"], res[k]["q_cut"]))
    else:
        pick = None
    L = ["Edge MoOP driving-setting search: minADE@6 6.4 s (m), Test500 + OOD-Val250 (750 clips pooled)", "",
         f"{'setting':<12}{'dense':>7}{'MoOP':>7}{'FM':>7}{'CE':>7}   {'MoOP-FM [95% CI]':<28}{'MoOP-CE [95% CI]':<28}{'gap':>7}"]
    for k in elig:
        v = res[k]
        pf, pc = v["pooled"]["pooled|moop-fm_only"], v["pooled"]["pooled|moop-ce_only"]
        a = v["pooled_abs"]

        def f(e):
            s = "*" if e["ci"][0] > 0 or e["ci"][1] < 0 else ""
            return f"{e['mean']:+.3f} [{e['ci'][0]:+.3f},{e['ci'][1]:+.3f}]{s}"
        L.append(f"{k:<12}{a['dense']:>7.3f}{a['moop']:>7.3f}{a['fm_only']:>7.3f}{a['ce_only']:>7.3f}   {f(pf):<28}{f(pc):<28}"
                 f"{v['gap']:>+7.3f}")
    L += ["", "per set (MoOP - FM / MoOP - CE, mean [CI]):"]
    for k, v in res.items():
        L.append(f"  {k:<12}" + " | ".join(f"{s}: FM {v['per_set'][f'{s}|moop-fm_only']['mean']:+.3f} "
                                          f"[{v['per_set'][f'{s}|moop-fm_only']['ci'][0]:+.3f},{v['per_set'][f'{s}|moop-fm_only']['ci'][1]:+.3f}]"
                                          f" CE {v['per_set'][f'{s}|moop-ce_only']['mean']:+.3f} "
                                          f"[{v['per_set'][f'{s}|moop-ce_only']['ci'][0]:+.3f},{v['per_set'][f'{s}|moop-ce_only']['ci'][1]:+.3f}]"
                                          for s in v["sets"]) + ("" if v["eligible"] else "   (not eligible: Test500 only)"))
    L += ["", f"pick by the pre-registered rule: {pick}"
          + ("" if pick is None or res[pick]["gap"] < 0 else "  (gap >= 0: no setting beats both halves on the pooled mean)")]
    (out / "metrics.json").write_text(json.dumps({"settings": res, "ranking": elig, "pick": pick}, indent=2))
    (out / "summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
