"""Confirmatory test of the Edge MoOP driving-setting search on OOD-train1178 (plans/2026-11-15_a2-edge-moop-driving-search.md).

Pre-registered primary: minADE@6 at 6.4 s, paired per clip, MoOP - FM-only and MoOP - CE-only of the picked setting; 200,000-draw
bootstrap (ci_openloop_lingoqa.boot_many, seed 0; seed 1 stored too), 95% two-sided CIs, Holm over the 2 contrasts.
Success = both Holm-adjusted p < 0.05 with negative means; partial = one. Secondary: minFDE@6 6.4 s, minADE@6 1.6 / 3.2 s,
each arm vs dense, bucket means, CoC readouts.

Usage: .venv/bin/python iclr27-rebuttal/code/analyze_edge_confirm.py --setting-suffix _u50 --name u50
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
from analyze_a100 import HORIZONS, load_rows, metric
from ci_openloop_lingoqa import summarise

OUT = REPO / "outputs" / "iclr27-rebuttal"
MANIFEST = OUT / "edge_moop" / "ood_train_1178.parquet"


def holm(ps):
    """Holm step-down adjusted p-values for a dict {name: p}."""
    names = sorted(ps, key=ps.get)
    m, out, run = len(names), {}, 0.0
    for i, k in enumerate(names):
        run = max(run, min(1.0, (m - i) * ps[k]))
        out[k] = run
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting-suffix", required=True, help="arm suffix of the picked setting, e.g. _u50")
    ap.add_argument("--name", required=True)
    ap.add_argument("--exp-id", default="iclr27-rebuttal/edge_driving_confirm")
    args = ap.parse_args()
    out = REPO / "outputs" / args.exp_id
    out.mkdir(parents=True, exist_ok=True)
    ids = list(pd.read_parquet(MANIFEST)["clip_id"])
    arms = {"dense": "dense", "moop": f"moop{args.setting_suffix}", "fm_only": f"fm_only{args.setting_suffix}",
            "ce_only": f"ce_only{args.setting_suffix}"}
    rows = {k: load_rows(OUT / f"edge_oodtrain_{v}", "edge") for k, v in arms.items()}
    for k, r in rows.items():
        missing = [c for c in ids if c not in r]
        assert not missing, f"{k}: {len(missing)} of {len(ids)} clips missing"
    absolute, paired = {}, {}
    for kind in ("ade", "fde"):
        for h in HORIZONS:
            for k in arms:
                absolute[f"{k}|{kind}|{h}"] = [metric(rows[k][c], kind, h) for c in ids]
            for a, b in (("moop", "fm_only"), ("moop", "ce_only"), ("fm_only", "ce_only"), ("moop", "dense"),
                         ("fm_only", "dense"), ("ce_only", "dense")):
                paired[f"{a}|{b}|{kind}|{h}"] = list(np.subtract(absolute[f"{a}|{kind}|{h}"], absolute[f"{b}|{kind}|{h}"]))
    sa, sp = summarise(absolute), summarise(paired)
    prim = {k: sp[f"moop|{k}|ade|6.4s"] for k in ("fm_only", "ce_only")}
    adj = holm({k: v["p"] for k, v in prim.items()})
    adj1 = holm({k: v["p_seed1"] for k, v in prim.items()})
    wins = {k: bool(prim[k]["mean"] < 0 and adj[k] < 0.05) for k in prim}
    verdict = "SUCCESS" if all(wins.values()) else ("PARTIAL" if any(wins.values()) else "NOT CONFIRMED")
    coc = {k: {"nominal": float(np.mean([rows[k][c]["nominal"] for c in ids])),
               "degenerate": float(np.mean([rows[k][c]["coc_degenerate"] for c in ids])),
               "gen_len_mean": float(np.mean([rows[k][c]["gen_len"] for c in ids]))} for k in arms}
    buckets = {}
    for c in ids:
        buckets.setdefault(rows["dense"][c]["bucket"], []).append(c)
    by_bucket = {bk: {"n": len(cs), **{k: float(np.mean([metric(rows[k][c], "ade", "6.4s") for c in cs])) for k in arms}}
                 for bk, cs in buckets.items()}
    res = {"setting": args.name, "suffix": args.setting_suffix, "n": len(ids), "verdict": verdict,
           "primary": {k: {"mean": v["mean"], "ci": v["ci"], "p": v["p"], "p_holm": adj[k], "p_seed1": v["p_seed1"],
                           "p_holm_seed1": adj1[k], "ci_seed1": v["ci_seed1"], "win": wins[k]} for k, v in prim.items()},
           "absolute": sa, "paired": sp, "coc": coc, "ade6_by_bucket": by_bucket}
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    L = [f"Confirmatory test on OOD-train1178 -- setting {args.name} ({args.setting_suffix}); Ada", "",
         "Primary (minADE@6 6.4 s, paired, Holm over 2):"]
    for k, v in res["primary"].items():
        L.append(f"  MoOP - {k}: {v['mean']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]  p {v['p']:.3g}  p_holm {v['p_holm']:.3g}"
                 f"  (seed 1 p_holm {v['p_holm_seed1']:.3g})  -> {'win' if v['win'] else 'no'}")
    L += [f"VERDICT: {verdict}", "", "absolute minADE@6 / minFDE@6 (mean, median):"]
    for k in arms:
        L.append(f"  {k:<8}" + "  ".join(f"ADE {h} {sa[f'{k}|ade|{h}']['mean']:.3f}" for h in HORIZONS)
                 + f"  FDE 6.4s {sa[f'{k}|fde|6.4s']['mean']:.3f}  median ADE 6.4s {sa[f'{k}|ade|6.4s']['median']:.3f}  CoC {coc[k]}")
    L += ["", "secondary paired (mean [CI]):"]
    for key in [f"{a}|{b}|{kind}|{h}" for a, b in (("moop", "fm_only"), ("moop", "ce_only"), ("fm_only", "ce_only"),
                                                    ("moop", "dense"), ("fm_only", "dense"), ("ce_only", "dense"))
                for kind, h in (("ade", "1.6s"), ("ade", "3.2s"), ("ade", "6.4s"), ("fde", "6.4s"))]:
        e = sp[key]
        s = "*" if e["ci"][0] > 0 or e["ci"][1] < 0 else ""
        L.append(f"  {key:<32} {e['mean']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]{s} p {e['p']:.3g}")
    L += ["", f"minADE@6 6.4 s by bucket: {json.dumps({k: {a: round(x, 3) for a, x in v.items()} for k, v in by_bucket.items()})}"]
    (out / "summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
