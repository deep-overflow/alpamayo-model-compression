"""Alpamayo 2 Super u55-q8 (8 Q heads + 16,384 MLP channels per layer, 46.9% removed): MoOP vs FM-only vs CE-only, Blackwell.

Plan: iclr27-rebuttal/plans/2026-11-14_a2-u55-q8-blackwell.md (pre-registered). Per set (test500, OOD-Val262): minADE@6 / minFDE@6
at 1.6 / 3.2 / 6.4 s, CoC degeneracy, paired deltas. Primary test: pooled test500 + OOD-Val (762 clips), minADE@6 6.4 s,
H1 MoOP < FM-only and H2 MoOP < CE-only, bootstrap over clips (200,000 draws), Holm over the two.

Usage: .venv/bin/python iclr27-rebuttal/code/analyze_a2_u55q8.py [--partial]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "cvpr27" / "code"))
from analyze_early_mlp_openloop import at6, rows

R = "iclr27-rebuttal"
SETS = {"test500": {"dense": f"{R}/a2_test500_dense_bw"}, "oodval": {"dense": f"{R}/a2_oodval_dense_bw"}}
for _s, _arms in SETS.items():
    _arms.update({k: f"{R}/a2_{_s}_{k}_u55_q8_bw" for k in ("moop", "fm_only", "ce_only")})
HZ = {"1.6s": "_h16", "3.2s": "_h32", "6.4s": ""}


def boot(d, n=200_000, seed=0, chunk=10_000):
    rng = np.random.default_rng(seed)
    ms = []
    for s in range(0, n, chunk):
        idx = rng.integers(0, len(d), (min(chunk, n - s), len(d)))
        ms.append(d[idx].mean(1))
    m = np.concatenate(ms)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float(min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean())))


def holm(ps):
    order = np.argsort(ps)
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true", help="use the clips every arm has so far")
    ap.add_argument("--exp-id", default=f"{R}/a2_u55q8_analysis")
    args = ap.parse_args()
    out = REPO / "outputs" / args.exp_id
    out.mkdir(parents=True, exist_ok=True)
    res, L, pooled = {"sets": {}}, ["Alpamayo 2 Super u55-q8 (46.9% removed), Blackwell"], {}
    for s, arms in SETS.items():
        data = {k: rows(v) for k, v in arms.items()}
        if not all(data.values()):
            L.append(f"\n{s}: missing {[k for k, v in data.items() if not v]}")
            continue
        ids = sorted(set.intersection(*(set(v) for v in data.values())))
        if not args.partial:
            assert len(ids) == {"test500": 500, "oodval": 262}[s], (s, len(ids))
        v = {k: {f"{m}_{h}": at6(r, ids, f"{m}_rollout_k{sfx}") for m in ("ade", "fde") for h, sfx in HZ.items()}
             for k, r in data.items()}  # arm -> metric -> (n,)
        degen = {k: float(np.mean([r[c]["coc_degenerate"] for c in ids])) for k, r in data.items()}
        res["sets"][s] = {"n": len(ids), "mean": {k: {m: float(x.mean()) for m, x in vv.items()} for k, vv in v.items()},
                          "degen": degen, "pairs": {}}
        L += ["", f"{s} ({len(ids)} clips)", f"  {'arm':<9}{'minADE@6 1.6/3.2/6.4':>24}{'minFDE@6 1.6/3.2/6.4':>24}{'degen':>8}"]
        for k in arms:
            mm = res["sets"][s]["mean"][k]
            L.append(f"  {k:<9}{mm['ade_1.6s']:>8.3f}{mm['ade_3.2s']:>8.3f}{mm['ade_6.4s']:>8.3f}{mm['fde_1.6s']:>8.3f}"
                     f"{mm['fde_3.2s']:>8.3f}{mm['fde_6.4s']:>8.3f}{degen[k] * 100:>7.1f}%")
        for b in ("fm_only", "ce_only", "dense"):
            d = v["moop"]["ade_6.4s"] - v[b]["ade_6.4s"]
            lo, hi, p = boot(d)
            res["sets"][s]["pairs"][f"moop - {b}"] = {"mean": float(d.mean()), "ci": [lo, hi], "p": p}
            L.append(f"  moop - {b:<8} minADE@6 6.4 s {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] p {p:.2g}")
            if b != "dense":
                pooled.setdefault(b, []).append(d)
    if len(pooled.get("fm_only", [])) == 2 or args.partial:
        prim = {}
        for b, ds in pooled.items():
            d = np.concatenate(ds)
            lo, hi, p = boot(d)
            prim[f"moop - {b}"] = {"n": len(d), "mean": float(d.mean()), "ci": [lo, hi], "p": p}
        for k, padj in zip(prim, holm([x["p"] for x in prim.values()]), strict=True):
            prim[k]["p_holm"] = padj
        res["primary_pooled"] = prim
        L += ["", "PRIMARY (pooled test500 + OOD-Val, minADE@6 6.4 s, Holm over 2):"]
        for k, x in prim.items():
            L.append(f"  {k:<16} {x['mean']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}] p {x['p']:.2g} p_holm {x['p_holm']:.2g} (n {x['n']})")
        ok = all(x["p_holm"] < 0.05 and x["mean"] < 0 for x in prim.values()) and len(prim) == 2
        L.append(f"  confirmed (both p_holm < 0.05, MoOP better): {ok}")
        res["confirmed"] = ok
    (out / "config.json").write_text(json.dumps({"sets": SETS, "plan": "iclr27-rebuttal/plans/2026-11-14_a2-u55-q8-blackwell.md",
                                                 "partial": args.partial}, indent=2))
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    (out / "summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
