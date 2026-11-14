"""Pre-registered multi-objective test: does u50-q8 MoOP beat both FM-only and CE-only (better on one objective, not worse on the
other), on data never used before? Plan: iclr27-rebuttal/plans/2026-11-14_a2-pareto-q8.md.

Driving: OOD-train1271 (Blackwell), minADE@6 at 6.4 s, paired per clip, bootstrap over clips (200,000 draws).
Language: LingoQA train sample 3 (seed 20261116, segments disjoint from samples 1-2), Lingo-Judge accuracy, paired per question,
bootstrap over segments (200,000 draws).
Criteria (95% CIs; all four must hold -- an intersection-union test, no multiplicity correction needed):
  D-FM  driving  MoOP - FM-only   upper bound < +0.05 m   (non-inferior)
  L-FM  language MoOP - FM-only   lower bound > 0         (superior)
  D-CE  driving  MoOP - CE-only   upper bound < 0         (superior)
  L-CE  language MoOP - CE-only   lower bound > -2.0 pp   (non-inferior)

Usage: .venv/bin/python iclr27-rebuttal/code/analyze_a2_pareto_q8.py
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "cvpr27" / "code"))
from analyze_early_mlp_openloop import at6, rows

R = REPO / "outputs" / "iclr27-rebuttal"
ARMS = ("moop", "fm_only", "ce_only")
D_MARGIN, L_MARGIN = 0.05, -0.02


def boot(d, groups=None, n=200_000, seed=0, chunk=10_000):
    if groups is None:
        groups = np.arange(len(d))
    keys, inv = np.unique(groups, return_inverse=True)
    S = np.bincount(inv, weights=d, minlength=len(keys))
    C = np.bincount(inv, minlength=len(keys)).astype(float)
    rng = np.random.default_rng(seed)
    ms = []
    for s in range(0, n, chunk):
        idx = rng.integers(0, len(keys), (min(chunk, n - s), len(keys)))
        ms.append(S[idx].sum(1) / C[idx].sum(1))
    m = np.concatenate(ms)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    out = R / "a2_pareto_q8_analysis"
    out.mkdir(parents=True, exist_ok=True)
    drv = {a: rows(f"iclr27-rebuttal/a2_oodtrain_{a}_u50_q8_bw") for a in ARMS}
    ids = sorted(set.intersection(*(set(v) for v in drv.values())))
    assert len(ids) == 1271, len(ids)
    dv = {a: at6(r, ids, "ade_rollout_k") for a, r in drv.items()}  # (1271,)
    lang = {}
    for a in ARMS:
        sc = json.loads((R / f"a2_lingotrain3_{a}_u50_q8" / "scored.json").read_text())
        lang[a] = {(x["question_id"], x["segment_id"]): float(x["correct"]) for x in sc}
    keys = sorted(set.intersection(*(set(v) for v in lang.values())))
    assert len(keys) == 2000, len(keys)
    seg = np.array([k[1] for k in keys])
    lv = {a: np.array([lang[a][k] for k in keys]) for a in ARMS}

    res = {"n_clips": len(ids), "n_questions": len(keys), "n_segments": len(set(seg)),
           "driving_mean": {a: float(v.mean()) for a, v in dv.items()},
           "language_acc": {a: float(v.mean()) for a, v in lv.items()},
           "degen": {a: float(np.mean([drv[a][c]["coc_degenerate"] for c in ids])) for a in ARMS}, "criteria": {}}
    for b in ("fm_only", "ce_only"):
        d = dv["moop"] - dv[b]
        dlo, dhi = boot(d)
        l = lv["moop"] - lv[b]
        llo, lhi = boot(l, seg)
        res["criteria"][f"D-{b}"] = {"mean": float(d.mean()), "ci": [dlo, dhi]}
        res["criteria"][f"L-{b}"] = {"mean": float(l.mean()), "ci": [llo, lhi]}
    c = res["criteria"]
    checks = {"D-FM non-inferior (upper < +0.05 m)": c["D-fm_only"]["ci"][1] < D_MARGIN,
              "L-FM superior (lower > 0)": c["L-fm_only"]["ci"][0] > 0,
              "D-CE superior (upper < 0)": c["D-ce_only"]["ci"][1] < 0,
              "L-CE non-inferior (lower > -2 pp)": c["L-ce_only"]["ci"][0] > L_MARGIN}
    res["checks"] = checks
    res["confirmed"] = all(checks.values())
    (out / "config.json").write_text(json.dumps({
        "plan": "iclr27-rebuttal/plans/2026-11-14_a2-pareto-q8.md", "driving": "OOD-train1271 Blackwell, minADE@6 6.4 s",
        "language": "LingoQA train sample 3 (seed 20261116, disjoint), Lingo-Judge", "margins": {"driving_m": D_MARGIN,
                                                                                              "language": L_MARGIN},
        "bootstrap": "200,000 draws, clips (driving) / segments (language)"}, indent=2))
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    L = ["u50-q8 multi-objective test (pre-registered, fresh data)",
         "  driving OOD-train1271 minADE@6 6.4 s: " + ", ".join(f"{a} {v:.3f}" for a, v in res["driving_mean"].items()),
         "  CoC degeneracy: " + ", ".join(f"{a} {v * 100:.1f}%" for a, v in res["degen"].items()),
         f"  language LingoQA train-3 (2000 q / {res['n_segments']} seg): "
         + ", ".join(f"{a} {v * 100:.1f}%" for a, v in res["language_acc"].items()), ""]
    for k, v in c.items():
        unit = 100 if k.startswith("L") else 1
        L.append(f"  {k:<10} MoOP - X {v['mean'] * unit:+.3f} [{v['ci'][0] * unit:+.3f}, {v['ci'][1] * unit:+.3f}]"
                 f"{' pp' if unit == 100 else ' m'}")
    L += [""] + [f"  {'PASS' if ok else 'FAIL'}  {k}" for k, ok in checks.items()]
    L.append(f"  CONFIRMED (all four): {res['confirmed']}")
    (out / "summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
