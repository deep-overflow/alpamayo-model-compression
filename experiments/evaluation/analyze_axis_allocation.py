"""Axis-allocation verdict: 4 Q heads cut per layer instead of 13, same budget.

plans/2026-09-05_axis-allocation.md. Every arm shares the criterion (max11),
the calibration set (calib_100 via importance_v2_ada) and the removed-parameter
total (2,657,452,032); only the split between the two axes differs.
qcut4 was measured on cvlab20, whose baseline path matches this box bit for bit.

Pre-registered on test500 paired median dminADE@6 against maxstep11_u40_v2:
  |d| < 0.01  -> H1 (allocation does not matter)
  d < -0.01   -> H2 (Q heads are the expensive axis)
  d > +0.01   -> H3 (MLP width is the expensive axis)
0.01 is eight times the noise floor G0b measured (+0.0012, p=0.68).
"""
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

O = Path("/home/cvlab21/project/chan/alpamayo-model-compression/.claude/worktrees/"
         "calib-stability-curve/outputs")
K = 6


def load(tag):
    rows = []
    for f in sorted(glob.glob(str(O / tag / "*_s*of*.json"))):
        rows.extend(json.loads(Path(f).read_text()))
    return {r["clip_id"]: r for r in rows if "ade_rollout_k" in r}


def a6(r, key="ade_rollout_k"):
    return float(np.min(np.asarray(r[key], dtype=float)[:K]))


def paired(a, b, key="ade_rollout_k"):
    ids = sorted(set(a) & set(b))
    d = np.array([a6(a[i], key) - a6(b[i], key) for i in ids])
    rng = np.random.default_rng(0)
    boot = [np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(10000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return len(ids), float(np.median(d)), float(lo), float(hi), float(wilcoxon(d).pvalue)


arm = load("maxstep11_qcut4_test")
base = load("maxstep11_u40_v2_test")
unp = load("baseline_ada_ps_test")
dual = load("dual_u40_v2_ps_test")
print(f"rows: qcut4 {len(arm)}, maxstep11 {len(base)}, unpruned {len(unp)}")

print(f"\n{'arm':28s} {'minADE@6':>9s} {'minFDE@6':>9s} {'CoC붕괴':>8s}")
for name, r in (("무압축", unp), ("dual_u40_v2", dual),
                ("maxstep11_u40_v2 (헤드13)", base), ("qcut4 (헤드4)", arm)):
    a = np.array([a6(x) for x in r.values()])
    f = np.array([a6(x, "fde_rollout_k") for x in r.values()])
    deg = np.mean([x["coc_degenerate"] for x in r.values()])
    print(f"{name:28s} {a.mean():9.4f} {f.mean():9.4f} {deg:8.3f}")

print("\n=== 사전 등록 판정: qcut4 - maxstep11_u40_v2 ===")
n, med, lo, hi, p = paired(arm, base)
sig = "*" if (lo > 0 or hi < 0) else " "
print(f"  페어드 중앙값 {med:+.4f}{sig} [{lo:+.4f}, {hi:+.4f}]  p={p:.2g}  n={n}")
verdict = ("H1 채택 — 축 배분은 중요하지 않다" if abs(med) < 0.01 else
           "H2 채택 — Q head가 비싼 축이다" if med < 0 else
           "H3 채택 — MLP 폭이 비싼 축이다")
print(f"  -> {verdict}")

print("\n=== 참고: 무압축 대비 ===")
vs_unp = {}
for name, r in (("maxstep11", base), ("qcut4", arm), ("dual", dual)):
    n2, m2, l2, h2, p2 = paired(r, unp)
    sig = "*" if (l2 > 0 or h2 < 0) else " "
    print(f"  {name:20s} {m2:+.4f}{sig} [{l2:+.4f}, {h2:+.4f}]")
    vs_unp[name] = {"median": m2, "lo": l2, "hi": h2, "p": p2,
                    "sig": bool(l2 > 0 or h2 < 0)}

out = O / "axis_allocation"
out.mkdir(parents=True, exist_ok=True)
metrics = {
    "absolute": {k: {"minADE6": float(np.mean([a6(x) for x in r.values()])),
                     "minFDE6": float(np.mean([a6(x, "fde_rollout_k") for x in r.values()])),
                     "coc_degenerate": float(np.mean([x["coc_degenerate"]
                                                      for x in r.values()])),
                     "n": len(r)}
                 for k, r in (("baseline", unp), ("dual", dual),
                              ("maxstep11", base), ("qcut4", arm))},
    "vs_unpruned": vs_unp,
    "verdict": {"contrast": "qcut4 - maxstep11_u40_v2", "n": n, "median": med,
                "lo": lo, "hi": hi, "p": p, "sig": bool(lo > 0 or hi < 0),
                "threshold": 0.01, "hypothesis": verdict},
    "shape": {"baseline": {"heads_cut": 13, "channels_cut": 4898},
              "qcut4": {"heads_cut": 4, "channels_cut": 5666},
              "removed_params": 2657452032},
}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
print("\n->", out)
