"""GH200 site gates (cvpr27/plans/2026-11-05_kisti6-gh200-site.md §2.6) on test_2000 shard 0/16 (125 clips).

G1  dense twice on GH200: every rollout list (ade/fde at 1.6/3.2/6.4 s) and the CoC text bit-identical.
G2  dense GH200 vs A100 on the same clips: recorded, no pass bar (one GPU type per table).
G3  MoOP l0keep (selection only) and tyr_off (slim_state.pt) load and land at A100's level: recorded the same way.
A100's own dense rerun on the same clips is printed as the reference for G1.

    python cvpr27/code/analyze_gh200_gate.py      # reads outputs/cvpr27/{gh200,a100}, writes outputs/cvpr27/gh200/gate
"""
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "cvpr27"
HORIZONS = {"1.6s": "_h16", "3.2s": "_h32", "6.4s": ""}
SAME_FIELDS = [f"{m}_rollout_k{h}" for h in HORIZONS.values() for m in ("ade", "fde")] + ["gen_coc"]
K = 6  # @6 = min over the first 6 of the 8 stored samples, as in analyze_a100.py
ARMS = {"dense": "baseline", "moop_l0keep": "slim_dual_u40_l0keep_v2", "tyr_off": "slim_tyr_off"}


def load(d, tag):
    rows = {}
    for p in sorted(d.glob(f"{tag}_s*of*.json")):
        for r in json.loads(p.read_text()):
            rows[r["clip_id"]] = r
    return rows


def metric(r, kind, h):
    return float(min(r[f"{kind}_rollout_k{HORIZONS[h]}"][:K]))


def identical(a, b):
    ids = sorted(set(a) & set(b))
    return len(ids), sum(all(a[c][f] == b[c][f] for f in SAME_FIELDS) for c in ids)


res, lines = {}, []
g_dense = load(OUT / "gh200" / "dense_test2000", "baseline")
n, same = identical(g_dense, load(OUT / "gh200" / "dense_test2000_rerun", "baseline"))
res["G1"] = {"n": n, "bit_identical": same, "pass": n == 125 and same == n}
lines.append(f"G1 {'PASS' if res['G1']['pass'] else 'FAIL'}: GH200 dense twice, {same}/{n} clips bit-identical "
             f"(rollout lists at 1.6/3.2/6.4 s + CoC text)")
a_dense = load(OUT / "a100" / "dense_test2000", "baseline")
a_rerun = {c: r for c, r in load(OUT / "a100" / "dense_test2000_rerun", "baseline").items() if c in g_dense}
n, same = identical({c: a_dense[c] for c in a_rerun}, a_rerun)
res["A100_rerun_reference"] = {"n": n, "bit_identical": same}
lines.append(f"   reference: A100 dense twice on the same clips present in its rerun: {same}/{n} bit-identical")

lines.append("")
lines.append("| arm | n | metric | GH200 | A100 | GH200 - A100 | mean abs diff | clips equal |")
lines.append("|---|---|---|---|---|---|---|---|")
for arm, tag in ARMS.items():
    g = load(OUT / "gh200" / f"{arm}_test2000", tag)
    a = load(OUT / "a100" / f"{arm}_test2000", tag)
    ids = sorted(set(g) & set(a))
    if not ids:
        lines.append(f"| {arm} | 0 | (no rows) | | | | | |")
        continue
    ent = {"n": len(ids), "coc_changed": float(np.mean([g[c]["gen_coc"] != a[c]["gen_coc"] for c in ids]))}
    for kind in ("ade", "fde"):
        for h in HORIZONS:
            gv = np.array([metric(g[c], kind, h) for c in ids])  # (n,)
            av = np.array([metric(a[c], kind, h) for c in ids])  # (n,)
            d = gv - av
            name = f"min{kind.upper()}@6 {h}"
            ent[name] = {"gh200": float(gv.mean()), "a100": float(av.mean()), "delta": float(d.mean()),
                         "mean_abs": float(np.abs(d).mean()), "equal_frac": float((d == 0).mean())}
            e = ent[name]
            lines.append(f"| {arm} | {len(ids)} | {name} | {e['gh200']:.4f} | {e['a100']:.4f} | {e['delta']:+.4f} | "
                         f"{e['mean_abs']:.4f} | {e['equal_frac']:.0%} |")
    res[arm] = ent
    lines.append(f"| {arm} | {len(ids)} | CoC text changed | | | | | {ent['coc_changed']:.0%} of clips |")

dst = OUT / "gh200" / "gate"
dst.mkdir(parents=True, exist_ok=True)
(dst / "metrics.json").write_text(json.dumps(res, indent=1))
(dst / "summary.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
