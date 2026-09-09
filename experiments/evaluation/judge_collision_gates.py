"""G2 and G3 of plans/2026-09-08_openloop-collision-proxy.md.

G2 -- is the collision proxy new information, or minADE in another form?
G3 -- does it separate arms that minADE cannot?

The pre-registered G2 is an arm-level Spearman between collision rate and minADE with
|rho| < 0.7. With four arms that statistic has essentially no power (|rho| = 0.8 at n = 4
is p = 0.33), so it is reported and then set aside in favour of two readings the data can
actually support:

  per-clip   within one arm, does the clip that collides also have the worse minADE?
             n ~ 489, so this is answerable.
  paired     between two arms, are the clips whose collision outcome flips the same clips
             whose minADE moves? If the proxy were minADE in disguise they would coincide.

G3 is a paired McNemar on `collide_any` for every arm pair, set beside the same pair's
minADE Wilcoxon. A pair that minADE calls indistinguishable while collisions separate it
is the whole point of building this.
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

O = Path("/mnt/nvme1n1/ad_vla/outputs/chan")
# @6 counts SAMPLES, not seconds. The stored `minADE_rollout` is the min over all 8
# and is off-protocol; the frozen protocol is the min over the first 6.
K = 6
SETS = ["test500", "val500", "OOD-val"]
SUFFIX = {"test500": "test", "val500": "indist", "OOD-val": "oodval"}


def load_collision(arm):
    p = O / f"{arm}_collision" / "metrics.json"
    return json.loads(p.read_text())["sets"] if p.exists() else None


def load_minade(arm, suffix):
    d = O / f"{arm}_{suffix}"
    rows = {}
    for f in sorted(d.glob("*_s*of*.json")):
        for r in json.loads(f.read_text()):
            rows[r["clip_id"]] = min(r["ade_rollout_k"][:K])
    return rows


def mcnemar(a, b):
    from scipy.stats import binomtest

    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    if n10 + n01 == 0:
        return 1.0, n10, n01
    return float(binomtest(n10, n10 + n01, 0.5).pvalue), n10, n01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True,
                    help="collision exp-id stems, e.g. baseline_pred dual_u40_v2_pred")
    ap.add_argument("--out", type=Path, default=O / "collision_gates")
    args = ap.parse_args()

    from scipy.stats import spearmanr, wilcoxon

    coll = {a: load_collision(a) for a in args.arms}
    missing = [a for a, v in coll.items() if v is None]
    if missing:
        raise SystemExit(f"채점 결과 없음: {missing}")

    report = {"arms": args.arms, "sets": {}}
    for s in SETS:
        if any(s not in coll[a] for a in args.arms):
            continue
        # per-clip tables, restricted to clips every arm scored
        per = {a: {r["clip_id"]: r for r in coll[a][s]["rows"]} for a in args.arms}
        ade = {a: load_minade(a, SUFFIX[s]) for a in args.arms}
        clips = sorted(set.intersection(*(set(per[a]) for a in args.arms)))
        e = {"n": len(clips)}

        # ---- G2a: arm-level, pre-registered but underpowered at n=4 -------------------
        cr = [100 * np.mean([per[a][c]["collide_any"] for c in clips]) for a in args.arms]
        mr = [float(np.mean([ade[a][c] for c in clips])) for a in args.arms]
        rho, prho = spearmanr(cr, mr)
        e["arm_level"] = {"collide_pct": cr, "minADE": mr,
                          "spearman_rho": float(rho), "p": float(prho),
                          "n_arms": len(args.arms)}

        # ---- G2b: per-clip, within each arm -------------------------------------------
        e["per_clip"] = {}
        for a in args.arms:
            hit = np.array([per[a][c]["collide_any"] for c in clips], bool)
            m = np.array([ade[a][c] for c in clips], float)
            r, p = spearmanr(hit.astype(float), m)
            e["per_clip"][a] = {"rho": float(r), "p": float(p),
                                "minADE_hit": float(m[hit].mean()) if hit.any() else None,
                                "minADE_clean": float(m[~hit].mean())}

        # ---- G3: every pair, collision vs minADE --------------------------------------
        e["pairs"] = {}
        for x, y in combinations(args.arms, 2):
            hx = [per[x][c]["collide_any"] for c in clips]
            hy = [per[y][c]["collide_any"] for c in clips]
            pc, n10, n01 = mcnemar(hx, hy)
            d = np.array([ade[y][c] - ade[x][c] for c in clips], float)
            dnz = d[d != 0]
            pa = float(wilcoxon(dnz).pvalue) if len(dnz) >= 5 else float("nan")
            e["pairs"][f"{y} - {x}"] = {
                "collide_pct": [100 * np.mean(hx), 100 * np.mean(hy)],
                "collide_p": pc, "n10": n10, "n01": n01,
                "minADE_delta": float(d.mean()), "minADE_p": pa,
                "collision_separates": pc < 0.05, "minade_separates": pa < 0.05,
            }
        report["sets"][s] = e

    # ---- print -----------------------------------------------------------------------
    for s, e in report["sets"].items():
        al = e["arm_level"]
        print(f"\n=== {s}  (n={e['n']}) ===")
        print("  arm별 충돌률 / minADE")
        for a, c, m in zip(report["arms"], al["collide_pct"], al["minADE"]):
            print(f"    {a:24s} {c:5.1f}%   {m:.4f}")
        print(f"  G2a arm 수준 Spearman rho={al['spearman_rho']:+.3f} "
              f"(p={al['p']:.2f}, n={al['n_arms']} — 검정력 없음, 참고용)")
        print("  G2b 클립 수준 (충돌 여부 vs minADE)")
        for a, v in e["per_clip"].items():
            print(f"    {a:24s} rho={v['rho']:+.3f} p={v['p']:.1e}   "
                  f"충돌 클립 minADE {v['minADE_hit']:.3f} vs 무충돌 {v['minADE_clean']:.3f}")
        print("  G3 쌍 비교")
        for k, v in e["pairs"].items():
            tag = ("충돌만 구분" if v["collision_separates"] and not v["minade_separates"]
                   else "둘 다 구분" if v["collision_separates"]
                   else "minADE만 구분" if v["minade_separates"] else "둘 다 못 구분")
            print(f"    {k:44s} 충돌 {v['collide_pct'][0]:.1f}->{v['collide_pct'][1]:.1f}% "
                  f"p={v['collide_p']:.1e}  minADE Δ{v['minADE_delta']:+.4f} "
                  f"p={v['minADE_p']:.3f}  [{tag}]")

    t = report["sets"].get("test500")
    if t:
        g2 = "PASS" if abs(t["arm_level"]["spearman_rho"]) < 0.7 else "FAIL"
        n_new = sum(1 for v in t["pairs"].values()
                    if v["collision_separates"] and not v["minade_separates"])
        g3 = "PASS" if n_new > 0 else "FAIL"
        print(f"\nG2 {g2} (test500 arm 수준 |rho|<0.7 — n=4라 참고용)")
        print(f"G3 {g3} — minADE가 못 가르는데 충돌이 가르는 쌍 {n_new}개")
        report["G2"], report["G3"] = g2, g3
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
