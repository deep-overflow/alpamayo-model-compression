"""CPU-only kept-set comparison of score-combination operators on importance_v2.

Builds the u40_v2-budget keep sets (19/32 Q heads, 7390/12288 MLP channels per layer)
under max / rank-sum / rank-product / raw-sum / raw-product and reports overlap with
dual plus the rank profile of the units each rule trades away. This is the screening
analysis behind plans/2026-08-20_combination-operator-ablation.md: raw-sum collapses
onto traj-only (the traj scale is ~12x CoC), raw-product is ill-defined where traj
importance is structurally zero (L35), and the rank-space rules differ from max
exactly on semi-specialist units.

Usage:
  .venv/bin/python experiments/head_analysis/compare_combine_ops.py
"""

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
Z = np.load(REPO / "outputs" / "importance_v2" / "importance.npz")


def rank_norm(x):  # (n,) -> [0,1], higher = more important
    return np.argsort(np.argsort(x)) / (len(x) - 1)


def kept(score, n_keep):
    return set(np.argsort(score, kind="stable")[-n_keep:].tolist())


for name, n_keep, n_tot in (("vlm_q", 19, 32), ("vlm_mlp", 7390, 12288)):
    t, c = Z[f"traj_{name}"], Z[f"coc_{name}"]  # (36, n)
    tot = 36 * n_keep
    ov = {"sum": 0, "prod": 0, "raw_sum": 0, "raw_prod": 0}
    raw_sum_vs_traj = 0
    # rank profiles (better, worse) of units max keeps but the rule drops, and gains
    dropped_by = {"sum": [], "prod": []}
    gained_by = {"sum": [], "prod": []}
    scale_ratio = []
    for li in range(36):
        rt, rc = rank_norm(t[li]), rank_norm(c[li])
        kmax = kept(np.maximum(rt, rc), n_keep)
        rules = {
            "sum": kept(rt + rc, n_keep),
            "prod": kept(rt * rc, n_keep),
            "raw_sum": kept(t[li] + c[li], n_keep),
            "raw_prod": kept(t[li] * c[li], n_keep),
        }
        raw_sum_vs_traj += len(rules["raw_sum"] & kept(t[li], n_keep))
        scale_ratio.append(np.mean(t[li]) / np.mean(c[li]))
        for r, ks in rules.items():
            ov[r] += len(kmax & ks)
        for r in ("sum", "prod"):
            for u in kmax - rules[r]:
                dropped_by[r].append((max(rt[u], rc[u]), min(rt[u], rc[u])))
            for u in rules[r] - kmax:
                gained_by[r].append((max(rt[u], rc[u]), min(rt[u], rc[u])))
    print(f"== {name} (keep {n_keep}/{n_tot} per layer, {tot} total kept) ==")
    print(f"  raw scale ratio mean(traj)/mean(coc): median {np.median(scale_ratio):.3g}, "
          f"range [{min(scale_ratio):.3g}, {max(scale_ratio):.3g}]")
    for r in ("sum", "prod", "raw_sum", "raw_prod"):
        print(f"  kept-overlap max vs {r:8s}: {ov[r] / tot:6.1%}")
    print(f"  raw_sum overlap with traj-only: {raw_sum_vs_traj / tot:6.1%}")
    for r in ("sum", "prod"):
        d = np.array(dropped_by[r]).reshape(-1, 2)
        g = np.array(gained_by[r]).reshape(-1, 2)
        print(f"  max keeps, {r} drops: n={len(d)}  rank profile (better, worse) = "
              f"({d[:, 0].mean():.2f}, {d[:, 1].mean():.2f})")
        print(f"  {r} keeps, max drops: n={len(g)}  rank profile (better, worse) = "
              f"({g[:, 0].mean():.2f}, {g[:, 1].mean():.2f})")
    print()
