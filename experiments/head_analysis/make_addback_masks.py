"""Add-back masks for the dual-saves question in the PRUNED context
(plans/2026-09-24_dual-saves-trunk.md, part 2). Removing the "dual kept, single arm dropped" trunk
units from the dense model measures their function with every redundant partner intact; here the
same units are switched back ON inside the single-criterion arm, where the partners are gone:

  trajarm            the shipped trajectory-only arm (slim_traj_u40_v2 kept set)
  trajarm+Straj      plus the trunk units dual kept and it dropped   (= dual's trunk selection, union)
  trajarm+MStraj     plus, instead, units the arm dropped that are NOT in Straj, matched per unit
                     on the pooled I_traj (same count, same first-order trajectory score, lower I_CoC)
  trajarm+RStraj0-2  plus random units the arm dropped, same count per layer
  cocarm, cocarm+Scoc, cocarm+MScoc, cocarm+RScoc0-2   the mirror for the CoC-only arm

The controls are drawn from the arm's own removed units so that every add-back restores the same
number of units per layer. Straj / Scoc come from make_dual_saves_sets.py (outputs/<sets>/sets.npz,
keep masks); the scores from the dense calib_100 anatomy (outputs/gradanat_v1). Writes
outputs/<sets>/addback/<name>.npz with q_mask / mlp_mask keep masks for run_token_ablation.py
--arm-masks.

Usage:
  python experiments/head_analysis/make_addback_masks.py --sets tokabl_sets_trunk_v1 --band trunk21
"""

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SPANS = {"trunk21": (0, 22), "mid": (6, 22)}


def kept(arm, axis, n):
    meta = json.loads((REPO / "outputs" / f"slim_{arm}_u40_v2" / "slim_meta.json").read_text())
    out = np.zeros((36, n), bool)
    for li, e in enumerate(meta["vlm"]):
        out[li, e[axis]] = True
    return out


def random_from_pool(sizes, pool, span, rng):
    out = np.zeros_like(pool)
    for li in range(*span):
        cand = np.where(pool[li])[0]
        out[li, rng.choice(cand, size=int(sizes[li]), replace=False)] = True
    return out


def matched_from_pool(sel, pool, score, span):
    """For every unit of sel pick the pool unit with the nearest score in that layer (greedy,
    without replacement, along the score order)."""
    out = np.zeros_like(sel)
    for li in range(*span):
        order = np.argsort(score[li], kind="stable")
        pos = {u: i for i, u in enumerate(order)}
        free = pool[li][order].copy()
        for u in np.where(sel[li])[0]:
            p, best, d = pos[u], -1, 0
            while best < 0 and d < len(order):
                d += 1
                for q in (p - d, p + d):
                    if (0 <= q < len(order) and free[q]) and (
                        best < 0 or abs(score[li, order[q]] - score[li, u]) < abs(score[li, order[best]] - score[li, u])
                    ):
                        best = q
            out[li, order[best]] = True
            free[best] = False
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="tokabl_sets_trunk_v1")
    ap.add_argument("--band", default="trunk21")
    ap.add_argument("--anatomy", default="gradanat_v1")
    ap.add_argument("--n-random", type=int, default=3)
    args = ap.parse_args()
    span = SPANS[args.band]
    sets = np.load(REPO / "outputs" / args.sets / "sets.npz")
    z = np.load(REPO / "outputs" / args.anatomy / "anatomy.npz")
    out = REPO / "outputs" / args.sets / "addback"
    out.mkdir(parents=True, exist_ok=True)
    lines, meta = [], {}
    for arm, sname, key, trunk_only in (("traj", "Straj", "fm_full", False), ("coc", "Scoc", "ce_full", False),
                                        ("traj", "Straj", "fm_full", True), ("coc", "Scoc", "ce_full", True)):
        base = {ax: kept(arm, ax, n) for ax, n in (("q", 32), ("mlp", 12288))}
        tag = f"{arm}trunk" if trunk_only else f"{arm}arm"
        if trunk_only:  # the arm's trunk selection with every late layer intact (armheld_v1's band masks)
            for m in base.values():
                m[span[1]:] = True
        np.savez(out / f"{tag}.npz", q_mask=base["q"].astype(np.float32), mlp_mask=base["mlp"].astype(np.float32))
        lines.append(f"{tag}: kept q {int(base['q'].sum())} mlp {int(base['mlp'].sum())}")
        arm = tag
        adds = {}
        for ax in ("q", "mlp"):
            S = sets[f"{ax}_{args.band}_{sname}"] == 0  # the set's units (removed in the ablation)
            assert not (S & base[ax]).any(), "the set must lie in the arm's removed units"
            pool = ~base[ax] & ~S
            pool[: span[0]] = False
            pool[span[1]:] = False
            adds.setdefault(sname, {})[ax] = S
            adds.setdefault(f"M{sname}", {})[ax] = matched_from_pool(S, pool, z[f"{key}_{ax}"], span)
            for i in range(args.n_random):
                adds.setdefault(f"R{sname}{i}", {})[ax] = random_from_pool(S.sum(1), pool, span, np.random.default_rng([4000 + i, span[0], ax == "q"]))
        for name, m in adds.items():
            keep = {ax: base[ax] | m[ax] for ax in ("q", "mlp")}
            np.savez(out / f"{arm}+{name}.npz", q_mask=keep["q"].astype(np.float32), mlp_mask=keep["mlp"].astype(np.float32))
            meta[f"{arm}+{name}"] = {ax: {"added": int(m[ax].sum()), "added_I_traj_share": float((z[f"fm_full_{ax}"][span[0]:span[1]] * m[ax][span[0]:span[1]]).sum() / z[f"fm_full_{ax}"][span[0]:span[1]].sum()),
                                         "added_I_CoC_share": float((z[f"ce_full_{ax}"][span[0]:span[1]] * m[ax][span[0]:span[1]]).sum() / z[f"ce_full_{ax}"][span[0]:span[1]].sum())} for ax in ("q", "mlp")}
            mm = meta[f"{arm}+{name}"]
            lines.append(f"{arm}+{name:8s}: +q {mm['q']['added']:3d} (band I_traj {mm['q']['added_I_traj_share']:.3f} I_CoC {mm['q']['added_I_CoC_share']:.3f}) "
                         f"+mlp {mm['mlp']['added']:6d} (band I_traj {mm['mlp']['added_I_traj_share']:.3f} I_CoC {mm['mlp']['added_I_CoC_share']:.3f})")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    (out / "meta.json").write_text(json.dumps({"sets": args.sets, "band": args.band, "anatomy": args.anatomy, "configs": meta}, indent=1))
    print("\n".join(lines))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
