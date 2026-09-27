"""Unit sets for the trunk-band "what the dual criterion saves" ablation
(plans/2026-09-24_dual-saves-trunk.md). Companion of make_token_sets.py, whose S-coc / S-traj
sets exist only for the late band; here the same sets are built for the trunk (layers 0-21)
and, as a robustness variant, layers 6-21, each with two controls:

  Straj    units the trajectory-only arm removed and the dual arm kept   (high I_CoC, low I_traj)
  Scoc     units the CoC-only arm removed and the dual arm kept          (high I_traj, low I_CoC)
  RStraj0-2 / RScoc0-2   random units, the same count per layer (size-matched null)
  MStraj   for every Straj unit, the unit not in Straj with the nearest pooled I_traj in that
           layer (same first-order trajectory score, lower I_CoC): the score-matched control
  MScoc    the same for Scoc, matched on pooled I_CoC

Per-axis configs (Q heads or MLP channels alone) go to sets.npz / sets.json in the format
run_token_ablation.py reads; joint configs (both axes at once, as in the shipped arms) go to
joint/<name>.npz as q_mask / mlp_mask keep masks for run_token_ablation.py --arm-masks.
Scores: the dense model's calib_100 anatomy (outputs/gradanat_v1, pooled full-seed gradients =
the shipped scores); kept sets: outputs/slim_{traj,coc,dual}_u40_v2/slim_meta.json.

Usage:
  python experiments/head_analysis/make_dual_saves_sets.py --exp-id tokabl_sets_trunk_v1
"""

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BANDS = {"trunk21": (0, 22), "mid": (6, 22)}
N_RANDOM = 3


def kept(arm, axis, n):
    meta = json.loads((REPO / "outputs" / f"slim_{arm}_u40_v2" / "slim_meta.json").read_text())
    out = np.zeros((36, n), bool)
    for li, e in enumerate(meta["vlm"]):
        out[li, e[axis]] = True
    return out


def in_band(mask, span):
    out = np.zeros_like(mask)
    out[span[0]:span[1]] = mask[span[0]:span[1]]
    return out


def random_like(sizes, n, span, rng):
    rem = np.zeros((36, n), bool)
    for li in range(*span):
        rem[li, rng.choice(n, size=int(sizes[li]), replace=False)] = True
    return rem


def matched(rem, score, span):
    """For every removed unit pick the not-removed unit with the nearest score in that layer
    (greedy without replacement, along the layer's score order, nearest free neighbour)."""
    out = np.zeros_like(rem)
    for li in range(*span):
        order = np.argsort(score[li], kind="stable")
        is_rem = rem[li][order]
        free = ~is_rem  # candidates: not removed, not yet taken
        for p in np.where(is_rem)[0]:
            best, d = -1, 0
            while best < 0 and d < len(order):
                d += 1
                for q in (p - d, p + d):
                    if (0 <= q < len(order) and free[q]) and (
                        best < 0 or abs(score[li, order[q]] - score[li, order[p]])
                        < abs(score[li, order[best]] - score[li, order[p]])
                    ):
                        best = q
            out[li, order[best]] = True
            free[best] = False
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="tokabl_sets_trunk_v1")
    ap.add_argument("--anatomy", default="gradanat_v1")
    args = ap.parse_args()
    z = np.load(REPO / "outputs" / args.anatomy / "anatomy.npz")
    out = REPO / "outputs" / args.exp_id
    (out / "joint").mkdir(parents=True, exist_ok=True)
    masks, meta, joint = {}, {}, {}
    for axis in ("q", "mlp"):
        t_pool, c_pool = z[f"fm_full_{axis}"], z[f"ce_full_{axis}"]  # (L, U)
        n = t_pool.shape[1]
        dual, traj, coc = (kept(a, axis, n) for a in ("dual", "traj", "coc"))
        for band, span in BANDS.items():
            base = {"Straj": in_band(dual & ~traj, span), "Scoc": in_band(dual & ~coc, span)}
            removed = {}
            for name, rem in base.items():
                removed[name] = rem
                for i in range(N_RANDOM):
                    removed[f"R{name}{i}"] = random_like(
                        rem.sum(1), n, span, np.random.default_rng([3000 + i, span[0], n, len(name)]))
                removed[f"M{name}"] = matched(rem, t_pool if name == "Straj" else c_pool, span)
            for name, rem in removed.items():
                cfg = f"{axis}_{band}_{name}"
                masks[cfg] = (~rem).astype(np.uint8)
                lo, hi = span
                meta[cfg] = {
                    "axis": axis, "band": band, "set": name, "layers": [lo, hi - 1],
                    "removed_per_layer_mean": float(rem[lo:hi].sum(1).mean()),
                    "removed_per_layer_min_max": [int(rem[lo:hi].sum(1).min()), int(rem[lo:hi].sum(1).max())],
                    "band_mass_removed": {
                        "I_traj": float((t_pool[lo:hi] * rem[lo:hi]).sum() / t_pool[lo:hi].sum()),
                        "I_CoC": float((c_pool[lo:hi] * rem[lo:hi]).sum() / c_pool[lo:hi].sum())},
                    "first_order": {"traj": float((t_pool * rem).sum()), "coc": float((c_pool * rem).sum())},
                }
                joint.setdefault(f"{band}_{name}", {})[axis] = ~rem
    for name, m in joint.items():
        np.savez(out / "joint" / f"both_{name}.npz",
                 q_mask=m["q"].astype(np.float32), mlp_mask=m["mlp"].astype(np.float32))
    np.savez_compressed(out / "sets.npz", **masks)
    (out / "sets.json").write_text(json.dumps({
        "plan": "plans/2026-09-24_dual-saves-trunk.md", "anatomy": args.anatomy,
        "bands": {k: [v[0], v[1] - 1] for k, v in BANDS.items()}, "configs": meta}, indent=2))
    header = (f"{'config':22s} {'rm/layer':>9s} {'min-max':>9s}   share of the band's mass held by "
              "the removed units: I_traj  I_CoC")
    lines = [f"dual-saves trunk sets -- scores from {args.anatomy}", "", header]
    for name, m in meta.items():
        b = m["band_mass_removed"]
        lines.append(f"{name:22s} {m['removed_per_layer_mean']:9.1f} {m['removed_per_layer_min_max']!s:>9s}"
                     f"   {b['I_traj']:6.3f} {b['I_CoC']:6.3f}")
    lines.append(f"\njoint (both axes) configs for --arm-masks: {', '.join('both_' + k for k in joint)}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n{len(masks)} per-axis configs + {len(joint)} joint -> {out}")


if __name__ == "__main__":
    main()
