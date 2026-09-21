"""Unit sets for the token-targeted ablation (plans/2026-09-21_importance-causal-validation.md, B).

Every set is chosen on the DENSE model's calib_100 scores (outputs/gradanat_v1, split by the
token type a unit acts on), per layer, inside one band of layers; the ablation is then
measured on held-out clips by run_token_ablation.py.

  Ttok   largest rank(I_traj at vision + history tokens) - rank(pooled I_CoC)
  Ctok   largest rank(I_CoC at CoC tokens)               - rank(pooled I_traj)
  Tpool  largest rank(pooled I_traj) - rank(pooled I_CoC)      what Figure 1 is about
  Cpool  largest rank(pooled I_CoC)  - rank(pooled I_traj)
  R0-4   k random units per layer                              the null
  Scoc   units coc_u40_v2 removed and dual_u40_v2 kept         what the dual criterion saves
  Straj  units traj_u40_v2 removed and dual_u40_v2 kept        (late band only, B5)
  RScoc0-2 / RStraj0-2   random sets of Scoc's / Straj's per-layer size

k = 6 of 32 heads and 2,304 of 12,288 channels (the same 19%): the number of heads per late
layer on which the two shipped single criteria disagree. Bands: late = layers 22-34, trunk
(the control) = layers 6-17.

Writes outputs/<exp-id>/sets.npz, one (36, U) uint8 KEEP mask per config (1 = keep), and
sets.json with, per config, what share of each score's band mass the removed units hold and
the first-order prediction of the damage (the summed pooled score of the removed units).

Usage:
  python experiments/head_analysis/make_token_sets.py --exp-id tokabl_sets_v1
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[2]
VIS, HIST, COC = 0, 1, 3  # rows of the (5, L, U) typed arrays
BANDS = {"late": (22, 35), "trunk": (6, 18)}
K_FRAC = 6 / 32


def kept(arm, axis, n):
    meta = json.loads((REPO / "outputs" / f"slim_{arm}_u40_v2" / "slim_meta.json").read_text())
    out = np.zeros((36, n), bool)
    for li, e in enumerate(meta["vlm"]):
        out[li, e[axis]] = True
    return out


def top_k(score, band, k):
    """score (L, U) -> (L, U) bool, the k largest per layer inside the band."""
    rem = np.zeros(score.shape, bool)
    for li in range(*band):
        rem[li, np.argsort(-score[li], kind="stable")[:k]] = True
    return rem


def random_like(sizes, n, band, rng):
    """sizes: units to remove per layer, indexed by layer -> (L, U) bool."""
    rem = np.zeros((36, n), bool)
    for li in range(*band):
        rem[li, rng.choice(n, size=int(sizes[li]), replace=False)] = True
    return rem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="tokabl_sets_v1")
    ap.add_argument("--anatomy", default="gradanat_v1")
    args = ap.parse_args()

    z = np.load(REPO / "outputs" / args.anatomy / "anatomy.npz")
    masks, meta = {}, {}
    for axis, slim_axis in (("q", "q"), ("mlp", "mlp")):
        t_pool, c_pool = z[f"fm_full_{axis}"], z[f"ce_full_{axis}"]  # (L, U)
        t_tok = z[f"fm_type_{axis}"][VIS] + z[f"fm_type_{axis}"][HIST]
        c_tok = z[f"ce_{axis}"][COC]
        currencies = {"I_traj": t_pool, "I_CoC": c_pool, "I_traj@vision+hist": t_tok,
                      "I_CoC@coc": c_tok}
        n = t_pool.shape[1]
        k = round(K_FRAC * n)

        def rk(a):
            return rankdata(a, axis=1) / n  # (L, U) within-layer rank in (0, 1]

        rules = {"Ttok": rk(t_tok) - rk(c_pool), "Ctok": rk(c_tok) - rk(t_pool),
                 "Tpool": rk(t_pool) - rk(c_pool), "Cpool": rk(c_pool) - rk(t_pool)}
        dual, traj, coc = (kept(a, slim_axis, n) for a in ("dual", "traj", "coc"))
        saved = {"Scoc": dual & ~coc, "Straj": dual & ~traj}

        removed = {}
        for band, span in BANDS.items():
            for name, score in rules.items():
                removed[f"{axis}_{band}_{name}"] = top_k(score, span, k)
            for i in range(5):
                removed[f"{axis}_{band}_R{i}"] = random_like(
                    np.full(36, k), n, span, np.random.default_rng([1000 + i, span[0], n]))
        span = BANDS["late"]
        for name, s in saved.items():
            rem = np.zeros((36, n), bool)
            rem[span[0]:span[1]] = s[span[0]:span[1]]
            removed[f"{axis}_late_{name}"] = rem
            for i in range(3):
                removed[f"{axis}_late_R{name}{i}"] = random_like(
                    rem.sum(1), n, span, np.random.default_rng([2000 + i, len(name), n]))

        for name, rem in removed.items():
            band = name.split("_")[1]
            lo, hi = BANDS[band]
            masks[name] = (~rem).astype(np.uint8)
            meta[name] = {
                "axis": axis, "band": band, "set": name.split("_", 2)[2],
                "layers": [lo, hi - 1], "removed_per_layer_mean": float(rem[lo:hi].sum(1).mean()),
                "removed_per_layer_min_max": [int(rem[lo:hi].sum(1).min()), int(rem[lo:hi].sum(1).max())],
                "band_mass_removed": {c: float((s[lo:hi] * rem[lo:hi]).sum() / s[lo:hi].sum())
                                      for c, s in currencies.items()},
                "first_order": {"traj": float((t_pool * rem).sum()), "coc": float((c_pool * rem).sum())},
            }
        for band in BANDS:
            a, b = removed[f"{axis}_{band}_Ttok"], removed[f"{axis}_{band}_Ctok"]
            meta[f"{axis}_{band}_Ttok"]["overlap_with_Ctok"] = float((a & b).sum() / a.sum())
            a, b = removed[f"{axis}_{band}_Ttok"], removed[f"{axis}_{band}_Tpool"]
            meta[f"{axis}_{band}_Ttok"]["overlap_with_Tpool"] = float((a & b).sum() / a.sum())
            a, b = removed[f"{axis}_{band}_Ctok"], removed[f"{axis}_{band}_Cpool"]
            meta[f"{axis}_{band}_Ctok"]["overlap_with_Cpool"] = float((a & b).sum() / a.sum())

    out = REPO / "outputs" / args.exp_id
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "sets.npz", **masks)
    (out / "sets.json").write_text(json.dumps({
        "plan": "plans/2026-09-21_importance-causal-validation.md", "anatomy": args.anatomy,
        "bands": {k: [v[0], v[1] - 1] for k, v in BANDS.items()}, "k_frac": K_FRAC,
        "configs": meta}, indent=2))
    lines = [f"token-targeted ablation sets -- scores from {args.anatomy}", "",
             f"{'config':22s} {'rm/layer':>9s}  share of the band's mass held by the removed units",
             f"{'':32s} {'I_traj':>7s} {'I_CoC':>7s} {'traj@vis+hist':>14s} {'CoC@coc':>8s}"]
    for name, m in meta.items():
        b = m["band_mass_removed"]
        lines.append(f"{name:22s} {m['removed_per_layer_mean']:9.1f}  {b['I_traj']:7.3f} {b['I_CoC']:7.3f} "
                     f"{b['I_traj@vision+hist']:14.3f} {b['I_CoC@coc']:8.3f}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n{len(masks)} configs -> {out}")


if __name__ == "__main__":
    main()
