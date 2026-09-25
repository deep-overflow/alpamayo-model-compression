"""Head sets for the CoC-position double dissociation (plans/2026-09-25_coc-position-content.md, B).

From the calib_100 anatomy's same-position scores at generated-CoC positions
(outputs/gradanat_v1/anatomy_perclip_q.npz: I_traj@CoC = |fm_full[:, coc]| and I_CoC@CoC =
|ce[:, coc]| averaged over clips) rank every Q head within its layer and take, per layer,

  T   the k heads with the largest rank(I_traj@CoC) - rank(I_CoC@CoC)   (FM-favoured at CoC)
  C   the k heads with the smallest                                    (CE-favoured at CoC)
  R0-2  k random heads per layer

each for all layers 0-34 (`all`) and for layers 0-21 only (`trunk`). Writes (L, H) uint8 arrays
(1 = switch the head's output OFF at CoC positions) to outputs/<exp-id>/masks.npz + masks.json.

Usage:
  python experiments/head_analysis/make_coc_position_masks.py --exp-id coc_posabl_sets_v1 --k 8
"""

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
COC = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="coc_posabl_sets_v1")
    ap.add_argument("--anatomy", default="gradanat_v1")
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    pq = np.load(REPO / "outputs" / args.anatomy / "anatomy_perclip_q.npz")
    Tc = np.abs(pq["fm_full"][:, COC].astype(np.float64)).mean(0)  # (36, 32)
    Cc = np.abs(pq["ce"][:, COC].astype(np.float64)).mean(0)
    L, H = Tc.shape
    rk = lambda x: np.argsort(np.argsort(x, axis=1), axis=1) / (H - 1)
    diff = rk(Tc) - rk(Cc)
    masks, meta = {}, {}
    for band, layers in (("all", range(35)), ("trunk", range(22))):
        sets = {"T": np.zeros((L, H), np.uint8), "C": np.zeros((L, H), np.uint8)}
        for l in layers:
            order = np.argsort(diff[l])
            sets["T"][l, order[-args.k:]] = 1
            sets["C"][l, order[: args.k]] = 1
        for i in range(3):
            rng = np.random.default_rng([5000 + i, len(band)])
            r = np.zeros((L, H), np.uint8)
            for l in layers:
                r[l, rng.choice(H, size=args.k, replace=False)] = 1
            sets[f"R{i}"] = r
        for name, m in sets.items():
            key = f"{name}_{band}"
            masks[key] = m
            meta[key] = {"set": name, "band": band, "layers": [min(layers), max(layers)], "heads": int(m.sum()),
                         "mass_share_I_traj_at_coc": float((Tc * m).sum() / Tc[list(layers)].sum()),
                         "mass_share_I_CoC_at_coc": float((Cc * m).sum() / Cc[list(layers)].sum())}
    out = REPO / "outputs" / args.exp_id
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "masks.npz", **masks)
    (out / "masks.json").write_text(json.dumps({"anatomy": args.anatomy, "k": args.k, "configs": meta}, indent=2))
    for key, m in meta.items():
        print(f"{key:9s} heads {m['heads']:4d} | share of the band's CoC-position mass: I_traj {m['mass_share_I_traj_at_coc']:.3f} I_CoC {m['mass_share_I_CoC_at_coc']:.3f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
