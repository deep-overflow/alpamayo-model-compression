"""Layer-0 exemption: the same kept sets with ALL 32 Q heads of layer 0 kept.

`analyze_l0_masks.py` located the cruise collapse of bad seed arms and bad calibration
draws in which layer-0 Q heads survive. This writes, for each source arm, two variants:

  <arm>_L0full    the identical slim_meta.json with layer 0's Q heads restored in full and
                  nothing else touched, so the only factor is whether layer 0 is cut. The
                  budget is NOT matched: 13 heads x 2 * head_dim * hidden = 13,631,488
                  fewer parameters are removed, 0.51% of 2,657,452,032.
  <arm>_L0fullM   the same, with the budget paid back in MLP channels. 13 heads are worth
                  1109.33 channels (one channel = 3 * hidden = 12,288 parameters), which
                  does not divide, so 1,109 more channels are dropped and the variant
                  removes 4,096 parameters (0.00015%) fewer than the source arm. They are
                  spread evenly -- 31 in each of layers 0-28, 30 in layers 29-35 -- and
                  within a layer they are the arm's own next-lowest dual scores, i.e.
                  exactly the channels the arm's criterion would have cut next. The script
                  first re-derives the arm's shipped MLP kept set from its importance file
                  and refuses to continue if that does not reproduce slim_meta.json.

Usage:
  python experiments/evaluation/make_l0_exempt.py
"""

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
# source slim dir, importance run it was selected from
SRC = {"ship": ("slim_dual_u40_v2", "importance_v2"),
       "ref": ("slim_dual_seed_ref", "importance_seed_ref"),
       "c3": ("slim_dual_seed_c3", "importance_seed_c3"),
       "n1": ("slim_dual_seed_n1", "importance_seed_n1"),
       **{f"nt{x}": (f"slim_dual_nt_{x}", f"importance_nt500_{x}") for x in "bcde"}}
# earlier conclusions the layer-0 census (analyze_l0_census.py) flagged for re-measurement.
# Unmatched variant only: one factor, and it needs no importance file -- dualsafe's score is
# composed at build time, so there is nothing to pick 1,109 "next" channels from.
RECHECK = {"self": "slim_dualself_u40_v2", "bestn6": "slim_dualbestn6_u40_v2",
           "st2000": "slim_dual_u40_st2000", "safe": "slim_dualsafe_u40_v2",
           **{f"rd{x}": f"slim_dual_rd_{x}" for x in "abc"},
           **{f"se{x}": f"slim_dual_se_{x}" for x in "abc"}}
N_MLP, DROP_MLP, EXTRA = 12288, 4898, 1109
HEAD_PARAMS, CH_PARAMS = 2 * 128 * 4096, 3 * 4096


def rank_norm(s):
    """Per-layer rank in [0, 1]. (L, U) -> (L, U); verbatim from run_cocsafe."""
    return np.argsort(np.argsort(s, axis=1), axis=1) / max(s.shape[1] - 1, 1)


def main():
    out = REPO / "outputs"
    extra = np.full(36, EXTRA // 36)
    extra[: EXTRA - extra.sum()] += 1                 # 31 x 29 layers + 30 x 7 = 1109
    assert extra.sum() == EXTRA
    for name, (d, imp) in SRC.items():
        m = json.loads((out / d / "slim_meta.json").read_text())
        dropped = sorted(set(range(32)) - set(m["vlm"][0]["q"]))
        m["vlm"][0]["q"] = list(range(32))
        assert all(len(l["q"]) == 19 for l in m["vlm"][1:]) and len(m["vlm"][0]["q"]) == 32
        assert all(len(l["mlp"]) == N_MLP - DROP_MLP for l in m["vlm"])
        m["config"] = f"swap_{name}_L0full"
        dd = out / f"slim_swap_{name}_L0full"
        dd.mkdir(exist_ok=True)
        (dd / "slim_meta.json").write_text(json.dumps(m))

        z = np.load(out / imp / "importance.npz")
        s = np.maximum(rank_norm(z["traj_vlm_mlp"]), rank_norm(z["coc_vlm_mlp"]))  # (36, 12288)
        order = np.argsort(s, axis=1)                  # ascending: first DROP_MLP were cut
        for li in range(36):
            assert sorted(order[li, DROP_MLP:].tolist()) == m["vlm"][li]["mlp"], (
                f"{name}: {imp} does not reproduce {d}'s layer-{li} MLP kept set")
            m["vlm"][li]["mlp"] = sorted(order[li, DROP_MLP + extra[li]:].tolist())
        m["config"] = f"swap_{name}_L0fullM"
        short = 13 * HEAD_PARAMS - EXTRA * CH_PARAMS
        m["params"] = {**m["params"], "removed": m["params"]["removed"] - short,
                       "slim": m["params"]["slim"] + short}
        dd = out / f"slim_swap_{name}_L0fullM"
        dd.mkdir(exist_ok=True)
        (dd / "slim_meta.json").write_text(json.dumps(m))
        print(f"{name:5s} <- {d:22s} restored layer-0 heads {dropped}; matched variant drops "
              f"{EXTRA} more channels, removed {m['params']['removed']:,} ({short:+,} short)")

    for name, d in RECHECK.items():
        m = json.loads((out / d / "slim_meta.json").read_text())
        dropped = sorted(set(range(32)) - set(m["vlm"][0]["q"]))
        assert all(len(l["q"]) == 19 and len(l["mlp"]) == N_MLP - DROP_MLP for l in m["vlm"]), name
        m["vlm"][0]["q"] = list(range(32))
        m["config"] = f"swap_{name}_L0full"
        dd = out / f"slim_swap_{name}_L0full"
        dd.mkdir(exist_ok=True)
        (dd / "slim_meta.json").write_text(json.dumps(m))
        print(f"{name:7s} <- {d:26s} restored layer-0 heads {dropped}")


if __name__ == "__main__":
    main()
