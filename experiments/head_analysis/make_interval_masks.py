"""Whole-layer interval masks for the interval-deletion landscape.

plans/2026-10-13_interval-landscape-pilot.md. One npz per config in the q_mask / mlp_mask format
of slim_to_mask.py, with every Q head and MLP channel of the layers in the interval(s) masked
(kv-only bypass: k/v projections still run). Three groups:

  single  [l, l] for every layer
  grid    intervals whose boundaries sit on --start, --start + --stride, ..., plus the last layer
  pair    A u B for the two-interval additivity test

Writes outputs/<exp-id>/masks/<name>.npz, intervals.json and arm_masks.txt (the NAME=NPZ list
run_token_ablation.py --arm-masks takes).

Usage:
  python experiments/head_analysis/make_interval_masks.py --exp-id ivl_masks_v1
"""

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
L, N_HEADS, INTERMEDIATE = 36, 32, 12288
PAIR_A = [(8, 10), (8, 13)]
PAIR_B = [(17, 19), (20, 22), (23, 25), (26, 28), (29, 31), (32, 34), (23, 28), (29, 34)]


def name_of(intervals):
    return "+".join(f"i{a}_{b}" for a, b in intervals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--start", type=int, default=2)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=18)
    args = ap.parse_args()

    cfgs = {}  # name -> (group, [(a, b), ...]) with inclusive layer indices
    for li in range(L):
        cfgs[name_of([(li, li)])] = ("single", [(li, li)])
    bounds = sorted(set(range(args.start, L, args.stride)) | {L})
    for i, s in enumerate(bounds):
        for e in bounds[i + 1 :]:
            if e - s <= args.max_len:
                cfgs.setdefault(name_of([(s, e - 1)]), ("grid", [(s, e - 1)]))
    for a in PAIR_A:
        for b in PAIR_B:
            assert name_of([a]) in cfgs and name_of([b]) in cfgs, "pair members must be measured alone"
            assert a[1] + 1 < b[0], "pair intervals must be separated by a kept layer"
            cfgs[name_of([a, b])] = ("pair", [a, b])

    out = REPO / "outputs" / args.exp_id
    (out / "masks").mkdir(parents=True, exist_ok=True)
    lines = []
    for name, (_, intervals) in cfgs.items():
        q = np.ones((L, N_HEADS), np.uint8)  # (36, 32)
        mlp = np.ones((L, INTERMEDIATE), np.uint8)  # (36, 12288)
        for a, b in intervals:
            q[a : b + 1] = 0
            mlp[a : b + 1] = 0
        np.savez_compressed(out / "masks" / f"{name}.npz", q_mask=q, mlp_mask=mlp)
        lines.append(f"{name}=outputs/{args.exp_id}/masks/{name}.npz")
    (out / "arm_masks.txt").write_text("\n".join(lines) + "\n")
    (out / "intervals.json").write_text(json.dumps({
        "start": args.start, "stride": args.stride, "max_len": args.max_len, "bounds": bounds,
        "configs": {n: {"group": g, "intervals": iv} for n, (g, iv) in cfgs.items()},
    }, indent=2))
    groups = [g for g, _ in cfgs.values()]
    print({g: groups.count(g) for g in ("single", "grid", "pair")}, "->", out)


if __name__ == "__main__":
    main()
