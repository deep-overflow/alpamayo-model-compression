"""Stratified random scene sample from public_2601 for the dense closed-loop baseline.

Origin150 (sorted(scene_id)[:150]) is random in principle -- every scene_id is a UUID v4 -- but the
draw came out easy: slower and shorter than the rest of the suite, and sangoh's unpruned 913 x 1 run
scores it 0.742 against 0.673 for the whole suite. This draws a replacement sample instead.

Frame: public_2601 minus the PhysicalAI-AV train-split clips (one scene). Strata: sangoh's per-scene
unpruned score in three levels (0 / in between / 1, it is one rollout so the values pile up at the
ends) crossed with hard_score tertiles over the frame -- 9 cells, allocated in proportion to cell size
(largest remainder), drawn without replacement within each cell. Proportional allocation keeps the
sample mean an unbiased estimate of the frame mean whatever the stratifier is; the stratifier only
buys precision and balance. Plan: plans/2026-10-28_dense-closedloop-full2601.md.

Writes <out>/scenes_s<N>.csv (the launcher's SCENES_CSV: test_suite_id, scene_id, uuid, plus the
stratum columns), <out>/scenes_smoke16.csv (16 sampled scenes outside Origin150, two per card so each
stack overlaps two rollouts as in the main run) and <out>/draw.json (cell table, overlaps, balance check).

Usage:
  python experiments/head_analysis/make_strat_suite.py --out outputs/alpasim_dense_s600
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SUITES = Path("/home/cvlab21/project/chan/alpasim/data/scenes/sim_suites.csv")
FEATS = REPO / "outputs" / "scene_difficulty" / "scene_difficulty_public2601.csv"
HARD100 = REPO / "outputs" / "scene_difficulty" / "hard100_suite.csv"
SANGOH = Path("/mnt/nvme1n1/ad_vla/results/sangoh/alpasim_runs/eval2601_a1_5/aggregate/results-summary.json")
CLIP_INDEX = Path("/mnt/nvme1n1/ad_vla/data/physicalai_av/clip_index.parquet")
BALANCE = ["hard_score", "v_mean", "yaw_total_deg", "gt_path_m", "stop_frac", "decel_max", "yawrate_max",
           "n_agents", "n_agents_30m", "n_ped_15m", "dmin_agent", "n_junctions", "n_signals", "sangoh"]


def max_smd(a, b):
    return max(abs((a[c].mean() - b[c].mean()) / np.sqrt((a[c].var() + b[c].var()) / 2)) for c in BALANCE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("-n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--suite", default="public_2601")
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else REPO / args.out
    out.mkdir(parents=True, exist_ok=True)

    su = pd.read_csv(SUITES)
    suite = su[su.test_suite_id == args.suite].sort_values("scene_id").reset_index(drop=True)
    d = pd.read_csv(FEATS).set_index("scene_id").loc[suite.scene_id].reset_index()
    sg = json.loads(SANGOH.read_text())["rollouts"]
    d["sangoh"] = d.scene_id.map({r["clipgt_id"]: float(r["score"]) for r in sg if r.get("score") is not None})
    if d.sangoh.isna().any():
        raise SystemExit(f"{d.sangoh.isna().sum()} scenes have no sangoh score")
    split = pd.read_parquet(CLIP_INDEX).split
    d["split"] = split.reindex(d.scene_id.str.replace("clipgt-", "", regex=False)).values
    d["origin150"] = d.scene_id.isin(set(suite.scene_id.iloc[:150]))
    d["hard100"] = d.scene_id.isin(set(pd.read_csv(HARD100).scene_id))

    f = d[d.split != "train"].copy()
    f["sg_level"] = np.select([f.sangoh <= 0.0, f.sangoh >= 0.999], ["0", "1"], "mid")
    # categorical, so the cells are visited lo -> mid -> hi; the draw order fixes the sample
    f["hs_tertile"] = pd.qcut(f.hard_score, 3, labels=["lo", "mid", "hi"])
    cells = f.groupby(["sg_level", "hs_tertile"], observed=True).size()
    exact = cells / len(f) * args.n
    alloc = np.floor(exact).astype(int)
    alloc[(exact - alloc).sort_values(ascending=False).index[:args.n - alloc.sum()]] += 1

    rng = np.random.default_rng(args.seed)
    pick = []
    for (lvl, ter), k in alloc.items():
        g = f[(f.sg_level == lvl) & (f.hs_tertile == ter)].sort_values("scene_id")
        pick += list(rng.choice(g.scene_id.values, k, replace=False))
    f["sampled"] = f.scene_id.isin(pick)
    s, rest = f[f.sampled], f[~f.sampled]

    null = []
    for _ in range(1000):
        m = np.zeros(len(f), bool)
        m[rng.choice(len(f), args.n, replace=False)] = True
        null.append(max_smd(f[m], f[~m]))
    obs = max_smd(s, rest)

    s = s.sort_values("scene_id").assign(test_suite_id=args.suite)
    cols = ["test_suite_id", "scene_id", "uuid", "sg_level", "hs_tertile", "sangoh", "hard_score",
            "origin150", "hard100", "split"]
    s[cols].to_csv(out / f"scenes_s{args.n}.csv", index=False)
    s[~s.origin150].head(16)[cols].to_csv(out / "scenes_smoke16.csv", index=False)

    info = {
        "suite": args.suite, "n": args.n, "seed": args.seed, "frame": len(f),
        "excluded_train_split": d.scene_id[d.split == "train"].tolist(),
        "cells": [{"sg_level": a, "hs_tertile": b, "N": int(cells[(a, b)]), "n": int(alloc[(a, b)])}
                  for a, b in cells.index],
        "overlap_origin150": int(s.origin150.sum()), "overlap_hard100": int(s.hard100.sum()),
        "max_smd": round(float(obs), 4), "max_smd_random_median": round(float(np.median(null)), 4),
        "max_smd_random_pct": round(float((np.array(null) < obs).mean()), 3),
        "sangoh_mean_sample": round(float(s.sangoh.mean()), 4), "sangoh_mean_frame": round(float(f.sangoh.mean()), 4),
    }
    (out / "draw.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != "cells"}, indent=1))
    print(pd.DataFrame(info["cells"]).to_string(index=False))


if __name__ == "__main__":
    main()
