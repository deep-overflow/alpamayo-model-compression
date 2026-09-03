"""Draw disjoint calibration blocks from the official-train clips that are already cached.

`2026-08-19_calibration-source.html` measured the calibration draw against exactly one
control (`calib_val100`) and put +0.0941 on it -- a point estimate, drawn from official
*val* while `calib_100` came from official *train*, so the draw effect and the split were
confounded. This draws several disjoint blocks from official train so the effect gets an
interval and the split is held fixed.

Why not `make_eval_sets.py --split train`: its pool is the whole official train split
(153,625 clips), and whatever it picks has to be streamed into a per-clip cache first.
This draws from the 9,654 train clips already in `pre_processed/train`, so a block costs
no download. Everything else is the same: the same six weighted attributes, the same
greedy matcher, the same target (the full official-train distribution), the same seed
rule -- the blocks have to be drawn the way `calib_100` was or "the draw changed" is not
the only thing that changed.

**One sample per clip.** `pre_processed/train` holds up to 12 windows per clip (recovery
training wanted several t0 per clip), so a manifest taken off that index weights some
clips twice. Each clip contributes the single sample whose t0 is nearest `CALIB_T0`
(5.1 s, what `calib_100` uses); ties take the earlier t0. t0 still varies clip to clip,
which `calib_100` does not -- the cache has no 5.1 s window and rebuilding one was judged
not worth 25 minutes of streaming.

Every block is disjoint from every other block, from `calib_100`, and from the OOD pool
(8 cached train clips appear in `ood_reasoning.parquet` and are dropped), hence from
every evaluation set -- verified, not assumed.

Writes one manifest per block plus the union, all readable by
`run_importance.py --calib-manifest <name> --cache train`:

  calib_tr100_a ... calib_tr100_e   one block each
  calib_tr500                       the union, carrying a `block` column

Usage:
  python experiments/evaluation/make_calib_blocks.py --blocks 5 --block-size 100
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from make_eval_sets import ATTR, derive, encode, greedy, random_reference, weighted_l1

REPO = Path(__file__).resolve().parents[2]
AV = Path("/mnt/nvme1n1/ad_vla/data/physicalai_av")
PRE = AV / "pre_processed"
CALIB_T0 = 5_100_000


def one_sample_per_clip(cache):
    """clip_id -> t0_us, taking the window nearest CALIB_T0 (ties: the earlier one)."""
    best = {}
    for stem in json.loads((PRE / cache / "index.json").read_text()):
        clip_id, t0 = stem.split("__t0_")
        t0 = int(t0)
        key = (abs(t0 - CALIB_T0), t0)
        if clip_id not in best or key < best[clip_id][0]:
            best[clip_id] = (key, t0)
    return {c: v[1] for c, v in best.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--block-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", default="train", help="pre_processed/<cache> to draw from")
    ap.add_argument("--exp-id", default="eval_sets")
    ap.add_argument("--prefix", default="calib_tr")
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    t0_of = one_sample_per_clip(args.cache)
    ci = pd.read_parquet(AV / "clip_index.parquet")
    dc = pd.read_parquet(AV / "metadata" / "data_collection.parquet")
    ood = set(pd.read_parquet(AV / "reasoning" / "ood_reasoning.parquet").index.astype(str))
    prior = set(pd.read_parquet(out_dir / "calib_100.parquet")["clip_id"])

    full = derive(ci[ci.split == "train"], dc)          # target: the whole official split
    pool = full[full.index.isin(t0_of)]                  # only what the cache can serve
    n_cached = len(pool)
    pool = pool[~pool.index.isin(ood) & ~pool.index.isin(prior)]
    need = args.blocks * args.block_size
    print(f"official train {len(full):,}  cached {n_cached:,}  "
          f"-OOD/-calib_100 {n_cached - len(pool)}  -> pool {len(pool):,}  need {need}",
          flush=True)
    assert len(pool) >= need, f"pool {len(pool)} < {need}"

    # each block is matched to the full-train distribution on its own, drawn from what the
    # earlier blocks left behind -- a slice of one long greedy order would not be matched,
    # since greedy corrects an early imbalance with a later pick
    names, rows, quality = [], [], []
    letters = "abcdefghijklmnopqrstuvwxyz"
    for b in range(args.blocks):
        codes, targets = encode(full, pool)
        order, counts = greedy(codes, targets, len(pool), args.block_size, args.seed)
        sel = pool.iloc[order]
        rmean, rstd = random_reference(codes, targets, len(pool), args.block_size, args.seed)
        wl1 = weighted_l1(counts, args.block_size, targets)
        name = f"{args.prefix}{args.block_size}_{letters[b]}"
        print(f"  {name}: weighted L1 {wl1:.4f} (random {rmean:.4f}+-{rstd:.4f})  "
              f"countries {sel['country'].nunique()}", flush=True)
        quality.append({"block": letters[b], "n": args.block_size, "weighted_l1": wl1,
                        "random_l1_mean": rmean, "random_l1_std": rstd,
                        "countries": int(sel["country"].nunique())})
        d = sel.reset_index(names="clip_id")
        d["t0_us"] = d["clip_id"].map(t0_of)
        d["block"] = letters[b]
        keep = ["clip_id", "t0_us", "chunk", "block", "country", "platform_class",
                "radar_config", "month", "hour_of_day", "time_of_day", "season"]
        d[keep].to_parquet(out_dir / f"{name}.parquet", index=False)
        names.append(name)
        rows.append(d[keep])
        pool = pool[~pool.index.isin(set(d["clip_id"]))]

    union = pd.concat(rows, ignore_index=True)
    union_name = f"{args.prefix}{args.blocks * args.block_size}"
    union.to_parquet(out_dir / f"{union_name}.parquet", index=False)

    # disjointness is the whole point of the design, so it is checked rather than assumed
    ids = [set(r["clip_id"]) for r in rows]
    assert len(union) == len(set(union["clip_id"])) == need, "blocks overlap"
    assert all(not (a & prior) for a in ids) and all(not (a & ood) for a in ids)
    evalsets = {}
    for s in ("test_500", "val_500", "indist_500", "ood_val", "calib_val100"):
        p = out_dir / f"{s}.parquet"
        if p.exists():
            e = pd.read_parquet(p)
            col = "clip_id" if "clip_id" in e.columns else e.columns[0]
            evalsets[s] = len(set(union["clip_id"]) & set(e[col].astype(str)))
    assert not any(evalsets.values()), f"blocks touch an evaluation set: {evalsets}"

    (out_dir / f"config_{union_name}.json").write_text(json.dumps({
        "purpose": "disjoint in-distribution calibration blocks for the draw-variance study",
        "blocks": args.blocks, "block_size": args.block_size, "seed": args.seed,
        "cache": args.cache, "manifests": names, "union": union_name,
        "attributes": ATTR, "t0_rule": f"nearest {CALIB_T0} among the cache's windows",
        "pool": {"official_train": len(full), "cached": n_cached, "eligible": len(pool) + need},
        "excluded": {"calib_100": len(prior), "ood_reasoning": len(ood)},
        "overlap_with_eval_sets": evalsets,
    }, indent=2))
    pd.DataFrame(quality).to_csv(out_dir / f"quality_{union_name}.csv", index=False)
    t0s = union["t0_us"].value_counts().sort_index()
    per_block = ", ".join("{}={:.4f}".format(q["block"], q["weighted_l1"]) for q in quality)
    (out_dir / f"summary_{union_name}.txt").write_text(
        f"{args.blocks} x {args.block_size} disjoint calibration blocks from "
        f"pre_processed/{args.cache}\n"
        f"pool {n_cached:,} cached official-train clips, {len(pool) + need:,} eligible\n"
        f"one sample per clip, t0 nearest {CALIB_T0} us\n"
        f"t0 spread: {dict(zip(t0s.index.tolist(), t0s.tolist()))}\n"
        f"weighted L1 per block: {per_block}\n"
        f"disjoint from calib_100 and every evaluation set (asserted)\n")
    print(f"wrote {union_name} + {len(names)} blocks to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
