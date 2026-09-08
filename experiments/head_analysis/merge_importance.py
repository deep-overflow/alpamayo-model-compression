"""Merge sharded run_importance outputs into one importance run.

`run_importance.py --shard i --n-shards k` splits the calibration list round-robin.
Because every clip is scored from its own seed (`sha256(f"{seed}:{clip_id}")`), a clip's
per-clip array does not depend on which shard measured it or in what order -- so merging
is concatenation, and the merged mean is exactly what an unsharded run would have
written.

Rows are restored to manifest order, not shard order, so the merged
`importance_perclip.npz` can still be split by block the way
`make_block_importance.py` does.

Usage:
  python experiments/head_analysis/merge_importance.py \
      --shards importance_st4000_s0 importance_st4000_s1 \
      --out importance_st4000 --manifest calib_st4000
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True,
                    help="eval_sets stem the shards were drawn from; sets the row order")
    args = ap.parse_args()

    man = pd.read_parquet(REPO / "outputs" / "eval_sets" / f"{args.manifest}.parquet")
    order = {c: i for i, c in enumerate(man["clip_id"])}

    per, ids, cfg = [], [], None
    for s in args.shards:
        d = REPO / "outputs" / s
        c = json.loads((d / "config.json").read_text())
        m = json.loads((d / "metrics.json").read_text())
        rows = [r["clip_id"] for r in m["per_clip"]]
        z = dict(np.load(d / "importance_perclip.npz"))
        n = next(iter(z.values())).shape[0]
        if n != len(rows):
            raise SystemExit(f"{s}: {n} per-clip rows but {len(rows)} records")
        print(f"  {s}: {n} clips", flush=True)
        per.append(z)
        ids.extend(rows)
        cfg = cfg or c

    dup = len(ids) - len(set(ids))
    if dup:
        raise SystemExit(f"{dup} clips appear in more than one shard -- shards overlap")
    missing = [c for c in ids if c not in order]
    if missing:
        raise SystemExit(f"{len(missing)} clips are not in {args.manifest}, e.g. {missing[:3]}")

    keys = sorted(per[0])
    stacked = {k: np.concatenate([p[k] for p in per], axis=0) for k in keys}
    idx = np.argsort([order[c] for c in ids])          # back to manifest order
    stacked = {k: v[idx] for k, v in stacked.items()}
    ordered_ids = [ids[i] for i in idx]

    out_dir = REPO / "outputs" / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "importance_perclip.npz", **stacked)
    np.savez(out_dir / "importance.npz",
             **{k: v.astype(np.float64).mean(0) for k, v in stacked.items()})
    cfg.update({"num_clips": len(ordered_ids), "clip_ids": ordered_ids,
                "merged_from": args.shards, "calib_manifest": args.manifest})
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps({"n_clips": len(ordered_ids)}, indent=2))
    print(f"merged {len(ordered_ids)} clips -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
