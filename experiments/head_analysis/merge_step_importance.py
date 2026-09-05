"""Merge sharded run_step_importance_vlm outputs.

Unlike run_importance, this script keeps no per-clip axis in the file it ships: the
arrays are already means over clips (`acc / n`), with denoising step as the first axis.
So merging is a clip-count-weighted average, not a concatenation:

    merged = (a0 * n0 + a1 * n1) / (n0 + n1)

which equals the mean an unsharded run would have written, because every clip is scored
from its own seed (`sha256(f"{seed}:{clip_id}")`) and a clip's contribution does not
depend on which shard measured it or in what order.

Usage:
  python experiments/head_analysis/merge_step_importance.py \
      --shards importance_stepvlm_st4000_s0 importance_stepvlm_st4000_s1 \
      --out importance_stepvlm_st4000
"""

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    total, acc, ids, cfg = 0, None, [], None
    for s in args.shards:
        d = REPO / "outputs" / s
        c = json.loads((d / "config.json").read_text())
        m = json.loads((d / "metrics.json").read_text())
        n = m["n_clips"]
        z = dict(np.load(d / "step_importance_vlm.npz"))
        # weight by the number of clips actually MEASURED, not by the shard's intended
        # slice: a shard killed part-way writes means over the clips it finished, and
        # config.json still lists the full slice
        measured = [r["clip_id"] for r in m.get("per_clip", [])] or c["clip_ids"][:n]
        if len(measured) != n:
            raise SystemExit(f"{s}: n_clips={n} but {len(measured)} clip ids")
        print(f"  {s}: {n} clips", flush=True)
        acc = ({k: v * n for k, v in z.items()} if acc is None
               else {k: acc[k] + z[k] * n for k in acc})
        total += n
        ids.extend(measured)
        cfg = cfg or c

    dup = len(ids) - len(set(ids))
    if dup:
        raise SystemExit(f"{dup} clips appear in more than one shard -- shards overlap")

    out_dir = REPO / "outputs" / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "step_importance_vlm.npz", **{k: v / total for k, v in acc.items()})
    cfg.update({"num_clips": total, "clip_ids": sorted(ids), "merged_from": args.shards})
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps({"n_clips": total}, indent=2))
    print(f"merged {total} clips -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
