"""Split one importance run into per-block and cumulative importance runs, with no GPU.

`run_importance.py` stores `acc / n` in importance.npz (`run_importance.py:267`) and the
untouched per-clip arrays in importance_perclip.npz (`:270`), stacked in manifest order.
Every clip is scored from its own seed (`sha256(f"{seed}:{clip_id}")`), independent of
order and sharding, so the mean over any subset of a run is exactly the mean a separate
run over that subset would have produced. That is what the pooled-200 arm of
`2026-08-19_calibration-source.html` already relied on.

So the draw-variance ladder costs one 500-clip run: this writes

  <exp>_a ... <exp>_e     one calibration block each (100 clips)
  <exp>_200 / _300 / _500 the cumulative unions a+b, a+b+c, a..e

each a directory `make_slim.py --importance <name>` reads like any other importance run.
Gate G0 of `plans/2026-09-03_calib-draw-variance.md` checks the synthesis against a
separately measured block before any of it is believed.

Usage:
  python experiments/evaluation/make_block_importance.py --importance importance_tr500
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def write_run(out_dir, per_clip, rows, src_cfg, clip_ids, note):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "importance.npz",
             **{k: v[rows].mean(axis=0) for k, v in per_clip.items()})
    cfg = dict(src_cfg)
    cfg.update({"num_clips": len(rows), "clip_ids": clip_ids, "derived_from": note})
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps({"n_clips": len(rows)}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--importance", default="importance_tr500")
    ap.add_argument("--manifest", default="calib_tr500",
                    help="parquet with the `block` column that defines the split")
    ap.add_argument("--cumulative", type=int, nargs="*", default=[200, 300, 500])
    args = ap.parse_args()

    src = REPO / "outputs" / args.importance
    cfg = json.loads((src / "config.json").read_text())
    per_clip = dict(np.load(src / "importance_perclip.npz"))
    man = pd.read_parquet(REPO / "outputs" / "eval_sets" / f"{args.manifest}.parquet")

    # the run may have stopped short of the manifest, and --num-clips takes a prefix
    n = len(cfg["clip_ids"])
    assert list(man["clip_id"][:n]) == list(cfg["clip_ids"]), \
        "run's clip order does not match the manifest -- the row split would be wrong"
    assert all(v.shape[0] == n for v in per_clip.values()), "per-clip rows != clip count"
    man = man.iloc[:n]

    written = []
    for b, g in man.groupby("block", sort=True):
        rows = np.asarray(g.index)
        write_run(REPO / "outputs" / f"{args.importance}_{b}", per_clip, rows, cfg,
                  list(g["clip_id"]), f"{args.importance} rows of block {b}")
        written.append((f"{args.importance}_{b}", len(rows)))

    # the union manifest is block a then b then ..., so a prefix is exactly a+b+...
    assert list(man["block"]) == sorted(man["block"]), "manifest is not block-ordered"
    for k in args.cumulative:
        if k > n:
            continue
        write_run(REPO / "outputs" / f"{args.importance}_c{k}", per_clip, np.arange(k),
                  cfg, list(man["clip_id"][:k]), f"{args.importance} first {k} clips")
        written.append((f"{args.importance}_c{k}", k))

    for name, k in written:
        print(f"{name:28s} {k:4d} clips", flush=True)


if __name__ == "__main__":
    main()
