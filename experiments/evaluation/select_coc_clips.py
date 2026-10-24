"""Pick the clips `run_coc_importance.py` re-scores, from the `run_coc_multimode.py` rows.

  high     every clip whose between-CoC endpoint RMS exceeds --thr (1 m): the clips where
           resampling the CoC actually moves the trajectory.
  control  the same number of clips drawn at random (seed 0) from the rest, as the
           reference for what a CoC resample does to importance on an ordinary clip.

Usage:
  python experiments/evaluation/select_coc_clips.py [--out outputs/coc_importance/clips.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

from analyze_coc_multimode import SETS, clip_metrics, load

CACHE = {"calib_100": "calib"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=1.0)
    ap.add_argument("--out", default="outputs/coc_importance/clips.json")
    args = ap.parse_args()
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    clips = []
    for name, stem in SETS:
        t0 = dict(pd.read_parquet(REPO / "outputs" / "eval_sets" / f"{stem}.parquet")
                  [["clip_id", "t0_us"]].itertuples(index=False))
        for r in load(stem):
            m = clip_metrics(r)
            clips.append({"clip_id": r["clip_id"], "set": name, "manifest": stem,
                          "cache": CACHE.get(stem, "calib_nt"), "t0_us": int(t0[r["clip_id"]]),
                          "bucket": r["bucket"], "end_rms_between": float(m["end_rms_between"]),
                          "end_rms_within": float(m["end_rms_within"]),
                          "n_text": int(m["n_text"]), "n_head": int(m["n_head"]),
                          "nll": float(m["nll"])})
    high = [c for c in clips if c["end_rms_between"] > args.thr]
    rest = [c for c in clips if c["end_rms_between"] <= args.thr]
    rng = np.random.default_rng(0)
    ctrl = [rest[i] for i in sorted(rng.choice(len(rest), len(high), replace=False))]
    for c in high:
        c["group"] = "high"
    for c in ctrl:
        c["group"] = "control"
    sel = high + ctrl
    out.write_text(json.dumps(sel, indent=1))
    for g, cs in (("high", high), ("control", ctrl)):
        per = {n: sum(c["set"] == n for c in cs) for n, _ in SETS}
        print(f"{g}: {len(cs)} clips {per}, between RMS median "
              f"{np.median([c['end_rms_between'] for c in cs]):.3f}, "
              f"distinct CoC mean {np.mean([c['n_text'] for c in cs]):.2f}")
    print("->", out)


if __name__ == "__main__":
    main()
