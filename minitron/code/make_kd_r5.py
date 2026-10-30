"""kd_r5: up to five R1 windows per KD clip (KISTI plan S3, minitron/plans/2026-10-26_kisti-training-plan.md).

Every clip keeps its kd_r1 window, so kd_r5 is a superset of kd_r1 and the existing cache and teacher
CoC carry over; up to four more R1 windows of the same clip are added at random (seed 0). Same clips,
same R1 rule (mt_data.r1_windows), same teacher CoC source (coc_train.parquet) -- only the number of
windows per clip changes.

Usage:
  .venv/bin/python minitron/code/make_kd_r5.py
Writes outputs/minitron/kd_sets/kd_r5.parquet (+ kd_r5_config.json).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import mt_data

OUT = mt_data.REPO / "outputs" / "minitron" / "kd_sets"
PER_CLIP, SEED = 5, 0


def main():
    kd1 = pd.read_parquet(OUT / "kd_r1.parquet")
    w = mt_data.r1_windows()
    w = w[w.clip_id.isin(set(kd1.clip_id))]
    key1 = set(zip(kd1.clip_id, kd1.t0_us.astype(int)))
    w["in_r1"] = [(c, int(t)) in key1 for c, t in zip(w.clip_id, w.t0_us)]
    assert w["in_r1"].sum() == len(kd1), "every kd_r1 window must pass R1"
    rng = np.random.default_rng(SEED)
    w = w.iloc[rng.permutation(len(w))]
    # kd_r1 window first, then the shuffled others; keep the first PER_CLIP of each clip
    w = w.sort_values("in_r1", ascending=False, kind="stable")
    sel = w.groupby("clip_id", sort=False).head(PER_CLIP).drop(columns="in_r1")
    sel = sel.sort_values(["clip_id", "t0_us"]).reset_index(drop=True)
    sel.to_parquet(OUT / "kd_r5.parquet", index=False)
    per = sel.groupby("clip_id").size()
    cfg = {"source": "mt_data.r1_windows() on the kd_r1 clips", "per_clip": PER_CLIP, "seed": SEED,
           "n_windows": len(sel), "n_clips": int(per.size), "contains_kd_r1": True,
           "windows_per_clip": per.value_counts().sort_index().to_dict(),
           "gt_bucket": sel.gt_bucket.value_counts(normalize=True).round(4).to_dict()}
    (OUT / "kd_r5_config.json").write_text(json.dumps(cfg, indent=2, default=int))
    # the 64 clips mtw24_kd held out (train_kd: seed-0 permutation of kd_r1, first 64 rows); kd_r5
    # holds out every window of the same clips, so its held-out loss is comparable with R0's
    perm = np.random.default_rng(0).permutation(len(kd1))
    held = sorted(kd1.iloc[perm[:64]].clip_id)
    (OUT / "heldout_clips_r0.json").write_text(json.dumps(held, indent=1))
    print(json.dumps(cfg, default=int))
    print(f"held-out clips {len(held)}, their kd_r5 windows {int(sel.clip_id.isin(held).sum())}")


if __name__ == "__main__":
    main()
