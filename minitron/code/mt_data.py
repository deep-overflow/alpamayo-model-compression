"""Shared data rules for the Minitron baseline (minitron/plans/2026-10-23_plan.md, D2 / D10).

R1 (decided 2026-10-24): per clip, distill on a window whose teacher-CoC head clause does
not assert the opposite of the GT bucket -- CoC decel/stop while GT keeps or gains speed,
or CoC accelerate while GT stops. Heads that map to no bucket ("Keep distance", "Adapt
speed", "Nudge ..."; 54% of windows) are kept: R1 drops confirmed contradictions only.
Measured on coc_train x bucket_pool_full: 90.5% of windows and 9,905 of 9,917 clips pass,
with the GT bucket mix of one window per clip essentially unchanged.
"""

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "evaluation"))

from coc_action_consistency import HEAD2BUCKET, head

# teacher (full-model) CoC rollouts on official train: 119,260 windows / 10,040 clips
COC_TRAIN = Path("/mnt/nvme1n1/ad_vla/data/coc_generated/train/coc_train.parquet")
# GT-path bucket per (clip, t0), eval_lib priority decel_stop > turn > accel > cruise
BUCKETS = REPO / "outputs" / "recovery_sets" / "bucket_pool_full.parquet"


def r1_windows():
    """Windows that pass R1: clip_id, t0_us, chunk, coc, minADE, gt_bucket, coc_bucket."""
    d = pd.read_parquet(COC_TRAIN)
    b = pd.read_parquet(BUCKETS)[["clip_id", "t0_us", "bucket"]]
    m = d.merge(b, on=["clip_id", "t0_us"], how="inner")
    m["gt_bucket"] = m["bucket"].replace({"turn_left": "turn", "turn_right": "turn"})
    m["coc_bucket"] = m["coc"].map(head).map(HEAD2BUCKET)
    opposite = (((m["coc_bucket"] == "decel_stop") & m["gt_bucket"].isin(["accel", "cruise"]))
                | ((m["coc_bucket"] == "accel") & (m["gt_bucket"] == "decel_stop")))
    cols = ["clip_id", "t0_us", "chunk", "coc", "minADE", "gt_bucket", "coc_bucket"]
    return m.loc[~opposite, cols].reset_index(drop=True)
