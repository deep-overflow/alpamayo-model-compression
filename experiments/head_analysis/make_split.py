"""Create the fixed train/val/test clip split used by every pruning experiment.

Calibration clips for importance estimation are drawn from train only, so the
test set stays clean.

Usage: python make_split.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/workspace/alpamayo-model-compression")
SEED = 20260721
N_VAL, N_TEST, N_CALIB = 131, 150, 50


def main():
    clips = pd.read_parquet(REPO / "notebooks" / "clip_ids.parquet")["clip_id"].tolist()
    rng = np.random.RandomState(SEED)
    order = rng.permutation(len(clips))
    ids = [clips[i] for i in order]

    test = ids[:N_TEST]
    val = ids[N_TEST : N_TEST + N_VAL]
    train = ids[N_TEST + N_VAL :]
    calib = train[:N_CALIB]

    out = REPO / "outputs" / "split.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "seed": SEED, "source": "notebooks/clip_ids.parquet", "n_total": len(clips),
        "train": train, "val": val, "test": test, "calib": calib,
    }, indent=2))
    print(f"total {len(clips)} -> train {len(train)} / val {len(val)} / test {len(test)}")
    print(f"calibration {len(calib)} (subset of train)")
    assert not (set(test) & set(train)) and not (set(test) & set(val))
    assert set(calib) <= set(train)
    print("saved ->", out)


if __name__ == "__main__":
    main()
