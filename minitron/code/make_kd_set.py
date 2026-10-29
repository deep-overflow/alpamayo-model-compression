"""Distillation manifest (plan D10-1) and the importance calibration subset (D9-1).

One R1 window per clip from the teacher-CoC pool on official train (mt_data.r1_windows),
picked at random -- not preferring windows that are already cached, because those were
chosen by the 2026-08 recovery sets (bucket balancing, teacher minADE <= 3.0) and would
carry that selection into the KD data. A random pick that happens to be cached is reused.
The calibration set is a random 1,024-window subset of the KD set, as Minitron draws its
calibration from the distillation data.

Asserts disjointness from every evaluation manifest (val500 / test500 / OOD / OOD-val /
the probe set val_official_238) -- they are official val/test, so this holds by
construction, and the assert keeps it that way if the pool ever changes.

Writes outputs/minitron/kd_sets/{kd_r1.parquet, calib_1024.parquet, config.json,
summary.txt}. Then stream the missing samples:
  ALPAMAYO_REPO=$PWD bash experiments/head_analysis/run_retry_host.sh 3 \\
      experiments/evaluation/build_cache.py \\
      --manifest outputs/minitron/kd_sets/kd_r1.parquet --cache train --workers 16
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPO / "experiments" / "evaluation"))

import mt_data
import sample_cache as sc

EVAL_MANIFESTS = ["eval_sets/indist_500.parquet", "eval_sets/val_500.parquet",
                  "eval_sets/test_500.parquet", "eval_sets/ood.parquet",
                  "eval_sets/ood_val.parquet", "recovery_sets/val_official_238.parquet"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--n-calib", type=int, default=1024)
    args = ap.parse_args()
    out = REPO / "outputs" / "minitron" / "kd_sets"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    r1 = mt_data.r1_windows()
    # shuffle, then the first window of each clip: a random window per clip
    kd = (r1.iloc[rng.permutation(len(r1))].drop_duplicates("clip_id")
          .sort_values(["clip_id", "t0_us"]).reset_index(drop=True))
    held_out = set()
    for m in EVAL_MANIFESTS:
        held_out |= set(pd.read_parquet(REPO / "outputs" / m)["clip_id"])
    overlap = held_out & set(kd["clip_id"])
    assert not overlap, f"{len(overlap)} KD clips are in an evaluation set"
    kd["cached"] = [sc.path_for("train", c, int(t)).exists()
                    for c, t in zip(kd["clip_id"], kd["t0_us"])]
    calib = kd.iloc[np.sort(rng.choice(len(kd), args.n_calib, replace=False))]

    kd.drop(columns="cached").to_parquet(out / "kd_r1.parquet", index=False)
    calib.drop(columns="cached").to_parquet(out / f"calib_{args.n_calib}.parquet", index=False)
    mix = kd["gt_bucket"].value_counts(normalize=True).round(3).to_dict()
    cfg = {**vars(args), "rule": "R1, one random window per clip", "pool": str(mt_data.COC_TRAIN),
           "n_windows_r1": len(r1), "n_kd": len(kd), "n_calib": len(calib),
           "already_cached": int(kd["cached"].sum()), "to_build": int((~kd["cached"]).sum()),
           "calib_cached": int(calib["cached"].sum()), "gt_bucket_mix": mix,
           "eval_manifests_checked": EVAL_MANIFESTS}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    lines = [f"KD set: {len(kd):,} clips (one R1 window each) from {len(r1):,} R1 windows",
             f"already cached {cfg['already_cached']:,}, to build {cfg['to_build']:,}",
             f"calibration subset {len(calib):,} (cached {cfg['calib_cached']:,})",
             f"GT bucket mix {mix}",
             f"disjoint from {len(EVAL_MANIFESTS)} evaluation manifests ({len(held_out):,} clips)"]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
