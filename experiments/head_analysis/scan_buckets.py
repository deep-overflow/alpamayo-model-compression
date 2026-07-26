"""Scan clips for scenario buckets using ONLY the egomotion feature (no images).

Full load_physical_aiavdataset is 10-20s/clip because of image decoding; the
scenario bucket depends only on the GT trajectory, which comes from egomotion in
a fraction of that. This lets us scan hundreds of clips to find the rare turn /
accel clips that the val split under-sampled.

Usage: python scan_buckets.py --pool train --limit 500 --out bucket_scan
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import transformers.utils.hub as _hub  # noqa: E402

_orig = _hub.cached_files


def _patched(path_or_repo_id, *a, **kw):
    if str(path_or_repo_id) == "nvidia/Cosmos-Reason2-8B":
        kw["local_files_only"] = True
    return _orig(path_or_repo_id, *a, **kw)


_hub.cached_files = _patched

import eval_lib as el  # noqa: E402
from alpamayo1_5 import load_physical_aiavdataset as L  # noqa: E402

REPO = Path("/workspace/alpamayo-model-compression")


def make_avdi():
    if L.PHYSICAL_AV_LOCAL_DIR:
        return L._get_local_avdi(L.PHYSICAL_AV_LOCAL_DIR), False
    import physical_ai_av
    return physical_ai_av.PhysicalAIAVDatasetInterface(), True


def future_xy(avdi, clip_id, maybe_stream, reader_cache,
              t0_us=5_100_000, num_history_steps=16, num_future_steps=64, time_step=0.1):
    """Return the GT future xy path in the ego frame at t0 (T, 2) -- trajectory only."""
    egomotion = L._get_feature(avdi, clip_id, avdi.features.LABELS.EGOMOTION,
                               maybe_stream, reader_cache)
    hist_off = np.arange(-(num_history_steps - 1) * time_step * 1e6,
                         time_step * 1e6 / 2, time_step * 1e6).astype(np.int64)
    fut_off = np.arange(time_step * 1e6, (num_future_steps + 0.5) * time_step * 1e6,
                        time_step * 1e6).astype(np.int64)
    ego_h = egomotion(t0_us + hist_off)
    ego_f = egomotion(t0_us + fut_off)
    t0_xyz = ego_h.pose.translation[-1].copy()
    t0_rot_inv = L.spt.Rotation.from_quat(ego_h.pose.rotation.as_quat()[-1].copy()).inv()
    fut_local = t0_rot_inv.apply(ego_f.pose.translation - t0_xyz)  # (T, 3)
    return fut_local[:, :2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=str, default="train")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--exclude", nargs="*", default=["calib", "test"])
    ap.add_argument("--out", type=str, default="bucket_scan")
    args = ap.parse_args()

    split = json.loads((REPO / "outputs" / "split.json").read_text())
    excluded = set()
    for key in args.exclude:
        excluded |= set(split.get(key, []))
    pool = [c for c in split[args.pool] if c not in excluded][args.offset : args.offset + args.limit]

    avdi, maybe_stream = make_avdi()
    reader_cache = {}
    records, counts = [], {b: 0 for b in ["cruise", "turn", "decel_stop", "accel"]}
    t0 = time.time()
    for i, clip_id in enumerate(pool):
        try:
            xy = future_xy(avdi, clip_id, maybe_stream, reader_cache)
        except Exception as e:
            print(f"[{i}] {clip_id} FAILED {type(e).__name__}: {e}", flush=True)
            continue
        g = el.gt_geometry(xy)
        b = el.bucket(xy)
        counts[b] += 1
        records.append({"clip_id": clip_id, "bucket": b, **g})
        if (i + 1) % 25 == 0:
            print(f"[{i + 1}/{len(pool)}] {counts} ({time.time() - t0:.0f}s)", flush=True)

    out_dir = REPO / "outputs" / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scan.json").write_text(json.dumps({
        "pool": args.pool, "n_scanned": len(records), "counts": counts,
        "records": records,
    }, indent=2))
    print(f"\nscanned {len(records)} clips: {counts}")
    print("saved ->", out_dir / "scan.json")


if __name__ == "__main__":
    main()
