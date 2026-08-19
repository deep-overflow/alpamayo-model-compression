"""Read pre-processed evaluation samples instead of the camera chunk zips.

`pre_processed/eval` holds one npz per val clip at t0=5.1s (the t0 every
in-distribution runner uses), and `pre_processed/ood` holds one per OOD clip at
that clip's own event timestamp. Each npz stores the 16 JPEG frames the model
actually consumes (4 cameras x 4 frames) plus the ego trajectory, which is 2-7 MB
against ~57 MB of mp4 per clip -- so the chunk zips (0.97 TB for val alone) are
only needed to *build* the cache, not to evaluate from it.

The val chunk zips were deleted on 2026-08-06 once this cache was verified against
a direct load; the train ones are still on disk, and the `train` cache that used to
sit beside `eval` was dropped because it was locked to t0=5.1s. So train clips load
through `load_physical_aiavdataset` as before, val and OOD clips through here.

The evaluation pipeline reads exactly six keys off the loader's return value
(`image_frames`, `camera_indices`, `ego_history_{xyz,rot}`, `ego_future_{xyz,rot}`);
timestamps are never used. `load_cached` returns those, so it is a drop-in for
`load_physical_aiavdataset` in `analysis_lib.build_inputs` and in the eval runners.

Frames come back through a JPEG round-trip, so they are close to but not
bit-identical with a direct load. Every model in a comparison reads the same
cache, so paired differences are unaffected; only cross-cache absolute
comparisons would carry the small encoding difference.
"""

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

AV = Path("/mnt/nvme1n1/ad_vla/data/physicalai_av")
PRE = AV / "pre_processed"

KEYS = ("image_frames", "camera_indices", "ego_history_xyz", "ego_history_rot",
        "ego_future_xyz", "ego_future_rot")

CALIB_T0 = 5_100_000


def clip_seed(global_seed, clip_id):
    """Stable per-clip seed. Independent of order, sharding and restarts.

    Every runner derives its seeds from this rather than from a loop index, so a
    sharded run and a whole run give the same result for the same clip.
    """
    h = hashlib.sha256(f"{global_seed}:{clip_id}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def calib_clips(repo, manifest="calib_100"):
    """Calibration clip ids.

    The manifest under `outputs/eval_sets/` is the current definition: drawn from
    official train only, distribution-matched, disjoint from every evaluation set.
    Passing an empty manifest falls back to `split.json`'s `calib`, which predates the
    official-split protocol and mixes val/test clips in -- kept only so older runs stay
    reproducible.
    """
    if not manifest:
        return json.loads((Path(repo) / "outputs" / "split.json").read_text())["calib"]
    p = Path(repo) / "outputs" / "eval_sets" / f"{manifest}.parquet"
    return pd.read_parquet(p)["clip_id"].tolist()


def index(split):
    """Sample stems available for a cache split ('eval', 'train', 'ood')."""
    return json.loads((PRE / split / "index.json").read_text())


def calib_samples(repo, manifest="calib_100"):
    """(clip_id, t0_us) for each calibration clip.

    calib_100 sits at the fixed CALIB_T0, but a calibration set drawn from the OOD pool
    carries its own per-clip t0 -- reading it from the manifest keeps the caller from
    assuming one. Falls back to split.json at CALIB_T0 when the manifest is empty, for
    the pre-2026-08-07 runs.
    """
    if not manifest:
        return [(c, CALIB_T0) for c in calib_clips(repo, manifest)]
    df = pd.read_parquet(Path(repo) / "outputs" / "eval_sets" / f"{manifest}.parquet")
    if "t0_us" not in df.columns:
        return [(c, CALIB_T0) for c in df["clip_id"]]
    return list(zip(df["clip_id"], df["t0_us"].astype(int)))


def path_for(split, clip_id, t0_us=5_100_000):
    return PRE / split / "samples" / f"{clip_id}__t0_{t0_us}.npz"


def load_cached(path, device=None):
    """Rebuild the loader's return dict from a cached npz.

    Returns image_frames (n_cam, n_frame, 3, H, W) uint8 and the ego tensors with
    their original (1, 1, T, ...) shapes.
    """
    z = np.load(path, allow_pickle=True)
    n_cam, n_frame = int(z["n_cam"]), int(z["n_frame"])
    H, W = int(z["H"]), int(z["W"])
    frames = np.empty((n_cam, n_frame, 3, H, W), dtype=np.uint8)
    for c in range(n_cam):
        for f in range(n_frame):
            img = np.asarray(Image.open(io.BytesIO(z["jpeg_bytes"][c * n_frame + f])))
            frames[c, f] = np.transpose(img, (2, 0, 1))  # (H, W, 3) -> (3, H, W)
    out = {
        "image_frames": torch.from_numpy(frames),
        "camera_indices": torch.from_numpy(z["camera_indices"]),
        "ego_history_xyz": torch.from_numpy(z["ego_history_xyz"]),
        "ego_history_rot": torch.from_numpy(z["ego_history_rot"]),
        "ego_future_xyz": torch.from_numpy(z["ego_future_xyz"]),
        "ego_future_rot": torch.from_numpy(z["ego_future_rot"]),
        "t0_us": int(z["t0_us"]),
        "clip_id": str(z["clip_id"]),
    }
    for extra in ("gt_coc", "cluster", "split", "chunk"):  # OOD cache only
        if extra in z.files:
            out[extra] = z[extra].item() if z[extra].shape == () else z[extra]
    if device is not None:
        for k in KEYS:
            out[k] = out[k].to(device)
    return out
