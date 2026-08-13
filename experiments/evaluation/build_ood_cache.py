"""Build the OOD evaluation sample cache from ood_reasoning.parquet.

Each OOD clip is loaded at its own `event_start_timestamp` (the moment the OOD
situation occurs), not at the fixed t0=5.1s the in-distribution cache uses, so the
existing `pre_processed/{eval,train}` caches cannot serve OOD evaluation.

Clips whose chunk zip is not on disk are streamed from HuggingFace: `open_file`
hands zipfile a seekable HfFileSystem object, so only the clip's own members are
fetched over the wire. Downloading the 840 missing chunks outright would move
4.28 TB; streaming per clip moves a small fraction of that.

Samples are written in the SAME schema as `pre_processed/eval` (16 JPEGs =
4 cameras x 4 frames, plus ego history/future), so the in-distribution and OOD
axes go through identical preprocessing and their numbers stay comparable.
Raw frames would be ~52 MB/clip; JPEG brings that to ~2.2 MB/clip.

Resumable: samples already on disk are skipped, so re-running after an
interruption continues where it stopped.

Usage:
  python experiments/head_analysis/build_ood_cache.py --workers 12
  python experiments/head_analysis/build_ood_cache.py --limit 20   # smoke
"""

import argparse
import io
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
AV = Path("/mnt/nvme1n1/ad_vla/data/physicalai_av")
OUT = AV / "pre_processed" / "ood"

CAM_DIR = AV / "camera" / "camera_front_wide_120fov"


def event_of(row):
    """First event carrying both a start timestamp and curated CoC text."""
    ev = row["events"]
    if not isinstance(ev, str):
        return None
    for e in json.loads(ev):
        if "event_start_timestamp" in e and "coc" in e:
            return e
    return None


def manifest():
    """OOD clips that have a usable event, joined with split/chunk metadata."""
    ci = pd.read_parquet(AV / "clip_index.parquet")
    ood = pd.read_parquet(AV / "reasoning" / "ood_reasoning.parquet")
    rows = []
    for cid, r in ood.iterrows():
        if cid not in ci.index:
            continue
        e = event_of(r)
        if e is None:
            continue
        rows.append({
            "clip_id": cid,
            "t0_us": int(e["event_start_timestamp"]),
            "gt_coc": e["coc"],
            "cluster": r["event_cluster"],
            "split": r["split"],
            "chunk": int(ci.loc[cid, "chunk"]),
        })
    return pd.DataFrame(rows)


def sample_path(clip_id, t0_us):
    return OUT / "samples" / f"{clip_id}__t0_{t0_us}.npz"


def build_one(task):
    """Load one clip at its event timestamp and write the npz. Returns a status dict."""
    clip_id, t0_us, gt_coc, cluster, split, chunk = task
    out = sample_path(clip_id, t0_us)
    if out.exists():
        return {"clip_id": clip_id, "status": "skip", "bytes": out.stat().st_size}

    # imported here so each worker process sets up its own dataset interface
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
    from PIL import Image

    t = time.time()
    try:
        d = load_physical_aiavdataset(clip_id, t0_us=t0_us, maybe_stream=True)
    except Exception as e:  # noqa: BLE001  per-clip failures are recorded, not fatal
        return {"clip_id": clip_id, "status": "error",
                "error": f"{type(e).__name__}: {str(e)[:200]}"}

    frames = d["image_frames"].numpy()  # (n_cam, n_frame, 3, H, W) uint8
    n_cam, n_frame = frames.shape[0], frames.shape[1]
    jpegs = []
    for c in range(n_cam):
        for f in range(n_frame):
            img = np.transpose(frames[c, f], (1, 2, 0))  # (H, W, 3)
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="JPEG", quality=95)
            jpegs.append(buf.getvalue())

    out.parent.mkdir(parents=True, exist_ok=True)
    # np.savez appends ".npz" to a path that lacks it, so hand it an open handle
    # instead and keep the temp name exact; the rename is what makes it atomic.
    tmp = out.with_suffix(".npz.tmp")
    with open(tmp, "wb") as fh:
        np.savez(
            fh,
            jpeg_bytes=np.array(jpegs, dtype=object),
            n_cam=np.int32(n_cam), n_frame=np.int32(n_frame),
            H=np.int32(frames.shape[3]), W=np.int32(frames.shape[4]),
            camera_indices=d["camera_indices"].numpy(),
            ego_history_xyz=d["ego_history_xyz"].numpy(),
            ego_history_rot=d["ego_history_rot"].numpy(),
            ego_future_xyz=d["ego_future_xyz"].numpy(),
            ego_future_rot=d["ego_future_rot"].numpy(),
            t0_us=np.int64(t0_us), clip_id=str(clip_id), chunk=np.int32(chunk),
            gt_coc=str(gt_coc), cluster=str(cluster), split=str(split),
        )
    tmp.rename(out)
    return {"clip_id": clip_id, "status": "ok", "sec": time.time() - t,
            "bytes": out.stat().st_size}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None, help="build only the first N (smoke)")
    args = ap.parse_args()

    (OUT / "samples").mkdir(parents=True, exist_ok=True)
    man = manifest()
    if args.limit:
        man = man.head(args.limit)

    local = {int(f.split("chunk_")[1].split(".zip")[0])
             for f in os.listdir(CAM_DIR) if "chunk_" in f}
    n_local = int(man["chunk"].isin(local).sum())
    print(f"OOD clips with a usable event: {len(man)}", flush=True)
    print(f"  chunk on disk {n_local} | streamed {len(man) - n_local}", flush=True)
    print(f"  split: {man['split'].value_counts().to_dict()}", flush=True)

    man.to_parquet(OUT / "manifest.parquet", index=False)

    tasks = [tuple(r) for r in man[
        ["clip_id", "t0_us", "gt_coc", "cluster", "split", "chunk"]].itertuples(index=False)]
    done, errors, t0 = [], [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(build_one, t): t[0] for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r["status"] == "error":
                errors.append(r)
            else:
                done.append(r)
            if i % 25 == 0 or i == len(tasks):
                el = time.time() - t0
                gb = sum(x.get("bytes", 0) for x in done) / 2**30
                eta = el / i * (len(tasks) - i) / 3600
                print(f"[{i}/{len(tasks)}] ok={len(done)} err={len(errors)} "
                      f"{gb:.2f} GB  elapsed {el / 3600:.2f}h  eta {eta:.2f}h", flush=True)

    index = sorted(p.stem for p in (OUT / "samples").glob("*.npz"))
    (OUT / "index.json").write_text(json.dumps(index))
    (OUT / "errors.json").write_text(json.dumps(errors, indent=2))
    total_gb = sum(p.stat().st_size for p in (OUT / "samples").glob("*.npz")) / 2**30
    (OUT / "summary.txt").write_text(
        f"OOD sample cache\n"
        f"clips with usable event: {len(man)}\n"
        f"built: {len(index)}   failed: {len(errors)}\n"
        f"size: {total_gb:.2f} GB\n"
        f"schema: same as pre_processed/eval (16 jpegs = 4 cam x 4 frame) "
        f"+ gt_coc/cluster/split\n"
        f"t0_us: per-clip event_start_timestamp (NOT the fixed 5.1s used in-distribution)\n")
    print(f"\ndone: {len(index)} samples, {total_gb:.2f} GB, {len(errors)} errors -> {OUT}",
          flush=True)


if __name__ == "__main__":
    main()
