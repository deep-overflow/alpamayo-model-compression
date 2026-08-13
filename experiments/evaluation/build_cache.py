"""Stream an arbitrary clip manifest into the per-clip npz cache.

`build_ood_cache.py` extracts the OOD manifest from ood_reasoning.parquet and builds its
cache in one step; this is the general form, for a manifest that already exists — such as
the test-split evaluation set, whose camera chunks were never downloaded.

Clips are fetched one at a time over the wire: `open_file(maybe_stream=True)` hands
zipfile a seekable HfFileSystem object, so only that clip's members move, not the 5 GB
chunk that contains it. Storage is ~6 MB/clip against ~57 MB of mp4.

The npz schema matches `pre_processed/eval`, so `sample_cache.load_cached` reads either.

Resumable: samples already on disk are skipped.

Usage:
  python experiments/evaluation/build_cache.py \
      --manifest outputs/eval_sets/test_500.parquet --cache test --workers 16
"""

import argparse
import io
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
AV = Path("/mnt/nvme1n1/ad_vla/data/physicalai_av")
PRE = AV / "pre_processed"


def build_one(task):
    """Load one clip at its t0 and write the npz. Returns a status dict."""
    clip_id, t0_us, chunk, out_root = task
    out = Path(out_root) / "samples" / f"{clip_id}__t0_{t0_us}.npz"
    if out.exists():
        return {"clip_id": clip_id, "status": "skip", "bytes": out.stat().st_size}

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
            buf = io.BytesIO()
            Image.fromarray(np.transpose(frames[c, f], (1, 2, 0))).save(
                buf, format="JPEG", quality=95)
            jpegs.append(buf.getvalue())

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".npz.tmp")
    with open(tmp, "wb") as fh:  # np.savez appends ".npz" to a path lacking it
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
        )
    tmp.rename(out)
    return {"clip_id": clip_id, "status": "ok", "sec": time.time() - t,
            "bytes": out.stat().st_size}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="parquet with clip_id, t0_us, chunk")
    ap.add_argument("--cache", required=True, help="subdirectory under pre_processed/")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out_root = PRE / args.cache
    (out_root / "samples").mkdir(parents=True, exist_ok=True)
    man = pd.read_parquet(REPO / args.manifest if not Path(args.manifest).is_absolute()
                          else args.manifest)
    if args.limit:
        man = man.head(args.limit)
    print(f"{len(man)} clips -> {out_root}", flush=True)

    tasks = [(r.clip_id, int(r.t0_us), int(r.chunk), str(out_root))
             for r in man.itertuples()]
    done, errors, t0 = [], [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(build_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            (errors if r["status"] == "error" else done).append(r)
            if i % 25 == 0 or i == len(tasks):
                el = time.time() - t0
                gb = sum(x.get("bytes", 0) for x in done) / 2**30
                print(f"[{i}/{len(tasks)}] ok={len(done)} err={len(errors)} {gb:.2f} GB "
                      f"elapsed {el / 60:.1f}m eta {el / i * (len(tasks) - i) / 60:.1f}m",
                      flush=True)

    index = sorted(p.stem for p in (out_root / "samples").glob("*.npz"))
    (out_root / "index.json").write_text(json.dumps(index))
    (out_root / "errors.json").write_text(json.dumps(errors, indent=2))
    total_gb = sum(p.stat().st_size for p in (out_root / "samples").glob("*.npz")) / 2**30
    (out_root / "summary.txt").write_text(
        f"cache {args.cache} from {args.manifest}\n"
        f"requested {len(man)}   built {len(index)}   failed {len(errors)}\n"
        f"size {total_gb:.2f} GB\n"
        f"schema: same as pre_processed/eval (16 jpegs = 4 cam x 4 frame), JPEG quality 95\n")
    print(f"\ndone: {len(index)} samples, {total_gb:.2f} GB, {len(errors)} errors -> {out_root}",
          flush=True)


if __name__ == "__main__":
    main()
