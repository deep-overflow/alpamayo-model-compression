"""Cache every 30 Hz camera frame in [t0 - window, t0] for a manifest, for timing probes.

`build_cache.py` keeps only the 4 frames per camera the model reads (t0-0.3 .. t0 at
10 Hz). Shifting a camera in time needs the frames in between and before, so this keeps
the whole window per camera with each frame's capture timestamp; `run_baseline.py
--lag-ms/--stride-ms` then picks frames out of it with the loader's own rule (closest
frame at or before each requested time).

Frames are decoded and JPEG-encoded (quality 95) exactly as `build_cache.py` does, so
the 4 frames the loader would pick must come out byte-identical to the existing cache.
Every clip is checked against `--ref-cache` and the count of matching frames is written
to the status rows (G0); a mismatch means the two caches would not be comparable.

Usage:
  python experiments/evaluation/build_timing_cache.py \
      --manifest outputs/eval_sets/test_2000.parquet --cache test_timing --ref-cache test
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

# camera index -> feature, in the order the loaders sort into; Alpamayo 1.5 reads (0, 1, 2, 6),
# Alpamayo 2's trajectory profile adds the rear pair (3, 5)
CAMS = {0: "CAMERA_CROSS_LEFT_120FOV", 1: "CAMERA_FRONT_WIDE_120FOV",
        2: "CAMERA_CROSS_RIGHT_120FOV", 3: "CAMERA_REAR_LEFT_70FOV",
        5: "CAMERA_REAR_RIGHT_70FOV", 6: "CAMERA_FRONT_TELE_30FOV"}

_AVDI = None


def _avdi():
    global _AVDI
    if _AVDI is None:
        import physical_ai_av
        _AVDI = physical_ai_av.PhysicalAIAVDatasetInterface()
    return _AVDI


def build_one(task):
    clip_id, t0_us, out_root, ref_root, window_us, cams = task
    out = Path(out_root) / "samples" / f"{clip_id}__t0_{t0_us}.npz"
    if out.exists():
        return {"clip_id": clip_id, "status": "skip", "bytes": out.stat().st_size}

    from PIL import Image

    t = time.time()
    avdi = _avdi()
    ref = None
    if ref_root:
        ref = np.load(Path(ref_root) / "samples" / f"{clip_id}__t0_{t0_us}.npz", allow_pickle=True)
        assert list(ref["camera_indices"]) == cams, ref["camera_indices"]
    slots = np.array([t0_us - (3 - i) * 100_000 for i in range(4)], dtype=np.int64)
    arrays, n_match = {}, 0
    try:
        for c, ci in enumerate(cams):
            name = CAMS[ci]
            cam = avdi.get_clip_feature(clip_id, getattr(avdi.features.CAMERA, name),
                                        maybe_stream=True)
            ts = cam.timestamps                                      # (n_video,) int64 us
            lo = np.searchsorted(ts, t0_us - window_us, side="right") - 1
            hi = np.searchsorted(ts, t0_us, side="right") - 1
            idx = np.arange(max(lo, 0), hi + 1)                      # (n_win,)
            frames = cam.decode_images_from_frame_indices(idx)       # (n_win, H, W, 3) uint8
            cam.close()
            jpegs = []
            for f in frames:
                buf = io.BytesIO()
                Image.fromarray(f).save(buf, format="JPEG", quality=95)
                jpegs.append(buf.getvalue())
            win_ts = ts[idx]
            # G0: the loader's 4 frames, picked out of the window, against the ref cache
            if ref is not None:
                pick = np.searchsorted(win_ts, slots, side="right") - 1  # (4,)
                n_match += sum(jpegs[p] == ref["jpeg_bytes"][c * 4 + f]
                               for f, p in enumerate(pick))
            arrays[f"jpeg_c{ci}"] = np.array(jpegs, dtype=object)
            arrays[f"ts_c{ci}"] = win_ts.astype(np.int64)
    except Exception as e:  # noqa: BLE001  per-clip failures are recorded, not fatal
        return {"clip_id": clip_id, "status": "error",
                "error": f"{type(e).__name__}: {str(e)[:200]}"}

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".npz.tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, camera_indices=np.array(cams, dtype=np.int64),
                 t0_us=np.int64(t0_us), clip_id=str(clip_id),
                 g0_match=np.int32(n_match if ref is not None else -1), **arrays)
    tmp.rename(out)
    return {"clip_id": clip_id, "status": "ok", "sec": time.time() - t,
            "bytes": out.stat().st_size, "g0_match": int(n_match),
            "n_win": [len(arrays[f"ts_c{ci}"]) for ci in cams]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="parquet with clip_id, t0_us")
    ap.add_argument("--cache", required=True, help="subdirectory under pre_processed/")
    ap.add_argument("--ref-cache", required=True,
                    help="4-frame cache to check G0 against, or 'none' (no cache holds these cameras)")
    ap.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2, 6], choices=sorted(CAMS))
    ap.add_argument("--window-ms", type=int, default=1000)
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

    ref_root = None if args.ref_cache == "none" else str(PRE / args.ref_cache)
    n_ref = 4 * len(args.cams) if ref_root else -1  # g0_match of a fully matching clip
    tasks = [(r.clip_id, int(r.t0_us), str(out_root), ref_root, args.window_ms * 1000,
              args.cams) for r in man.itertuples()]
    done, errors, t0 = [], [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(build_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            (errors if r["status"] == "error" else done).append(r)
            if i % 25 == 0 or i == len(tasks):
                el = time.time() - t0
                gb = sum(x.get("bytes", 0) for x in done) / 2**30
                bad = sum(1 for x in done if ref_root and x.get("g0_match", n_ref) != n_ref)
                print(f"[{i}/{len(tasks)}] ok={len(done)} err={len(errors)} g0_bad={bad} "
                      f"{gb:.2f} GB elapsed {el / 60:.1f}m "
                      f"eta {el / i * (len(tasks) - i) / 60:.1f}m", flush=True)

    (out_root / "errors.json").write_text(json.dumps(errors, indent=1))
    paths = sorted((out_root / "samples").glob("*.npz"))
    index = [p.stem for p in paths]
    (out_root / "index.json").write_text(json.dumps(index))
    # G0 over everything on disk, resumed runs included: npz loads a key at a time
    g0_bad = [p.stem for p in paths if int(np.load(p)["g0_match"]) != n_ref]
    total_gb = sum(p.stat().st_size for p in paths) / 2**30
    want = {f"{r.clip_id}__t0_{int(r.t0_us)}" for r in man.itertuples()}
    complete = want <= set(index) and not g0_bad
    (out_root / "summary.txt").write_text(
        f"cache {args.cache} from {args.manifest}\n"
        f"requested {len(man)}   on disk {len(index)}   failed this run {len(errors)}\n"
        f"G0 vs {args.ref_cache}: {len(paths) - len(g0_bad)}/{len(paths)} clips with all "
        f"{n_ref} loader frames byte-identical\n"
        f"size {total_gb:.2f} GB\n"
        f"schema: per camera c in {tuple(args.cams)}: jpeg_c{{c}} (n_win,) JPEG q95, ts_c{{c}} "
        f"(n_win,) us, every frame in [t0 - {args.window_ms} ms, t0]\n")
    print(f"\ndone: {len(index)} samples, {total_gb:.2f} GB, {len(errors)} errors, "
          f"G0 mismatches {len(g0_bad)} -> {out_root}", flush=True)
    if complete:
        (out_root / "READY").write_text(f"{len(want)} clips, G0 vs {args.ref_cache}: all match\n")
    raise SystemExit(0 if complete else 1)


if __name__ == "__main__":
    main()
