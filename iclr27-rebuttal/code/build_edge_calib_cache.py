"""Edge input cache for the calib_100 clips (t0 = 5.1 s), streamed from the Hub through Edge's own loader.

Edge's input profile is 6 cameras / 22 frames (cameras 1 and 6 at -2/-1/-0.5/-0.1/0 s, cameras 0/2/3/5 at -1/-0.1/0 s)
and 20 history poses (-1.9..0 s). The A2 Super cache (calib_a2: 7 cameras x 4 frames, 16 poses) cannot serve it.
Edge's loader reads no future, so the 64 GT poses (t0 + 0.1..6.4 s) are read from the same egomotion and moved into the
t0 frame with the loader's own math (t0_inv.apply(x - x_t0), t0_inv * R).

npz per clip: 22 JPEGs (q95, the round trip every other cache uses), camera_indices (22,), absolute_timestamps (22,),
ego_history_{xyz,rot} (1, 20, ...), ego_future_{xyz,rot} (1, 64, ...), t0_us, clip_id, dataset_revision.
No navigation text is stored (plan decision Q1).

--verify checks the GT against calib_a2 (A2 Super loader, same clips and t0): the future must match, and the last 16
history poses must match A2's 16.

Usage (Edge venv):
  /mnt/nvme1n1/ad_vla/venvs/alpamayo2-edge/bin/python iclr27-rebuttal/code/build_edge_calib_cache.py --workers 8
  /mnt/nvme1n1/ad_vla/venvs/alpamayo2-edge/bin/python iclr27-rebuttal/code/build_edge_calib_cache.py --verify
"""

import argparse
import io
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import edge_common as ec

ec.setup_hf_env()

import numpy as np
import pandas as pd

A2_CACHE = ec.AV / "pre_processed" / "calib_a2"
_AVDI = None


def _avdi():
    global _AVDI
    if _AVDI is None:
        import physical_ai_av

        _AVDI = physical_ai_av.PhysicalAIAVDatasetInterface()
    return _AVDI


def build_one(task):
    clip_id, t0_us, root = task
    out = ec.path_for(clip_id, t0_us, root)
    if out.exists():
        return {"clip_id": clip_id, "status": "skip", "bytes": out.stat().st_size}
    from alpamayo2_edge.load_physical_aiavdataset import load_physical_aiavdataset
    from PIL import Image

    t = time.time()
    try:
        avdi = _avdi()
        d = load_physical_aiavdataset(clip_id, t0_us=t0_us, nav_text=ec.NAV_PLACEHOLDER, avdi=avdi)
        ego = avdi.get_clip_feature(clip_id, avdi.features.LABELS.EGOMOTION, maybe_stream=True)
        t_now = ego(np.array([t0_us], dtype=np.int64))
        fut = ego(t0_us + np.arange(100_000, 6_400_001, 100_000, dtype=np.int64))
        inv = t_now.pose.rotation[-1].inv()
        x0 = np.asarray(t_now.pose.translation)[-1]
        fx = inv.apply(np.asarray(fut.pose.translation) - x0)  # (64, 3)
        fr = (inv * fut.pose.rotation).as_matrix()  # (64, 3, 3)
    except Exception as e:  # noqa: BLE001  per-clip failures are recorded, not fatal
        return {"clip_id": clip_id, "status": "error", "error": f"{type(e).__name__}: {str(e)[:300]}"}
    frames = d["image_frames"].numpy()  # (22, 1, 3, H, W) uint8
    jpegs = []
    for i in range(frames.shape[0]):
        buf = io.BytesIO()
        Image.fromarray(np.transpose(frames[i, 0], (1, 2, 0))).save(buf, format="JPEG", quality=95)
        jpegs.append(buf.getvalue())
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".npz.tmp")
    with open(tmp, "wb") as fh:
        np.savez(
            fh,
            jpeg_bytes=np.array(jpegs, dtype=object),
            H=np.int32(frames.shape[3]), W=np.int32(frames.shape[4]),
            camera_indices=d["camera_indices"].numpy(),
            absolute_timestamps=d["absolute_timestamps"].numpy(),
            ego_history_xyz=d["ego_history_xyz"].numpy(),
            ego_history_rot=d["ego_history_rot"].numpy(),
            ego_future_xyz=fx[None].astype(np.float32),
            ego_future_rot=fr[None].astype(np.float32),
            t0_us=np.int64(t0_us), clip_id=str(clip_id),
            dataset_revision=str(d["provenance"].get("dataset_revision", "")),
        )
    tmp.rename(out)
    return {"clip_id": clip_id, "status": "ok", "sec": round(time.time() - t, 1), "bytes": out.stat().st_size}


def verify(man, root):
    rows = []
    for r in man.itertuples():
        e, a = ec.path_for(r.clip_id, r.t0_us, root), ec.path_for(r.clip_id, r.t0_us, A2_CACHE)
        if not e.exists() or not a.exists():
            rows.append({"clip_id": r.clip_id, "missing": True})
            continue
        ze, za = np.load(e, allow_pickle=True), np.load(a, allow_pickle=True)
        rows.append({
            "clip_id": r.clip_id, "missing": False,
            "future_xyz": float(np.abs(ze["ego_future_xyz"][0] - za["ego_future_xyz"][0, 0]).max()),
            "future_rot": float(np.abs(ze["ego_future_rot"][0] - za["ego_future_rot"][0, 0]).max()),
            "hist16_xyz": float(np.abs(ze["ego_history_xyz"][0, -16:] - za["ego_history_xyz"][0, 0]).max()),
            "hist16_rot": float(np.abs(ze["ego_history_rot"][0, -16:] - za["ego_history_rot"][0, 0]).max()),
            "n_frames": len(ze["jpeg_bytes"]),
        })
    df = pd.DataFrame(rows)
    ok = df[~df.missing]
    worst = {k: float(ok[k].max()) for k in ("future_xyz", "future_rot", "hist16_xyz", "hist16_rot")}
    passed = bool(len(ok) == len(man) and max(worst.values()) < 1e-4 and (ok.n_frames == 22).all())
    print(f"verify vs calib_a2: {len(ok)}/{len(man)} compared, max |diff| {worst}, all 22 frames "
          f"{bool((ok.n_frames == 22).all())} -> {'PASS' if passed else 'FAIL'}", flush=True)
    (root / "verify.json").write_text(json.dumps({"passed": passed, "max_abs_diff": worst, "n": len(ok)}, indent=2))
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--manifest", default=str(ec.CALIB_MANIFEST))
    ap.add_argument("--root", default=str(ec.CALIB_CACHE))
    args = ap.parse_args()
    root = Path(args.root)
    man = pd.read_parquet(args.manifest)
    if args.limit:
        man = man.head(args.limit)
    if args.verify:
        sys.exit(0 if verify(man, root) else 1)
    (root / "samples").mkdir(parents=True, exist_ok=True)
    tasks = [(r.clip_id, int(r.t0_us), root) for r in man.itertuples()]
    done, errors, t0 = [], [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(build_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            (errors if r["status"] == "error" else done).append(r)
            if r["status"] == "error":
                print(f"  error {r['clip_id'][:8]}: {r['error']}", flush=True)
            if i % 10 == 0 or i == len(tasks):
                el = time.time() - t0
                print(f"[{i}/{len(tasks)}] ok={len(done)} err={len(errors)} "
                      f"{sum(x.get('bytes', 0) for x in done) / 2**30:.2f} GB  elapsed {el / 60:.1f} min", flush=True)
    (root / "errors.json").write_text(json.dumps(errors, indent=2))
    n = len(list((root / "samples").glob("*.npz")))
    gb = sum(p.stat().st_size for p in (root / "samples").glob("*.npz")) / 2**30
    (root / "summary.txt").write_text(
        f"Alpamayo 2 Edge input cache (calib_100)\nclips in manifest: {len(man)}\nbuilt: {n}   failed: {len(errors)}\n"
        f"size: {gb:.2f} GB\nschema: 22 jpegs (cams 1/6 x 5, cams 0/2/3/5 x 3, q95) + 20 history + 64 future poses\n"
        f"navigation: none stored (plan decision Q1)\nbuilder: iclr27-rebuttal/code/build_edge_calib_cache.py\n")
    print(f"done: {n} samples, {gb:.2f} GB, {len(errors)} errors -> {root}", flush=True)


if __name__ == "__main__":
    main()
