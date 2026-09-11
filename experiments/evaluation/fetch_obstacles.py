"""Fetch obstacle.offline labels for the evaluation sets, one clip at a time.

The chunks these sets span are almost entirely not downloaded -- 0/500, 0/500, 0/262
against 96/100 for calib_100, because only the calibration chunks were ever pulled.
Taking whole chunks would be ~34 GB for a few hundred clips; `maybe_stream=True` hands
zipfile a seekable HfFileSystem so only the requested member crosses the wire, the same
trick build_cache.py uses to turn ~1 TB into 24 minutes.

Results land per clip under labels/obstacle_perclip/, NOT rewritten into chunk zips:
the zips are read-only shared data and a clip written into one would be invisible to
anyone checking the chunk's completeness.

Gated repo -- needs the `full_right` token, which lives on cvlab21 only.

    python experiments/evaluation/fetch_obstacles.py --sets test_500 indist_500 ood_val
"""

import argparse
import configparser
import os
import sys
import time
from pathlib import Path

import pandas as pd

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HF_HUB_CACHE", "/mnt/nvme1n1/ad_vla/cache/hub")
if "HF_TOKEN" not in os.environ:
    _cp = configparser.ConfigParser()
    _cp.read(os.path.expanduser("~/.cache/huggingface/stored_tokens"))
    if "full_right" in _cp:
        os.environ["HF_TOKEN"] = _cp["full_right"]["hf_token"]

DATA = Path("/mnt/nvme1n1/ad_vla/data/physicalai_av")
OUT = DATA / "labels" / "obstacle_perclip"
O = Path("/mnt/nvme1n1/ad_vla/outputs/chan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["test_500", "indist_500", "ood_val"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import physical_ai_av

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(local_dir=str(DATA))
    OUT.mkdir(parents=True, exist_ok=True)

    want, chunks = [], set()
    for s in args.sets:
        man = pd.read_parquet(O / "eval_sets" / f"{s}.parquet")
        want += [str(c) for c in man.clip_id]
        chunks |= {int(c) for c in man.chunk}
    want = sorted(set(want))
    # not every clip has this feature -- 96% of the evaluation sets do. metadata says so
    # up front, so the missing ones are skipped rather than discovered as a KeyError per
    # clip after paying for the request.
    fp = DATA / "metadata" / "feature_presence.parquet"
    if fp.exists():
        pres = pd.read_parquet(fp).reindex(want)["obstacle.offline"].fillna(False)
        absent = [c for c in want if not pres.get(c, False)]
        want = [c for c in want if pres.get(c, False)]
        print(f"라벨 없는 클립 {len(absent)}개 제외", flush=True)
    if args.limit:
        want = want[:args.limit]
        chunks = set()
    print(f"대상 클립 {len(want)}개 ({', '.join(args.sets)})", flush=True)

    # the ego footprint comes from vehicle_dimensions, and those chunks are as absent as
    # the obstacle ones -- a tiny per-chunk parquet covering ~100 clips each, so they are
    # fetched whole rather than streamed per clip
    missing = [c for c in sorted(chunks)
               if not (DATA / "calibration" / "vehicle_dimensions"
                       / f"vehicle_dimensions.chunk_{c:04d}.parquet").exists()]
    if missing:
        print(f"vehicle_dimensions 청크 {len(missing)}개 내려받는 중...", flush=True)
        for j, c in enumerate(missing):
            try:
                avdi.download_file(
                    f"calibration/vehicle_dimensions/vehicle_dimensions.chunk_{c:04d}.parquet")
            except Exception as e:  # noqa: BLE001
                print(f"  ERROR chunk {c}: {type(e).__name__}: {str(e)[:100]}", flush=True)
            if (j + 1) % 50 == 0:
                print(f"  [{j + 1}/{len(missing)}]", flush=True)
        print("vehicle_dimensions 완료", flush=True)

    ok = skip = err = absent_late = 0
    t0 = time.time()
    for i, clip_id in enumerate(want):
        p = OUT / f"{clip_id}.parquet"
        if p.exists():
            skip += 1
            continue
        try:
            d = avdi.get_clip_feature(clip_id, "obstacle.offline", maybe_stream=True)
        except KeyError:
            # the member is not in the chunk: the dataset has no labels for this clip.
            # A normal state, not a failure -- 4% of the evaluation sets are like this,
            # and conflating it with a real error makes the exit code useless to a caller.
            absent_late += 1
            continue
        except Exception as e:  # noqa: BLE001  per-clip failures are recorded, not fatal
            err += 1
            print(f"  ERROR {clip_id[:8]} {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        if isinstance(d, dict):  # zip features come back keyed by member name
            d = next(iter(d.values()))
        d.to_parquet(p)
        ok += 1
        if (ok + skip) % 25 == 0:
            el = time.time() - t0
            done = i + 1
            print(f"  [{done}/{len(want)}] 받음 {ok} 건너뜀 {skip} 실패 {err}  "
                  f"{el / max(ok, 1):.1f}s/클립  누적 "
                  f"{sum(f.stat().st_size for f in OUT.glob('*.parquet')) / 1e6:.0f} MB",
                  flush=True)

    size = sum(f.stat().st_size for f in OUT.glob("*.parquet")) / 1e6
    print(f"\n완료: 받음 {ok} 건너뜀 {skip} 라벨없음 {absent_late} 실패 {err}"
          f" / {len(want)}")
    print(f"저장 위치 {OUT}  총 {size:.0f} MB")
    if err:
        sys.exit(1)


if __name__ == "__main__":
    main()
