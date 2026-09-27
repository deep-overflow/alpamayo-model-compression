"""Pull real front-camera mp4s for the OOD open-loop qualitative deck's selected clips.

Reuses the streaming trick from `experiments/head_analysis/fetch_scene_camera.py`
(`HfFileSystem.open()` is seekable, so `zipfile` reads only the one `<clip_id>.<camera>.mp4`
member out of the multi-GB chunk zip) against `outputs/eval_sets/ood_val.parquet` directly,
since these are OOD-val clip_ids rather than alpasim scene_ids.

Usage:
  python experiments/evaluation/fetch_qual_videos.py --prefixes 892eb6e7 93274ef5 ... \
      --out outputs/ood_qual_videos
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
from fetch_scene_camera import pull, hf_token, DEFAULT_CAM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefixes", nargs="+", required=True)
    ap.add_argument("--manifest", default="outputs/eval_sets/ood_val.parquet")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--camera", default=DEFAULT_CAM)
    args = ap.parse_args()

    man = pd.read_parquet(REPO / args.manifest)
    man["pfx"] = man["clip_id"].str[:8]
    sub = man[man["pfx"].isin(args.prefixes)]
    missing = set(args.prefixes) - set(sub["pfx"])
    if missing:
        raise SystemExit(f"not found in manifest: {missing}")

    out = args.out if args.out.is_absolute() else REPO / args.out
    token = hf_token()
    for i, r in enumerate(sub.itertuples(), 1):
        dest = out / f"{r.clip_id[:8]}.mp4"
        p, sec, how = pull(r.clip_id, int(r.chunk), dest, args.camera, token)
        if p is None:
            print(f"[{i}/{len(sub)}] {r.clip_id[:8]} FAILED: {how}")
        else:
            print(f"[{i}/{len(sub)}] {p.name}  {p.stat().st_size / 1e6:5.1f} MB  "
                  f"{sec:4.1f}s  [{how}]")


if __name__ == "__main__":
    main()
