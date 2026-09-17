"""Detect the frame-slide artifact across rendered videos, without eyeballing each one.

The bug was a mismatch between the frame size the writer declared to ffmpeg and the size
of the buffers it handed over: every frame shifted by a row, so by the end of a clip the
previous frame's bottom sat at the top. Even dimensions are necessary to avoid it but not
sufficient evidence that a given file escaped it, and checking 200 videos by eye is not a
check.

The signal used here is that the top band of these figures is static: background plus the
suptitle, whose text does not change over a clip. In a correct render an early and a late
frame agree there almost exactly; under the slide they do not, because moving content has
migrated into it.

Reported per file as the max absolute difference over that band, 0-255.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def n_frames(video):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True).stdout.strip().rstrip(",")
    return int(out) if out.isdigit() else 0


def grab(video, n, out):
    """Extract one frame. The unlink is load-bearing: `select` matching nothing still exits
    0 and writes nothing, so a stale file from the previous video would be read as this
    one's frame -- which is exactly how two short clips were reported as torn when they
    were fine (165 and 162 frames against a requested frame 190)."""
    out.unlink(missing_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(video),
                    "-vf", f"select='eq(n\\,{n})'", "-vframes", "1", str(out)],
                   check=True)
    if not out.exists():
        raise FileNotFoundError(f"{video.name}: 프레임 {n} 없음")
    return np.asarray(Image.open(out).convert("L"), dtype=np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--early", type=int, default=5)
    ap.add_argument("--late", type=int, default=190)
    ap.add_argument("--band", type=int, default=30, help="top rows compared")
    ap.add_argument("--thresh", type=int, default=12,
                    help="max abs diff above this is called drift")
    args = ap.parse_args()

    vids = sorted(args.root.rglob("*.mp4"))
    print(f"{len(vids)}개 검사  (프레임 {args.early} vs {args.late}, 상단 {args.band}행)")
    bad = []
    with tempfile.TemporaryDirectory() as td:
        a_png, b_png = Path(td) / "a.png", Path(td) / "b.png"
        for v in vids:
            # a short clip has no frame 190; compare its own last frame instead of
            # silently falling back to whatever the previous video left behind
            nf = n_frames(v)
            late = min(args.late, nf - 2) if nf else args.late
            try:
                a = grab(v, args.early, a_png)[:args.band]
                b = grab(v, late, b_png)[:args.band]
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(f"  SKIP (프레임 추출 실패) {v.name}")
                continue
            if a.shape != b.shape:
                bad.append((v, -1))
                print(f"  DRIFT shape {a.shape} != {b.shape}  {v}")
                continue
            d = int(np.abs(a - b).max())
            if d > args.thresh:
                bad.append((v, d))
                print(f"  DRIFT maxdiff {d:3d}  {v.relative_to(args.root)}")

    print(f"\n밀림 의심 {len(bad)} / {len(vids)}")
    for v, d in bad:
        print(f"  {d:4d}  {v}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
