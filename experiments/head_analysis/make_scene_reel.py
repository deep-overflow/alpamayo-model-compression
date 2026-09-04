"""Stack a scene's real camera video and its top-down BEV into one side-by-side mp4.

Pairs `cam_<short>.mp4` from `fetch_scene_camera.py` with `bev_<short>.mp4` from
`render_scene_bev.py`, where `<short>` is the first 8 characters of the scene uuid. Both must
be real time for the halves to stay in step -- `render_scene_bev.py` defaults to that.

The two differ in rate (30 vs 10 fps) and shape (16:9 vs 1:1), so both are lifted to 30 fps and
a common 1080 height, giving 1920 + 1080 = 3000 px wide. `hstack` takes `shortest` as an option
(`hstack=inputs=2:shortest=1`); writing it as `hstack=inputs=2,shortest=1` makes ffmpeg look for
a filter named "shortest" and fail.

Usage:
  python make_scene_reel.py --cam-dir outputs/scene_difficulty/media/full \
      --bev-dir outputs/scene_difficulty/media/full \
      --out outputs/scene_difficulty/media/side
  python make_scene_reel.py ... --labels outputs/scene_difficulty/media/manifest_all.json
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

FILTER = ("[0:v]scale=-2:{h},fps={fps},setsar=1[a];"
          "[1:v]scale={h}:{h},fps={fps},setsar=1[b];"
          "[a][b]hstack=inputs=2:shortest=1[v]")


def stack(cam, bev, out, height=1080, fps=30, crf=22):
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(cam), "-i", str(bev),
         "-filter_complex", FILTER.format(h=height, fps=fps), "-map", "[v]",
         "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out)],
        capture_output=True, text=True, check=False)   # 실패는 씬 단위로 보고하고 계속
    if r.returncode != 0:
        return None, r.stderr.strip()[:200]
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam-dir", type=Path, required=True)
    ap.add_argument("--bev-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--labels", type=Path, default=None,
                    help="manifest json with short/group, used to prefix the output name")
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--crf", type=int, default=22)
    args = ap.parse_args()

    def abs_(p):
        return p if p.is_absolute() else REPO / p

    cam_dir, bev_dir, out_dir = abs_(args.cam_dir), abs_(args.bev_dir), abs_(args.out)
    group = {}
    if args.labels and abs_(args.labels).exists():
        group = {e["short"]: e.get("group", "") for e in
                 json.loads(abs_(args.labels).read_text())}

    shorts = sorted(m.group(1) for m in
                    (re.match(r"cam_(.+)\.mp4$", p.name) for p in cam_dir.glob("cam_*.mp4"))
                    if m)
    if not shorts:
        raise SystemExit(f"{cam_dir} 에 cam_*.mp4 가 없습니다")

    ok = 0
    for i, short in enumerate(shorts, 1):
        bev = bev_dir / f"bev_{short}.mp4"
        if not bev.exists():
            print(f"[{i}/{len(shorts)}] {short}: BEV 없음, 건너뜀")
            continue
        prefix = f"{group[short]}_" if group.get(short) else ""
        out = out_dir / f"{prefix}{short}.mp4"
        p, err = stack(cam_dir / f"cam_{short}.mp4", bev, out, args.height, args.fps, args.crf)
        if p is None:
            print(f"[{i}/{len(shorts)}] {short} 실패: {err}")
            continue
        ok += 1
        print(f"[{i}/{len(shorts)}] {p.name}  {p.stat().st_size / 1e6:5.1f} MB")
    total = sum(p.stat().st_size for p in out_dir.glob("*.mp4")) if out_dir.exists() else 0
    print(f"\n{ok}개 생성, {out_dir} 합계 {total / 1e6:.0f} MB "
          f"({args.height * 16 // 9 + args.height}x{args.height} @{args.fps}fps)")


if __name__ == "__main__":
    main()
