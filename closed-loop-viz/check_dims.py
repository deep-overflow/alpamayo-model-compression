"""Every rendered video, its dimensions, and whether any is odd.

libx264 + yuv420p needs even width and height. An odd one is not merely a warning: the
declared-size-vs-buffer mismatch that produced the one-row-per-frame slide showed up first
as an odd height, and the fix was to plan even pixel counts and assert them. This is the
cheap half of the verification -- no frame decoding, so it runs over the whole tree in
seconds -- and `check_drift.py` is the other half.
"""
import json
import subprocess
from collections import Counter
from pathlib import Path

V = Path("/home/cvlab21/project/chan/alpamayo-model-compression/closed-loop-viz/video")

vids = sorted(V.rglob("*.mp4"))
sizes = Counter()
odd = []
bad = []
for v in vids:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", str(v)],
        capture_output=True, text=True)
    try:
        st = json.loads(out.stdout)["streams"][0]
        w, h = int(st["width"]), int(st["height"])
    except Exception:
        bad.append(v)
        continue
    sizes[(w, h)] += 1
    if w % 2 or h % 2:
        odd.append((v, w, h))

print(f"{len(vids)}개 검사")
for (w, h), n in sorted(sizes.items(), key=lambda kv: -kv[1]):
    print(f"  {w}x{h}  {n}개")
print(f"홀수 변 {len(odd)}개, 읽기 실패 {len(bad)}개")
for v, w, h in odd[:10]:
    print(f"  ODD {w}x{h}  {v.relative_to(V)}")
for v in bad[:10]:
    print(f"  UNREADABLE {v.relative_to(V)}")
