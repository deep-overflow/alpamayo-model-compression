"""Do the arms actually show different driving, or did the replay reuse one set of poses?

The whole premise of per-arm replay is that each arm's rollout.asl carries its own
RGBRenderRequests -- its own ego poses -- so the same scene must look different per arm. If
the footage were identical the replay would be drawing one arm's driving eight times, which
would be worse than useless: it would look like evidence.

Compares a late frame of the same scene across arms. The control is the same file against
itself, which must be exactly 0.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

V = Path("/home/cvlab21/project/chan/alpamayo-model-compression/closed-loop-viz/video")
SUITE = "origin150"
KIND = sys.argv[1] if len(sys.argv) > 1 else "camera"
ARMS = ["baseline", "dual", "wanda", "tyr", "coc", "traj", "tyrK", "llm-pruner"]


def find(arm, scene):
    hits = list((V / arm / SUITE / KIND).glob(f"*_{scene}.mp4"))
    return hits[0] if hits else None


def grab(v, n, out):
    out.unlink(missing_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(v),
                    "-vf", f"select='eq(n\\,{n})'", "-vframes", "1", str(out)], check=True)
    return np.asarray(Image.open(out).convert("L"), dtype=np.int16)


# a scene every arm rendered
scenes = None
for a in ARMS:
    got = {p.name.split("_", 1)[1][:-4] for p in (V / a / SUITE / KIND).glob("*.mp4")}
    scenes = got if scenes is None else (scenes & got)
scene = sorted(scenes)[0]
print(f"{KIND} / {SUITE} / {scene}  (모든 arm 공통 {len(scenes)}씬 중 첫 번째)")

with tempfile.TemporaryDirectory() as td:
    a_png, b_png = Path(td) / "a.png", Path(td) / "b.png"
    frames = {}
    for a in ARMS:
        v = find(a, scene)
        frames[a] = grab(v, 120, a_png).copy()
        score = v.name.split("_", 1)[0]
        print(f"  {a:<11} score {score}")

    ref = ARMS[0]
    print(f"\n프레임 120, {ref} 대비 평균 절대차 (0 = 동일 영상):")
    ctrl = np.abs(frames[ref] - grab(find(ref, scene), 120, b_png))
    print(f"  {'(대조: 자기 자신)':<24} {ctrl.mean():7.3f}")
    for a in ARMS[1:]:
        d = np.abs(frames[ref] - frames[a])
        print(f"  {a:<24} {d.mean():7.3f}   |d|>8 {100 * (d > 8).mean():5.1f}%")
