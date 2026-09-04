"""Pull one alpasim scene's real front-camera video out of the PhysicalAI-AV chunk zips.

The usdz holds no frames (it is a neural reconstruction), and the model-input loader
`load_physical_aiavdataset` returns only the 4 sampled timesteps per camera it feeds the VLM --
neither gives a watchable clip. The footage lives in the PhysicalAI-AV camera chunks as one
`<clip_id>.<camera>.mp4` per clip, ~3-40 MB inside a ~5 GB chunk zip.

The trick that makes this cheap: `HfFileSystem.open()` is seekable, so `zipfile` can read the
central directory and then just that one member. Pulling a 20 s 1080p clip took 5-11 s per scene
against downloading the whole chunk. Chunks already mirrored under `data/physicalai_av/camera`
are used directly; the local mirror is sparse (350 of ~3140), so most scenes stream.

Only ~22% of alpasim scene ids exist in the PhysicalAI-AV clip index (198 of `public_2601`'s
913) -- a scene set drawn on other criteria will mostly have no footage, and this reports which.

Usage:
  python fetch_scene_camera.py --scenes outputs/scene_difficulty/hard100_suite.csv \
      --out outputs/scene_difficulty/media/full
  python fetch_scene_camera.py --scenes ... --out ... --list-only
"""

import argparse
import configparser
import csv
import os
import time
import zipfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PAV = Path("/mnt/nvme1n1/ad_vla/data/physicalai_av")
HF_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
DEFAULT_CAM = "camera_front_wide_120fov"


def hf_token():
    """The `full_right` token, which is the only one with gated-dataset access.

    `stored_tokens` is INI, and there are two of them: the shared cache on /mnt/nvme1n1 that
    `HF_HOME` usually points at holds other members' tokens only, while `full_right` lives in
    $HOME. Check both rather than trusting HF_HOME.
    """
    seen = []
    for base in (Path.home() / ".cache/huggingface",
                 Path(os.environ.get("HF_HOME", "")) if os.environ.get("HF_HOME") else None):
        if base is None:
            continue
        f = base / "stored_tokens"
        if not f.exists():
            continue
        cp = configparser.ConfigParser()
        cp.read(f)
        seen.append(f"{f} -> {list(cp.sections())}")
        if "full_right" in cp:
            return cp["full_right"]["hf_token"]
    raise SystemExit("full_right 토큰을 찾지 못했습니다 (gated 접근 불가):\n  "
                     + "\n  ".join(seen or ["stored_tokens 파일 없음"]))


def pull(clip_id, chunk, out_path, cam=DEFAULT_CAM, token=None):
    """Extract one clip's camera mp4. Returns (path, seconds, source) or (None, sec, reason)."""
    out = Path(out_path)
    if out.exists():
        return out, 0.0, "이미 있음"
    member = f"{clip_id}.{cam}.mp4"
    local_zip = PAV / "camera" / cam / f"{cam}.chunk_{chunk:04d}.zip"
    t0 = time.time()
    if local_zip.exists():
        src, how = zipfile.ZipFile(local_zip), "로컬 청크"
    else:
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem(token=token or hf_token())
        remote = f"datasets/{HF_REPO}/camera/{cam}/{cam}.chunk_{chunk:04d}.zip"
        src, how = zipfile.ZipFile(fs.open(remote, "rb")), "HF 스트리밍"
    if member not in src.namelist():
        return None, time.time() - t0, f"멤버 없음 ({member})"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".mp4.tmp")
    with src.open(member) as fh, tmp.open("wb") as w:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            w.write(b)
    tmp.rename(out)
    return out, time.time() - t0, how


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=Path, required=True,
                    help="suite CSV with a scene_id column")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--camera", default=DEFAULT_CAM)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--list-only", action="store_true",
                    help="report which scenes have footage and stop")
    args = ap.parse_args()

    idx = pd.read_parquet(PAV / "clip_index.parquet")
    rows = [r["scene_id"] for r in csv.DictReader(args.scenes.open())]
    have = [(s, s.replace("clipgt-", "")) for s in rows]
    have = [(s, c, int(idx.loc[c, "chunk"])) for s, c in have if c in idx.index]
    print(f"{len(rows)}개 중 카메라 데이터 보유 {len(have)}개 "
          f"({100 * len(have) / max(len(rows), 1):.0f}%)")
    if args.list_only:
        for s, c, ch in have:
            print(f"  {c[:8]}  chunk {ch}")
        return
    if args.limit:
        have = have[:args.limit]

    out = args.out if args.out.is_absolute() else REPO / args.out
    token = hf_token()
    for i, (_sid, cid, chunk) in enumerate(have, 1):
        p, sec, how = pull(cid, chunk, out / f"cam_{cid[:8]}.mp4", args.camera, token)
        if p is None:
            print(f"[{i}/{len(have)}] {cid[:8]} 실패: {how}")
        else:
            print(f"[{i}/{len(have)}] {p.name}  {p.stat().st_size / 1e6:5.1f} MB  "
                  f"{sec:4.1f}s  [{how}]")


if __name__ == "__main__":
    main()
