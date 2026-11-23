"""Side-by-side comparison video of several arms on one scene: per arm a column of BEV (cl_bev.py),
optionally with the front camera (cl_dump_camera.py) on top. Arm label burned in top-left; from each
arm's event frame on, a red frame and an event pill. Returns the data entry (per-frame CoC as runs).

Used by cl_gallery.py (BEV only) and for the MoOP project page (camera + BEV):
  python closed-loop-viz/cl_compose.py --entries entries.json --video-dir DIR --data out.json [--camera]
entries.json: [{"id": ..., "label": ..., "arms": [{"label": "MoOP", "dir": ".../moop"}, ...], ...}, ...]
(any extra keys are copied into the data entry).
"""
import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
CW, CH, BW, BH, GAP = 640, 364, 640, 480, 6
RED = (229, 72, 77)
EVENT_TXT = {"collision": "AT-FAULT COLLISION", "offroad": "OFF-ROAD"}


def pill(d, xy, text, font, fill, right=False):
    x, y = xy
    w = d.textlength(text, font=font)
    pad = font.size * 0.45
    if right:
        x -= w + 2 * pad
    d.rounded_rectangle([x, y, x + w + 2 * pad, y + font.size * 1.55], radius=font.size * 0.35, fill=fill)
    d.text((x + pad, y + font.size * 0.22), text, font=font, fill=(255, 255, 255))


def load_arms(entry):
    return [dict(a, bev=json.loads((Path(a["dir"]) / "bev_meta.json").read_text())) for a in entry["arms"]]


def entry_data(entry, video, arms=None):
    """The data entry for a composed video: scene fields plus per-arm verdict, event and per-frame CoC."""
    arms = arms or load_arms(entry)
    video = Path(video)
    out = {k: v for k, v in entry.items() if k != "arms"}
    out.update(n_frames=max(a["bev"]["n_frames"] for a in arms), t_offset=arms[0]["bev"]["t_offset"], video=video.name,
               poster=video.with_suffix(".jpg").name,
               arms=[{"name": a["label"], "verdict": a["bev"]["verdict"], "score": a["bev"]["score"],
                      "dist_m": round(a["bev"]["dist_traveled_m"]), "event_frame": a["bev"]["event_frame"],
                      "event_t": a["bev"]["event_t"], "n_frames": a["bev"]["n_frames"], "coc": a["bev"]["coc"]}
                     for a in arms])
    return out


def compose(entry, video, camera=False, crf=27, tmp_root=None):
    arms = load_arms(entry)
    if camera:
        for a in arms:
            cam = json.loads((Path(a["dir"]) / "meta.json").read_text())
            assert cam["rollout_id"] == a["bev"]["rollout_id"], f"{a['dir']}: camera and BEV are different rollouts"
            assert cam["n_frames"] == a["bev"]["n_frames"], f"{a['dir']}: camera/BEV frame counts differ"
    n = max(a["bev"]["n_frames"] for a in arms)
    top = CH if camera else 0
    font = ImageFont.truetype(BOLD, 23)
    tmp = Path(tempfile.mkdtemp(prefix="clcompose_", dir=tmp_root))
    try:
        for i in range(n):
            canvas = Image.new("RGB", (len(arms) * (BW + GAP) - GAP, top + BH), (12, 12, 14))
            d = ImageDraw.Draw(canvas, "RGBA")
            for j, a in enumerate(arms):
                b = a["bev"]
                f = min(i, b["n_frames"] - 1)
                x0 = j * (BW + GAP)
                if camera:
                    canvas.paste(Image.open(Path(a["dir"]) / "frames" / f"{f:05d}.jpg").resize((CW, CH), Image.LANCZOS),
                                 (x0, 0))
                canvas.paste(Image.open(Path(a["dir"]) / "bev" / f"{f:05d}.jpg"), (x0, top))
                if camera:
                    d.line([(x0, top), (x0 + BW, top)], fill=(12, 12, 14), width=2)
                pill(d, (x0 + 12, 12), a["label"], font, (0, 0, 0, 150))
                if b["event_frame"] is not None and f >= b["event_frame"]:
                    d.rectangle([x0 + 2, 2, x0 + BW - 3, top + BH - 3], outline=RED, width=5)
                    pill(d, (x0 + BW - 12, 12), EVENT_TXT[b["verdict"]], font, RED + (235,), right=True)
            canvas.save(tmp / f"{i:05d}.jpg", quality=93)
        video = Path(video)
        video.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "10", "-i", str(tmp / "%05d.jpg"),
                        "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", "-threads", "4", str(video)], check=True)
        evs = [a["bev"]["event_frame"] for a in arms if a["bev"]["event_frame"] is not None]
        pf = entry.get("poster", min(max(evs) + 6, n - 1) if evs else n // 2)
        Image.open(tmp / f"{pf:05d}.jpg").save(video.with_suffix(".jpg"), quality=82)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return entry_data(entry, video, arms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entries", type=Path, required=True)
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--camera", action="store_true", help="stack the front camera above the BEV")
    args = ap.parse_args()
    data = {"fps": 10, "scenes": []}
    for e in json.loads(args.entries.read_text()):
        data["scenes"].append(compose(e, args.video_dir / f"{e['id']}.mp4", camera=args.camera))
        print(f"{e['id']}: {data['scenes'][-1]['n_frames']} frames", flush=True)
    args.data.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    print(f"wrote {args.data}")


if __name__ == "__main__":
    main()
