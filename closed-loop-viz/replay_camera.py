"""Re-render a stored rollout's camera view by replaying its recorded render requests.

The driver does NOT run. Every `batch_render_request` in `rollout.asl` is already a
`BatchRGBRenderRequest` whose items carry an `RGBRenderRequest` complete with scene_id,
ftheta intrinsics, the rolling-shutter sensor pose (start and end), the frame window, and
every dynamic object's pose for that frame. So this forwards the recorded messages to a
standalone renderer and keeps the images it sends back.

What that buys: the trajectory, the traffic and the reasoning are fixed by the log, so the
footage is of the rollout that produced the published score -- not of a re-run that would
differ. It also means the GPU this runs on cannot change the driving, which is why it is
free to sit on a Blackwell card while the Ada cards stay with the closed-loop evaluations.

Why the images are not simply in the log: the launchers set ALPASIM_ASL_SKIP_IMAGES=1,
whose LogWriter drops driver camera frames (alpasim_runtime/event_loop.py:68). The requests
were recorded; the returned images were thrown away.

**All four cameras come back on every call.** The driver requests exactly the four Alpamayo
consumes -- front_wide_120, front_tele_30, cross_left_120, cross_right_120 -- in one batch
(the scene also carries rear_left_70 and rear_right_70, which the driver never asks for). The
render cost is therefore already paid whether one tile or four is kept, so `--camera all` is
the default: it shows what the model actually saw.

Frames are composited and written to disk one at a time rather than held in a list: four
1900x1080 tiles over 199 frames is ~4.9 GB of RAM if accumulated, and the encode does not
need them all at once.

Start the renderer first (GPU and port are yours to pick):

    GPU=1 PORT=16007 bash start_renderer.sh

The renderer only serves the scenes its --artifact-glob matched, so check the scene is in
its "Available scenes" line before pointing this at it.
"""
import argparse
import asyncio
import io
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import grpc
from alpasim_grpc.v0 import sensorsim_pb2_grpc
from PIL import Image, ImageDraw, ImageFont

RUNS = Path("/home/cvlab21/project/chan/alpasim-runs")

# The four the driver requests, laid out two per row: the forward pair on top (wide, then
# its zoomed twin), the side pair beneath in physical left-right order.
GRID = [("camera_front_wide_120fov", "front wide 120"),
        ("camera_front_tele_30fov", "front tele 30"),
        ("camera_cross_left_120fov", "cross left 120"),
        ("camera_cross_right_120fov", "cross right 120")]

BG = (13, 13, 15)
INK = (232, 230, 225)
DIM = (150, 148, 143)


def run_dir(config):
    p = Path(config)
    return p if p.is_absolute() else RUNS / f"m2601_merged_{config}"


async def read_requests(asl):
    """The recorded batch requests, in order, each tagged with the CoC in force at the time.

    The driver writes its reasoning on `driver_return`, which precedes the render request
    for the next frame, so the caption carried here is the text the model was acting on
    when that frame was drawn.
    """
    import pickle

    from alpasim_utils.logs import async_read_pb_log

    reqs = []
    last_coc = ""
    async for e in async_read_pb_log(str(asl)):
        if e.HasField("driver_return"):
            blob = e.driver_return.debug_info.unstructured_debug_info
            if blob:
                try:
                    t = pickle.loads(blob).get("reasoning_text")
                except (pickle.UnpicklingError, EOFError, AttributeError,
                        ImportError, IndexError, TypeError, ValueError):
                    t = None          # keep the previous caption rather than blanking it
                if t:
                    s = str(t).strip()
                    if s.startswith("['") and s.endswith("']"):
                        s = s[2:-2]
                    last_coc = s.strip()
        elif e.HasField("batch_render_request"):
            reqs.append((e.batch_render_request, last_coc))
    return reqs


def load_font(size):
    import matplotlib
    p = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    return ImageFont.truetype(str(p), size) if p.exists() else ImageFont.load_default()


def wrap_to_width(draw, text, font, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines[:2]


def compose(tiles, order, head, coc, tile_w, fonts):
    """One canvas: a grid of labelled tiles, a header strip and a caption strip."""
    font_h, font_l, font_c = fonts
    ncol = 2 if len(order) > 1 else 1
    nrow = (len(order) + ncol - 1) // ncol
    t0 = tiles[order[0][0]]
    tile_h = round(tile_w * t0.height / t0.width)
    pad, head_h, cap_h, lab_h = 6, 34, 46, 18
    width = ncol * tile_w + (ncol + 1) * pad
    height = head_h + nrow * (tile_h + lab_h + pad) + pad + cap_h

    canvas = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(canvas)
    d.text((width // 2, head_h // 2), head, font=font_h, fill=INK, anchor="mm")

    for i, (cam, label) in enumerate(order):
        row, col = divmod(i, ncol)
        x = pad + col * (tile_w + pad)
        y = head_h + pad + row * (tile_h + lab_h + pad)
        canvas.paste(tiles[cam].resize((tile_w, tile_h), Image.BILINEAR), (x, y))
        d.text((x + 4, y + tile_h + 3), label, font=font_l, fill=DIM)

    lines = wrap_to_width(d, coc or "(empty reasoning output)", font_c, width - 40)
    for j, line in enumerate(lines):
        d.text((width // 2, height - cap_h + 10 + j * 17), line,
               font=font_c, fill=INK, anchor="ma")
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--config", required=True,
                    help="run name (m2601_merged_<config>) or an absolute run directory")
    ap.add_argument("--camera", default="all",
                    help="'all' for the four the driver requests, or one camera name")
    ap.add_argument("--worst", action="store_true", default=True)
    ap.add_argument("--endpoint", default="localhost:16007")
    ap.add_argument("--tile-width", type=int, default=950)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    d = json.loads((run_dir(args.config) / "aggregate/results-summary.json").read_text())
    rs = sorted((r for r in d["rollouts"] if r["clipgt_id"] == args.scene),
                key=lambda r: float(r["score"]))
    if not rs:
        raise SystemExit(f"{args.config}: {args.scene} 없음")
    r = rs[0] if args.worst else rs[-1]
    m = r.get("metrics") or {}
    verdict = ("OFFROAD" if float(m.get("offroad") or 0) else
               ("COLLISION" if float(m.get("collision_at_fault") or 0) else "pass"))
    asl = run_dir(args.config) / "rollouts" / args.scene / r["rollout_id"] / "rollout.asl"
    if not asl.exists():
        raise SystemExit(f"{args.scene}: rollout.asl 이 없습니다 -- {asl}")
    print(f"{args.config}  rollout {r['rollout_id'][:8]}  "
          f"score {float(r['score']):.3f}  {verdict}")

    reqs = asyncio.run(read_requests(asl))
    if args.max_frames:
        reqs = reqs[:args.max_frames]
    want = list(GRID) if args.camera == "all" else \
        [(args.camera, args.camera.replace("camera_", "").replace("_", " "))]
    print(f"기록된 렌더 요청 {len(reqs)}개, 카메라 {[c for c, _ in want]}")

    # a 1 GiB message cap: four 1900x1080 JPEGs sit far under it, but the default 4 MiB
    # does not, and the failure surfaces as an opaque transport error
    chan = grpc.insecure_channel(
        args.endpoint,
        options=[("grpc.max_receive_message_length", 1 << 30),
                 ("grpc.max_send_message_length", 1 << 30)])
    stub = sensorsim_pb2_grpc.SensorsimServiceStub(chan)

    fonts = (load_font(17), load_font(13), load_font(15))
    head = f"{args.scene}    {args.config}    score {float(r['score']):.3f}  {verdict}"

    tmp = Path(tempfile.mkdtemp(prefix="camreplay_"))
    t0 = time.time()
    try:
        for i, (batch, coc) in enumerate(reqs):
            ret = stub.batch_render_rgb(batch)
            # BatchRGBRenderReturnItem: camera_name / result.image_bytes (JPEG) / success
            tiles = {}
            for item in ret.items:
                if not item.success:
                    raise SystemExit(f"프레임 {i}: {item.camera_name} success=False")
                tiles[item.camera_name] = Image.open(
                    io.BytesIO(item.result.image_bytes)).convert("RGB")
            missing = [c for c, _ in want if c not in tiles]
            if missing:
                raise SystemExit(f"프레임 {i}: 응답에 없는 카메라 {missing} "
                                 f"(받은 것 {sorted(tiles)})")
            canvas = compose(tiles, want, f"{head}    t = {i / args.fps:4.1f} s",
                             coc, args.tile_width, fonts)
            canvas.save(tmp / f"{i:05d}.jpg", quality=92)
            if i == 0:
                print(f"첫 프레임 OK: 합성 {canvas.size}, {time.time() - t0:.1f}s")
            elif (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(reqs)}  {(time.time() - t0) / (i + 1):.2f} s/frame")

        print(f"렌더+합성 {len(reqs)}프레임, {time.time() - t0:.0f}s")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        # even dimensions: libx264 with yuv420p rejects an odd width or height
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(args.fps),
             "-i", str(tmp / "%05d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-crf", "20", str(args.out)],
            check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"-> {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
