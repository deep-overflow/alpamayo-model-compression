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

Start the renderer first (GPU and port are yours to pick):

    GPU=1 PORT=16007 bash start_renderer.sh

The renderer only serves the scenes its --artifact-glob matched, so check the scene is in
its "Available scenes" line before pointing this at it.
"""
import argparse
import asyncio
import io
import time
from pathlib import Path

import grpc
import numpy as np
from alpasim_grpc.v0 import sensorsim_pb2_grpc
from PIL import Image

RUNS = Path("/home/cvlab21/project/chan/alpasim-runs")


def run_dir(config):
    p = Path(config)
    return p if p.is_absolute() else RUNS / f"m2601_merged_{config}"


async def read_requests(asl, camera):
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
                    if t:
                        s = str(t).strip()
                        if s.startswith("['") and s.endswith("']"):
                            s = s[2:-2]
                        last_coc = s.strip()
                except Exception:
                    pass
        elif e.HasField("batch_render_request"):
            items = [it for it in e.batch_render_request.items
                     if camera is None or it.camera_name == camera]
            if items:
                reqs.append((e.batch_render_request, items[0].camera_name, last_coc))
    return reqs


def decode(image_bytes):
    return np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--config", required=True,
                    help="run name (m2601_merged_<config>) or an absolute run directory")
    ap.add_argument("--camera", default="camera_front_wide_120fov")
    ap.add_argument("--worst", action="store_true", default=True)
    ap.add_argument("--endpoint", default="localhost:16007")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import json
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
    print(f"{args.config}  rollout {r['rollout_id'][:8]}  score {float(r['score']):.3f}  {verdict}")

    reqs = asyncio.run(read_requests(asl, args.camera))
    if args.max_frames:
        reqs = reqs[:args.max_frames]
    print(f"기록된 렌더 요청 {len(reqs)}개  (카메라 {args.camera})")

    # a 1 GiB message cap: a batch of four 1920x1080 JPEGs is far under it, but the
    # default 4 MiB is not always enough and the failure looks like a transport error
    chan = grpc.insecure_channel(
        args.endpoint,
        options=[("grpc.max_receive_message_length", 1 << 30),
                 ("grpc.max_send_message_length", 1 << 30)])
    stub = sensorsim_pb2_grpc.SensorsimServiceStub(chan)

    frames, t0 = [], time.time()
    for i, (batch, cam, coc) in enumerate(reqs):
        ret = stub.batch_render_rgb(batch)
        # BatchRGBRenderReturnItem: camera_name / result.image_bytes (JPEG) / success
        img = None
        for item in ret.items:
            if item.camera_name != cam:
                continue
            if not item.success:
                raise SystemExit(f"프레임 {i}: 렌더러가 success=False 를 돌려줬습니다")
            img = decode(item.result.image_bytes)
            break
        if img is None:
            raise SystemExit(f"프레임 {i}: {cam} 가 응답에 없습니다 "
                             f"({[it.camera_name for it in ret.items]})")
        frames.append((img, coc))
        if i == 0:
            print(f"첫 프레임 OK: {img.shape}, {time.time() - t0:.1f}s")
        elif (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(reqs)}  {(time.time() - t0) / (i + 1):.2f} s/frame")

    print(f"렌더 {len(frames)}프레임, {time.time() - t0:.0f}s")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    h, w = frames[0][0].shape[:2]
    fig = plt.figure(figsize=(w / 160, h / 160 + 0.9), facecolor="#0d0d0f")
    ax = fig.add_axes([0, 0.11, 1, 0.83])
    ax.axis("off")
    im = ax.imshow(frames[0][0])
    title = fig.text(0.5, 0.965, "", ha="center", va="top", color="#e8e6e1", fontsize=9)
    cap = fig.text(0.5, 0.055, "", ha="center", va="center", color="#e8e6e1", fontsize=8.5,
                   wrap=True)
    head = (f"{args.scene[:28]}   {args.config}   score {float(r['score']):.3f}  {verdict}"
            f"   ·   {args.camera}")

    def update(k):
        img, coc = frames[k]
        im.set_data(img)
        title.set_text(f"{head}    t = {k / args.fps:4.1f} s")
        cap.set_text(coc if coc else "(empty reasoning output)")
        return [im, title, cap]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(str(args.out), writer=animation.FFMpegWriter(
        fps=args.fps, bitrate=6000, codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"]))
    print(f"-> {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
