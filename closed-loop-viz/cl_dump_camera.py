"""Dump one arm's front-wide camera frames for a scene by replaying the recorded render requests.

Same rollout as cl_bev.py / replay_camera.py (the worse of the scene's two). Needs the NuRec renderer
running (start_renderer.sh) with the scene in its sceneset; the driver does not run.
Writes <out>/frames/%05d.jpg (native 1900x1080) and <out>/meta.json (rollout_id, n_frames, per-frame CoC).

  python closed-loop-viz/cl_dump_camera.py --scene clipgt-... --config /path/to/merged_run --out DIR/moop
"""
import argparse
import asyncio
import io
import json
import sys
from pathlib import Path

import grpc
from alpasim_grpc.v0 import sensorsim_pb2_grpc
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cl_bev import verdict_of, worst_rollout
from replay_camera import read_requests

CAM = "camera_front_wide_120fov"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--config", required=True, type=Path, help="merged run directory")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--endpoint", default="localhost:16007")
    args = ap.parse_args()

    r = worst_rollout(args.config, args.scene)
    rdir = args.config / "rollouts" / args.scene / r["rollout_id"]
    reqs = asyncio.run(read_requests(rdir / "rollout.asl"))
    chan = grpc.insecure_channel(args.endpoint, options=[("grpc.max_receive_message_length", 1 << 30),
                                                          ("grpc.max_send_message_length", 1 << 30)])
    stub = sensorsim_pb2_grpc.SensorsimServiceStub(chan)
    (args.out / "frames").mkdir(parents=True, exist_ok=True)
    cocs = []
    for i, (batch, coc) in enumerate(reqs):
        item = next(it for it in stub.batch_render_rgb(batch).items if it.camera_name == CAM)
        if not item.success:
            raise SystemExit(f"frame {i}: render failed")
        Image.open(io.BytesIO(item.result.image_bytes)).convert("RGB").save(args.out / "frames" / f"{i:05d}.jpg",
                                                                             quality=92)
        cocs.append(coc)
    meta = {"scene": args.scene, "config": str(args.config), "rollout_id": r["rollout_id"],
            "verdict": verdict_of(r), "n_frames": len(reqs), "coc": cocs}
    (args.out / "meta.json").write_text(json.dumps(meta))
    print(f"{args.out.name}: {len(reqs)} camera frames, {meta['verdict']}", flush=True)


if __name__ == "__main__":
    main()
