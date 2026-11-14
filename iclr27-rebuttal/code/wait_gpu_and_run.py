"""Wait for an empty card, then run a command on it; retry if the command fails (e.g. a co-tenant OOM).

A card counts as empty when nvidia-smi reports < --max-used-mib of memory in use (the lab rule: 2 MiB cards only).
The command must be resumable (run_importance_edge.py skips clips already on disk).

Usage:
  python iclr27-rebuttal/code/wait_gpu_and_run.py --cards 4,5,6,7 --attempts 20 -- <python> <script> [args...]
"""

import argparse
import os
import subprocess
import sys
import time


def free_cards(cards, max_used):
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    used = {int(i): int(m) for i, m in (line.split(",") for line in out.strip().splitlines())}
    return [c for c in cards if used.get(c, 10**9) < max_used]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", default="4,5,6,7")
    ap.add_argument("--max-used-mib", type=int, default=100)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--attempts", type=int, default=20)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    cards = [int(c) for c in args.cards.split(",")]
    for attempt in range(1, args.attempts + 1):
        while not (free := free_cards(cards, args.max_used_mib)):
            time.sleep(args.poll)
        card = free[0]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(card), "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
               "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        print(f"[wait_gpu] attempt {attempt}: card {card} empty at {time.strftime('%H:%M:%S')} UTC, launching",
              flush=True)
        rc = subprocess.run(cmd, env=env, check=False).returncode
        if rc == 0:
            print(f"[wait_gpu] done on attempt {attempt}", flush=True)
            return
        print(f"[wait_gpu] exit code {rc}; waiting for an empty card again", flush=True)
        time.sleep(args.poll)
    sys.exit(f"[wait_gpu] gave up after {args.attempts} attempts")


if __name__ == "__main__":
    main()
