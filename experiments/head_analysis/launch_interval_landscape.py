"""Queue worker for the interval-deletion landscape: one GPU, shards claimed one at a time.

run_token_ablation.py checkpoints but does not resume, so a long run is cut into many small
shards and a failure costs one of them. Start one worker per card; each claims the next shard
that is neither finished nor held by a live worker, waits until its card is empty, and runs it.
A shard is finished only when its metrics.json holds every clip of the shard.

Sets what run_retry_host.sh sets (token from $HOME, blobs from the shared cache, PCI_BUS_ID
device order) plus OMP_NUM_THREADS=8.

Usage (one per card, from the checkout the outputs/ and .venv symlinks live in):
  nohup .venv/bin/python experiments/head_analysis/launch_interval_landscape.py --gpu 4 \
      --masks ivl_masks_main --prefix ivl_main_v1 --num-clips 150 --n-shards 30 &
"""

import argparse
import configparser
import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def gpu_used_mib(gpu):
    out = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used",
                          "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    return int(out.stdout.strip())


def alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--num-clips", type=int, default=150)
    ap.add_argument("--n-shards", type=int, default=30)
    ap.add_argument("--max-attempts", type=int, default=2)
    args = ap.parse_args()

    home = Path.home()
    cp = configparser.ConfigParser()
    cp.read(home / ".cache/huggingface/stored_tokens")
    env = dict(os.environ)
    env.update({
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HOME": str(home / ".cache/huggingface"), "HF_HUB_CACHE": "/mnt/nvme1n1/ad_vla/cache/hub",
        "HF_HUB_ENABLE_HF_TRANSFER": "0", "HF_TOKEN": cp["full_right"]["hf_token"],
        "OMP_NUM_THREADS": "8",
    })
    masks = (REPO / "outputs" / args.masks / "arm_masks.txt").read_text().split()
    py = str(REPO / ".venv" / "bin" / "python")
    locks = REPO / "outputs" / f"{args.prefix}_locks"
    locks.mkdir(exist_ok=True)
    attempts = {}

    def log(msg):
        print(f"{time.strftime('%m-%d %H:%M:%S')} gpu{args.gpu} {msg}", flush=True)

    def done(i):
        f = REPO / "outputs" / f"{args.prefix}_s{i}" / "metrics.json"
        return f.exists() and json.loads(f.read_text())["n_clips"] == len(range(i, args.num_clips, args.n_shards))

    def claim():
        for i in range(args.n_shards):
            if done(i) or attempts.get(i, 0) >= args.max_attempts:
                continue
            lock = locks / f"s{i}"
            if lock.exists():
                pid = int(lock.read_text() or 0)
                if alive(pid):
                    continue
                lock.unlink()  # its worker died
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return i
        return None

    while (i := claim()) is not None:
        while (used := gpu_used_mib(args.gpu)) > 100:
            log(f"card holds {used} MiB, waiting")
            time.sleep(60)
        attempts[i] = attempts.get(i, 0) + 1
        log(f"shard {i} start (attempt {attempts[i]})")
        with open(REPO / "outputs" / f"{args.prefix}_s{i}.launch.log", "w") as f:
            rc = subprocess.run(
                [py, "experiments/head_analysis/run_token_ablation.py", "--gpu", str(args.gpu),
                 "--exp-id", f"{args.prefix}_s{i}", "--num-clips", str(args.num_clips),
                 "--shard", str(i), "--n-shards", str(args.n_shards), "--arm-masks", *masks],
                check=False, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        (locks / f"s{i}").unlink(missing_ok=True)
        log(f"shard {i} exit {rc} done={done(i)}")
        if rc != 0:
            time.sleep(60)
    log("no shard left to claim")


if __name__ == "__main__":
    main()
