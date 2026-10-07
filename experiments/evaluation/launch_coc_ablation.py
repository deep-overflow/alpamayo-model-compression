"""Run run_coc_ablation.py as one strided shard per GPU, with run_retry_host.sh's environment.

Usage:
  .venv/bin/python experiments/evaluation/launch_coc_ablation.py --gpus 4 5 6 7
"""

import argparse
import configparser
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

ap = argparse.ArgumentParser()
ap.add_argument("--gpus", type=int, nargs="+", default=[4, 5, 6, 7])
ap.add_argument("--exp-id", default="coc_ablation_ood")
ap.add_argument("--manifest", default=None)
args = ap.parse_args()

cp = configparser.ConfigParser()
cp.read(Path.home() / ".cache/huggingface/stored_tokens")
env = dict(os.environ)
env.update({
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "HF_HOME": str(Path.home() / ".cache/huggingface"),
    "HF_HUB_CACHE": "/mnt/nvme1n1/ad_vla/cache/hub",
    "HF_HUB_ENABLE_HF_TRANSFER": "0",
    "HF_TOKEN": cp["full_right"]["hf_token"],
    "OMP_NUM_THREADS": "8",
})
logs = REPO / "outputs" / args.exp_id / "logs"
logs.mkdir(parents=True, exist_ok=True)
procs = []
for s, g in enumerate(args.gpus):
    cmd = [sys.executable, str(REPO / "experiments/evaluation/run_coc_ablation.py"),
           "--exp-id", args.exp_id, "--shard", str(s), "--n-shards", str(len(args.gpus)),
           "--gpu", str(g)]
    if args.manifest:
        cmd += ["--manifest", args.manifest]
    procs.append(subprocess.Popen(cmd, cwd=REPO, env=env, stdout=open(logs / f"s{s}.log", "w"),
                                  stderr=subprocess.STDOUT))
codes = [p.wait() for p in procs]
print("exit codes", codes)
sys.exit(max(codes))
