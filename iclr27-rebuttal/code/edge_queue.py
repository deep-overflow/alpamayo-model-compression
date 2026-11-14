"""Job queue for the Edge u40 runs on Ada cards 4-7 (plans/2026-11-14_a2-edge-moop-u40.md). Edge venv or any python.

Jobs: 4 arms x {LingoQA, Test500, OOD-Val250}. A card takes new jobs only while no foreign process is on it (the lab rule:
never share a card with someone else's job; this queue's own jobs may share, up to --slots per card). Open-loop jobs wait until
their Edge cache passed its GT gate (verify.json "passed": true). A failed job (e.g. a co-tenant OOM) goes back to the queue,
up to --retries times; every runner is resumable, so a retry continues where it stopped.

--arms / --kinds / --shards / --log pick the job set (2026-11-14, plans/2026-11-14_a2-edge-l0keep-u40.md: the three _l0keep arms on
Test500 only). Masks are outputs/iclr27-rebuttal/edge_moop/masks_<arm>.npz, outputs edge_<kind>_<arm>.

Usage: python iclr27-rebuttal/code/edge_queue.py --cards 4,5,6,7 --slots 2
       python iclr27-rebuttal/code/edge_queue.py --arms moop_l0keep,fm_only_l0keep,ce_only_l0keep --kinds test500 \
           --shards test500=8 --log logs_l0keep
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = "/mnt/nvme1n1/ad_vla/venvs/alpamayo2-edge/bin/python"
CACHE = Path("/mnt/nvme1n1/ad_vla/data/physicalai_av/pre_processed")
LOG_ROOT = REPO / "outputs" / "iclr27-rebuttal" / "edge_moop"
LOG = LOG_ROOT / "logs_ada"
ARMS = ("dense", "moop", "fm_only", "ce_only")


SHARDS = {"test500": 2, "oodval": 1}  # open-loop shards per arm (rows files edge_s{i}of{n}.json, merged by load_rows)


def jobs(arms=ARMS, kinds=("lingo", "test500", "oodval"), shards=None):
    shards = {**SHARDS, **(shards or {})}
    out = []
    for kind in kinds:
        for arm in arms:
            mask = [] if arm == "dense" else ["--masks", f"outputs/iclr27-rebuttal/edge_moop/masks_{arm}.npz"]
            if kind == "lingo":
                cmd = [PY, "iclr27-rebuttal/code/run_edge_lingoqa.py", "--exp-id", f"iclr27-rebuttal/edge_lingo_{arm}"]
                out.append({"name": f"lingo_{arm}", "cmd": cmd + mask, "ready": None, "tries": 0})
                continue
            n = shards[kind]
            for s in range(n):
                cmd = [PY, "iclr27-rebuttal/code/run_edge_openloop.py", "--set", kind, "--exp-id",
                       f"iclr27-rebuttal/edge_{kind}_{arm}", "--shard", str(s), "--n-shards", str(n)]
                out.append({"name": f"{kind}_{arm}_s{s}of{n}", "cmd": cmd + mask,
                            "ready": CACHE / f"{kind}_edge" / "verify.json", "tries": 0})
    return out


def gpu_state():
    """{card index: set of pids on it}"""
    idx = {}
    for line in subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True,
                               text=True, check=True).stdout.strip().splitlines():
        i, u = (x.strip() for x in line.split(","))
        idx[u] = int(i)
    pids = {i: set() for i in idx.values()}
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], capture_output=True,
                         text=True, check=True).stdout.strip()
    for line in out.splitlines():
        if line.strip():
            u, p = (x.strip() for x in line.split(","))
            pids[idx[u]].add(int(p))
    return pids


def is_ready(job):
    if job["ready"] is None:
        return True
    return job["ready"].exists() and json.loads(job["ready"].read_text()).get("passed") is True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", default="4,5,6,7")
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--threads", type=int, default=4, help="OMP/MKL/OpenBLAS threads per job")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--kinds", default="lingo,test500,oodval")
    ap.add_argument("--shards", nargs="*", default=[], help="kind=n, e.g. test500=8")
    ap.add_argument("--log", default="logs_ada", help="log dir under outputs/iclr27-rebuttal/edge_moop")
    args = ap.parse_args()
    global LOG
    LOG = LOG_ROOT / args.log
    cards = [int(c) for c in args.cards.split(",")]
    LOG.mkdir(parents=True, exist_ok=True)
    shards = {k: int(v) for k, v in (s.split("=") for s in args.shards)}
    queue, running, done, failed = jobs(args.arms.split(","), args.kinds.split(","), shards), {}, [], []
    os.chdir(REPO)

    def log(msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(LOG / "queue.log", "a") as f:
            f.write(line + "\n")

    log(f"queue start: {len(queue)} jobs, cards {cards}, {args.slots} slots per card, {args.threads} CPU threads per job")
    while queue or running:
        for name, (proc, card, job, fh) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            fh.close()
            del running[name]
            if rc == 0:
                done.append(name)
                log(f"done {name} on card {card}")
            elif job["tries"] < args.retries:
                queue.insert(0, job)
                log(f"FAILED {name} on card {card} (exit {rc}, try {job['tries']}); requeued")
            else:
                failed.append(name)
                log(f"FAILED {name} on card {card} (exit {rc}); giving up")
        pids = gpu_state()
        mine = {p.pid for p, *_ in running.values()}
        for card in cards:
            foreign = pids.get(card, set()) - mine
            n_mine = sum(1 for _, c, *_ in running.values() if c == card)
            while not foreign and n_mine < args.slots:
                job = next((j for j in queue if is_ready(j)), None)
                if job is None:
                    break
                queue.remove(job)
                job["tries"] += 1
                t = str(args.threads)  # torch's default CPU pool is all 64 cores per process: oversubscribes the box
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(card), "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "OMP_NUM_THREADS": t,
                       "MKL_NUM_THREADS": t, "OPENBLAS_NUM_THREADS": t}
                fh = open(LOG / f"{job['name']}.log", "a")  # noqa: SIM115 -- outlives this block; closed when the job ends
                proc = subprocess.Popen(job["cmd"], env=env, stdout=fh, stderr=subprocess.STDOUT)
                running[job["name"]] = (proc, card, job, fh)
                n_mine += 1
                log(f"start {job['name']} on card {card} (try {job['tries']}, pid {proc.pid})")
        time.sleep(args.poll)
    log(f"queue end: done {len(done)}, failed {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
