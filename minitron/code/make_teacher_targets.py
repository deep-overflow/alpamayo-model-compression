"""Precompute the dense teacher's KD targets for every KD window (plan D10-5, 2026-10-25).

Online, the teacher runs inside every training step; here it runs once per window, with the
same functions train_kd uses (make_batch / vlm_forward / sup_logits / velocity), so the stored
tensors are what the online step computed:
  - logits at the supervised positions (CoC + cot_end + traj_future_start + 128 GT trajectory
    tokens), full vocabulary, bf16 -- the lm_head output under bf16 autocast, no top-K
    truncation (2407.14679: K <= 100 hurts);
  - the expert velocity v_T(x_t, t) for epoch slots 0 .. E-1, with (noise, t) from
    mt_data.kd_draw(clip, t0, slot) -- train_kd regenerates the same draw for the student.
One file per window, outputs/minitron/teacher_targets/<clip>__t0_<t0>.pt (~45 MB):
  {"logits": (N, V) bf16, "n_sup": N, "prompt_len": P, "v": (E, 64, 2) fp32, "t_idx": (E,)}
Resumable (existing files are skipped), shardable (--shard / --n-shards), written via .tmp +
rename so a killed shard never leaves a half file that a resume would skip.

Usage (one shard per card):
  ALPAMAYO_REPO=$PWD bash experiments/head_analysis/run_retry_host.sh 3 \\
      minitron/code/make_teacher_targets.py --shard 0 --n-shards 4 --gpu 4
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

import mt_data
import train_kd as tk

OUT = tk.REPO / "outputs" / "minitron" / "teacher_targets"


def target_path(clip_id, t0_us):
    return OUT / f"{clip_id}__t0_{int(t0_us)}.pt"


@torch.no_grad()
def run(teacher, processor, tok, rows, raws, epochs, max_coc):
    b = tk.make_batch(teacher, processor, tok, raws, max_coc, "cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = tk.vlm_forward(teacher, b, use_cache=True)
        lg = tk.sup_logits(teacher, out, b)  # (N_total, V) bf16
        vs, tis = [], []
        for slot in range(epochs):
            draws = [mt_data.kd_draw(r.clip_id, r.t0_us, slot) for r in rows]
            noise = torch.stack([d[0] for d in draws]).to("cuda")  # (B, 64, 2)
            t_idx = [d[1] for d in draws]
            t = torch.tensor([tk.T_GRID[i] for i in t_idx], device="cuda").view(-1, 1, 1)
            x_t = (1.0 - t) * noise + t * b["x1"]
            vs.append(tk.velocity(teacher, out.past_key_values, out.rope_deltas, b, x_t, t))
            tis.append(t_idx)
    v = torch.stack(vs, dim=1).cpu()  # (B, E, 64, 2)
    counts = torch.bincount(b["b_idx"], minlength=len(rows)).tolist()
    pieces = torch.split(lg.cpu(), counts)
    for i, r in enumerate(rows):
        prompt_len = int(b["h_idx"][b["b_idx"] == i][0]) + 1
        # clone: a split piece is a view, and torch.save would write the whole batch's storage
        rec = {"logits": pieces[i].clone(), "n_sup": counts[i], "prompt_len": prompt_len,
               "v": v[i].clone(), "t_idx": torch.tensor([tis[e][i] for e in range(epochs)])}
        dst = target_path(r.clip_id, r.t0_us)
        tmp = dst.with_suffix(".pt.tmp")
        torch.save(rec, tmp)
        tmp.rename(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="outputs/minitron/kd_sets/kd_r1.parquet")
    ap.add_argument("--epochs", type=int, default=3, help="velocity slots = training epochs")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-coc", type=int, default=256)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--threads", type=int, default=0, help="0: cpu_count / n_shards")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    # the image processor runs on CPU; four shards at torch's default 64 threads each
    # oversubscribed the 64 cores (load 131) and left the GPUs idle at 3.7 s/window
    args.threads = args.threads or max(1, os.cpu_count() // args.n_shards)
    torch.set_num_threads(args.threads)

    rows = pd.read_parquet(tk.REPO / args.manifest).iloc[args.shard :: args.n_shards]
    rows = rows[[not target_path(c, t).exists() for c, t in zip(rows.clip_id, rows.t0_us)]]
    if args.limit:
        rows = rows.iloc[: args.limit]
    from expert_per_clip import reserve_gpu
    from run_baseline import set_determinism

    set_determinism(warn_only=True)
    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    gpu = torch.cuda.get_device_name(device)
    teacher = tk.Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", revision=tk.MODEL_REV,
                                             dtype=torch.bfloat16).to("cuda")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    processor = tk.helper.get_processor(teacher.tokenizer)
    tok = teacher.tokenizer
    if args.shard == 0:
        (OUT / "config.json").write_text(json.dumps({
            **vars(args), "teacher": "nvidia/Alpamayo-1.5-10B", "revision": tk.MODEL_REV,
            "gpu": gpu, "t_grid": tk.T_GRID, "draw": "mt_data.kd_draw(clip, t0, slot)",
            "logits": "bf16, full vocabulary, supervised positions only"}, indent=2))
    print(f"shard {args.shard}/{args.n_shards} | {len(rows)} windows to do | {gpu}", flush=True)

    pool = ThreadPoolExecutor(4)
    t0 = time.time()
    for k, raws in enumerate(tk.batches(rows, args.batch, pool, tk.load_raw)):
        chunk = rows.iloc[k * args.batch : (k + 1) * args.batch]
        run(teacher, processor, tok, list(chunk.itertuples()), raws, args.epochs, args.max_coc)
        done = min((k + 1) * args.batch, len(rows))
        if (k + 1) % 25 == 0 or done == len(rows):
            rate = (time.time() - t0) / done
            print(f"[{done}/{len(rows)}] {rate:.2f} s/window, eta "
                  f"{rate * (len(rows) - done) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
