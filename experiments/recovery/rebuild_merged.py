"""Materialize a merged (slim + LoRA) checkpoint from an adapter-only save.

Adapters are the stored artifact (~0.3 GB); the 17 GB merged checkpoint is rebuilt on
demand for evaluation and for the alpasim driver (whose vendored load_slim requires
slim_state.pt). The merged directory is a normal slim_lib checkpoint: run_baseline.py
takes it as --model, launch_alpasim_*.sh as a driver dir.

Usage:
  bash experiments/head_analysis/run_retry_host.sh 10 experiments/recovery/rebuild_merged.py \
      --adapter outputs/recover_dual_u55/adapter_best.pt --out outputs/recover_dual_u55/merged \
      --gpu 5
"""

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))

import recover_lib as rl
from expert_per_clip import reserve_gpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default=None,
                    help="slim checkpoint dir; default: the one recorded in the adapter")
    ap.add_argument("--no-state", action="store_true")
    ap.add_argument("--reserve-gb", type=float, default=24.0)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    payload = torch.load(REPO / args.adapter, map_location="cpu", weights_only=True)
    extra = payload["extra"]
    ckpt = args.ckpt or extra["ckpt"]
    print(f"adapter from step {extra.get('step')} (val minADE {extra.get('val_minADE')}), "
          f"base {ckpt}", flush=True)

    reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    peft_model, base, meta = rl.load_slim_lora(REPO / ckpt, r=extra["r"],
                                               alpha=extra["alpha"], device="cuda")
    rl.load_adapter_(peft_model, REPO / args.adapter)
    meta["recovered_from"] = {"adapter": args.adapter, **{k: extra.get(k) for k in
                                                          ("step", "val_minADE")}}
    out_dir = REPO / args.out
    rl.merge_save(peft_model, base, meta, out_dir, write_state=not args.no_state)
    (out_dir / "summary.txt").write_text(
        f"merged slim+LoRA from {args.adapter} (step {extra.get('step')})\n"
        f"base {ckpt}  config {meta['config']}\n")
    print("merged ->", out_dir, flush=True)


if __name__ == "__main__":
    main()
