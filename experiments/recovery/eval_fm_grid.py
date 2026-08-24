"""Frozen-grid flow-matching loss for any checkpoint — the expert's own objective.

`train_recover` logs this during training; this runs it standalone so already-finished
checkpoints can be compared on the same (clip, t, noise) grid. That is the direct test of
the capacity hypothesis behind u70's residual gap: the teacher-forced minADE split says
77% of the u70-vs-u55 gap survives a perfect CoC, but minADE is the 10-step integrated
result. This measures the quantity the expert is actually trained on.

Every checkpoint sees the identical grid (t = 0.05..0.95, noise seeded per clip), the
identical clips, and the curated gt_coc as context, so differences are the model alone.

Usage:
  python experiments/recovery/eval_fm_grid.py --ckpt outputs/slim_recover_dual_u70 \
      --exp-id fmgrid_u70 --clips 100 --gpu 6
  python experiments/recovery/eval_fm_grid.py --ckpt baseline --exp-id fmgrid_base --gpu 7
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
sys.path.insert(0, str(REPO / "experiments" / "evaluation"))

import analysis_lib as lib
import pandas as pd
import recover_lib as rl
import slim_lib as sl
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from expert_per_clip import reserve_gpu
from train_recover import FM_T_GRID, ce_mean, fm_stats, load_samples, val_fm_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="slim checkpoint dir, or 'baseline'")
    ap.add_argument("--adapter", default=None, help="optional adapter to merge first")
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--clips", type=int, default=100)
    ap.add_argument("--ce-train-clips", type=int, default=100,
                    help="held-IN OOD train clips for the teacher-forced CE (0 disables)")
    ap.add_argument("--ce-official-clips", type=int, default=0,
                    help="held-IN OFFICIAL train clips, whose CoC is the teacher rollout "
                         "rather than curated gt_coc. Splitting held-in CE by source says "
                         "whether the two supervision styles are fitted differently.")
    ap.add_argument("--max-coc", type=int, default=256)
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    print(f"using {device}", flush=True)

    if args.ckpt == "baseline":
        model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16,
                                            revision=sl.MODEL_REV).to("cuda")
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    elif args.adapter:
        peft_model, model, _ = rl.load_slim_lora(REPO / args.ckpt, device="cuda")
        rl.load_adapter_(peft_model, REPO / args.adapter)
        peft_model.merge_and_unload()
    else:
        model = sl.load_slim(REPO / args.ckpt, device="cuda")
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")
    processor = helper.get_processor(model.tokenizer)

    ood = pd.read_parquet(REPO / "outputs" / "eval_sets" / "ood_val.parquet")
    all_ood = [("ood", r.clip_id, int(r.t0_us), "ood") for r in ood.itertuples()]
    samples = all_ood[: args.clips]
    ce_samples = all_ood[: max(args.clips, args.ce_train_clips) or 100]
    t0 = time.time()
    rows = val_fm_rows(model, processor, model.tokenizer, samples, args.max_coc)
    stats = fm_stats(rows)
    if not rows:
        # --clips 0: skip the FM grid and read the held-out CE from a CE-only pass
        stats = {"ce": ce_mean(val_fm_rows(model, processor, model.tokenizer,
                                           ce_samples, args.max_coc, t_grid=()))}
    if args.ce_train_clips:
        # same code, same distribution, same gt_coc column -- only "was it trained on"
        # differs, so held_out - held_in is the VLM channel's generalisation gap
        train, _ = load_samples(REPO)
        held_in = [(c, i, t, src) for c, i, t, _, src in train
                   if src == "ood"][: args.ce_train_clips]
        stats["ce_heldin"] = ce_mean(val_fm_rows(model, processor, model.tokenizer,
                                                 held_in, args.max_coc, t_grid=()))
        stats["ce_gap"] = stats["ce"] - stats["ce_heldin"]
    if args.ce_official_clips:
        train, _ = load_samples(REPO)
        off = [s for s in train if s[4] == "official"][: args.ce_official_clips]
        stats["ce_heldin_official"] = ce_mean(val_fm_rows(
            model, processor, model.tokenizer, off, args.max_coc, t_grid=()))

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({
        "ckpt": args.ckpt, "adapter": args.adapter, "clips": args.clips,
        "t_grid": list(FM_T_GRID), "context": "teacher-forced gt_coc",
        "gpu": torch.cuda.get_device_name(device)}, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps({"stats": stats, "rows": rows}, indent=2))
    line = (f"{args.ckpt}: FM(frozen grid) mean {stats['mean']:.5f}  "
            f"low {stats['low']:.5f}  mid {stats['mid']:.5f}  high {stats['high']:.5f}  "
            f"({stats['n_clips']} clips x {len(FM_T_GRID)} t, {(time.time() - t0) / 60:.1f}m)"
            ) if rows else f"{args.ckpt}: FM grid skipped (--clips 0)"
    if "ce_gap" in stats:
        line += (f"\n{args.ckpt}: CoC CE held-out {stats['ce']:.4f}  "
                 f"held-in {stats['ce_heldin']:.4f}  gap {stats['ce_gap']:+.4f}")
    if "ce_heldin_official" in stats:
        line += (f"\n{args.ckpt}: CoC CE held-in OFFICIAL (teacher rollout) "
                 f"{stats['ce_heldin_official']:.4f}")
    (out_dir / "summary.txt").write_text(line + "\n")
    print(line, flush=True)
    print("saved ->", out_dir, flush=True)


if __name__ == "__main__":
    main()
