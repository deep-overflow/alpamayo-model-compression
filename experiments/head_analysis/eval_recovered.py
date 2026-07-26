"""Paired K=8 evaluation of a recovered (merged slim+LoRA) model vs the full baseline.

Two sequential passes (models never co-resident), same 80 combined_eval_clips, same
denoise seeds, same baseline CoC context (the full model's rollout is teacher-forced into
both), so deltas are paired and noise cancels:

  --pass ref : full model. Per clip: rollout -> save seq_tf; K=8 eval -> baseline ade/nll.
  --pass rec : merged slim+LoRA. Load saved seq_tf/seeds -> K=8 eval -> recovered ade/nll,
               paired delta vs baseline, bootstrap CI, scenario buckets, p95, regressions.

The pre-recovery "slim" numbers are already established (== masking, verify_slim), so the
3-point story is: baseline -> slim (known masking delta) -> recovered (this script).

Usage:
  bash experiments/head_analysis/run_retry.sh 20 experiments/head_analysis/eval_recovered.py \
      --pass ref  --exp-id eval_rec_int --gpu 0
  bash experiments/head_analysis/run_retry.sh 20 experiments/head_analysis/eval_recovered.py \
      --pass rec --merged outputs/lora_int/merged --exp-id eval_rec_int --gpu 0
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

import analysis_lib as lib  # noqa: E402
import eval_lib as el  # noqa: E402
import slim_lib as sl  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  gated-repo hub patch
from run_eval import eval_config  # noqa: E402

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

REPO = Path("/workspace/alpamayo-model-compression")


def eval_pass(model, processor, clips, out_dir, which, args):
    results = {"ade": [], "fde": [], "nll": []}
    buckets, ids = [], []
    for ci, clip_id in enumerate(clips):
        t0 = time.time()
        data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
        inputs = lib.build_inputs(model, processor, data, "cuda")
        prompt_len = inputs["input_ids"].shape[1]
        gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()
        seeds = [args.seed + ci * 100 + k for k in range(args.k)]
        art_path = out_dir / f"seq_{ci:03d}.pt"

        if which == "ref":
            torch.manual_seed(args.seed + ci)
            torch.cuda.manual_seed_all(args.seed + ci)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
            coc_end = roll["eos_pos"] + 1
            seq_tf = roll["sequences"][:, :coc_end].clone()
            del roll
            torch.save({"clip_id": clip_id, "seq_tf": seq_tf.cpu(),
                        "coc_start": prompt_len, "coc_end": coc_end}, art_path)
        else:
            art = torch.load(art_path, weights_only=False)
            assert art["clip_id"] == clip_id
            seq_tf = art["seq_tf"].to("cuda")
            prompt_len, coc_end = art["coc_start"], art["coc_end"]

        ade, fde, nll = eval_config(model, inputs, seq_tf, prompt_len, coc_end, gt_xy, seeds)
        results["ade"].append(ade)
        results["fde"].append(fde)
        results["nll"].append(nll)
        buckets.append(el.bucket(gt_xy))
        ids.append(clip_id)
        print(f"[{ci + 1}/{len(clips)}] {clip_id} {buckets[-1]:10s} "
              f"ade={ade:.3f} nll={nll:.3f} ({time.time() - t0:.0f}s)", flush=True)
    return results, buckets, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="which", required=True, choices=["ref", "rec"])
    ap.add_argument("--merged", type=str, default=None)
    ap.add_argument("--exp-id", type=str, required=True)
    ap.add_argument("--clip-file", type=str, default="outputs/combined_eval_clips.json")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = json.loads((REPO / args.clip_file).read_text())

    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    print(f"using {device} pass={args.which}", flush=True)

    if args.which == "ref":
        model = Alpamayo1_5.from_pretrained(
            "nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16).to("cuda")
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    else:
        assert args.merged, "--merged required for --pass rec"
        model = sl.load_slim(REPO / args.merged, device="cuda")
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")

    results, buckets, ids = eval_pass(model, processor, clips, out_dir, args.which, args)
    tag = args.which
    (out_dir / f"{tag}_metrics.json").write_text(json.dumps({
        "pass": tag, "merged": args.merged, "k": args.k, "clip_ids": ids,
        "buckets": buckets, "per_clip": results,
    }, indent=2))

    if args.which == "rec":
        ref = json.loads((out_dir / "ref_metrics.json").read_text())
        b_ade = np.array(ref["per_clip"]["ade"])
        r_ade = np.array(results["ade"])
        b_nll = np.array(ref["per_clip"]["nll"])
        r_nll = np.array(results["nll"])
        delta = r_ade - b_ade
        lo, hi = el.paired_bootstrap_ci(delta)
        buck = np.array(buckets)
        by_bucket = {}
        for bname in ("cruise", "turn", "decel_stop", "accel"):
            m = buck == bname
            if m.sum():
                by_bucket[bname] = {"n": int(m.sum()), "d_ade": float(delta[m].mean())}
        summary = {
            "n": len(clips), "baseline_minADE": float(b_ade.mean()),
            "recovered_minADE": float(r_ade.mean()), "d_ade": float(delta.mean()),
            "d_ade_ci": [float(lo), float(hi)], "d_nll": float((r_nll - b_nll).mean()),
            "p95": float(np.percentile(delta, 95)),
            "regressions_gt_0.5": int((delta > 0.5).sum()), "by_bucket": by_bucket,
        }
        (out_dir / "recovery_summary.json").write_text(json.dumps(summary, indent=2))
        print("\n=== RECOVERY ===", flush=True)
        print(f"baseline {summary['baseline_minADE']:.4f} -> recovered "
              f"{summary['recovered_minADE']:.4f}  Δ{summary['d_ade']:+.4f} "
              f"CI[{lo:+.3f},{hi:+.3f}] ΔNLL{summary['d_nll']:+.4f} "
              f"turn Δ{by_bucket.get('turn', {}).get('d_ade', float('nan')):+.4f}", flush=True)


if __name__ == "__main__":
    main()
