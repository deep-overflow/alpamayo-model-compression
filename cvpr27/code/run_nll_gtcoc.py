"""Reference-CoC NLL on the OOD sets: how plausible an arm finds the curated correct reasoning.

cvpr27/plans/2026-11-02_oodtrain-coc-quality.md, CoC quality (1). The number is `nll_gtcoc` of
`run_baseline.py`'s teacher-forced OOD condition -- same span, same reduction -- without that
condition's eight trajectory denoisings, and without the rollout: `[prompt] + tokenize(gt_coc) +
<cot_end> + <traj_future_start>` (`gt_coc_seq`), one forward, mean per-token cross-entropy in nats
over the appended span (`run_nll_dense.span_nll`, which is `run_eval.eval_config_samples`'s `nll`).
No sampling, so seed-independent. It is a language column; the driving numbers stay rollout-only.

Gate (the plan's G1): dense and MoOP on OOD-val must land on the stored Ada `nll_gtcoc`
(`baseline_pred_oodval` 3.139, `dual_u40_v2_pred_oodval` 3.185) up to hardware noise.

Usage:
  python cvpr27/code/run_nll_gtcoc.py --manifest ood_val --model baseline --exp-id cvpr27/a100/nll_dense_oodval --gpu 0
"""

import os

# must precede any CUDA context creation for deterministic cuBLAS reductions
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
sys.path.insert(0, str(REPO / "experiments" / "evaluation"))

import analysis_lib as lib
import eval_lib as el
import sample_cache as sc
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from expert_per_clip import reserve_gpu
from run_baseline import MODEL_REV, gt_coc_seq, load_manifest, set_determinism
from run_nll_dense import span_nll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="OOD manifest stem: ood_val, ood_train or ood")
    ap.add_argument("--model", default="baseline", help="'baseline' or a slim ckpt dir")
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--sets-id", default="eval_sets")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="first N clips of the shard")
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--strict-deterministic", action="store_true",
                    help="error instead of warn on a non-deterministic kernel")
    args = ap.parse_args()

    tag = "baseline" if args.model == "baseline" else Path(args.model).name
    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"{tag}_s{args.shard}of{args.n_shards}.json"

    man = load_manifest("ood", REPO / "outputs" / args.sets_id, 0, args.manifest)
    man = man[args.shard::args.n_shards]          # strided, as run_baseline shards
    if args.limit:
        man = man[: args.limit]

    set_determinism(warn_only=not args.strict_deterministic)
    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    gpu_name = torch.cuda.get_device_name(device)
    print(f"{args.exp_id} {tag} | manifest={args.manifest} shard {args.shard}/{args.n_shards} "
          f"| {len(man)} clips | {gpu_name}", flush=True)

    if args.model == "baseline":
        model = Alpamayo1_5.from_pretrained(
            "nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, dtype=torch.bfloat16).to("cuda")
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    else:
        import slim_lib as sl
        model = sl.load_slim(REPO / args.model, device="cuda")

    processor = helper.get_processor(model.tokenizer)
    tok = model.tokenizer
    lib.set_vlm_attn_impl(model, "sdpa")   # as run_baseline, whose TF nll_gtcoc this reproduces

    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    done = {r["clip_id"] for r in rows}
    (out_dir / "config.json").write_text(json.dumps({
        "model": args.model, "manifest": args.manifest, "n_clips": len(man),
        "shard": [args.shard, args.n_shards], "sets_id": args.sets_id, "gpu": gpu_name,
        "model_revision": MODEL_REV,
        "span": "[prompt] + tokenize(gt_coc) + cot_end + traj_future_start, "
                "cross_entropy mean over the appended span (= run_baseline nll_gtcoc)",
        "deterministic": {"use_deterministic_algorithms": True,
                          "warn_only": not args.strict_deterministic,
                          "cudnn_deterministic": True, "tf32": False,
                          "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"]},
    }, indent=2))

    t_start = time.time()
    for i, m in enumerate(man):
        if m["clip_id"] in done:
            continue
        t0 = time.time()
        data = sc.load_cached(sc.path_for(m["cache"], m["clip_id"], m["t0_us"]))
        inputs = lib.build_inputs(model, processor, data, "cuda")
        prompt_len = inputs["input_ids"].shape[1]
        gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()  # (64, 2)

        seq = gt_coc_seq(tok, inputs["input_ids"], m["gt_coc"], "cuda")  # (1, L)
        per_tok = span_nll(model, inputs, seq, prompt_len, seq.shape[1])  # (n_tok,)
        rec = {"clip_id": m["clip_id"], "bucket": el.bucket(gt_xy), "cluster": m["cluster"],
               "split": m["split"], "nll_gtcoc": float(per_tok.mean()), "n_tok": len(per_tok),
               "nll_tok": [round(float(x), 5) for x in per_tok]}
        rows.append(rec)
        if len(rows) % 10 == 0 or i + 1 == len(man):
            rows_path.write_text(json.dumps(rows, indent=2))
        print(f"[{i + 1}/{len(man)}] {m['clip_id'][:8]} {rec['bucket']:10s} "
              f"nll={rec['nll_gtcoc']:.3f} n_tok={rec['n_tok']} ({time.time() - t0:.1f}s)", flush=True)

    rows_path.write_text(json.dumps(rows, indent=2))
    a = np.array([r["nll_gtcoc"] for r in rows])
    line = (f"{tag} {args.manifest} shard {args.shard}/{args.n_shards}: {len(rows)} clips, "
            f"nll_gtcoc mean {a.mean():.4f} median {np.median(a):.4f}")
    (out_dir / f"summary_s{args.shard}of{args.n_shards}.txt").write_text(line + "\n")
    print(f"\n{line}\n{(time.time() - t_start) / 60:.1f} min -> {rows_path}", flush=True)


if __name__ == "__main__":
    main()
