"""What is a CORRECT CoC worth? Teacher-force different CoC texts, same clips and seeds.

Companion of run_coc_ablation.py (which stored `rollout` and the no-CoC conditions). Here
the cache is given a CoC the model did not write, crossing content with style:

  gt        this clip's curated CoC                     correct content, human style
  gt_swap   another clip's curated CoC                  unrelated content, human style
  gen_swap  the CoC the model wrote for another clip    unrelated content, model style
  gen_retok this clip's own generated CoC, re-tokenized (check only, --check-retok): must
            reproduce the stored `rollout`, or re-tokenizing is itself a treatment

The partner clip is the fixed derangement score_coc_similarity.py used (`perm`), so its
judge score for the "shuffled" pair says how wrong `gen_swap`'s text is for this clip.
Every text is tokenized and closed with <|cot_end|><|traj_future_start|>, exactly as
run_baseline.gt_coc_seq does. Seeds are `clip_seed(seed, clip_id) + k`, as everywhere.

Pre-registered questions: plans/2026-10-07_coc-vs-nococ-ood.md, "CoC 교체 실험".

Usage:
  .venv/bin/python experiments/evaluation/launch_coc_ablation.py \
      --script run_coc_swap.py --exp-id coc_swap_ood --gpus 4 5 6 7
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
sys.path.insert(0, str(Path(__file__).parent))

import analysis_lib as lib
import eval_lib as el
import sample_cache as sc
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from expert_per_clip import reserve_gpu
from run_baseline import MODEL_REV, gt_coc_seq, load_manifest, set_determinism
from run_coc_ablation import record
from run_eval import eval_config_samples
from sample_cache import clip_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="coc_swap_ood")
    ap.add_argument("--src", default="coc_ablation_ood", help="run holding gen_coc and perm")
    ap.add_argument("--sets-id", default="eval_sets")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--check-retok", action="store_true")
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"baseline_s{args.shard}of{args.n_shards}.json"

    src = REPO / "outputs" / args.src
    gen = {}
    for f in sorted(src.glob("baseline_s*of*.json")):
        for r in json.loads(f.read_text()):
            gen[r["clip_id"]] = r
    sim = json.loads((src / "coc_similarity.json").read_text())
    ids, perm = sim["clip_id"], sim["perm"]
    partner = {c: ids[perm[i]] for i, c in enumerate(ids)}  # clip -> the clip whose CoC it gets
    assert all(partner[c] != c for c in ids)

    man = {m["clip_id"]: m for m in load_manifest("ood", REPO / "outputs" / args.sets_id, 500)}
    assert set(man) == set(ids)
    todo = [man[c] for c in ids][args.shard::args.n_shards]
    if args.limit:
        todo = todo[: args.limit]

    set_determinism(warn_only=True)
    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    gpu_name = torch.cuda.get_device_name(device)
    print(f"{args.exp_id} shard {args.shard}/{args.n_shards} | {len(todo)} clips | {gpu_name}",
          flush=True)
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    tok = model.tokenizer
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")

    conds = ["gt", "gt_swap", "gen_swap"] + (["gen_retok"] if args.check_retok else [])
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    done = {r["clip_id"] for r in rows}
    (out_dir / "config.json").write_text(json.dumps({
        "model": "baseline", "set": "ood", "n_clips": len(todo), "k": args.k, "seed": args.seed,
        "seed_rule": "sha256(f'{seed}:{clip_id}')[:4], +k per sample; shared with " + args.src,
        "shard": [args.shard, args.n_shards], "gpu": gpu_name, "model_revision": MODEL_REV,
        "conditions": conds, "src": args.src, "partner": "coc_similarity.json perm",
    }, indent=2))

    t_start = time.time()
    for i, m in enumerate(todo):
        c = m["clip_id"]
        if c in done:
            continue
        t0 = time.time()
        data = sc.load_cached(sc.path_for(m["cache"], c, m["t0_us"]))
        inputs = lib.build_inputs(model, processor, data, "cuda")
        prompt_len = inputs["input_ids"].shape[1]
        gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()  # (64, 2)
        base = clip_seed(args.seed, c)
        assert base == gen[c]["seed"]
        seeds = [base + k for k in range(args.k)]
        p = partner[c]
        texts = {"gt": m["gt_coc"], "gt_swap": man[p]["gt_coc"], "gen_swap": gen[p]["gen_coc"],
                 "gen_retok": gen[c]["gen_coc"]}
        rec = {"clip_id": c, "split": m["split"], "bucket": el.bucket(gt_xy), "seed": base,
               "partner": p, **{f"text_{n}": texts[n] for n in conds}}
        for n in conds:
            seq = gt_coc_seq(tok, inputs["input_ids"], texts[n], "cuda")  # (1, T_prompt + L)
            tc = time.time()
            with torch.no_grad():
                ade_k, fde_k, nll, pred_k = eval_config_samples(
                    model, inputs, seq, prompt_len, seq.shape[1], gt_xy, seeds)  # pred (K, 64, 2)
            record(rec, n, ade_k, fde_k, pred_k, gt_xy, nll, time.time() - tc)
        rows.append(rec)
        if len(rows) % 10 == 0 or i + 1 == len(todo):
            rows_path.write_text(json.dumps(rows))
        ro = min(gen[c]["ade_rollout_k"][:6])
        print(f"[{i + 1}/{len(todo)}] {c[:8]} rollout={ro:.3f} " + " ".join(
            f"{n}={min(rec[f'ade_{n}_k'][:6]):.3f}" for n in conds)
            + f" ({time.time() - t0:.0f}s)", flush=True)

    rows_path.write_text(json.dumps(rows))
    line = f"shard {args.shard}/{args.n_shards}: {len(rows)} clips, minADE@6 mean " + ", ".join(
        f"{n} {np.mean([min(r[f'ade_{n}_k'][:6]) for r in rows]):.4f}" for n in conds)
    (out_dir / f"summary_s{args.shard}of{args.n_shards}.txt").write_text(line + "\n")
    print(f"\n{line}\n{(time.time() - t_start) / 60:.1f} min -> {rows_path}", flush=True)


if __name__ == "__main__":
    main()
