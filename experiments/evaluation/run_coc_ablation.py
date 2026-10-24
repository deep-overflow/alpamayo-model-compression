"""Does writing the CoC help the action? Same clips, same denoising seeds, with and without.

The expert reads the VLM's KV cache, so the conditions differ only in what that cache
holds when denoising starts:

  rollout     prompt(...<|cot_start|>) + the model's own CoC + <|cot_end|> + <|traj_future_start|>
              -- release inference, the reference
  empty       prompt(...<|cot_start|>) + <|cot_end|> + <|traj_future_start|>
              -- same prompt, zero reasoning tokens (what an empty-output collapse yields)
  skip        prompt without <|cot_start|> + <|traj_future_start|>
              -- no reasoning section at all
  trajprompt  instruction replaced by "output the future trajectory.", assistant turn
              starts at <|traj_future_start|> -- the model is not even asked
              to reason. This is the model's native no-CoC mode: under this instruction it
              emits <|traj_future_start|> itself with p=1.0 (8/8 clips checked)

Denoising noise is a function of the seed alone (`clip_seed(seed, clip_id) + k`), so every
condition of a clip draws the same K noises and the differences are paired.

`nll_<cond>` is the NLL of the forced tokens that replace the CoC (e.g. for `empty`, of
<|cot_end|><|traj_future_start|> right after <|cot_start|>) -- how surprised the model is
by the shortcut, not a quality measure.

Gates (plans/2026-10-07_coc-vs-nococ-ood.md), delta = minADE@6(empty) - minADE@6(rollout):
  G1 CI lower bound > 0 on both OOD-train and OOD-val; G2 mean >= 0.05 m on all 1,533;
  G3 `skip` agrees in sign with a CI excluding 0.

Usage:
  bash experiments/head_analysis/run_retry_host.sh 3 \
      experiments/evaluation/run_coc_ablation.py --shard 0 --n-shards 4 --gpu 4
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
from run_baseline import COT_END, MODEL_REV, TFS, load_manifest, seed_all, set_determinism
from run_eval import eval_config_samples
from sample_cache import clip_seed

COT_PROMPT = ("output the chain-of-thought reasoning of the driving process, "
              "then output the future trajectory.")
TRAJ_PROMPT = "output the future trajectory."
NOCOC = ["empty", "skip", "trajprompt"]


def trajprompt_ids(model, processor, data, device):
    """Fused prompt whose instruction asks for the trajectory only -> (1, T), ends in TFS."""
    messages = helper.create_message(frames=data["image_frames"].flatten(0, 1),
                                     camera_indices=data["camera_indices"])
    text = messages[1]["content"][-1]["text"]
    assert COT_PROMPT in text
    messages[1]["content"][-1]["text"] = text.replace(COT_PROMPT, TRAJ_PROMPT)
    messages[2]["content"][0]["text"] = "<|traj_future_start|>"
    ids = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, continue_final_message=True,
        return_dict=True, return_tensors="pt")["input_ids"].to(device)
    return model.fuse_traj_tokens(ids, {"ego_history_xyz": data["ego_history_xyz"].to(device),
                                        "ego_history_rot": data["ego_history_rot"].to(device)})


def record(rec, name, ade_k, fde_k, pred_k, gt_xy, nll, dt):
    rec.update({f"ade_{name}_k": [round(float(x), 6) for x in ade_k],
                f"fde_{name}_k": [round(float(x), 6) for x in fde_k],
                f"nll_{name}": nll, f"sec_{name}": round(dt, 3)})
    for h in (16, 32):  # 1.6 s / 3.2 s at 10 Hz
        a_h, f_h = el.ade_fde(pred_k[:, :h], gt_xy[:h])
        rec.update({f"ade_{name}_k_h{h}": [round(float(x), 6) for x in a_h],
                    f"fde_{name}_k_h{h}": [round(float(x), 6) for x in f_h]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="coc_ablation_ood")
    ap.add_argument("--sets-id", default="eval_sets")
    ap.add_argument("--manifest", default=None, help="OOD manifest stem (default: all 1,533)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"baseline_s{args.shard}of{args.n_shards}.json"
    man = load_manifest("ood", REPO / "outputs" / args.sets_id, 500, args.manifest)
    man = man[args.shard::args.n_shards]
    if args.limit:
        man = man[: args.limit]

    set_determinism(warn_only=True)
    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    gpu_name = torch.cuda.get_device_name(device)
    print(f"{args.exp_id} shard {args.shard}/{args.n_shards} | {len(man)} clips | {gpu_name}",
          flush=True)

    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    tok = model.tokenizer
    cot_start = tok.convert_tokens_to_ids("<|cot_start|>")
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")

    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    done = {r["clip_id"] for r in rows}
    (out_dir / "config.json").write_text(json.dumps({
        "model": "baseline", "set": "ood", "manifest": args.manifest, "n_clips": len(man),
        "k": args.k, "seed": args.seed,
        "seed_rule": "sha256(f'{seed}:{clip_id}')[:4], +k per sample; shared by conditions",
        "max_gen": args.max_gen, "shard": [args.shard, args.n_shards], "gpu": gpu_name,
        "model_revision": MODEL_REV, "conditions": ["rollout"] + NOCOC,
        "traj_prompt": TRAJ_PROMPT,
    }, indent=2))

    t_start = time.time()
    for i, m in enumerate(man):
        if m["clip_id"] in done:
            continue
        t0 = time.time()
        data = sc.load_cached(sc.path_for(m["cache"], m["clip_id"], m["t0_us"]))
        inputs = lib.build_inputs(model, processor, data, "cuda")
        prompt = inputs["input_ids"]  # (1, T_prompt), ends in <|cot_start|>
        prompt_len = prompt.shape[1]
        assert prompt[0, -1].item() == cot_start
        gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()  # (64, 2)
        base = clip_seed(args.seed, m["clip_id"])
        seeds = [base + k for k in range(args.k)]
        rec = {"clip_id": m["clip_id"], "bucket": el.bucket(gt_xy), "seed": base,
               "cluster": m["cluster"], "split": m["split"], "gt_coc": m["gt_coc"]}

        # rollout: identical to run_baseline.py's own-rollout condition
        seed_all(base)
        torch.cuda.synchronize()
        tg = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
        torch.cuda.synchronize()
        rec["sec_generate"] = round(time.time() - tg, 3)  # prefill + CoC decode
        coc_end = roll["eos_pos"] + 1
        seq_gen = roll["sequences"][:, :coc_end].clone()  # (1, T_prompt + L_coc)
        gen_ids = roll["sequences"][0, prompt_len:coc_end].tolist()
        gen_coc = tok.decode([t for t in gen_ids if t not in (COT_END, TFS)],
                             skip_special_tokens=True)
        del roll
        rec.update({"gen_coc": gen_coc, "gen_len": len(gen_ids),
                    **{f"coc_{k}": v for k, v in el.coc_degenerate(gen_coc).items()}})

        tail = torch.tensor([[COT_END, TFS]], device="cuda")
        seqs = {
            "rollout": (seq_gen, prompt_len, coc_end),
            "empty": (torch.cat([prompt, tail], dim=1), prompt_len, prompt_len + 2),
            "skip": (torch.cat([prompt[:, :-1], tail[:, 1:]], dim=1), prompt_len - 1, prompt_len),
        }
        tp = trajprompt_ids(model, processor, data, "cuda")  # (1, T_tp)
        assert tp[0, -1].item() == TFS
        seqs["trajprompt"] = (tp, tp.shape[1] - 1, tp.shape[1])

        for name, (seq, a, b) in seqs.items():
            torch.cuda.synchronize()
            tc = time.time()
            with torch.no_grad():
                ade_k, fde_k, nll, pred_k = eval_config_samples(
                    model, inputs, seq, a, b, gt_xy, seeds)  # pred_k (K, 64, 2)
            torch.cuda.synchronize()
            record(rec, name, ade_k, fde_k, pred_k, gt_xy, nll, time.time() - tc)
            rec[f"ctx_len_{name}"] = int(seq.shape[1])

        rows.append(rec)
        if len(rows) % 10 == 0 or i + 1 == len(man):
            rows_path.write_text(json.dumps(rows))
        mins = " ".join(f"{n}={min(rec[f'ade_{n}_k'][:6]):.3f}" for n in seqs)
        print(f"[{i + 1}/{len(man)}] {m['clip_id'][:8]} {m['split']:5s} {rec['bucket']:10s} "
              f"{mins} len={rec['gen_len']} ({time.time() - t0:.0f}s)", flush=True)

    rows_path.write_text(json.dumps(rows))
    line = f"shard {args.shard}/{args.n_shards}: {len(rows)} clips, minADE@6 mean " + ", ".join(
        f"{n} {np.mean([min(r[f'ade_{n}_k'][:6]) for r in rows]):.4f}"
        for n in ["rollout"] + NOCOC)
    (out_dir / f"summary_s{args.shard}of{args.n_shards}.txt").write_text(line + "\n")
    print(f"\n{line}\n{(time.time() - t_start) / 60:.1f} min -> {rows_path}", flush=True)


if __name__ == "__main__":
    main()
