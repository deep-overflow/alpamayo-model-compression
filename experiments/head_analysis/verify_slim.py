"""Masked-vs-slim functional equivalence check.

Pass 1 (--pass masked): full model. Per clip an UNMASKED rollout fixes seq_tf, then for
each config the masks are applied and the teacher-forced eval runs; saves seq_tf and the
K raw predicted trajectories per config (a far tighter equivalence signal than metrics).

Pass 2 (--pass slim): a slim checkpoint re-runs the same teacher-forced eval on the saved
seq_tf with the same seeds; compares trajectories / NLL / minADE per clip. Rollout token
identity is NOT a criterion -- sampling at temperature 0.6 diverges on bf16-level logit
differences by design.

Passes run sequentially in separate processes so the two models are never co-resident.

Usage:
  bash experiments/head_analysis/run_retry.sh 20 experiments/head_analysis/verify_slim.py \
      --pass masked --gpu 4 --exp-id slim_verify
  bash experiments/head_analysis/run_retry.sh 20 experiments/head_analysis/verify_slim.py \
      --pass slim --slim-ckpt outputs/slim_integrated_mag --gpu 4 --exp-id slim_verify
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
import mask_lib as ml  # noqa: E402
import slim_lib as sl  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  also installs the gated-repo hub patch
from make_slim import build_masks  # noqa: E402

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

REPO = Path("/workspace/alpamayo-model-compression")
CONFIGS = ["integrated_mag", "cocsafe_full_r20"]
# soft gates: bf16 reduction-order noise expected, bit-exactness is not
GATE_XY, GATE_NLL, GATE_ADE = 0.05, 1e-2, 0.02


@torch.no_grad()
def eval_with_preds(model, inputs, seq_tf, coc_start, coc_end, gt_xy, seeds):
    """run_eval.eval_config, but also returns the K raw trajectories (K, 64, 2)."""
    attention_mask = torch.ones_like(seq_tf)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.vlm.model(
            input_ids=seq_tf, attention_mask=attention_mask,
            pixel_values=inputs["tokenized_data"]["pixel_values"],
            image_grid_thw=inputs["tokenized_data"]["image_grid_thw"], use_cache=True,
        )
        logits = model.vlm.lm_head(out.last_hidden_state[:, coc_start - 1 : coc_end - 1]).float()
        nll = torch.nn.functional.cross_entropy(logits[0], seq_tf[0, coc_start:coc_end]).item()
        cache = out.past_key_values
        prefill = cache.get_seq_length()
        offset = torch.tensor([prefill], device=seq_tf.device)
        prefix_mask = torch.ones(1, prefill, device=seq_tf.device, dtype=torch.long)
        preds = []
        for s in seeds:
            action = lib.denoise_with_cache(model, cache, out.rope_deltas, offset, prefix_mask,
                                            seed=s)  # cropped back to prefill each call
            pred_xyz, _ = model.action_space.action_to_traj(
                action.float(), inputs["ego_history_xyz"][:, -1].float(),
                inputs["ego_history_rot"][:, -1].float(),
            )
            preds.append(pred_xyz[0, :, :2].cpu().numpy())
    preds = np.stack(preds)  # (K, 64, 2)
    ade, fde = el.min_metrics(preds, gt_xy)
    del out, cache
    return ade, fde, nll, preds


def pass_masked(args, out_dir, clips):
    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    print(f"using {device}", flush=True)
    model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")

    imp = dict(np.load(REPO / "outputs" / args.importance / "importance.npz"))
    masks = {c: build_masks(c, imp, model) for c in CONFIGS}  # (vq, vm, eq, em, kvonly)
    tc = model.vlm.config.text_config
    ec = model.expert.config
    vmasks = ml.PruneMasks(model.vlm.model.language_model.layers, tc.num_attention_heads,
                           tc.head_dim, tc.intermediate_size, "cuda")
    emasks = ml.PruneMasks(model.expert.layers, ec.num_attention_heads, ec.head_dim,
                           ec.intermediate_size, "cuda")

    summary = {c: {"ade": [], "nll": []} for c in CONFIGS}
    for ci, clip_id in enumerate(clips):
        t0 = time.time()
        data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
        inputs = lib.build_inputs(model, processor, data, "cuda")
        prompt_len = inputs["input_ids"].shape[1]
        gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()
        seeds = [args.seed + ci * 100 + k for k in range(args.k)]

        vmasks.reset(); emasks.reset()
        torch.manual_seed(args.seed + ci)
        torch.cuda.manual_seed_all(args.seed + ci)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
        coc_start, coc_end = prompt_len, roll["eos_pos"] + 1
        seq_tf = roll["sequences"][:, :coc_end].clone()
        del roll

        art = {"clip_id": clip_id, "seq_tf": seq_tf.cpu(), "coc_start": coc_start,
               "coc_end": coc_end, "gt_xy": gt_xy, "seeds": seeds, "configs": {}}
        for cfg in CONFIGS:
            vq, vm, eq, em, _ = masks[cfg]
            vmasks.set(q=vq, mlp=vm)
            emasks.set(q=eq, mlp=em)
            ade, fde, nll, preds = eval_with_preds(model, inputs, seq_tf, coc_start, coc_end,
                                                   gt_xy, seeds)
            art["configs"][cfg] = {"ade": ade, "fde": fde, "nll": nll, "preds": preds}
            summary[cfg]["ade"].append(ade)
            summary[cfg]["nll"].append(nll)
        vmasks.reset(); emasks.reset()
        torch.save(art, out_dir / f"clip_{ci:02d}.pt")
        print(f"[{ci + 1}/{len(clips)}] {clip_id} ({time.time() - t0:.0f}s)", flush=True)

    (out_dir / "masked_metrics.json").write_text(json.dumps({
        "clips": clips, "k": args.k, "seed": args.seed,
        "summary": {c: {"mean_ade": float(np.mean(v["ade"])),
                        "mean_nll": float(np.mean(v["nll"]))} for c, v in summary.items()},
    }, indent=2))
    print("masked pass saved ->", out_dir, flush=True)


def pass_identity(args, out_dir, clips):
    """Attribution control: full model + identity gather + the SAME masks.

    Zero pruning, identical math to the masked reference -- only the attention pathway
    differs (stock enable_gqa vs materializing gather). Its deviation vs the saved
    artifacts is the pure bf16 pathway-noise floor; slim-vs-masked beyond this floor
    would indicate a real surgery bug.
    """
    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    print(f"using {device}", flush=True)
    model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    sl.bind_identity(model)
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")

    imp = dict(np.load(REPO / "outputs" / args.importance / "importance.npz"))
    masks = {c: build_masks(c, imp, model) for c in CONFIGS}
    tc = model.vlm.config.text_config
    ec = model.expert.config
    vmasks = ml.PruneMasks(model.vlm.model.language_model.layers, tc.num_attention_heads,
                           tc.head_dim, tc.intermediate_size, "cuda")
    emasks = ml.PruneMasks(model.expert.layers, ec.num_attention_heads, ec.head_dim,
                           ec.intermediate_size, "cuda")

    rows = {c: [] for c in CONFIGS}
    for ci, clip_id in enumerate(clips):
        t0 = time.time()
        art = torch.load(out_dir / f"clip_{ci:02d}.pt", weights_only=False)
        assert art["clip_id"] == clip_id
        data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
        inputs = lib.build_inputs(model, processor, data, "cuda")
        seq_tf = art["seq_tf"].to("cuda")
        for cfg in CONFIGS:
            vq, vm, eq, em, _ = masks[cfg]
            vmasks.set(q=vq, mlp=vm)
            emasks.set(q=eq, mlp=em)
            ade, fde, nll, preds = eval_with_preds(model, inputs, seq_tf, art["coc_start"],
                                                   art["coc_end"], art["gt_xy"], art["seeds"])
            ref = art["configs"][cfg]
            rows[cfg].append({
                "clip_id": clip_id,
                "max_dxy": float(np.abs(preds - ref["preds"]).max()),
                "d_nll": nll - ref["nll"], "d_ade": ade - ref["ade"],
            })
        vmasks.reset(); emasks.reset()
        print(f"[{ci + 1}/{len(clips)}] {clip_id} ({time.time() - t0:.0f}s)", flush=True)

    report = {}
    for cfg in CONFIGS:
        report[cfg] = {
            "worst": {"max_dxy": max(r["max_dxy"] for r in rows[cfg]),
                      "d_nll": max(abs(r["d_nll"]) for r in rows[cfg]),
                      "d_ade": max(abs(r["d_ade"]) for r in rows[cfg])},
            "per_clip": rows[cfg],
        }
        w = report[cfg]["worst"]
        print(f"NOISE FLOOR {cfg}: max|dxy| {w['max_dxy']:.4f} "
              f"|dNLL| {w['d_nll']:.5f} |dADE| {w['d_ade']:.4f}", flush=True)
    (out_dir / "verify_identity.json").write_text(json.dumps(report, indent=2))


def pass_slim(args, out_dir, clips):
    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    print(f"using {device}", flush=True)
    model = sl.load_slim(REPO / args.slim_ckpt, device="cuda")
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")
    cfg = json.loads((REPO / args.slim_ckpt / "slim_meta.json").read_text())["config"]
    print(f"slim config: {cfg}", flush=True)

    rows = []
    for ci, clip_id in enumerate(clips):
        t0 = time.time()
        art = torch.load(out_dir / f"clip_{ci:02d}.pt", weights_only=False)
        assert art["clip_id"] == clip_id
        data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
        inputs = lib.build_inputs(model, processor, data, "cuda")
        seq_tf = art["seq_tf"].to("cuda")
        ade, fde, nll, preds = eval_with_preds(model, inputs, seq_tf, art["coc_start"],
                                               art["coc_end"], art["gt_xy"], art["seeds"])
        ref = art["configs"][cfg]
        rows.append({
            "clip_id": clip_id,
            "max_dxy": float(np.abs(preds - ref["preds"]).max()),
            "d_nll": nll - ref["nll"],
            "d_ade": ade - ref["ade"],
            "ade_masked": ref["ade"], "ade_slim": ade,
        })
        print(f"[{ci + 1}/{len(clips)}] {clip_id} max|dxy|={rows[-1]['max_dxy']:.4f} "
              f"dNLL={rows[-1]['d_nll']:+.5f} dADE={rows[-1]['d_ade']:+.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    worst_xy = max(r["max_dxy"] for r in rows)
    worst_nll = max(abs(r["d_nll"]) for r in rows)
    worst_ade = max(abs(r["d_ade"]) for r in rows)
    ok = worst_xy < GATE_XY and worst_nll < GATE_NLL and worst_ade < GATE_ADE
    verdict = {
        "config": cfg, "slim_ckpt": args.slim_ckpt, "n_clips": len(clips), "k": args.k,
        "worst": {"max_dxy": worst_xy, "d_nll": worst_nll, "d_ade": worst_ade},
        "gates": {"max_dxy": GATE_XY, "d_nll": GATE_NLL, "d_ade": GATE_ADE},
        "pass": bool(ok), "per_clip": rows,
    }
    (out_dir / f"verify_{cfg}.json").write_text(json.dumps(verdict, indent=2))
    print(f"VERDICT {cfg}: {'PASS' if ok else 'FAIL'} "
          f"(max|dxy| {worst_xy:.4f}/{GATE_XY}, |dNLL| {worst_nll:.5f}/{GATE_NLL}, "
          f"|dADE| {worst_ade:.4f}/{GATE_ADE})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="which", required=True,
                    choices=["masked", "slim", "identity"])
    ap.add_argument("--slim-ckpt", type=str, default=None)
    ap.add_argument("--exp-id", type=str, default="slim_verify")
    ap.add_argument("--num-clips", type=int, default=10)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--importance", type=str, default="importance_v1")
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    split = json.loads((REPO / "outputs" / "split.json").read_text())
    clips = split["val"][: args.num_clips]

    if args.which == "masked":
        pass_masked(args, out_dir, clips)
    elif args.which == "identity":
        pass_identity(args, out_dir, clips)
    else:
        assert args.slim_ckpt, "--slim-ckpt required for --pass slim"
        pass_slim(args, out_dir, clips)


if __name__ == "__main__":
    main()
