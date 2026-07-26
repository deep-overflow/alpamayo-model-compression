"""Strengthened evaluation: multi-sample minADE/minFDE on the P2 headline configs.

For each clip the reference CoC is generated once (shared VLM prefill). Every config
then runs one teacher-forced forward and K expert denoisings with different seeds;
minADE/minFDE take the min over the K samples. baseline and pruned configs share the
same K seeds, so sampling noise cancels in the paired difference.

Usage:
  bash run_retry.sh 20 experiments/head_analysis/run_eval.py --gpu 0 --split val --k 8
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
from expert_per_clip import reserve_gpu  # noqa: E402  also installs the gated-repo hub patch

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

REPO = Path("/workspace/alpamayo-model-compression")
LATE_START = 22


def headline_configs(imp, mag, n_layers, n_heads, intermediate, seed):
    """The ~15 configs whose 'free' claims from P2 we want to re-verify."""
    late = list(range(LATE_START, n_layers))
    all_l = list(range(n_layers))
    rng = np.random.RandomState(seed)
    tq, tm = imp["traj_vlm_q"], imp["traj_vlm_mlp"]
    cq, cm = imp["coc_vlm_q"], imp["coc_vlm_mlp"]

    def width(qs, ms, ratio, scope, r=None):
        return (ml.select_mask(qs, ratio, scope, rng=r),
                ml.select_mask(ms, ratio, scope, rng=r))

    cfgs = [("baseline", {"kind": "baseline"}, None, None)]
    q = np.ones((n_layers, n_heads)); q[-1] = 0
    m = np.ones((n_layers, intermediate)); m[-1] = 0
    cfgs.append(("drop_last_layer", {"kind": "structural"}, q, m))

    for ratio in (0.30, 0.50):
        for crit, (qs, ms) in [("traj", (tq, tm)), ("coc", (cq, cm))]:
            qq, mm = width(qs, ms, ratio, late)
            cfgs.append((f"{crit}_late_r{int(ratio * 100)}",
                         {"kind": "width", "criterion": crit, "scope": "late", "ratio": ratio},
                         qq, mm))
    qq, mm = width(mag[0], mag[1], 0.50, late)
    cfgs.append(("magnitude_late_r50", {"kind": "width", "criterion": "magnitude",
                                        "scope": "late", "ratio": 0.5}, qq, mm))
    q = ml.select_mask(np.zeros((n_layers, n_heads)), 0.5, late, rng=rng)
    m = ml.select_mask(np.zeros((n_layers, intermediate)), 0.5, late, rng=rng)
    cfgs.append(("random_late_r50", {"kind": "width", "criterion": "random",
                                     "scope": "late", "ratio": 0.5}, q, m))

    for ratio in (0.20, 0.30):
        for crit, (qs, ms) in [("traj", (tq, tm)), ("coc", (cq, cm))]:
            qq, mm = width(qs, ms, ratio, all_l)
            cfgs.append((f"{crit}_all_r{int(ratio * 100)}",
                         {"kind": "width", "criterion": crit, "scope": "all", "ratio": ratio},
                         qq, mm))

    for crit, key in [("traj", "traj_kv"), ("coc", "coc_kv"), ("random", None)]:
        for n_drop in (1, 2):
            if key is None:
                sc = np.zeros((n_layers, imp["traj_kv_k"].shape[1])); r = rng
            else:
                sc = imp[f"{key}_k"] + imp[f"{key}_v"]; r = None
            q = ml.kv_group_mask(sc, n_drop, n_heads, all_l, rng=r)
            cfgs.append((f"kv_{crit}_drop{n_drop}",
                         {"kind": "kv_group", "criterion": crit, "n_drop": n_drop}, q, None))
    return cfgs


def combined_configs(imp, n_layers, n_heads, intermediate, seed):
    """Stack the confirmed-free units to test whether their deltas are independent.

    Graded depth: 0-21 @30% width, 22-34 @50% width, layer 35 fully removed, plus
    1 KV group/layer -- all by trajectory Taylor. References measured on the same
    clips let us compare combined delta vs sum-of-individual deltas.
    """
    early = list(range(0, LATE_START))
    late = list(range(LATE_START, n_layers - 1))  # 22..34; layer 35 handled separately
    all_l = list(range(n_layers))
    tq, tm = imp["traj_vlm_q"], imp["traj_vlm_mlp"]
    kv = imp["traj_kv_k"] + imp["traj_kv_v"]

    def graded_width():
        q = ml.select_mask(tq, 0.30, early) * ml.select_mask(tq, 0.50, late)
        m = ml.select_mask(tm, 0.30, early) * ml.select_mask(tm, 0.50, late)
        q[n_layers - 1] = 0.0  # layer 35 attn fully removed
        m[n_layers - 1] = 0.0  # layer 35 mlp fully removed
        return q, m

    cfgs = [("baseline", {"kind": "baseline"}, None, None)]
    # individual references (same clips, for the independence check)
    qd = np.ones((n_layers, n_heads)); qd[-1] = 0
    md = np.ones((n_layers, intermediate)); md[-1] = 0
    cfgs.append(("ref_drop_last", {"kind": "ref"}, qd, md))
    cfgs.append(("ref_all_r30", {"kind": "ref"},
                 ml.select_mask(tq, 0.30, all_l), ml.select_mask(tm, 0.30, all_l)))
    cfgs.append(("ref_late_r50", {"kind": "ref"},
                 ml.select_mask(tq, 0.50, late), ml.select_mask(tm, 0.50, late)))
    cfgs.append(("ref_kv_drop1", {"kind": "ref"},
                 ml.kv_group_mask(kv, 1, n_heads, all_l), None))

    qw, mw = graded_width()
    cfgs.append(("combined_width", {"kind": "combined", "note": "graded width + layer35"}, qw, mw))
    qk = qw * ml.kv_group_mask(kv, 1, n_heads, all_l)
    cfgs.append(("combined_width_kv1", {"kind": "combined", "note": "+ KV drop1"}, qk, mw))
    qk2 = qw * ml.kv_group_mask(kv, 2, n_heads, all_l)
    cfgs.append(("combined_width_kv2", {"kind": "combined", "note": "+ KV drop2 (known-bad)"},
                 qk2, mw))
    return cfgs


@torch.no_grad()
def eval_config(model, inputs, seq_tf, coc_start, coc_end, gt_xy, seeds):
    """Teacher-forced forward + K denoisings. Returns (minADE, minFDE, CoC NLL)."""
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
    ade, fde = el.min_metrics(np.stack(preds), gt_xy)
    del out, cache
    return ade, fde, nll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-clips", type=int, default=50)
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--exp-id", type=str, default=None)
    ap.add_argument("--importance", type=str, default="importance_v1")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--clip-offset", type=int, default=0)
    ap.add_argument("--clip-file", type=str, default=None,
                    help="JSON list of clip_ids to evaluate; overrides split selection")
    ap.add_argument("--mode", type=str, default="headline", choices=["headline", "combined"])
    args = ap.parse_args()
    exp_id = args.exp_id or f"eval_{args.split}_k{args.k}"

    out_dir = REPO / "outputs" / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    split = json.loads((REPO / "outputs" / "split.json").read_text())
    if args.clip_file:
        clips = json.loads(Path(args.clip_file).read_text())
        clips = clips[args.clip_offset : args.clip_offset + args.num_clips]
    else:
        clips = split[args.split][args.clip_offset : args.clip_offset + args.num_clips]
    imp = dict(np.load(REPO / "outputs" / args.importance / "importance.npz"))

    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    print(f"using {device}", flush=True)

    model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")

    tc = model.vlm.config.text_config
    layers = model.vlm.model.language_model.layers
    masks = ml.PruneMasks(layers, tc.num_attention_heads, tc.head_dim, tc.intermediate_size, "cuda")
    mag = ml.magnitude_scores(layers, tc.num_attention_heads, tc.head_dim, tc.intermediate_size)
    if args.mode == "combined":
        cfgs = combined_configs(imp, tc.num_hidden_layers, tc.num_attention_heads,
                                tc.intermediate_size, args.seed)
    else:
        cfgs = headline_configs(imp, mag, tc.num_hidden_layers, tc.num_attention_heads,
                                tc.intermediate_size, args.seed)
    print(f"{len(cfgs)} configs x {len(clips)} clips x K={args.k}", flush=True)

    (out_dir / "config.json").write_text(json.dumps({
        "model": "nvidia/Alpamayo-1.5-10B", "eval_split": args.split, "num_clips": len(clips),
        "clip_ids": clips, "clip_offset": args.clip_offset, "k_samples": args.k,
        "importance_from": args.importance,
        "metrics": ["minADE", "minFDE"], "buckets": ["cruise", "turn", "decel_stop", "accel"],
        "protocol": "shared VLM prefill, K denoisings with shared seeds, teacher-forced CoC",
        "configs": [{"name": n, **m} for n, m, _, _ in cfgs],
        "gpu": torch.cuda.get_device_name(device),
    }, indent=2))

    results = {n: {"ade": [], "fde": [], "nll": []} for n, _, _, _ in cfgs}
    buckets, clip_ids_done = [], []
    for ci, clip_id in enumerate(clips):
        t0 = time.time()
        data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
        inputs = lib.build_inputs(model, processor, data, "cuda")
        prompt_len = inputs["input_ids"].shape[1]
        gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()
        buckets.append(el.bucket(gt_xy))
        clip_ids_done.append(clip_id)
        seeds = [args.seed + ci * 100 + k for k in range(args.k)]

        masks.reset()
        torch.manual_seed(args.seed + ci)
        torch.cuda.manual_seed_all(args.seed + ci)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
        coc_start, coc_end = prompt_len, roll["eos_pos"] + 1
        seq_tf = roll["sequences"][:, :coc_end].clone()
        del roll

        for name, _, q, m in cfgs:
            masks.set(q=q, mlp=m)
            ade, fde, nll = eval_config(model, inputs, seq_tf, coc_start, coc_end, gt_xy, seeds)
            results[name]["ade"].append(ade)
            results[name]["fde"].append(fde)
            results[name]["nll"].append(nll)
        masks.reset()

        b = np.mean(results["baseline"]["ade"])
        print(f"[{ci + 1}/{len(clips)}] {clip_id} {buckets[-1]:10s} "
              f"baseMinADE={b:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if (ci + 1) % 5 == 0 or ci + 1 == len(clips):
            save(out_dir, cfgs, results, buckets, clip_ids_done, ci + 1)
    save(out_dir, cfgs, results, buckets, clip_ids_done, len(clips))
    print("saved ->", out_dir, flush=True)


def save(out_dir, cfgs, results, buckets, clip_ids, n):
    (out_dir / "metrics.json").write_text(json.dumps({
        "n_clips": n, "buckets": buckets, "clip_ids": clip_ids,
        "configs": [n_ for n_, _, _, _ in cfgs],
        "meta": {n_: m for n_, m, _, _ in cfgs},
        "per_clip": {k: v for k, v in results.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
