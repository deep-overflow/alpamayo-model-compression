"""Phase P2: mask ablation comparing the two importance criteria.

For each clip the unpruned model produces a reference CoC once. Every config then
runs a teacher-forced VLM forward on that same token sequence, so the CoC text is
held fixed and the measurement isolates "does the pruned cache still support the
trajectory". CoC NLL comes out of the same forward for free.

P1 found the two criteria agree in early layers and diverge after layer ~22, so
each ratio is run twice: pruning all layers, and pruning only layers 22-35.

Usage:
  bash run_retry.sh 20 experiments/head_analysis/run_ablation.py --gpu 0 --num-clips 50
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
import mask_lib as ml  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  also installs the gated-repo hub patch

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

REPO = Path("/workspace/alpamayo-model-compression")
LATE_START = 22  # where P1 saw the two criteria diverge


def build_configs(imp, mag, n_layers, n_heads, intermediate, ratios, seed):
    """(name, meta, vlm_q_mask, vlm_mlp_mask) for every ablation config."""
    rng = np.random.RandomState(seed)
    all_layers = list(range(n_layers))
    late_layers = list(range(LATE_START, n_layers))
    scopes = {"all": all_layers, "late": late_layers}
    criteria = {
        "traj": (imp["traj_vlm_q"], imp["traj_vlm_mlp"]),
        "coc": (imp["coc_vlm_q"], imp["coc_vlm_mlp"]),
        "magnitude": mag,
        "random": None,
    }
    cfgs = [("baseline", {"kind": "baseline"}, None, None)]

    for scope, layers_scope in scopes.items():
        for ratio in ratios:
            for crit, sc in criteria.items():
                r = rng if crit == "random" else None
                if sc is None:
                    qs = np.zeros((n_layers, n_heads))
                    ms = np.zeros((n_layers, intermediate))
                else:
                    qs, ms = sc
                q = ml.select_mask(qs, ratio, layers_scope, rng=r)
                m = ml.select_mask(ms, ratio, layers_scope, rng=r)
                cfgs.append((
                    f"{crit}_{scope}_r{int(ratio * 100)}",
                    {"kind": "width", "criterion": crit, "scope": scope, "ratio": ratio},
                    q, m,
                ))

    # KV groups: drop the n weakest per layer, over all layers
    for crit, key in [("traj", "traj_kv"), ("coc", "coc_kv"), ("random", None)]:
        for n_drop in (1, 2, 3):
            if key is None:
                sc = np.zeros((n_layers, imp["traj_kv_k"].shape[1]))
                r = rng
            else:
                sc = imp[f"{key}_k"] + imp[f"{key}_v"]
                r = None
            q = ml.kv_group_mask(sc, n_drop, n_heads, list(range(n_layers)), rng=r)
            cfgs.append((
                f"kv_{crit}_drop{n_drop}",
                {"kind": "kv_group", "criterion": crit, "n_drop": n_drop},
                q, None,
            ))

    # layer 35 attention + MLP: exactly zero trajectory importance (P1)
    q = np.ones((n_layers, n_heads)); q[n_layers - 1] = 0.0
    m = np.ones((n_layers, intermediate)); m[n_layers - 1] = 0.0
    cfgs.append(("drop_last_layer", {"kind": "structural", "note": "layer 35 attn+mlp"}, q, m))
    return cfgs


@torch.no_grad()
def evaluate(model, inputs, seq_tf, coc_start, coc_end, gt_xy, seed):
    """Teacher-forced forward -> cache -> denoise. Returns (ADE, CoC NLL)."""
    attention_mask = torch.ones_like(seq_tf)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.vlm.model(
            input_ids=seq_tf, attention_mask=attention_mask,
            pixel_values=inputs["tokenized_data"]["pixel_values"],
            image_grid_thw=inputs["tokenized_data"]["image_grid_thw"],
            use_cache=True,
        )
        logits = model.vlm.lm_head(out.last_hidden_state[:, coc_start - 1 : coc_end - 1]).float()
        nll = torch.nn.functional.cross_entropy(
            logits[0], seq_tf[0, coc_start:coc_end]
        ).item()
        cache = out.past_key_values
        prefill = cache.get_seq_length()
        offset = torch.tensor([prefill], device=seq_tf.device)
        prefix_mask = torch.ones(1, prefill, device=seq_tf.device, dtype=torch.long)
        action = lib.denoise_with_cache(
            model, cache, out.rope_deltas, offset, prefix_mask, seed=seed
        )
        pred_xyz, _ = model.action_space.action_to_traj(
            action.float(),
            inputs["ego_history_xyz"][:, -1].float(),
            inputs["ego_history_rot"][:, -1].float(),
        )
    ade = float(np.linalg.norm(pred_xyz[0, :, :2].cpu().numpy() - gt_xy, axis=-1).mean())
    del out, cache
    return ade, nll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-clips", type=int, default=50)
    ap.add_argument("--exp-id", type=str, default="ablation_v1")
    ap.add_argument("--importance", type=str, default="importance_v1")
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.10, 0.25, 0.40, 0.50])
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    split = json.loads((REPO / "outputs" / "split.json").read_text())
    clips = split[args.split][: args.num_clips]
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
    masks = ml.PruneMasks(layers, tc.num_attention_heads, tc.head_dim,
                          tc.intermediate_size, "cuda")
    mag = ml.magnitude_scores(layers, tc.num_attention_heads, tc.head_dim, tc.intermediate_size)
    cfgs = build_configs(imp, mag, tc.num_hidden_layers, tc.num_attention_heads,
                         tc.intermediate_size, args.ratios, args.seed)
    print(f"{len(cfgs)} configs x {len(clips)} clips", flush=True)

    (out_dir / "config.json").write_text(json.dumps({
        "model": "nvidia/Alpamayo-1.5-10B",
        "purpose": "mask ablation: does the trajectory criterion beat the CoC criterion?",
        "eval_split": args.split, "num_clips": len(clips), "clip_ids": clips,
        "importance_from": args.importance, "ratios": args.ratios,
        "late_start_layer": LATE_START, "seed": args.seed,
        "protocol": "teacher-forced reference CoC from the unpruned model, fixed per clip",
        "configs": [{"name": n, **m} for n, m, _, _ in cfgs],
        "gpu": torch.cuda.get_device_name(device),
    }, indent=2))

    results = {n: {"ade": [], "nll": []} for n, _, _, _ in cfgs}
    for ci, clip_id in enumerate(clips):
        t0 = time.time()
        data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
        inputs = lib.build_inputs(model, processor, data, "cuda")
        prompt_len = inputs["input_ids"].shape[1]
        gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()

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
            ade, nll = evaluate(model, inputs, seq_tf, coc_start, coc_end, gt_xy,
                                args.seed + ci)
            results[name]["ade"].append(ade)
            results[name]["nll"].append(nll)
        masks.reset()

        base = np.mean(results["baseline"]["ade"])
        print(f"[{ci + 1}/{len(clips)}] {clip_id} coc={coc_end - coc_start} "
              f"baseADE={base:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if (ci + 1) % 5 == 0 or ci + 1 == len(clips):
            save(out_dir, cfgs, results, ci + 1)
    save(out_dir, cfgs, results, len(clips))
    print("saved ->", out_dir, flush=True)


def save(out_dir, cfgs, results, n):
    rows = []
    base_ade = np.mean(results["baseline"]["ade"]) if results["baseline"]["ade"] else float("nan")
    base_nll = np.mean(results["baseline"]["nll"]) if results["baseline"]["nll"] else float("nan")
    for name, meta, _, _ in cfgs:
        a, l = np.array(results[name]["ade"]), np.array(results[name]["nll"])
        if len(a) == 0:
            continue
        rows.append({
            "name": name, **meta,
            "ade_mean": float(a.mean()), "ade_median": float(np.median(a)),
            "ade_se": float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0,
            "ade_delta": float(a.mean() - base_ade),
            "nll_mean": float(l.mean()), "nll_delta": float(l.mean() - base_nll),
            "n": len(a),
        })
    (out_dir / "metrics.json").write_text(json.dumps(
        {"n_clips": n, "rows": rows,
         "per_clip": {k: v for k, v in results.items()}}, indent=2))


if __name__ == "__main__":
    main()
