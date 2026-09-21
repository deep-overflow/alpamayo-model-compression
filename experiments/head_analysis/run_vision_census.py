"""Where do VISION queries put their attention, per head and layer?

plans/2026-09-21_importance-causal-validation.md, part C (the observational half). For every
VLM head, the attention mass of the 2,880 vision-token queries on five disjoint key groups:

  sink | text (system text, camera labels) | own image | same-camera earlier frames |
  other cameras

The last three are the vision-vision interaction the nested knockouts of run_pathway2.py
--cuts remove (V3, E1, E2). Same clips, same rollout seeds as the pathway map, so the census
describes the forward passes those knockouts intervene on. Eager attention, one layer's
probabilities at a time (a (32, T, T) fp32 map is 1.2 GB; nothing is kept but the sums).

Usage:
  bash experiments/head_analysis/run_pathway.sh 3 --census --gpu 4 --num-clips 50 \
      --exp-id vision_census_v1
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
from expert_per_clip import reserve_gpu  # noqa: E402  also installs the gated-repo hub patch
from run_pathway2 import clip_seed, image_runs  # noqa: E402

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
GROUPS = ("sink", "text", "own_image", "cross_frame", "cross_camera")


def key_groups(spans, T, n_frames=4):
    """(vis_idx (Nv,), onehot (5, Nv, T) float32 on cuda): the group of key k for vision query q."""
    vis = spans["vision"]
    runs = image_runs(vis)
    img = torch.full((T,), -1, dtype=torch.long)
    for i, (a, b) in enumerate(runs):
        img[a:b] = i
    sink = torch.zeros(T, dtype=torch.bool)
    sink[: spans["sink"].shape[0]] = spans["sink"]
    vis_idx = torch.nonzero(img >= 0).flatten()
    qi = img[vis_idx].unsqueeze(1)  # (Nv, 1) image of the query
    ki = img.unsqueeze(0)  # (1, T) image of the key, -1 for non-vision
    own = ki == qi
    same_cam = (ki >= 0) & (ki // n_frames == qi // n_frames) & ~own
    other_cam = (ki >= 0) & (ki // n_frames != qi // n_frames)
    is_sink = sink.unsqueeze(0).expand_as(own)
    text = ~(own | same_cam | other_cam | is_sink)
    onehot = torch.stack([is_sink, text, own, same_cam, other_cam]).float()  # (5, Nv, T)
    return vis_idx.to("cuda"), onehot.to("cuda"), len(runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-clips", type=int, default=50)
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--clip-offset", type=int, default=0)
    ap.add_argument("--exp-id", type=str, default="vision_census_v1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--reserve-gb", type=float, default=34.0)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--outputs-root", type=str, default=None)
    args = ap.parse_args()

    root = Path(args.outputs_root) if args.outputs_root else REPO / "outputs"
    out_dir = root / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    split = json.loads((root / "split.json").read_text())
    clips = split[args.split][args.clip_offset : args.clip_offset + args.num_clips]

    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    print(f"using {device}", flush=True)
    model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    lib.set_expert_attn_impl(model, "sdpa")
    layers = model.vlm.model.language_model.layers
    n_l, n_h = len(layers), model.vlm.config.text_config.num_attention_heads

    state = {}

    def make_hook(li):
        def hook(module, a, kw, output):
            attn = output[1]  # (1, H, T, T) attention probabilities (eager)
            assert attn is not None, "eager attention required"
            rows = attn[0].index_select(1, state["vis_idx"]).float()  # (H, Nv, T)
            m = rows.reshape(n_h, -1) @ state["onehot"].reshape(len(GROUPS), -1).T  # (H, 5)
            state["mass"][li] = (m / state["vis_idx"].numel()).cpu().numpy()
        return hook

    (out_dir / "config.json").write_text(json.dumps({
        "model": "nvidia/Alpamayo-1.5-10B",
        "plan": "plans/2026-09-21_importance-causal-validation.md (part C, observational)",
        "eval_split": args.split, "num_clips": len(clips), "clip_ids": clips,
        "clip_offset": args.clip_offset, "seed": args.seed,
        "seed_from": "sha256(f'{seed}:{clip_id}')[:4], as run_pathway2",
        "groups": GROUPS, "queries": "vision tokens only",
        "gpu": torch.cuda.get_device_name(device),
    }, indent=2))

    per_clip, done, n_vis = [], [], []
    for ci, clip_id in enumerate(clips):
        t0 = time.time()
        data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
        inputs = lib.build_inputs(model, processor, data, "cuda")
        prompt_len = inputs["input_ids"].shape[1]
        spans = lib.compute_spans(model, inputs["input_ids"])
        base_seed = clip_seed(args.seed, clip_id)
        torch.manual_seed(base_seed)
        torch.cuda.manual_seed_all(base_seed)
        lib.set_vlm_attn_impl(model, "sdpa")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
        seq_tf = roll["sequences"][:, : roll["eos_pos"] + 1].clone()  # (1, T)
        del roll

        T = seq_tf.shape[1]
        state["vis_idx"], state["onehot"], n_img = key_groups(spans, T)
        state["mass"] = np.zeros((n_l, n_h, len(GROUPS)))
        lib.set_vlm_attn_impl(model, "eager")
        handles = [layer.self_attn.register_forward_hook(make_hook(i), with_kwargs=True)
                   for i, layer in enumerate(layers)]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model.vlm.model(
                input_ids=seq_tf, attention_mask=torch.ones_like(seq_tf),
                pixel_values=inputs["tokenized_data"]["pixel_values"],
                image_grid_thw=inputs["tokenized_data"]["image_grid_thw"], use_cache=False,
            )
        for h in handles:
            h.remove()
        per_clip.append(state["mass"].copy())
        done.append(clip_id)
        n_vis.append(int(state["vis_idx"].numel()))
        lay = state["mass"].mean(1)  # (L, 5)
        print(f"[{ci + 1}/{len(clips)}] {clip_id} T={T} prompt={prompt_len} images={n_img} Nv={n_vis[-1]} "
              f"row-sum {state['mass'].sum(-1).mean():.4f} | cross-frame+camera mass L3 {lay[3, 3:].sum():.3f} "
              f"L12 {lay[12, 3:].sum():.3f} L20 {lay[20, 3:].sum():.3f} L30 {lay[30, 3:].sum():.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        del state["onehot"]
        if (ci + 1) % 5 == 0 or ci + 1 == len(clips):
            arr = np.stack(per_clip)  # (N, L, H, 5)
            np.savez(out_dir / "census.npz", mass=arr.mean(0), mass_by_clip_layer=arr.mean(2))
            (out_dir / "metrics.json").write_text(json.dumps({
                "n_clips": len(done), "clip_ids": done, "n_vision_tokens": n_vis, "groups": GROUPS,
                "layer_mean_mass": arr.mean((0, 2)).tolist()}, indent=2))
    lay = np.stack(per_clip).mean((0, 2))  # (L, 5)
    lines = [f"vision-query attention census -- {len(done)} clips", "",
             "layer  " + "  ".join(f"{g:>12s}" for g in GROUPS)]
    lines += [f"{li:5d}  " + "  ".join(f"{v:12.3f}" for v in lay[li]) for li in range(n_l)]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("saved ->", out_dir, flush=True)


if __name__ == "__main__":
    main()
