"""Pathway map, Stage 2: VLM-internal edge knockout with layer windows.

Stage 1 blocked what the expert may read out of the finished cache. Stage 2 blocks
edges *inside* the VLM forward, so the cache itself is built differently -- which lets
the same knockout be read on two channels at once: the CoC NLL (language) and the
trajectory (minADE). Map the Flow has only the first; the second is what makes this a
reasoning-action map rather than a VideoQA one.

Mechanics. Qwen3VL calls `self_attn(attention_mask=...)` by keyword, and sdpa sets
`is_causal = ... and attention_mask is None`, so supplying a mask turns the implicit
causal mask off. The injected mask therefore carries causality itself: start from the
additive causal mask and add finfo.min at the (query span x key span) block. A
forward_pre_hook installs it on the layers in the window and leaves every other layer
on the stock fast path.

Token order in the fused prompt (verified from helper.create_message) is
  sink | system text | [vision images, camera-label text interleaved] | traj-history |
  instruction text | <|cot_start|> | CoC
so causality already forbids e.g. hist->vision; only the edges below can exist.

Usage:
  bash experiments/head_analysis/run_pathway.sh 20 --stage2 --gpu 4 \
      --num-clips 13 --clip-offset 0 --k 8 --exp-id pathway_e_s0
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

import analysis_lib as lib  # noqa: E402
import eval_lib as el  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  also installs the gated-repo hub patch

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
# k=9 disjoint windows covering all 36 layers, matching Map the Flow's window size.
BANDS = {"L0-8": (0, 9), "L9-17": (9, 18), "L18-26": (18, 27),
         "L27-35": (27, 36), "all": (0, 36)}


def clip_seed(seed, clip_id):
    return int(hashlib.sha256(f"{seed}:{clip_id}".encode()).hexdigest()[:4], 16)


class EdgeBlocker:
    """Injects a per-layer additive attention mask into the VLM text stack."""

    def __init__(self, layers):
        self.layers = layers
        self.handles = []
        self.mask = None      # (1, 1, T, T) additive, or None to stay on the fast path
        self.active = set()   # layer indices the mask applies to

    def register(self):
        for i, layer in enumerate(self.layers):
            self.handles.append(
                layer.self_attn.register_forward_pre_hook(self._hook(i), with_kwargs=True))

    def _hook(self, i):
        def fn(_mod, args, kwargs):
            if self.mask is not None and i in self.active:
                kwargs["attention_mask"] = self.mask
            return args, kwargs
        return fn

    def set(self, mask, layers):
        self.mask, self.active = mask, set(layers)

    def clear(self):
        self.mask, self.active = None, set()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def image_runs(vision_mask):
    """Contiguous vision runs = one per image (4 cameras x 4 frames = 16)."""
    pos = torch.nonzero(vision_mask).flatten().tolist()
    runs, start, prev = [], pos[0], pos[0]
    for p in pos[1:]:
        if p != prev + 1:
            runs.append((start, prev + 1))
            start = p
        prev = p
    runs.append((start, prev + 1))
    return runs


def edge_specs(spans, prompt_len, coc_start, coc_end, n_frames=4):
    """(name, [(query_idx, key_idx), ...]) for each VLM-internal edge type."""
    T = coc_end
    vis = spans["vision"]
    runs = image_runs(vis)
    n_cam = len(runs) // n_frames
    cam_imgs = [[runs[c * n_frames + f] for f in range(n_frames)] for c in range(n_cam)]

    def idx(mask):
        return torch.nonzero(mask[:T] if mask.shape[0] >= T else mask).flatten()

    def rng(a, b):
        return torch.arange(a, b)

    def cat(blocks):
        return torch.cat([rng(a, b) for a, b in blocks]) if blocks else torch.empty(0, dtype=torch.long)

    vis_i = idx(vis)
    hist_i = idx(spans["hist"])
    sink_i = idx(spans["sink"])
    last_vis = int(vis_i[-1])
    text_i = idx(spans["text"])
    instr_i = text_i[text_i > last_vis]          # traj markers + instruction + <|cot_start|>
    coc_i = rng(coc_start - 1, coc_end)          # the token that predicts CoC, plus the CoC
    all_i = rng(0, T)

    specs = {}
    # E1 cross-frame: image j of a camera <- earlier images of the SAME camera
    e1 = []
    for imgs in cam_imgs:
        for j in range(1, n_frames):
            e1.append((cat([imgs[j]]), cat(imgs[:j])))
    specs["E1_crossframe"] = e1
    # E2 cross-camera: camera c <- all earlier cameras
    e2 = []
    for c in range(1, n_cam):
        e2.append((cat(cam_imgs[c]), cat([b for cc in range(c) for b in cam_imgs[cc]])))
    specs["E2_crosscam"] = e2
    specs["E3_hist_vision"] = [(hist_i, vis_i)]
    specs["E4_instr_vision"] = [(instr_i, vis_i)]
    specs["E5_coc_vision"] = [(coc_i, vis_i)]
    specs["E6_coc_hist"] = [(coc_i, hist_i)]
    specs["E7_coc_instr"] = [(coc_i, instr_i)]
    specs["E8_all_sink"] = [(all_i, sink_i)]
    return specs


def build_mask(T, blocks, device, dtype):
    """Additive causal mask with the (query x key) blocks additionally forbidden."""
    neg = torch.finfo(dtype).min
    m = torch.full((T, T), neg, device=device, dtype=dtype).triu_(1)
    for q, k in blocks:
        if len(q) == 0 or len(k) == 0:
            continue
        m[q.to(device).unsqueeze(1), k.to(device).unsqueeze(0)] = neg
    return m.view(1, 1, T, T)


@torch.no_grad()
def eval_edge(model, inputs, seq_tf, coc_start, coc_end, gt_xy, seeds, blocker, mask, layers):
    """One masked teacher-forced forward -> CoC NLL, then K denoisings -> minADE."""
    blocker.set(mask, layers)
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.vlm.model(
                input_ids=seq_tf, attention_mask=torch.ones_like(seq_tf),
                pixel_values=inputs["tokenized_data"]["pixel_values"],
                image_grid_thw=inputs["tokenized_data"]["image_grid_thw"], use_cache=True,
            )
            logits = model.vlm.lm_head(
                out.last_hidden_state[:, coc_start - 1 : coc_end - 1]).float()
            nll = torch.nn.functional.cross_entropy(
                logits[0], seq_tf[0, coc_start:coc_end]).item()
            cache = out.past_key_values
            prefill = cache.get_seq_length()
            offset = torch.tensor([prefill], device=seq_tf.device)
            pm = torch.ones(1, prefill, device=seq_tf.device, dtype=torch.long)
            preds = []
            for s in seeds:
                action = lib.denoise_with_cache(model, cache, out.rope_deltas, offset, pm, seed=s)
                pred_xyz, _ = model.action_space.action_to_traj(
                    action.float(), inputs["ego_history_xyz"][:, -1].float(),
                    inputs["ego_history_rot"][:, -1].float(),
                )
                preds.append(pred_xyz[0, :, :2].cpu().numpy())
    finally:
        blocker.clear()
    ade, fde = el.min_metrics(np.stack(preds), gt_xy)
    del out, cache
    return ade, fde, nll


def save(out_dir, cfg_names, cfg_meta, results, buckets, clip_ids, n):
    (out_dir / "metrics.json").write_text(json.dumps({
        "n_clips": n, "buckets": buckets, "clip_ids": clip_ids,
        "configs": cfg_names, "meta": cfg_meta, "per_clip": results,
    }, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-clips", type=int, default=13)
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--exp-id", type=str, default="pathway_e_v1")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--reserve-gb", type=float, default=34.0)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--clip-offset", type=int, default=0)
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
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")

    layers = model.vlm.model.language_model.layers
    blocker = EdgeBlocker(layers)
    blocker.register()

    results, buckets, clip_ids_done = {}, [], []
    cfg_names, cfg_meta = None, None

    for ci, clip_id in enumerate(clips):
        t0 = time.time()
        data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
        inputs = lib.build_inputs(model, processor, data, "cuda")
        prompt_len = inputs["input_ids"].shape[1]
        spans = lib.compute_spans(model, inputs["input_ids"])
        gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()
        base_seed = clip_seed(args.seed, clip_id)
        seeds = [base_seed + k for k in range(args.k)]

        torch.manual_seed(base_seed)
        torch.cuda.manual_seed_all(base_seed)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
        coc_start, coc_end = prompt_len, roll["eos_pos"] + 1
        seq_tf = roll["sequences"][:, :coc_end].clone()
        del roll

        specs = edge_specs(spans, prompt_len, coc_start, coc_end)
        cfgs = [("E0_none", {"edge": "none", "band": "-", "n_blocks": 0}, None, None),
                # integrity check: an injected mask with no blocks is just the causal mask,
                # so this must reproduce E0_none. If it does not, the mask construction is
                # wrong and every other row is meaningless.
                ("E0_causalonly", {"edge": "causal_mask_only", "band": "all", "n_blocks": 0},
                 [], list(range(36)))]
        for ename, blocks in specs.items():
            for bname, (blo, bhi) in BANDS.items():
                cfgs.append((f"{ename}@{bname}",
                             {"edge": ename, "band": bname, "n_blocks": len(blocks),
                              "n_pairs": int(sum(len(q) * len(k) for q, k in blocks))},
                             blocks, list(range(blo, bhi))))
        if cfg_names is None:
            cfg_names = [c[0] for c in cfgs]
            cfg_meta = {c[0]: c[1] for c in cfgs}
            results = {c: {"ade": [], "fde": [], "nll": []} for c in cfg_names}
            print(f"{len(cfgs)} configs x {len(clips)} clips x K={args.k}", flush=True)
            (out_dir / "config.json").write_text(json.dumps({
                "model": "nvidia/Alpamayo-1.5-10B", "stage": "E (VLM-internal edges)",
                "eval_split": args.split, "num_clips": len(clips), "clip_ids": clips,
                "clip_offset": args.clip_offset, "k_samples": args.k, "seed": args.seed,
                "seed_from": "sha256(f'{seed}:{clip_id}')[:4] -- shard-invariant",
                "bands": {k: list(v) for k, v in BANDS.items()},
                "protocol": ("per config: one masked teacher-forced VLM forward (readout 1 = "
                             "CoC NLL) then K denoisings on that cache (readout 2 = minADE); "
                             "injected mask carries causality since sdpa drops is_causal when "
                             "a mask is present"),
                "configs": [{"name": n, **m} for n, m, _, _ in cfgs],
                "plan": "plans/2026-08-25_pathway-map.md",
                "gpu": torch.cuda.get_device_name(device),
            }, indent=2))

        T = coc_end
        for name, _, blocks, blayers in cfgs:
            mask = None if blocks is None else build_mask(T, blocks, "cuda", torch.bfloat16)
            ade, fde, nll = eval_edge(model, inputs, seq_tf, coc_start, coc_end, gt_xy,
                                      seeds, blocker, mask, blayers or [])
            results[name]["ade"].append(ade)
            results[name]["fde"].append(fde)
            results[name]["nll"].append(nll)
            del mask

        buckets.append(el.bucket(gt_xy))
        clip_ids_done.append(clip_id)
        b = results["E0_none"]
        print(f"[{ci + 1}/{len(clips)}] {clip_id} {buckets[-1]:10s} T={T} "
              f"base={b['ade'][-1]:.3f}/nll={b['nll'][-1]:.3f} "
              f"cocvis={results['E5_coc_vision@all']['ade'][-1]:.3f}/"
              f"{results['E5_coc_vision@all']['nll'][-1]:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if (ci + 1) % 2 == 0 or ci + 1 == len(clips):
            save(out_dir, cfg_names, cfg_meta, results, buckets, clip_ids_done, ci + 1)

    blocker.remove()
    save(out_dir, cfg_names, cfg_meta, results, buckets, clip_ids_done, len(clips))
    print("saved ->", out_dir, flush=True)


if __name__ == "__main__":
    main()
