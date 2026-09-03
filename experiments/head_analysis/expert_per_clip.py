"""Per-clip expert cache-attention breakdown, with prompt-text and generated-CoC separated.

The Phase A run lumped prompt text and generated CoC into one "text" region. Prompt
text is ~400 tokens while CoC is ~15, so that curve could be dominated by prompt text.
This script separates them and keeps per-clip records so outlier samples are identifiable.

Usage: python expert_per_clip.py --num-clips 50 --exp-id expert_per_clip_v1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

import transformers.utils.hub as _hub  # noqa: E402

_orig_cached_files = _hub.cached_files


def _patched_cached_files(path_or_repo_id, *a, **kw):
    if str(path_or_repo_id) == "nvidia/Cosmos-Reason2-8B":
        kw["local_files_only"] = True
    return _orig_cached_files(path_or_repo_id, *a, **kw)


_hub.cached_files = _patched_cached_files

import analysis_lib as lib  # noqa: E402

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

REPO = Path("/workspace/alpamayo-model-compression")
REGIONS = ["vision", "prompt_text", "coc", "hist", "sink", "own"]


class ExpertRegionCollector:
    """Per-layer, per-head expert attention mass with prompt-text / CoC separated."""

    def __init__(self, n_layers, n_heads, spans, prompt_len, coc_start, coc_end, prefill_len):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.spans = spans
        self.prompt_len = prompt_len
        self.coc_start = coc_start
        self.coc_end = coc_end
        self.prefill_len = prefill_len
        self.sums = {r: np.zeros((n_layers, n_heads)) for r in REGIONS}
        # per-token mass, so a small region is not penalised purely for being small
        self.per_token = {r: np.zeros((n_layers, n_heads)) for r in REGIONS}
        self.sizes = {}
        self.calls = np.zeros(n_layers)
        self._handles = []

    def _masks(self, tk, device):
        m = {}
        for k in ("vision", "hist", "sink"):
            full = torch.zeros(tk, dtype=torch.bool, device=device)
            full[: self.prompt_len] = self.spans[k].to(device)
            m[k] = full
        ptext = torch.zeros(tk, dtype=torch.bool, device=device)
        ptext[: self.prompt_len] = self.spans["text"].to(device)
        m["prompt_text"] = ptext
        coc = torch.zeros(tk, dtype=torch.bool, device=device)
        coc[self.coc_start : min(self.coc_end, tk)] = True
        m["coc"] = coc
        own = torch.zeros(tk, dtype=torch.bool, device=device)
        own[self.prefill_len :] = True
        m["own"] = own
        return m

    def make_hook(self, layer_idx):
        def hook(module, args, kwargs, output):
            a = output[1][0].double()  # (H, 64, Tk)
            tk = a.shape[-1]
            nq = a.shape[1]
            masks = self._masks(tk, a.device)
            for r in REGIONS:
                mass = a[:, :, masks[r]].sum((1, 2)).cpu().numpy() / nq
                size = int(masks[r].sum().item())
                self.sums[r][layer_idx] += mass
                self.per_token[r][layer_idx] += mass / max(size, 1)
                self.sizes[r] = size
            self.calls[layer_idx] += 1
        return hook

    def register(self, model):
        for i, layer in enumerate(model.expert.layers):
            self._handles.append(
                layer.self_attn.register_forward_hook(self.make_hook(i), with_kwargs=True)
            )

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def finalize(self):
        c = np.maximum(self.calls, 1)[:, None]
        return (
            {r: self.sums[r] / c for r in REGIONS},
            {r: self.per_token[r] / c for r in REGIONS},
            dict(self.sizes),
        )


def reserve_gpu(need_gb, chunk_gb=1.0, devices=None):
    """Return the first device where need_gb can actually be reserved.

    The reservation is freed but stays in PyTorch's caching allocator, so it is
    held against other processes for the lifetime of this process.
    devices restricts the scan, so parallel runs can be pinned to distinct GPUs.
    """
    for idx in devices if devices is not None else range(torch.cuda.device_count()):
        dev = torch.device(f"cuda:{idx}")
        chunks = []
        try:
            while sum(c.numel() * 2 for c in chunks) < need_gb * 1024**3:
                chunks.append(
                    torch.empty(int(chunk_gb * 1024**3 // 2), dtype=torch.bfloat16, device=dev)
                )
        except (torch.OutOfMemoryError, torch.AcceleratorError) as e:
            # A card with no free memory at all raises AcceleratorError, not
            # OutOfMemoryError -- the two are siblings under RuntimeError in torch 2.8,
            # so catching only the latter aborts the whole scan on the first full card
            # instead of moving to the next one.
            if not isinstance(e, torch.OutOfMemoryError) and "out of memory" not in str(e):
                raise
        got = sum(c.numel() * 2 for c in chunks) / 1024**3
        del chunks
        if got >= need_gb:
            torch.cuda.set_device(dev)
            print(f"reserved {got:.1f} GiB on cuda:{idx}", flush=True)
            return dev
        print(f"  cuda:{idx}: only {got:.1f} GiB available, skipping", flush=True)
        torch.cuda.empty_cache()
    raise torch.OutOfMemoryError(f"no GPU with {need_gb} GiB free")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-clips", type=int, default=50)
    ap.add_argument("--exp-id", type=str, default="expert_per_clip_v1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--reserve-gb", type=float, default=24.0)
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # same clip selection as head_analysis_v1 so results are directly comparable
    clip_df = pd.read_parquet(REPO / "notebooks" / "clip_ids.parquet")
    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(clip_df))
    calib_ids = clip_df.iloc[order[: args.num_clips]]["clip_id"].tolist()

    (out_dir / "config.json").write_text(json.dumps({
        "model": "nvidia/Alpamayo-1.5-10B",
        "purpose": "per-clip expert cache attention with prompt_text/CoC separated",
        "num_clips": len(calib_ids),
        "clip_ids": calib_ids,
        "seed": args.seed,
        "max_generation_length": args.max_gen,
        "regions": REGIONS,
    }, indent=2))

    # Pick the GPU by actually reserving memory on it, not by reading nvidia-smi:
    # on this shared box nvidia-smi's free figure does not match what a process can
    # allocate, and any gap between checking and loading is a race another process wins.
    # Chunked because a single contiguous block fails on a fragmented device.
    device = reserve_gpu(args.reserve_gb)
    print(f"using {device}", flush=True)

    print("Loading model...", flush=True)
    model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "eager")

    LE = len(model.expert.layers)
    HE = model.expert.config.num_attention_heads

    per_clip = {r: [] for r in REGIONS}
    per_clip_pt = {r: [] for r in REGIONS}
    records = []

    for ci, clip_id in enumerate(calib_ids):
        t0 = time.time()
        try:
            data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
        except Exception as e:
            print(f"[{ci}] {clip_id} LOAD FAILED: {e}", flush=True)
            continue

        inputs = lib.build_inputs(model, processor, data, "cuda")
        spans = lib.compute_spans(model, inputs["input_ids"])
        prompt_len = inputs["input_ids"].shape[1]

        torch.manual_seed(args.seed + ci)
        torch.cuda.manual_seed_all(args.seed + ci)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
        eos_pos = roll["eos_pos"]
        coc_start, coc_end = prompt_len, eos_pos + 1
        prefill = roll["past_key_values"].get_seq_length()

        coll = ExpertRegionCollector(LE, HE, spans, prompt_len, coc_start, coc_end, prefill)
        coll.register(model)
        offset = torch.tensor([eos_pos + 1], device="cuda")
        prefix_mask = torch.ones(1, prefill, device="cuda", dtype=torch.long)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            action = lib.denoise_with_cache(
                model, roll["past_key_values"], roll["rope_deltas"], offset, prefix_mask,
                seed=args.seed + ci,
            )
            pred_xyz, _ = model.action_space.action_to_traj(
                action.float(),
                inputs["ego_history_xyz"][:, -1].float(),
                inputs["ego_history_rot"][:, -1].float(),
            )
        coll.remove()
        mass, per_token, sizes = coll.finalize()

        gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()
        ade = float(np.linalg.norm(pred_xyz[0, :, :2].cpu().numpy() - gt_xy, axis=-1).mean())
        coc_text = model.tokenizer.decode(
            roll["sequences"][0, coc_start:coc_end].tolist()
        )

        for r in REGIONS:
            per_clip[r].append(mass[r])
            per_clip_pt[r].append(per_token[r])
        records.append({
            "clip_id": clip_id,
            "prompt_len": prompt_len,
            "coc_len": coc_end - coc_start,
            "minADE": ade,
            "coc_text": coc_text,
            "sizes": sizes,
            "L0_coc": float(mass["coc"][0].mean()),
            "L0_prompt_text": float(mass["prompt_text"][0].mean()),
            "L0_vision": float(mass["vision"][0].mean()),
            "mean_coc": float(np.mean([mass["coc"][i].mean() for i in range(LE)])),
        })
        print(
            f"[{ci + 1}/{len(calib_ids)}] {clip_id} coc_len={coc_end - coc_start} "
            f"L0: coc={records[-1]['L0_coc']:.3f} ptext={records[-1]['L0_prompt_text']:.3f} "
            f"vis={records[-1]['L0_vision']:.3f} ({time.time() - t0:.0f}s)",
            flush=True,
        )
        del roll
        torch.cuda.empty_cache()

    arrays = {}
    for r in REGIONS:
        arrays[f"mass_{r}"] = np.stack(per_clip[r])          # (N, 36, 16)
        arrays[f"pertoken_{r}"] = np.stack(per_clip_pt[r])   # (N, 36, 16)
    np.savez(out_dir / "expert_per_clip.npz", **arrays)
    (out_dir / "metrics.json").write_text(json.dumps(records, indent=2))
    print(f"saved -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
