"""Cache-use map: how much does the action expert read the VLM's KV cache, per layer
and per denoising step?  (plans/2026-08-28_cache-use-map.md)

Everything measured so far about the expert's cache reads is summed over layers and
steps (vision 72.6% / text 16.7% / hist 3.9% / CoC 2.3% / sink 1.1% / own 3.6%). This
runner resolves the 36 layers x 10 Euler steps, on ONE cache per clip (rollout ->
teacher-forced VLM forward), so every number below is clip- and seed-paired.

Stage A (observational, eager attention, one denoise per clip): per (step, layer, head)
  attention mass on the cache vs the 64 own tokens, the cache mass by span, entropy, and
  the READOUT share ||sum_cache a v|| / (||sum_cache a v|| + ||sum_own a v||) -- mass
  ignores the size of V, the readout does not.
Stage B (causal, sdpa): block the cache at one (layer, step) cell -- a per-layer pre-hook
  swaps that layer's 4D attention mask for a copy with finfo.min on the cache positions,
  only while the step counter (a hook on action_in_proj, which step_fn calls once per
  step) says it is that step. Measured as `move` = mean waypoint distance from the
  same-seed unblocked trajectory (m), plus minADE vs GT. Cells: (layer, step) x 1 seed
  (360), (layer, head) at all steps x 1 seed (576, the head's mask rows only -- head h reads
  KV group h//2, so this is the expert-side reliance on each cache group), a fixed random
  sample of (layer, head, step) cells x 1 seed (checks whether the 3D map factorises), and
  layer / step marginals, all-block and none-block x K seeds.
Stage C (the pruning question): a second teacher-forced forward with dual_u40_v2's VLM
  masks gives the PRUNED cache for the same text; swapping one (layer, KV group) of the
  dense cache for its pruned version and denoising measures how much that group's actual
  shift moves the trajectory -- per (layer, group), per layer, and all at once (which is
  the cache-shift report's A10 cell). Reliance (Stage B) says which cells the expert
  reads; this says which cells' shift hurts.

Pre-registered gates (judged by analyze_cacheuse.py):
  G0  none-block == unblocked bitwise; all-block reproduces the pathway prefix_mask
      catastrophe; the step counter sees exactly 10 steps
  G1  Spearman(mass, move) and Spearman(readout share, move) over the 360 cells
  G2  step-marginal move monotone in the step (early steps read the cache most)
  G3  cells for 80% of the grid's total move; share of cells under the seed-noise floor
  G4  layer-marginal move vs the sum of that layer's single-cell moves (redundancy)

Usage:
  bash experiments/head_analysis/run_retry_host.sh 60 experiments/head_analysis/run_cacheuse.py \
      --gpu 0 --manifest calib_100 --cache calib --num-clips 25 --clip-offset 0 \
      --exp-id cacheuse_v1_s0
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "evaluation"))
import analysis_lib as lib  # noqa: E402
import eval_lib as el  # noqa: E402
import mask_lib as ml  # noqa: E402
import sample_cache as sc  # noqa: E402
from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  also installs the gated-repo hub patch
from make_slim import build_masks  # noqa: E402
from run_cachediff import span_index, tf_forward  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MODEL_REV = "7aba8293c09993f2e125c6819df05d7fa3e873ea"
N_STEPS = 10
SPANS = ("vision", "text", "hist", "sink", "coc")
STAT_KEYS = ("mass_cache", "mass_own", *[f"mass_{s}" for s in SPANS], "entropy",
             "read_cache", "read_own", "read_share")


class StepCounter:
    """Which Euler step the expert is in. step_fn calls action_in_proj once per step,
    before the expert forward, so during layer hooks `step` is the current index."""

    def __init__(self, model):
        self.step = -1
        self.h = model.action_in_proj.register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        self.step += 1

    def reset(self):
        self.step = -1

    def remove(self):
        self.h.remove()


class CacheBlocker:
    """Blocks the expert's reads of the cache positions at chosen (layer, step) cells by
    handing those layers a copy of the 4D mask with finfo.min over [0, prefill)."""

    def __init__(self, model, counter, prefill):
        self.counter = counter
        self.prefill = prefill
        self.layers, self.steps = set(), set()
        self._blocked = {}  # id(mask) -> (mask, blocked copy); keeps the mask alive
        self.handles = [
            layer.self_attn.register_forward_pre_hook(self._make(li), with_kwargs=True)
            for li, layer in enumerate(model.expert.layers)
        ]

    def set(self, layers, steps, heads=None):
        """heads=None blocks every head of the chosen layers; a list blocks only those
        heads (the mask is expanded to (1, H, 64, T), which eager and sdpa both accept)."""
        self.layers, self.steps = set(layers), set(steps)
        self.heads = None if heads is None else list(heads)
        self._blocked = {}

    def _make(self, li):
        def hook(module, args, kwargs):
            if li not in self.layers or self.counter.step not in self.steps:
                return None
            in_kwargs = "attention_mask" in kwargs
            m = kwargs["attention_mask"] if in_kwargs else args[2]
            ent = self._blocked.get(id(m))
            if ent is None or ent[0] is not m:
                if self.heads is None:
                    b = m.clone()  # (1, 1, 64, prefill + 64)
                    b[..., : self.prefill] = torch.finfo(b.dtype).min
                else:
                    b = m.expand(m.shape[0], module.config.num_attention_heads, *m.shape[2:]).clone()
                    b[:, self.heads, :, : self.prefill] = torch.finfo(b.dtype).min
                self._blocked[id(m)] = (m, b)
                ent = (m, b)
            if in_kwargs:
                return args, {**kwargs, "attention_mask": ent[1]}
            args = list(args)
            args[2] = ent[1]
            return tuple(args), kwargs
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


class StepStatsCollector:
    """Stage A: per (step, layer, head) attention mass, entropy and readout share.
    Registered on the expert's self_attn (eager attention -> output[1] is (1, H, 64, Tk));
    the layer's cache already holds the 64 own tokens' V when the hook fires, so the
    readout split needs no second pass."""

    def __init__(self, model, counter, cache, prefill, sidx):
        ec = model.expert.config
        self.counter, self.cache, self.prefill, self.sidx = counter, cache, prefill, sidx
        self.rep = ec.num_attention_heads // ec.num_key_value_heads
        L, H = ec.num_hidden_layers, ec.num_attention_heads
        self.sums = {k: np.zeros((N_STEPS, L, H)) for k in STAT_KEYS}
        self.calls = np.zeros((N_STEPS, L))
        self.handles = [
            layer.self_attn.register_forward_hook(self._make(li), with_kwargs=True)
            for li, layer in enumerate(model.expert.layers)
        ]

    def _make(self, li):
        def hook(module, args, kwargs, output):
            attn = output[1]
            assert attn is not None, "eager attention required"
            a = attn[0].float()  # (H, 64, Tk)
            P, s = self.prefill, self.counter.step
            S = self.sums
            S["mass_cache"][s, li] += a[:, :, :P].sum(-1).mean(-1).cpu().numpy()
            S["mass_own"][s, li] += a[:, :, P:].sum(-1).mean(-1).cpu().numpy()
            for name in SPANS:
                m = self.sidx[name]
                S[f"mass_{name}"][s, li] += a[:, :, :P][:, :, m].sum(-1).mean(-1).cpu().numpy()
            ent = -(a.clamp_min(1e-12).log() * a).sum(-1) / np.log(a.shape[-1])  # (H, 64)
            S["entropy"][s, li] += ent.mean(-1).cpu().numpy()
            _, v = lib.cache_layer_kv(self.cache, li)  # (1, G, Tk, D) incl. the 64 own tokens
            vrep = v[0].float().repeat_interleave(self.rep, dim=0)  # (H, Tk, D)
            c_cache = torch.einsum("hqk,hkd->hqd", a[:, :, :P], vrep[:, :P])
            c_own = torch.einsum("hqk,hkd->hqd", a[:, :, P:], vrep[:, P:])
            nc = c_cache.norm(dim=-1).mean(-1)  # (H)
            no = c_own.norm(dim=-1).mean(-1)
            S["read_cache"][s, li] += nc.cpu().numpy()
            S["read_own"][s, li] += no.cpu().numpy()
            S["read_share"][s, li] += (nc / (nc + no).clamp_min(1e-12)).cpu().numpy()
            self.calls[s, li] += 1
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def save(out_dir, meta, per_clip, stats_sum, n):
    arrays = {k: np.array(v) for k, v in per_clip.items() if k not in ("clip_ids", "buckets")}
    for k, v in stats_sum.items():
        arrays[f"stat_{k}"] = v / max(n, 1)
    np.savez(out_dir / "cacheuse.npz", **arrays)
    (out_dir / "metrics.json").write_text(json.dumps(
        {**meta, "n_clips": n, "clip_ids": per_clip["clip_ids"], "buckets": per_clip["buckets"],
         "move_all_mean": float(np.mean(per_clip["move_all"])) if n else None,
         "move_none_max": float(np.max(per_clip["move_none"])) if n else None,
         "noise_floor_mean": float(np.mean(per_clip["noise_floor"])) if n else None},
        indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="cacheuse_v1")
    ap.add_argument("--manifest", default="calib_100")
    ap.add_argument("--sets-id", default="eval_sets")
    ap.add_argument("--cache", default="calib")
    ap.add_argument("--num-clips", type=int, default=25)
    ap.add_argument("--clip-offset", type=int, default=0)
    ap.add_argument("--k", type=int, default=4, help="seeds for marginals / all / reference")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--importance", default="importance_stepexp_bw_znorm",
                    help="importance file for the VLM masks of --vlm-config (VLM keys == "
                         "importance_v2, so the VLM half is dual_u40_v2 bit for bit)")
    ap.add_argument("--vlm-config", default="dualexp_u40_e25",
                    help="build_masks config whose VLM half supplies the pruned cache")
    ap.add_argument("--no-swap", action="store_true", help="skip Stage C")
    ap.add_argument("--n-sample3d", type=int, default=120,
                    help="fixed random (layer, head, step) cells, shared by every clip")
    ap.add_argument("--smoke", action="store_true",
                    help="grid restricted to layers {0, 18, 35} x steps {0, 5, 9}")
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--gpu", type=str, default=None)
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(REPO / "outputs" / args.sets_id / f"{args.manifest}.parquet")
    t0_col = df["t0_us"].astype(int) if "t0_us" in df.columns else [sc.CALIB_T0] * len(df)
    rows = [{"clip_id": c, "t0_us": int(t)} for c, t in zip(df["clip_id"], t0_col)]
    rows = rows[args.clip_offset: args.clip_offset + args.num_clips]

    devices = None if args.gpu is None else [int(x) for x in args.gpu.split(",")]
    device = reserve_gpu(args.reserve_gb, devices=devices)
    print(f"using {device}", flush=True)
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")
    ec, tc = model.expert.config, model.vlm.config.text_config
    L, H = ec.num_hidden_layers, ec.num_attention_heads
    G = ec.num_key_value_heads
    vmasks = None
    if not args.no_swap:
        imp = dict(np.load(REPO / "outputs" / args.importance / "importance.npz"))
        vq, vm, _, _, kvonly = build_masks(args.vlm_config, imp, model)
        assert not kvonly
        vmasks = ml.PruneMasks(model.vlm.model.language_model.layers, tc.num_attention_heads,
                               tc.head_dim, tc.intermediate_size, "cuda")
        vmasks.reset()
        print(f"VLM masks for the pruned cache: keep q={vq.mean():.4f} mlp={vm.mean():.4f}",
              flush=True)
    all_l, all_s = list(range(L)), list(range(N_STEPS))
    grid_l = [0, 18, 35] if args.smoke else all_l
    grid_s = [0, 5, 9] if args.smoke else all_s
    head_l = [0, 18, 35] if args.smoke else all_l
    rng = np.random.default_rng(0)  # the same 3D cells for every clip and shard
    cells3d = sorted({(int(l), int(h), int(st)) for l, h, st in zip(
        rng.integers(0, L, args.n_sample3d), rng.integers(0, H, args.n_sample3d),
        rng.integers(0, N_STEPS, args.n_sample3d))})
    if args.smoke:
        cells3d = cells3d[:6]

    meta = {
        "model": "nvidia/Alpamayo-1.5-10B", "model_revision": MODEL_REV,
        "plan": "plans/2026-08-28_cache-use-map.md",
        "manifest": args.manifest, "cache": args.cache, "clip_offset": args.clip_offset,
        "k_marginal": args.k, "k_grid": 1, "seed": args.seed,
        "seed_rule": "sha256(f'{seed}:{clip_id}')[:4] + k",
        "n_steps": N_STEPS, "grid_layers": grid_l, "grid_steps": grid_s, "spans": SPANS,
        "head_layers": head_l, "cells3d": cells3d,
        "swap": None if args.no_swap else {"vlm_config": args.vlm_config,
                                           "importance": args.importance},
        "move": "mean over 64 waypoints of |xy - xy_ref(same seed)| in m; move_end = last waypoint",
        "gpu": torch.cuda.get_device_name(device),
    }
    (out_dir / "config.json").write_text(json.dumps(
        {**meta, "clip_ids": [r["clip_id"] for r in rows]}, indent=2))

    per_clip = {k: [] for k in (
        "clip_ids", "buckets", "nll", "noise_floor", "steps_seen", "ade_ref", "ade_ref1",
        "move_none", "move_all", "move_all_end", "ade_all",
        "move_layer", "move_layer_end", "ade_layer", "move_step", "move_step_end", "ade_step",
        "move_grid", "move_grid_end", "ade_grid", "move_head", "move_head_end", "ade_head",
        "move_3d", "move_3d_end", "ade_3d", "stat_mass_cache", "stat_read_share",
        "nll_pruned", "move_swap", "move_swap_end", "ade_swap", "move_swap_layer",
        "move_swap_layer_end", "ade_swap_layer", "move_swap_all", "move_swap_all_end",
        "ade_swap_all")}
    stats_sum = {k: np.zeros((N_STEPS, L, H)) for k in STAT_KEYS}
    counter = StepCounter(model)

    for ci, r in enumerate(rows):
        t_clip = time.time()
        data = sc.load_cached(sc.path_for(args.cache, r["clip_id"], r["t0_us"]))
        inputs = lib.build_inputs(model, processor, data, "cuda")
        prompt_len = inputs["input_ids"].shape[1]
        spans = lib.compute_spans(model, inputs["input_ids"])
        gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()  # (64, 2)
        base = sc.clip_seed(args.seed, r["clip_id"])
        seeds = [base + k for k in range(args.k)]

        torch.manual_seed(base)
        torch.cuda.manual_seed_all(base)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
            coc_start, coc_end = prompt_len, roll["eos_pos"] + 1
            seq_tf = roll["sequences"][:, :coc_end].clone()
            del roll
            cache, rope, nll = tf_forward(model, seq_tf, inputs, coc_start, coc_end)
            cache_p, nll_p = None, float("nan")
            if vmasks is not None:
                vmasks.set(q=vq, mlp=vm)
                cache_p, rope_p, nll_p = tf_forward(model, seq_tf, inputs, coc_start, coc_end)
                vmasks.reset()
                assert cache_p.get_seq_length() == cache.get_seq_length()
                assert torch.equal(rope_p, rope)
        prefill = cache.get_seq_length()
        sidx = span_index(spans, prompt_len, coc_start, coc_end, prefill, "cuda")
        offset = torch.tensor([prefill], device="cuda")
        prefix_ones = torch.ones(1, prefill, device="cuda", dtype=torch.long)
        blocker = CacheBlocker(model, counter, prefill)

        @torch.no_grad()
        def traj(seed):
            counter.reset()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                action = lib.denoise_with_cache(model, cache, rope, offset, prefix_ones, seed=seed)
                xyz, _ = model.action_space.action_to_traj(
                    action.float(), inputs["ego_history_xyz"][:, -1].float(),
                    inputs["ego_history_rot"][:, -1].float())
            per_clip_steps.append(counter.step + 1)
            return xyz[0, :, :2].float().cpu().numpy()  # (64, 2)

        def measure(layers, steps, use_seeds, heads=None):
            blocker.set(layers, steps, heads)
            mv, mv_end, preds = [], [], []
            for s in use_seeds:
                xy = traj(s)
                d = np.linalg.norm(xy - ref[s], axis=-1)  # (64,)
                mv.append(d.mean())
                mv_end.append(d[-1])
                preds.append(xy)
            ade, _ = el.min_metrics(np.stack(preds), gt_xy)
            return float(np.mean(mv)), float(np.mean(mv_end)), float(ade)

        per_clip_steps = []
        blocker.set([], [])
        ref = {s: traj(s) for s in seeds}
        pairs = [np.linalg.norm(ref[a] - ref[b], axis=-1).mean()
                 for a, b in itertools.combinations(seeds, 2)]
        ade_ref, _ = el.min_metrics(np.stack([ref[s] for s in seeds]), gt_xy)
        ade_ref1, _ = el.min_metrics(ref[seeds[0]][None], gt_xy)

        # Stage A on the unblocked model, one seed
        lib.set_expert_attn_impl(model, "eager")
        col = StepStatsCollector(model, counter, cache, prefill, sidx)
        traj(seeds[0])
        col.remove()
        lib.set_expert_attn_impl(model, "sdpa")
        assert (col.calls == 1).all(), col.calls
        for k in STAT_KEYS:
            stats_sum[k] += col.sums[k]
        per_clip["stat_mass_cache"].append(col.sums["mass_cache"].copy())  # (S, L, H)
        per_clip["stat_read_share"].append(col.sums["read_share"].copy())

        # Stage B
        m_none, _, _ = measure([], [], seeds[:1])
        m_all, m_all_end, a_all = measure(all_l, all_s, seeds)
        lay = np.array([measure([l], all_s, seeds) for l in all_l])  # (L, 3)
        stp = np.array([measure(all_l, [s], seeds) for s in all_s])  # (S, 3)
        grid = np.full((L, N_STEPS, 3), np.nan)
        for l in grid_l:
            for s in grid_s:
                grid[l, s] = measure([l], [s], seeds[:1])
        head = np.full((L, H, 3), np.nan)
        for l in head_l:
            for h in range(H):
                head[l, h] = measure([l], all_s, seeds[:1], heads=[h])
        if args.smoke:  # blocking every head of a layer must equal blocking the layer
            for l in head_l:
                full = measure([l], all_s, seeds[:1], heads=list(range(H)))
                one = measure([l], all_s, seeds[:1])
                print(f"  smoke: layer {l} all-heads-mask {full[0]:.4f} vs layer-mask "
                      f"{one[0]:.4f} | per-head moves "
                      f"{' '.join(f'{x:.2f}' for x in head[l, :, 0])}", flush=True)
        samp = np.array([measure([l], [st], seeds[:1], heads=[h]) for l, h, st in cells3d])
        blocker.remove()

        # Stage C: swap (layer, group) cells of the dense cache for their pruned version
        swap = np.full((L, G, 3), np.nan)
        swap_layer = np.full((L, 3), np.nan)
        swap_all = (np.nan, np.nan, np.nan)
        if cache_p is not None:
            def measure_swap(cells):
                saved = {}
                for l, groups in cells.items():
                    k_d, v_d = lib.cache_layer_kv(cache, l)
                    k_p, v_p = lib.cache_layer_kv(cache_p, l)
                    saved[l] = (k_d, v_d)
                    k_m, v_m = k_d.clone(), v_d.clone()
                    # the dense cache turns float32 once a denoise has appended the expert's
                    # own K/V; the pruned cache is still bf16 -- cast, exactly
                    k_m[:, groups] = k_p[:, groups].to(k_m.dtype)
                    v_m[:, groups] = v_p[:, groups].to(v_m.dtype)
                    lib.set_cache_layer_kv(cache, l, k_m, v_m)
                try:
                    xy = traj(seeds[0])
                finally:
                    for l, (k_d, v_d) in saved.items():
                        lib.set_cache_layer_kv(cache, l, k_d, v_d)
                d = np.linalg.norm(xy - ref[seeds[0]], axis=-1)
                ade, _ = el.min_metrics(xy[None], gt_xy)
                return float(d.mean()), float(d[-1]), float(ade)
            for l in head_l:
                for g in range(G):
                    swap[l, g] = measure_swap({l: [g]})
                swap_layer[l] = measure_swap({l: list(range(G))})
            swap_all = measure_swap({l: list(range(G)) for l in all_l})
            del cache_p

        per_clip["clip_ids"].append(r["clip_id"])
        per_clip["buckets"].append(el.bucket(gt_xy))
        per_clip["nll"].append(nll)
        per_clip["noise_floor"].append(float(np.mean(pairs)))
        per_clip["steps_seen"].append(int(np.max(per_clip_steps)) if len(set(per_clip_steps)) == 1
                                      else -1)
        per_clip["ade_ref"].append(float(ade_ref))
        per_clip["ade_ref1"].append(float(ade_ref1))
        per_clip["move_none"].append(m_none)
        per_clip["move_all"].append(m_all)
        per_clip["move_all_end"].append(m_all_end)
        per_clip["ade_all"].append(a_all)
        per_clip["move_layer"].append(lay[:, 0])
        per_clip["move_layer_end"].append(lay[:, 1])
        per_clip["ade_layer"].append(lay[:, 2])
        per_clip["move_step"].append(stp[:, 0])
        per_clip["move_step_end"].append(stp[:, 1])
        per_clip["ade_step"].append(stp[:, 2])
        per_clip["move_grid"].append(grid[:, :, 0])
        per_clip["move_grid_end"].append(grid[:, :, 1])
        per_clip["ade_grid"].append(grid[:, :, 2])
        per_clip["move_head"].append(head[:, :, 0])
        per_clip["move_head_end"].append(head[:, :, 1])
        per_clip["ade_head"].append(head[:, :, 2])
        per_clip["move_3d"].append(samp[:, 0])
        per_clip["move_3d_end"].append(samp[:, 1])
        per_clip["ade_3d"].append(samp[:, 2])
        per_clip["nll_pruned"].append(nll_p)
        per_clip["move_swap"].append(swap[:, :, 0])
        per_clip["move_swap_end"].append(swap[:, :, 1])
        per_clip["ade_swap"].append(swap[:, :, 2])
        per_clip["move_swap_layer"].append(swap_layer[:, 0])
        per_clip["move_swap_layer_end"].append(swap_layer[:, 1])
        per_clip["ade_swap_layer"].append(swap_layer[:, 2])
        per_clip["move_swap_all"].append(swap_all[0])
        per_clip["move_swap_all_end"].append(swap_all[1])
        per_clip["ade_swap_all"].append(swap_all[2])
        torch.cuda.empty_cache()

        print(f"[{ci + 1}/{len(rows)}] {r['clip_id'][:8]} {per_clip['buckets'][-1]:10s} "
              f"steps={per_clip['steps_seen'][-1]} none={m_none:.2e} all={m_all:.3f} "
              f"floor={pairs and np.mean(pairs):.3f} "
              f"step-marg={' '.join(f'{x:.2f}' for x in stp[:, 0])} "
              f"cache-mass(s0/s9)={col.sums['mass_cache'][0].mean():.3f}/"
              f"{col.sums['mass_cache'][9].mean():.3f} swap-all={swap_all[0]:.3f} "
              f"({time.time() - t_clip:.0f}s)",
              flush=True)
        if (ci + 1) % 5 == 0 or ci + 1 == len(rows):
            save(out_dir, meta, per_clip, stats_sum, ci + 1)

    counter.remove()
    save(out_dir, meta, per_clip, stats_sum, len(rows))
    (out_dir / "summary.txt").write_text(
        f"cache-use map, {len(rows)} clips ({args.manifest} offset {args.clip_offset})\n"
        f"steps seen: {sorted(set(per_clip['steps_seen']))}  none-block move max "
        f"{np.max(per_clip['move_none']):.2e}\n"
        f"all-block move mean {np.mean(per_clip['move_all']):.3f} m, noise floor "
        f"{np.mean(per_clip['noise_floor']):.3f} m\n"
        f"step marginal move: {np.round(np.mean(per_clip['move_step'], 0), 3).tolist()}\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
