"""Per-denoising-step VLM trajectory importance -- the half the shipped configs actually use.

`run_importance.py` lets the expert accumulate ten steps of gradient onto the cache leaves
and then seeds ONE VLM backward with the sum, so the shipped `traj_vlm_*` is
|sum_s dL_s/dg|. The step decomposition of the expert tower showed that summing a step axis
whose mass varies 7.7x costs real accuracy; on the VLM that ratio is 60x (measured), and
this score is one half of every shipped `*_u40_v2` criterion via
max(rank traj, rank coc). So it is worth measuring properly.

Backprop is linear in the seed, so per-step VLM scores need one VLM backward per step:
+5.9 s/clip measured, ~25 min for calib_100. The loop is interleaved rather than staged --
expert step s, then immediately the VLM backward for step s, then discard that step's cache
gradient -- because holding all ten cache-gradient sets peaked at 45.0 GB on a 47.4 GB card.

Two scores come out of the same pass:
  exact       per-step signed dL_s/dg_vlm, from ten backwards
  seedznorm   one extra backward seeded with sum_s (dL_s/dcache)/mass_s, the cheap
              approximation of a step-normalised score (mass_s comes free from the expert
              side; it correlated 0.9861 with the VLM gate mass in the probe). If it
              tracks the exact znorm, later VLM re-measurements cost one backward, not ten.

The CoC objective is deliberately NOT recomputed: it has no step axis (a single NLL
backward), so the drop-in importance files copy `coc_*` from the reference run and only
`traj_vlm_*` / `traj_kv_*` are replaced. That keeps the comparison one-factor.

Usage:
  bash experiments/head_analysis/run_retry_host.sh 60 \
      experiments/head_analysis/run_step_importance_vlm.py --gpu 6 \
      --num-clips 100 --exp-id importance_stepvlm_v1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

import analysis_lib as lib  # noqa: E402
import prune_lib as pl  # noqa: E402
import sample_cache as sc  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  also installs the gated-repo hub patch

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MODEL_REV = "7aba8293c09993f2e125c6819df05d7fa3e873ea"


def process_clip(model, processor, data, args, seed):
    """One clip -> per-step signed VLM gate grads, per-step KV scores, seedznorm scores."""
    inputs = lib.build_inputs(model, processor, data, "cuda")
    prompt_len = inputs["input_ids"].shape[1]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
    coc_end = roll["eos_pos"] + 1
    seq_tf = roll["sequences"][:, :coc_end]  # (1, T) prompt + generated CoC
    del roll

    x1 = lib.gt_actions(model, data, "cuda").to(torch.float32)  # (1, 64, 2)

    tc = model.vlm.config.text_config
    vlm_gates = pl.UnitGates(
        model.vlm.model.language_model.layers, tc.num_attention_heads, tc.head_dim,
        tc.intermediate_size, "cuda", torch.float32,
    )

    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, cache, rope_deltas = pl.vlm_forward_with_grad(
            model, seq_tf, inputs["tokenized_data"], use_cache=True
        )
    n_vlm = len(model.vlm.model.language_model.layers)
    cache_t = pl.retain_cache_grads(cache, n_vlm)
    prefill = cache.get_seq_length()

    leaves = []
    for i in range(n_vlm):
        k, v = lib.cache_layer_kv(cache, i)
        leaves.append((k.detach().requires_grad_(True), v.detach().requires_grad_(True)))

    n_kv = leaves[0][0].shape[1]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q_steps, mlp_steps, kv_k_steps, kv_v_steps, losses, masses = [], [], [], [], [], []
    # running sum_s (dL_s/dcache)/mass_s -- one extra cache-sized buffer, not ten
    acc = [(torch.zeros_like(k), torch.zeros_like(v)) for k, v in leaves]

    vlm_gates.zero_grads()
    for s in range(args.fm_steps):
        noise = torch.randn(x1.shape, generator=gen).to("cuda")  # (1, 64, 2)
        loss = pl.expert_step_cache_grads(model, cache, rope_deltas, x1, s, args.fm_steps,
                                          prefill, leaves, noise)
        losses.append(loss)

        # per-step KV score, same form as kv_group_scores, straight off the leaves
        kk = np.zeros((n_vlm, n_kv))
        vv = np.zeros((n_vlm, n_kv))
        mass = 0.0
        with torch.no_grad():
            for i, (k, v) in enumerate(leaves):
                if k.grad is not None:
                    kk[i] = (k * k.grad).abs().sum((0, 2, 3)).float().cpu().numpy()
                    mass += float(k.grad.abs().sum())
                if v.grad is not None:
                    vv[i] = (v * v.grad).abs().sum((0, 2, 3)).float().cpu().numpy()
                    mass += float(v.grad.abs().sum())
        kv_k_steps.append(kk)
        kv_v_steps.append(vv)
        masses.append(mass)

        seed_grads = [(k.grad, v.grad) for k, v in leaves]
        pl.vlm_backward_from_cache(cache_t, seed_grads, retain=True)
        q_steps.append(vlm_gates.q_signed())
        mlp_steps.append(vlm_gates.mlp_signed())
        vlm_gates.zero_grads()

        with torch.no_grad():
            w = 1.0 / max(mass, 1e-30)
            for (ak, av), (k, v) in zip(acc, leaves):
                if k.grad is not None:
                    ak.add_(k.grad, alpha=w)
                if v.grad is not None:
                    av.add_(v.grad, alpha=w)
        for k, v in leaves:
            k.grad = None
            v.grad = None
        for k, v in cache_t:
            k.grad = None
            v.grad = None

    # cheap approximation: one backward with the mass-normalised seed
    pl.vlm_backward_from_cache(cache_t, acc, retain=False)
    seedz_q = np.abs(vlm_gates.q_signed())
    seedz_mlp = np.abs(vlm_gates.mlp_signed())

    peak = torch.cuda.max_memory_allocated() / 1024**3
    vlm_gates.remove()
    out = {
        "q": np.stack(q_steps), "mlp": np.stack(mlp_steps),
        "kv_k": np.stack(kv_k_steps), "kv_v": np.stack(kv_v_steps),
        "seedz_q": seedz_q, "seedz_mlp": seedz_mlp,
        "mass": np.array(masses),
    }
    del cache, cache_t, leaves, acc, inputs
    return out, {"coc_len": int(coc_end - prompt_len), "fm_loss": float(np.mean(losses)),
                 "losses": [round(x, 6) for x in losses], "peak_gb": round(peak, 2),
                 "mass": [float(x) for x in masses]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-clips", type=int, default=100)
    ap.add_argument("--exp-id", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--calib-manifest", default="calib_100")
    ap.add_argument("--cache", default="calib")
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--fm-steps", type=int, default=10)
    ap.add_argument("--reserve-gb", type=float, default=44.0)
    ap.add_argument("--gpu", type=str, default=None)
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    calib = sc.calib_samples(REPO, args.calib_manifest)[: args.num_clips]

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
    # the gates sit downstream of k/v, so without this the early layers' cache tensors
    # carry no autograd graph at all
    model.vlm.enable_input_require_grads()

    tc = model.vlm.config.text_config
    S, L = args.fm_steps, tc.num_hidden_layers
    shapes = {"q": (S, L, tc.num_attention_heads), "mlp": (S, L, tc.intermediate_size),
              "kv_k": (S, L, tc.num_key_value_heads), "kv_v": (S, L, tc.num_key_value_heads)}
    acc = {f"{k}_abs_step": np.zeros(s) for k, s in shapes.items()}
    acc.update({f"{k}_shipped": np.zeros(s[1:]) for k, s in shapes.items()})
    acc.update({"seedz_q": np.zeros(shapes["q"][1:]),
                "seedz_mlp": np.zeros(shapes["mlp"][1:])})
    # fp32 throughout: these gradients sit near 1e-7 and fp16 storage silently underflowed
    # 75% of an earlier MLP array to zero (see plans/2026-08-21, section 3A)
    per_clip = {"q_abs_step": [], "q_shipped": [], "mlp_abs_step": [], "mlp_shipped": []}

    (out_dir / "config.json").write_text(json.dumps({
        "model": "nvidia/Alpamayo-1.5-10B", "model_revision": MODEL_REV,
        "purpose": "per-denoising-step VLM trajectory importance (the shipped criterion's half)",
        "objective": "flow-matching MSE vs GT action, per diffusion step, seeded into the VLM",
        "num_clips": len(calib), "clip_ids": [c for c, _ in calib], "seed": args.seed,
        "seed_rule": "sha256(f'{seed}:{clip_id}')[:4]",
        "calib_manifest": args.calib_manifest, "cache": args.cache,
        "fm_steps": args.fm_steps, "max_gen": args.max_gen,
        "coc_note": "not recomputed here; the drop-in files copy coc_* from the reference",
        "gpu": torch.cuda.get_device_name(device),
        "shapes": {k: list(v) for k, v in shapes.items()},
    }, indent=2))

    records = []
    for ci, (clip_id, clip_t0) in enumerate(calib):
        t0 = time.time()
        data = sc.load_cached(sc.path_for(args.cache, clip_id, clip_t0))
        torch.cuda.reset_peak_memory_stats()
        g, rec = process_clip(model, processor, data, args,
                             sc.clip_seed(args.seed, clip_id))
        for k in shapes:
            acc[f"{k}_abs_step"] += np.abs(g[k])
            acc[f"{k}_shipped"] += np.abs(g[k].sum(0))
        acc["seedz_q"] += g["seedz_q"]
        acc["seedz_mlp"] += g["seedz_mlp"]
        per_clip["q_abs_step"].append(np.abs(g["q"]).astype(np.float32))
        per_clip["q_shipped"].append(np.abs(g["q"].sum(0)).astype(np.float32))
        per_clip["mlp_abs_step"].append(np.abs(g["mlp"]).astype(np.float32))
        per_clip["mlp_shipped"].append(np.abs(g["mlp"].sum(0)).astype(np.float32))
        rec["clip_id"] = clip_id
        records.append(rec)
        del g
        print(f"[{ci + 1}/{len(calib)}] {clip_id} coc={rec['coc_len']} "
              f"fm={rec['fm_loss']:.4f} peak={rec['peak_gb']:.1f}GB "
              f"({time.time() - t0:.0f}s)", flush=True)
        if (ci + 1) % 10 == 0 or ci + 1 == len(calib):
            save(out_dir, acc, per_clip, records, ci + 1)
    save(out_dir, acc, per_clip, records, len(records))
    print("saved ->", out_dir, flush=True)


def save(out_dir, acc, per_clip, records, n):
    np.savez(out_dir / "step_importance_vlm.npz",
             **{k: v / max(n, 1) for k, v in acc.items()})
    np.savez(out_dir / "step_importance_vlm_perclip.npz",
             **{k: np.stack(v) for k, v in per_clip.items() if v})
    (out_dir / "metrics.json").write_text(json.dumps({"n_clips": n, "per_clip": records},
                                                     indent=2))


if __name__ == "__main__":
    main()
