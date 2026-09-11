"""G1: does the self-anchored FM loss survive tying the noise to the sample?

`I_traj` anchors the flow-matching loss on the GT action, while `I_CoC` anchors on the
model's own rollout. Replacing the GT with the model's own denoised sample removes that
asymmetry -- but there are two ways to pair the sample with a noise, and they are not
equivalent (plans/2026-09-10_self-anchored-traj-importance.md):

  gt    x_t = (1-t) eps + t x1_gt        u = x1_gt - eps     (shipped)
  A     x_t = (1-t) eps0 + t xhat(eps0)  u = xhat - eps0     tied: the model's OWN coupling
  B     x_t = (1-t) eps + t xhat(eps0)   u = xhat - eps      independent, fresh eps per step

A is the only pairing the model's flow actually produces -- the ODE maps eps0 to xhat
deterministically. The worry is that a well-rectified flow makes the instantaneous field
v(x_t, t) nearly equal to the chord xhat - eps0, collapsing the residual so that
dL/dg = 2 (approx 0) dv/dg ranks units by alignment with a tiny residual rather than by
contribution -- the same shape as the identity-refit trap in `cache-preservation-is-a-chain`.

Whether that happens is an unmeasured assumption about how straight Alpamayo's flow is.
This probe measures it. No gradients are taken: only the loss magnitude at g=1 decides,
against the shipped baseline (importance_v2 calib_100: fm_loss median 0.1933, mean 0.2564).

Pre-registered reading:
  A median >= 0.019 (a tenth of shipped)  -> adopt A, the model's own coupling
  A collapses below that                  -> adopt B
  ambiguous                               -> build both, let G2/G3 decide

Also reports, as a by-product this repo does not have: the straightness of the learned
flow, as ||v(x_t,t) - chord|| relative to ||chord||, and the ADE of a single sampled
trajectory against GT (which must land near the 1.76 m of the 500-clip baseline draw
distribution, or the sampler is being driven wrong).

Usage:
  bash experiments/head_analysis/run_retry_host.sh 3 \
      experiments/head_analysis/run_selftraj_probe.py --gpu 0 --num-clips 20
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

import analysis_lib as lib
import prune_lib as pl
import sample_cache as sc
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from expert_per_clip import reserve_gpu

REPO = Path(__file__).resolve().parents[2]
SELF_XOR = 0x5E1F  # keeps the sampling noise off the loss generator's stream


def expert_field(model, cache, position_ids, attention_mask, forward_kwargs, prefill,
                 n_tok, x, t_val):
    """One expert evaluation of the vector field at (x, t). Returns (1, 64, 2)."""
    t = torch.full((1, 1, 1), t_val, device=x.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        embeds = model.action_in_proj(x.to(torch.bfloat16), t)
        if embeds.dim() == 2:
            embeds = embeds.view(1, n_tok, -1)
        out = model.expert(
            inputs_embeds=embeds, position_ids=position_ids, past_key_values=cache,
            attention_mask=attention_mask, use_cache=True, **forward_kwargs,
        )
        cache.crop(prefill)
        return model.action_out_proj(out.last_hidden_state[:, -n_tok:]).float()


def probe_clip(model, processor, data, args, seed):
    inputs = lib.build_inputs(model, processor, data, "cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
        seq_tf = roll["sequences"][:, : roll["eos_pos"] + 1]
        del roll
    # the cache is built exactly as run_importance does, minus the autograd graph
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        _, cache, rope_deltas = pl.vlm_forward_with_grad(
            model, seq_tf, inputs["tokenized_data"], use_cache=True)
    prefill = cache.get_seq_length()
    n_tok = model.action_space.get_action_space_dims()[0]  # 64
    dims = model.action_space.get_action_space_dims()      # (64, 2)

    offset = torch.tensor([prefill], device="cuda")
    prefix_mask = torch.ones(1, prefill, device="cuda", dtype=torch.long)
    position_ids, attention_mask = model._build_expert_pos_ids_and_attn_mask(
        offset=offset, rope_deltas=rope_deltas, kv_cache_seq_len=prefill,
        n_diffusion_tokens=n_tok, b_star=1, device="cuda", prefix_mask=prefix_mask)
    fk = {"is_causal": False} if model.config.expert_non_causal_attention else {}

    # ---- the model's own sample, from a noise we keep ----
    gen0 = torch.Generator(device="cpu").manual_seed(seed ^ SELF_XOR)
    eps0 = torch.randn(1, *dims, generator=gen0).to("cuda")  # (1, 64, 2)
    dt = 1.0 / args.fm_steps
    x = eps0.clone()
    with torch.no_grad():
        for s in range(args.fm_steps):
            x = x + dt * expert_field(model, cache, position_ids, attention_mask, fk,
                                      prefill, n_tok, x, s * dt)
    xhat = x  # (1, 64, 2)
    assert cache.get_seq_length() == prefill, "the Euler loop left the cache cropped wrong"
    assert xhat.shape == (1, *dims), xhat.shape

    x1_gt = lib.gt_actions(model, data, "cuda").float()  # (1, 64, 2)

    # sanity: what does this sampled trajectory score against GT?
    pred_xyz, _ = model.action_space.action_to_traj(
        xhat, inputs["ego_history_xyz"][:, -1].float(),
        inputs["ego_history_rot"][:, -1].float())
    gt_xy = data["ego_future_xyz"][0, 0, :, :2].to("cuda").float()
    ade_sample = float(torch.norm(pred_xyz[0, :, :2] - gt_xy, dim=-1).mean())

    # ---- the three losses at g=1, on the same clip and the same xhat ----
    gen = torch.Generator(device="cpu").manual_seed(seed)
    losses = {"gt": [], "A": [], "B": []}
    straight = []
    with torch.no_grad():
        for s in range(args.fm_steps):
            t_val = (s + 0.5) / args.fm_steps
            eps = torch.randn(x1_gt.shape, generator=gen).to("cuda")

            for tag, x1, noise in (("gt", x1_gt, eps), ("A", xhat, eps0), ("B", xhat, eps)):
                x_t = (1.0 - t_val) * noise + t_val * x1
                v_target = x1 - noise
                pred = expert_field(model, cache, position_ids, attention_mask, fk,
                                    prefill, n_tok, x_t, t_val)
                losses[tag].append(float(F.mse_loss(pred, v_target)))
                if tag == "A":
                    chord = v_target
                    straight.append(float(torch.norm(pred - chord) / torch.norm(chord)))

    del cache
    return {"fm_gt": float(np.mean(losses["gt"])),
            "fm_A": float(np.mean(losses["A"])),
            "fm_B": float(np.mean(losses["B"])),
            "fm_gt_steps": losses["gt"], "fm_A_steps": losses["A"],
            "fm_B_steps": losses["B"],
            "straightness": float(np.mean(straight)),
            "ade_sample_vs_gt": ade_sample,
            "coc_len": int(seq_tf.shape[1]), "prefill": prefill}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-clips", type=int, default=20)
    ap.add_argument("--exp-id", type=str, default="selftraj_probe")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--calib-manifest", default="calib_100")
    ap.add_argument("--cache", default="calib")
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--fm-steps", type=int, default=10)
    ap.add_argument("--reserve-gb", type=float, default=40.0)
    ap.add_argument("--gpu", type=str, default=None)
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    calib = sc.calib_samples(REPO, args.calib_manifest)[: args.num_clips]

    devices = None if args.gpu is None else [int(x) for x in args.gpu.split(",")]
    device = reserve_gpu(args.reserve_gb, devices=devices)
    print(f"using {device}", flush=True)

    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", revision="7aba8293c09993f2e125c6819df05d7fa3e873ea",
        dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")

    (out_dir / "config.json").write_text(json.dumps({
        "model": "nvidia/Alpamayo-1.5-10B",
        "purpose": "G1: FM loss magnitude at g=1 for GT / self-tied (A) / self-independent (B)",
        "couplings": {"gt": "x_t=(1-t)eps+t x1_gt, u=x1_gt-eps",
                      "A": "x_t=(1-t)eps0+t xhat(eps0), u=xhat-eps0",
                      "B": "x_t=(1-t)eps+t xhat(eps0), u=xhat-eps"},
        "baseline_importance_v2": {"fm_loss_median": 0.1933, "fm_loss_mean": 0.2564},
        "adopt_A_if": "A median >= 0.019",
        "num_clips": len(calib), "clip_ids": [c for c, _ in calib], "seed": args.seed,
        "self_seed_xor": SELF_XOR, "fm_steps": args.fm_steps,
        "model_revision": "7aba8293c09993f2e125c6819df05d7fa3e873ea",
        "gpu": torch.cuda.get_device_name(device),
    }, indent=2))

    recs = []
    for ci, (clip_id, clip_t0) in enumerate(calib):
        t0 = time.time()
        data = sc.load_cached(sc.path_for(args.cache, clip_id, clip_t0))
        r = probe_clip(model, processor, data, args, sc.clip_seed(args.seed, clip_id))
        r["clip_id"] = clip_id
        recs.append(r)
        print(f"[{ci + 1}/{len(calib)}] {clip_id} gt={r['fm_gt']:.4f} A={r['fm_A']:.4f} "
              f"B={r['fm_B']:.4f} straight={r['straightness']:.3f} "
              f"ade={r['ade_sample_vs_gt']:.2f} ({time.time() - t0:.0f}s)", flush=True)
        (out_dir / "metrics.json").write_text(json.dumps({"per_clip": recs}, indent=2))

    g = np.array([r["fm_gt"] for r in recs])
    a = np.array([r["fm_A"] for r in recs])
    b = np.array([r["fm_B"] for r in recs])
    st = np.array([r["straightness"] for r in recs])
    ade = np.array([r["ade_sample_vs_gt"] for r in recs])
    verdict = ("A" if np.median(a) >= 0.019 else "B")
    lines = [
        f"G1 coupling probe, {len(recs)} calib clips, {args.fm_steps} FM steps",
        "",
        f"{'':22s} {'median':>9s} {'mean':>9s} {'min':>9s} {'max':>9s}",
        f"{'gt (shipped)':22s} {np.median(g):9.4f} {g.mean():9.4f} {g.min():9.4f} {g.max():9.4f}",
        f"{'A (tied eps0)':22s} {np.median(a):9.4f} {a.mean():9.4f} {a.min():9.4f} {a.max():9.4f}",
        f"{'B (independent eps)':22s} {np.median(b):9.4f} {b.mean():9.4f} {b.min():9.4f} {b.max():9.4f}",
        "",
        f"A / gt  median ratio : {np.median(a) / np.median(g):.4f}",
        f"B / gt  median ratio : {np.median(b) / np.median(g):.4f}",
        "adopt-A threshold    : A median >= 0.019   (a tenth of importance_v2's 0.1933)",
        f"VERDICT              : adopt {verdict}",
        "",
        (f"flow straightness ||v - chord|| / ||chord|| : median {np.median(st):.3f} "
         f"mean {st.mean():.3f}  (0 = perfectly rectified)"),
        (f"sampled-trajectory ADE vs GT               : median {np.median(ade):.3f} m "
         f"mean {ade.mean():.3f} m  (500-clip single-draw reference: mean 1.76 m)"),
    ]
    txt = "\n".join(lines)
    print("\n" + txt)
    (out_dir / "summary.txt").write_text(txt + "\n")
    print("saved ->", out_dir, flush=True)


if __name__ == "__main__":
    main()
