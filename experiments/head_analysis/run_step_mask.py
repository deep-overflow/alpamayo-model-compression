"""Step-limited expert masking: does pruning damage depend on WHICH denoising step it hits?

The gradient-free counterpart to run_step_importance. It makes no Taylor assumption at all:
take a fixed expert mask and apply it during exactly one Euler step, leaving the other nine
dense, then read minADE. A flat curve over s means the steps are interchangeable and the
step axis cannot explain why the trajectory Taylor criterion loses to weight magnitude on
this tower; a peaked curve means it can.

Configs (r = --ratio, per-layer uniform over all 36 expert layers):
  baseline            no mask
  full_<crit>         mask at every step -- the ordinary static prune, for scale
  only<s>_<crit>      mask at step s only
  except<s>_traj      mask at every step except s -- the complement, which separates
                      "step s is fragile" from "step s is the only one that matters"

Criteria are the shipped trajectory Taylor score (importance.npz `traj_exp_*`, i.e.
|sum_s dL_s/dg|) and weight magnitude, the pair whose ordering this whole track is about.

Two readouts per config: minADE/minFDE against GT (the task metric) and `dev_k`, the mean
waypoint distance from the unmasked path drawn with the SAME noise seed. A one-step mask
moves minADE by ~1/10 of a full mask, which n=60 cannot resolve; dev is the identical
perturbation measured against its own control, so it resolves the step curve.

Evaluated on indist_500, never on the calibration clips the criterion was measured on.
One rollout + one teacher-forced VLM forward per clip serves every config, because an
expert mask cannot change either.

Usage:
  bash experiments/head_analysis/run_retry_host.sh 240 \
      experiments/head_analysis/run_step_mask.py --gpu 4 --exp-id stepmask_v1
"""

import os

# must precede any CUDA context creation for deterministic cuBLAS reductions
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

import analysis_lib as lib  # noqa: E402
import eval_lib as el  # noqa: E402
import mask_lib as ml  # noqa: E402
import sample_cache as sc  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  also installs the gated-repo hub patch

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MODEL_REV = "7aba8293c09993f2e125c6819df05d7fa3e873ea"


def set_determinism():
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_plans(model, imp, ratio, n_steps, criteria):
    """(name, meta, plan) per config; plan[s] is a (q, mlp) keep-mask pair or None."""
    ec = model.expert.config
    layers = list(range(ec.num_hidden_layers))
    mag_q, mag_mlp = ml.magnitude_scores(model.expert.layers, ec.num_attention_heads,
                                         ec.head_dim, ec.intermediate_size)
    scores = {"traj": (imp["traj_exp_q"], imp["traj_exp_mlp"]), "magnitude": (mag_q, mag_mlp)}
    masks = {}
    for crit in criteria:
        q, m = scores[crit]
        masks[crit] = (torch.as_tensor(ml.select_mask(q, ratio, layers), dtype=torch.float32,
                                       device="cuda"),
                       torch.as_tensor(ml.select_mask(m, ratio, layers), dtype=torch.float32,
                                       device="cuda"))

    cfgs = [("baseline", {"kind": "baseline"}, [None] * n_steps)]
    for crit in criteria:
        cfgs.append((f"full_{crit}", {"kind": "full", "criterion": crit, "ratio": ratio},
                     [masks[crit]] * n_steps))
        for s in range(n_steps):
            plan = [None] * n_steps
            plan[s] = masks[crit]
            cfgs.append((f"only{s}_{crit}",
                         {"kind": "only", "step": s, "criterion": crit, "ratio": ratio}, plan))
    crit = criteria[0]
    for s in range(n_steps):
        plan = [masks[crit]] * n_steps
        plan[s] = None
        cfgs.append((f"except{s}_{crit}",
                     {"kind": "except", "step": s, "criterion": crit, "ratio": ratio}, plan))
    return cfgs


@torch.no_grad()
def denoise_step_masked(model, cache, rope_deltas, offset, prefix_mask, seed, masks, plan):
    """denoise_with_cache, but the expert mask is swapped between Euler steps.

    The official sampler is reused rather than reimplemented, so the t grid and the Euler
    update stay exactly what inference does; the step index comes from the call order and
    is checked against t so a sampler change cannot silently mis-assign a mask.
    """
    device = prefix_mask.device
    n_tok = model.action_space.get_action_space_dims()[0]  # 64
    prefill = cache.get_seq_length()
    position_ids, attention_mask = model._build_expert_pos_ids_and_attn_mask(
        offset=offset, rope_deltas=rope_deltas, kv_cache_seq_len=prefill,
        n_diffusion_tokens=n_tok, b_star=1, device=device, prefix_mask=prefix_mask,
    )
    forward_kwargs = {}
    if model.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False
    n_steps = len(plan)
    calls = [0]

    def step_fn(x, t):
        s = calls[0]
        assert abs(float(t.flatten()[0]) - s / n_steps) < 1e-6, "step index vs t mismatch"
        entry = plan[s]
        if entry is None:
            masks.reset()
        else:
            masks.q_mask.copy_(entry[0])
            masks.mlp_mask.copy_(entry[1])
        calls[0] += 1
        embeds = model.action_in_proj(x, t)  # (1, 64, 2048)
        if embeds.dim() == 2:
            embeds = embeds.view(1, n_tok, -1)
        out = model.expert(
            inputs_embeds=embeds, position_ids=position_ids, past_key_values=cache,
            attention_mask=attention_mask, use_cache=True, **forward_kwargs,
        )
        cache.crop(prefill)
        last = out.last_hidden_state[:, -n_tok:]  # (1, 64, 2048)
        return model.action_out_proj(last).view(-1, *model.action_space.get_action_space_dims())

    torch.cuda.manual_seed_all(seed)
    action = model.diffusion.sample(batch_size=1, step_fn=step_fn, device=device,
                                    inference_step=n_steps, return_all_steps=False)
    assert calls[0] == n_steps, f"sampler took {calls[0]} steps, plan has {n_steps}"
    masks.reset()
    return action  # (1, 64, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", type=str, required=True)
    ap.add_argument("--num-clips", type=int, default=60)
    ap.add_argument("--manifest", default="indist_500")
    ap.add_argument("--cache", default="eval")
    ap.add_argument("--importance", default="importance_v2")
    ap.add_argument("--ratio", type=float, default=0.40)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--criteria", nargs="+", default=["traj", "magnitude"])
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--gpu", type=str, default=None)
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(REPO / "outputs" / "eval_sets" / f"{args.manifest}.parquet")
    man = [(r.clip_id, int(r.t0_us)) for r in df.itertuples()][: args.num_clips]

    set_determinism()
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

    ec = model.expert.config
    masks = ml.PruneMasks(model.expert.layers, ec.num_attention_heads, ec.head_dim,
                          ec.intermediate_size, "cuda")
    imp = dict(np.load(REPO / "outputs" / args.importance / "importance.npz"))
    cfgs = build_plans(model, imp, args.ratio, args.n_steps, args.criteria)
    print(f"{len(cfgs)} configs x {len(man)} clips x K={args.k}", flush=True)

    (out_dir / "config.json").write_text(json.dumps({
        "model": "nvidia/Alpamayo-1.5-10B", "model_revision": MODEL_REV, "tower": "expert",
        "purpose": "is pruning damage step-dependent? (gradient-free)",
        "manifest": args.manifest, "cache": args.cache, "n_clips": len(man),
        "clip_ids": [c for c, _ in man], "ratio": args.ratio, "k": args.k,
        "n_steps": args.n_steps, "seed": args.seed,
        "seed_rule": "sha256(f'{seed}:{clip_id}')[:4], +k per sample",
        "importance_from": args.importance, "criteria": args.criteria,
        "configs": [{"name": n, **m} for n, m, _ in cfgs],
        "gpu": torch.cuda.get_device_name(device),
    }, indent=2))

    rows_path = out_dir / "rows.json"
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    done = {r["clip_id"] for r in rows}

    for ci, (clip_id, t0_us) in enumerate(man):
        if clip_id in done:
            continue
        t_start = time.time()
        data = sc.load_cached(sc.path_for(args.cache, clip_id, t0_us))
        inputs = lib.build_inputs(model, processor, data, "cuda")
        prompt_len = inputs["input_ids"].shape[1]
        gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()  # (64, 2)
        base = sc.clip_seed(args.seed, clip_id)
        seeds = [base + k for k in range(args.k)]

        masks.reset()
        seed_all(base)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
            coc_end = roll["eos_pos"] + 1
            seq_tf = roll["sequences"][:, :coc_end].clone()
            del roll
            out = model.vlm.model(
                input_ids=seq_tf, attention_mask=torch.ones_like(seq_tf),
                pixel_values=inputs["tokenized_data"]["pixel_values"],
                image_grid_thw=inputs["tokenized_data"]["image_grid_thw"], use_cache=True,
            )
        cache, rope_deltas = out.past_key_values, out.rope_deltas
        prefill = cache.get_seq_length()
        offset = torch.tensor([prefill], device="cuda")
        prefix_mask = torch.ones(1, prefill, device="cuda", dtype=torch.long)
        del out

        rec = {"clip_id": clip_id, "bucket": el.bucket(gt_xy), "seed": base,
               "coc_len": int(coc_end - prompt_len), "configs": {}}
        base_pred = None
        for name, _, plan in cfgs:
            preds = []
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for s in seeds:
                    action = denoise_step_masked(model, cache, rope_deltas, offset,
                                                 prefix_mask, s, masks, plan)
                    pred_xyz, _ = model.action_space.action_to_traj(
                        action.float(), inputs["ego_history_xyz"][:, -1].float(),
                        inputs["ego_history_rot"][:, -1].float(),
                    )
                    preds.append(pred_xyz[0, :, :2].cpu().numpy())
            pred_k = np.stack(preds)  # (K, 64, 2)
            ade, fde = el.ade_fde(pred_k, gt_xy)
            if base_pred is None:                      # baseline is the first config
                base_pred = pred_k
            # deviation from the unmasked path at the SAME noise seed. minADE vs GT is the
            # task metric but its per-clip spread swamps a one-step perturbation; this is
            # the same perturbation measured against its own control, so the step curve is
            # resolvable at n=60 instead of needing thousands of clips.
            dev = np.linalg.norm(pred_k - base_pred, axis=2).mean(1)  # (K,)
            rec["configs"][name] = {"ade_k": [round(float(x), 6) for x in ade],
                                    "fde_k": [round(float(x), 6) for x in fde],
                                    "dev_k": [round(float(x), 6) for x in dev]}
        rows.append(rec)
        del cache, inputs

        b = rec["configs"]["baseline"]["ade_k"]
        print(f"[{ci + 1}/{len(man)}] {clip_id} {rec['bucket']:10s} "
              f"baseMinADE={min(b):.3f} ({time.time() - t_start:.0f}s)", flush=True)
        rows_path.write_text(json.dumps(rows))
        if (ci + 1) % 10 == 0 or ci + 1 == len(man):
            save(out_dir, cfgs, rows)
    save(out_dir, cfgs, rows)
    print("saved ->", out_dir, flush=True)


def save(out_dir, cfgs, rows):
    (out_dir / "metrics.json").write_text(json.dumps({
        "n_clips": len(rows),
        "configs": [n for n, _, _ in cfgs],
        "meta": {n: m for n, m, _ in cfgs},
        "rows": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
