"""Phase A + B head-pruning analysis for Alpamayo 1.5.

Per clip:
  1. rollout      — sampled CoC generation + expert denoising (expert stats hooks active)
  2. stats pass   — teacher-forced VLM forward with eager attention (Phase A VLM stats)
  3. taylor pass1 — teacher-forced VLM forward + CoC NLL backward (VLM Q / KV-v gates)
  4. taylor pass2 — no-grad cache build, cache k/v as leaves, expert FM MSE backward
                    (expert Q gates + cache Taylor per VLM KV head)

Usage: python run_analysis.py --num-clips 50 --exp-id head_analysis_v1
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

# The account's HF token is not authorized for the gated nvidia/Cosmos-Reason2-8B repo,
# but all files the model needs from it are already in the local HF cache. Force
# cache-only resolution for that repo while keeping the network for dataset streaming.
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
CLIP_IDS_PARQUET = REPO / "notebooks" / "clip_ids.parquet"


def teacher_forced_vlm_forward(model, seq_tf, tokenized_data, use_cache=False):
    """Forward the full (prompt + CoC) sequence through the VLM base model.

    Returns (last_hidden (1, T, 4096), past_key_values or None, rope_deltas (1, 1)).
    """
    attention_mask = torch.ones_like(seq_tf)
    out = model.vlm.model(
        input_ids=seq_tf,
        attention_mask=attention_mask,
        pixel_values=tokenized_data["pixel_values"],
        image_grid_thw=tokenized_data["image_grid_thw"],
        use_cache=use_cache,
    )
    return out.last_hidden_state, out.past_key_values, out.rope_deltas


def coc_nll(model, hidden, seq_tf, coc_start, coc_end):
    """CoC NLL over generated positions [coc_start, coc_end)."""
    # hidden at p predicts token at p+1 -> slice [coc_start-1, coc_end-1)
    h = hidden[:, coc_start - 1 : coc_end - 1]  # (1, Tc, 4096)
    logits = model.vlm.lm_head(h).float()  # (1, Tc, V)
    targets = seq_tf[:, coc_start:coc_end]  # (1, Tc)
    return F.cross_entropy(logits[0], targets[0])


def fm_taylor_pass(model, seq_tf, tokenized_data, x1, expert_gates, fm_steps, seed):
    """Pass 2: FM MSE backward through expert; cache k/v as leaf tensors.

    Returns (fm_loss_mean, cache_k_taylor (L, KV), cache_v_taylor (L, KV)).
    """
    device = seq_tf.device
    n_layers = len(model.expert.layers)
    n_kv = model.expert.config.num_key_value_heads
    head_dim = model.expert.config.head_dim

    with torch.no_grad():
        _, cache, rope_deltas = teacher_forced_vlm_forward(
            model, seq_tf, tokenized_data, use_cache=True
        )
    prefill = cache.get_seq_length()
    # promote cache tensors to autograd leaves
    leaves = []
    for i in range(n_layers):
        k, v = lib.cache_layer_kv(cache, i)
        k = k.detach().requires_grad_(True)  # (1, KV, T, D)
        v = v.detach().requires_grad_(True)
        lib.set_cache_layer_kv(cache, i, k, v)
        leaves.append((k, v))

    offset = torch.tensor([prefill], device=device)
    prefix_mask = torch.ones(1, prefill, device=device, dtype=torch.long)
    n_tok = model.action_space.get_action_space_dims()[0]  # 64
    position_ids, attention_mask = model._build_expert_pos_ids_and_attn_mask(
        offset=offset,
        rope_deltas=rope_deltas,
        kv_cache_seq_len=prefill,
        n_diffusion_tokens=n_tok,
        b_star=1,
        device=device,
        prefix_mask=prefix_mask,
    )
    forward_kwargs = {}
    if model.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False

    gen = torch.Generator(device="cpu").manual_seed(seed)
    x1 = x1.to(device)  # (1, 64, 2)
    losses = []
    for s in range(fm_steps):
        t_val = (s + 0.5) / fm_steps
        noise = torch.randn(x1.shape, generator=gen).to(device)  # (1, 64, 2)
        x_t = (1.0 - t_val) * noise + t_val * x1
        v_target = x1 - noise  # (1, 64, 2)
        t = torch.full((1, 1, 1), t_val, device=device)
        # restore leaf tensors: update()+crop() leave graph-attached slices from the
        # previous step in the cache, whose graph is freed after backward()
        for i, (k, v) in enumerate(leaves):
            lib.set_cache_layer_kv(cache, i, k, v)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            embeds = model.action_in_proj(x_t.to(torch.bfloat16), t)  # (1, 64, 2048)
            if embeds.dim() == 2:
                embeds = embeds.view(1, n_tok, -1)
            out = model.expert(
                inputs_embeds=embeds,
                position_ids=position_ids,
                past_key_values=cache,
                attention_mask=attention_mask,
                use_cache=True,
                **forward_kwargs,
            )
            cache.crop(prefill)
            pred = model.action_out_proj(out.last_hidden_state[:, -n_tok:])  # (1, 64, 2)
        loss = F.mse_loss(pred.float(), v_target)
        loss.backward()
        losses.append(loss.item())

    k_taylor = np.zeros((n_layers, n_kv))
    v_taylor = np.zeros((n_layers, n_kv))
    with torch.no_grad():
        for i, (k, v) in enumerate(leaves):
            if k.grad is not None:
                k_taylor[i] = (k * k.grad).abs().sum((0, 2, 3)).float().cpu().numpy()
            if v.grad is not None:
                v_taylor[i] = (v * v.grad).abs().sum((0, 2, 3)).float().cpu().numpy()
    del cache, leaves
    return float(np.mean(losses)), k_taylor, v_taylor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-clips", type=int, default=50)
    ap.add_argument("--exp-id", type=str, default="head_analysis_v1")
    ap.add_argument("--fm-steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--smoke", action="store_true", help="1 clip, verbose checks")
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_df = pd.read_parquet(CLIP_IDS_PARQUET)
    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(clip_df))
    calib_ids = clip_df.iloc[order[: args.num_clips]]["clip_id"].tolist()
    heldout_ids = clip_df.iloc[order[args.num_clips : args.num_clips + 12]]["clip_id"].tolist()
    if args.smoke:
        calib_ids = calib_ids[:1]

    config = {
        "model": "nvidia/Alpamayo-1.5-10B",
        "num_clips": len(calib_ids),
        "calib_clip_ids": calib_ids,
        "heldout_clip_ids": heldout_ids,
        "fm_steps": args.fm_steps,
        "seed": args.seed,
        "max_generation_length": args.max_gen,
        "temperature": 0.6,
        "top_p": 0.98,
        "t0_us": 5_100_000,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    print("Loading model...", flush=True)
    model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    lib.set_expert_attn_impl(model, "eager")  # expert always eager (cheap, hooks need weights)

    L = len(model.vlm.model.language_model.layers)  # 36
    H = model.vlm.config.text_config.num_attention_heads  # 32
    KV = model.vlm.config.text_config.num_key_value_heads  # 8
    D = model.vlm.config.text_config.head_dim  # 128
    LE = len(model.expert.layers)  # 36
    HE = model.expert.config.num_attention_heads  # 16

    vlm_stats_total = None
    expert_stats_total = None
    acc = {
        "taylor_vlm_q": np.zeros((L, H)),
        "taylor_vlm_kv_coc": np.zeros((L, KV)),
        "taylor_cache_k_fm": np.zeros((L, KV)),
        "taylor_cache_v_fm": np.zeros((L, KV)),
        "taylor_expert_q": np.zeros((LE, HE)),
    }
    metrics = {"clips": []}
    n_done = 0

    for ci, clip_id in enumerate(calib_ids):
        t_start = time.time()
        try:
            data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
        except Exception as e:  # dataset streaming can fail per clip; skip and log
            print(f"[{ci}] {clip_id} LOAD FAILED: {e}", flush=True)
            metrics["clips"].append({"clip_id": clip_id, "status": f"load_failed: {e}"})
            continue

        inputs = lib.build_inputs(model, processor, data, "cuda")
        spans = lib.compute_spans(model, inputs["input_ids"])
        prompt_len = inputs["input_ids"].shape[1]
        n_cameras = spans["per_camera"].shape[0] if spans["per_camera"] is not None else 0

        # ---- 1) rollout (VLM sdpa fast path, expert eager + stats hooks) ----
        lib.set_vlm_attn_impl(model, "sdpa")
        torch.manual_seed(args.seed + ci)
        torch.cuda.manual_seed_all(args.seed + ci)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(
                model, inputs, max_generation_length=args.max_gen
            )
        eos_pos = roll["eos_pos"]
        coc_start, coc_end = prompt_len, eos_pos + 1
        seq_tf = roll["sequences"][:, : eos_pos + 1].contiguous()  # (1, T_tf) teacher-forced seq

        expert_stats = lib.ExpertStatsCollector(
            LE, HE, spans, prompt_len, coc_start, coc_end,
            prefill_len=roll["past_key_values"].get_seq_length(),
        )
        expert_stats.register(model)
        offset = torch.tensor([eos_pos + 1], device="cuda")
        prefix_mask = torch.ones(1, roll["past_key_values"].get_seq_length(), device="cuda", dtype=torch.long)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            action = lib.denoise_with_cache(
                model, roll["past_key_values"], roll["rope_deltas"], offset, prefix_mask,
                seed=args.seed + ci,
            )
            hist_xyz = inputs["ego_history_xyz"][:, -1]  # (1, 16, 3)
            hist_rot = inputs["ego_history_rot"][:, -1]
            pred_xyz, pred_rot = model.action_space.action_to_traj(
                action.float(), hist_xyz.float(), hist_rot.float()
            )
        expert_stats.remove()

        gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()  # (64, 2)
        pred_xy = pred_xyz[0, :, :2].detach().cpu().numpy()  # (64, 2)
        ade = float(np.linalg.norm(pred_xy - gt_xy, axis=-1).mean())
        coc_text = model.tokenizer.decode(seq_tf[0, coc_start:coc_end].tolist())
        del roll
        torch.cuda.empty_cache()

        # ---- 2) Phase A VLM stats (eager, teacher-forced, no grad) ----
        lib.set_vlm_attn_impl(model, "eager")
        vlm_stats = lib.VLMStatsCollector(
            L, H, spans, prompt_len, coc_start, coc_end, n_cameras
        )
        vlm_stats.register(model)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hidden, _, _ = teacher_forced_vlm_forward(
                model, seq_tf, inputs["tokenized_data"], use_cache=False
            )
            nll_eval = coc_nll(model, hidden, seq_tf, coc_start, coc_end).item()
        vlm_stats.remove()
        torch.cuda.empty_cache()

        # ---- 3) Phase B pass 1: CoC NLL Taylor (sdpa, grads to gates only) ----
        lib.set_vlm_attn_impl(model, "sdpa")
        gates = lib.HeadGates(
            model.vlm.model.language_model.layers, H, KV, D, "cuda", torch.bfloat16
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden, _, _ = teacher_forced_vlm_forward(
                model, seq_tf, inputs["tokenized_data"], use_cache=False
            )
            loss = coc_nll(model, hidden, seq_tf, coc_start, coc_end)
        loss.backward()
        acc["taylor_vlm_q"] += gates.q_scores()
        acc["taylor_vlm_kv_coc"] += gates.v_scores()
        gates.remove()
        del hidden, loss
        torch.cuda.empty_cache()

        # ---- 4) Phase B pass 2: FM Taylor (expert gates + cache leaves) ----
        expert_gates = lib.HeadGates(
            model.expert.layers, HE, model.expert.config.num_key_value_heads,
            model.expert.config.head_dim, "cuda", torch.bfloat16, gate_v=False,
        )
        x1 = lib.gt_actions(model, data, "cuda")  # (1, 64, 2)
        fm_loss, k_tay, v_tay = fm_taylor_pass(
            model, seq_tf, inputs["tokenized_data"], x1, expert_gates,
            fm_steps=args.fm_steps, seed=args.seed + ci,
        )
        acc["taylor_expert_q"] += expert_gates.q_scores()
        acc["taylor_cache_k_fm"] += k_tay
        acc["taylor_cache_v_fm"] += v_tay
        expert_gates.remove()
        torch.cuda.empty_cache()

        # ---- accumulate Phase A ----
        if vlm_stats_total is None:
            vlm_stats_total = vlm_stats
            expert_stats_total = expert_stats
        else:
            for g in ("all", "coc"):
                for k in vlm_stats.STAT_KEYS:
                    vlm_stats_total.sums[g][k] += vlm_stats.sums[g][k]
            if n_cameras == vlm_stats_total.cam_sums.shape[2]:
                vlm_stats_total.cam_sums += vlm_stats.cam_sums
            vlm_stats_total.headnorm += vlm_stats.headnorm
            vlm_stats_total.js += vlm_stats.js
            vlm_stats_total.cka += vlm_stats.cka
            for k in expert_stats.sums:
                expert_stats_total.sums[k] += expert_stats.sums[k]
            expert_stats_total.headnorm += expert_stats.headnorm
            expert_stats_total.calls += expert_stats.calls
        n_done += 1

        metrics["clips"].append({
            "clip_id": clip_id,
            "status": "ok",
            "prompt_len": prompt_len,
            "coc_len": coc_end - coc_start,
            "minADE": ade,
            "coc_nll": nll_eval,
            "fm_loss": fm_loss,
            "n_image_runs": spans["n_image_runs"],
            "sec": round(time.time() - t_start, 1),
        })
        print(
            f"[{ci + 1}/{len(calib_ids)}] {clip_id} ok: prompt={prompt_len} coc={coc_end - coc_start} "
            f"ADE={ade:.3f} NLL={nll_eval:.3f} FM={fm_loss:.4f} ({time.time() - t_start:.0f}s)",
            flush=True,
        )
        if args.smoke:
            print("CoC:", coc_text[:500], flush=True)

        if (ci + 1) % 10 == 0 or (ci + 1) == len(calib_ids):
            save_results(out_dir, vlm_stats_total, expert_stats_total, acc, n_done, metrics)

    save_results(out_dir, vlm_stats_total, expert_stats_total, acc, n_done, metrics)
    ades = [c["minADE"] for c in metrics["clips"] if c["status"] == "ok"]
    summary = (
        f"head_analysis: {n_done}/{len(calib_ids)} clips ok\n"
        f"mean minADE={np.mean(ades):.3f}  mean NLL="
        f"{np.mean([c['coc_nll'] for c in metrics['clips'] if c['status'] == 'ok']):.3f}\n"
    )
    (out_dir / "summary.txt").write_text(summary)
    print(summary, flush=True)


def save_results(out_dir, vlm_stats, expert_stats, acc, n_done, metrics):
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if vlm_stats is None or n_done == 0:
        return
    arrays = {"n_clips": np.array(n_done)}
    for g in ("all", "coc"):
        for k in vlm_stats.STAT_KEYS:
            arrays[f"vlm_{k}_{g}"] = vlm_stats.sums[g][k] / n_done
    arrays["vlm_cam_mass_coc"] = vlm_stats.cam_sums / n_done
    arrays["vlm_headnorm"] = vlm_stats.headnorm / n_done
    arrays["vlm_js"] = vlm_stats.js / n_done
    arrays["vlm_cka"] = vlm_stats.cka / n_done
    for k in expert_stats.sums:
        arrays[f"expert_{k}"] = expert_stats.sums[k] / np.maximum(expert_stats.calls, 1)[:, None]
    arrays["expert_headnorm"] = expert_stats.headnorm / np.maximum(expert_stats.calls, 1)[:, None]
    for k, v in acc.items():
        arrays[k] = v / n_done
    np.savez(out_dir / "head_stats.npz", **arrays)
    print(f"  saved -> {out_dir / 'head_stats.npz'} (n={n_done})", flush=True)


if __name__ == "__main__":
    main()
