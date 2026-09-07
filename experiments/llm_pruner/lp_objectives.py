# Ported from soowon's alpamayo1.5 research tree
#   /home/cvlab21/project/soowon/vla-ad/alpamayo1.5/experiments/llm_pruner/lp_objectives.py
# copied 2026-09-06 to run the second-order (param_mix) arm that was never executed there.
# That tree had itself vendored analysis_lib / sample_cache / expert_per_clip / slim_lib
# FROM this repo on 2026-08-09; those copies are deliberately NOT brought back -- this
# repo's versions are newer (expert_per_clip carries an AcceleratorError fix, slim_lib a
# pinned MODEL_REV and write_state, sample_cache calib_samples()), and this code was
# written against them in the first place.
# Changes from the original are marked `PORT:`; the importance formulas are otherwise
# untouched, because upstream fidelity is the whole point of this file.
"""The two differentiable objectives LLM-Pruner's Taylor estimator backprops here.

Upstream backprops plain causal-LM next-token cross-entropy over 10 bookcorpus samples
(`hf_prune.py:146-148`).  Alpamayo ships no `forward` and no loss at all, so both have to
be written:

  (A) `coc_nll`  -- the direct analogue: teacher-forced cross-entropy over the
      chain-of-causation span the model itself just generated.  This is the headline
      objective, the one that makes this a reproduction of LLM-Pruner rather than
      something else.

  (B) `traj_fm_backward` -- flow-matching MSE on the action expert.  Not
      LLM-Pruner-canonical, but it is the only signal that measures trajectory damage,
      and gradient reaches the VLM through the KV cache the expert reads.  Ten denoise
      steps accumulate dL/d(cache) on detached leaves and then ONE chained VLM backward
      replays them, so the expensive tower is traversed once, not ten times.

Structure of (B) follows the in-lab implementation in chan's `prune_lib.expert_fm_grads`,
which is the version empirically validated against this model.
"""

import numpy as np
import torch
import torch.nn.functional as F

import analysis_lib as lib


def vlm_forward(model, seq_tf, tokenized_data, use_cache):
    out = model.vlm.model(
        input_ids=seq_tf,
        attention_mask=torch.ones_like(seq_tf),
        pixel_values=tokenized_data["pixel_values"],
        image_grid_thw=tokenized_data["image_grid_thw"],
        use_cache=use_cache,
    )
    return out.last_hidden_state, out.past_key_values, out.rope_deltas


def coc_nll(model, hidden, seq_tf, coc_start, coc_end):
    """Mean per-token NLL over the CoC span; hidden at p predicts token p+1.

    Only `coc_end - coc_start` positions (~16 tokens) go through `lm_head`, so the
    155697-wide logits never materialise for the ~3086-token prompt.
    """
    h = hidden[:, coc_start - 1 : coc_end - 1]
    logits = model.vlm.lm_head(h).float()
    return F.cross_entropy(logits[0], seq_tf[0, coc_start:coc_end])


def retain_cache_grads(cache, n_layers):
    """Keep .grad on the non-leaf cache tensors so the expert's grads can be replayed."""
    tensors = []
    for i in range(n_layers):
        k, v = lib.cache_layer_kv(cache, i)
        if not k.requires_grad:
            raise RuntimeError(
                f"cache layer {i} carries no graph -- call "
                "model.vlm.enable_input_require_grads() before the forward"
            )
        k.retain_grad()
        v.retain_grad()
        tensors.append((k, v))
    return tensors


def traj_fm_backward(model, cache, rope_deltas, x1, fm_steps, seed, prefill):
    """Flow-matching MSE through the expert onto detached cache leaves.

    Returns (mean loss, [(dL/dk, dL/dv) per layer]).  The caller chains them into the VLM
    with a single `torch.autograd.backward(cache_tensors, grads)`.
    """
    device = x1.device
    n_layers = len(model.expert.layers)
    leaves = []
    for i in range(n_layers):
        k, v = lib.cache_layer_kv(cache, i)
        leaves.append((k.detach().requires_grad_(True), v.detach().requires_grad_(True)))

    n_tok = model.action_space.get_action_space_dims()[0]  # 64
    position_ids, attention_mask = model._build_expert_pos_ids_and_attn_mask(
        offset=torch.tensor([prefill], device=device),
        rope_deltas=rope_deltas,
        kv_cache_seq_len=prefill,
        n_diffusion_tokens=n_tok,
        b_star=1,
        device=device,
        prefix_mask=torch.ones(1, prefill, device=device, dtype=torch.long),
    )
    forward_kwargs = {}
    if model.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False

    gen = torch.Generator(device="cpu").manual_seed(seed)
    losses = []
    for s in range(fm_steps):
        t_val = (s + 0.5) / fm_steps
        noise = torch.randn(x1.shape, generator=gen).to(device)
        x_t = (1.0 - t_val) * noise + t_val * x1
        v_target = x1 - noise
        t = torch.full((1, 1, 1), t_val, device=device)
        # update()+crop() leave graph-attached slices from the previous step behind and
        # that graph is freed by backward(); reinstall the leaves every step.
        for i, (k, v) in enumerate(leaves):
            lib.set_cache_layer_kv(cache, i, k, v)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            embeds = model.action_in_proj(x_t.to(torch.bfloat16), t)
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
            pred = model.action_out_proj(out.last_hidden_state[:, -n_tok:])
        loss = F.mse_loss(pred.float(), v_target)
        loss.backward()
        losses.append(loss.item())

    return float(np.mean(losses)), [(k.grad, v.grad) for k, v in leaves]
