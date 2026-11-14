"""Evaluation helpers for Alpamayo 2 Edge (plans/2026-11-14_a2-edge-moop-u40.md). Edge venv.

EdgeVlmMasks         run_a2_openloop.VlmMasks for Edge: zero masked Q heads at the o_proj input and masked MLP channels at the
                     mlp.fc2 input of every VLM text layer (forward pre-hooks; zeroing there = removing the unit)
rollout_then_k_traj  run_oodval.rollout_then_k_traj (the A2 open-loop protocol) on Edge: one CoC rollout through Edge's own
                     generation path (edge_grad_lib.generate), then one 10-step denoising per seed on that rollout's KV cache
                     via Alpamayo2Edge._sample_actions (release code). If the CoC never closes with the anchor within max_gen,
                     the expert reads the whole generated text (A2's find_eos_offset behaviour) and has_eos is False.
"""

import time

import edge_grad_lib as gl
import numpy as np
import torch
from alpamayo2_edge.models.expert_utils import (
    build_expert_pos_ids_and_attn_mask,
    replace_padding_after_eos,
)
from alpamayo2_edge.models.token_utils import extract_text_tokens


def seed_all(seed):
    import random

    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_determinism():
    """run_oodval.set_determinism."""
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class EdgeVlmMasks:
    """Forward pre-hooks that zero masked Q heads (o_proj input) and MLP channels (mlp.fc2 input)."""

    def __init__(self, model, q_mask, mlp_mask):
        layers = model.vlm.model.language_model.layers
        assert q_mask.shape[0] == mlp_mask.shape[0] == len(layers), (q_mask.shape, mlp_mask.shape, len(layers))
        self.handles = []
        for i, layer in enumerate(layers):
            attn = layer.self_attn
            n_heads, head_dim = q_mask.shape[1], attn.head_dim
            dev = attn.o_proj.weight.device
            qm = torch.tensor(q_mask[i], device=dev, dtype=torch.bfloat16).view(1, 1, n_heads, 1)
            mm = torch.tensor(mlp_mask[i], device=dev, dtype=torch.bfloat16).view(1, 1, -1)

            def q_hook(module, args, qm=qm, n_heads=n_heads, head_dim=head_dim):
                x = args[0]  # (B, T, H*D)
                b, t, _ = x.shape
                return ((x.view(b, t, n_heads, head_dim) * qm.to(x.dtype)).view(b, t, -1),)

            def m_hook(module, args, mm=mm):
                return (args[0] * mm.to(args[0].dtype),)  # (B, T, I)

            self.handles.append(attn.o_proj.register_forward_pre_hook(q_hook))
            self.handles.append(layer.mlp.fc2.register_forward_pre_hook(m_hook))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.inference_mode()
def rollout_then_k_traj(model, model_inputs, base_seed, traj_seeds, max_gen, inference_step=10):
    """Returns (pred_k (K, 64, 2) numpy, cot, gen_len, has_eos, sec_gen, sec_denoise, prompt_len)."""
    td = gl.fused_tokenized(model, model_inputs)
    input_ids = td["input_ids"]
    prompt_len = input_ids.shape[1]
    anchor = model.config.traj_ids["future_start"]
    dev = input_ids.device

    t_gen = time.time()
    seed_all(base_seed)
    out = gl.generate(model, td, max_gen)
    torch.cuda.synchronize()
    t_gen = time.time() - t_gen
    seq = out.sequences.clone()
    gen = seq[:, prompt_len:]
    has_eos = bool((gen == anchor).any())
    if has_eos:
        seq[:, prompt_len:] = replace_padding_after_eos(gen, anchor, model.tokenizer.pad_token_id)
        offset = int((gen[0] == anchor).nonzero()[0, 0]) + 1 + prompt_len
    else:
        offset = seq.shape[1]
    cache, rope_deltas = out.past_key_values, out.rope_deltas
    prefix_mask = td["attention_mask"]
    rope_deltas = rope_deltas - prefix_mask.eq(0).sum(dim=-1, keepdim=True)  # batch 1, no padding: unchanged
    cached = cache.get_seq_length()
    if cached < offset:  # the last token(s) were sampled but never forwarded (Alpamayo2Edge release branch)
        missing = seq[:, cached:offset]
        pos = torch.arange(cached, offset, device=dev)[None, None, :].expand(3, 1, -1) + rope_deltas[None]
        mask = torch.cat([prefix_mask, prefix_mask.new_ones((1, offset - prompt_len))], dim=1)
        model.vlm.model.language_model(input_ids=missing, position_ids=pos, attention_mask=mask, past_key_values=cache,
                                       use_cache=True)
    cache.crop(offset)
    position_ids, attention_mask = build_expert_pos_ids_and_attn_mask(
        offset=torch.tensor([offset], device=dev), rope_deltas=rope_deltas, kv_cache_seq_len=offset,
        n_diffusion_tokens=model.expert.action_space.get_action_space_dims()[0], b_star=1, device=dev,
        prefix_mask=prefix_mask)
    if model.expert.expert.config._attn_implementation == "flash_attention_2":
        attention_mask = attention_mask[:, 0, 0].eq(0).to(prefix_mask.dtype)

    hist_xyz = model_inputs["ego_history_xyz"][:, -1]  # (1, 20, 3)
    hist_rot = model_inputs["ego_history_rot"][:, -1]  # (1, 20, 3, 3)
    t_den = time.time()
    preds = []
    for s in traj_seeds:
        seed_all(s)
        action = model._sample_actions(cache, offset, position_ids, attention_mask, 1, dev,
                                       {"inference_step": inference_step})
        pred_xyz, _ = model.expert.action_space.action_to_traj(action, hist_xyz, hist_rot)
        preds.append(pred_xyz[0, :, :2].float().cpu().numpy())  # (64, 2)
    torch.cuda.synchronize()
    t_den = time.time() - t_den
    cot = extract_text_tokens(model.tokenizer, seq[:, :offset])["cot"][0]
    return np.stack(preds), cot, offset - prompt_len, has_eos, t_gen, t_den, prompt_len
