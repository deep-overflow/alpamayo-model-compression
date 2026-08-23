"""Pruning-unit gates and dual-objective Taylor importance (Phase P1).

Three structured units, per the approved plan:
  - Q head        : gate on o_proj input          (VLM and expert, uncoupled)
  - MLP channel   : gate on down_proj input       (VLM and expert, uncoupled)
  - KV group      : scored on the VLM cache k/v   (joint across both towers)

The KV group is scored on the cache rather than through a gate on k_proj: Qwen3
applies k_norm (RMSNorm over head_dim) after k_proj, which normalises away any
per-head scalar gate, so such a gate would carry no gradient. The cache tensors
are the ones attention actually consumes, in both the VLM's own attention and the
expert's read, so scoring there measures exactly "remove this KV group".
"""

import numpy as np
import torch
import torch.nn.functional as F

import analysis_lib as lib


class UnitGates:
    """Multiplicative gates (value 1.0) on Q heads and MLP channels of one tower."""

    def __init__(self, layers, n_heads, head_dim, intermediate, device, dtype):
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.intermediate = intermediate
        self.q_gates = [
            torch.ones(n_heads, device=device, dtype=dtype, requires_grad=True) for _ in layers
        ]
        self.mlp_gates = [
            torch.ones(intermediate, device=device, dtype=dtype, requires_grad=True)
            for _ in layers
        ]
        self._handles = []
        for i, layer in enumerate(layers):
            self._handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(self._make_q_hook(i))
            )
            self._handles.append(
                layer.mlp.down_proj.register_forward_pre_hook(self._make_mlp_hook(i))
            )

    def _make_q_hook(self, i):
        def hook(module, args):
            x = args[0]  # (B, T, H*D)
            b, t, _ = x.shape
            x = x.view(b, t, self.n_heads, self.head_dim) * self.q_gates[i].view(1, 1, -1, 1)
            return (x.view(b, t, -1),)
        return hook

    def _make_mlp_hook(self, i):
        def hook(module, args):
            x = args[0]  # (B, T, intermediate)
            return (x * self.mlp_gates[i].view(1, 1, -1),)
        return hook

    @staticmethod
    def _scores(gates, size):
        # gate value is 1, so |g * dL/dg| == |dL/dg|
        return np.stack([
            g.grad.abs().float().cpu().numpy() if g.grad is not None else np.zeros(size)
            for g in gates
        ])

    def q_scores(self):
        return self._scores(self.q_gates, self.n_heads)  # (L, H)

    def mlp_scores(self):
        return self._scores(self.mlp_gates, self.intermediate)  # (L, I)

    def zero_grads(self):
        for g in self.q_gates + self.mlp_gates:
            g.grad = None

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def retain_cache_grads(cache, n_layers):
    """Keep .grad on the (non-leaf) cache k/v so KV groups can be scored.

    Requires the VLM input embeddings to require grad (see
    enable_input_require_grads in the driver): the gates sit downstream of k/v,
    so without that the early layers' cache tensors carry no graph at all.
    """
    tensors = []
    for i in range(n_layers):
        k, v = lib.cache_layer_kv(cache, i)
        if not k.requires_grad or not v.requires_grad:
            raise RuntimeError(
                f"cache layer {i} k/v do not require grad; call "
                "model.vlm.enable_input_require_grads() before the forward"
            )
        k.retain_grad()
        v.retain_grad()
        tensors.append((k, v))
    return tensors


def kv_group_scores(tensors):
    """Taylor score per KV group from retained cache grads. Returns (L, KV) x2."""
    n_layers = len(tensors)
    n_kv = tensors[0][0].shape[1]
    k_s = np.zeros((n_layers, n_kv))
    v_s = np.zeros((n_layers, n_kv))
    with torch.no_grad():
        for i, (k, v) in enumerate(tensors):
            if k.grad is not None:
                k_s[i] = (k * k.grad).abs().sum((0, 2, 3)).float().cpu().numpy()
            if v.grad is not None:
                v_s[i] = (v * v.grad).abs().sum((0, 2, 3)).float().cpu().numpy()
    return k_s, v_s


def vlm_forward_with_grad(model, seq_tf, tokenized_data, use_cache):
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
    """CoC NLL over generated positions; hidden at p predicts token p+1."""
    h = hidden[:, coc_start - 1 : coc_end - 1]  # (1, Tc, 4096)
    logits = model.vlm.lm_head(h).float()  # (1, Tc, V)
    return F.cross_entropy(logits[0], seq_tf[0, coc_start:coc_end])


def expert_infer_grads(model, cache, rope_deltas, gt_xy, hist_xyz, hist_rot, seed,
                       prefill, n_steps=10, k_draws=4, x0=None):
    """Same two-stage trick as expert_fm_grads, but on the model's OWN denoising path.

    The shipped trajectory criterion measures on the training path -- x_t = (1-t) eps +
    t x1 is an interpolation toward the GT action, with a fresh eps per step, i.e. ten
    independent regression problems. Inference instead integrates the model's own field
    from t = 0, so its iterates carry its own accumulated error. This reproduces that
    Euler loop with grad on (FlowMatching.sample is @torch.no_grad) and backpropagates
    MSE(final xy, GT xy) through the whole chain onto the detached cache leaves, so the
    single VLM backward that follows scores VLM units for the trajectory the model
    actually produces.

    One shared gate spans all ten steps, so the chain rule already sums the step axis --
    the score stays |sum_s dL/dg_s|, exactly the aggregation the training-path twin uses.
    Nothing is step-normalised here: that axis was measured separately and closing it is
    a different factor (plans/2026-08-22_vlm-step-aggregation.md).

    K noise draws are accumulated on the leaves before the abs, because the training path
    effectively averages ten independent eps draws while one Euler chain is a single draw.

    Returns (mean loss, [(dL/dk, dL/dv) per layer], leaves), matching expert_fm_grads.
    """
    device = gt_xy.device
    n_layers = len(model.expert.layers)
    leaves = []
    for i in range(n_layers):
        k, v = lib.cache_layer_kv(cache, i)
        leaves.append((k.detach().requires_grad_(True), v.detach().requires_grad_(True)))

    offset = torch.tensor([prefill], device=device)
    prefix_mask = torch.ones(1, prefill, device=device, dtype=torch.long)
    n_tok = model.action_space.get_action_space_dims()[0]  # 64
    position_ids, attention_mask = model._build_expert_pos_ids_and_attn_mask(
        offset=offset, rope_deltas=rope_deltas, kv_cache_seq_len=prefill,
        n_diffusion_tokens=n_tok, b_star=1, device=device, prefix_mask=prefix_mask,
    )
    forward_kwargs = {}
    if model.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False

    dims = model.action_space.get_action_space_dims()  # (64, 2)
    dt = 1.0 / n_steps
    flat_leaves = [t for kv in leaves for t in kv]

    def euler_step(x, t, *leaf_tensors):
        # update()+crop() leave graph-attached slices from the previous step in the
        # cache; re-seat the leaves so each step reads the same prefill
        for i in range(n_layers):
            lib.set_cache_layer_kv(cache, i, leaf_tensors[2 * i], leaf_tensors[2 * i + 1])
        # x stays fp32, as the release step_fn passes it -- autocast casts inside
        # action_in_proj's Linears
        embeds = model.action_in_proj(x, t)  # (1, 64, 2048)
        if embeds.dim() == 2:
            embeds = embeds.view(1, n_tok, -1)
        out = model.expert(
            inputs_embeds=embeds, position_ids=position_ids, past_key_values=cache,
            attention_mask=attention_mask, use_cache=True, **forward_kwargs,
        )
        cache.crop(prefill)
        field = model.action_out_proj(out.last_hidden_state[:, -n_tok:]).view(1, *dims)
        return x + dt * field.float()

    losses = []
    for draw in range(k_draws):
        if x0 is None:
            gen = torch.Generator(device="cpu").manual_seed(seed + draw)
            x = torch.randn(1, *dims, generator=gen).to(device)
        else:
            x = x0.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for s in range(n_steps):
                t = torch.full((1, 1, 1), s * dt, device=device)
                # The ten step graphs would sit next to the VLM's (~40 GB on its own) and
                # OOM a 47 GB card, so each step is checkpointed: only x (1, 64, 2) and the
                # leaves survive between steps and the step is recomputed on backward.
                # The estimator is unchanged -- recomputation is deterministic here (no
                # dropout, noise drawn outside the loop).
                x = torch.utils.checkpoint.checkpoint(
                    euler_step, x, t, *flat_leaves, use_reentrant=False)
        pred_xyz, _ = model.action_space.action_to_traj(
            x.float(), hist_xyz[:, -1].float(), hist_rot[:, -1].float()
        )  # (1, 64, 3)
        loss = F.mse_loss(pred_xyz[0, :, :2], gt_xy)
        loss.backward()
        losses.append(loss.item())

    grads = [(k.grad, v.grad) for k, v in leaves]
    return float(np.mean(losses)), grads, leaves


def expert_fm_grads(model, cache, rope_deltas, x1, fm_steps, seed, prefill):
    """Run the FM loss backward through the expert onto detached cache leaves.

    Returns the accumulated dL/d(cache) so a single VLM backward can follow, and
    the mean FM loss. Doing it in two stages costs one VLM backward instead of
    fm_steps of them.
    """
    device = x1.device
    n_layers = len(model.expert.layers)
    leaves = []
    for i in range(n_layers):
        k, v = lib.cache_layer_kv(cache, i)
        leaves.append((k.detach().requires_grad_(True), v.detach().requires_grad_(True)))

    offset = torch.tensor([prefill], device=device)
    prefix_mask = torch.ones(1, prefill, device=device, dtype=torch.long)
    n_tok = model.action_space.get_action_space_dims()[0]  # 64
    position_ids, attention_mask = model._build_expert_pos_ids_and_attn_mask(
        offset=offset, rope_deltas=rope_deltas, kv_cache_seq_len=prefill,
        n_diffusion_tokens=n_tok, b_star=1, device=device, prefix_mask=prefix_mask,
    )
    forward_kwargs = {}
    if model.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False

    gen = torch.Generator(device="cpu").manual_seed(seed)
    losses = []
    for s in range(fm_steps):
        t_val = (s + 0.5) / fm_steps
        noise = torch.randn(x1.shape, generator=gen).to(device)  # (1, 64, 2)
        x_t = (1.0 - t_val) * noise + t_val * x1
        v_target = x1 - noise  # (1, 64, 2)
        t = torch.full((1, 1, 1), t_val, device=device)
        # update()+crop() leave graph-attached slices from the previous step in the
        # cache; that graph is freed after backward(), so reset the leaves each step
        for i, (k, v) in enumerate(leaves):
            lib.set_cache_layer_kv(cache, i, k, v)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            embeds = model.action_in_proj(x_t.to(torch.bfloat16), t)  # (1, 64, 2048)
            if embeds.dim() == 2:
                embeds = embeds.view(1, n_tok, -1)
            out = model.expert(
                inputs_embeds=embeds, position_ids=position_ids, past_key_values=cache,
                attention_mask=attention_mask, use_cache=True, **forward_kwargs,
            )
            cache.crop(prefill)
            pred = model.action_out_proj(out.last_hidden_state[:, -n_tok:])  # (1, 64, 2)
        loss = F.mse_loss(pred.float(), v_target)
        loss.backward()
        losses.append(loss.item())

    grads = [(k.grad, v.grad) for k, v in leaves]
    return float(np.mean(losses)), grads, leaves
