"""Gradient-side helpers for dual-objective Taylor importance on Alpamayo 2 Edge.

Port of alpamayo2-distill/experiments/importance/a2_grad_lib.py (itself a port of head_analysis prune_lib /
analysis_lib). The gate math (UnitGates, _TypedGate, TypedUnitGates), the token-type rule, the CoC NLL, the KV-group
scores, the cache-read decomposition and the FM estimator are unchanged. a2_grad_lib imports alpamayo2_super at module
level and cannot be imported in the Edge venv (transformers 5.17), hence the copy.

What differs on Edge:
  - MLP channels are gated at the input of `mlp.fc2` (relu^2 MLP: fc1 -> relu2 -> fc2; VLM 9,216 channels, expert
    3,840). Edge has no `down_proj`. Q heads stay at the input of `self_attn.o_proj` (16 heads x 128).
  - One card: the 2.65 B model needs no layer split, so every gate sits on the model's device.
  - The rollout is the generation half of Alpamayo2Edge.sample_trajectories_from_data: trajectory vocab masked, text
    EOS blocked, StopAfterEOS(<|traj_future_start|>), shared-prefill generation.
  - The teacher-forced forward passes mm_token_type_ids (HF5 M-RoPE needs it; generated tokens are text = 0) and reads
    rope_deltas from model.vlm.model after the forward.
  - The expert step follows Alpamayo2Edge._sample_actions: action_in_proj(x, t) reshaped to (1, 64, -1), expert with
    the explicit 4D non-causal mask, crop to the prefix after every step, action_out_proj.
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F
from alpamayo2_edge.models.alpamayo2_edge import (
    MaskDiscreteTrajectoryLogitsProcessor,
    MaskTokenIdsLogitsProcessor,
)
from alpamayo2_edge.models.expert_utils import (
    StopAfterEOS,
    build_expert_pos_ids_and_attn_mask,
    find_eos_offset,
    replace_padding_after_eos,
)
from alpamayo2_edge.models.token_utils import extract_text_tokens
from alpamayo2_edge.models.utils import fuse_traj_tokens
from transformers import StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList

TYPES = ("vision", "hist", "prompt_text", "coc", "sink")  # same order as run_gradient_anatomy / a2_grad_lib


def module_device(module):
    return next(module.parameters()).device


def cache_layer_kv(cache, i):
    return cache.layers[i].keys, cache.layers[i].values


def set_cache_layer_kv(cache, i, k, v):
    cache.layers[i].keys = k
    cache.layers[i].values = v


class UnitGates:
    """Multiplicative gates (value 1.0) on Q heads and MLP channels, one per unit and layer."""

    def __init__(self, layers, n_heads, head_dim, intermediate, mlp=True):
        self.n_heads, self.head_dim, self.intermediate = n_heads, head_dim, intermediate
        self.q_gates = [torch.ones(n_heads, device=module_device(ly), requires_grad=True) for ly in layers]
        self.mlp_gates = [torch.ones(intermediate, device=module_device(ly), requires_grad=True)
                          for ly in layers] if mlp else []
        self._handles = []
        for i, layer in enumerate(layers):
            self._handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self._make_q_hook(i)))
            if mlp:
                self._handles.append(layer.mlp.fc2.register_forward_pre_hook(self._make_mlp_hook(i)))

    def _make_q_hook(self, i):
        def hook(module, args):
            x = args[0]  # (B, T, H*D)
            b, t, _ = x.shape
            x = x.view(b, t, self.n_heads, self.head_dim) * self.q_gates[i].view(1, 1, -1, 1)
            return (x.view(b, t, -1),)
        return hook

    def _make_mlp_hook(self, i):
        def hook(module, args):
            return (args[0] * self.mlp_gates[i].view(1, 1, -1),)  # (B, T, I)
        return hook

    @staticmethod
    def _signed(gates, size):
        return np.stack([g.grad.float().cpu().numpy() if g.grad is not None else np.zeros(size) for g in gates])

    def q_signed(self):
        return self._signed(self.q_gates, self.n_heads)  # (L, H)

    def mlp_signed(self):
        return self._signed(self.mlp_gates, self.intermediate)  # (L, I)

    def zero_grads(self):
        for g in self.q_gates + self.mlp_gates:
            g.grad = None

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


class _TypedGate(torch.autograd.Function):
    """y = x * gate[type(p)], with dL/dgate split by the token type of position p (a2_grad_lib._TypedGate)."""

    @staticmethod
    def _per_pos(x, gate, type_idx):
        g = gate.index_select(0, type_idx)  # (T, U)
        return g.view(1, x.shape[1], x.shape[2], *([1] * (x.dim() - 3)))  # (1, T, U[, 1])

    @staticmethod
    def forward(ctx, x, gate, type_idx):
        # x (1, T, U) or (1, T, U, D); gate (n_types, U) fp32; type_idx (T,) long
        ctx.save_for_backward(x, gate, type_idx)
        return x * _TypedGate._per_pos(x, gate, type_idx)

    @staticmethod
    def backward(ctx, grad_y):
        x, gate, type_idx = ctx.saved_tensors
        prod = grad_y.float() * x.float()  # (1, T, U[, D])
        if prod.dim() == 4:
            prod = prod.sum(-1)  # (1, T, U)
        grad_gate = torch.zeros_like(gate).index_add_(0, type_idx, prod[0])  # (n_types, U)
        return (grad_y * _TypedGate._per_pos(x, gate, type_idx)).to(x.dtype), grad_gate, None


class TypedUnitGates:
    """UnitGates with one gate per (token type, unit). Summed over types = the plain gate's grad."""

    def __init__(self, layers, n_heads, head_dim, intermediate, n_types):
        self.n_heads, self.head_dim, self.intermediate = n_heads, head_dim, intermediate
        self.devices = [module_device(ly) for ly in layers]
        self.type_idx = {}
        self.q_gates = [torch.ones(n_types, n_heads, device=d, requires_grad=True) for d in self.devices]
        self.mlp_gates = [torch.ones(n_types, intermediate, device=d, requires_grad=True) for d in self.devices]
        self._handles = []
        for i, layer in enumerate(layers):
            self._handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self._make_q_hook(i)))
            self._handles.append(layer.mlp.fc2.register_forward_pre_hook(self._make_mlp_hook(i)))

    def set_types(self, type_idx):
        """type_idx: (T,) long, the token type of every position of the coming forward."""
        self.type_idx = {d: type_idx.to(d) for d in set(self.devices)}

    def _make_q_hook(self, i):
        def hook(module, args):
            x = args[0]  # (1, T, H*D)
            b, t, _ = x.shape
            y = _TypedGate.apply(x.view(b, t, self.n_heads, self.head_dim), self.q_gates[i],
                                 self.type_idx[self.devices[i]])  # (1, T, H, D)
            return (y.view(b, t, -1),)
        return hook

    def _make_mlp_hook(self, i):
        def hook(module, args):
            return (_TypedGate.apply(args[0], self.mlp_gates[i], self.type_idx[self.devices[i]]),)  # (1, T, I)
        return hook

    @staticmethod
    def _signed(gates):
        return np.stack([g.grad.float().cpu().numpy() if g.grad is not None
                         else np.zeros(tuple(g.shape)) for g in gates], 1)

    def q_signed(self):
        return self._signed(self.q_gates)  # (n_types, L, H)

    def mlp_signed(self):
        return self._signed(self.mlp_gates)  # (n_types, L, I)

    def zero_grads(self):
        for g in self.q_gates + self.mlp_gates:
            g.grad = None

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def fused_tokenized(model, model_inputs):
    """tokenized_data with the history placeholders replaced by trajectory tokens, as sample_trajectories_from_data."""
    td = dict(model_inputs["tokenized_data"])
    traj = {"ego_history_xyz": model_inputs["ego_history_xyz"], "ego_history_rot": model_inputs["ego_history_rot"]}
    td["input_ids"] = fuse_traj_tokens(model.history_traj_tokenizer, model.future_traj_tokenizer,
                                       td["input_ids"], traj, model.config.traj_ids)  # (1, L_prompt)
    td.setdefault("attention_mask", torch.ones_like(td["input_ids"]))
    return td


@torch.inference_mode()
def rollout_sequence(model, model_inputs, max_gen):
    """The release CoC rollout (generation half of Alpamayo2Edge.sample_trajectories_from_data, 1 sample).

    Seed the RNG before calling. Returns (seq (1, T) = prompt + generated up to and including <|traj_future_start|>,
    prompt_len, has_eos, cot text, rope_deltas from the prefill (1, 1)).
    """
    td = fused_tokenized(model, model_inputs)
    input_ids = td["input_ids"]
    anchor = model.config.traj_ids["future_start"]
    gc = copy.deepcopy(model.vlm.generation_config)
    gc.top_p, gc.top_k, gc.temperature = 0.98, None, 0.6
    gc.do_sample, gc.num_beams = True, 1
    gc.max_new_tokens = max_gen
    gc.output_logits = gc.output_scores = False
    gc.return_dict_in_generate = True
    gc.use_cache = True
    gc.pad_token_id = model.tokenizer.pad_token_id
    eos_ids = gc.eos_token_id
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids or [])
    gc.eos_token_id = None
    tr = model.config.traj_ids
    procs = LogitsProcessorList([
        MaskDiscreteTrajectoryLogitsProcessor(tr["history_id0"], model.config.history_vocab_size),
        MaskDiscreteTrajectoryLogitsProcessor(tr["future_id0"], model.config.future_vocab_size)])
    blocked = [t for t in eos_ids if t != anchor]
    if blocked:
        procs.append(MaskTokenIdsLogitsProcessor(blocked))
    out = model._generate_with_shared_prefill(td, generation_config=gc, n_samples_total=1, logits_processor=procs,
                                              stopping_criteria=StoppingCriteriaList([StopAfterEOS(anchor)]))
    prompt_len = input_ids.shape[1]
    gen = out.sequences[:, prompt_len:]
    has_eos = bool((gen == anchor).any())
    seq = out.sequences.clone()
    if has_eos:
        seq[:, prompt_len:] = replace_padding_after_eos(seq[:, prompt_len:], anchor, model.tokenizer.pad_token_id)
        end = int(find_eos_offset(seq[:, prompt_len:], anchor, seq.device)[0]) + prompt_len
    else:
        end = seq.shape[1]
    cot = extract_text_tokens(model.tokenizer, seq[:, :end])["cot"][0]
    return seq[:, :end].clone(), prompt_len, has_eos, cot, out.rope_deltas.clone()


def token_types(model, seq, prompt_len):
    """(T,) long index into TYPES: vision = image pad, hist = trajectory tokens, sink = position 0, coc = generated,
    prompt_text = the rest (system prompt, camera and time labels, vision start/end, task sentence)."""
    ids = seq[0].cpu()  # (T,)
    tr = model.config.traj_ids
    lo = min(tr["history_id0"], tr["future_id0"])
    idx = torch.full((ids.shape[0],), TYPES.index("prompt_text"), dtype=torch.long)
    idx[ids == model.config.vlm_config.image_token_id] = TYPES.index("vision")
    idx[(ids >= lo) & (ids < lo + model.config.traj_vocab_size)] = TYPES.index("hist")
    idx[0] = TYPES.index("sink")
    idx[prompt_len:] = TYPES.index("coc")
    return idx


def vlm_forward_with_grad(model, seq, td):
    """Teacher-forced VLM forward over prompt + CoC that builds the KV cache the expert reads. Grad enabled.

    Returns (last hidden (1, T, 2048), cache, rope_deltas (1, 1)).
    """
    prompt_len = td["input_ids"].shape[1]
    mm = td["mm_token_type_ids"]
    mm = torch.cat([mm, mm.new_zeros((1, seq.shape[1] - prompt_len))], dim=1)  # generated tokens are text
    out = model.vlm.model(input_ids=seq, attention_mask=torch.ones_like(seq), pixel_values=td["pixel_values"],
                          image_grid_thw=td["image_grid_thw"], mm_token_type_ids=mm, use_cache=True)
    return out.last_hidden_state, out.past_key_values, model.vlm.model.rope_deltas.clone()


def coc_nll(model, hidden, seq, coc_start, coc_end):
    """CoC NLL over the generated positions; hidden at p predicts token p+1."""
    h = hidden[:, coc_start - 1 : coc_end - 1]  # (1, Tc, 2048)
    logits = model.vlm.lm_head(h).float()  # (1, Tc, V)
    return F.cross_entropy(logits[0], seq[0, coc_start:coc_end].to(logits.device))


def kv_group_scores(tensors):
    """Taylor score per KV group from retained cache grads. Returns (L, KV) twice."""
    n_kv = tensors[0][0].shape[1]
    k_s, v_s = np.zeros((len(tensors), n_kv)), np.zeros((len(tensors), n_kv))
    with torch.no_grad():
        for i, (k, v) in enumerate(tensors):
            if k.grad is not None:
                k_s[i] = (k * k.grad).abs().sum((0, 2, 3)).float().cpu().numpy()
            if v.grad is not None:
                v_s[i] = (v * v.grad).abs().sum((0, 2, 3)).float().cpu().numpy()
    return k_s, v_s


def gt_actions(model, data, device):
    """GT future in Edge's normalised course-heading action space. (1, 64, 2) fp32."""
    hx = data["ego_history_xyz"].to(device).float()  # (1, 20, 3)
    hr = data["ego_history_rot"].to(device).float()  # (1, 20, 3, 3)
    fx = data["ego_future_xyz"].to(device).float()  # (1, 64, 3)
    fr = data["ego_future_rot"].to(device).float()  # (1, 64, 3, 3)
    with torch.autocast("cuda", enabled=False):
        action = model.expert.action_space.traj_to_action(
            traj_history_xyz=hx, traj_history_rot=hr, traj_future_xyz=fx, traj_future_rot=fr)
    return action.reshape(1, *model.expert.action_space.get_action_space_dims()).float()


def expert_fm_grads(model, cache, rope_deltas, x1, fm_steps, seed, prefill, step_gates=None):
    """FM loss backward through the expert onto detached cache leaves (a2_grad_lib.expert_fm_grads).

    t_s = (s + 0.5) / S, fresh CPU-generator noise per step seeded with the clip seed, x_t = t x1 + (1 - t) eps,
    target x1 - eps (Edge's FlowMatching.construct_training_data / compute_loss_from_pred).
    Returns (mean FM loss, [(dL/dk, dL/dv)] per cache layer summed over steps, leaves,
    per-step signed expert gate grads {"q": (S, L, H), "mlp": (S, L, I)} or None).
    """
    dev = x1.device
    n_layers = len(model.expert.expert.layers)
    leaves = []
    for i in range(n_layers):
        k, v = cache_layer_kv(cache, i)
        leaves.append((k.detach().requires_grad_(True), v.detach().requires_grad_(True)))
    action_dims = model.expert.action_space.get_action_space_dims()  # (64, 2)
    n_tok = action_dims[0]
    position_ids, attention_mask = build_expert_pos_ids_and_attn_mask(
        offset=torch.tensor([prefill], device=dev), rope_deltas=rope_deltas.to(dev), kv_cache_seq_len=prefill,
        n_diffusion_tokens=n_tok, b_star=1, device=dev,
        prefix_mask=torch.ones(1, prefill, device=dev, dtype=torch.long))
    if model.expert.expert.config._attn_implementation == "flash_attention_2":
        attention_mask = attention_mask[:, 0, 0].eq(0).long()

    losses, step_q, step_mlp = [], [], []
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for s in range(fm_steps):
        t_val = (s + 0.5) / fm_steps
        noise = torch.randn(x1.shape, generator=gen).to(dev)  # (1, 64, 2)
        x_t = t_val * x1 + (1.0 - t_val) * noise
        v_target = x1 - noise
        t = torch.full((1, 1, 1), t_val, device=dev)
        for i, (k, v) in enumerate(leaves):  # the previous step's update()+crop() left graph slices in the cache
            set_cache_layer_kv(cache, i, k, v)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            embeds = model.expert.action_in_proj(x_t, t).reshape(1, n_tok, -1)  # (1, 64, 512)
            out = model.expert.expert(inputs_embeds=embeds, position_ids=position_ids, past_key_values=cache,
                                      attention_mask=attention_mask, use_cache=True, is_causal=False)
            cache.crop(prefill)
            pred = model.expert.action_out_proj(out.last_hidden_state).reshape(-1, *action_dims)  # (1, 64, 2)
        loss = F.mse_loss(pred.float(), v_target)
        loss.backward()
        losses.append(loss.item())
        if step_gates is not None:
            step_q.append(step_gates.q_signed())
            step_mlp.append(step_gates.mlp_signed())
            step_gates.zero_grads()
    grads = [(k.grad, v.grad) for k, v in leaves]
    steps = {"q": np.stack(step_q), "mlp": np.stack(step_mlp)} if step_gates is not None else None
    return float(np.mean(losses)), grads, leaves, steps


def cache_read(leaves, grads, type_idx, n_types):
    """|k dL/dk| + |v dL/dv| on the expert-side leaves, by (cache layer, position type). (L, n_types)."""
    d = np.zeros((len(leaves), n_types))
    with torch.no_grad():
        for m, ((k, v), (gk, gv)) in enumerate(zip(leaves, grads)):
            if gk is None or gv is None:
                raise RuntimeError(f"expert left no gradient on cache layer {m}")
            a = (k * gk).abs().sum((0, 1, 3)) + (v * gv).abs().sum((0, 1, 3))  # (T,)
            d[m] = torch.zeros(n_types, device=a.device, dtype=torch.float64).index_add_(
                0, type_idx.to(a.device), a.double()).cpu().numpy()
    return d
