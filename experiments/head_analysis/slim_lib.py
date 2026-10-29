"""Physical removal (slim) of pruned units from Alpamayo-1.5.

Converts the 0/1 keep-masks of a validated config (mask_lib semantics) into real weight
surgery: q_proj rows / o_proj columns per kept Q head, gate/up rows / down columns per
kept MLP channel. KV heads, k/v projections and the KV cache are never touched (8x128
everywhere) -- the expert reads the VLM's per-layer cache, so the cache layout is a hard
interface.

Ragged GQA: pruning Q heads unevenly across KV groups breaks repeat_kv / enable_gqa
(both need a uniform group size per layer). The slim attention forward therefore gathers
K/V per kept Q head after the cache update (index_select on the head dim) and runs with
num_key_value_groups=1. bind_identity() puts the unpruned model on the same code path
for paired latency comparisons -- stock sdpa decode takes the enable_gqa zero-copy path
while the gather materializes, so stock-vs-slim alone would conflate the two effects.

Layer 35 of the integrated config loses its whole q/o path and MLP but must keep
producing k/v for the expert -> kvonly_layer_forward (bias-free projections make the
masked layer's residual contributions exactly zero, so passthrough is the faithful
physical equivalent).

The Minitron width baseline (minitron/) also narrows the VLM residual stream:
slice_hidden(), recorded as `vlm_hidden` in the meta. Checkpoints without that key are
untouched by it.

Checkpoints are torch.save state_dicts + a meta.json of kept indices; save_pretrained
cannot round-trip non-uniform layer shapes.
"""

import json
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn

from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    ALL_ATTENTION_FUNCTIONS,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

# The base snapshot every checkpoint in this track was cut from. Several revisions of the
# gated repo are cached on this box, so an unpinned from_pretrained is not reproducible.
MODEL_REV = "7aba8293c09993f2e125c6819df05d7fa3e873ea"


def slim_attn_forward(self, hidden_states, position_embeddings, attention_mask,
                      past_key_values=None, cache_position=None, **kwargs):
    """Stock Qwen3VLTextAttention.forward + K/V gather for ragged GQA."""
    input_shape = hidden_states.shape[:-1]  # (B, T)
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states,
                                                          self.layer_idx, cache_kwargs)

    # (B, 8, S, D) -> (B, n_keep, S, D): one K/V copy per kept Q head; groups become 1.
    # Must happen AFTER the cache update so the cache stays compact (8 KV heads).
    key_states = key_states.index_select(1, self.kv_head_index)
    value_states = value_states.index_select(1, self.kv_head_index)

    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self, query_states, key_states, value_states, attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling, **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def kvonly_layer_forward(self, hidden_states, position_embeddings, attention_mask=None,
                         position_ids=None, past_key_values=None, use_cache=False,
                         cache_position=None, **kwargs):
    """Layer whose q/o path and MLP are fully pruned: only produce k/v for the cache."""
    attn = self.self_attn
    normed = self.input_layernorm(hidden_states)
    hidden_shape = (*normed.shape[:-1], -1, attn.head_dim)
    k = attn.k_norm(attn.k_proj(normed).view(hidden_shape)).transpose(1, 2)  # (B, 8, T, D)
    v = attn.v_proj(normed).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    _, k = apply_rotary_pos_emb(k, k, cos, sin)
    if past_key_values is not None:
        past_key_values.update(k, v, attn.layer_idx,
                               {"sin": sin, "cos": cos, "cache_position": cache_position})
    return hidden_states


def _swap(linear, weight):
    """In-place Parameter swap -- a fresh nn.Linear would silently be fp32."""
    linear.weight = nn.Parameter(weight.contiguous(), requires_grad=False)
    linear.out_features, linear.in_features = weight.shape


def slice_layer(layer, kept_q, kept_mlp, head_dim):
    """Physical surgery on one decoder layer. kept_q/kept_mlp: ascending original indices.

    `kept_mlp` may be empty: gate/up become Linear(hidden, 0), down Linear(0, hidden), and
    the block returns exact zeros, so the residual passes through and the MLP sublayer is
    gone rather than merely narrow (the endpoint of the expert-MLP ladder,
    plans/2026-08-31_dualrwl-expert-mlp.md). `kept_q` must still be non-empty -- a layer
    with no attention heads is what make_kvonly() is for.
    """
    assert len(kept_q) > 0
    attn = layer.self_attn
    assert attn.q_proj.bias is None and attn.o_proj.bias is None
    qi = torch.as_tensor(kept_q, dtype=torch.long, device=attn.q_proj.weight.device)
    rows = (qi[:, None] * head_dim
            + torch.arange(head_dim, device=qi.device)).reshape(-1)  # (n_keep*D,)
    _swap(attn.q_proj, attn.q_proj.weight.data[rows])
    _swap(attn.o_proj, attn.o_proj.weight.data[:, rows])
    mlp = layer.mlp
    mi = torch.as_tensor(kept_mlp, dtype=torch.long, device=mlp.gate_proj.weight.device)
    _swap(mlp.gate_proj, mlp.gate_proj.weight.data[mi])
    _swap(mlp.up_proj, mlp.up_proj.weight.data[mi])
    _swap(mlp.down_proj, mlp.down_proj.weight.data[:, mi])
    mlp.intermediate_size = mi.numel()


def bind_slim_attention(attn, kept_q, n_heads_orig, n_kv_heads):
    """kv_head_index[i] = KV head of the i-th kept Q head; groups collapse to 1."""
    per_group = n_heads_orig // n_kv_heads
    idx = torch.as_tensor(kept_q, dtype=torch.long,
                          device=attn.k_proj.weight.device) // per_group
    attn.register_buffer("kv_head_index", idx, persistent=False)
    attn.num_key_value_groups = 1
    attn.forward = types.MethodType(slim_attn_forward, attn)


def make_kvonly(layer):
    """Strip a fully-pruned layer down to its k/v-producing path."""
    attn = layer.self_attn
    attn.q_proj = None
    attn.o_proj = None
    attn.q_norm = None
    layer.mlp = None
    layer.post_attention_layernorm = None
    layer.forward = types.MethodType(kvonly_layer_forward, layer)


KVONLY_EXTRA = {"q_norm", "post_attention_layernorm"}  # removed beyond the mask accounting


def _keep_rows(linear, idx):
    """Output-row slice of a Linear, bias included."""
    _swap(linear, linear.weight.data[idx])
    if linear.bias is not None:
        linear.bias = nn.Parameter(linear.bias.data[idx].contiguous(), requires_grad=False)


def _keep_norm(norm, idx):
    norm.weight = nn.Parameter(norm.weight.data[idx].contiguous(), requires_grad=False)


def slice_hidden(model, kept_h):
    """Narrow the VLM text tower's residual stream to `kept_h` (ascending channel ids).

    One kept set for every layer -- the residual carries it between layers (Minitron,
    Sheared-LLaMA). Sliced: embed_tokens and lm_head columns, every RMSNorm weight, q/k/v
    and gate/up input columns, o/down output rows, the final norm, and the output rows of
    the vision merger and the DeepStack mergers, which write straight into the residual.
    k/v_proj keep their 8x128 outputs, so the cache layout the expert reads is unchanged;
    its values are not. RMSNorm is not rescaled -- it now averages over the kept channels,
    and Minitron leaves that shift to distillation. The ViT is untouched.
    """
    lm = model.vlm.model.language_model
    h = torch.as_tensor(kept_h, dtype=torch.long, device=lm.embed_tokens.weight.device)
    lm.embed_tokens.weight = nn.Parameter(
        lm.embed_tokens.weight.data[:, h].contiguous(), requires_grad=False)
    lm.embed_tokens.embedding_dim = h.numel()
    _swap(model.vlm.lm_head, model.vlm.lm_head.weight.data[:, h])
    for layer in lm.layers:
        attn = layer.self_attn
        _keep_norm(layer.input_layernorm, h)
        for proj in (attn.q_proj, attn.k_proj, attn.v_proj):
            if proj is not None:
                _swap(proj, proj.weight.data[:, h])
        if attn.o_proj is not None:
            _keep_rows(attn.o_proj, h)
        if layer.mlp is not None:  # None on kv-only layers
            _keep_norm(layer.post_attention_layernorm, h)
            _swap(layer.mlp.gate_proj, layer.mlp.gate_proj.weight.data[:, h])
            _swap(layer.mlp.up_proj, layer.mlp.up_proj.weight.data[:, h])
            _keep_rows(layer.mlp.down_proj, h)
    _keep_norm(lm.norm, h)
    vis = model.vlm.model.visual
    for merger in (vis.merger, *vis.deepstack_merger_list):
        _keep_rows(merger.linear_fc2, h)
    model.vlm.config.text_config.hidden_size = h.numel()
    model.vlm.config.vision_config.out_hidden_size = h.numel()


def apply_surgery(model, vlm_q, vlm_mlp, exp_q, exp_mlp, kvonly_layers=(), vlm_hidden=None):
    """Apply keep-masks ((L,H)/(L,I) 0/1 arrays, mask_lib semantics) as real removal.

    `vlm_hidden` (ascending kept residual channels) additionally runs slice_hidden().
    """
    meta = {"vlm": [], "expert": [], "kvonly_layers": list(kvonly_layers)}
    tc = model.vlm.config.text_config
    for li, layer in enumerate(model.vlm.model.language_model.layers):
        kq = np.nonzero(vlm_q[li])[0].tolist()
        km = np.nonzero(vlm_mlp[li])[0].tolist()
        meta["vlm"].append({"q": kq, "mlp": km})
        if li in kvonly_layers:
            assert not kq and not km
            make_kvonly(layer)
        else:
            slice_layer(layer, kq, km, tc.head_dim)
            bind_slim_attention(layer.self_attn, kq, tc.num_attention_heads,
                                tc.num_key_value_heads)
    ec = model.expert.config
    for li, layer in enumerate(model.expert.layers):
        kq = np.nonzero(exp_q[li])[0].tolist()
        km = np.nonzero(exp_mlp[li])[0].tolist()
        meta["expert"].append({"q": kq, "mlp": km})
        slice_layer(layer, kq, km, ec.head_dim)
        bind_slim_attention(layer.self_attn, kq, ec.num_attention_heads,
                            ec.num_key_value_heads)
    if vlm_hidden is not None:
        slice_hidden(model, vlm_hidden)
        meta["vlm_hidden"] = [int(i) for i in vlm_hidden]
    return meta


def bind_identity(model):
    """Unpruned model on the slim attention path (latency control for the gather cost)."""
    tc = model.vlm.config.text_config
    for layer in model.vlm.model.language_model.layers:
        bind_slim_attention(layer.self_attn, list(range(tc.num_attention_heads)),
                            tc.num_attention_heads, tc.num_key_value_heads)
    ec = model.expert.config
    for layer in model.expert.layers:
        bind_slim_attention(layer.self_attn, list(range(ec.num_attention_heads)),
                            ec.num_attention_heads, ec.num_key_value_heads)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def check_slim(model):
    """Every parameter must have stayed bf16 and contiguous through surgery."""
    for name, p in model.named_parameters():
        assert p.dtype == torch.bfloat16, f"{name}: {p.dtype}"
        assert p.is_contiguous(), name


def save_slim(model, meta, out_dir, write_state=True):
    """Write the recipe, and the 16.8 GB state_dict unless told not to.

    load_slim() rebuilds the identical weights from the meta alone, so the state file is
    only needed to move a checkpoint to a machine that cannot reach the base weights.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "slim_meta.json").write_text(json.dumps(meta))
    if not write_state:
        return None
    torch.save(model.state_dict(), out_dir / "slim_state.pt")
    return out_dir / "slim_state.pt"


def load_slim(ckpt_dir, device="cuda", revision=MODEL_REV):
    """Full skeleton -> structural surgery from meta -> strict state_dict load.

    `slim_state.pt` is optional. apply_surgery slices the base weights in place, so the
    surgically-modified skeleton already *is* the slim checkpoint -- the state_dict load
    is an identity restated for safety when the file is there, and skipped when it is
    not. That makes slim_meta.json (3.4 MB) a complete recipe: it reconstructs the same
    model the 16.8 GB state file holds, bit for bit (verified tensor-by-tensor).

    `revision` is pinned because the base repo has several snapshots and an unpinned
    from_pretrained picks whichever the cache resolves; a different revision either
    fails the strict load or, worse, matches shapes and silently loads another model.

    Caller must have imported expert_per_clip first (gated-hub patch), like every runner.
    """
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    ckpt_dir = Path(ckpt_dir)
    meta = json.loads((ckpt_dir / "slim_meta.json").read_text())
    model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", revision=revision,
                                        dtype=torch.bfloat16)
    tc = model.vlm.config.text_config
    ec = model.expert.config
    vq = np.zeros((tc.num_hidden_layers, tc.num_attention_heads))
    vm = np.zeros((tc.num_hidden_layers, tc.intermediate_size))
    for li, m in enumerate(meta["vlm"]):
        vq[li, m["q"]] = 1.0
        vm[li, m["mlp"]] = 1.0
    eq = np.zeros((ec.num_hidden_layers, ec.num_attention_heads))
    em = np.zeros((ec.num_hidden_layers, ec.intermediate_size))
    for li, m in enumerate(meta["expert"]):
        eq[li, m["q"]] = 1.0
        em[li, m["mlp"]] = 1.0
    apply_surgery(model, vq, vm, eq, em, kvonly_layers=tuple(meta["kvonly_layers"]),
                  vlm_hidden=meta.get("vlm_hidden"))
    state = ckpt_dir / "slim_state.pt"
    if state.exists():
        sd = torch.load(state, map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)
