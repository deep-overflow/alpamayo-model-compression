"""Functional expert denoise: cache-immutable fast path + torch.compile.

The release denoise appends 64 diffusion tokens to the shared DynamicCache on every
flow-matching step and crops them back (10x per denoise). That append/crop churn
allocates every step and makes the step impure, which blocks CUDA-graph capture.
StaticPrefixCache replaces it: update() returns cat(prefix, current) WITHOUT storing,
so each step is a pure function of (x, t) -- same op sequence, graph-capturable.
(The 4D expert mask early-exits transformers' mask creation before any cache access,
so the stand-in only needs update() and get_seq_length().)

The prefix K/V is padded up to a bucket multiple so tensor shapes repeat across clips
(a graph is captured per distinct shape). Padding is exact by construction:
_build_expert_pos_ids_and_attn_mask assigns -inf to every KV position in
[offset, kv_len - 64), which covers the padding.

Two dispatch-amortization backends:
  - get_compiled_step: torch.compile(reduce-overhead). Needs inductor/triton, which
    builds a C extension at runtime -- unavailable in this image (no python3.12-dev
    headers; adding them to the Docker image would enable this path + kernel fusion).
  - GraphedDenoiser: manual torch.cuda.CUDAGraph capture of the pure step. No
    compilation toolchain needed; amortizes launches, which is the dominant cost.
    Run with PYTORCH_CUDA_ALLOC_CONF="" -- expandable_segments conflicts with capture.
"""

import torch


class StaticPrefixCache:
    """Read-only Cache stand-in: the expert attends over [prefix || own 64 tokens]."""

    def __init__(self, keys, values, seq_len):
        self.keys = keys  # per layer (1, 8, S_pad, 128)
        self.values = values
        self._seq = seq_len

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        return (torch.cat([self.keys[layer_idx], key_states], dim=2),
                torch.cat([self.values[layer_idx], value_states], dim=2))

    def get_seq_length(self, layer_idx=0):
        return self._seq


def build_prefix(prompt_cache, n_layers, pad_multiple=128):
    """Snapshot the VLM cache into padded static tensors (own memory)."""
    import analysis_lib as lib

    s = prompt_cache.get_seq_length()
    s_pad = -(-s // pad_multiple) * pad_multiple if pad_multiple else s
    keys, values = [], []
    for i in range(n_layers):
        k, v = lib.cache_layer_kv(prompt_cache, i)
        pad = s_pad - k.shape[2]
        if pad:
            k = torch.nn.functional.pad(k, (0, 0, 0, pad))  # (1, 8, S_pad, 128)
            v = torch.nn.functional.pad(v, (0, 0, 0, pad))
        keys.append(k.contiguous())
        values.append(v.contiguous())
    return StaticPrefixCache(keys, values, s_pad)


_COMPILED = {}


def get_compiled_step(model, mode="reduce-overhead"):
    """One compiled step per model; explicit tensor args so clips share the graph."""
    key = id(model)
    if key not in _COMPILED:
        def _step(x, t, position_ids, attention_mask, keys, values, seq_len, n_tok, non_causal):
            prefix = StaticPrefixCache(list(keys), list(values), seq_len)
            emb = model.action_in_proj(x, t)  # (1, 64, 2048)
            if emb.dim() == 2:
                emb = emb.view(1, n_tok, -1)
            kw = {"is_causal": False} if non_causal else {}
            out = model.expert(inputs_embeds=emb, position_ids=position_ids,
                               past_key_values=prefix, attention_mask=attention_mask,
                               use_cache=True, **kw)
            return model.action_out_proj(out.last_hidden_state[:, -n_tok:])  # (1, 64, 2)

        _COMPILED[key] = torch.compile(_step, mode=mode)
    return _COMPILED[key]


class GraphedDenoiser:
    """Manual CUDA-graph capture of the fast step; one instance per (model, S_pad).

    Static input buffers (x, t, prefix K/V, pos ids, mask) are baked into the graph;
    clips with the same padded length reuse it via load_clip() copies. Captured with
    zero-filled buffers -- the kernels are data-independent, so values only flow at
    replay time.
    """

    def __init__(self, model, s_pad, device="cuda"):
        self.model = model
        self.s_pad = s_pad
        ec = model.expert.config
        self.n_tok = model.action_space.get_action_space_dims()[0]  # 64
        kv = (1, ec.num_key_value_heads, s_pad, ec.head_dim)
        self.keys = [torch.zeros(kv, dtype=torch.bfloat16, device=device)
                     for _ in range(ec.num_hidden_layers)]
        self.values = [torch.zeros(kv, dtype=torch.bfloat16, device=device)
                       for _ in range(ec.num_hidden_layers)]
        self.x = torch.zeros(1, self.n_tok, 2, device=device)  # (1, 64, 2) f32
        self.t = torch.zeros(1, 1, 1, device=device)
        self.pos = torch.zeros(3, 1, self.n_tok, dtype=torch.long, device=device)
        self.mask = torch.zeros(1, 1, self.n_tok, s_pad + self.n_tok, device=device)
        self._capture()

    @torch.no_grad()
    def _raw(self):
        prefix = StaticPrefixCache(self.keys, self.values, self.s_pad)
        kw = {"is_causal": False} if self.model.config.expert_non_causal_attention else {}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            emb = self.model.action_in_proj(self.x, self.t)  # (1, 64, 2048)
            if emb.dim() == 2:
                emb = emb.view(1, self.n_tok, -1)
            out = self.model.expert(inputs_embeds=emb, position_ids=self.pos,
                                    past_key_values=prefix, attention_mask=self.mask,
                                    use_cache=True, **kw)
            return self.model.action_out_proj(out.last_hidden_state[:, -self.n_tok:])

    def _capture(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):  # warmup allocations off the capture stream
            for _ in range(2):
                self._raw()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = self._raw()  # (1, 64, 2) bf16, overwritten on every replay

    def load_clip(self, prefix, rope_deltas, offset):
        assert prefix.get_seq_length() == self.s_pad
        for a, b in zip(self.keys, prefix.keys):
            a.copy_(b)
        for a, b in zip(self.values, prefix.values):
            a.copy_(b)
        pos, mask = self.model._build_expert_pos_ids_and_attn_mask(
            offset=offset, rope_deltas=rope_deltas, kv_cache_seq_len=self.s_pad,
            n_diffusion_tokens=self.n_tok, b_star=1, device=self.x.device, prefix_mask=None,
        )
        self.pos.copy_(pos)
        self.mask.copy_(mask)

    def step(self, x, t):
        self.x.copy_(x)
        self.t.copy_(t)
        self.graph.replay()
        return self.out.clone().view(-1, self.n_tok, 2)


def graphed_denoise(model, graphed, seed=42):
    """10-step denoise replaying the captured graph. load_clip() must be current."""
    torch.cuda.manual_seed_all(seed)
    return model.diffusion.sample(batch_size=1, step_fn=graphed.step,
                                  device=graphed.x.device, return_all_steps=False)


def fast_denoise(model, prefix, rope_deltas, offset, seed=42, compiled=None):
    """Drop-in equivalent of analysis_lib.denoise_with_cache on a StaticPrefixCache."""
    device = prefix.keys[0].device
    n_tok = model.action_space.get_action_space_dims()[0]  # 64
    s_pad = prefix.get_seq_length()
    position_ids, attention_mask = model._build_expert_pos_ids_and_attn_mask(
        offset=offset, rope_deltas=rope_deltas, kv_cache_seq_len=s_pad,
        n_diffusion_tokens=n_tok, b_star=1, device=device, prefix_mask=None,
    )
    non_causal = bool(model.config.expert_non_causal_attention)

    if compiled is None:
        kw = {"is_causal": False} if non_causal else {}

        def step_fn(x, t):
            emb = model.action_in_proj(x, t)
            if emb.dim() == 2:
                emb = emb.view(1, n_tok, -1)
            out = model.expert(inputs_embeds=emb, position_ids=position_ids,
                               past_key_values=prefix, attention_mask=attention_mask,
                               use_cache=True, **kw)
            last = out.last_hidden_state[:, -n_tok:]
            return model.action_out_proj(last).view(-1, *model.action_space.get_action_space_dims())
    else:
        keys, values = tuple(prefix.keys), tuple(prefix.values)

        def step_fn(x, t):
            a = compiled(x, t, position_ids, attention_mask, keys, values, s_pad, n_tok,
                         non_causal)
            return a.view(-1, *model.action_space.get_action_space_dims())

    torch.cuda.manual_seed_all(seed)
    return model.diffusion.sample(batch_size=1, step_fn=step_fn, device=device,
                                  return_all_steps=False)  # (1, 64, 2)
