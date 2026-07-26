"""Graphed CoC decode: manual static-KV loop + CUDA-graph single-step capture.

Decode is dispatch-bound at batch 1 (~450 kernel launches per token), so the step is
captured once as a CUDA graph and replayed per token. A transformers StaticCache holds
KV in fixed-capacity buffers; the current length lives in DATA (cache_position index +
additive mask), not in shapes, so ONE capture serves every clip and every step.

The Qwen3VLModel wrapper branches on tensor values for its prefill/decode position
logic, so the graphed step calls language_model directly and computes the mrope decode
position as pure tensor math: pos = cache_pos + rope_delta (all three mrope rows equal
in decode). Prefill runs eager through the wrapper (compute-bound; also sets
rope_deltas and writes cache positions [0, S)).

Sampling (temperature/top-p + traj-vocab masking a la ExpertLogitsProcessor) runs
between replays in Python; sampled token streams differ from vlm.generate() in RNG
consumption, so equivalence is checked teacher-forced: per-token replay logits of a
fixed sequence vs the batched TF forward.

Run with PYTORCH_CUDA_ALLOC_CONF="" (expandable_segments conflicts with capture).
"""

import torch
from transformers import StaticCache


class GraphedDecoder:
    """One capture per (model, capacity); every clip/step reuses it via buffer writes."""

    def __init__(self, model, cap, device="cuda"):
        self.model = model
        self.cap = cap
        self.token = torch.zeros(1, 1, dtype=torch.long, device=device)
        self.cache_pos = torch.zeros(1, dtype=torch.long, device=device)
        self.delta = torch.zeros(1, 1, dtype=torch.long, device=device)  # rope_deltas
        # bf16 additive mask: a float32 mask mismatches the autocast query dtype and
        # kicks sdpa to the math backend (fp32 upcast of the whole KV -> ~3x step time)
        self.mask = torch.full((1, 1, 1, cap), torch.finfo(torch.bfloat16).min,
                               dtype=torch.bfloat16, device=device)
        self.cache = StaticCache(config=model.vlm.config.text_config, max_cache_len=cap)
        self._capture()

    @torch.no_grad()
    def _raw(self):
        # decode mrope position: all 3 rows = cache_pos + rope_delta
        pos3 = (self.cache_pos.view(1, 1) + self.delta).unsqueeze(0).expand(3, 1, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model.vlm.model.language_model(
                input_ids=self.token, attention_mask=self.mask, position_ids=pos3,
                past_key_values=self.cache, cache_position=self.cache_pos, use_cache=True,
            )
            logits = self.model.vlm.lm_head(out.last_hidden_state[:, -1:])
        return logits.float()  # (1, 1, V)

    def _capture(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):  # warmup: lazy-inits the static KV, allocs
            for _ in range(2):
                self._raw()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.logits = self._raw()  # overwritten on every replay

    def reset_clip(self):
        self.mask.fill_(torch.finfo(torch.bfloat16).min)

    def step(self, token, pos, graphed=True):
        """One decode step at absolute position `pos`; returns logits (1, 1, V)."""
        self.mask[..., pos] = 0.0
        self.cache_pos.fill_(pos)
        self.token.copy_(token)
        if graphed:
            self.graph.replay()
            return self.logits
        return self._raw()


def _sample(logits, generator, temperature=0.6, top_p=0.98):
    """HF sampling semantics: temperature warp -> nucleus top-p -> multinomial."""
    scores = logits[0, -1] / temperature  # (V,)
    sorted_scores, idx = torch.sort(scores, descending=True)
    cum = torch.softmax(sorted_scores, dim=-1).cumsum(-1)
    remove = cum - torch.softmax(sorted_scores, dim=-1) > top_p  # keep first token above p
    sorted_scores[remove] = float("-inf")
    probs = torch.softmax(sorted_scores, dim=-1)
    pick = torch.multinomial(probs, 1, generator=generator)
    return idx[pick].view(1, 1)


@torch.no_grad()
def rollout_graphed(model, dec, inputs, temperature=0.6, top_p=0.98,
                    max_generation_length=256, seed=None, graphed=True):
    """Drop-in analog of analysis_lib.run_rollout on the graphed decoder."""
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]
    device = input_ids.device
    from alpamayo1_5.models.token_utils import to_special_token
    eos_id = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    off = model.config.traj_token_start_idx
    size = model.config.traj_vocab_size

    dec.reset_clip()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    ev[0].record()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.vlm.model(  # eager prefill into the static cache, sets rope_deltas
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
            pixel_values=inputs["tokenized_data"]["pixel_values"],
            image_grid_thw=inputs["tokenized_data"]["image_grid_thw"],
            past_key_values=dec.cache, use_cache=True,
            cache_position=torch.arange(prompt_len, device=device),
        )
        first_logits = model.vlm.lm_head(out.last_hidden_state[:, -1:]).float()
    dec.delta.copy_(out.rope_deltas.to(device))
    dec.mask[..., :prompt_len] = 0.0
    ev[1].record()

    gen = torch.Generator(device=device)
    gen.manual_seed(seed if seed is not None else torch.seed() % 2**31)
    tokens = []
    logits = first_logits
    for t in range(max_generation_length):
        logits = logits.clone()
        logits[..., off : off + size] = float("-inf")  # ExpertLogitsProcessor
        tok = _sample(logits, gen, temperature, top_p)
        tokens.append(tok)
        if tok.item() == eos_id or t == max_generation_length - 1:
            break
        logits = dec.step(tok, prompt_len + t, graphed=graphed)
    ev[2].record()
    torch.cuda.synchronize()

    sequences = torch.cat([input_ids] + tokens, dim=1)
    eos_mask = sequences[0] == eos_id
    eos_pos = int(eos_mask.int().argmax().item()) if eos_mask.any() else sequences.shape[1] - 1
    return {"sequences": sequences, "eos_pos": eos_pos, "prompt_len": prompt_len,
            "n_steps": len(tokens), "true_len": prompt_len + len(tokens),
            "prefill_ms": ev[0].elapsed_time(ev[1]),  # ViT + prefill + first logits
            "loop_ms": ev[1].elapsed_time(ev[2])}  # sampling loop only


@torch.no_grad()
def tf_replay_logits(model, dec, inputs, seq_tf, graphed=True):
    """Teacher-forced per-token replay of seq_tf; returns stacked decode logits.

    Feeds seq_tf[prompt_len:] one token at a time through the graphed step -- the
    equivalence probe against the batched TF forward (same tokens, same cache math).
    """
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]
    device = input_ids.device
    dec.reset_clip()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.vlm.model(
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
            pixel_values=inputs["tokenized_data"]["pixel_values"],
            image_grid_thw=inputs["tokenized_data"]["image_grid_thw"],
            past_key_values=dec.cache, use_cache=True,
            cache_position=torch.arange(prompt_len, device=device),
        )
        logits = [model.vlm.lm_head(out.last_hidden_state[:, -1:]).float()]
    dec.delta.copy_(out.rope_deltas.to(device))
    dec.mask[..., :prompt_len] = 0.0
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()  # time the decode loop only -- prefill would swamp a per-token metric
    for t in range(prompt_len, seq_tf.shape[1] - 1):
        logits.append(dec.step(seq_tf[:, t : t + 1], t, graphed=graphed).clone())
    e.record()
    torch.cuda.synchronize()
    # (1, T_coc, V) predicting seq_tf[prompt_len:], and the loop-only wall time
    return torch.cat(logits, dim=1), s.elapsed_time(e)
