"""Shared library for Alpamayo 1.5 head-pruning attention analysis (Phase A + B).

Runs against the alpamayo1_5 package using /workspace/alpamayo1.5/.venv.
"""

import numpy as np
import torch
import torch.nn.functional as F

from alpamayo1_5 import helper
from alpamayo1_5.models.token_utils import to_special_token

IMAGE_TOKEN_ID = 151655  # <|image_pad|> in Qwen3-VL


# ---------------------------------------------------------------------------
# attention implementation switching
# ---------------------------------------------------------------------------


def set_vlm_attn_impl(model, impl: str) -> None:
    """Switch attention implementation for the VLM (text + vision) in place."""
    model.vlm.config._attn_implementation = impl
    model.vlm.config.text_config._attn_implementation = impl
    model.vlm.config.vision_config._attn_implementation = impl


def set_expert_attn_impl(model, impl: str) -> None:
    model.expert.config._attn_implementation = impl


# ---------------------------------------------------------------------------
# prompt building / spans
# ---------------------------------------------------------------------------


def build_inputs(model, processor, data, device):
    """Build fused input_ids + tokenized data exactly as the release inference does.

    Returns dict with input_ids (1, T) already containing fused traj-history tokens.
    """
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"],
    )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, device)
    tokenized = model_inputs["tokenized_data"]
    input_ids = tokenized.pop("input_ids")  # (1, T_prompt)
    traj_data_vlm = {
        "ego_history_xyz": model_inputs["ego_history_xyz"],
        "ego_history_rot": model_inputs["ego_history_rot"],
    }
    input_ids = model.fuse_traj_tokens(input_ids, traj_data_vlm)  # (1, T_prompt)
    return {
        "input_ids": input_ids,
        "tokenized_data": dict(tokenized),  # pixel_values, image_grid_thw, attention_mask
        "ego_history_xyz": model_inputs["ego_history_xyz"],
        "ego_history_rot": model_inputs["ego_history_rot"],
    }


def compute_spans(model, input_ids: torch.Tensor) -> dict:
    """Identify token-type spans in the fused prompt.

    Args:
        input_ids: (1, T_prompt) fused prompt ids.

    Returns dict of boolean masks (T_prompt,) on CPU plus per-camera vision masks.
    """
    ids = input_ids[0].cpu()  # (T,)
    t = ids.shape[0]
    vision = ids == IMAGE_TOKEN_ID
    start = model.config.traj_token_start_idx
    hist = (ids >= start) & (ids < start + model.config.traj_vocab_size)
    sink = torch.zeros(t, dtype=torch.bool)
    sink[0] = True
    text = ~(vision | hist | sink)

    # split vision positions into contiguous runs -> one run per image
    pos = torch.nonzero(vision).flatten()
    runs = []
    if len(pos) > 0:
        run_start = pos[0].item()
        prev = pos[0].item()
        for p in pos[1:].tolist():
            if p != prev + 1:
                runs.append((run_start, prev + 1))
                run_start = p
            prev = p
        runs.append((run_start, prev + 1))
    per_camera = None
    if len(runs) % 4 == 0 and len(runs) >= 4:
        n_cam = len(runs) // 4  # 4 frames per camera
        per_camera = torch.zeros(n_cam, t, dtype=torch.bool)
        for i, (s, e) in enumerate(runs):
            per_camera[i // 4, s:e] = True
    return {
        "vision": vision,
        "hist": hist,
        "sink": sink,
        "text": text,
        "per_camera": per_camera,
        "n_image_runs": len(runs),
    }


# ---------------------------------------------------------------------------
# rollout (replicates sample_trajectories_from_data_with_vlm_rollout, b=1, n=1)
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_rollout(model, inputs, temperature=0.6, top_p=0.98, max_generation_length=256):
    """Autoregressive CoC rollout with num_traj_samples=1. Returns sequences etc."""
    from transformers import StoppingCriteriaList
    from transformers.generation.logits_process import LogitsProcessorList
    from alpamayo1_5.models.alpamayo1_5 import ExpertLogitsProcessor
    from alpamayo1_5.models.token_utils import StopAfterEOS, replace_padding_after_eos

    input_ids = inputs["input_ids"]
    tokenized_data = inputs["tokenized_data"]

    generation_config = model.vlm.generation_config
    generation_config.top_p = top_p
    generation_config.temperature = temperature
    generation_config.do_sample = True
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = max_generation_length
    generation_config.output_logits = False
    generation_config.return_dict_in_generate = True
    generation_config.top_k = None
    generation_config.pad_token_id = model.tokenizer.pad_token_id

    eos_token_id = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])
    logits_processor = LogitsProcessorList(
        [
            ExpertLogitsProcessor(
                traj_token_offset=model.config.traj_token_start_idx,
                traj_vocab_size=model.config.traj_vocab_size,
            )
        ]
    )
    vlm_outputs = model.vlm.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        stopping_criteria=stopping_criteria,
        logits_processor=logits_processor,
        **tokenized_data,
    )
    sequences = replace_padding_after_eos(
        token_ids=vlm_outputs.sequences,
        eos_token_id=eos_token_id,
        pad_token_id=model.tokenizer.pad_token_id,
    )  # (1, T_total)
    eos_mask = sequences[0] == eos_token_id
    if eos_mask.any():
        eos_pos = int(eos_mask.int().argmax().item())
    else:
        eos_pos = sequences.shape[1] - 1
    return {
        "sequences": sequences,
        "eos_pos": eos_pos,  # position of <|traj_future_start|>
        "prompt_len": input_ids.shape[1],
        "past_key_values": vlm_outputs.past_key_values,
        "rope_deltas": model.vlm.model.rope_deltas,
    }


def denoise_with_cache(model, prompt_cache, rope_deltas, offset, prefix_mask, seed=42):
    """Run the 10-step flow matching denoising on a given cache (b_star=1).

    Returns pred_xyz (1, 64, 3), pred_rot (1, 64, 3, 3).
    """
    device = prefix_mask.device
    n_diffusion_tokens = model.action_space.get_action_space_dims()[0]  # 64
    prefill_seq_len = prompt_cache.get_seq_length()
    position_ids, attention_mask = model._build_expert_pos_ids_and_attn_mask(
        offset=offset,
        rope_deltas=rope_deltas,
        kv_cache_seq_len=prefill_seq_len,
        n_diffusion_tokens=n_diffusion_tokens,
        b_star=1,
        device=device,
        prefix_mask=prefix_mask,
    )
    forward_kwargs = {}
    if model.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False

    def step_fn(x, t):
        # x: (1, 64, 2), t: (1, 1, 1)
        future_token_embeds = model.action_in_proj(x, t)  # (1, 64, 2048)
        if future_token_embeds.dim() == 2:
            future_token_embeds = future_token_embeds.view(1, n_diffusion_tokens, -1)
        expert_out = model.expert(
            inputs_embeds=future_token_embeds,
            position_ids=position_ids,
            past_key_values=prompt_cache,
            attention_mask=attention_mask,
            use_cache=True,
            **forward_kwargs,
        )
        prompt_cache.crop(prefill_seq_len)
        last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]  # (1, 64, 2048)
        return model.action_out_proj(last_hidden).view(-1, *model.action_space.get_action_space_dims())

    torch.cuda.manual_seed_all(seed)
    sampled_action = model.diffusion.sample(
        batch_size=1, step_fn=step_fn, device=device, return_all_steps=False
    )  # (1, 64, 2)
    return sampled_action


# ---------------------------------------------------------------------------
# Phase A: attention statistics hooks
# ---------------------------------------------------------------------------


class VLMStatsCollector:
    """Collects per-head attention statistics from eager attention weights.

    Registered as forward hooks on each VLM text layer's self_attn module.
    Accumulates fp64 sums on CPU; call finalize() for means.
    """

    STAT_KEYS = [
        "mass_vision", "mass_text", "mass_hist", "mass_sink",
        "entropy", "attdist",
    ]

    def __init__(self, n_layers, n_heads, spans, prompt_len, coc_start, coc_end, n_cameras, js_queries=128):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.spans = spans
        self.prompt_len = prompt_len
        self.coc_start = coc_start  # first generated position
        self.coc_end = coc_end      # exclusive, includes <|traj_future_start|>
        self.n_cameras = n_cameras
        self.js_queries = js_queries
        # accumulators: {query_group: {stat: (L, H)}}
        self.sums = {
            g: {k: np.zeros((n_layers, n_heads)) for k in self.STAT_KEYS}
            for g in ("all", "coc")
        }
        self.cam_sums = np.zeros((n_layers, n_heads, n_cameras))  # coc queries only
        self.headnorm = np.zeros((n_layers, n_heads))
        self.js = np.zeros((n_layers, n_heads, n_heads))
        self.cka = np.zeros((n_layers, n_heads, n_heads))
        self.count = 0  # clips
        self._handles = []

    def _region_masks(self, t_total, device):
        # key-region masks padded to full teacher-forced length (T_total,)
        m = {}
        for k in ("vision", "text", "hist", "sink"):
            full = torch.zeros(t_total, dtype=torch.bool, device=device)
            full[: self.prompt_len] = self.spans[k].to(device)
            m[k] = full
        coc = torch.zeros(t_total, dtype=torch.bool, device=device)
        coc[self.coc_start : self.coc_end] = True
        m["text"] = m["text"] | coc  # generated CoC counts as text keys
        if self.spans["per_camera"] is not None:
            cams = torch.zeros(self.n_cameras, t_total, dtype=torch.bool, device=device)
            cams[:, : self.prompt_len] = self.spans["per_camera"].to(device)
            m["cams"] = cams
        else:
            m["cams"] = None
        return m

    def make_hook(self, layer_idx):
        def hook(module, args, kwargs, output):
            attn = output[1]  # (1, H, Tq, Tk) fp32 attention probs (eager)
            assert attn is not None, "eager attention required for stats capture"
            attn = attn[0].float()  # (H, Tq, Tk)
            h, tq, tk = attn.shape
            device = attn.device
            masks = self._region_masks(tk, device)
            qpos = torch.arange(tq, device=device, dtype=torch.float64)
            kpos = torch.arange(tk, device=device, dtype=torch.float64)
            coc_q = torch.zeros(tq, dtype=torch.bool, device=device)
            coc_q[self.coc_start : self.coc_end] = True
            groups = {"all": torch.ones(tq, dtype=torch.bool, device=device), "coc": coc_q}

            attn64 = attn.double()
            # entropy per query, normalized by log(#visible keys) (causal)
            ent = -(attn64.clamp_min(1e-12).log() * attn64).sum(-1)  # (H, Tq)
            denom = torch.log(qpos + 1.0).clamp_min(1.0)  # (Tq,)
            ent_norm = ent / denom
            # mean attention distance per query
            dist = (attn64 * (qpos[None, :, None] - kpos[None, None, :]).abs()).sum(-1)  # (H, Tq)

            for g, qmask in groups.items():
                nq = int(qmask.sum().item())
                if nq == 0:
                    continue
                a = attn64[:, qmask, :]  # (H, nq, Tk)
                s = self.sums[g]
                s["mass_vision"][layer_idx] += a[:, :, masks["vision"]].sum((1, 2)).cpu().numpy() / nq
                s["mass_text"][layer_idx] += a[:, :, masks["text"]].sum((1, 2)).cpu().numpy() / nq
                s["mass_hist"][layer_idx] += a[:, :, masks["hist"]].sum((1, 2)).cpu().numpy() / nq
                s["mass_sink"][layer_idx] += a[:, :, masks["sink"]].sum((1, 2)).cpu().numpy() / nq
                s["entropy"][layer_idx] += ent_norm[:, qmask].mean(1).cpu().numpy()
                s["attdist"][layer_idx] += dist[:, qmask].mean(1).cpu().numpy()
                if g == "coc" and masks["cams"] is not None:
                    for c in range(self.n_cameras):
                        self.cam_sums[layer_idx, :, c] += (
                            a[:, :, masks["cams"][c]].sum((1, 2)).cpu().numpy() / nq
                        )

            # JS divergence between heads over subsampled CoC queries
            qidx = torch.nonzero(coc_q).flatten()
            if len(qidx) > self.js_queries:
                sel = torch.linspace(0, len(qidx) - 1, self.js_queries, device=device).long()
                qidx = qidx[sel]
            a = attn64[:, qidx, :].clamp_min(1e-12)  # (H, nq, Tk)
            js_mat = np.zeros((h, h))
            for i in range(h):
                ai = a[i : i + 1]  # (1, nq, Tk)
                m = 0.5 * (ai + a)  # (H, nq, Tk)
                kl_i = (ai * (ai.log() - m.log())).sum(-1)  # (H, nq)
                kl_j = (a * (a.log() - m.log())).sum(-1)  # (H, nq)
                js_mat[i] = (0.5 * (kl_i + kl_j)).mean(1).cpu().numpy()
            self.js[layer_idx] += js_mat

        return hook

    def make_headout_hook(self, layer_idx, head_dim):
        def hook(module, args):
            x = args[0]  # (1, T, H*D) input to o_proj
            t = x.shape[1]
            xh = x[0].view(t, self.n_heads, head_dim).float()  # (T, H, D)
            self.headnorm[layer_idx] += xh.norm(dim=-1).mean(0).cpu().numpy()
            # linear CKA between heads
            xc = xh - xh.mean(0, keepdim=True)  # (T, H, D)
            cross = torch.einsum("tad,tbe->abde", xc, xc)  # (H, H, D, D)
            hsic = (cross ** 2).sum((-1, -2))  # (H, H)
            diag = hsic.diagonal().clamp_min(1e-12).sqrt()
            cka = hsic / (diag[:, None] * diag[None, :])
            self.cka[layer_idx] += cka.cpu().numpy()
        return hook

    def register(self, model):
        head_dim = model.vlm.config.text_config.head_dim
        for i, layer in enumerate(model.vlm.model.language_model.layers):
            self._handles.append(
                layer.self_attn.register_forward_hook(self.make_hook(i), with_kwargs=True)
            )
            self._handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(self.make_headout_hook(i, head_dim))
            )

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


class ExpertStatsCollector:
    """Per-head stats for the expert during denoising (eager attention)."""

    def __init__(self, n_layers, n_heads, spans, prompt_len, coc_start, coc_end, prefill_len):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.spans = spans
        self.prompt_len = prompt_len
        self.coc_start = coc_start
        self.coc_end = coc_end
        self.prefill_len = prefill_len
        self.sums = {
            k: np.zeros((n_layers, n_heads))
            for k in ("mass_own", "mass_vision", "mass_text", "mass_hist", "mass_sink", "entropy")
        }
        self.headnorm = np.zeros((n_layers, n_heads))
        self.calls = np.zeros(n_layers)  # denoise steps seen
        self._handles = []

    def make_hook(self, layer_idx):
        def hook(module, args, kwargs, output):
            attn = output[1]  # (1, H, 64, Tk)
            assert attn is not None, "eager attention required for expert stats"
            a = attn[0].double()  # (H, 64, Tk)
            tk = a.shape[-1]
            device = a.device
            own = torch.zeros(tk, dtype=torch.bool, device=device)
            own[self.prefill_len :] = True
            masks = {}
            for k in ("vision", "text", "hist", "sink"):
                full = torch.zeros(tk, dtype=torch.bool, device=device)
                full[: self.prompt_len] = self.spans[k].to(device)
                masks[k] = full
            coc = torch.zeros(tk, dtype=torch.bool, device=device)
            coc[self.coc_start : min(self.coc_end, tk)] = True
            masks["text"] = masks["text"] | coc
            s = self.sums
            s["mass_own"][layer_idx] += a[:, :, own].sum((1, 2)).cpu().numpy() / a.shape[1]
            for k in ("vision", "text", "hist", "sink"):
                s[f"mass_{k}"][layer_idx] += a[:, :, masks[k]].sum((1, 2)).cpu().numpy() / a.shape[1]
            ent = -(a.clamp_min(1e-12).log() * a).sum(-1)  # (H, 64)
            s["entropy"][layer_idx] += (ent / np.log(tk)).mean(1).cpu().numpy()
            self.calls[layer_idx] += 1
        return hook

    def make_headout_hook(self, layer_idx, head_dim):
        def hook(module, args):
            x = args[0]  # (1, 64, H*D)
            xh = x[0].view(x.shape[1], self.n_heads, head_dim).float()
            self.headnorm[layer_idx] += xh.norm(dim=-1).mean(0).cpu().numpy()
        return hook

    def register(self, model):
        head_dim = model.expert.config.head_dim
        for i, layer in enumerate(model.expert.layers):
            self._handles.append(
                layer.self_attn.register_forward_hook(self.make_hook(i), with_kwargs=True)
            )
            self._handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(self.make_headout_hook(i, head_dim))
            )

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


# ---------------------------------------------------------------------------
# Phase B: Taylor importance gates
# ---------------------------------------------------------------------------


class HeadGates:
    """Per-head multiplicative gates (value 1.0) on o_proj inputs and v_proj outputs."""

    def __init__(self, layers, n_heads, n_kv_heads, head_dim, device, dtype, gate_v=True):
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.q_gates = [
            torch.ones(n_heads, device=device, dtype=dtype, requires_grad=True) for _ in layers
        ]
        self.v_gates = (
            [torch.ones(n_kv_heads, device=device, dtype=dtype, requires_grad=True) for _ in layers]
            if gate_v
            else None
        )
        self._handles = []
        for i, layer in enumerate(layers):
            self._handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(self._make_q_hook(i))
            )
            if gate_v:
                self._handles.append(
                    layer.self_attn.v_proj.register_forward_hook(self._make_v_hook(i))
                )

    def _make_q_hook(self, i):
        def hook(module, args):
            x = args[0]  # (B, T, H*D)
            b, t, _ = x.shape
            x = x.view(b, t, self.n_heads, self.head_dim) * self.q_gates[i].view(1, 1, -1, 1)
            return (x.view(b, t, -1),)
        return hook

    def _make_v_hook(self, i):
        def hook(module, args, output):
            b, t, _ = output.shape  # (B, T, KV*D)
            out = output.view(b, t, self.n_kv_heads, self.head_dim) * self.v_gates[i].view(1, 1, -1, 1)
            return out.view(b, t, -1)
        return hook

    def q_scores(self):
        # (L, H) |grad| — gate value is 1 so |g * dL/dg| == |dL/dg|
        return np.stack(
            [g.grad.abs().float().cpu().numpy() if g.grad is not None else np.zeros(self.n_heads)
             for g in self.q_gates]
        )

    def v_scores(self):
        return np.stack(
            [g.grad.abs().float().cpu().numpy() if g.grad is not None else np.zeros(self.n_kv_heads)
             for g in self.v_gates]
        )

    def zero_grads(self):
        for g in self.q_gates:
            g.grad = None
        if self.v_gates:
            for g in self.v_gates:
                g.grad = None

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def cache_layer_kv(cache, i):
    """Return (keys, values) tensors of cache layer i across transformers versions."""
    if hasattr(cache, "layers"):
        return cache.layers[i].keys, cache.layers[i].values
    return cache.key_cache[i], cache.value_cache[i]


def set_cache_layer_kv(cache, i, k, v):
    if hasattr(cache, "layers"):
        cache.layers[i].keys = k
        cache.layers[i].values = v
    else:
        cache.key_cache[i] = k
        cache.value_cache[i] = v


def gt_actions(model, data, device):
    """Encode GT future trajectory into normalized action space. Returns (1, 64, 2)."""
    hx = data["ego_history_xyz"][:, 0].to(device)  # (1, 16, 3)
    hr = data["ego_history_rot"][:, 0].to(device)  # (1, 16, 3, 3)
    fx = data["ego_future_xyz"][:, 0].to(device)  # (1, 64, 3)
    fr = data["ego_future_rot"][:, 0].to(device)  # (1, 64, 3, 3)
    action = model.action_space.traj_to_action(hx.float(), hr.float(), fx.float(), fr.float())
    return action  # (1, 64, 2)
