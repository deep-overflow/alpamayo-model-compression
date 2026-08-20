"""KI-LoRA recovery for slim (physically-pruned) Alpamayo-1.5 checkpoints.

Knowledge Insulation (pi0.5): one VLM forward over the teacher-forced sequence
[prompt + CoC + cot_end + traj_future_start] (use_cache=True) yields both the CoC logits
and the KV cache. CE on the CoC span is the only loss that reaches VLM LoRA. The cache is
then detached layer-by-layer, so the expert's flow-matching loss trains expert LoRA alone
-- the reasoning channel is supervised in language space, the action channel adapts to
the cache it reads, and the trajectory gradient never touches the VLM. This is the
opposite of the 2026-07 Design B trainer (train_lora.py), which kept the cache attached
on purpose; that path needed phase21 losses from the deleted sibling repo and never ran.

FM convention matches prune_lib.expert_fm_grads: x_t = (1-t)*noise + t*x1,
v_target = x1 - noise, one random t ~ U(0,1) per micro-step.

LoRA is attached with peft to the surviving q/k/v/o + gate/up/down of both towers'
decoder layers. The slim surgery only resized these nn.Linears in place, so peft's
name-based targeting wraps them transparently; the MethodType-bound slim attention
forward keeps calling self.q_proj(...) and picks up the wrapper. After training,
merge_and_unload folds W += BA back into the slim-shaped weights, so every eval path
(open-loop runners, alpasim driver) applies to the recovered model unchanged.
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "head_analysis"))
import analysis_lib as lib
import slim_lib as sl

# token ids of <cot_end> / <traj_future_start> (same as run_baseline.py)
COT_END, TFS = 155678, 155681

LORA_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def lora_target_modules(model):
    """Every surviving q/k/v/o + gate/up/down under the two towers' decoder layers."""
    targets = []
    for name, mod in model.named_modules():
        if mod is None or not name.endswith(LORA_SUFFIXES):
            continue
        if "language_model.layers." in name or "expert.layers." in name:
            targets.append(name)
    return targets


def load_slim_lora(ckpt_dir, r=32, alpha=64, dropout=0.0, device="cuda"):
    """slim checkpoint -> peft LoRA-wrapped. Returns (peft_model, base, meta).

    Unlike the Design B loader this does NOT enable input grads: with the cache detached,
    the only gradient path into the VLM is CE -> lm_head -> hidden states -> LoRA params,
    which needs no grad on the (frozen) embedding output.
    """
    base = sl.load_slim(ckpt_dir, device=device)
    meta = json.loads((Path(ckpt_dir) / "slim_meta.json").read_text())
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                     target_modules=lora_target_modules(base))
    peft_model = get_peft_model(base, cfg)
    return peft_model, base, meta


def param_summary(peft_model):
    total = sum(p.numel() for p in peft_model.parameters())
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    return trainable, total


def coc_ids(tok, text, max_tokens=256):
    """CoC continuation ids for one reasoning text: tokenize + [cot_end, traj_future_start].

    Same encoding as run_baseline.gt_coc_seq, so the FM prefill context here matches the
    context every teacher-forced evaluation uses. Truncation keeps the two terminals.
    """
    rt = tok(str(text), add_special_tokens=False)["input_ids"]
    return rt[: max_tokens - 2] + [COT_END, TFS]


def vlm_forward(base, inputs, seq_tf):
    """Grad VLM forward over the teacher-forced sequence. Returns (out, cache, rope_deltas)."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = base.vlm.model(
            input_ids=seq_tf, attention_mask=torch.ones_like(seq_tf),
            pixel_values=inputs["tokenized_data"]["pixel_values"],
            image_grid_thw=inputs["tokenized_data"]["image_grid_thw"], use_cache=True,
        )
    return out, out.past_key_values, out.rope_deltas


def ce_loss(base, out, seq_tf, coc_start, coc_end):
    """CE over the CoC span (cot_end/traj_future_start included -> learns to terminate)."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = base.vlm.lm_head(out.last_hidden_state[:, coc_start - 1 : coc_end - 1])
    return F.cross_entropy(logits[0].float(), seq_tf[0, coc_start:coc_end])


def detach_cache_(model, cache):
    """Replace every layer's K/V with detached views -- the insulation boundary."""
    for i in range(len(model.expert.layers)):
        k, v = lib.cache_layer_kv(cache, i)
        lib.set_cache_layer_kv(cache, i, k.detach(), v.detach())


def fm_loss_insulated(model, cache, rope_deltas, x1, prefill, t_val, noise):
    """One flow-matching step on the detached cache. Gradient reaches expert LoRA only.

    Mirrors prune_lib.expert_fm_grads' forward exactly, minus the requires_grad leaves:
    here dL/d(cache) is deliberately never formed. The caller draws t_val and noise.
    """
    device = x1.device
    detach_cache_(model, cache)
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

    x_t = (1.0 - t_val) * noise + t_val * x1  # (1, 64, 2)
    v_target = x1 - noise  # (1, 64, 2)
    t = torch.full((1, 1, 1), t_val, device=device)
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
    return F.mse_loss(pred.float(), v_target)


def adapter_state(peft_model):
    return {k: v.detach().cpu().clone() for k, v in peft_model.state_dict().items()
            if "lora_" in k}


def save_adapter(peft_model, path, extra=None):
    """Adapter-only checkpoint (~0.3 GB): the recipes-style substitute for 17 GB merges."""
    payload = {"lora_state": adapter_state(peft_model), "extra": extra or {}}
    torch.save(payload, path)


def load_adapter_(peft_model, path, device="cuda"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    sd = peft_model.state_dict()
    missing = [k for k in payload["lora_state"] if k not in sd]
    assert not missing, f"adapter keys not in model: {missing[:3]}"
    sd.update({k: v.to(device) for k, v in payload["lora_state"].items()})
    peft_model.load_state_dict(sd)
    return payload["extra"]


def merge_save(peft_model, base, meta, out_dir, write_state=True):
    """Fold LoRA into the slim weights and re-save in slim_lib format (shape unchanged)."""
    peft_model.merge_and_unload()  # W += BA, adapters removed, base modules restored
    sl.check_slim(base)  # every param still bf16 + contiguous after the merge
    return sl.save_slim(base, meta, out_dir, write_state=write_state)
