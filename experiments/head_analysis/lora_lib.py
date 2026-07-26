"""LoRA recovery for slim (physically-pruned) Alpamayo-1.5 checkpoints.

Design B (cache attached): a single grad VLM forward (use_cache=True) produces a KV
cache that still requires grad, the expert runs a flow-matching step on it, and one
backward sends the trajectory gradient through expert LoRA AND -- via the cache -- VLM
LoRA. This directly recovers the trajectory damage that VLM pruning inflicts through the
cache (the "cache is the interface" thesis), unlike phase21 which detaches the cache and
trains VLM only via CoC.

LoRA is attached with peft to the surviving q/k/v/o + gate/up/down of both towers'
decoder layers. The slim surgery only resized these nn.Linears (in-place Parameter swap),
so peft's name-based targeting wraps them transparently; the MethodType-bound slim
attention forward keeps calling self.q_proj(...) and picks up the wrapper. Layer-35
kv-only (integrated) has q_proj/o_proj = None -> skipped by name; its k/v_proj are still
wrapped (they produce the cache the expert reads -> a recovery lever).

After training, merge_and_unload folds W += BA back into the slim-shaped weights, so the
graphed latency (1.62x / 1.43x) and every eval path apply to the recovered model unchanged.
"""

import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).parent))
import slim_lib as sl  # noqa: E402

# phase21 ships the convention-correct flow-matching / CoC training losses
_PH21 = "/workspace/alpamayo1.5/experiments/phase21_kv_selfrecon"
sys.path.insert(0, _PH21)
import losses as ph21  # noqa: E402  flow_matching_loss, sample_flow_inputs, coc_ce_loss

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
    """slim checkpoint -> peft LoRA-wrapped. Returns (peft_model, base, meta)."""
    import json
    base = sl.load_slim(ckpt_dir, device=device)
    meta = json.loads((Path(ckpt_dir) / "slim_meta.json").read_text())
    # base weights already frozen (requires_grad_(False) in load_slim). Make embeddings
    # emit grad so early-layer cache tensors sit on a live graph even where LoRA is thin.
    base.vlm.enable_input_require_grads()
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                     target_modules=lora_target_modules(base))
    peft_model = get_peft_model(base, cfg)
    return peft_model, base, meta


def param_summary(peft_model):
    total = sum(p.numel() for p in peft_model.parameters())
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    return trainable, total


def merge_save(peft_model, base, meta, out_dir):
    """Fold LoRA into the slim weights and re-save in slim_lib format (shape unchanged)."""
    peft_model.merge_and_unload()  # W += BA, adapters removed, base modules restored
    sl.check_slim(base)  # every param still bf16 + contiguous after the merge
    return sl.save_slim(base, meta, out_dir)
