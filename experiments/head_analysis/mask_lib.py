"""Structured pruning masks and unit-selection criteria (Phase P2).

Masks are applied as 0/1 multipliers on the same tensors the Phase P1 gates sat
on, so "masked" is functionally identical to "removed" for these units:
  - Q head      : o_proj input, per head
  - MLP channel : down_proj input, per channel

A KV group is masked by masking the Q heads bound to it. Zeroing K in the cache
would not remove the group -- it would make every score 0 and leave softmax
spreading uniform weight over those positions. Under GQA, group h feeds exactly
VLM Q heads [4h, 4h+4) and expert Q heads [2h, 2h+2), so zeroing those heads'
contribution is the faithful ablation.
"""

import numpy as np
import torch


class PruneMasks:
    """0/1 masks on Q heads and MLP channels of one tower, swappable in place."""

    def __init__(self, layers, n_heads, head_dim, intermediate, device):
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.intermediate = intermediate
        self.q_mask = torch.ones(len(layers), n_heads, device=device)
        self.mlp_mask = torch.ones(len(layers), intermediate, device=device)
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
            m = self.q_mask[i].to(x.dtype)
            return ((x.view(b, t, self.n_heads, self.head_dim) * m.view(1, 1, -1, 1))
                    .view(b, t, -1),)
        return hook

    def _make_mlp_hook(self, i):
        def hook(module, args):
            x = args[0]  # (B, T, intermediate)
            return (x * self.mlp_mask[i].to(x.dtype).view(1, 1, -1),)
        return hook

    def reset(self):
        self.q_mask.fill_(1.0)
        self.mlp_mask.fill_(1.0)

    def set(self, q=None, mlp=None):
        self.reset()
        if q is not None:
            self.q_mask.copy_(torch.as_tensor(q, device=self.q_mask.device, dtype=torch.float32))
        if mlp is not None:
            self.mlp_mask.copy_(
                torch.as_tensor(mlp, device=self.mlp_mask.device, dtype=torch.float32)
            )

    def kept_fraction(self):
        return float(self.q_mask.mean()), float(self.mlp_mask.mean())

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def magnitude_scores(layers, n_heads, head_dim, intermediate):
    """Weight-norm importance, the standard structured-pruning baseline."""
    q = np.zeros((len(layers), n_heads))
    m = np.zeros((len(layers), intermediate))
    with torch.no_grad():
        for i, layer in enumerate(layers):
            w = layer.self_attn.o_proj.weight.float()  # (hidden, H*D)
            q[i] = w.view(w.shape[0], n_heads, head_dim).norm(dim=(0, 2)).cpu().numpy()
            d = layer.mlp.down_proj.weight.float()  # (hidden, intermediate)
            m[i] = d.norm(dim=0).cpu().numpy()
    return q, m


def select_mask(scores, ratio, layers_scope, rng=None):
    """Keep-mask that prunes the lowest-scoring `ratio` of units within each layer.

    Selection is per layer rather than globally so that the depth trend in the
    scores does not silently turn a width sweep into a depth sweep.
    """
    mask = np.ones_like(scores)
    n_units = scores.shape[1]
    k = int(round(n_units * ratio))
    if k == 0:
        return mask
    for li in layers_scope:
        s = rng.random(n_units) if rng is not None else scores[li]
        mask[li, np.argsort(s)[:k]] = 0.0
    return mask


def select_mask_ratios(scores, ratios):
    """Keep-mask with a per-layer prune ratio: ratios[li] of layer li's units are dropped.

    Same within-layer rule as select_mask, but the budget varies by depth, which is what
    separates the layerwise-allocation factor from the within-layer criterion.
    """
    mask = np.ones_like(scores)
    n_units = scores.shape[1]
    for li, r in enumerate(ratios):
        k = round(float(r) * n_units)
        if k > 0:
            mask[li, np.argsort(scores[li])[:k]] = 0.0
    return mask


def kv_group_mask(scores_kv, n_drop, n_heads, layers_scope, rng=None):
    """Keep-mask over Q heads implied by dropping the n_drop weakest KV groups."""
    n_layers, n_kv = scores_kv.shape
    per_group = n_heads // n_kv
    mask = np.ones((n_layers, n_heads))
    for li in layers_scope:
        s = rng.random(n_kv) if rng is not None else scores_kv[li]
        for g in np.argsort(s)[:n_drop]:
            mask[li, g * per_group : (g + 1) * per_group] = 0.0
    return mask
