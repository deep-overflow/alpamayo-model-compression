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

    @staticmethod
    def _signed(gates, size):
        return np.stack([
            g.grad.float().cpu().numpy() if g.grad is not None else np.zeros(size)
            for g in gates
        ])

    def q_signed(self):
        """dL/dg without the abs, for callers that aggregate the sign themselves."""
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


def expert_selffm_grads(model, cache, rope_deltas, seed, prefill, n_steps=10,
                        k_draws=10, coupling="tied", self_xor=0x5E1F):
    """FM gradients anchored on the model's OWN denoised sample instead of the GT.

    The shipped criterion pairs a fresh noise with the GT action; I_CoC meanwhile scores
    the model's own rollout, so only the trajectory half is GT-anchored. The trajectory
    distribution is multimodal and the GT is one mode of it -- on a clip where the model
    produces a different mode the residual v(x_t,t) - (x1_gt - eps) is dominated by mode
    mismatch, and the Taylor score becomes a projection onto "what moves the output
    toward the GT" rather than "what this model uses".

    Here the target is xhat = Denoise_10(eps0), a trajectory the model actually produces.
    `coupling` picks how the loss noise pairs with it:

      tied         x_t = (1-t) eps0 + t xhat,  u = xhat - eps0   (the model's OWN coupling:
                   the ODE maps eps0 to xhat, so this is the only pair the flow produces)
      independent  x_t = (1-t) eps  + t xhat,  u = xhat - eps    (fresh eps per step)

    `tied` was suspected of degenerating -- a perfectly rectified flow would make the
    instantaneous field equal the chord and collapse the residual -- but the probe
    measured it at 29.9% of the GT-anchored loss because Alpamayo's flow is not straight
    (||v - chord||/||chord|| = 0.168 median). See
    plans/2026-09-10_self-anchored-traj-importance.md and outputs/selftraj_probe/.

    K matters more here than on the shipped path: under `tied` a draw's n_steps
    evaluations are n_steps points on ONE chord, so the independent-noise count is
    k_draws, not k_draws * n_steps. The shipped path draws a fresh eps per step and so
    averages n_steps independent problems per clip; k_draws=n_steps restores parity.

    The Euler chain runs under no_grad: xhat is a target, held constant exactly as
    seq_tf is for the CoC half. Gradients accumulate on the detached cache leaves across
    all draws, so the single VLM backward that follows is unchanged.

    Returns (mean loss, [(dL/dk, dL/dv) per layer], leaves), matching expert_fm_grads.
    """
    device = next(model.expert.parameters()).device
    n_layers = len(model.expert.layers)
    leaves = []
    for i in range(n_layers):
        k, v = lib.cache_layer_kv(cache, i)
        leaves.append((k.detach().requires_grad_(True), v.detach().requires_grad_(True)))

    offset = torch.tensor([prefill], device=device)
    prefix_mask = torch.ones(1, prefill, device=device, dtype=torch.long)
    n_tok = model.action_space.get_action_space_dims()[0]  # 64
    dims = model.action_space.get_action_space_dims()  # (64, 2)
    position_ids, attention_mask = model._build_expert_pos_ids_and_attn_mask(
        offset=offset, rope_deltas=rope_deltas, kv_cache_seq_len=prefill,
        n_diffusion_tokens=n_tok, b_star=1, device=device, prefix_mask=prefix_mask,
    )
    forward_kwargs = {}
    if model.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False

    def field(x, t_val, cast=True):
        """cast=False keeps x in fp32 into action_in_proj, which is what the release
        step_fn and expert_infer_grads do; the shipped FM loss casts first. The Euler
        chain therefore runs uncast so xhat is the trajectory the release path would
        produce, while the loss keeps the shipped cast so only the TARGET differs from
        expert_fm_grads. Measured effect of the cast on the sample: 0.005 m mean /
        0.012 m max over 4 clips, against a 1.76 m single-draw distance from GT."""
        # re-seat the leaves: update()+crop() leave the previous step's slices behind
        for i, (k, v) in enumerate(leaves):
            lib.set_cache_layer_kv(cache, i, k, v)
        t = torch.full((1, 1, 1), t_val, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            embeds = model.action_in_proj(x.to(torch.bfloat16) if cast else x, t)
            if embeds.dim() == 2:
                embeds = embeds.view(1, n_tok, -1)
            out = model.expert(
                inputs_embeds=embeds, position_ids=position_ids, past_key_values=cache,
                attention_mask=attention_mask, use_cache=True, **forward_kwargs,
            )
            cache.crop(prefill)
            return model.action_out_proj(out.last_hidden_state[:, -n_tok:])

    dt = 1.0 / n_steps
    losses = []
    for draw in range(k_draws):
        gen0 = torch.Generator(device="cpu").manual_seed((seed ^ self_xor) + draw)
        eps0 = torch.randn(1, *dims, generator=gen0).to(device)
        with torch.no_grad():
            x = eps0.clone()
            for s in range(n_steps):
                x = x + dt * field(x, s * dt, cast=False).float()
        xhat = x.detach()  # (1, 64, 2) -- a target, like seq_tf for the CoC half

        # a separate stream, and one that cannot arithmetically land on gen0's: gen0 is
        # (seed ^ self_xor) + draw over [0, k_draws), so offsetting by k_draws keeps the
        # two windows disjoint whatever the seed
        gen = torch.Generator(device="cpu").manual_seed(seed + k_draws + draw)
        for s in range(n_steps):
            t_val = (s + 0.5) / n_steps
            noise = eps0 if coupling == "tied" else \
                torch.randn(1, *dims, generator=gen).to(device)
            x_t = (1.0 - t_val) * noise + t_val * xhat
            v_target = xhat - noise
            pred = field(x_t, t_val)
            loss = F.mse_loss(pred.float(), v_target)
            loss.backward()
            losses.append(loss.item())

    grads = [(k.grad, v.grad) for k, v in leaves]
    return float(np.mean(losses)), grads, leaves


def best_of_n_target(model, cache, rope_deltas, prefill, gt_xy, hist_xyz, hist_rot,
                     seed, n=6, n_steps=10, self_xor=0x5E1F):
    """The model's own sample that lands closest to the GT, as the flow-matching target.

    The GT anchor pairs a noise with a trajectory the model may never produce; the pure
    self anchor (expert_selffm_grads) puts the target on the model's manifold but throws
    the GT away, and open-loop says that costs +0.0540 m and quadruples CoC degeneracy.
    best-of-n keeps both: draw n chains, decode each, keep the one nearest the GT. The
    target is on-manifold AND chosen by the GT.

    n defaults to 6 because the evaluation protocol is minADE@6 -- the criterion then
    optimises the same "best of six" the metric reads.

    Returns (x1 (1, 64, 2), diagnostics). The caller hands x1 to expert_fm_grads
    UNCHANGED, so the estimator, its fresh-noise-per-step schedule and its |sum_s|
    aggregation stay the shipped ones and the only factor that differs is the target.
    That matters: raising k_draws instead extends the signed sum inside |sum|, which is
    a different statistic rather than the same one with more samples -- measured on
    2026-09-10, the traj map's magnitude grew 8.8x from K=1 to K=10 and the per-clip
    dispersion did not fall at all.
    """
    device = gt_xy.device
    n_tok = model.action_space.get_action_space_dims()[0]  # 64
    dims = model.action_space.get_action_space_dims()  # (64, 2)
    offset = torch.tensor([prefill], device=device)
    prefix_mask = torch.ones(1, prefill, device=device, dtype=torch.long)
    position_ids, attention_mask = model._build_expert_pos_ids_and_attn_mask(
        offset=offset, rope_deltas=rope_deltas, kv_cache_seq_len=prefill,
        n_diffusion_tokens=n_tok, b_star=1, device=device, prefix_mask=prefix_mask,
    )
    forward_kwargs = {}
    if model.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False

    def field(x, t_val):
        t = torch.full((1, 1, 1), t_val, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # fp32 in, as the release step_fn passes it
            embeds = model.action_in_proj(x, t)
            if embeds.dim() == 2:
                embeds = embeds.view(1, n_tok, -1)
            out = model.expert(
                inputs_embeds=embeds, position_ids=position_ids, past_key_values=cache,
                attention_mask=attention_mask, use_cache=True, **forward_kwargs,
            )
            cache.crop(prefill)
            return model.action_out_proj(out.last_hidden_state[:, -n_tok:]).float()

    dt = 1.0 / n_steps
    ades, best, best_ade = [], None, float("inf")
    with torch.no_grad():
        for k in range(n):
            gen = torch.Generator(device="cpu").manual_seed((seed ^ self_xor) + k)
            x = torch.randn(1, *dims, generator=gen).to(device)
            for s in range(n_steps):
                x = x + dt * field(x, s * dt)
            xyz, _ = model.action_space.action_to_traj(
                x, hist_xyz[:, -1].float(), hist_rot[:, -1].float())
            ade = float(torch.norm(xyz[0, :, :2] - gt_xy, dim=-1).mean())
            ades.append(ade)
            if ade < best_ade:
                best_ade, best = ade, x.detach()
    return best, {"ade_of_draws": ades, "ade_best": best_ade,
                  "ade_mean": float(np.mean(ades)), "argbest": int(np.argmin(ades))}


def expert_fm_grads(model, cache, rope_deltas, x1, fm_steps, seed, prefill, k_draws=1,
                    dims=None, wt=None):
    """Run the FM loss backward through the expert onto detached cache leaves.

    Returns the accumulated dL/d(cache) so a single VLM backward can follow, and
    the mean FM loss. Doing it in two stages costs one VLM backward instead of
    fm_steps of them.

    k_draws repeats the whole fm_steps sweep with a fresh noise stream, accumulating on
    the same leaves. k_draws=1 is the shipped path bit-for-bit (draw 0 keeps the original
    seed). It exists so the GT anchor can be given the SAME number of gradient
    evaluations as expert_selffm_grads, whose k_draws x fm_steps sweep would otherwise
    make any stability comparison a comparison of sample counts
    (plans/2026-09-10_self-anchored-traj-importance.md).

    dims restricts the loss to a subset of the action channels -- the action is
    (1, 64, 2) in unicycle space, so dims=[0] and dims=[1] give the two components their
    own importance map. The reduction stays a mean over the kept elements, so a
    single-channel loss is that channel's MSE rather than a halved total; raw magnitudes
    are therefore comparable across dims, and rank_norm makes the selection insensitive
    to the choice either way. dims=None is the shipped path bit-for-bit.

    wt is an optional (n_tok,) weight over the 64 trajectory waypoints -- a different axis
    from the fm_steps denoising axis that znorm11/maxstep11 decompose. The shipped loss
    weights waypoints equally, which under-weights early actions relative to their effect
    on the path: an action error at step t displaces every position after t, so its
    position-space influence goes as (n_tok - t). Closed loop points the same way -- it
    executes 0.1 s per plan and only ~1.9 s of the 6.4 s is realised at all
    (plans/2026-09-11_effective-plan-horizon.md). wt=None is the shipped path.
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

    losses = []
    for draw in range(k_draws):
        gen = torch.Generator(device="cpu").manual_seed(seed + draw)
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
            p_, t_ = pred.float(), v_target
            if dims is not None:
                idx = torch.as_tensor(dims, device=device, dtype=torch.long)
                p_, t_ = p_.index_select(-1, idx), t_.index_select(-1, idx)
            if wt is None:
                loss = F.mse_loss(p_, t_)
            else:
                # weighted over the 64 trajectory waypoints. Normalised to sum to n_tok so
                # the loss keeps the scale of the unweighted one -- the gate score is
                # |sum_s dL/dg| and rank_norm is scale-invariant, but the reported fm_loss
                # would otherwise not be comparable across arms.
                w = wt.to(p_.device).view(1, -1, 1)
                loss = ((p_ - t_) ** 2 * w).sum() / (w.expand_as(p_).sum())
            loss.backward()
            losses.append(loss.item())

    grads = [(k.grad, v.grad) for k, v in leaves]
    return float(np.mean(losses)), grads, leaves


# ---------------------------------------------------------------------------
# Denoising-step decomposition (2026-08-21)
#
# The shipped expert criterion accumulates ten backwards into one gate grad and reads
# |sum_s dL_s/dg|, while the clip axis accumulates sum_clips |.|. The two axes therefore
# aggregate by different rules, and steps of opposite sign cancel on the step axis only.
# Everything below keeps the step axis so that asymmetry is measurable, not assumed.
# ---------------------------------------------------------------------------


class StepGates:
    """Per-denoising-step gates on one tower, for a single backward through the chain.

    One leaf per layer carrying a leading step axis, plus a mutable `step` the hooks read.
    Running the Euler loop with grad enabled and bumping `step` each iteration means one
    backward from the final loss yields dL/dg separately for every step, which is what
    "how much does unit u matter AT step s, for the final trajectory" requires. UnitGates
    cannot express this: it holds one gate per unit for the whole rollout.
    """

    def __init__(self, layers, n_heads, head_dim, intermediate, n_steps, device, dtype):
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.intermediate = intermediate
        self.n_steps = n_steps
        self.step = 0
        self.q_gates = [
            torch.ones(n_steps, n_heads, device=device, dtype=dtype, requires_grad=True)
            for _ in layers
        ]
        self.mlp_gates = [
            torch.ones(n_steps, intermediate, device=device, dtype=dtype, requires_grad=True)
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
            g = self.q_gates[i][self.step].view(1, 1, -1, 1)
            x = x.view(b, t, self.n_heads, self.head_dim) * g
            return (x.view(b, t, -1),)
        return hook

    def _make_mlp_hook(self, i):
        def hook(module, args):
            x = args[0]  # (B, T, intermediate)
            return (x * self.mlp_gates[i][self.step].view(1, 1, -1),)
        return hook

    @staticmethod
    def _grads(gates, n_steps, size):
        # (L, S, U) -> (S, L, U); the gate value is 1, so g * dL/dg == dL/dg
        g = np.stack([
            x.grad.float().cpu().numpy() if x.grad is not None else np.zeros((n_steps, size))
            for x in gates
        ])
        return g.transpose(1, 0, 2)

    def q_grads(self):
        return self._grads(self.q_gates, self.n_steps, self.n_heads)  # (S, L, H)

    def mlp_grads(self):
        return self._grads(self.mlp_gates, self.n_steps, self.intermediate)  # (S, L, I)

    def zero_grads(self):
        for g in self.q_gates + self.mlp_gates:
            g.grad = None

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def expert_fm_grads_stepwise(model, cache, rope_deltas, x1, fm_steps, seed, prefill,
                             gates, noise_mode="shared"):
    """Per-step flow-matching gradients on the expert gates, signed.

    Identical construction to expert_fm_grads -- same t grid ((s+0.5)/S), same
    x_t = (1-t) eps + t x1, same target x1 - eps -- except the gate grads are read and
    zeroed after every step. The shipped path never zeroes them, so its score is
    |sum_s dL_s/dg| and steps of opposite sign cancel; keeping the signed per-step term
    lets that cancellation be measured, and any other step aggregation be built.

    noise_mode:
      "per_step"  a fresh eps per step, exactly as expert_fm_grads draws it -- summing the
                  returned grads over s then reproduces the shipped score, the integrity check.
      "shared"    one eps for the whole t grid, so step-to-step differences are paired and
                  carry no noise-draw variance. The analysis default.

    Returns (losses (S,), q (S,L,H), mlp (S,L,I), kv_k (S,L,KV), kv_v (S,L,KV)).
    kv_* use the same within-step form as kv_group_scores (|k * dL/dk| summed over
    positions) on detached cache leaves, so no VLM backward is needed.
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
    shared = torch.randn(x1.shape, generator=gen).to(device) if noise_mode == "shared" else None

    n_kv = leaves[0][0].shape[1]
    losses, q_s, mlp_s, kv_k_s, kv_v_s = [], [], [], [], []
    gates.zero_grads()
    for s in range(fm_steps):
        t_val = (s + 0.5) / fm_steps
        noise = shared if shared is not None \
            else torch.randn(x1.shape, generator=gen).to(device)  # (1, 64, 2)
        x_t = (1.0 - t_val) * noise + t_val * x1
        v_target = x1 - noise  # (1, 64, 2)
        t = torch.full((1, 1, 1), t_val, device=device)
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
        q_s.append(gates.q_signed())
        mlp_s.append(gates.mlp_signed())
        kk = np.zeros((n_layers, n_kv))
        vv = np.zeros((n_layers, n_kv))
        with torch.no_grad():
            for i, (k, v) in enumerate(leaves):
                if k.grad is not None:
                    kk[i] = (k * k.grad).abs().sum((0, 2, 3)).float().cpu().numpy()
                if v.grad is not None:
                    vv[i] = (v * v.grad).abs().sum((0, 2, 3)).float().cpu().numpy()
        kv_k_s.append(kk)
        kv_v_s.append(vv)

        gates.zero_grads()
        for k, v in leaves:
            k.grad = None
            v.grad = None

    del leaves
    return (np.array(losses), np.stack(q_s), np.stack(mlp_s),
            np.stack(kv_k_s), np.stack(kv_v_s))


def vlm_backward_from_cache(cache_t, seed_grads, retain=True):
    """Seed the VLM graph with a per-layer cache gradient and backprop through it.

    This is the operation run_importance performs exactly once, with seed
    sum_s dL_s/dcache, which is why the shipped VLM trajectory score is |sum_s dL_s/dg|.
    Backprop is linear in the seed, so calling this per step recovers the step axis, and
    calling it once with sum_s w_s (dL_s/dcache) returns sum_s w_s (dL_s/dg) for any fixed
    weighting -- one backward instead of ten.

    Caller owns zeroing: gate .grad and cache_t .grad both accumulate across calls.
    """
    ts, gs = [], []
    for (k, v), (gk, gv) in zip(cache_t, seed_grads):
        for t_, g_ in ((k, gk), (v, gv)):
            if g_ is not None:
                ts.append(t_)
                gs.append(g_.to(t_.dtype))
    torch.autograd.backward(ts, gs, retain_graph=retain)


def expert_step_cache_grads(model, cache, rope_deltas, x1, s, fm_steps, prefill,
                            leaves, noise):
    """One flow-matching step's gradient onto the detached cache leaves. Returns the loss.

    Split out of expert_fm_grads so the caller can interleave a VLM backward after each
    step instead of accumulating ten of them: holding all ten cache-gradient sets at once
    peaked at 45.0 GB on a 47.4 GB card, while one set at a time stays at the ~41 GB the
    shipped pass already uses.
    """
    device = x1.device
    n_tok = model.action_space.get_action_space_dims()[0]  # 64
    offset = torch.tensor([prefill], device=device)
    prefix_mask = torch.ones(1, prefill, device=device, dtype=torch.long)
    position_ids, attention_mask = model._build_expert_pos_ids_and_attn_mask(
        offset=offset, rope_deltas=rope_deltas, kv_cache_seq_len=prefill,
        n_diffusion_tokens=n_tok, b_star=1, device=device, prefix_mask=prefix_mask,
    )
    forward_kwargs = {}
    if model.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False

    t_val = (s + 0.5) / fm_steps
    x_t = (1.0 - t_val) * noise + t_val * x1
    v_target = x1 - noise  # (1, 64, 2)
    t = torch.full((1, 1, 1), t_val, device=device)
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
    return float(loss.item())


def expert_infer_grads_stepwise(model, cache, rope_deltas, prefill, step_gates, gt_xy,
                                hist_xyz, hist_rot, seed, n_steps=10, x0=None):
    """Per-step gradients along the model's OWN Euler trajectory, w.r.t. the final path.

    The shipped criterion measures on the training path: x_t = (1-t) eps + t x1, a point on
    the straight line to the GT action, at t = 0.05 .. 0.95. Inference walks a different
    path -- x_{i+1} = x_i + dt v(x_i), t = 0.0 .. 0.9, one noise draw -- whose iterates carry
    the model's own accumulated error. This runs that loop with grad on and a separate gate
    per step, so one backward from MSE(pred_xy, gt_xy) gives d(final path error)/d(gate at
    step s): the deployment-relevant score.

    FlowMatching.sample is decorated @torch.no_grad, so its Euler loop is reproduced here.
    The prefill cache is re-seated from detached leaves each step, so the graph holds only
    the expert's own contribution rather than ten nested cats of the VLM cache.

    x0 overrides the initial noise, which is what lets the reimplementation be checked
    against the official sampler on identical input (see verify_step_euler.py); left None
    it is drawn from a CPU generator so the draw does not depend on the GPU architecture.

    Returns (loss, q (S,L,H), mlp (S,L,I), action (1,64,2)); the grads are signed.
    """
    device = gt_xy.device
    n_layers = len(model.expert.layers)
    leaves = []
    for i in range(n_layers):
        k, v = lib.cache_layer_kv(cache, i)
        leaves.append((k.detach(), v.detach()))

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
    if x0 is None:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        x = torch.randn(1, *dims, generator=gen).to(device)
    else:
        x = x0.to(device)
    dt = 1.0 / n_steps

    step_gates.zero_grads()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(n_steps):
            step_gates.step = s
            t = torch.full((1, 1, 1), s * dt, device=device)
            for i, (k, v) in enumerate(leaves):
                lib.set_cache_layer_kv(cache, i, k, v)
            # x stays fp32 here, as the release step_fn and denoise_with_cache pass it --
            # autocast casts inside action_in_proj's Linears, and pre-casting to bf16
            # instead cost ~1% of the final trajectory (verify_step_euler.py). The
            # training-path twin above does cast, matching the shipped expert_fm_grads.
            embeds = model.action_in_proj(x, t)  # (1, 64, 2048)
            if embeds.dim() == 2:
                embeds = embeds.view(1, n_tok, -1)
            out = model.expert(
                inputs_embeds=embeds, position_ids=position_ids, past_key_values=cache,
                attention_mask=attention_mask, use_cache=True, **forward_kwargs,
            )
            cache.crop(prefill)
            v_field = model.action_out_proj(out.last_hidden_state[:, -n_tok:]).view(1, *dims)
            x = x + dt * v_field.float()

    pred_xyz, _ = model.action_space.action_to_traj(
        x.float(), hist_xyz[:, -1].float(), hist_rot[:, -1].float()
    )  # (1, 64, 3)
    loss = F.mse_loss(pred_xyz[0, :, :2], gt_xy)
    loss.backward()
    del leaves
    return (float(loss.item()), step_gates.q_grads(), step_gates.mlp_grads(),
            x.detach())
