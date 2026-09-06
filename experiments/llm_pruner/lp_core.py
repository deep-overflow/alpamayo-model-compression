# Ported from soowon's alpamayo1.5 research tree
#   /home/cvlab21/project/soowon/vla-ad/alpamayo1.5/experiments/llm_pruner/lp_core.py
# copied 2026-09-06 to run the second-order (param_mix) arm that was never executed there.
# That tree had itself vendored analysis_lib / sample_cache / expert_per_clip / slim_lib
# FROM this repo on 2026-08-09; those copies are deliberately NOT brought back -- this
# repo's versions are newer (expert_per_clip carries an AcceleratorError fix, slim_lib a
# pinned MODEL_REV and write_state, sample_cache calib_samples()), and this code was
# written against them in the first place.
# Changes from the original are marked `PORT:`; the importance formulas are otherwise
# untouched, because upstream fidelity is the whole point of this file.
"""LLM-Pruner (arXiv:2305.11627) stages 1-2, ported to the Alpamayo 1.5 VLM text tower.

Stage 1 -- Discovery.  `hf_prune.py --block_wise` roots a dependency group at each
`layers[i].self_attn.q_proj` and `layers[i].mlp.gate_proj` and lets the tracer close it
transitively.  For a LLaMA/Qwen block that yields

    ATTN(i) = {q_proj.out, k_proj.out, v_proj.out, o_proj.in}      # + reshape/expand dummies
    MLP(i)  = {gate_proj.out, up_proj.out, down_proj.in}

and terminates at `o_proj`/`down_proj` because `Linear.prune_in_channels is not
Linear.prune_out_channels`, so `hidden_size` is never touched.  We reproduce that table
directly instead of running the vendored tracer against the live model, because on
Qwen3-VL the tracer is provably wrong (class-keyed `customized_pruners` hands 4096-space
indices to the 128-element `q_norm`/`k_norm`; the GQA `expand` node does not exist under
flash-attention-2 or `sdpa(enable_gqa=True)`).  `trace_check.py` runs the real tracer on a
2-layer CPU meta-config and diffs it against this table.

Two deliberate deviations from upstream, both forced by the action expert reading the
VLM's per-layer KV cache verbatim (see docs/llm_pruner.md §1):

  * `k_proj.out` and `v_proj.out` are dropped from ATTN(i).  Upstream would keep them
    (and `llama3.py` roots the group at `k_proj`), but their out-features ARE the cache
    width.  Upstream's own mechanism for this is post-filtering `group._group`;
    `ignored_layers` cannot trim a member, it kills the whole group.
  * Dropping k/v also removes upstream's GQA importance bug: with members of length 4096
    and 1024, `hf_llama_pruner.py:335` folds `imp.view(4, 1024).sum(0)`, which aggregates
    q-head h with h+8, h+16, h+24 -- four *different* kv groups.  With k/v filtered out
    every member is 4096 long and no folding happens.

Stage 2 -- Estimation.  `group_importance()` is formula-identical to
`LLMPruner/pruner/hf_llama_pruner.py::TaylorImportance` (all four `--taylor` variants, all
six `--grouping_strategy` reductions) and `magnitude_importance()` to
`MagnitudeImportance`.  The head fold `imp.view(-1, head_dim).sum(1)` lives in upstream's
MetaPruner, not in the importance object; it is applied here for the ATTN groups.

The gradient the formulas consume is accumulated by `GradAccumulator` over the whole
calibration set before any score is computed -- upstream backprops one batch of 10
examples in a single pass, so the abs must come after the sum over examples, which means
the elementwise sum has to be materialised.  6.6 B fp32 does not fit next to a 22 GB model
on a 49 GB card, hence the CPU accumulator and the post-accumulate hook that frees each
weight's `.grad` the instant it has been consumed.
"""

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Stage 1: the coupled structures
# ---------------------------------------------------------------------------

# (module path relative to the decoder layer, which axis of .weight is the group axis)
#   "out" -> rows    (weight[c, :])  = Linear out-channels
#   "in"  -> columns (weight[:, c])  = Linear in-channels
ATTN_MEMBERS = (("self_attn.q_proj", "out"), ("self_attn.o_proj", "in"))
MLP_MEMBERS = (("mlp.gate_proj", "out"), ("mlp.up_proj", "out"), ("mlp.down_proj", "in"))

# Never in any group: their out-features are the expert's KV interface, or they are
# per-head_dim norms, or they live on the hidden dim.
FROZEN_MEMBERS = (
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.q_norm",
    "self_attn.k_norm",
    "input_layernorm",
    "post_attention_layernorm",
)

PRUNABLE_TENSORS = tuple(name for name, _ in ATTN_MEMBERS + MLP_MEMBERS)

TAYLOR_VARIANTS = ("param_first", "param_second", "param_mix", "vectorize")
GROUP_REDUCTIONS = ("sum", "mean", "max", "prod", "first", "second")


def get_submodule(layer, path):
    obj = layer
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def text_layers(model):
    return model.vlm.model.language_model.layers


def text_geometry(model):
    tc = model.vlm.config.text_config
    return {
        "n_layers": tc.num_hidden_layers,
        "n_heads": tc.num_attention_heads,
        "n_kv_heads": tc.num_key_value_heads,
        "head_dim": tc.head_dim,
        "hidden": tc.hidden_size,
        "intermediate": tc.intermediate_size,
    }


def layer_scope(n_layers, start, end):
    """LLM-Pruner's --block_*_layer_start/end, a half-open range of decoder indices.

    Upstream's scripts/llama_prune.sh uses 4/30 on 32 layers: skip the first four and the
    last two.  The same pattern on 36 layers is 4/34.
    """
    scope = list(range(start, end))
    assert scope and scope[0] >= 0 and scope[-1] < n_layers, (start, end, n_layers)
    return scope


def group_param_cost(geom):
    """Params removed per pruned group, for the ratio -> parameter-reduction table."""
    return {
        "attn": 2 * geom["hidden"] * geom["head_dim"],  # q_proj rows + o_proj cols
        "mlp": 3 * geom["hidden"],  # gate/up rows + down col
    }


# ---------------------------------------------------------------------------
# Stage 2a: gradient accumulation
# ---------------------------------------------------------------------------


class GradAccumulator:
    """Elementwise sum (and optionally sum of squares) of weight gradients over clips.

    `register_post_accumulate_grad_hook` fires as soon as a parameter's `.grad` exists, so
    the grad is copied out and dropped mid-backward: GPU gradient residency stays at about
    one tensor (~100 MB) instead of the 13 GB the full text tower would hold.

    One accumulator owns the buffers for *all* objectives and routes each backward by
    `self.active`.  Registering a second accumulator on the same parameters would not
    work: the hooks fire in registration order and the first one nulls `.grad`, so the
    second silently records nothing while the first absorbs every objective.
    """

    def __init__(self, model, scope, objectives=("coc",), second_order=False, device="cpu"):
        self.scope = list(scope)
        self.objectives = tuple(objectives)
        self.device = device
        self.second_order = second_order
        self.active = None
        self.n_samples = {o: 0 for o in self.objectives}
        self._handles = []
        self.grad = {o: {} for o in self.objectives}
        self.grad_sq = {o: {} for o in self.objectives} if second_order else None
        layers = text_layers(model)
        for li in self.scope:
            for mname in PRUNABLE_TENSORS:
                p = get_submodule(layers[li], mname).weight
                p.requires_grad_(True)
                key = (li, mname)
                for o in self.objectives:
                    self.grad[o][key] = torch.zeros(tuple(p.shape), dtype=torch.float32,
                                                    device=device)
                    if second_order:
                        self.grad_sq[o][key] = torch.zeros(tuple(p.shape),
                                                           dtype=torch.float32, device=device)
                self._handles.append(
                    p.register_post_accumulate_grad_hook(self._make_hook(key))
                )

    def _make_hook(self, key):
        def hook(param):
            g = param.grad
            if g is None:
                return
            if self.active is None:
                raise RuntimeError("gradient produced with no active objective set")
            # move in the parameter dtype (bf16 = half the PCIe traffic), widen on the
            # far side; bf16 -> fp32 is exact.
            g_local = g.detach().to(self.device, copy=True).float()
            self.grad[self.active][key].add_(g_local)
            if self.grad_sq is not None:
                self.grad_sq[self.active][key].add_(g_local * g_local)
            param.grad = None

        return hook

    def begin(self, objective):
        assert objective in self.objectives, objective
        self.active = objective

    def end(self):
        self.n_samples[self.active] += 1
        self.active = None

    def nbytes(self):
        n = sum(t.numel() for d in self.grad.values() for t in d.values())
        if self.grad_sq is not None:
            n += sum(t.numel() for d in self.grad_sq.values() for t in d.values())
        return n * 4

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def state(self, objective):
        out = {f"g|{li}|{m}": t.numpy() for (li, m), t in self.grad[objective].items()}
        if self.grad_sq is not None:
            out.update({f"g2|{li}|{m}": t.numpy()
                        for (li, m), t in self.grad_sq[objective].items()})
        out["n_samples"] = np.asarray(self.n_samples[objective])
        return out


# ---------------------------------------------------------------------------
# Stage 2b: importance
# ---------------------------------------------------------------------------


def _salience(w, g, g2, taylor, n_samples, first_order="mean"):
    """LLMPruner/pruner/hf_llama_pruner.py:266-271, elementwise.

    PORT: `first_order` is new, and it matters only for `param_mix`.

    Upstream runs TWO passes (`_upstream_reference/hf_prune.py:128-148`): a per-example
    loop that accumulates `acc_grad = Σ_j g_j² / N`, and then ONE batched backward whose
    `.grad` is the mean-loss gradient, i.e. `g_batch ≈ (1/N) Σ_j g_j`. So upstream's two
    terms are both **per-sample averages** and `param_mix = w·g_batch − ½·w²·acc_grad`
    balances them correctly.

    `GradAccumulator` here instead sums the signed per-clip gradients (deliberately -- it
    reproduces the batched backward's direction without holding N gradients), so `g` is
    `Σ_j g_j = N · g_batch`. Feeding that to the upstream expression makes the first-order
    term N times too large; at N=100 the second-order correction is numerically invisible
    and `param_mix` silently collapses onto `param_first`. `first_order="mean"` divides it
    back out, which is what upstream actually computes.

    `param_first` and `param_second` are unaffected in *ranking* either way: the whole
    pipeline below (abs-sum over a channel, sum over group members, sum over head_dim) is
    homogeneous of degree 1 in the salience, so a global factor cannot reorder anything.
    Only `group_reduction="prod"` breaks that, and it is not the default. Use
    `first_order="sum"` to reproduce the stored arrays of the original tree bit for bit.
    """
    g1 = g / n_samples if first_order == "mean" else g
    if taylor == "param_second":
        return w * (g2 / n_samples) * w
    s = w * g1
    if taylor == "param_mix":
        s = s - 0.5 * w * (g2 / n_samples) * w
    return s


def _channel_reduce(salience, axis_kind, taylor):
    """hf_llama_pruner.py:274-292 -- element-wise (`param_*`) vs vector-wise (`vectorize`)."""
    dim = 1 if axis_kind == "out" else 0
    if taylor == "vectorize":
        return salience.sum(dim=dim).abs()
    if "param" in taylor:
        return salience.abs().sum(dim=dim)
    raise NotImplementedError(taylor)


def _reduce_group(stacked, group_reduction):
    """hf_llama_pruner.py:221-238."""
    if group_reduction == "sum":
        return stacked.sum(dim=0)
    if group_reduction == "mean":
        return stacked.mean(dim=0)
    if group_reduction == "max":
        return stacked.max(dim=0)[0]
    if group_reduction == "prod":
        return torch.prod(stacked, dim=0)
    if group_reduction == "first":
        return stacked[0]
    if group_reduction == "second":
        return stacked[1]
    raise NotImplementedError(group_reduction)


class WeightCache:
    """fp32 CPU mirror of the prunable weights.

    Every criterion arm needs the same `w` tensors; without this each arm re-copies
    ~5.5 B params off the GPU and re-widens them, which dominates the score pass.
    """

    def __init__(self):
        self._d = {}

    def get(self, layers, li, mname):
        key = (li, mname)
        if key not in self._d:
            self._d[key] = get_submodule(layers[li], mname).weight.detach().to("cpu").float()
        return self._d[key]

    def nbytes(self):
        return sum(t.numel() for t in self._d.values()) * 4


def group_importance(model, accum, objective, taylor="param_first", group_reduction="sum",
                     weights=None, first_order="mean"):
    """Grouped Taylor importance for every ATTN and MLP group in the accumulator's scope.

    Returns (q_scores (L, n_heads), mlp_scores (L, intermediate)) as float64 numpy, zero
    outside the scope.  Computed on CPU: the accumulated gradient lives there and one
    weight at a time is copied over, which keeps this off the GPU entirely.
    """
    geom = text_geometry(model)
    layers = text_layers(model)
    weights = weights if weights is not None else WeightCache()
    n = max(accum.n_samples[objective], 1)
    grad = accum.grad[objective]
    grad_sq = accum.grad_sq[objective] if accum.grad_sq is not None else None
    q_scores = np.zeros((geom["n_layers"], geom["n_heads"]), dtype=np.float64)
    mlp_scores = np.zeros((geom["n_layers"], geom["intermediate"]), dtype=np.float64)

    if taylor in ("param_second", "param_mix") and grad_sq is None:
        raise ValueError(f"{taylor} needs a second-order accumulator")

    for li in accum.scope:
        for members, out_arr, fold_heads in (
            (ATTN_MEMBERS, q_scores, True),
            (MLP_MEMBERS, mlp_scores, False),
        ):
            local = []
            for mname, axis_kind in members:
                w = weights.get(layers, li, mname)
                g = grad[(li, mname)]
                g2 = grad_sq[(li, mname)] if grad_sq is not None else None
                s = _salience(w, g, g2, taylor, n, first_order)
                local.append(_channel_reduce(s, axis_kind, taylor))
                del s
            # Upstream length-aligns members here; with k/v filtered out of ATTN every
            # member already has the same length, so the fold is a no-op. Assert it.
            assert len({t.numel() for t in local}) == 1, [t.numel() for t in local]
            imp = _reduce_group(torch.stack(local, dim=0), group_reduction)
            if fold_heads:  # metapruner.py:263-264, head score = sum over its head_dim
                imp = imp.view(-1, geom["head_dim"]).sum(1)
            out_arr[li] = imp.double().numpy()
    return q_scores, mlp_scores


def magnitude_importance(model, scope, p=2, group_reduction="mean", weights=None):
    """hf_llama_pruner.py::MagnitudeImportance -- LLM-Pruner's `--pruner_type l1|l2`.

    The scripts construct it as `MagnitudeImportance(p=…)`, so `--grouping_strategy` is
    ignored and the reduction is `mean`.
    """
    geom = text_geometry(model)
    layers = text_layers(model)
    weights = weights if weights is not None else WeightCache()
    q_scores = np.zeros((geom["n_layers"], geom["n_heads"]), dtype=np.float64)
    mlp_scores = np.zeros((geom["n_layers"], geom["intermediate"]), dtype=np.float64)
    for li in scope:
        for members, out_arr, fold_heads in (
            (ATTN_MEMBERS, q_scores, True),
            (MLP_MEMBERS, mlp_scores, False),
        ):
            local = []
            for mname, axis_kind in members:
                w = weights.get(layers, li, mname)
                dim = 1 if axis_kind == "out" else 0
                local.append(w.abs().pow(p).sum(dim=dim))
            imp = _reduce_group(torch.stack(local, dim=0), group_reduction)
            if fold_heads:
                imp = imp.view(-1, geom["head_dim"]).sum(1)
            out_arr[li] = imp.double().numpy()
    return q_scores, mlp_scores


def random_importance(model, scope, seed=0):
    geom = text_geometry(model)
    rng = np.random.default_rng(seed)
    q_scores = np.zeros((geom["n_layers"], geom["n_heads"]), dtype=np.float64)
    mlp_scores = np.zeros((geom["n_layers"], geom["intermediate"]), dtype=np.float64)
    for li in scope:
        q_scores[li] = rng.random(geom["n_heads"])
        mlp_scores[li] = rng.random(geom["intermediate"])
    return q_scores, mlp_scores


# ---------------------------------------------------------------------------
# Stage 2c: fusing two objectives into one score (the `dual` arm)
# ---------------------------------------------------------------------------


def rank_norm(scores, scope):
    """Per-layer ascending rank in [0, 1], zero outside `scope`.

    The in-lab reference (`head_analysis/run_cocsafe.py:43-50`) ranks every layer; here
    the calibration pass only scores `scope`, so the other rows are identically zero and
    `np.argsort` on them is stable -- it would hand layer 0 the fabricated ranking
    0, 1/(n-1), 2/(n-1), ... and `keep_masks_for_ratio` would then prune those layers by
    head index with nothing detecting it.  Rows outside `scope` are left at zero instead.

    Magnitude is discarded on purpose: this is the only step where two objectives with
    different units (CoC NLL in nats, flow-matching MSE) have to share one scale.  For a
    single criterion it is a no-op, because `keep_masks_for_ratio` already ranks within
    a layer.
    """
    out = np.zeros_like(scores, dtype=np.float64)
    n = scores.shape[1]
    for li in scope:
        out[li] = np.argsort(np.argsort(scores[li], kind="stable"), kind="stable") / max(n - 1, 1)
    return out


def fuse_max(parts, scope):
    """`make_slim.py:101-111`: elementwise max over per-layer rank-normalised scores.

    "Keep a unit if EITHER objective calls it important" -- only units that both
    objectives rank low are dropped.  Not a weighted sum (there is no principled weight
    between nats and MSE) and not a keep-set union (that cannot hit a parameter budget).
    """
    assert parts, "fuse_max needs at least one score array"
    out = rank_norm(parts[0], scope)
    for other in parts[1:]:
        out = np.maximum(out, rank_norm(other, scope))
    return out


# ---------------------------------------------------------------------------
# Stage 2d: ratio -> kept indices
# ---------------------------------------------------------------------------


def n_pruned_channels(n_channels, ratio):
    """metapruner.py:249-252: `current - int(init * (1 - target_sparsity))`."""
    return n_channels - int(n_channels * (1.0 - ratio))


def keep_masks_for_ratio(q_scores, mlp_scores, ratio, scope, geom):
    """Local (per-layer) pruning: drop the lowest-scoring groups inside each layer.

    `--global_pruning` is deliberately off; a global threshold is confounded by the depth
    trend in the scores and turns a width sweep into a depth sweep.
    """
    n_layers, n_heads = q_scores.shape
    intermediate = mlp_scores.shape[1]
    head_dim = geom["head_dim"]

    n_q_ch = n_pruned_channels(n_heads * head_dim, ratio)
    n_q_drop = n_q_ch // head_dim  # whole heads only
    n_mlp_drop = n_pruned_channels(intermediate, ratio)

    q_keep = np.ones((n_layers, n_heads), dtype=np.float32)
    mlp_keep = np.ones((n_layers, intermediate), dtype=np.float32)
    for li in scope:
        if n_q_drop:
            q_keep[li, np.argsort(q_scores[li], kind="stable")[:n_q_drop]] = 0.0
        if n_mlp_drop:
            mlp_keep[li, np.argsort(mlp_scores[li], kind="stable")[:n_mlp_drop]] = 0.0
    return q_keep, mlp_keep, {"heads_dropped_per_layer": int(n_q_drop),
                              "mlp_dropped_per_layer": int(n_mlp_drop),
                              "layers": len(scope)}


def removed_params(plan, geom):
    cost = group_param_cost(geom)
    return plan["layers"] * (plan["heads_dropped_per_layer"] * cost["attn"]
                             + plan["mlp_dropped_per_layer"] * cost["mlp"])


def text_layer_params(geom):
    """Every parameter in the 36 decoder layers -- the denominator for "% of text layers".

    This is the pool the user-facing "prune N% of the VLM" number is quoted against.  It
    counts each projection ONCE; an earlier revision of lp_prune.py added q_proj and
    o_proj a second time (8,154,031,104 instead of 6,946,071,552) and shipped the
    resulting ~6-point understatement in four checkpoints' summary.txt.
    """
    return geom["n_layers"] * (
        2 * geom["hidden"] ** 2                                        # q_proj + o_proj
        + 2 * geom["hidden"] * geom["n_kv_heads"] * geom["head_dim"]   # k_proj + v_proj
        + 3 * geom["hidden"] * geom["intermediate"]                    # gate + up + down
        + 2 * geom["head_dim"]                                         # q_norm + k_norm
        + 2 * geom["hidden"]                                           # the two layernorms
    )


def summary_text(cfg, meta):
    """The human-readable sidecar, rendered from `meta` so the two cannot drift apart.

    `cfg` needs only .arm / .ratio / .layer_start / .layer_end, so an argparse Namespace
    and a checkpoint's own slim_meta.json both work -- `regen_summaries.py` reuses this to
    rewrite the sidecar of an already-built checkpoint without touching its weights.
    """
    p, plan = meta["params"], meta["config"]["plan"]
    return (
        f"arm={cfg.arm} ratio={cfg.ratio} scope=[{cfg.layer_start},{cfg.layer_end})\n"
        f"params {p['full']:,} -> {p['slim']:,} (removed {p['removed']:,}, "
        f"{p['pct_of_model']:.2f}% of model, "
        f"{p['pct_of_text_layers']:.2f}% of text layers)\n"
        f"per layer: -{plan['heads_dropped_per_layer']} heads, "
        f"-{plan['mlp_dropped_per_layer']} mlp channels\n"
        f"smoke coc_len={meta['smoke']['coc_len']}\n"
    )
