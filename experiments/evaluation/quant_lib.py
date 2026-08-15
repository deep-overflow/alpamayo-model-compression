"""Per-row fake quantization of the VLM tower (QVLA layout, Alpamayo pool).

Fake quant means quantize->dequantize back into bf16: the tensor keeps its dtype and
shape, so runtime, memory and determinism are identical to the unquantized model and
`run_baseline.py` needs no other change. Storage savings are therefore *projected*, not
measured -- `effective_bits()` reports what the allocation would cost packed, and every
plot axis derived from it must say "projected".

Layout follows QVLA in the one place that matters for the method: the decision variable
is a bit-width **per output row** of each weight, drawn from {0, 2, 4, 8, 16}, with 0
meaning the row is removed (structured pruning) and 16 meaning untouched. It departs
from QVLA in the scale granularity -- QVLA keeps one scale+zero-point per row, which was
measured on this checkpoint to cost 2.5x more error at 4 bits than a group of 64, and to
be catastrophic on lm_head (0.462 vs 0.016 relative MSE; 155697x4096 has outliers that
one scale per row cannot cover). Scale granularity and bit granularity are orthogonal --
a row's bit-width is uniform across its groups either way -- so grouping does not disturb
the allocation algorithm. `group=0` restores the exact QVLA layout for the ablation.

The pool is the VLM tower only: VLM text Linear + lm_head + ViT Linear (370 tensors,
8.158 B). The expert (2.279 B), `embed_tokens` (a lookup, no GEMM) and the action
projections stay bf16. That is not a preference: the CoC NLL graph does not contain the
expert at all (see `prune_lib.coc_nll` -- it ends at `model.vlm.lm_head`), so a CoC-only
criterion scores every expert row at exactly zero and would demote the whole tower to
0-bit. Excluding it is the criterion's own consequence.
"""

import re

import torch

BITS = (0, 2, 4, 8, 16)

# name patterns for the quantizable pool, matched against `model.named_modules()`
POOL_PATTERNS = {
    "vlm_text": re.compile(r"^vlm\.model\.language_model\.layers\.\d+\."),
    "lm_head": re.compile(r"^vlm\.lm_head$"),
    "vit": re.compile(r"^vlm\.model\.visual\."),
    # Not in the default pool: a CoC-only criterion scores every expert row at exactly
    # zero (the CoC graph ends at vlm.lm_head), so a guided allocation would demote the
    # whole tower to 0-bit. A *uniform* budget has no criterion and no such problem, and
    # the expert is 2.279 B of the 2.921 B that otherwise stays bf16 -- more storage than
    # 24% structured pruning buys. Opt in with parts=(..., "expert").
    "expert": re.compile(r"^expert\.layers\.\d+\."),
}
# never quantized, even though they live under `vlm.`
EXCLUDE = re.compile(r"embed_tokens")


def pool_modules(model, parts=("vlm_text", "lm_head", "vit")):
    """-> {name: nn.Linear} for every quantizable weight in the requested pool parts."""
    out = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear) or EXCLUDE.search(name):
            continue
        for part in parts:
            if POOL_PATTERNS[part].search(name):
                out[name] = mod
                break
    return out


def _groups(in_features, group):
    """(group_size, n_groups, padded_in) -- group<=0 means one group per row (QVLA)."""
    g = in_features if group <= 0 else min(group, in_features)
    n = -(-in_features // g)
    return g, n, n * g


def quantize_dequantize(W, bits, group=64):
    """Asymmetric per-(row, group) RTN. `bits` is an int or a per-row int tensor.

    W is (O, I). Returns the quantize->dequantize round trip in W's dtype. Rows with
    bits==0 come back zero (removed), rows with bits==16 come back untouched.
    """
    O, I = W.shape
    g, n_g, pad_I = _groups(I, group)
    b = (torch.full((O,), int(bits), device=W.device, dtype=torch.long)
         if isinstance(bits, int) else bits.to(W.device).long())

    out = W.clone()
    for bit in b.unique().tolist():
        rows = (b == bit).nonzero(as_tuple=True)[0]
        if bit == 16:
            continue
        if bit == 0:
            out[rows] = 0
            continue
        w = W[rows].float()                                    # (R, I)
        if pad_I != I:
            # pad the tail group with the row mean so the padding cannot widen the
            # group's range and inflate its step size
            fill = w.mean(1, keepdim=True).expand(-1, pad_I - I)
            w = torch.cat([w, fill], 1)
        wg = w.reshape(len(rows), n_g, g)                      # (R, n_g, g)
        lo = wg.amin(-1, keepdim=True)
        hi = wg.amax(-1, keepdim=True)
        qmax = 2**bit - 1
        s = ((hi - lo) / qmax).clamp_min(1e-8)
        z = torch.round(-lo / s)
        q = torch.clamp(torch.round(wg / s) + z, 0, qmax)
        deq = ((q - z) * s).reshape(len(rows), pad_I)[:, :I]
        out[rows] = deq.to(W.dtype)
    return out


def quant_error(W, bits, group=64):
    """eps = Q_b(W) - W, the exact deterministic RTN perturbation. Same shape as W."""
    return quantize_dequantize(W, bits, group) - W


def rel_mse(W, bits, group=64):
    e = quant_error(W.float(), bits, group)
    return (e.pow(2).sum() / W.float().pow(2).sum()).item()


def row_bits_cost(in_features, bits, group=64):
    """Projected storage in bits for ONE row at `bits`.

    bf16 scale (16 b) plus a zero-point stored at the row's own bit-width, per group --
    the packing GPTQ/AWQ kernels actually use. A 16-bit row is stored as bf16 with no
    quantization metadata; a 0-bit row costs nothing.
    """
    if bits == 0:
        return 0
    if bits == 16:
        return in_features * 16
    _, n_g, pad_I = _groups(in_features, group)
    return pad_I * bits + n_g * (16 + bits)


def effective_bits(shapes, spec, group=64):
    """Pool-average projected bits per parameter.

    `shapes` is {name: (O, I)}, `spec` is {name: per-row bit array}. The denominator is
    the pool's unquantized parameter count, so a 0-bit row lowers the average rather
    than leaving the pool.
    """
    total_bits = total_params = 0
    for name, (O, I) in shapes.items():
        b = spec[name]
        for bit in (0, 2, 4, 8, 16):
            n = int((b == bit).sum())
            if n:
                total_bits += n * row_bits_cost(I, bit, group)
        total_params += O * I
    return total_bits / total_params, total_bits, total_params


def uniform_spec(shapes, bits):
    return {name: torch.full((O,), bits, dtype=torch.long) for name, (O, I) in shapes.items()}


@torch.no_grad()
def apply_quant(model, spec, group=64, parts=("vlm_text", "lm_head", "vit")):
    """In-place fake quantization. Returns a per-tensor report."""
    mods = pool_modules(model, parts)
    missing = set(spec) - set(mods)
    if missing:
        raise KeyError(f"spec names not in pool: {sorted(missing)[:5]}")
    report = {}
    for name, mod in mods.items():
        if name not in spec:
            continue
        W = mod.weight.data
        b = spec[name]
        q = quantize_dequantize(W, b, group)
        err = (q.float() - W.float()).pow(2).sum() / W.float().pow(2).sum().clamp_min(1e-12)
        report[name] = {
            "shape": list(W.shape),
            "rel_mse": float(err),
            "bit_hist": {int(x): int((b == x).sum()) for x in torch.tensor(BITS)},
        }
        mod.weight.data = q
    return report


def pool_shapes(model, parts=("vlm_text", "lm_head", "vit")):
    return {n: tuple(m.weight.shape) for n, m in pool_modules(model, parts).items()}
