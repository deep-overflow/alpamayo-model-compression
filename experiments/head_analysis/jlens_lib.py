"""Jacobian lens (J-lens) and J-space unit scoring for the VLM tower (Stage A/B).

Follows Anthropic's "Verbalizable Representations Form a Global Workspace in
Language Models" (transformer-circuits, 2026-07-06):

    J_l = E[ d h_final,t' / d h_l,t ]     over source positions t, all t' >= t

The J-lens vector of token tau at layer l is v_tau^l = J_l^T Wu[tau], i.e. the
direction in the layer-l residual stream that raises tau's logit at the present
or any future position. J-space is the set of sparse nonnegative combinations of
those vectors.

Why cotangents are tokens rather than a random output basis: a random sketch on
the *output* side gives U J_l, which cannot be composed with Wu (wrong
dimension), so the readout that defines J-space is lost. Taking the cotangent to
be a row of Wu instead yields v_tau^l directly, and one backward produces it for
*every* layer at once. The token set is therefore the sketch: a subsample of the
vocabulary spanning the driving terms the model actually emits plus a random
slice for coverage.

Unit scoring. A unit's write enters the residual stream as a fixed direction
scaled by a varying activation, so its alignment with the whole dictionary
factorises and needs no per-unit backward:

  MLP channel c : write = a_c(t) * down_proj[:, c]
                  j = || V_l down_proj[:, c] || * rms_t(a_c)
  Q head h      : write = o_proj[:, h] @ y_h(t)          (y_h in R^head_dim)
                  j^2 = tr( (V_l O_h)^T (V_l O_h) E[y_h y_h^T] )

Both are exact under the linearisation; only second moments of the activations
are needed, collected by hooks over the same span the Jacobian is averaged on.
`write_norm` repeats each formula with V_l dropped, so jfrac = j / write_norm
separates "writes a lot, none of it verbalizable" from the converse -- which is
the quantity hypothesis H is about.

Source positions are restricted to text and generated-CoC tokens: an Alpamayo
prompt is dominated by vision tokens, which are not carriers of verbalizable
content, and averaging over them only dilutes J_l.
"""

from collections import Counter

import numpy as np
import torch


# ---------------------------------------------------------------------------
# token dictionary
# ---------------------------------------------------------------------------


def banned_ids(model):
    """Token ids that must not enter the dictionary (traj tokens, image pad)."""
    start = model.config.traj_token_start_idx
    ids = set(range(start, start + model.config.traj_vocab_size))
    ids.add(151655)  # <|image_pad|>
    return ids


def select_tokens(coc_token_ids, model, n_freq, n_random, seed=20260728):
    """Dictionary = most frequent tokens the model emits in CoC + a random slice.

    The frequent half makes the readout interpretable on this model's own
    vocabulary; the random half keeps the score from being defined only on terms
    we already expect to matter.
    """
    banned = banned_ids(model)
    counts = Counter(int(t) for t in coc_token_ids if int(t) not in banned)
    freq = [t for t, _ in counts.most_common(n_freq)]
    taken = set(freq) | banned
    vocab = model.vlm.lm_head.weight.shape[0]
    rng = np.random.default_rng(seed)
    pool = rng.choice(vocab, size=min(vocab, 8 * n_random + len(taken)), replace=False)
    rand = [int(t) for t in pool if int(t) not in taken][:n_random]
    # order matters: the first len(freq) entries are the CoC half, which freq_auc
    # and the report rely on to separate driving vocabulary from random tokens
    return freq + rand, len(freq)


def unembed_rows(model, token_ids):
    """Centered unembedding rows for the dictionary. Returns (S, d) fp32 cuda.

    Centering over the full vocabulary removes the component softmax ignores, so
    a direction that lifts every logit equally scores zero.
    """
    w = model.vlm.lm_head.weight  # (V, d)
    mu = w.float().mean(0)  # (d,)
    idx = torch.as_tensor(token_ids, device=w.device, dtype=torch.long)
    return w[idx].float() - mu  # (S, d)


# ---------------------------------------------------------------------------
# probe taps: residual-stream tensors + on-the-fly grad reduction
# ---------------------------------------------------------------------------


class ProbeTaps:
    """Capture per-layer residual-stream tensors and mean-reduce their grads.

    Two tap points per layer:
      mid[l] : input to post_attention_layernorm = residual after the attention
               write, the correct source point for Q-head contributions
      out[l] : the decoder layer's output, the source point for MLP channels

    Grads are reduced to (d,) inside a tensor hook and the full (1, T, d) buffer
    is dropped immediately, so a backward costs O(T*d) transient memory instead
    of holding 2L of them. This is why the backward is a plain .backward() with
    no retain_grad / inputs=: both would materialise every probe grad at once.
    """

    def __init__(self, layers):
        self.n_layers = len(layers)
        self.mid = [None] * self.n_layers
        self.out = [None] * self.n_layers
        self.g_mid = [None] * self.n_layers
        self.g_out = [None] * self.n_layers
        self.span = None  # (n_src,) long, source positions
        self._handles = []
        for i, layer in enumerate(layers):
            self._handles.append(
                layer.post_attention_layernorm.register_forward_pre_hook(self._cap("mid", i))
            )
            self._handles.append(
                layer.register_forward_hook(self._cap_out(i), with_kwargs=True)
            )

    def _reduce(self, store, i):
        def hook(grad):  # (1, T, d)
            store[i] = grad[0, self.span].float().mean(0)  # (d,)
        return hook

    def _cap(self, which, i):
        def hook(module, args):
            t = args[0]
            getattr(self, which)[i] = t
            t.register_hook(self._reduce(self.g_mid if which == "mid" else self.g_out, i))
        return hook

    def _cap_out(self, i):
        def hook(module, args, kwargs, output):
            t = output[0] if isinstance(output, tuple) else output
            self.out[i] = t
            t.register_hook(self._reduce(self.g_out, i))
        return hook

    def set_span(self, span_idx):
        self.span = span_idx

    def collect(self, d, device):
        """Stack the reduced grads of the last backward. Returns (2, L, d)."""
        zero = torch.zeros(d, device=device)
        mid = torch.stack([g if g is not None else zero for g in self.g_mid])
        out = torch.stack([g if g is not None else zero for g in self.g_out])
        return torch.stack([mid, out])  # (2, L, d)

    def clear_tensors(self):
        self.mid = [None] * self.n_layers
        self.out = [None] * self.n_layers

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def build_jlens(taps, last_hidden, wu_rows, chunk_log=None):
    """One backward per dictionary token -> J-lens vectors at every layer.

    Args:
        last_hidden: (1, T, d) final (already RMSNormed) hidden state. Using the
            post-norm tensor folds the norm into the linearisation, which is what
            the lens applies anyway.
        wu_rows: (S, d) centered unembedding rows.

    Returns (2, L, S, d) fp32 on cuda: [mid/out, layer, token, direction].
    """
    s, d = wu_rows.shape
    n_layers = taps.n_layers
    v = torch.zeros(2, n_layers, s, d, device=wu_rows.device, dtype=torch.float32)
    for i in range(s):
        # sum over all output positions t'; causality gives t' >= t for free
        target = (last_hidden[0] * wu_rows[i]).sum()
        target.backward(retain_graph=True)
        v[:, :, i, :] = taps.collect(d, wu_rows.device)
        if chunk_log is not None and (i + 1) % chunk_log == 0:
            print(f"    jlens {i + 1}/{s}", flush=True)
    return v


# ---------------------------------------------------------------------------
# activation second moments over the same span
# ---------------------------------------------------------------------------


class WriteStats:
    """Second moments of the quantities each pruning unit scales its write by.

    MLP: E[a_c^2] per channel (down_proj input).
    Q head: E[y y^T] per head (o_proj input, reshaped per head) -- the full
    head_dim x head_dim matrix, because the head's write direction rotates with
    y and a per-component rms would not be rotation-correct.
    """

    def __init__(self, layers, n_heads, head_dim, intermediate, device):
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.mlp_sq = torch.zeros(len(layers), intermediate, device=device, dtype=torch.float64)
        self.head_cov = torch.zeros(
            len(layers), n_heads, head_dim, head_dim, device=device, dtype=torch.float64
        )
        self.count = 0
        self.span = None
        self._handles = []
        for i, layer in enumerate(layers):
            self._handles.append(layer.mlp.down_proj.register_forward_pre_hook(self._mlp(i)))
            self._handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self._attn(i)))

    def set_span(self, span_idx):
        self.span = span_idx

    # detach() is load-bearing: these hook inputs are live graph tensors, so
    # accumulating them without it attaches a grad_fn to the persistent buffers
    # and pins the whole forward graph (~16 GB) for the process lifetime.

    def _mlp(self, i):
        def hook(module, args):
            x = args[0].detach()[0, self.span].double()  # (n_src, I)
            self.mlp_sq[i] += (x ** 2).sum(0)
        return hook

    def _attn(self, i):
        def hook(module, args):
            x = args[0].detach()[0, self.span].double()  # (n_src, H*D)
            y = x.view(-1, self.n_heads, self.head_dim)  # (n_src, H, D)
            self.head_cov[i] += torch.einsum("thd,the->hde", y, y)
        return hook

    def add_positions(self, n):
        self.count += n

    def finalize(self):
        n = max(self.count, 1)
        return (self.mlp_sq / n).float(), (self.head_cov / n).float()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


# ---------------------------------------------------------------------------
# unit scores
# ---------------------------------------------------------------------------


def unit_jscores(model, v, mlp_sq, head_cov):
    """J-space alignment and total write norm per unit, per layer.

    Args:
        v: (2, L, S, d) J-lens vectors; index 0 (mid) scores Q heads, index 1
           (out) scores MLP channels.
        mlp_sq: (L, I) E[a_c^2];  head_cov: (L, H, D, D) E[y y^T].

    Returns dict of (L, n_units) numpy arrays: q_j, q_w, mlp_j, mlp_w.
    """
    layers = model.vlm.model.language_model.layers
    tc = model.vlm.config.text_config
    n_heads, head_dim = tc.num_attention_heads, tc.head_dim
    n_layers = len(layers)
    out = {
        "q_j": np.zeros((n_layers, n_heads)), "q_w": np.zeros((n_layers, n_heads)),
        "mlp_j": np.zeros((n_layers, mlp_sq.shape[1])),
        "mlp_w": np.zeros((n_layers, mlp_sq.shape[1])),
    }
    for li, layer in enumerate(layers):
        with torch.no_grad():
            # --- MLP channels: fixed write direction, scalar activation
            dw = layer.mlp.down_proj.weight.float()  # (d, I)
            rms = mlp_sq[li].clamp_min(0).sqrt()  # (I,)
            a = v[1, li] @ dw  # (S, I)
            out["mlp_j"][li] = (a.norm(dim=0) * rms).cpu().numpy()
            out["mlp_w"][li] = (dw.norm(dim=0) * rms).cpu().numpy()
            del a

            # --- Q heads: write direction rotates with y, so use E[y y^T]
            ow = layer.self_attn.o_proj.weight.float()  # (d, H*D)
            o = ow.view(ow.shape[0], n_heads, head_dim)  # (d, H, D)
            vo = torch.einsum("sd,dhe->she", v[0, li], o)  # (S, H, D)
            m = torch.einsum("she,shf->hef", vo, vo)  # (H, D, D)
            c = head_cov[li].float()  # (H, D, D)
            out["q_j"][li] = (m * c).sum((-1, -2)).clamp_min(0).sqrt().cpu().numpy()
            mw = torch.einsum("dhe,dhf->hef", o, o)  # (H, D, D)
            out["q_w"][li] = (mw * c).sum((-1, -2)).clamp_min(0).sqrt().cpu().numpy()
            del vo, m, mw
    return out


# ---------------------------------------------------------------------------
# Stage A validation: readout + layer bands
# ---------------------------------------------------------------------------


def readout(v_layer, h, token_ids, tokenizer, topk=15):
    """Top dictionary tokens for an activation h under the lens at one layer.

    v_layer: (S, d); h: (d,). Restricted to the dictionary, so this ranks within
    the sketch rather than the full vocabulary -- enough to tell driving terms
    from noise, which is what G1 asks.
    """
    with torch.no_grad():
        scores = v_layer @ h.float()  # (S,)
        top = torch.topk(scores, min(topk, scores.shape[0]))
    return [(tokenizer.decode([token_ids[i]]), float(s))
            for i, s in zip(top.indices.tolist(), top.values.tolist())]


def freq_auc(v_layer, acts, n_freq):
    """Do CoC-vocabulary tokens outrank random-vocabulary tokens in the readout?

    The dictionary is built as [CoC-frequent | random], so this is a falsifiable
    version of "the readout names driving concepts": AUC 0.5 means the lens can
    not tell the model's own reasoning vocabulary from arbitrary tokens, which is
    G1 failing. Mann-Whitney U, averaged over probe activations.
    """
    s = v_layer.shape[0]
    n_rand = s - n_freq
    if n_freq == 0 or n_rand == 0:
        return float("nan")
    with torch.no_grad():
        scores = acts.float() @ v_layer.T  # (n_probe, S)
        ranks = scores.argsort(dim=1).argsort(dim=1).float() + 1.0  # (n_probe, S)
        r_freq = ranks[:, :n_freq].sum(1)
        u = r_freq - n_freq * (n_freq + 1) / 2.0
    return float((u / (n_freq * n_rand)).mean())


def excess_kurtosis(v_layer, acts):
    """Excess kurtosis of the lens readout, averaged over probe activations.

    The paper's sparsity signature: a layer whose readout is peaked on a few
    tokens is carrying discrete verbalizable content; a flat readout is not.

    Pass only the RANDOM half of the dictionary. The CoC half is selected for
    being frequent, so including it makes the readout bimodal by construction
    and the kurtosis measures our sampling, not the model.
    """
    with torch.no_grad():
        s = acts.float() @ v_layer.T  # (n_probe, S)
        s = s - s.mean(1, keepdim=True)
        var = s.var(1, unbiased=False).clamp_min(1e-12)
        k = ((s ** 4).mean(1) / var ** 2) - 3.0
    return float(k.mean())


def cka_matrix(v):
    """Linear CKA between layers' J-lens dictionaries. v: (L, S, d) -> (L, L).

    Block structure in this matrix is how the paper locates the sensory /
    workspace / motor transition points.
    """
    n = v.shape[0]
    g = []
    with torch.no_grad():
        for i in range(n):
            x = v[i] - v[i].mean(0, keepdim=True)  # (S, d)
            g.append(x @ x.T)  # (S, S) gram
        m = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                num = (g[i] * g[j]).sum()
                den = (g[i].norm() * g[j].norm()).clamp_min(1e-12)
                m[i, j] = float(num / den)
    return m
