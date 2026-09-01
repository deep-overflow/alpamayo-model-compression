"""Pruned VLM internals: what changed inside, not just how much minADE moved.

plans/2026-09-01_pruned-vlm-internals.md. Two measurements per clip, both off the same
pair of teacher-forced forwards (dense, then the same model with dual_u40_v2's masks on):

(A) VLM residual stream. Three points per layer -- h_in, h_mid = h_in + a, h_out = h_mid + m
    -- compared dense vs pruned by cos and rel = ||x_p - x_d|| / ||x_d||, and compared
    WITHIN each model by the transition cos(h_in, h_mid) / cos(h_mid, h_out) and the
    relative write size ||a||/||h_in|| / ||m||/||h_mid||. The cross-model differences
    localise where inside a layer the streams separate; the transitions say whether
    pruning changes how the block transforms the stream at all.

(B) Expert attention over the VLM cache. dual_u40_v2 leaves the expert untouched (16/16
    heads), so this is ONE expert reading TWO caches. With eager attention the probs are
    materialised per (layer, step) and compared per (layer, head, step, query token) by
    total variation (primary -- the fraction of attention mass relocated), both KL
    directions (which side drops keys the other keeps) and JS.

Instrument: one dense model with mask_lib.PruneMasks toggled, not two checkpoints
(run_cachediff.py's approach). Two checkpoints need 36.3 GiB of weights and, with the
captures here, exceed a 48 GB card; masks need 20.6 GiB and hold the code path fixed, so
the measured difference is the pruning and nothing else.

Taps: `output_hidden_states=True` is NOT usable -- deepstack writes vision features into
the recorded tensors in place after text layers 0-2, and the last entry is replaced by the
RMSNorm'ed output, so layer 35's raw h_out is unavailable. Hooks on the decoder layers
themselves (args[0] = h_in, output = h_out, cloned inside the hook) plus a pre-hook on
post_attention_layernorm (h_mid) give all three points exactly.

Pre-registered gates (G0-G4) are in the plan; --verify-taps runs G1.

Usage:
  ALPAMAYO_REPO=$PWD bash experiments/head_analysis/run_retry_host.sh 240 \
      experiments/head_analysis/run_prunedvlm.py --gpu 4 --exp-id prunedvlm_dual \
      --num-clips 32
  # wiring check first:
  ... run_prunedvlm.py --gpu 4 --exp-id prunedvlm_smoke --num-clips 1 --verify-taps
"""

import os

# must precede any CUDA context creation for deterministic cuBLAS reductions
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

import analysis_lib as lib  # noqa: E402
import mask_lib as ml  # noqa: E402
import sample_cache as sc  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  also installs the gated-repo hub patch
from make_slim import build_masks  # noqa: E402
from run_cacheuse import StepCounter  # noqa: E402
from run_cachediff import span_index, tf_forward  # noqa: E402
from slim_lib import MODEL_REV  # noqa: E402

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
N_STEPS = 10
COARSE = ["all", "vision", "text", "hist", "sink", "coc"]
FINE = ["special", "sys_text", "cam_text", "instr"]
POINTS = ["h_in", "h_mid", "h_out"]
TRANS = ["attn", "mlp"]


def set_determinism():
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def fine_spans(model, input_ids, spans, prompt_len):
    """Split the coarse `text` bucket into special / sys_text / cam_text / instr.

    compute_spans' `text` is everything that is not vision, traj-history or the sink, so
    it lumps the system prompt, the per-camera labels, the instruction sentence and every
    `<|...|>` marker together. Special ids come from two places: the base Cosmos tokenizer
    knows `<|im_start|>`/`<|vision_start|>`/`<|image_pad|>` etc., but `<|cot_start|>` and
    the traj markers are Alpamayo additions and live in model.config.traj_token_ids.
    """
    ids = input_ids[0, :prompt_len].cpu()
    text = spans["text"]
    sp_ids = {int(i) for i in (model.tokenizer.all_special_ids or [])}
    tt = getattr(model.config, "traj_token_ids", None)
    if tt is not None:
        vals = tt.values() if hasattr(tt, "values") else vars(tt).values()
        sp_ids |= {int(v) for v in vals if isinstance(v, int)}
    special = torch.zeros(prompt_len, dtype=torch.bool)
    if sp_ids:
        special = torch.isin(ids, torch.tensor(sorted(sp_ids)))
    special &= text  # only the markers that landed in the text bucket

    vis = torch.nonzero(spans["vision"]).flatten()
    first_vis = int(vis[0]) if len(vis) else prompt_len
    last_vis = int(vis[-1]) if len(vis) else -1
    pos = torch.arange(prompt_len)
    plain = text & ~special
    return {"special": special,
            "sys_text": plain & (pos < first_vis),
            "cam_text": plain & (pos >= first_vis) & (pos <= last_vis),
            "instr": plain & (pos > last_vis)}


class ResidualTaps:
    """h_in / h_mid / h_out per layer, cloned so deepstack's in-place write cannot reach us.

    The decoder-layer hook fires before the caller's _deepstack_process, so cloning inside
    it is what makes h_out trustworthy on layers 0-2; output_hidden_states does not have
    that property. Tensors are kept on the host (bf16) because the dense pass must survive
    until the pruned pass runs.
    """

    def __init__(self, layers, keep_device="cpu"):
        self.n = len(layers)
        self.dev = keep_device
        self.h_in = [None] * self.n
        self.h_mid = [None] * self.n
        self.h_out = [None] * self.n
        self._handles = []
        for i, layer in enumerate(layers):
            self._handles.append(layer.register_forward_hook(self._make_layer(i)))
            self._handles.append(
                layer.post_attention_layernorm.register_forward_pre_hook(self._make_mid(i)))

    def _make_layer(self, i):
        def hook(module, args, output):
            self.h_in[i] = args[0].detach().clone().to(self.dev)
            out = output[0] if isinstance(output, tuple) else output
            self.h_out[i] = out.detach().clone().to(self.dev)
        return hook

    def _make_mid(self, i):
        def hook(module, args):
            self.h_mid[i] = args[0].detach().clone().to(self.dev)
        return hook

    def clear(self):
        self.h_in = [None] * self.n
        self.h_mid = [None] * self.n
        self.h_out = [None] * self.n

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


class BlockTaps:
    """o_proj / down_proj OUTPUTS -- only for the G1 wiring check (--verify-taps).

    These are the tensors the layer adds to the residual (modeling_qwen3_vl.py:455-457,
    472-474), so h_out == (h_in + a).add(m) must hold bitwise when recomposed in the
    model's own order and dtype.
    """

    def __init__(self, layers):
        self.n = len(layers)
        self.a = [None] * self.n
        self.m = [None] * self.n
        self._handles = []
        for i, layer in enumerate(layers):
            self._handles.append(
                layer.self_attn.o_proj.register_forward_hook(self._make(i, "a")))
            self._handles.append(
                layer.mlp.down_proj.register_forward_hook(self._make(i, "m")))

    def _make(self, i, which):
        def hook(module, args, output):
            getattr(self, which)[i] = output.detach().clone()
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


class ExpertAttnCapture:
    """Eager expert attention per (layer, step), either stored or compared against a store.

    mode="store"   -> keep (L, S) probs on the host as fp16
    mode="compare" -> reduce against the stored dense probs into per (L, H, S, Q) metrics

    Probs arrive bf16 (eager_attention_forward casts the fp32 softmax back to the query
    dtype), which alone puts a ~1e-3 floor on TV, so everything is recomputed in fp32 and
    renormalised before any divergence is taken.
    """

    def __init__(self, model, counter, prefill, sidx, n_layers, n_heads, mode, store=None):
        self.counter, self.prefill, self.sidx = counter, prefill, sidx
        self.L, self.H, self.mode = n_layers, n_heads, mode
        self.store = store if store is not None else {}
        self.spans = list(sidx)
        if mode == "compare":
            z = lambda *s: np.zeros(s, dtype=np.float32)  # noqa: E731
            self.out = {k: z(self.L, self.H, N_STEPS, 64)
                        for k in ("tv", "js", "kl_pq", "kl_qp")}
            for tag in ("d", "p"):
                self.out[f"mass_{tag}"] = z(self.L, self.H, N_STEPS, len(self.spans))
                self.out[f"ent_{tag}"] = z(self.L, self.H, N_STEPS)
            self.out["own_d"] = z(self.L, self.H, N_STEPS)
            self.out["own_p"] = z(self.L, self.H, N_STEPS)
        self.rowsum_bf16 = []
        self._handles = []
        for li, layer in enumerate(model.expert.layers):
            self._handles.append(layer.self_attn.register_forward_hook(
                self._make(li), with_kwargs=True))

    def _norm(self, attn):
        a = attn[0].float()  # (H, 64, Tk)
        self.rowsum_bf16.append(float((a.sum(-1) - 1).abs().max()))
        return a / a.sum(-1, keepdim=True).clamp_min(1e-30)

    def _stats(self, a, tag, li, s):
        P = self.prefill
        for j, name in enumerate(self.spans):
            m = self.sidx[name]
            self.out[f"mass_{tag}"][li, :, s, j] = \
                a[:, :, :P][:, :, m].sum(-1).mean(-1).cpu().numpy()
        self.out[f"own_{tag}"][li, :, s] = a[:, :, P:].sum(-1).mean(-1).cpu().numpy()
        ent = -(a.clamp_min(1e-12).log() * a).sum(-1) / np.log(a.shape[-1])
        self.out[f"ent_{tag}"][li, :, s] = ent.mean(-1).cpu().numpy()

    def _make(self, li):
        def hook(module, args, kwargs, output):
            attn = output[1]
            assert attn is not None, "eager attention required on the expert"
            s = self.counter.step
            a = self._norm(attn)  # (H, 64, Tk) fp32, renormalised
            if self.mode == "store":
                # fp32, not fp16: attention probs over ~3.2k keys are routinely below
                # fp16's subnormal threshold, and rounding them there put a TV floor of
                # 9e-5 on the no-mask control instead of the exact 0 it must be
                self.store[(li, s)] = a.cpu()
                return
            d = self.store[(li, s)].to(a.device).float()
            d = d / d.sum(-1, keepdim=True).clamp_min(1e-30)
            p, q = d.clamp_min(1e-12), a.clamp_min(1e-12)
            mmix = 0.5 * (p + q)
            self.out["tv"][li, :, s] = (0.5 * (p - q).abs().sum(-1)).cpu().numpy()
            self.out["kl_pq"][li, :, s] = (p * (p.log() - q.log())).sum(-1).cpu().numpy()
            self.out["kl_qp"][li, :, s] = (q * (q.log() - p.log())).sum(-1).cpu().numpy()
            self.out["js"][li, :, s] = (
                0.5 * (p * (p.log() - mmix.log())).sum(-1)
                + 0.5 * (q * (q.log() - mmix.log())).sum(-1)).cpu().numpy()
            self._stats(d, "d", li, s)
            self._stats(a, "p", li, s)
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def verify_metrics(taps, blk, masks):
    """G1 + the bf16 floor, from the DENSE pass only.

    Must be called while `blk` still holds the dense pass's o_proj/down_proj outputs --
    the pruned pass would overwrite them. h_out == (h_in + a).add(m) is bitwise exact when
    recomposed in the model's own order and dtype; recomposing in fp32 instead leaves the
    rounding residue, which is the floor every residual claim is read against.
    """
    names = list(masks)
    L = taps.n
    out = {"g1_bitwise": np.zeros(L, np.float32),
           "bf16_floor_rel": np.zeros((L, len(names)), np.float32),
           "bf16_floor_cos": np.zeros((L, len(names)), np.float32)}

    def by_span(v):
        return np.array([float(v[masks[n]].mean()) for n in names], dtype=np.float32)

    for li in range(L):
        a, m = blk.a[li], blk.m[li]
        h_in, h_out = taps.h_in[li].to(a.device), taps.h_out[li].to(a.device)
        out["g1_bitwise"][li] = float(torch.equal(h_out, (h_in + a).add(m)))
        r = (h_in.float() + a.float() + m.float())[0].cpu()
        ho = h_out[0].float().cpu()
        out["bf16_floor_rel"][li] = by_span(
            ((r - ho).norm(dim=-1) / ho.norm(dim=-1).clamp_min(1e-12)).numpy())
        out["bf16_floor_cos"][li] = by_span(F.cosine_similarity(r, ho, dim=-1).numpy())
    return out


def residual_metrics(taps_d, taps_p, masks):
    """Per (layer, point/transition, span): cross-model cos/rel and within-model transitions.

    Metrics are computed per token first and averaged inside a span -- averaging the
    vectors first would mix directions and destroy the cosine.
    """
    names = list(masks)
    L = taps_d.n
    out = {"cross_cos": np.zeros((L, 3, len(names)), np.float32),
           "cross_rel": np.zeros((L, 3, len(names)), np.float32)}
    for tag in ("d", "p"):
        out[f"trans_cos_{tag}"] = np.zeros((L, 2, len(names)), np.float32)
        out[f"trans_mag_{tag}"] = np.zeros((L, 2, len(names)), np.float32)

    def by_span(v):  # v (T,) -> (n_span,)
        return np.array([float(v[masks[n]].mean()) for n in names], dtype=np.float32)

    for li in range(L):
        pts_d = [taps_d.h_in[li][0].float(), taps_d.h_mid[li][0].float(),
                 taps_d.h_out[li][0].float()]
        pts_p = [taps_p.h_in[li][0].float(), taps_p.h_mid[li][0].float(),
                 taps_p.h_out[li][0].float()]
        for pi in range(3):
            d, p = pts_d[pi], pts_p[pi]
            out["cross_cos"][li, pi] = by_span(F.cosine_similarity(d, p, dim=-1).numpy())
            out["cross_rel"][li, pi] = by_span(
                ((p - d).norm(dim=-1) / d.norm(dim=-1).clamp_min(1e-12)).numpy())
        for tag, pts in (("d", pts_d), ("p", pts_p)):
            for ti, (lo, hi) in enumerate(((0, 1), (1, 2))):
                w = pts[hi] - pts[lo]
                out[f"trans_cos_{tag}"][li, ti] = by_span(
                    F.cosine_similarity(pts[lo], pts[hi], dim=-1).numpy())
                out[f"trans_mag_{tag}"][li, ti] = by_span(
                    (w.norm(dim=-1) / pts[lo].norm(dim=-1).clamp_min(1e-12)).numpy())
    return out


def save(out_dir, rows, res, attn, meta, n):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({**meta, "n_done": n}, indent=1))
    (out_dir / "metrics.json").write_text(json.dumps(rows, indent=1))
    if res:
        np.savez_compressed(out_dir / "residual.npz",
                            **{k: np.stack([r[k] for r in res]) for k in res[0]})
    if attn:
        np.savez_compressed(out_dir / "attn.npz",
                            **{k: np.stack([a[k] for a in attn]) for k in attn[0]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--config", default="dual_u40_v2")
    ap.add_argument("--importance", default="importance_v2")
    ap.add_argument("--num-clips", type=int, default=32)
    ap.add_argument("--clip-offset", type=int, default=0)
    ap.add_argument("--manifest", default="indist_500")
    ap.add_argument("--sets-id", default="eval_sets")
    ap.add_argument("--cache", default="eval")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--reserve-gb", type=float, default=26.0)
    ap.add_argument("--gpu", type=str, default=None)
    ap.add_argument("--verify-taps", action="store_true",
                    help="G1: also hook o_proj/down_proj outputs and assert the bitwise "
                         "residual decomposition; measures the bf16 floor as a side effect")
    ap.add_argument("--no-mask", action="store_true",
                    help="G3-1: leave the masks off in BOTH passes; every metric must be "
                         "exactly 0, which checks the determinism setup")
    args = ap.parse_args()

    set_determinism()
    out_dir = REPO / "outputs" / args.exp_id
    df = pd.read_parquet(REPO / "outputs" / args.sets_id / f"{args.manifest}.parquet")
    rows_man = [(r.clip_id, int(r.t0_us)) for r in df.itertuples()]
    rows_man = rows_man[args.clip_offset: args.clip_offset + args.num_clips]

    devices = None if args.gpu is None else [int(x) for x in args.gpu.split(",")]
    device = reserve_gpu(args.reserve_gb, devices=devices)
    print(f"using {device}", flush=True)

    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")  # never eager: (1,32,3102,3102) probs is ~616 GiB
    lib.set_expert_attn_impl(model, "eager")

    imp = dict(np.load(REPO / "outputs" / args.importance / "importance.npz"))
    vq, vm, eq, em, kvonly = build_masks(args.config, imp, model)
    assert not kvonly, "this diagnostic assumes no KV-only layer"
    assert eq.min() == 1 and em.min() == 1, "config must leave the expert untouched"
    tc = model.vlm.config.text_config
    ec = model.expert.config
    vlayers = model.vlm.model.language_model.layers
    vmasks = ml.PruneMasks(vlayers, tc.num_attention_heads, tc.head_dim,
                           tc.intermediate_size, "cuda")
    print(f"VLM keep q={vq.mean():.4f} mlp={vm.mean():.4f} | expert untouched", flush=True)

    meta = {"exp_id": args.exp_id, "config": args.config, "importance": args.importance,
            "manifest": args.manifest, "clip_offset": args.clip_offset,
            "num_clips": len(rows_man), "seed": args.seed,
            "seed_rule": "sha256(f'{seed}:{clip_id}')[:4]", "cache": args.cache,
            "model_revision": MODEL_REV, "instrument": "one dense model + PruneMasks",
            "spans": COARSE + FINE, "points": POINTS, "trans": TRANS,
            "verify_taps": args.verify_taps, "no_mask": args.no_mask,
            "gpu": torch.cuda.get_device_name(device),
            "plan": "plans/2026-09-01_pruned-vlm-internals.md"}

    rows, res_all, attn_all = [], [], []
    for ci, (clip_id, t0_us) in enumerate(rows_man):
        t0 = time.time()
        data = sc.load_cached(sc.path_for(args.cache, clip_id, t0_us))
        inputs = lib.build_inputs(model, processor, data, "cuda")
        prompt_len = inputs["input_ids"].shape[1]
        base = sc.clip_seed(args.seed, clip_id)

        vmasks.reset()
        torch.manual_seed(base)
        torch.cuda.manual_seed_all(base)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
            coc_start, coc_end = prompt_len, roll["eos_pos"] + 1
            seq_tf = roll["sequences"][:, :coc_end].clone()
            del roll

        spans = lib.compute_spans(model, inputs["input_ids"])
        fs = fine_spans(model, inputs["input_ids"], spans, prompt_len)

        # ---- dense pass -------------------------------------------------------
        # taps must be removed before the pruned forward or its hooks overwrite the dense
        # tensors; the captures themselves are already cloned to the host by then.
        taps_d = ResidualTaps(vlayers)
        blk = BlockTaps(vlayers) if args.verify_taps else None
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            cache_d, rope_d, nll_d = tf_forward(model, seq_tf, inputs, coc_start, coc_end)
        taps_d.remove()
        prefill = cache_d.get_seq_length()
        assert prefill == seq_tf.shape[1], "cache length must equal the forced sequence"
        sidx = span_index(spans, prompt_len, coc_start, coc_end, prefill, "cuda")
        for k, v in fs.items():
            m = torch.zeros(prefill, dtype=torch.bool, device="cuda")
            m[:prompt_len] = v.to("cuda")
            sidx[k] = m
        # residual tensors span the same positions as the cache, so the same masks serve
        span_cpu = {n: sidx[n].cpu() for n in sidx}
        span_cpu = {n: m for n, m in span_cpu.items() if bool(m.any())}

        ver = None
        if blk is not None:
            ver = verify_metrics(taps_d, blk, span_cpu)
            blk.remove()

        offset = torch.tensor([prefill], device="cuda")
        prefix_mask = torch.ones(1, prefill, device="cuda", dtype=torch.long)
        counter = StepCounter(model)
        store = {}
        cap = ExpertAttnCapture(model, counter, prefill, sidx, ec.num_hidden_layers,
                               ec.num_attention_heads, "store", store)
        counter.reset()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            lib.denoise_with_cache(model, cache_d, rope_d, offset, prefix_mask, seed=base)
        cap.remove()
        assert counter.step == N_STEPS - 1, f"expected {N_STEPS} steps, got {counter.step + 1}"

        # ---- pruned pass ------------------------------------------------------
        if not args.no_mask:
            vmasks.set(q=vq, mlp=vm)
        taps_p = ResidualTaps(vlayers)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            cache_p, rope_p, nll_p = tf_forward(model, seq_tf, inputs, coc_start, coc_end)
        taps_p.remove()
        assert cache_p.get_seq_length() == prefill and torch.equal(rope_p, rope_d), \
            "G0: teacher forcing must give both passes identical cache positions"
        k0d, _ = lib.cache_layer_kv(cache_d, 0)
        k0p, _ = lib.cache_layer_kv(cache_p, 0)
        l0dk = float((k0d - k0p).abs().max())

        cap2 = ExpertAttnCapture(model, counter, prefill, sidx, ec.num_hidden_layers,
                                ec.num_attention_heads, "compare", store)
        counter.reset()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            lib.denoise_with_cache(model, cache_p, rope_p, offset, prefix_mask, seed=base)
        cap2.remove()
        counter.remove()
        vmasks.reset()

        r = residual_metrics(taps_d, taps_p, span_cpu)
        if ver is not None:
            r.update(ver)
        res_all.append(r)
        attn_all.append(cap2.out)

        rec = {"clip_id": clip_id, "prompt_len": int(prompt_len), "prefill": int(prefill),
               "coc_len": int(coc_end - coc_start), "seed": base,
               "nll_dense": nll_d, "nll_pruned": nll_p, "layer0_max_dk": l0dk,
               "rowsum_bf16_max": float(np.max(cap2.rowsum_bf16)),
               "spans_present": list(span_cpu),
               "tv_mean": float(cap2.out["tv"].mean()),
               "cross_rel_hout_all": float(r["cross_rel"][:, 2, 0].mean()),
               "sec": round(time.time() - t0, 1)}
        if blk is not None:
            rec["g1_bitwise_frac"] = float(r["g1_bitwise"].mean())
            rec["bf16_floor_rel"] = float(r["bf16_floor_rel"][:, 0].mean())
        rows.append(rec)
        del taps_d, taps_p
        print(f"[{ci + 1}/{len(rows_man)}] {clip_id[:8]} prefill={prefill} "
              f"coc={rec['coc_len']} L0dk={l0dk:.1e} tv={rec['tv_mean']:.4f} "
              f"rel={rec['cross_rel_hout_all']:.4f} ({rec['sec']}s)", flush=True)

        del cache_d, cache_p, store, inputs
        torch.cuda.empty_cache()
        if (ci + 1) % 5 == 0 or ci + 1 == len(rows_man):
            save(out_dir, rows, res_all, attn_all, meta, ci + 1)

    save(out_dir, rows, res_all, attn_all, meta, len(rows))
    print("saved ->", out_dir, flush=True)


if __name__ == "__main__":
    main()
