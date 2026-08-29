"""Cache-targeted reconstruction (plans/2026-08-29_cache-targeted-reconstruction.md).

Keeps dual_u40_v2's selection everywhere and OSSCAR-refits o_proj / down_proj ONLY in the
layers whose outputs feed the cache the expert is sensitive to (layers >= --start; the
cache-use map put 92% of the sensitivity-weighted shift in layers 20-35). Two things
differ from the dualr supernet (run_tyr_supernet.py --selection dual):

  * the Hessian is a per-token WEIGHTED sum, H = sum_t w_t x_t x_t^T, where a prefill
    token's weight is its span's expert attention share (vision 0.72, text 0.17,
    hist 0.04, sink 0.01 -- run_cacheuse Stage A, per-clip token-mean normalised) and
    the model's own rolled-out CoC (racfit_v1/rollouts.json, K seeds) carries
    --decode-share of the mixture (racfit's d10 corner, 0.16, the mixture that cut the
    held-out decode error without hurting the prefill streams). Tyr/dualr fit prefill
    only, which is what damaged the decode path.
  * block-sequential, like the Tyr / dualr supernet: for target layer l the forward runs
    with dual's masks on layers < l (and the already-refitted layers >= start written back,
    zero columns included) while layer l's OWN o_proj / down_proj inputs are unmasked --
    the removed columns must be present in H, or the least-squares refit is the identity
    (H_{k,removed} = 0 makes W_kept = H_kk^-1 H_kk W_k = W_k; the first build of this
    script masked every layer and reproduced dual_u40_v2 bit for bit).

Writes outputs/<exp-id>/layers.NN.{self_attn.o_proj,mlp.down_proj}/0.pth in the tyr
supernet layout (full (out, in) bf16 tensors, removed columns exactly zero), plus
metadata.json. make_slim.py --config dualrc_u40_s<start> loads them; a supernet built
with --start 16 also serves dualrc_u40_s24 because layers are refitted independently.

Usage:
  bash experiments/head_analysis/run_retry_host.sh 60 experiments/head_analysis/run_cache_recon.py \
      --gpu 0 --start 16 --exp-id dualrc_supernet_u40
  # smoke: --num-clips 2 --start 34 --k-seeds 1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
import analysis_lib as lib  # noqa: E402
import mask_lib as ml  # noqa: E402
import sample_cache as sc  # noqa: E402
import tyr_lib as tyr  # noqa: E402
from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  also installs the gated-repo hub patch
from make_slim import build_masks  # noqa: E402
from slim_lib import MODEL_REV  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
# expert attention mass per cache span, run_cacheuse Stage A (calib_100, step/layer/head mean)
SPAN_SHARE = {"vision": 0.7223, "text": 0.1656, "hist": 0.0418, "sink": 0.0110}


class WeightedHessianHook:
    """H = sum_t w_t x_t x_t^T at a linear module's input; `weights` (T,) is set by the
    caller before each forward (zero drops a token)."""

    def __init__(self, module):
        d = module.in_features
        self.H = torch.zeros((d, d), device=module.weight.device, dtype=torch.float32)
        self.wsum = 0.0
        self.weights = None
        self.handle = module.register_forward_pre_hook(self._hook)

    @torch.no_grad()
    def _hook(self, _module, args):
        x = args[0].reshape(-1, args[0].shape[-1]).float()  # (T, d)
        w = self.weights
        keep = w > 0
        xs = x[keep] * w[keep].sqrt()[:, None]
        self.H += xs.t() @ xs
        self.wsum += float(w[keep].sum())

    def remove(self):
        self.handle.remove()


def token_weights(spans, prompt_len, n_coc, decode_share, first_pass, k_seeds):
    """Per-token weights for one [prompt; CoC] forward. Prefill spans get their expert
    attention share (only on the first of the K passes -- prefill activations are causal
    and identical across them), the CoC gets decode_share / K; each stream is
    token-mean normalised within the clip."""
    T = prompt_len + n_coc
    w = torch.zeros(T, dtype=torch.float32)
    if first_pass:
        for name, share in SPAN_SHARE.items():
            m = spans[name].cpu()
            n = int(m.sum())
            if n:
                w[:prompt_len][m] = (1.0 - decode_share) * share / n
    if n_coc:
        w[prompt_len:] = decode_share / (k_seeds * n_coc)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="dualrc_supernet_u40")
    ap.add_argument("--start", type=int, default=16, help="first layer to refit")
    ap.add_argument("--end", type=int, default=36, help="one past the last layer to refit")
    ap.add_argument("--calib-manifest", default="calib_100")
    ap.add_argument("--cache", default="calib")
    ap.add_argument("--num-clips", type=int, default=100)
    ap.add_argument("--rollouts-from", default="racfit_v1",
                    help="exp-id holding rollouts.json (K on-policy CoCs per calib clip)")
    ap.add_argument("--k-seeds", type=int, default=4)
    ap.add_argument("--decode-share", type=float, default=0.16)
    ap.add_argument("--damp", type=float, default=1e-2)
    ap.add_argument("--importance", default="importance_v2")
    ap.add_argument("--reserve-gb", type=float, default=40.0)
    ap.add_argument("--gpu", type=str, default=None)
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    calib = sc.calib_samples(REPO, args.calib_manifest)[: args.num_clips]
    rolls = json.loads((REPO / "outputs" / args.rollouts_from / "rollouts.json").read_text())
    devices = None if args.gpu is None else [int(x) for x in args.gpu.split(",")]
    device = reserve_gpu(args.reserve_gb, devices=devices)
    print(f"using {device}", flush=True)
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")
    tc = model.vlm.config.text_config
    layers = model.vlm.model.language_model.layers

    imp = dict(np.load(REPO / "outputs" / args.importance / "importance.npz"))
    vq, vm, _, _, kvonly = build_masks("dual_u40_v2", imp, model)  # (36, 32), (36, 12288)
    assert not kvonly
    vmasks = ml.PruneMasks(layers, tc.num_attention_heads, tc.head_dim, tc.intermediate_size,
                           "cuda")
    vmasks.set(q=vq, mlp=vm)
    keep_q = int(vq[0].sum())
    keep_m = int(vm[0].sum())
    print(f"dual masks applied: keep {keep_q}/32 heads, {keep_m}/12288 channels per layer",
          flush=True)

    # inputs once (pixel tensors on CPU), token weights once per (clip, coc)
    print(f"preloading {len(calib)} clip inputs...", flush=True)
    store = []
    for clip_id, t0_us in calib:
        data = sc.load_cached(sc.path_for(args.cache, clip_id, t0_us))
        inp = lib.build_inputs(model, processor, data, "cuda")
        spans = lib.compute_spans(model, inp["input_ids"])
        prompt_len = inp["input_ids"].shape[1]
        rec = rolls["clips"][clip_id]
        assert rec["prompt_len"] == prompt_len, (clip_id, rec["prompt_len"], prompt_len)
        cocs = rec["coc"][: args.k_seeds]
        passes = []
        for j, coc in enumerate(cocs):
            ids = torch.cat([inp["input_ids"].cpu(),
                             torch.tensor(coc, dtype=inp["input_ids"].dtype)[None, :]], dim=1)
            w = token_weights(spans, prompt_len, len(coc), args.decode_share, j == 0, len(cocs))
            passes.append((ids, w))
        store.append({"clip_id": clip_id, "passes": passes, "prompt_len": prompt_len,
                      "pixel_values": inp["tokenized_data"]["pixel_values"].cpu(),
                      "image_grid_thw": inp["tokenized_data"]["image_grid_thw"].cpu()})
        del inp, data
    n_tok = {"prefill": sum(it["prompt_len"] for it in store),
             "decode": sum(ids.shape[1] - it["prompt_len"] for it in store
                           for ids, _ in it["passes"])}

    nh, hd = tc.num_attention_heads, tc.head_dim
    layer_names = []
    t_all = time.time()
    for i in range(args.start, args.end):
        t0 = time.time()
        # masks on layers < i (layers >= start already carry their refitted, zero-column
        # weights); layer i's own inputs unmasked so H sees the removed columns
        q_i, m_i = vq.copy(), vm.copy()
        q_i[i:] = 1.0
        m_i[i:] = 1.0
        vmasks.set(q=q_i, mlp=m_i)
        o_proj, down = layers[i].self_attn.o_proj, layers[i].mlp.down_proj
        hooks = {"self_attn.o_proj": WeightedHessianHook(o_proj),
                 "mlp.down_proj": WeightedHessianHook(down)}
        for item in store:
            for ids, w in item["passes"]:
                wc = w.to("cuda")
                for h in hooks.values():
                    h.weights = wc
                ids = ids.cuda()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model.vlm.model(
                        input_ids=ids, attention_mask=torch.ones_like(ids),
                        pixel_values=item["pixel_values"].cuda(),
                        image_grid_thw=item["image_grid_thw"].cuda(), use_cache=False)
        for h in hooks.values():
            h.remove()
        t_fwd = time.time() - t0
        for suffix, mod, keep, n_groups, gs, mask_row in (
            ("self_attn.o_proj", o_proj, keep_q, nh, hd, vq[i]),
            ("mlp.down_proj", down, keep_m, tc.intermediate_size, 1, vm[i]),
        ):
            name = f"layers.{i:02d}.{suffix}"
            hook = hooks[suffix]
            kept_g = [g for g in range(n_groups) if mask_row[g] > 0]
            kept_cols = [g * gs + d for g in kept_g for d in range(gs)]
            sols = tyr.reconstruct_levels(mod, hook.H, {keep: kept_cols}, damp=args.damp)
            w = sols[keep]
            n_zero = int((w.abs().sum(0).reshape(n_groups, gs).sum(1) == 0).sum())
            assert n_zero == n_groups - keep, (name, n_zero, keep)
            rel = float((w - mod.weight.data.float()).norm() / mod.weight.data.float().norm())
            d = out_dir / name
            d.mkdir(parents=True, exist_ok=True)
            torch.save(w.to(torch.bfloat16).cpu(), d / "0.pth")
            layer_names.append(name)
            # error accumulation: the refitted weight (zero columns included) goes into
            # the model, so the next block sees the pruned-and-refitted upstream
            mod.weight.data.copy_(w.to(mod.weight.dtype))
            print(f"  {name}: refit rel change {rel:.4f}", flush=True)
        del hooks
        torch.cuda.empty_cache()
        print(f"[block {i - args.start + 1}/{args.end - args.start}] layer {i} fwd {t_fwd:.0f}s "
              f"total {time.time() - t0:.0f}s", flush=True)
    vmasks.reset()
    t_fwd = time.time() - t_all

    (out_dir / "metadata.json").write_text(json.dumps({
        "model_revision": MODEL_REV, "num_clips": len(calib),
        "clip_ids": [c for c, _ in calib], "selection": "dual",
        "importance": args.importance, "damp": args.damp,
        "start": args.start, "end": args.end, "layer_names": layer_names,
        "levels_q": {"0": keep_q}, "levels_mlp": {"0": keep_m}, "num_levels": 1,
        "hessian_tokens": ("per-token weighted: prefill spans by expert attention share "
                           f"{SPAN_SHARE}, own CoC (K={args.k_seeds}, {args.rollouts_from}) "
                           f"at decode share {args.decode_share}; block-sequential, dual "
                           "masks on layers < l, refitted layers written back, layer l's "
                           "own inputs unmasked"),
        "tokens": n_tok, "plan": "plans/2026-08-29_cache-targeted-reconstruction.md",
    }, indent=2))
    (out_dir / "summary.txt").write_text(
        f"cache-targeted reconstruction: layers {args.start}..{args.end - 1}, {len(calib)} "
        f"clips x K={args.k_seeds} CoCs, decode share {args.decode_share}, damp {args.damp}; "
        f"forwards {t_fwd:.0f}s, total {time.time() - t_all:.0f}s\n")
    print("saved ->", out_dir, flush=True)


if __name__ == "__main__":
    main()
