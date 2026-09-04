"""The same token-type decomposition, on the PROPAGATED quantities.

plans/2026-09-04_stream-error-decomposition.md, part 2.

`run_streamerr.py` measures a **local** quantity: how much one sublayer's write into
the residual stream changes, with dense inputs, layer by layer. That is what
`tyr_lib.recon_error` has always been, and it is why the reconstruction family's
error ranking need not match its capability ranking -- local errors can cancel or
compound once they propagate.

This runner measures the two propagated quantities the local one is upstream of:

  hidden   ||h_l(arm) - h_l(dense)|| / ||h_l(dense)||   accumulated residual stream
  K, V     the same on the per-layer KV cache the expert actually reads

both decomposed by the five token types (vision / prompt_text / hist / sink / coc).
K is direction-dominated (Qwen3 applies k_norm per head), so a mean per-head cosine
is reported beside the relative L2.

Alignment: both models are teacher-forced on the SAME sequence -- prompt plus the
DENSE model's own rollout -- so token positions correspond. An arm's own rollout
would drift and make position-wise differences meaningless.

Shapes are comparable by construction: pruning is within-layer (heads / channels),
so hidden_size stays 4096, and k/v projections are never touched, so the cache stays
(1, 8, T, 128).

Dense and arm live on separate GPUs: load_slim rebuilds a full skeleton before
surgery, so one card would transiently hold 44 GB.

Usage:
  bash experiments/head_analysis/run_retry_host.sh 30 \
      experiments/head_analysis/run_streamprop.py --num-clips 24 \
      --gpu-dense 4 --gpu-arm 5
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

import analysis_lib as lib
import sample_cache as sc
import slim_lib as sl
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from expert_per_clip import reserve_gpu  # also installs the gated-repo hub patch
from slim_lib import MODEL_REV

REPO = Path(__file__).resolve().parents[2]
STREAMS = ["vision", "prompt_text", "hist", "sink", "coc"]
ARMS = {"dualr": "slim_dualr_u40", "dualr_rep": "slim_dualr_rep_u40",
        "dualr_w": "slim_dualr_w_u40", "dualr_wl": "slim_dualr_wl_u40"}


def masks_for(spans, prompt_len, total, device):
    m = {}
    for s in ("vision", "hist", "sink"):
        full = torch.zeros(total, dtype=torch.bool, device=device)
        full[:prompt_len] = spans[s].to(device)
        m[s] = full
    pt = torch.zeros(total, dtype=torch.bool, device=device)
    pt[:prompt_len] = spans["text"].to(device)
    m["prompt_text"] = pt
    coc = torch.zeros(total, dtype=torch.bool, device=device)
    coc[prompt_len:] = True
    m["coc"] = coc
    return m


@torch.no_grad()
def forward_states(model, ids, pixel_values, image_grid_thw):
    """Per-layer hidden states and the KV cache from one teacher-forced pass."""
    out = model.vlm.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                          pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                          use_cache=True, output_hidden_states=True)
    n = len(model.vlm.model.language_model.layers)
    kv = [tuple(t.detach() for t in lib.cache_layer_kv(out.past_key_values, i))
          for i in range(n)]
    hs = [h.detach() for h in out.hidden_states]        # n+1 tensors (1, T, 4096)
    return hs, kv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-clips", type=int, default=24)
    ap.add_argument("--set", default="indist")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--gpu-dense", type=int, required=True)
    ap.add_argument("--gpu-arm", type=int, default=None)
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--exp-id", default="streamprop_v1")
    ap.add_argument("--energy-only", action="store_true",
                    help="dense pass only: write the per-stream energies into an existing "
                         "run's metrics.json (one GPU, no arm forwards)")
    args = ap.parse_args()

    dev_d = reserve_gpu(args.reserve_gb, devices=[args.gpu_dense])
    dev_a = dev_d if args.energy_only else reserve_gpu(args.reserve_gb,
                                                       devices=[args.gpu_arm])
    out = REPO / "outputs" / args.exp_id
    (out / "plots").mkdir(parents=True, exist_ok=True)

    dense = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, dtype=torch.bfloat16).to(dev_d)
    dense.eval()
    for p in dense.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(dense.tokenizer)
    lib.set_vlm_attn_impl(dense, "sdpa")
    lib.set_expert_attn_impl(dense, "sdpa")
    n_layers = len(dense.vlm.model.language_model.layers)

    df = pd.read_parquet(REPO / "outputs" / "eval_sets" / "indist_500.parquet")
    cache = "eval" if args.set == "indist" else "test"
    manifest = [{"clip_id": r.clip_id, "t0_us": int(r.t0_us)}
                for r in df.itertuples()][: args.num_clips]

    # one dense pass per clip, kept on CPU: the same reference serves every arm, so the
    # dense model is loaded and run once instead of four times
    ref = []
    tok_counts = {s: 0 for s in STREAMS}
    for ci, m in enumerate(manifest):
        t0 = time.time()
        data = sc.load_cached(sc.path_for(cache, m["clip_id"], m["t0_us"]))
        inp = lib.build_inputs(dense, processor, data, dev_d)
        spans = lib.compute_spans(dense, inp["input_ids"])
        prompt_len = inp["input_ids"].shape[1]
        s = sc.clip_seed(args.seed, m["clip_id"])
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(dense, inp, max_generation_length=args.max_gen)
        coc = roll["sequences"][0, prompt_len:roll["eos_pos"]].cpu()
        ids = torch.cat([inp["input_ids"][0].cpu(), coc]).unsqueeze(0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hs, kv = forward_states(dense, ids.to(dev_d),
                                    inp["tokenized_data"]["pixel_values"],
                                    inp["tokenized_data"]["image_grid_thw"])
        msk = masks_for(spans, prompt_len, ids.shape[1], "cpu")
        for k, v in msk.items():
            tok_counts[k] += int(v.sum())
        ref.append({
            "ids": ids, "prompt_len": prompt_len, "masks": msk,
            "pixel_values": inp["tokenized_data"]["pixel_values"].cpu(),
            "image_grid_thw": inp["tokenized_data"]["image_grid_thw"].cpu(),
            "hs": [h.cpu() for h in hs],
            "kv": [(k.cpu(), v.cpu()) for k, v in kv],
        })
        del roll, inp, hs, kv
        torch.cuda.empty_cache()
        print(f"[dense {ci + 1}/{len(manifest)}] {m['clip_id']} T={ids.shape[1]} "
              f"coc={len(coc)} ({time.time() - t0:.0f}s)", flush=True)

    if args.energy_only:
        # ||dense||^2 per (quantity, stream, layer): the weights that turn a per-stream
        # relative error into what it costs the model. Comparing an energy-weighted
        # verdict against an unweighted one is how a wrong conclusion gets made.
        den = {}
        for q in ("h", "k", "v"):
            nl = n_layers + 1 if q == "h" else n_layers
            for s in STREAMS:
                den[f"{q}_{s}"] = [0.0] * nl
        for r in ref:
            for q in ("h", "k", "v"):
                series = (r["hs"] if q == "h"
                          else [kv[0 if q == "k" else 1] for kv in r["kv"]])
                for li, xd0 in enumerate(series):
                    xd = xd0.to(dev_d, torch.float32)
                    xd = (xd[0].permute(1, 0, 2).reshape(xd.shape[2], -1)
                          if xd.dim() == 4 else xd[0])
                    for s, mk0 in r["masks"].items():
                        mk = mk0.to(dev_d)
                        if mk.sum() == 0:
                            continue
                        den[f"{q}_{s}"][li] += float(xd[mk].pow(2).sum())
        mp = out / "metrics.json"
        res = json.loads(mp.read_text())
        res["den"], res["token_counts"] = den, tok_counts
        mp.write_text(json.dumps(res, indent=1))
        print(f"energies written into {mp}")
        return

    del dense
    torch.cuda.empty_cache()

    acc = {}
    for a in ARMS:
        for q in ("h", "k", "v"):
            for s in STREAMS:
                for li in range(n_layers + 1 if q == "h" else n_layers):
                    acc[(a, q, s, li, "num")] = 0.0
                    acc[(a, q, s, li, "den")] = 0.0
                    acc[(a, q, s, li, "cos")] = 0.0
                    acc[(a, q, s, li, "cnt")] = 0.0

    for a, slim_dir in ARMS.items():
        t0 = time.time()
        arm = sl.load_slim(str(REPO / "outputs" / slim_dir), device=str(dev_a))
        arm.eval()
        lib.set_vlm_attn_impl(arm, "sdpa")
        lib.set_expert_attn_impl(arm, "sdpa")
        print(f"[arm] {a} loaded in {time.time() - t0:.0f}s", flush=True)
        for ci, r in enumerate(ref):
            t1 = time.time()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                hs, kv = forward_states(arm, r["ids"].to(dev_a),
                                        r["pixel_values"].to(dev_a),
                                        r["image_grid_thw"].to(dev_a))
            for q, series in (("h", list(zip(hs, r["hs"]))),
                              ("k", [(kv[i][0], r["kv"][i][0]) for i in range(n_layers)]),
                              ("v", [(kv[i][1], r["kv"][i][1]) for i in range(n_layers)])):
                for li, (xa, xd) in enumerate(series):
                    xa = xa.to(dev_d, torch.float32)
                    xd = xd.to(dev_d, torch.float32)
                    if xa.dim() == 4:                       # (1, 8, T, 128) -> (T, 8*128)
                        xa = xa[0].permute(1, 0, 2).reshape(xa.shape[2], -1)
                        xd = xd[0].permute(1, 0, 2).reshape(xd.shape[2], -1)
                    else:
                        xa, xd = xa[0], xd[0]               # (T, 4096)
                    d = xa - xd
                    for s, mk in r["masks"].items():
                        mk = mk.to(dev_d)
                        if mk.sum() == 0:
                            continue
                        acc[(a, q, s, li, "num")] += float(d[mk].pow(2).sum())
                        acc[(a, q, s, li, "den")] += float(xd[mk].pow(2).sum())
                        cs = torch.nn.functional.cosine_similarity(
                            xa[mk], xd[mk], dim=-1)
                        acc[(a, q, s, li, "cos")] += float(cs.sum())
                        acc[(a, q, s, li, "cnt")] += float(mk.sum())
            del hs, kv
            torch.cuda.empty_cache()
            print(f"  [{a} {ci + 1}/{len(ref)}] ({time.time() - t1:.0f}s)", flush=True)
        del arm
        torch.cuda.empty_cache()

    res = {"streams": STREAMS, "arms": list(ARMS), "n_clips": len(ref),
           "token_counts": tok_counts, "n_layers": n_layers, "rel": {}, "cos": {},
           "den": {}}
    # the dense energy per stream, so the propagated trade can be weighted the same way
    # the local one was -- comparing an energy-weighted verdict against an unweighted one
    # is exactly how the tail-selection error happened
    a0 = next(iter(ARMS))
    for q in ("h", "k", "v"):
        nl = n_layers + 1 if q == "h" else n_layers
        for s in STREAMS:
            res["den"][f"{q}_{s}"] = [acc[(a0, q, s, li, "den")] for li in range(nl)]
    for a in ARMS:
        res["rel"][a], res["cos"][a] = {}, {}
        for q in ("h", "k", "v"):
            nl = n_layers + 1 if q == "h" else n_layers
            for s in STREAMS:
                res["rel"][a][f"{q}_{s}"] = [
                    float(np.sqrt(acc[(a, q, s, li, "num")] / acc[(a, q, s, li, "den")]))
                    if acc[(a, q, s, li, "den")] > 0 else float("nan") for li in range(nl)]
                res["cos"][a][f"{q}_{s}"] = [
                    float(acc[(a, q, s, li, "cos")] / acc[(a, q, s, li, "cnt")])
                    if acc[(a, q, s, li, "cnt")] > 0 else float("nan") for li in range(nl)]
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    (out / "config.json").write_text(json.dumps(vars(args) | {"arms": ARMS}, indent=1))
    lines = [f"propagated per-token-type divergence, {len(ref)} held-out clips", ""]
    for q, lab in (("h", "hidden"), ("k", "cache K"), ("v", "cache V")):
        lines.append(f"{lab} -- relative L2, median over layers")
        lines.append(f"{'arm':11s} " + " ".join(f"{s:>13s}" for s in STREAMS))
        for a in ARMS:
            lines.append(f"{a:11s} " + " ".join(
                f"{np.nanmedian(res['rel'][a][f'{q}_{s}']):13.4f}" for s in STREAMS))
        lines.append("")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
