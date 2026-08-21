"""Wanda baseline scores for the u40_v2 cell (plans/2026-08-20_wanda-baseline.md).

Structured adaptation of Wanda (Sun et al. 2023, github.com/locuslab/wanda): per input
dim j of o_proj / down_proj, accumulate the official WrappedGPT statistic
scaler_row[j] = running mean over calibration clips of sum_tokens x_{t,j}^2 (fp32),
then score units with |W|-column norms:

  MLP channel c : sqrt(scaler_mlp[c]) * ||W_down[:, c]||_2
  Q head h      : sqrt( sum_{j in head} scaler_o[j] * ||W_o[:, j]||_2^2 )

L1-aggregation variants are stored alongside for the pre-registered sensitivity check;
selection uses the L2 scores. Comparison group is within-layer across units
(select_mask_ratios), the structured stand-in for the original per-output-row grouping.

Two token protocols (--tokens):
  all   every position of the calib_100 fused prompt, prefill-only forward (the analog
        of Wanda's unlabeled C4 pass). 93% of those positions are vision tokens.
  text  only non-vision positions: the prompt's text span (compute_spans: not vision,
        not traj-history, not sink) plus the model's OWN rolled-out CoC tokens, read
        in one teacher-forced forward over prompt+CoC. Still label-free -- the CoC is
        the model's, seeded from the clip id -- and it is the token set the reasoning
        channel actually runs on (the jlens precedent of restricting to text/CoC).

Gates restated from the plan: W0 masks must remove exactly 2,657,452,032 params at the
u40_v2 budget; W1 judges paired dminADE@6 (arm - dual) on test_500; W2 records degen,
delta vs baseline, and kept-set overlaps; W3 (text variant) judges paired
dminADE@6 (wanda_txt - wanda) -- does restricting to text tokens recover the collapse.

Usage:
  bash experiments/head_analysis/run_retry_host.sh 30 \
      experiments/head_analysis/run_wanda.py --gpu 4 [--num-clips 100] [--tokens text]
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

import analysis_lib as lib
import sample_cache as sc
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from expert_per_clip import reserve_gpu  # also installs the gated-repo hub patch
from slim_lib import MODEL_REV

REPO = Path(__file__).resolve().parents[2]


class WandaStats:
    """Official WrappedGPT accumulation, one clip = one sample (tmp=1)."""

    def __init__(self, layers):
        self.n = 0
        self.pos = None  # (T,) bool on cuda: positions to accumulate, None = all
        d_o = layers[0].self_attn.o_proj.in_features
        d_m = layers[0].mlp.down_proj.in_features
        self.scaler_o = [torch.zeros(d_o, dtype=torch.float32, device="cuda")
                         for _ in layers]
        self.scaler_m = [torch.zeros(d_m, dtype=torch.float32, device="cuda")
                         for _ in layers]
        self.handles = []
        for i, layer in enumerate(layers):
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(
                self._hook(self.scaler_o, i)))
            self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(
                self._hook(self.scaler_m, i)))

    def _hook(self, store, i):
        def fn(_module, args):
            x = args[0].reshape(-1, args[0].shape[-1])  # (T, D)
            if self.pos is not None:
                x = x[self.pos]
            x = x.float()
            # running mean over clips of per-clip token sum-of-squares; step() pre-scales
            # the previous mean before the clip's forward, done() advances n after it
            store[i] += (x * x).sum(0) / (self.n + 1)
        return fn

    def step(self):
        for s in (self.scaler_o, self.scaler_m):
            for t in s:
                t *= self.n / (self.n + 1)

    def done(self):
        self.n += 1

    def remove(self):
        for h in self.handles:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-clips", type=int, default=100)
    ap.add_argument("--exp-id", type=str, default="wanda_v1")
    ap.add_argument("--calib-manifest", default="calib_100")
    ap.add_argument("--cache", default="calib")
    ap.add_argument("--tokens", choices=["all", "text"], default="all")
    ap.add_argument("--max-gen", type=int, default=256, help="text mode: CoC rollout cap")
    ap.add_argument("--seed", type=int, default=42, help="text mode: rollout seed base")
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--gpu", type=str, default=None,
                    help="comma-separated card ids to restrict the scan")
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    calib = sc.calib_samples(REPO, args.calib_manifest)[: args.num_clips]

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

    layers = model.vlm.model.language_model.layers
    tc = model.vlm.config.text_config
    stats = WandaStats(layers)

    (out_dir / "config.json").write_text(json.dumps({
        "model": "nvidia/Alpamayo-1.5-10B", "model_revision": MODEL_REV,
        "purpose": "Wanda (|W| * ||X||_2) structured unit scores, official "
                   "WrappedGPT accumulation, no labels",
        "num_clips": len(calib), "clip_ids": [c for c, _ in calib],
        "calib_manifest": args.calib_manifest, "cache": args.cache,
        "tokens": args.tokens,
        "tokens_desc": {"all": "full fused prompt (vision + traj-history + text), "
                               "prefill only",
                        "text": "prompt text span (not vision/traj-history/sink) + "
                                "the model's own rolled-out CoC, teacher-forced forward"
                        }[args.tokens],
        "max_gen": args.max_gen, "seed": args.seed,
        "seed_rule": "sha256(f'{seed}:{clip_id}')[:4]",
        "aggregation": {"primary": "L2 over unit columns", "secondary": "L1"},
        "gpu": torch.cuda.get_device_name(device),
    }, indent=2))

    tokens_total = 0
    records = []
    for ci, (clip_id, clip_t0) in enumerate(calib):
        t0 = time.time()
        data = sc.load_cached(sc.path_for(args.cache, clip_id, clip_t0))
        inputs = lib.build_inputs(model, processor, data, "cuda")
        prompt_len = inputs["input_ids"].shape[1]
        if args.tokens == "all":
            seq, td, pos = inputs["input_ids"], inputs["tokenized_data"], None
            n_pos = prompt_len
            rec = {"clip_id": clip_id, "T": prompt_len}
        else:
            seed = sc.clip_seed(args.seed, clip_id)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
            eos_pos = roll["eos_pos"]
            seq = roll["sequences"][:, : eos_pos + 1]  # (1, T_prompt + T_coc + eos)
            del roll
            spans = lib.compute_spans(model, inputs["input_ids"])
            pos = torch.zeros(seq.shape[1], dtype=torch.bool)
            pos[:prompt_len] = spans["text"]          # prompt text, no vision/hist/sink
            pos[prompt_len:eos_pos] = True             # the model's own CoC tokens
            n_pos = int(pos.sum())
            td = dict(inputs["tokenized_data"])
            td["attention_mask"] = torch.ones_like(seq)
            rec = {"clip_id": clip_id, "T": int(seq.shape[1]),
                   "n_text_prompt": int(spans["text"].sum()),
                   "n_coc": int(eos_pos - prompt_len), "n_pos": n_pos}
            pos = pos.to("cuda")
        stats.pos = pos
        stats.step()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model.vlm(input_ids=seq, use_cache=False, **td)
        stats.done()
        stats.pos = None
        tokens_total += n_pos
        records.append(rec)
        extra = (f" text={rec['n_text_prompt']} coc={rec['n_coc']}"
                 if args.tokens == "text" else "")
        print(f"[{ci + 1}/{len(calib)}] {clip_id} T={rec['T']} pos={n_pos}{extra} "
              f"({time.time() - t0:.0f}s)", flush=True)
    stats.remove()

    L, H, hd = tc.num_hidden_layers, tc.num_attention_heads, tc.head_dim
    inter = tc.intermediate_size
    q_w = np.zeros((L, H)); q_w_l1 = np.zeros((L, H))
    mlp_w = np.zeros((L, inter)); mlp_w_l1 = np.zeros((L, inter))
    sc_o = np.zeros((L, H * hd), dtype=np.float32)
    sc_m = np.zeros((L, inter), dtype=np.float32)
    for i, layer in enumerate(layers):
        so = stats.scaler_o[i]                                  # (H*hd,)
        sm = stats.scaler_m[i]                                  # (inter,)
        wo = layer.self_attn.o_proj.weight.float()              # (hidden, H*hd)
        wd = layer.mlp.down_proj.weight.float()                 # (hidden, inter)
        col2_o = (wo * wo).sum(0)                               # ||W_o[:,j]||^2
        col1_o = wo.abs().sum(0)                                # ||W_o[:,j]||_1
        q_w[i] = (so * col2_o).view(H, hd).sum(1).sqrt().cpu().numpy()
        q_w_l1[i] = (so.sqrt() * col1_o).view(H, hd).sum(1).cpu().numpy()
        mlp_w[i] = (sm.sqrt() * (wd * wd).sum(0).sqrt()).cpu().numpy()
        mlp_w_l1[i] = (sm.sqrt() * wd.abs().sum(0)).cpu().numpy()
        sc_o[i] = so.cpu().numpy()
        sc_m[i] = sm.cpu().numpy()

    np.savez(out_dir / "wanda.npz", q_w=q_w, mlp_w=mlp_w,
             q_w_l1=q_w_l1, mlp_w_l1=mlp_w_l1, scaler_o=sc_o, scaler_m=sc_m)
    (out_dir / "records.json").write_text(json.dumps(records, indent=1))
    finite = all(np.isfinite(a).all() for a in (q_w, mlp_w, q_w_l1, mlp_w_l1))
    (out_dir / "summary.txt").write_text(
        f"wanda scores over {len(calib)} clips, tokens={args.tokens}, "
        f"{tokens_total} accumulated positions\n"
        f"finite: {finite}\n"
        f"q_w    range [{q_w.min():.4g}, {q_w.max():.4g}]\n"
        f"mlp_w  range [{mlp_w.min():.4g}, {mlp_w.max():.4g}]\n")
    print(f"saved -> {out_dir} | finite={finite} positions={tokens_total}", flush=True)
    assert finite, "non-finite wanda scores"


if __name__ == "__main__":
    main()
