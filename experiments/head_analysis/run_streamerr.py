"""Reconstruction error of the dualr arms, decomposed by token type.

plans/2026-09-04_stream-error-decomposition.md.

`analyze_dualrw.py` plotted "rel err on own-CoC tokens" per layer. That plot carries
no information: the least-squares refit minimises exactly the error that `H_fit`
weights, so an arm that put own-CoC in `H_fit` wins on own-CoC by construction, and
those numbers were in-sample besides. The informative question is what happens to the
token types the fit did NOT buy -- reconstruction is a budget, so paying for one
stream sells another. Nobody has measured that exchange rate here.

What the metric is (the earlier label was ambiguous): `tyr_lib.recon_error` is
`||(W - W_hat) X||_F / ||W X||_F` with X the input to that layer's o_proj / down_proj.
So it is the relative change in **that sublayer's write into the residual stream**, at
the selected token positions -- not the KV cache (layer l's write only reaches the
cache through layer l+1's k/v_proj) and not the accumulated hidden state (each layer is
measured independently on dense inputs). Token type selects WHERE the error is read.

Five streams, the repo's standard split (expert_per_clip.REGIONS):

    vision · prompt_text (instruction) · hist (ego history) · sink · coc (own rollout)

which refines racfit's V/T/D, where V lumped vision+hist+sink together.

Hessians are never formed. At a forward hook we already have y = Wx (the module's own
output) and can form y_hat = W_hat x_kept directly, so accumulating scalar
sum||y-y_hat||^2 and sum||y||^2 per stream gives the same ratio with constant memory --
a 5-way split of down_proj Hessians would otherwise cost 5 x 604 MB per layer.

Clips come from `indist_500`, never from the fit's calibration set, so unlike the
original figure these errors are held out.

Gates (from the plan):
  H1  err_coc falls and err_vision rises between dualr_rep (CoC share 0) and dualr_w
      (0.16), both CIs excluding 0 -> the exchange is real.
  H2  within an arm, per-type error ranks anti-correlate with that type's fit weight.
  H3  hist / sink are structurally abandoned (large error in every arm).

Usage:
  bash experiments/head_analysis/run_retry_host.sh 30 \
      experiments/head_analysis/run_streamerr.py --num-clips 24 --gpu 6
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
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from expert_per_clip import reserve_gpu  # also installs the gated-repo hub patch
from slim_lib import MODEL_REV

REPO = Path(__file__).resolve().parents[2]
STREAMS = ["vision", "prompt_text", "hist", "sink", "coc"]
# each arm's slim dir and the supernet whose metadata records what its H was fitted on
ARMS = {
    "dualr": ("slim_dualr_u40", "dualr_supernet_u40"),
    "dualr_rep": ("slim_dualr_rep_u40", "dualr_rep_supernet_u40"),
    "dualr_w": ("slim_dualr_w_u40", "dualr_w_supernet_u40"),
    "dualr_wl": ("slim_dualr_wl_u40", "dualr_wl_supernet_u40"),
}


def load_arm_projections(slim_dir, n_layers, device):
    """The arm's refitted o_proj / down_proj weights plus the kept input columns.

    The checkpoint stores physically narrowed matrices, so the kept indices from
    slim_meta.json say which dense input columns they consume. o_proj's kept columns
    are per Q head (head_dim wide); down_proj's are per MLP channel.
    """
    d = REPO / "outputs" / slim_dir
    meta = json.loads((d / "slim_meta.json").read_text())
    state = torch.load(d / "slim_state.pt", map_location="cpu", mmap=True)
    head_dim = 128
    out = {}
    for li in range(n_layers):
        q = torch.tensor(sorted(meta["vlm"][li]["q"]))
        cols_o = (q[:, None] * head_dim + torch.arange(head_dim)).reshape(-1)
        cols_m = torch.tensor(sorted(meta["vlm"][li]["mlp"]))
        po = f"vlm.model.language_model.layers.{li}.self_attn.o_proj.weight"
        pm = f"vlm.model.language_model.layers.{li}.mlp.down_proj.weight"
        out[(li, "o")] = (state[po].to(device, torch.bfloat16), cols_o.to(device))
        out[(li, "m")] = (state[pm].to(device, torch.bfloat16), cols_m.to(device))
    del state
    return out, meta


class StreamErrorHook:
    """Per-(arm, stream) squared error of W_hat x against the dense module's own y."""

    def __init__(self, module, key, arms, acc):
        self.key, self.arms, self.acc = key, arms, acc
        self.masks = None
        self.handle = module.register_forward_hook(self._hook)

    @torch.no_grad()
    def _hook(self, _m, args, output):
        x = args[0].reshape(-1, args[0].shape[-1])        # (T, d_in)
        y = output.reshape(-1, output.shape[-1]).float()  # (T, d_out)
        for s, m in self.masks.items():
            ys = y[m]
            if ys.shape[0] == 0:
                continue
            self.acc[(self.key, "den", s)] += float(ys.pow(2).sum())
        for arm, proj in self.arms.items():
            w, cols = proj[self.key]
            yh = torch.nn.functional.linear(x[:, cols], w).float()   # (T, d_out)
            dy = y - yh
            for s, m in self.masks.items():
                if m.sum() == 0:
                    continue
                self.acc[(self.key, arm, s)] += float(dy[m].pow(2).sum())

    def remove(self):
        self.handle.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-clips", type=int, default=24)
    ap.add_argument("--set", default="indist", help="eval manifest the clips come from")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--reserve-gb", type=float, default=40.0)
    ap.add_argument("--exp-id", default="streamerr_v1")
    args = ap.parse_args()

    device = reserve_gpu(args.reserve_gb,
                         devices=None if args.gpu is None else [args.gpu])
    out = REPO / "outputs" / args.exp_id
    (out / "plots").mkdir(parents=True, exist_ok=True)

    # dense reference model: its own o_proj / down_proj outputs are the denominator
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, dtype=torch.bfloat16).to(device)
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")
    layers = model.vlm.model.language_model.layers
    n_layers = len(layers)

    arms, metas = {}, {}
    for name, (slim_dir, supernet) in ARMS.items():
        t0 = time.time()
        arms[name], metas[name] = load_arm_projections(slim_dir, n_layers, device)
        sup = REPO / "outputs" / supernet / "metadata.json"
        metas[name] = json.loads(sup.read_text()).get("hessian_tokens", "?") if sup.exists() else "?"
        print(f"[arm] {name:10s} loaded in {time.time() - t0:.0f}s | H fitted on: "
              f"{metas[name][:90]}", flush=True)

    acc = {}
    for li in range(n_layers):
        for kind in ("o", "m"):
            for s in STREAMS:
                acc[((li, kind), "den", s)] = 0.0
                for a in arms:
                    acc[((li, kind), a, s)] = 0.0

    hooks = {}
    for li in range(n_layers):
        hooks[(li, "o")] = StreamErrorHook(layers[li].self_attn.o_proj, (li, "o"), arms, acc)
        hooks[(li, "m")] = StreamErrorHook(layers[li].mlp.down_proj, (li, "m"), arms, acc)

    stem = "indist_500" if args.set == "indist" else f"{args.set}_500"
    df = pd.read_parquet(REPO / "outputs" / "eval_sets" / f"{stem}.parquet")
    cache = "eval" if args.set == "indist" else "test"
    manifest = [{"clip_id": r.clip_id, "t0_us": int(r.t0_us)}
                for r in df.itertuples()][: args.num_clips]
    tok_counts = {s: 0 for s in STREAMS}
    for ci, m in enumerate(manifest):
        t0 = time.time()
        data = sc.load_cached(sc.path_for(cache, m["clip_id"], m["t0_us"]))
        inp = lib.build_inputs(model, processor, data, device)
        spans = lib.compute_spans(model, inp["input_ids"])
        prompt_len = inp["input_ids"].shape[1]
        s = sc.clip_seed(args.seed, m["clip_id"])
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
        for h in hooks.values():
            h.masks = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(model, inp, max_generation_length=args.max_gen)
        coc = roll["sequences"][0, prompt_len:roll["eos_pos"]]
        ids = torch.cat([inp["input_ids"][0], coc]).unsqueeze(0)
        t_total = ids.shape[1]
        masks = {}
        for s_ in ("vision", "hist", "sink"):
            full = torch.zeros(t_total, dtype=torch.bool, device=device)
            full[:prompt_len] = spans[s_].to(device)
            masks[s_] = full
        pt = torch.zeros(t_total, dtype=torch.bool, device=device)
        pt[:prompt_len] = spans["text"].to(device)
        masks["prompt_text"] = pt
        cm = torch.zeros(t_total, dtype=torch.bool, device=device)
        cm[prompt_len:] = True
        masks["coc"] = cm
        for k, v in masks.items():
            tok_counts[k] += int(v.sum())
        for h in hooks.values():
            h.masks = masks
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model.vlm.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                            pixel_values=inp["tokenized_data"]["pixel_values"],
                            image_grid_thw=inp["tokenized_data"]["image_grid_thw"],
                            use_cache=False)
        for h in hooks.values():
            h.masks = {}
        del roll, inp
        torch.cuda.empty_cache()
        print(f"[{ci + 1}/{len(manifest)}] {m['clip_id']} prompt={prompt_len} "
              f"coc={len(coc)} ({time.time() - t0:.0f}s)", flush=True)

    for h in hooks.values():
        h.remove()

    res = {"streams": STREAMS, "arms": list(ARMS), "hessian_tokens": metas,
           "token_counts": tok_counts, "n_clips": len(manifest), "err": {}, "den": {}}
    # the dense output energy per stream: a stream's share of it is what an error there
    # actually costs the model, and it is not the token share (vision tokens carry less
    # energy each than CoC tokens). Without this, "vision got 4% worse and CoC 24%
    # better" cannot be added up.
    for kind in ("o", "m"):
        for s in STREAMS:
            res["den"][f"{kind}_{s}"] = [acc[((li, kind), "den", s)] for li in range(n_layers)]
    for a in arms:
        res["err"][a] = {}
        for kind in ("o", "m"):
            for s in STREAMS:
                res["err"][a][f"{kind}_{s}"] = [
                    float(np.sqrt(acc[((li, kind), a, s)] / acc[((li, kind), "den", s)]))
                    if acc[((li, kind), "den", s)] > 0 else float("nan")
                    for li in range(n_layers)]
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    (out / "config.json").write_text(json.dumps(vars(args) | {"arms": ARMS}, indent=1))
    lines = [(f"per-token-type reconstruction error, {len(manifest)} held-out clips "
              f"from {args.set}"), ""]
    lines.append(f"{'arm':11s} {'module':9s} " + " ".join(f"{s:>12s}" for s in STREAMS))
    for a in arms:
        for kind, lab in (("o", "o_proj"), ("m", "down_proj")):
            med = [np.nanmedian(res["err"][a][f"{kind}_{s}"]) for s in STREAMS]
            lines.append(f"{a:11s} {lab:9s} " + " ".join(f"{v:12.4f}" for v in med))
    joined = "  ".join(f"{s} {tok_counts[s]}" for s in STREAMS)
    lines += ["", f"tokens per stream (all clips): {joined}"]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
