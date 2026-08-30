"""G0 in FUNCTION space: do the rep refit and dualr's refit agree on the data, even where
their weights differ? o_proj inputs are collinear across heads, so the least-squares
solution is not identifiable in weight space (layer-1 o_proj differs 58x) -- what has to
agree is W' X on calibration activations. For a few layers this recollects H = sum x x^T
at the module input on the DENSE model (10 clips) and reports
  between = ||(W'_rep - W'_dualr) X|| / ||W X||     (agreement of the two refits)
  rep, dualr = ||(W - W') X|| / ||W X||               (each refit's own error, context)

Usage:
  .venv/bin/python experiments/head_analysis/check_dualr_rep_fn.py --gpu 0 --layers 0 1 16 35
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
import analysis_lib as lib  # noqa: E402
import sample_cache as sc  # noqa: E402
import tyr_lib as tyr  # noqa: E402
from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402
from slim_lib import MODEL_REV  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--supernet", default="dualr_rep_supernet_u40")
    ap.add_argument("--ref", default="slim_dualr_u40")
    ap.add_argument("--layers", nargs="+", type=int, default=[0, 1, 16, 35])
    ap.add_argument("--num-clips", type=int, default=10)
    ap.add_argument("--gpu", type=str, default=None)
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    args = ap.parse_args()
    reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [int(args.gpu)])
    model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", revision=MODEL_REV,
                                        dtype=torch.bfloat16).to("cuda")
    model.eval()
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    layers = model.vlm.model.language_model.layers
    meta = json.loads((REPO / "outputs" / args.ref / "slim_meta.json").read_text())
    sd = torch.load(REPO / "outputs" / args.ref / "slim_state.pt", map_location="cpu",
                    weights_only=True)
    hooks = {}
    for li in args.layers:
        hooks[(li, "self_attn.o_proj")] = tyr.HessianHook(layers[li].self_attn.o_proj)
        hooks[(li, "mlp.down_proj")] = tyr.HessianHook(layers[li].mlp.down_proj)
    for clip_id, t0 in sc.calib_samples(REPO, "calib_100")[: args.num_clips]:
        data = sc.load_cached(sc.path_for("calib", clip_id, t0))
        inp = lib.build_inputs(model, processor, data, "cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model.vlm.model(input_ids=inp["input_ids"],
                            attention_mask=torch.ones_like(inp["input_ids"]),
                            pixel_values=inp["tokenized_data"]["pixel_values"],
                            image_grid_thw=inp["tokenized_data"]["image_grid_thw"],
                            use_cache=False)
    out = {}
    for (li, kind), h in hooks.items():
        h.remove()
        mod = layers[li].self_attn.o_proj if "o_proj" in kind else layers[li].mlp.down_proj
        W = mod.weight.data.float()
        H = h.H
        name = f"layers.{li:02d}.{kind}"
        w_rep = torch.load(REPO / "outputs" / args.supernet / name / "0.pth",
                           map_location="cuda").float()
        kept = ([q * 128 + d for q in meta["vlm"][li]["q"] for d in range(128)]
                if "o_proj" in kind else meta["vlm"][li]["mlp"])
        w_ref = torch.zeros_like(W)
        w_ref[:, kept] = sd[f"vlm.model.language_model.layers.{li}.{kind}.weight"].float().cuda()
        denom = torch.trace(W @ H @ W.t())
        diff = w_rep - w_ref
        between = float(torch.sqrt(torch.trace(diff @ H @ diff.t()) / denom))
        out[name] = {"between": between,
                     "rep_err": float(tyr.recon_error(W, w_rep, H)),
                     "dualr_err": float(tyr.recon_error(W, w_ref, H)),
                     "weight_rel_diff": float(diff.norm() / w_ref.norm())}
        print(f"{name:28s} between {between:.4f} | rep err {out[name]['rep_err']:.4f} "
              f"dualr err {out[name]['dualr_err']:.4f} | weight rel diff "
              f"{out[name]['weight_rel_diff']:.3f}", flush=True)
    (REPO / "outputs" / args.supernet / "g0_function_space.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
