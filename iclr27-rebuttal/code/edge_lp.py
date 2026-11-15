"""LLM-Pruner (paper arm `lp_r50` / u50 `lp_r60`) on Alpamayo 2 Edge (plans/2026-11-15_a2-edge-baselines-u50.md). Edge venv.

The paper recipe (`experiments/llm_pruner/run_lp_importance.py` objective `coc` + `lp_prune.py --arm coc_param_first`) ported:
  objective    per calib_100 clip: the dense model's own CoC rollout (release sampling, seed sha256(42:clip)), one teacher-forced
               VLM forward over prompt + CoC (no cache), mean CE over the CoC positions (edge_grad_lib.coc_nll = lp_objectives.coc_nll);
               weight gradients summed over the 100 clips (lp_core.GradAccumulator, fp32 on CPU)
  importance   lp_core.group_importance(taylor="param_first", group_reduction="sum"), element-wise first-order Taylor s = W * g:
               head  I(h) = sum_{d in h} [ sum_j |W_q[d,j] g| + sum_i |W_o[i,d] g| ]
               MLP   I(c) = sum_j |W_fc1[c,j] g| + sum_i |W_fc2[i,c] g|      (Edge relu^2 MLP; A1.5 had gate/up/down)
               k_proj / v_proj (the expert's KV interface) and norms are frozen, as in the paper port
  selection    upstream local per-layer pruning, lp_core.keep_masks_for_ratio in the layer scope; paper rule = skip the first 4 and
               the last 2 layers (A1.5 [4, 34), Edge [4, 26) = 22 layers) with the ratio that matches MoOP u50's budget:
               0.5 x 28 / 22 (A1.5: 0.5 x 36 / 30 = 0.6). Heads floor to whole heads (10/16), MLP 5,865/9,216 per scoped layer.
  recovery     none (the paper omits LoRA recovery)
lp_core is imported unchanged; only its member tables are set to Edge's module names.

Usage: CUDA_VISIBLE_DEVICES=5 /mnt/nvme1n1/ad_vla/venvs/alpamayo2-edge/bin/python iclr27-rebuttal/code/edge_lp.py
"""

import os
import sys

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    raise SystemExit("set CUDA_VISIBLE_DEVICES to one empty card")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import edge_common as ec

ec.setup_hf_env(offline=True)

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "llm_pruner"))
import edge_eval_lib as eel
import edge_grad_lib as gl
import lp_core as lp

# Edge's coupled structures: ATTN(i) unchanged ({q_proj.out, o_proj.in}); MLP(i) = {fc1.out, fc2.in} (no gate)
lp.MLP_MEMBERS = (("mlp.fc1", "out"), ("mlp.fc2", "in"))
lp.PRUNABLE_TENSORS = tuple(name for name, _ in lp.ATTN_MEMBERS + lp.MLP_MEMBERS)


def tf_forward(model, seq, td):
    """Teacher-forced VLM forward over prompt + CoC without cache (lp_objectives.vlm_forward, use_cache=False)."""
    prompt_len = td["input_ids"].shape[1]
    mm = torch.cat([td["mm_token_type_ids"], td["mm_token_type_ids"].new_zeros((1, seq.shape[1] - prompt_len))], 1)
    out = model.vlm.model(input_ids=seq, attention_mask=torch.ones_like(seq), pixel_values=td["pixel_values"],
                          image_grid_thw=td["image_grid_thw"], mm_token_type_ids=mm, use_cache=False)
    return out.last_hidden_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="iclr27-rebuttal/edge_lp_importance_v1")
    ap.add_argument("--scope", type=int, nargs=2, default=[4, 26], help="half-open layer range (paper rule on 28 layers)")
    ap.add_argument("--ratio", type=float, default=0.5 * 28 / 22, help="upstream --pruning_ratio inside the scope")
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default="lp_u50")
    args = ap.parse_args()
    out = REPO / "outputs" / args.exp_id
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    from alpamayo2_edge import helper

    model, _ = ec.load_model("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer, model.config)
    geom = lp.text_geometry(model)
    scope = lp.layer_scope(geom["n_layers"], *args.scope)
    accum = lp.GradAccumulator(model, scope, objectives=("coc",))
    print(f"accumulator {accum.nbytes() / 2**30:.1f} GB on CPU, scope {scope[0]}..{scope[-1]} ({len(scope)} layers)", flush=True)
    man = pd.read_parquet(ec.CALIB_MANIFEST)
    recs, t0 = [], time.time()
    for i, r in enumerate(man.itertuples()):
        data = ec.load_cached(ec.path_for(r.clip_id, int(r.t0_us)))
        inputs, _ = ec.prepare_inputs_no_nav(data, model.config, model.tokenizer, processor)
        inputs = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in inputs.items()}
        inputs["tokenized_data"] = {k: v.to("cuda") if torch.is_tensor(v) else v
                                    for k, v in inputs["tokenized_data"].items()}
        seed = ec.clip_seed(args.seed, r.clip_id)
        eel.seed_all(seed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            seq, prompt_len, has_eos, cot, _ = gl.rollout_sequence(model, inputs, args.max_gen)
        seq = seq.clone()
        td = gl.fused_tokenized(model, inputs)
        accum.begin("coc")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = tf_forward(model, seq, td)
        nll = gl.coc_nll(model, hidden, seq, prompt_len, seq.shape[1])
        nll.backward()
        accum.end()
        recs.append({"clip_id": r.clip_id, "coc_len": int(seq.shape[1] - prompt_len), "nll": float(nll.detach()),
                     "has_eos": has_eos, "cot": cot})
        del hidden, nll
        print(f"[{i + 1}/{len(man)}] {r.clip_id[:8]} coc={recs[-1]['coc_len']} nll={recs[-1]['nll']:.3f} "
              f"({time.time() - t0:.0f}s) | {cot[:60]!r}", flush=True)
    accum.remove()
    q_scores, mlp_scores = lp.group_importance(model, accum, "coc", taylor="param_first", group_reduction="sum")
    np.savez(out / "importance.npz", coc_param_first_q=q_scores, coc_param_first_mlp=mlp_scores)
    q_keep, mlp_keep, plan = lp.keep_masks_for_ratio(q_scores, mlp_scores, args.ratio, scope, geom)
    head_cost, ch_cost = 2 * geom["hidden"] * geom["head_dim"], 2 * geom["hidden"]  # Edge: relu^2 MLP = 2 x hidden
    removed = int((q_keep == 0).sum() * head_cost + (mlp_keep == 0).sum() * ch_cost)
    mask_path = REPO / "outputs" / "iclr27-rebuttal" / "edge_moop" / f"masks_{args.name}.npz"
    np.savez(mask_path, q_mask=q_keep, mlp_mask=mlp_keep)
    moop_u50 = 645_922_816
    cfg = {"method": "LLM-Pruner coc_param_first (paper lp_r50 / lp_r60 recipe) ported to Edge", "model": ec.MODEL_ID,
           "model_revision": ec.MODEL_REV, "calib": str(ec.CALIB_MANIFEST), "cache": str(ec.CALIB_CACHE),
           "prompt": "Edge driving prompt without the navigation line", "objective": "own-rollout CoC NLL (mean over CoC)",
           "taylor": "param_first", "group_reduction": "sum", "members": {"attn": lp.ATTN_MEMBERS, "mlp": lp.MLP_MEMBERS},
           "scope": [scope[0], scope[-1] + 1], "ratio": args.ratio, "plan": plan, "removed_params": removed,
           "removed_vs_moop_u50": removed - moop_u50, "n_samples": accum.n_samples["coc"], "masks": str(mask_path),
           "coc_len_mean": float(np.mean([x["coc_len"] for x in recs])), "nll_mean": float(np.mean([x["nll"] for x in recs])),
           "gpu": torch.cuda.get_device_name(0), "recovery": "none"}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "metrics.json").write_text(json.dumps({"per_clip": recs}, indent=2))
    (out / "summary.txt").write_text(f"Edge LLM-Pruner {args.name}: scope {scope[0]}-{scope[-1]}, ratio {args.ratio:.4f} -> "
                                     f"{plan['heads_dropped_per_layer']} heads + {plan['mlp_dropped_per_layer']} channels per "
                                     f"scoped layer, removed {removed:,} ({removed - moop_u50:+,} vs MoOP u50)\n")
    print(json.dumps(cfg, indent=2, default=str))


if __name__ == "__main__":
    main()
