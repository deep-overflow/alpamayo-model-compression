# Ported from soowon's alpamayo1.5 research tree
#   /home/cvlab21/project/soowon/vla-ad/alpamayo1.5/experiments/llm_pruner/lp_prune.py
# copied 2026-09-06 to run the second-order (param_mix) arm that was never executed there.
# That tree had itself vendored analysis_lib / sample_cache / expert_per_clip / slim_lib
# FROM this repo on 2026-08-09; those copies are deliberately NOT brought back -- this
# repo's versions are newer (expert_per_clip carries an AcceleratorError fix, slim_lib a
# pinned MODEL_REV and write_state, sample_cache calib_samples()), and this code was
# written against them in the first place.
# Changes from the original are marked `PORT:`; the importance formulas are otherwise
# untouched, because upstream fidelity is the whole point of this file.
"""LLM-Pruner stage 1+2 output -> a physically pruned Alpamayo 1.5 checkpoint.

Takes the group scores from `run_lp_importance.py`, applies upstream's local (per-layer)
selection rule at a given `--pruning_ratio`, and does real weight surgery via
`slim_lib.apply_surgery`: q_proj rows and o_proj columns for every dropped query head,
gate/up rows and down columns for every dropped MLP channel.  k_proj / v_proj / q_norm /
k_norm / the 8x128 cache and the expert tower are never touched.

Upstream serialises `torch.save({'model': module, 'tokenizer': tok})` -- a full pickle of
a module whose per-layer widths differ.  That cannot work here (22 GB, pickles the
hydra-instantiated action space, breaks on any transformers bump), and Alpamayo's
config.json carries no text geometry at all, so `from_pretrained` cannot rebuild the
shapes either.  The in-lab format is used instead: `slim_state.pt` + `slim_meta.json` of
kept indices, replayed onto a fresh skeleton by `slim_lib.load_slim`.

Usage:
  bash experiments/llm_pruner/run_host.sh 20 experiments/llm_pruner/lp_prune.py \
      --importance lp_importance_v1 --arm coc_param_first --ratio 0.20 \
      --out outputs/lp_r20 --gpu 4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
sys.path.insert(0, str(REPO / "experiments" / "evaluation"))

import analysis_lib as lib  # noqa: E402
import lp_core as lp  # noqa: E402
import sample_cache as sc  # noqa: E402
import slim_lib as sl  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  also installs the gated-repo hub patch

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

MODEL_REV = "7aba8293c09993f2e125c6819df05d7fa3e873ea"
SMOKE_CLIP = None  # first calib clip


def smoke(model, processor, clip_id):
    """One rollout + teacher-forced NLL. Asserts the expert's KV interface survived."""
    data = sc.load_cached(sc.path_for("calib", clip_id, sc.CALIB_T0))
    inputs = lib.build_inputs(model, processor, data, "cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        roll = lib.run_rollout(model, inputs, max_generation_length=64)
    cache = roll["past_key_values"]
    for i in range(len(lp.text_layers(model))):
        k, v = lib.cache_layer_kv(cache, i)
        assert k.shape[1] == 8 and k.shape[3] == 128, (i, tuple(k.shape))
        assert v.shape[1] == 8 and v.shape[3] == 128, (i, tuple(v.shape))
    text = model.tokenizer.decode(
        roll["sequences"][0, inputs["input_ids"].shape[1]: roll["eos_pos"]],
        skip_special_tokens=True,
    )
    return {"coc_len": roll["eos_pos"] + 1 - inputs["input_ids"].shape[1], "coc": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--importance", required=True, help="exp-id under outputs/ holding lp_importance.npz")
    ap.add_argument("--arm", default="coc_param_first", help="score arm, e.g. coc_param_first")
    ap.add_argument("--ratio", type=float, required=True, help="LLM-Pruner --pruning_ratio")
    ap.add_argument("--out", required=True, help="output dir, relative to REPO")
    ap.add_argument("--layer-start", type=int, default=4)
    ap.add_argument("--layer-end", type=int, default=34)
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--no-save", action="store_true", help="build and smoke only")
    args = ap.parse_args()

    imp_dir = REPO / "outputs" / args.importance
    z = np.load(imp_dir / "lp_importance.npz")
    if f"{args.arm}_q" not in z.files:
        arms = sorted({n.rsplit("_", 1)[0] for n in z.files if n.endswith(("_q", "_mlp"))})
        raise SystemExit(
            f"arm {args.arm!r} is not in {imp_dir / 'lp_importance.npz'} (have: {arms}). "
            f"param_mix / param_second are only written when run_lp_importance.py was "
            f"given --second-order."
        )
    q_scores, mlp_scores = z[f"{args.arm}_q"], z[f"{args.arm}_mlp"]
    out_dir = REPO / args.out if not Path(args.out).is_absolute() else Path(args.out)

    # The npz stores identically-zero rows for layers the calibration pass never scored.
    # np.argsort of an all-zero row is stable, i.e. it returns 0,1,2,... -- so a --layer
    # scope wider than the one importance was computed over prunes those layers *by index*
    # and nothing detects it: the parameter-count assert below still passes and the smoke
    # rollout still produces text.  Fail here, before the 22 GB model load.
    if "scope" not in z.files:
        raise SystemExit(f"{imp_dir / 'lp_importance.npz'} has no `scope` key; rebuild it")
    imp_scope = {int(x) for x in z["scope"]}
    want = lp.layer_scope(q_scores.shape[0], args.layer_start, args.layer_end)
    missing = sorted(set(want) - imp_scope)
    if missing:
        raise SystemExit(
            f"layers {missing} carry no importance in {args.importance} "
            f"(npz scope {sorted(imp_scope)}); re-run run_lp_importance.py over them, "
            f"or narrow --layer-start/--layer-end"
        )

    devices = None if args.gpu is None else [int(x) for x in args.gpu.split(",")]
    reserve_gpu(args.reserve_gb, devices=devices)

    t0 = time.time()
    model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", revision=MODEL_REV,
                                        dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")
    full_total = sl.n_params(model)
    print(f"loaded {full_total:,} params in {time.time() - t0:.0f}s", flush=True)

    geom = lp.text_geometry(model)
    scope = lp.layer_scope(geom["n_layers"], args.layer_start, args.layer_end)
    vq, vm, plan = lp.keep_masks_for_ratio(q_scores, mlp_scores, args.ratio, scope, geom)
    expected = lp.removed_params(plan, geom)
    print(f"ratio={args.ratio}: {plan['heads_dropped_per_layer']}/{geom['n_heads']} heads and "
          f"{plan['mlp_dropped_per_layer']}/{geom['intermediate']} MLP ch per layer "
          f"x {plan['layers']} layers -> {expected:,} params "
          f"({100 * expected / full_total:.2f}% of the model)", flush=True)

    # expert tower is out of scope for this study -> all ones
    ec = model.expert.config
    eq = np.ones((ec.num_hidden_layers, ec.num_attention_heads), dtype=np.float32)
    em = np.ones((ec.num_hidden_layers, ec.intermediate_size), dtype=np.float32)

    meta = sl.apply_surgery(model, vq, vm, eq, em, kvonly_layers=())
    sl.check_slim(model)
    slim_total = sl.n_params(model)
    assert slim_total == full_total - expected, (slim_total, full_total, expected)

    calib = sc.calib_clips(REPO, "calib_100")
    sm = smoke(model, processor, calib[0])
    print(f"smoke: coc_len={sm['coc_len']} coc={sm['coc'][:160]!r}", flush=True)

    text_layer_params = lp.text_layer_params(geom)
    meta["config"] = {
        "method": "LLM-Pruner block-wise, grouped Taylor",
        "importance_from": args.importance, "arm": args.arm, "ratio": args.ratio,
        "layer_scope": [args.layer_start, args.layer_end], "plan": plan,
        "importance_scope": sorted(imp_scope),   # so the scope guard is auditable after the fact
        "model_revision": MODEL_REV,
    }
    meta["params"] = {
        "full": full_total, "slim": slim_total, "removed": expected,
        "pct_of_model": 100 * expected / full_total,
        "pct_of_text_layers": 100 * expected / text_layer_params,
    }
    meta["smoke"] = sm

    if args.no_save:
        print("--no-save: skipping checkpoint write", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    t1 = time.time()
    sl.save_slim(model, meta, out_dir)
    (out_dir / "config.json").write_text(json.dumps(meta["config"] | meta["params"], indent=2))
    # interpolate from meta["params"] rather than recomputing: the two drifted apart once
    # already (lp_r10/r20/r24/r50 shipped a summary.txt whose text-layer denominator
    # double-counted q_proj+o_proj, reading ~6 points low against their own config.json).
    (out_dir / "summary.txt").write_text(lp.summary_text(args, meta))
    print(f"saved -> {out_dir} (write {time.time() - t1:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
