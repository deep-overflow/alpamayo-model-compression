"""Build a physically-slimmed Alpamayo-1.5 checkpoint from a validated mask config.

Configs:
  integrated_mag   -- VLM graded 30/50 + layer35(kv-only) + KV1 (trajectory Taylor),
                      expert early40/late10 (magnitude). Expected -3.25B.
  cocsafe_full_r20 -- VLM dual max(rank_traj, rank_coc) 20% + KV1(dual),
                      expert early40/late10 (magnitude). Expected -2.01B.

The mask recipes are imported from run_integrated / run_cocsafe -- no duplicated math.
Writes slim_state.pt + slim_meta.json to --out, then smoke-tests one val clip
(short rollout, teacher-forced NLL, one denoise, cache-shape assert).

Usage:
  bash experiments/head_analysis/run_retry.sh 20 experiments/head_analysis/make_slim.py \
      --config integrated_mag --gpu 4 --out outputs/slim_integrated_mag
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

import analysis_lib as lib  # noqa: E402
import mask_lib as ml  # noqa: E402
import slim_lib as sl  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  also installs the gated-repo hub patch
from run_cocsafe import rank_norm  # noqa: E402
from run_eval import eval_config  # noqa: E402
from run_integrated import expert_masks, vlm_combined_masks  # noqa: E402

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

REPO = Path("/workspace/alpamayo-model-compression")


def build_masks(cfg_name, imp, model):
    tc = model.vlm.config.text_config
    ec = model.expert.config
    emag = ml.magnitude_scores(model.expert.layers, ec.num_attention_heads, ec.head_dim,
                               ec.intermediate_size)
    eq, em = expert_masks(imp, emag, ec.num_hidden_layers, "magnitude")
    if cfg_name == "integrated_mag":
        vq, vm = vlm_combined_masks(imp, tc.num_hidden_layers, tc.num_attention_heads,
                                    tc.intermediate_size)
        kvonly = (tc.num_hidden_layers - 1,)
    else:  # cocsafe_full_r20 / cocsafe_full_r30 -- dual width r% + KV1(dual) + expert magnitude
        ratio = 0.30 if cfg_name == "cocsafe_full_r30" else 0.20
        all_l = list(range(tc.num_hidden_layers))
        dual_q = np.maximum(rank_norm(imp["traj_vlm_q"]), rank_norm(imp["coc_vlm_q"]))
        dual_m = np.maximum(rank_norm(imp["traj_vlm_mlp"]), rank_norm(imp["coc_vlm_mlp"]))
        dual_kv = np.maximum(rank_norm(imp["traj_kv_k"] + imp["traj_kv_v"]),
                             rank_norm(imp["coc_kv_k"] + imp["coc_kv_v"]))
        vq = ml.select_mask(dual_q, ratio, all_l) * ml.kv_group_mask(
            dual_kv, 1, tc.num_attention_heads, all_l)
        vm = ml.select_mask(dual_m, ratio, all_l)
        kvonly = ()
    return vq, vm, eq, em, kvonly


def expected_removed(model, vq, vm, eq, em, kvonly):
    tc = model.vlm.config.text_config
    ec = model.expert.config
    p_vq = 2 * tc.hidden_size * tc.head_dim
    p_vm = 3 * tc.hidden_size
    p_eq = 2 * ec.hidden_size * ec.head_dim
    p_em = 3 * ec.hidden_size
    removed = (int((vq == 0).sum()) * p_vq + int((vm == 0).sum()) * p_vm
               + int((eq == 0).sum()) * p_eq + int((em == 0).sum()) * p_em)
    # kv-only layers also lose q_norm (head_dim) and post_attention_layernorm (hidden)
    removed += len(kvonly) * (tc.head_dim + tc.hidden_size)
    return removed


def smoke(model, processor, clip_id, seed=42):
    """Short rollout + TF eval + cache-shape assert on the slim model."""
    data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
    inputs = lib.build_inputs(model, processor, data, "cuda")
    prompt_len = inputs["input_ids"].shape[1]
    gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        roll = lib.run_rollout(model, inputs, max_generation_length=64)
    coc_start, coc_end = prompt_len, roll["eos_pos"] + 1
    seq_tf = roll["sequences"][:, :coc_end].clone()
    for i in range(len(model.vlm.model.language_model.layers)):
        k, _ = lib.cache_layer_kv(roll["past_key_values"], i)
        assert k.shape[1] == 8, f"layer {i}: cache heads {k.shape[1]}"
    del roll
    ade, fde, nll = eval_config(model, inputs, seq_tf, coc_start, coc_end, gt_xy, [seed])
    return {"clip_id": clip_id, "coc_len": coc_end - coc_start,
            "minADE_k1": ade, "minFDE_k1": fde, "nll": nll}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    choices=["integrated_mag", "cocsafe_full_r20", "cocsafe_full_r30"])
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--importance", type=str, default="importance_v1")
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    out_dir = REPO / args.out
    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    print(f"using {device}", flush=True)

    model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")

    imp = dict(np.load(REPO / "outputs" / args.importance / "importance.npz"))
    vq, vm, eq, em, kvonly = build_masks(args.config, imp, model)

    full_total = sl.n_params(model)
    t0 = time.time()
    meta = sl.apply_surgery(model, vq, vm, eq, em, kvonly_layers=kvonly)
    meta["config"] = args.config
    meta["importance_from"] = args.importance
    slim_total = sl.n_params(model)
    removed = expected_removed(model, vq, vm, eq, em, kvonly)
    print(f"surgery {time.time() - t0:.0f}s: {full_total:,} -> {slim_total:,} "
          f"(-{full_total - slim_total:,}, {(full_total - slim_total) / full_total * 100:.1f}%)",
          flush=True)
    assert slim_total == full_total - removed, (slim_total, full_total - removed)
    sl.check_slim(model)
    meta["params"] = {"full": full_total, "slim": slim_total, "removed": removed}

    t0 = time.time()
    sl.save_slim(model, meta, out_dir)
    print(f"saved checkpoint in {time.time() - t0:.0f}s -> {out_dir}", flush=True)

    split = json.loads((REPO / "outputs" / "split.json").read_text())
    result = smoke(model, processor, split["val"][0])
    print(f"smoke: {result}", flush=True)

    (out_dir / "config.json").write_text(json.dumps({
        "model": "nvidia/Alpamayo-1.5-10B", "config": args.config,
        "importance_from": args.importance, "params": meta["params"],
        "kvonly_layers": list(kvonly),
        "kept_q_per_layer": {"vlm": [len(m["q"]) for m in meta["vlm"]],
                             "expert": [len(m["q"]) for m in meta["expert"]]},
        "smoke": result, "gpu": torch.cuda.get_device_name(device),
    }, indent=2))
    (out_dir / "summary.txt").write_text(
        f"slim checkpoint {args.config}\n"
        f"params {full_total:,} -> {slim_total:,} "
        f"(-{removed:,}, {removed / full_total * 100:.1f}%)\n"
        f"smoke clip {result['clip_id']}: NLL {result['nll']:.4f} "
        f"minADE@1 {result['minADE_k1']:.3f}\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
