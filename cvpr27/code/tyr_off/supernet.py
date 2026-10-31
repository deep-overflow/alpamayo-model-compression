"""Tyr-the-Pruner algorithm 1 (prune-to-supernet), following the official repo step for step.

cvpr27/plans/2026-10-31_tyr-a100.md. Official code: /home/cvlab21/project/chan/Tyr-the-Pruner
(prune_to_supernet.py, src/pruner.py, src/local_pruner.py, script/example.sh).

What follows the official code exactly:
  - units: input columns of o_proj (one Q head = head_dim columns) and down_proj (one channel);
    every other projection, k/v included, is untouched
  - per-level solve: LocalPruner.prune + step (dead-input zeroing, damping 1e-2 x mean diag,
    G = H @ B, sparsity 0.0 -> dense, 1.0 -> zeros, otherwise local_prune_core with
    int(cin * (1 - sp)) kept, 1 round for heads / 16 for channels) -- tyr_lib.local_prune_core
    is bitwise equal to the official one (check_osscar_vs_official.py, G0)
  - level grid (pruner.py _prune_group): delta = weights_diff / numel,
    levels -min(int(s // delta), 4) .. +min(int((1 - s) // delta), 4), sparsity s + l * delta
  - weights_diff floors (prune_to_supernet.py): mlp >= 32 * hidden, mha >= head_size * hidden
  - R >= 2: per-layer centre = (s_prev * numel + level * weights_diff_prev) / numel from the
    previous supernet's metadata and the previous search's levels
Where it differs (decided by the user, 2026-10-31):
  - calibration data is calib_100 (fused driving prompts, all prefill tokens), not FineWeb-Edu
  - error accumulation is the paper's Eq. 5: the next block sees the (1 - S_e)-weighted mean of
    every level's weight (the official code's commented-out branch; its live "median" branch is
    a no-op because of `expected_sparsity = 0`). Accumulated in fp32.
  - each block's inputs come from a full prefill through the already-processed blocks (same
    activations as the official block chaining, and Qwen3-VL's deepstack features stay intact)

Outputs (outputs/<exp-id>/): layers.NN.{self_attn.o_proj,mlp.down_proj}/<level>.pth (bf16, full
(out, in) weights, official layout), metadata.json, summary.txt.

  .venv/bin/python cvpr27/code/tyr_off/supernet.py --exp-id cvpr27/tyr_off/supernet_r1 --iteration 1
  .venv/bin/python cvpr27/code/tyr_off/supernet.py --exp-id cvpr27/tyr_off/supernet_r2 --iteration 2 \
      --weights-diff-mha 1048576 --weights-diff-mlp 3145728 \
      --prev-supernet cvpr27/tyr_off/supernet_r1 --prev-config cvpr27/tyr_off/search_r1/final_config.json
  # smoke: --num-clips 2 --blocks 2
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
sys.path.insert(0, str(REPO / "experiments" / "evaluation"))

import analysis_lib as lib
import sample_cache as sc
import tyr_lib as tyr
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from expert_per_clip import reserve_gpu
from run_tyr_supernet import prefill, preload_inputs
from slim_lib import MODEL_REV

# official local_pruner.py sets both off at import
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def level_grid(s, delta, half):
    """pruner.py _prune_group: {level: sparsity} around centre s."""
    lo = min(int(s // delta), half)
    hi = min(int((1 - s) // delta), half)
    return {lv: s + lv * delta for lv in range(-lo, hi + 1)}


@torch.no_grad()
def official_levels(module, H, n_groups, sparsities, update_iter):
    """LocalPruner.prune + step for one module: {level: (out, in) fp32 weight}."""
    W = module.weight.data.clone().float()
    B = W.t()                                            # (in, out)
    dead = torch.diag(H) == 0
    B[dead, :] = 0
    H = H + torch.eye(B.shape[0], device=H.device) * torch.mean(torch.diag(H)) * 1e-2
    G = H @ B
    out = {}
    for lv, sp in sparsities.items():
        if sp == 0.0:
            B_sol = B.clone()
        elif sp == 1.0:
            B_sol = torch.zeros_like(B)
        else:
            B_sol, _ = tyr.local_prune_core(B.clone(), H, G, n_groups, int(n_groups * (1 - sp)),
                                            update_iter)
        out[lv] = B_sol.t().contiguous()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--iteration", type=int, required=True)
    ap.add_argument("--sparsity", type=float, default=0.39856,
                    help="uniform start (R=1); 0.39856 cuts 13/32 heads and 4898/12288 channels "
                         "per layer = MoOP's 2,657,452,032 removed parameters")
    ap.add_argument("--weights-diff-mha", type=int, default=2097152,
                    help="12.5%% of o_proj (example.sh's value is 12.5%% of Llama-2-7B's)")
    ap.add_argument("--weights-diff-mlp", type=int, default=6291456,
                    help="12.5%% of down_proj (4096 x 12288)")
    ap.add_argument("--num-levels", type=int, default=9)
    ap.add_argument("--prev-supernet", default=None)
    ap.add_argument("--prev-config", default=None)
    ap.add_argument("--num-clips", type=int, default=100)
    ap.add_argument("--calib-manifest", default="calib_100")
    ap.add_argument("--cache", default="calib")
    ap.add_argument("--blocks", type=int, default=None, help="first N blocks only (smoke)")
    ap.add_argument("--reserve-gb", type=float, default=40.0)
    ap.add_argument("--gpu", type=str, default=None)
    args = ap.parse_args()
    assert (args.prev_supernet is None) == (args.prev_config is None)
    assert (args.iteration == 1) == (args.prev_supernet is None)

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

    tc = model.vlm.config.text_config
    layers = model.vlm.model.language_model.layers
    n_layers = len(layers)
    n_blocks = min(args.blocks or n_layers, n_layers)
    half = args.num_levels // 2
    numel_o = layers[0].self_attn.o_proj.weight.numel()
    numel_d = layers[0].mlp.down_proj.weight.numel()
    wd_mha = max(args.weights_diff_mha, tc.head_dim * tc.hidden_size)
    wd_mlp = max(args.weights_diff_mlp, 32 * tc.hidden_size)

    if args.prev_supernet is None:
        mha_sp = [args.sparsity] * n_layers
        ffn_sp = [args.sparsity] * n_layers
    else:
        pmeta = json.loads((REPO / "outputs" / args.prev_supernet / "metadata.json").read_text())
        pcfg = json.loads((REPO / "outputs" / args.prev_config).read_text())
        if args.blocks is None:
            assert len(pcfg) == 2 * n_layers, len(pcfg)
        # a smoke supernet (--blocks) only has its first blocks in the config: the rest stay put
        mha_sp = [(pmeta["mha_sparsity"][i] * numel_o
                   + pcfg.get(f"layers.{i:02d}.self_attn.o_proj", 0) * pmeta["weights_diff_mha"]) / numel_o
                  for i in range(n_layers)]
        ffn_sp = [(pmeta["ffn_sparsity"][i] * numel_d
                   + pcfg.get(f"layers.{i:02d}.mlp.down_proj", 0) * pmeta["weights_diff_mlp"]) / numel_d
                  for i in range(n_layers)]

    grids, layer_names = {}, []
    for i in range(n_blocks):
        for suffix, s, wd, numel, cin in (
                ("mlp.down_proj", ffn_sp[i], wd_mlp, numel_d, tc.intermediate_size),
                ("self_attn.o_proj", mha_sp[i], wd_mha, numel_o, tc.num_attention_heads)):
            name = f"layers.{i:02d}.{suffix}"
            layer_names.append(name)
            g = level_grid(s, wd / numel, half)
            grids[name] = {lv: {"sparsity": sp,
                                "keep": cin if sp == 0.0 else 0 if sp == 1.0 else int(cin * (1 - sp))}
                           for lv, sp in g.items()}
    keep0_q = sum(grids[f"layers.{i:02d}.self_attn.o_proj"][0]["keep"] for i in range(n_blocks))
    keep0_m = sum(grids[f"layers.{i:02d}.mlp.down_proj"][0]["keep"] for i in range(n_blocks))
    removed0 = ((n_blocks * tc.num_attention_heads - keep0_q) * 2 * tc.head_dim * tc.hidden_size
                + (n_blocks * tc.intermediate_size - keep0_m) * 3 * tc.hidden_size)
    print(f"R={args.iteration}: centre keeps {keep0_q} heads / {keep0_m} channels -> "
          f"removed {removed0:,}", flush=True)

    meta = {
        "model_revision": MODEL_REV, "iteration": args.iteration, "sparsity": args.sparsity,
        "mha_sparsity": mha_sp, "ffn_sparsity": ffn_sp,
        "weights_diff_mha": wd_mha, "weights_diff_mlp": wd_mlp, "num_levels": args.num_levels,
        "prev_supernet": args.prev_supernet, "prev_config": args.prev_config,
        "num_clips": len(calib), "clip_ids": [c for c, _ in calib],
        "layer_names": layer_names,
        "levels": {n: {str(lv): v for lv, v in g.items()} for n, g in grids.items()},
        "centre_keep_heads": keep0_q, "centre_keep_channels": keep0_m, "centre_removed": removed0,
        "hessian_tokens": "full fused prompt prefill, no labels (calib_100)",
        "error_accumulation": "paper Eq. 5: (1 - S_e)-weighted mean of all level weights, fp32",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=1))

    print(f"preloading {len(calib)} clip inputs...", flush=True)
    store = preload_inputs(model, processor, calib, args.cache)

    t_all = time.time()
    for i in range(n_blocks):
        t0 = time.time()
        o_proj, down = layers[i].self_attn.o_proj, layers[i].mlp.down_proj
        hooks = {"o": tyr.HessianHook(o_proj), "m": tyr.HessianHook(down)}
        for item in store:
            prefill(model, item)
        for h in hooks.values():
            h.remove()
        t_fwd = time.time() - t0
        for mod, hook, suffix, n_groups, upd in (
                (o_proj, hooks["o"], "self_attn.o_proj", tc.num_attention_heads, 1),
                (down, hooks["m"], "mlp.down_proj", tc.intermediate_size, 16)):
            name = f"layers.{i:02d}.{suffix}"
            sps = {lv: v["sparsity"] for lv, v in grids[name].items()}
            sols = official_levels(mod, hook.H, n_groups, sps, upd)
            d = out_dir / name
            d.mkdir(parents=True, exist_ok=True)
            gs = mod.in_features // n_groups
            acc = torch.zeros_like(sols[0])
            wsum = 0.0
            for lv, w in sols.items():
                n_alive = int((w.abs().sum(0).reshape(n_groups, gs).sum(1) != 0).sum())
                assert n_alive == grids[name][lv]["keep"], (name, lv, n_alive, grids[name][lv])
                torch.save(w.to(torch.bfloat16).cpu(), d / f"{lv}.pth")
                acc += w * (1.0 - sps[lv])
                wsum += 1.0 - sps[lv]
            # Eq. 5: the block's output for the next block is the expectation over its levels
            mod.weight.data.copy_((acc / wsum).to(mod.weight.dtype))
        del hooks
        torch.cuda.empty_cache()
        print(f"[block {i + 1}/{n_blocks}] fwd {t_fwd:.0f}s total {time.time() - t0:.0f}s", flush=True)

    (out_dir / "summary.txt").write_text(
        f"tyr_off supernet R={args.iteration}: {n_blocks} blocks, {len(calib)} clips, "
        f"{time.time() - t_all:.0f}s\n"
        f"weights_diff mha {wd_mha} ({wd_mha // (tc.head_dim * tc.hidden_size)} heads), "
        f"mlp {wd_mlp} ({wd_mlp // tc.hidden_size} channels)\n"
        f"centre: {keep0_q} heads, {keep0_m} channels kept -> removed {removed0:,}\n")
    print("saved ->", out_dir, flush=True)


if __name__ == "__main__":
    main()
