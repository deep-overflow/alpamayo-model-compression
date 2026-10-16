"""Greedy layer-0 Q-head selection by the propagated FM loss (P2) on an arm's own calibration clips.

plans/2026-10-14_l0-functional-redundancy.md, step 3 as amended 2026-10-16. Step 2 found that
the mean flow-matching loss on calibration clips, with an arm's mask in layers 1-35 and a
candidate set in layer 0, ranks stored arms by their damage (rho +0.78 / +0.63) where nothing
local to layer 0 does. This turns it into a selector:

  band     the arm's own layer-0 dual score (its importance file, as make_slim built it):
           top 13 heads kept, bottom 6 dropped, the middle 13 the band, 7 of which are dropped
  search   forward greedy from the 6 fixed drops: add the band head whose drop gives the lowest
           P2, seven times, every candidate's score recorded
  P2       mean FM loss over the first --num-clips clips of the arm's OWN calibration manifest
           and cache (read from its importance run's config), the dense model's reference text
           built once per clip, run_token_ablation.eval_config with no trajectory sampling

The arm's layers 1-35 are held fixed, so the budget is the arm's own (13 heads in layer 0).
Writes outputs/<exp-id>/{config.json, metrics.json, summary.txt}; the selected set is
metrics["selected_drop"].

Usage:
  ALPAMAYO_REPO=$PWD bash experiments/head_analysis/run_retry_host.sh 3 \
      experiments/evaluation/run_l0_select.py --arm dual_seed_n1 --exp-id l0sel_n1 --gpu 5
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "head_analysis"))
sys.path.insert(0, str(Path(__file__).parent))

import analysis_lib as lib
import mask_lib as ml
import sample_cache as sc
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from expert_per_clip import reserve_gpu
from run_cocsafe import rank_norm
from run_token_ablation import MODEL_REV, eval_config

REPO = Path(__file__).resolve().parents[2]
N_HEADS, N_FIXED_DROP, N_BAND, N_DROP = 32, 6, 13, 13


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="slim dir name without the slim_ prefix")
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--num-clips", type=int, default=50)
    ap.add_argument("--steps", type=int, default=N_DROP - N_FIXED_DROP, help="greedy steps (7)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fm-steps", type=int, default=10)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--gpu", type=str, default=None)
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((REPO / "outputs" / f"slim_{args.arm}" / "slim_meta.json").read_text())
    imp_run = meta["importance_from"]
    icfg = json.loads((REPO / "outputs" / imp_run / "config.json").read_text())
    manifest = icfg.get("calib_manifest") or icfg.get("manifest")
    cache = icfg.get("cache") or "calib"
    clips = sc.calib_samples(REPO, manifest)
    if icfg.get("clip_ids"):           # a block of a larger manifest (the nt blocks)
        want = set(icfg["clip_ids"])
        clips = [c for c in clips if c[0] in want]
    clips = clips[: args.num_clips]

    # the arm's kept set as a keep mask on the full model
    q_arm = np.zeros((36, N_HEADS), np.float32)
    m_arm = np.zeros((36, 12288), np.float32)
    for li, e in enumerate(meta["vlm"]):
        q_arm[li, e["q"]] = 1.0
        m_arm[li, e["mlp"]] = 1.0
    as_built = sorted(set(range(N_HEADS)) - set(meta["vlm"][0]["q"]))

    # the band, from the arm's own layer-0 dual score in make_slim's order
    imp = np.load(REPO / "outputs" / imp_run / "importance.npz")
    score = np.maximum(rank_norm(imp["traj_vlm_q"]), rank_norm(imp["coc_vlm_q"]))[0]
    order = np.argsort(score)                                   # as select_mask_ratios
    fixed = sorted(int(h) for h in order[:N_FIXED_DROP])
    band = [int(h) for h in order[N_FIXED_DROP:N_FIXED_DROP + N_BAND]]
    assert sorted(int(h) for h in order[:N_DROP]) == as_built, (order[:N_DROP], as_built)
    print(f"{args.arm}: {len(clips)} clips of {manifest} ({cache}); fixed drops {fixed}; "
          f"band {sorted(band)}; as built drops {as_built}", flush=True)

    devices = None if args.gpu is None else [int(x) for x in args.gpu.split(",")]
    device = reserve_gpu(args.reserve_gb, devices=devices)
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")
    tc = model.vlm.config.text_config
    masks = ml.PruneMasks(model.vlm.model.language_model.layers, tc.num_attention_heads,
                          tc.head_dim, tc.intermediate_size, "cuda")
    (out_dir / "config.json").write_text(json.dumps({
        "plan": "plans/2026-10-14_l0-functional-redundancy.md (step 3, amended 2026-10-16)",
        "model": "nvidia/Alpamayo-1.5-10B", "model_revision": MODEL_REV, "arm": args.arm,
        "importance_from": imp_run, "manifest": manifest, "cache": cache,
        "clip_ids": [c for c, _ in clips], "seed": args.seed, "fm_steps": args.fm_steps,
        "max_gen": args.max_gen, "fixed_drop": fixed, "band": band, "as_built_drop": as_built,
        "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
    }, indent=2))

    # dense reference text once per clip; inputs kept on the CPU between configs
    t0 = time.time()
    ref = []
    for clip_id, clip_t0 in clips:
        data = sc.load_cached(sc.path_for(cache, clip_id, clip_t0))
        inputs = lib.build_inputs(model, processor, data, "cuda")
        prompt_len = inputs["input_ids"].shape[1]
        x1 = lib.gt_actions(model, data, "cuda").to(torch.float32)          # (1, 64, 2)
        base = sc.clip_seed(args.seed, clip_id)
        masks.reset()
        torch.manual_seed(base)
        torch.cuda.manual_seed_all(base)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
        coc_end = roll["eos_pos"] + 1
        ref.append({"pv": inputs["tokenized_data"]["pixel_values"].cpu(),
                    "grid": inputs["tokenized_data"]["image_grid_thw"].cpu(),
                    "seq": roll["sequences"][:, :coc_end].cpu(), "start": prompt_len,
                    "end": coc_end, "x1": x1.cpu(), "base": base})
        del roll, inputs
    print(f"reference text for {len(ref)} clips in {time.time() - t0:.0f}s", flush=True)

    def p2(drop0):
        """Mean FM loss (and NLL) with the arm's layers 1-35 and `drop0` removed in layer 0."""
        q = q_arm.copy()
        if drop0 is None:
            masks.reset()
        else:
            q[0] = 1.0
            q[0, list(drop0)] = 0.0
            masks.set(q=q, mlp=m_arm)
        fms, nlls = [], []
        for r in ref:
            inp = {"tokenized_data": {"pixel_values": r["pv"].to("cuda"),
                                      "image_grid_thw": r["grid"].to("cuda")}}
            nll, fm, _, _ = eval_config(model, inp, r["seq"].to("cuda"), r["start"], r["end"],
                                        None, r["x1"].to("cuda"), r["base"], args.fm_steps, [])
            fms.append(fm)
            nlls.append(nll)
        masks.reset()
        return float(np.mean(fms)), float(np.mean(nlls)), fms

    lines = []

    def say(s):
        print(s, flush=True)
        lines.append(s)

    t0 = time.time()
    dense_fm, dense_nll, dense_pc = p2(None)
    built_fm, built_nll, built_pc = p2(as_built)
    say(f"dense P2 {dense_fm:.5f}; as built {built_fm:.5f} (drops {as_built}) "
        f"[{time.time() - t0:.0f}s for 2 configs]")
    cur, trace = list(fixed), []

    def save(done):
        sel_fm = trace[-1]["best_fm"] if trace else None
        (out_dir / "metrics.json").write_text(json.dumps({
            "arm": args.arm, "n_clips": len(ref), "fixed_drop": fixed, "band": band,
            "as_built_drop": as_built, "dense_fm": dense_fm, "dense_nll": dense_nll,
            "as_built_fm": built_fm, "as_built_nll": built_nll, "trace": trace,
            "selected_drop": sorted(cur) if done else None, "selected_fm": sel_fm,
            "per_clip_fm": {"dense": dense_pc, "as_built": built_pc,
                            **({"selected": trace[-1]["best_per_clip"]} if done else {})},
            "complete": done}, indent=1))

    for step in range(args.steps):
        t1 = time.time()
        cands = [h for h in band if h not in cur]
        scored = {h: p2(cur + [h]) for h in cands}
        best = min(cands, key=lambda h: scored[h][0])
        cur.append(best)
        trace.append({"step": step + 1, "added": best, "best_fm": scored[best][0],
                      "best_nll": scored[best][1], "best_per_clip": scored[best][2],
                      "candidates": {str(h): {"fm": v[0], "nll": v[1]} for h, v in scored.items()}})
        ranked = sorted(cands, key=lambda h: scored[h][0])
        say(f"step {step + 1}: drop {best} -> P2 {scored[best][0]:.5f}; candidates best->worst "
            + " ".join(f"{h}:{scored[h][0]:.4f}" for h in ranked) + f" [{time.time() - t1:.0f}s]")
        save(done=False)
    save(done=len(trace) == N_DROP - N_FIXED_DROP)
    say(f"selected drops {sorted(cur)} (P2 {trace[-1]['best_fm']:.5f}) vs as built {as_built} "
        f"(P2 {built_fm:.5f}); kept from the band: {sorted(set(band) - set(cur))}")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
