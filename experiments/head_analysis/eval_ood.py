"""OOD evaluation of compressed models against the GT OOD reasoning set.

Each OOD clip is loaded at its event_start_timestamp (the moment the OOD situation
occurs). The curated GT reasoning (ood_reasoning.parquet `coc`) is used two ways:

  1. as the teacher-forced denoise CONTEXT  -> trajectory minADE given correct reasoning
  2. as the target of an NLL                 -> can the model predict the correct OOD reasoning
Plus each model's OWN rollout CoC text is saved for qualitative comparison vs GT.

GT CoC is model-independent, so every model (baseline + each slim) runs an independent
pass -> no cross-model seq sharing, GPU-parallelizable. Report is baseline absolutes +
per-slim deltas, broken down by OOD cluster.

GT CoC teacher-forcing matches the generated format exactly (verified round-trip):
  [prompt] + tokenize(reasoning text) + <|cot_end|>(155678) + <|traj_future_start|>(155681)

Usage:
  bash experiments/head_analysis/run_retry.sh 30 experiments/head_analysis/eval_ood.py \
      --model baseline --exp-id ood_eval --gpu 0
  ... --model outputs/slim_integrated_mag --exp-id ood_eval --gpu 1
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

import analysis_lib as lib  # noqa: E402
import eval_lib as el  # noqa: E402
import slim_lib as sl  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  gated-repo hub patch
from run_eval import eval_config  # noqa: E402

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset  # noqa: E402
from alpamayo1_5.models.token_utils import to_special_token  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

REPO = Path("/workspace/alpamayo-model-compression")
AV = Path("/mnt/nvme1n1/ad_vla/data/physicalai_av")
COT_END, TFS = 155678, 155681


def build_manifest():
    """OOD clips with local frames (test-excluded): {clip_id, t0_us, cluster, gt_coc}."""
    cache = REPO / "outputs" / "ood_manifest.json"
    if cache.exists():
        return json.loads(cache.read_text())
    df = pd.read_parquet(AV / "reasoning" / "ood_reasoning.parquet")
    ci = pd.read_parquet(AV / "clip_index.parquet")
    cam = AV / "camera" / "camera_front_wide_120fov"
    local = {int(f.split("chunk_")[1].split(".zip")[0])
             for f in os.listdir(cam) if "chunk_" in f}
    test = set(json.loads((REPO / "outputs" / "split.json").read_text())["test"])
    man = []
    for cid, row in df.iterrows():
        if cid in test or cid not in ci.index:
            continue
        if int(ci.loc[cid, "chunk"]) not in local:
            continue
        evs = json.loads(row["events"])
        ev = next((e for e in evs if "event_start_timestamp" in e and "coc" in e), None)
        if ev is None:
            continue
        man.append({"clip_id": cid, "t0_us": int(ev["event_start_timestamp"]),
                    "cluster": row["event_cluster"], "gt_coc": ev["coc"]})
    cache.write_text(json.dumps(man))
    return man


def gt_coc_seq(tok, prompt_ids, gt_text, device):
    """[prompt] + tokenize(reasoning) + cot_end + traj_future_start -> (1, L) seq_tf."""
    rt = tok(gt_text, add_special_tokens=False)["input_ids"]
    coc = torch.tensor([rt + [COT_END, TFS]], device=device)
    return torch.cat([prompt_ids, coc], dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="'baseline' or a slim/merged ckpt dir")
    ap.add_argument("--exp-id", type=str, default="ood_eval")
    ap.add_argument("--num-clips", type=int, default=10_000)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    tag = "baseline" if args.model == "baseline" else Path(args.model).name
    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    man = build_manifest()[: args.num_clips]

    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    print(f"using {device}  model={tag}  {len(man)} OOD clips", flush=True)

    if args.model == "baseline":
        model = Alpamayo1_5.from_pretrained(
            "nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16).to("cuda")
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    else:
        model = sl.load_slim(REPO / args.model, device="cuda")
    processor = helper.get_processor(model.tokenizer)
    tok = model.tokenizer
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")
    eos_id = tok.convert_tokens_to_ids(to_special_token("traj_future_start"))

    rows = []
    out_path = out_dir / f"{tag}.json"
    done = {r["clip_id"] for r in json.loads(out_path.read_text())} if out_path.exists() else set()
    if done:
        rows = json.loads(out_path.read_text())
    for i, m in enumerate(man):
        if m["clip_id"] in done:
            continue
        t0 = time.time()
        try:
            data = load_physical_aiavdataset(m["clip_id"], t0_us=m["t0_us"])
        except Exception as e:
            print(f"[skip {m['clip_id'][:8]}] load failed: {type(e).__name__}", flush=True)
            continue
        inputs = lib.build_inputs(model, processor, data, "cuda")
        prompt_len = inputs["input_ids"].shape[1]
        gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()
        seeds = [args.seed + i * 100 + k for k in range(args.k)]

        # (1)+(2) teacher-force GT CoC as context -> minADE_tf, and NLL of GT CoC
        seq_gt = gt_coc_seq(tok, inputs["input_ids"], m["gt_coc"], "cuda")
        ade_tf, fde_tf, nll = eval_config(model, inputs, seq_gt, prompt_len, seq_gt.shape[1],
                                          gt_xy, seeds)
        # (3) own rollout: generated CoC text (qualitative) + rollout-context trajectory
        torch.manual_seed(args.seed + i)
        torch.cuda.manual_seed_all(args.seed + i)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
        coc_end = roll["eos_pos"] + 1
        seq_gen = roll["sequences"][:, :coc_end].clone()
        gen_ids = roll["sequences"][0, prompt_len:coc_end].tolist()
        gen_coc = tok.decode([t for t in gen_ids if t not in (COT_END, TFS)],
                             skip_special_tokens=True)
        del roll
        # trajectory under the model's OWN reasoning context (end-to-end)
        ade_ro, fde_ro, _ = eval_config(model, inputs, seq_gen, prompt_len, coc_end, gt_xy, seeds)

        rows.append({"clip_id": m["clip_id"], "cluster": m["cluster"], "bucket": el.bucket(gt_xy),
                     "minADE_tf": ade_tf, "minFDE_tf": fde_tf, "nll_gtcoc": nll,
                     "minADE_rollout": ade_ro, "minFDE_rollout": fde_ro,
                     "gt_coc": m["gt_coc"], "gen_coc": gen_coc, "gen_len": len(gen_ids)})
        if (i + 1) % 10 == 0 or i + 1 == len(man):
            out_path.write_text(json.dumps(rows, indent=2))
        print(f"[{i + 1}/{len(man)}] {m['clip_id'][:8]} {m['cluster'][:22]:22s} "
              f"ade_tf={ade_tf:.3f} ade_ro={ade_ro:.3f} nll={nll:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"saved {len(rows)} -> {out_path}  "
          f"minADE tf={np.mean([r['minADE_tf'] for r in rows]):.4f} "
          f"rollout={np.mean([r['minADE_rollout'] for r in rows]):.4f}  "
          f"NLL(GT CoC)={np.mean([r['nll_gtcoc'] for r in rows]):.4f}", flush=True)


if __name__ == "__main__":
    main()
