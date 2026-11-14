"""Smoke for the Alpamayo 2 Edge track (plan step 2): the release inference path on the no-nav prompt.

Per clip (calib_edge cache):
  - release sample_trajectories_from_data, 6 independent CoCs x 1 trajectory, 10 Euler steps, seed = clip seed
    -> minADE@6 / minFDE@6 at 6.4 s against the GT (xy), CoC texts, how many CoCs closed with the anchor
  - G-rollout: edge_grad_lib.rollout_sequence and the release generation (1 CoC) under the same seed must produce
    the same tokens, so the importance run's rollout is the release rollout

Usage:
  CUDA_VISIBLE_DEVICES=6 /mnt/nvme1n1/ad_vla/venvs/alpamayo2-edge/bin/python iclr27-rebuttal/code/edge_smoke.py --n 3
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import edge_common as ec

ec.setup_hf_env(offline=True)

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]


def to_dev(x, dev):
    if torch.is_tensor(x):
        return x.to(dev)
    if isinstance(x, dict):
        return {k: to_dev(v, dev) for k, v in x.items()}
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--exp-id", default="iclr27-rebuttal/edge_smoke")
    args = ap.parse_args()
    out = REPO / "outputs" / args.exp_id
    out.mkdir(parents=True, exist_ok=True)

    import edge_grad_lib as gl
    from alpamayo2_edge import helper

    model, _ = ec.load_model("cuda")
    model.eval()
    processor = helper.get_processor(model.tokenizer, model.config)
    man = pd.read_parquet(ec.CALIB_MANIFEST).head(args.n)
    rows = []
    for r in man.itertuples():
        seed = ec.clip_seed(42, r.clip_id)
        data = ec.load_cached(ec.path_for(r.clip_id, int(r.t0_us)))
        inputs, text = ec.prepare_inputs_no_nav(data, model.config, model.tokenizer, processor)
        inputs = to_dev(inputs, "cuda")
        if not rows:
            (out / "prompt_example.txt").write_text(text)
        t = time.time()
        torch.manual_seed(seed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            xyz, _, _, extra = model.sample_trajectories_from_data(
                inputs, num_traj_sets=6, num_traj_samples=1, max_generation_length=256,
                diffusion_kwargs={"inference_step": 10, "generator": torch.Generator("cuda").manual_seed(seed)},
                return_extra=True)
        sec6 = time.time() - t
        pred = xyz[0, :, 0, :, :2].float().cpu().numpy()  # (6, 64, 2)
        gt = data["ego_future_xyz"][0, :, :2].numpy()  # (64, 2)
        d = np.linalg.norm(pred - gt[None], axis=-1)  # (6, 64)
        cots = [str(c) for c in np.asarray(extra["cot"]).reshape(-1)]

        # G-rollout: release generation (1 CoC) vs edge_grad_lib.rollout_sequence, same seed
        torch.manual_seed(seed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, _, _, ex1 = model.sample_trajectories_from_data(
                inputs, num_traj_sets=1, num_traj_samples=1, max_generation_length=256,
                diffusion_kwargs={"inference_step": 10, "generator": torch.Generator("cuda").manual_seed(seed)},
                return_extra=True)
        rel = ex1["generated_ids"][0]
        rel = rel[rel >= 0].cpu()
        torch.manual_seed(seed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            seq, plen, _, _, _ = gl.rollout_sequence(model, inputs, 256)
        mine = seq[0, plen:].cpu()
        same = bool(mine.shape == rel.shape and torch.equal(mine, rel))
        rows.append({"clip_id": r.clip_id, "minADE6": float(d.mean(1).min()), "minFDE6": float(d[:, -1].min()),
                     "sec_6cocs": round(sec6, 1), "prompt_tokens": int(plen), "rollout_eq_release": same,
                     "coc_len": int(mine.numel()), "cots": cots})
        print(f"{r.clip_id[:8]} minADE6={rows[-1]['minADE6']:.3f} minFDE6={rows[-1]['minFDE6']:.3f} "
              f"prompt={plen} tok, rollout==release {same}, {sec6:.1f}s for 6 CoCs", flush=True)
        for c in cots[:3]:
            print("   CoC:", c[:160], flush=True)
    df = pd.DataFrame(rows)
    summary = {"n": len(df), "minADE6_mean": float(df.minADE6.mean()), "minFDE6_mean": float(df.minFDE6.mean()),
               "all_rollout_eq_release": bool(df.rollout_eq_release.all())}
    (out / "metrics.json").write_text(json.dumps({**summary, "per_clip": rows}, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
