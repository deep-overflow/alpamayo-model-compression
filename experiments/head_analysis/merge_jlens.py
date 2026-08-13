"""Combine sharded run_jlens accumulators into one jlens.npz.

`run_jlens.py --shard i --n-shards n` writes raw sums rather than final scores, because
the J-lens is a mean over clips and the write statistics are a mean over source
positions -- both merge by adding the sums and dividing once at the end. A shard costs
~280 s/clip, so splitting 100 clips over four cards turns 8 h into 2 h.

Split-half here is across shards rather than odd/even clips within one run. That is a
coarser split (4 groups instead of alternating clips) but measures the same thing: how
much of the J-score is Jacobian noise rather than signal.

Usage:
  python experiments/head_analysis/merge_jlens.py --exp-id jlens_v2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))

import jlens_lib as jl
import mask_lib as ml
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from expert_per_clip import reserve_gpu

REPO = Path(__file__).resolve().parents[2]
READOUT_LAYERS = [0, 6, 12, 18, 24, 30, 35]
MODEL_REV = "7aba8293c09993f2e125c6819df05d7fa3e873ea"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--save-vectors", action="store_true")
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    shards = sorted(out_dir.glob("shard_*of*.pt"))
    if not shards:
        raise SystemExit(f"no shard_*of*.pt in {out_dir}")

    parts = [torch.load(p, map_location="cpu", weights_only=False) for p in shards]
    n_expected = int(shards[0].stem.split("of")[1])
    if len(parts) != n_expected:
        raise SystemExit(f"found {len(parts)} shards but the names say {n_expected}; "
                         "merging an incomplete set would silently reweight the mean")
    # token dictionary must match across shards or the lens axes do not line up
    token_ids = parts[0]["token_ids"]
    for p, f in zip(parts[1:], shards[1:]):
        if list(p["token_ids"]) != list(token_ids):
            raise SystemExit(f"{f.name} has a different token dictionary")

    n_clips = sum(p["n_clips"] for p in parts)
    n_pos = sum(p["wstats_count"] for p in parts)
    print(f"{len(parts)} shards, {n_clips} clips, {n_pos} source positions", flush=True)

    v_acc = sum(p["v_sum"] for p in parts) / n_clips
    mlp_sq = (sum(p["mlp_sq_sum"] for p in parts) / n_pos).float()
    head_cov = (sum(p["head_cov_sum"] for p in parts) / n_pos).float()
    n_probe = min(p["probes"].shape[1] for p in parts)
    probes = sum(p["probes"][:, :n_probe] for p in parts) / n_clips
    n_freq = parts[0]["n_freq"]

    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, dtype=torch.bfloat16).to("cuda")
    model.eval()
    layers = model.vlm.model.language_model.layers
    tc = model.vlm.config.text_config

    v_acc = v_acc.to("cuda")
    scores = jl.unit_jscores(model, v_acc, mlp_sq.to("cuda"), head_cov.to("cuda"))

    kurt = np.array([jl.excess_kurtosis(v_acc[1, li, n_freq:], probes[li].to("cuda"))
                     for li in range(len(layers))])
    cka = jl.cka_matrix(v_acc[1])
    auc = np.array([jl.freq_auc(v_acc[1, li], probes[li].to("cuda"), n_freq)
                    for li in range(len(layers))])
    readouts = {str(li): jl.readout(v_acc[1, li], probes[li].mean(0).to("cuda"),
                                    list(token_ids), model.tokenizer)
                for li in READOUT_LAYERS if li < len(layers)}

    # split-half across shards: same write stats both sides, so this isolates the
    # Jacobian noise rather than activation-statistic noise
    half = len(parts) // 2
    split_half = {}
    if half:
        a = sum(p["v_sum"] for p in parts[:half]) / sum(p["n_clips"] for p in parts[:half])
        b = sum(p["v_sum"] for p in parts[half:]) / sum(p["n_clips"] for p in parts[half:])
        sa = jl.unit_jscores(model, a.to("cuda"), mlp_sq.to("cuda"), head_cov.to("cuda"))
        sb = jl.unit_jscores(model, b.to("cuda"), mlp_sq.to("cuda"), head_cov.to("cuda"))
        for k in ("q_j", "mlp_j"):
            split_half[k] = np.array([spearmanr(sa[k][li], sb[k][li]).statistic
                                      for li in range(len(layers))])
        del sa, sb, a, b

    mag_q, mag_mlp = ml.magnitude_scores(layers, tc.num_attention_heads, tc.head_dim,
                                         tc.intermediate_size)
    np.savez(out_dir / "jlens.npz", token_ids=np.array(token_ids), kurtosis=kurt, cka=cka,
             freq_auc=auc, n_freq=n_freq, mag_q=mag_q, mag_mlp=mag_mlp,
             mlp_sq=mlp_sq.numpy(), head_cov=head_cov.numpy(),
             **{f"split_{k}": v for k, v in split_half.items()},
             **{k: v for k, v in scores.items()})
    if args.save_vectors:
        torch.save({"v": v_acc.half().cpu(), "token_ids": token_ids},
                   out_dir / "jlens_vectors.pt")

    clip_ids = [c for p in parts for c in p["clip_ids"]]
    records = [r for p in parts for r in p["records"]]
    (out_dir / "config.json").write_text(json.dumps({
        "model": "nvidia/Alpamayo-1.5-10B", "model_revision": MODEL_REV,
        "purpose": "J-lens over the VLM tower + J-space unit scores (Stage A/B)",
        "merged_from": [p.name for p in shards],
        "num_clips": n_clips, "clip_ids": clip_ids, "n_source_positions": n_pos,
        "dict_tokens": len(token_ids), "n_freq": int(n_freq),
        "gpu": torch.cuda.get_device_name(device),
        "shapes": {k: list(v.shape) for k, v in scores.items()},
    }, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps({
        "per_clip": records, "kurtosis": kurt.tolist(), "freq_auc": auc.tolist(),
        "readouts": readouts,
        "split_half": {k: v.tolist() for k, v in split_half.items()},
    }, indent=2))

    lines = [f"J-lens merged: {len(parts)} shards, {n_clips} clips, {n_pos} positions"]
    if split_half:
        lines += ["", "split-half reproducibility of the J-score (shard groups)",
                  "  this is the noise floor: any G2 margin smaller than 1-rho is unreadable"]
        for k, v in split_half.items():
            lines.append(f"  {k:6s} median rho={np.median(v):+.3f}  "
                         f"min={v.min():+.3f}  max={v.max():+.3f}")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print("saved ->", out_dir, flush=True)


if __name__ == "__main__":
    main()
