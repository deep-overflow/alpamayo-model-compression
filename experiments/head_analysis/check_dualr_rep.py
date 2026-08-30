"""G0 of plans/2026-08-30_dualr-weighted-hessian.md: does the uniform / decode-0 supernet
from run_cache_recon.py reproduce dualr_u40's refitted weights?

Compares every module's kept columns in <supernet>/layers.NN.*/0.pth against the kept
weights stored in outputs/slim_dualr_u40/slim_state.pt (relative Frobenius difference),
and prints the median / max over the 72 modules. Pass: max < 1e-2.

Usage:
  .venv/bin/python experiments/head_analysis/check_dualr_rep.py --supernet dualr_rep_supernet_u40
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--supernet", required=True)
    ap.add_argument("--ref", default="slim_dualr_u40")
    ap.add_argument("--tol", type=float, default=1e-2)
    args = ap.parse_args()
    sup = REPO / "outputs" / args.supernet
    meta = json.loads((REPO / "outputs" / args.ref / "slim_meta.json").read_text())
    sd = torch.load(REPO / "outputs" / args.ref / "slim_state.pt", map_location="cpu",
                    weights_only=True)
    rels = {}
    for name in json.loads((sup / "metadata.json").read_text())["layer_names"]:
        li = int(name.split(".")[1])
        kind = "self_attn.o_proj" if "o_proj" in name else "mlp.down_proj"
        w = torch.load(sup / name / "0.pth", map_location="cpu").float()
        kept = ([h * 128 + d for h in meta["vlm"][li]["q"] for d in range(128)]
                if "o_proj" in kind else meta["vlm"][li]["mlp"])
        ref = sd[f"vlm.model.language_model.layers.{li}.{kind}.weight"].float()
        assert ref.shape[1] == len(kept), (name, ref.shape, len(kept))
        rels[name] = float((w[:, kept] - ref).norm() / ref.norm())
    v = np.array(list(rels.values()))
    worst = max(rels, key=rels.get)
    verdict = "PASS" if v.max() < args.tol else "FAIL"
    text = (f"G0 dualr reproduction ({args.supernet} vs {args.ref}): {len(v)} modules, rel diff "
            f"median {np.median(v):.2e} max {v.max():.2e} ({worst}) -> {verdict}")
    print(text)
    (sup / "g0_reproduction.json").write_text(json.dumps({"rel": rels, "verdict": verdict}, indent=1))


if __name__ == "__main__":
    main()
