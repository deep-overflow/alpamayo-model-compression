"""Build the Minitron width checkpoint from mt_importance's kept sets (plan D7 / D8).

VLM text tower only: hidden 4096 -> 3328 (slim_lib.slice_hidden, the vision merger and
DeepStack mergers' outputs included), MLP 12288 -> 9216 per layer; every Q head, KV group,
layer, the expert and the ViT kept. Asserts the removed parameter count (2,659,847,424,
+0.09% over MoOP's 2,657,452,032).

Writes the recipe only (slim_meta.json): the zero-shot arm is the base weights cut, so
slim_lib.load_slim rebuilds it exactly, and run_baseline.py --model reads it as is. The KD
arm saves its own slim_state.pt. Runs on CPU.

Usage:
  .venv/bin/python minitron/code/make_mt_slim.py \\
      --kept outputs/minitron/mt_importance/kept.json --out outputs/minitron/slim_mtw24
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))

import expert_per_clip  # noqa: F401  -- gated-hub patch, imported before any from_pretrained
import slim_lib as sl
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

EXPECTED_REMOVED = 2_659_847_424


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kept", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    kept = json.loads((REPO / args.kept).read_text())

    model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", revision=sl.MODEL_REV,
                                        dtype=torch.bfloat16)
    tc = model.vlm.config.text_config
    ec = model.expert.config
    n_dense = sl.n_params(model)
    vm = np.zeros((tc.num_hidden_layers, tc.intermediate_size))
    for li, idx in enumerate(kept["vlm_mlp"]):
        vm[li, idx] = 1.0
    meta = sl.apply_surgery(
        model, np.ones((tc.num_hidden_layers, tc.num_attention_heads)), vm,
        np.ones((ec.num_hidden_layers, ec.num_attention_heads)),
        np.ones((ec.num_hidden_layers, ec.intermediate_size)),
        vlm_hidden=np.asarray(kept["vlm_hidden"]))
    sl.check_slim(model)
    removed = n_dense - sl.n_params(model)
    assert removed == EXPECTED_REMOVED, f"removed {removed:,}, expected {EXPECTED_REMOVED:,}"
    meta["minitron"] = {"kept_from": args.kept, "removed": int(removed),
                        "hidden": len(kept["vlm_hidden"]), "mlp": len(kept["vlm_mlp"][0])}
    sl.save_slim(model, meta, REPO / args.out, write_state=False)
    print(f"{args.out}: removed {removed:,} of {n_dense:,} ({removed / n_dense:.2%})")


if __name__ == "__main__":
    main()
